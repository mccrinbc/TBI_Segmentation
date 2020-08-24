import os 
import random
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as ParentDataset

import torchvision 
from torchvision import transforms
from torchvision.transforms import Compose

import segmentation_models_pytorch as smp #model we're using for now. 

#we need to create a dataset class to obtain the training, validation, and testing sets
#of our model. we use this as a map-style dataset object. 

class TBI_dataset(ParentDataset): #Obtain the attributes of ParentDataset from torch.utils.data
#Finds Image and Label locations, creates random list of indicies for training / val / testing sets to be called
    def __init__(
        self,
        images_dir,
        labels_dir,
        train_size = 0.75, #fraction of total number of samples to be used in training set
        subset="train",
        transform = None, #need to include this for the iterator to work. 
        random_sampling=True,
        seed=42,
    ):
        #filter and sort the list
        self.ImageIds = sorted(list(filter(('.DS_Store').__ne__,os.listdir(images_dir)))) 
        self.LabelIds = sorted(list(filter(('.DS_Store').__ne__,os.listdir(labels_dir))))
        
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ImageIds] #full_paths to slices
        self.labels_fps = [os.path.join(labels_dir, image_id) for image_id in self.LabelIds] #full_paths to labels
        
        if random_sampling == True:
            samples = list(range(0,len(self.images_fps))) #create a list of numbers
            random.seed(seed) #set the seed
            
            #random sample train_size amount and then do a train/validation split 
            indicies = random.sample(samples,round(train_size*len(samples)))
            self.val_indicies = indicies[0:round(len(indicies)*0.15)]
            self.train_indicies = indicies[round(len(indicies)*0.15) : len(indicies)]
            
            test_indicies = samples
            for j in sorted(indicies, reverse = True): #remove the train/val indicies from test set
                del test_indicies[j]
            
            #suffles without replacement. 
            self.test_indicies = random.sample(test_indicies, len(test_indicies)) 

        #We define a mapping to use when calling the Dataset loader based on the parameter "subset"
        if subset == "train":
            self.mapping = self.train_indicies
        elif subset == "val":
            self.mapping = self.val_indicies
        elif subset == "test":
            self.mapping = self.test_indicies
        else:
            print("subset parameter requires train, val, or test exactly.")
            
        self.transform = transform #trasform given by transform_function
            
    def __getitem__(self, ii): #ii is the index
        
        #Current implementations of transforms only use PIL images.
        image = Image.open(self.images_fps[self.mapping[ii]]) #open as PIL image.
        label = Image.open(self.labels_fps[self.mapping[ii]])
        
        image = self.transform(image)
        label = self.transform(label)
             
        return image, label, self.images_fps[self.mapping[ii]],self.labels_fps[self.mapping[ii]]
    
    def __len__(self):
        return len(self.mapping)
    
    
def datasets(images_dir, labels_dir, train_size, aug_angle, aug_scale, flip_prob):
    train = TBI_dataset(
        images_dir = images_dir,
        labels_dir = labels_dir,
        train_size = 0.75,
        subset="train",
        transform=transform_function(degrees=aug_angle, scale=aug_scale, flip_prob=flip_prob),
    )
    valid = TBI_dataset(
        images_dir=images_dir,
        labels_dir = labels_dir,
        train_size = 0.75,
        subset="val",
        transform=transform_function(degrees=aug_angle, scale=aug_scale, flip_prob=flip_prob),
    )
    return train, valid

def transform_function(degrees,scale,flip_prob):
    transform_list = []
    
    transform_list.append(transforms.RandomAffine(degrees, scale = scale))
    transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))
    transform_list.append(transforms.Pad(37)) #all images should be 182x182.
    transform_list.append(transforms.ToTensor())
    
    return Compose(transform_list)


# helper function for data visualization
def visualize(**images):
    
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
train_size = 0.75
batch_size = 1
EPOCHS = 100
lr = 0.0001
aug_angle = 25
aug_scale = [1,1.5]
flip_prob = 0.5
num_workers = 1
images_dir = "/Users/brianmccrindle/Documents/Research/TBIFinder_Final/Registered_Brains_FA/normalized_slices"
labels_dir = "/Users/brianmccrindle/Documents/Research/TBIFinder_Final/Registered_Brains_FA/slice_labels"


#smp specific variables
ENCODER = 'resnet101'
aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation='softmax2d',    # activation function, default is None. This is the output activation. softmax2d specifies dim = 1 
    classes=1,                 # define number of output labels
)

model = smp.Unet(encoder_name = ENCODER, in_channels=1, aux_params = aux_params)

def train_validate():
    if torch.cuda.is_available():
        dev ="cuda:0"
    else:
        dev = "cpu"
        
    dev = torch.device(dev)
    train_dataset, valid_dataset = datasets(images_dir, labels_dir, train_size, aug_angle, aug_scale, flip_prob)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle = True, num_workers = num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle = True, num_workers = num_workers)
    
    model.to(dev) #cast the model onto the device 
    optimizer = optim.Adam(model.parameters(), lr = lr) #learning rate should change 
    loss_function = smp.utils.losses.DiceLoss()
    #metrics = [smp.utils.metrics.IoU(threshold=0.5)]
    
    loss_train = []
    loss_valid = []
    
    for epoch in range(EPOCHS):
        for phase in ["train","val"]:
            
            #This determines which portions of the model will have gradients turned off or on. 
            if phase == "train":
                model.train() #put into training mode
                loader = train_loader
            else:
                model.eval() #evaluation mode.
                loader = valid_loader
            
            print(phase)
            for ii, data in enumerate(loader): 
                
                brains = data[0] #[batch_size,1,256,256] 
                labels = data[1]
                
                brains,labels = brains.to(dev), labels.to(dev) #put the data onto the device
                predictions, masks = model(brains)
                
                loss = loss_function(predictions, labels)
                if phase == "train":
                    model.zero_grad()
                    loss_train.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    
                else:
                    loss_valid.append(loss.item())
                
            print(f"Phase: {phase}. Epoch: {epoch}. Loss: {loss.item()}")
        
        return loss_train, loss_valid
                

loss_train, loss_valid = train_validate()


















# NOTES: 

#PIL is some type of holder for data so if you want the actual array, you need to apply
# img = np.array(Image.open(img_path))




