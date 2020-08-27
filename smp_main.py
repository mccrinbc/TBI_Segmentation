import os 
import random
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from datetime import datetime
import pickle 

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
             
        return image, label #, self.images_fps[self.mapping[ii]],self.labels_fps[self.mapping[ii]]
    
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


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc
    
train_size = 0.75
batch_size = 12
EPOCHS = 3
lr = 0.0001
aug_angle = 25
aug_scale = [1,1.5]
flip_prob = 0.5
num_workers = 1
#images_dir = "/Users/brianmccrindle/Documents/Research/TBIFinder_Final/Registered_Brains_FA/normalized_slices"
#labels_dir = "/Users/brianmccrindle/Documents/Research/TBIFinder_Final/Registered_Brains_FA/slice_labels"

images_dir = "/Users/brianmccrindle/Documents/Research/TBIFinder_Final/Registered_Brains_FA/test_slices"
labels_dir = "/Users/brianmccrindle/Documents/Research/TBIFinder_Final/Registered_Brains_FA/test_labels"

#smp specific variables
ENCODER = 'resnet101'
aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    #activation='softmax2d',    # activation function, default is None. This is the output activation. softmax2d specifies dim = 1 
    classes=1,                 # define number of output labels
)

#classes = 2 for the softmax transformation. 
model = smp.Unet(encoder_name = ENCODER, in_channels=1, classes = 1, aux_params = aux_params)

def Weights(labels):
    #expects an [batch_size,c,n,n] input 
    
    weights = []
    for batch_num in range(0,labels.shape[0]):
        num_ones = torch.sum(labels[batch_num,0,:,:]);
        resolution = labels.shape[2] * labels.shape[3]
        num_zeros = resolution - num_ones 
        weights.append(num_zeros / (num_ones + 1))
        
    #this keeps the clas imbalance in check
    return torch.Tensor(weights) #to ensure that we're getting a real number in the division  
    

def train_validate(lr):
    
    earlystop = False 
    
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
    
    loss_function = torch.nn.BCELoss() #this takes in a weighted input and incorporates a sigmoid transformation
    #loss_function = smp.utils.losses.DiceLoss()
    #loss_function = DiceLoss()
    #metrics = [smp.utils.metrics.IoU(threshold=0.5)]
    
    loss_train = []
    loss_valid = []
    epochLoss_train = []
    epochLoss_valid = []
        
    for epoch in range(EPOCHS):
        image_count = 0
        for phase in ["train","val"]:
            
            #This determines which portions of the model will have gradients turned off or on. 
            if phase == "train":
                model.train() #put into training mode
                loader = train_loader
            else:
                model.eval() #evaluation mode.
                loader = valid_loader
                  
            for ii, data in enumerate(loader): 
                
                brains = data[0] #[batch_size,channels,height,width] 
                labels = data[1]
                
                image_count += len(brains)
                print(epoch, phase, ii, image_count)
                
                brains,labels = brains.to(dev), labels.to(dev) #put the data onto the device
                predictions, single_class = model(brains) #single class is not a useful output. 
                
                predictions = torch.sigmoid(predictions) #using this so that the output is bounded [0,1]
                single_class = torch.sigmoid(single_class)
                
                weights = Weights(labels) #generate the weights for each slice in the batch
                loss_function.pos_weight = weights                    
                
                loss = loss_function(predictions, labels) #loss changes here. 
                
                if phase == "train":
                    #employ this so we don't get multiples in the same list. 
                    if (loss_valid and ii == 0): #if loss_valid is NOT empty AND it's the first time we see this
                        epochLoss_valid.append(loss_valid[-1]) #append the last value in the 
                        
                    model.zero_grad()
                    loss_train.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    
                    print(f"Phase: {phase}. Epoch: {epoch}. Loss: {loss.item()}") 
               
                else:
                    if (loss_train and ii == 0):#if loss_valid is NOT empty AND it's the first time we see this
                        epochLoss_train.append(loss_train[-1]) #append the last value in the loss_train list.
                        
                    loss_valid.append(loss.item())
                    print(f"Phase: {phase}. Epoch: {epoch}. Loss: {loss.item()}") 
                    
                    #learning rate changes and early stopping
                    if epoch > 0:
                        if (epoch % 10) == 0: #if the epoch is divisable by 10
                            meanVal = np.mean(loss_valid[epoch - 10 : epoch])
                            if np.abs((meanVal - loss.item()) / meanVal) <= 0.05: #if the %difference is small
                                for param_group in optimizer.param_groups:
                                    lr = lr * 0.1 #reduce the learning rate by a factor of 10. 
                                    param_group['lr'] = lr
                        
                        if (epoch % 50) == 0:
                            meanVal = np.mean(loss_valid[epoch - 50 : epoch])
                            if np.abs((meanVal - loss.item()) / meanVal) <= 0.05:
                                earlystop = True 
               
                #Implementation of early stopping
                if earlystop == True:
                    torch.save(model.state_dict(), os.getcwd()) #save the model 
                    break
            else:
                continue
            break
        else:
            #save the model at the end of this epoch.
            date = datetime.now()
            torch.save(model.state_dict(), os.path.join(os.getcwd(), "Registered_Brains_FA/models_saved", "TBI_model-epoch" + str(epoch) + '-' + str(date.date()) + '-' + str(date.hour) + '-' + str(date.minute) +".pt"))
            continue
        break
    
    #Need to add the last element from loss_valid to epochLoss_valid to equal the number of epochs. 
    epochLoss_valid.append(loss_valid[-1])
    return brains, labels, predictions, single_class, loss_train, loss_valid, epochLoss_train, epochLoss_valid, model.state_dict()

brains, labels, predictions, single_class, loss_train, loss_valid, epochLoss_train, epochLoss_valid, model_state = train_validate(lr)

date = datetime.now()
torch.save(model_state, os.path.join(os.getcwd(), "Registered_Brains_FA/models_saved", "TBI_model-End-" + str(date.date()) + '-' + str(date.hour) + '-' + str(date.minute) +".pt"))

# Saving the objects:
with open('results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([brains, labels, predictions, single_class, loss_train, loss_valid, epochLoss_train, epochLoss_valid], f)
    
# # Getting back the objects:
#with open('/Users/brianmccrindle/Documents/Research/TBIFinder_Final/Registered_Brains_FA/models_saved/1/results.pkl','rb') as f:  
#    brains, labels, predictions, single_class, loss_train, loss_valid, epochLoss_train, epochLoss_valid = pickle.load(f)















# NOTES: 

#PIL is some type of holder for data so if you want the actual array, you need to apply
# img = np.array(Image.open(img_path))

#The learning rate should be reduced 
#by a factor of 10 when the validation loss fails to improve for 10 consecutive epochs.

#Early stopping should be implemented if validation loss does not improve over 50 epochs


