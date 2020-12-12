import os 
import random
import numpy as np
from PIL import Image

from datetime import datetime
import pickle
from tqdm import tqdm #loading bar  

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as ParentDataset

from torchvision import transforms
from torchvision.transforms import Compose

import sklearn.metrics

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
        #Apparently we can use np.array(Image.open(...)) to remove the error that happens each epoch
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
        subset = "train",
        transform = transform_function(degrees=aug_angle, scale=aug_scale, flip_prob=flip_prob),
    )
    valid = TBI_dataset(
        images_dir = images_dir,
        labels_dir = labels_dir,
        train_size = 0.75,
        subset = "val",
        transform = transform_function(degrees=aug_angle, scale=aug_scale, flip_prob=flip_prob),
    )
    
    test = TBI_dataset(
        images_dir = images_dir,
        labels_dir = labels_dir,
        train_size = 0.75,
        subset="test",
        transform = transform_function(degrees=aug_angle, scale=aug_scale, flip_prob=flip_prob),
    )
    
    return train, valid, test

def transform_function(degrees,scale,flip_prob):
    transform_list = []
    
    transform_list.append(transforms.RandomAffine(degrees, scale = scale))
    transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))
    transform_list.append(transforms.Pad(37)) #all images should be 182x182 before padding. 
    transform_list.append(transforms.ToTensor())
    
    return Compose(transform_list)

def transform_function_postTesting():
    transform_list = []

    transform_list.append(transforms.Pad(37)) #all images should be 182x182 before padding. 
    transform_list.append(transforms.ToTensor())
    
    return Compose(transform_list)


train_size = 0.75
batch_size = 12
EPOCHS = 40
lr = 0.0001
aug_angle = 25
aug_scale = [1,1.5]
flip_prob = 0.5
num_workers = 1
images_dir = "/home/mccrinbc/Registered_Brains_FA/normalized_slices"
labels_dir = "/home/mccrinbc/Registered_Brains_FA/slice_labels"

#images_dir = "/Users/brianmccrindle/Documents/Research/TBIFinder_Final/Registered_Brains_FA/test_slices"
#labels_dir = "/Users/brianmccrindle/Documents/Research/TBIFinder_Final/Registered_Brains_FA/test_labels"

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
train_dataset, valid_dataset, test_dataset = datasets(images_dir, labels_dir, train_size, aug_angle, aug_scale, flip_prob)

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
    

def train_validate(train_dataset, valid_dataset, lr):
    
    earlystop = False 
    
    if torch.cuda.is_available():
        dev ="cuda:2"
    else:
        dev = "cpu"
        
    dev = torch.device(dev)
    
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
                        
                    model.zero_grad()# for p in model.parameters(): p.grad = None # for more efficient computation
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
                    date = datetime.now()
                    torch.save(model.state_dict(), os.path.join(os.getcwd(), "Registered_Brains_FA/models_saved", "TBI_model-epoch" + str(epoch) + '-' + str(date.date()) + '-' + str(date.hour) + '-' + str(date.minute) +"-EARLYSTOP.pt")) #save the model 
                    break
            else:
                continue
            break
        else:
            #save the model at the end of this epoch.
            #date = datetime.now()
            #torch.save(model.state_dict(), os.path.join(os.getcwd(), "Registered_Brains_FA/models_saved", "TBI_model-epoch" + str(epoch) + '-' + str(date.date()) + '-' + str(date.hour) + '-' + str(date.minute) + ".pt"))
            continue
        break
    
    #Need to add the last element from loss_valid to epochLoss_valid to equal the number of epochs. 
    epochLoss_valid.append(loss_valid[-1])
    return brains, labels, predictions, single_class, loss_train, loss_valid, epochLoss_train, epochLoss_valid, model.state_dict()

def testModel(test_dataset, modelPath, threshold): #model = the model class = smp.UNet()

    total_images = 0
    CM_values = [0,0,0,0] #tp, fn, fp, tn
    model.load_state_dict(torch.load(modelPath))
    
    if torch.cuda.is_available():
        dev ="cuda:2"
        print("GPU is active")
    else:
        dev = "cpu"
        
    dev = torch.device(dev)
    model.to(dev) 
    model.eval() #evaluation mode to turn off the gradients / training. 
    
    loader = DataLoader(test_dataset, batch_size, shuffle = True, num_workers = num_workers)
    for ii, data in tqdm(enumerate(loader)):
        
        brains = data[0]
        labels = data[1]
        
        #move the data to the GPU 
        brains = brains.to(dev)
        labels = labels.to(dev)
        
        total_images += brains.shape[0] #this would be the same if we used labels or predictions. 
        #print(total_images)
        
        predictions, _ = model(brains)
        predictions = torch.sigmoid(predictions) 
        
        predictions_numpy = predictions.cpu().detach().numpy()
        labels_numpy = labels.cpu().detach().numpy()
        for j in range(predictions.shape[0]):
            #labels = [False, True] are needed to make sure we don't have errors with the shape of CM
            CM = sklearn.metrics.confusion_matrix(labels_numpy[j,0,:,:].ravel(), predictions_numpy[j,0,:,:].ravel() > threshold, labels = [False,True])
            try: 
                CM_values[0] = CM_values[0] + CM[0][0]
                CM_values[1] = CM_values[1] + CM[0][1]
                CM_values[2] = CM_values[2] + CM[1][0]
                CM_values[3] = CM_values[3] + CM[1][1]
            except:
                print("Error in Appending")
                return CM, CM_values
            
    del loader #delete loader
    return np.divide(CM_values , (total_images*(256*256)))



###################################################################################################################



#This is a funky way of getting around the Pickle issue of not being able to find "TBI_dataset" 
mode = input("Train/Val (tv), Test (t), or Analyze (a)")
if mode == 'tv':
    brains, labels, predictions, single_class, loss_train, loss_valid, epochLoss_train, epochLoss_valid, model_state = train_validate(train_dataset, valid_dataset,lr)
    
    date = datetime.now()
    torch.save(model_state, os.path.join(os.getcwd(), "Registered_Brains_FA/models_saved", "TBI_model-End-" + str(date.date()) + '-' + str(date.hour) + '-' + str(date.minute) +".pt"))
    
    # Saving the objects:
    with open('results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([brains, labels, predictions, single_class, loss_train, loss_valid, epochLoss_train, epochLoss_valid, test_dataset], f)
        
elif mode == 't':
# # Getting back the objects:
    del model 
    model = smp.Unet(encoder_name = ENCODER, in_channels=1, classes = 1, aux_params = aux_params)
    
    #read pickle
    with open('results.pkl','rb') as f:  
        brains, labels, predictions, single_class, loss_train, loss_valid, epochLoss_train, epochLoss_valid, test_dataset = pickle.load(f)
        
    modelPath = input("Filepath to the model you're Looking to instantiate: ")
    #modelPath = "/Users/brianmccrindle/Documents/Research/TBIFinder_Final/Registered_Brains_FA/models_saved/TBI_model-epoch2-2020-08-27-9-55.pt"
    
    thresholds = np.array(range(101)) / 100
    #thresholds = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    TPR_list = []
    FPR_list = []
    for threshold in tqdm(thresholds): #skip every other one for now
        #test the model to capture performance. Reported in the Confusion Matrix values
        CM_values = testModel(test_dataset, modelPath, threshold) #tp, fn, fp, tn
    
        TPR = CM_values[0] / (CM_values[0] + CM_values[1])
        FPR = CM_values[2] / (CM_values[2] + CM_values[3])
        TPR_list.append(TPR)
        FPR_list.append(FPR)
        print(TPR_list)
        print(FPR_list)
        
    with open('test_results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([TPR_list,FPR_list], f)
    
    #graph of the ROC curve with AUC in legend. 
    #evaluateModel.ROC_AUC(FPR_list, TPR_list) #from the utility scripts.

elif mode == 'a':
    modelPath = input("Filepath to the model you're Looking to instantiate: ")
    model.load_state_dict(torch.load(modelPath))
    model.eval() #put into evaluation mode
    
    with open('results.pkl','rb') as f:  
        brains, labels, predictions, single_class, loss_train, loss_valid, epochLoss_train, epochLoss_valid, test_dataset = pickle.load(f)
        
    loader = DataLoader(test_dataset, batch_size, shuffle = True, num_workers = num_workers)

else:
    print("Invalid Input. Only Accept tv or t for train/val and test, respectively.")



#caluclate the total number of parameters for the model
#numParams = sum(p.numel() for p in model.parameters())



# NOTES: 

#PIL is some type of holder for data so if you want the actual array, you need to apply
#img = np.array(Image.open(img_path))

#The learning rate should be reduced 
#by a factor of 10 when the validation loss fails to improve for 10 consecutive epochs.

#Early stopping should be implemented if validation loss does not improve over 50 epochs