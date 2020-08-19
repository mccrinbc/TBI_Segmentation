import os 
import random
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as ParentDataset

from skimage.transform import resize, rescale, rotate

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
        
       # image = cv2.imread(self.images_fps[self.mapping[ii]],-1)
        #label = cv2.imread(self.labels_fps[self.mapping[ii]],-1)
        
        #Current implementations of transforms only use PIL images.
        image = Image.open(self.images_fps[self.mapping[ii]]) #open as PIL image.
        label = Image.open(self.labels_fps[self.mapping[ii]])
        
        image = self.transform(image)
        label = self.transform(label)
             
        return image, label, self.images_fps[self.mapping[ii]],self.labels_fps[self.mapping[ii]]
    
    def __len__(self):
        return len(self.ids)
    
    
def datasets(images_dir, labels_dir, train_size, aug_scale, aug_angle):
    train = TBI_dataset(
        images_dir = images_dir,
        labels_dir = labels_dir,
        train_size = 0.75,
        subset="train",
        transform=transform_function(scale=aug_scale, angle=aug_angle, flip_prob=0.5),
    )
    valid = TBI_dataset(
        images_dir=images_dir,
        labels_dir = labels_dir,
        train_size = 0.75,
        subset="val",
        transform=transform_function(scale=aug_scale, angle=aug_angle, flip_prob=0.5),
    )
    return train, valid

def transform_function(degrees,scale,flip_prob):
    transform_list = []
    
    transform_list.append(transforms.RandomAffine(degrees, scale = scale))
    transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))
    #transform_list.append(transforms.ToPILImage())
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
    
class Scale(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image, label = sample
        scale = np.random.uniform(low=1.0, high=1.0 + self.scale)

        image = rescale(
            image,
            scale,
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )
        label = rescale(
            label,
            scale,
            order=0,
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )

        return image, label


class Rotate(object):

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, label = sample

        angle = np.random.uniform(low = -1 * self.angle, high = self.angle)
        image = rotate(image, angle, resize=False, preserve_range=True, mode="constant")
        label = rotate(
            label, angle, resize=False, order=0, preserve_range=True, mode="constant"
        )
        return image, label

class HorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, label = sample

        if np.random.rand() > self.flip_prob:
            return image, label

        image = np.fliplr(image).copy()
        label = np.fliplr(label).copy()

        return image, label





#PIL is some type of holder for data so if you want the actual array, you need to apply
# img = np.array(Image.open(img_path))




