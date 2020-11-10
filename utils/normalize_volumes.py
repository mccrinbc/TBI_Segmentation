#Ad-hoc script to make the background of the neural network zero. 
#We could potentially use a NN that specifically takes a brain mask and disregards the 
#background, but we're going to try and implement a regular 2D U-Net. 

#With the U-Net, we are requried to use a 256 x 256 image, but our sets are 182x182. 
#This means that by zero-padding the images, almost 50% of the image is going to be useless. 

#Currently, each slice has been normalized to itself while disregarding the background. 
#We're going to attempt to solve this by implementing a whole volume normalization with use of the brain
#mask. This removes the influence of excess zeros within the distribution. Further, we don't have 
#to worry about the background becoming non-zero, which would cause issues if there was a zero-pad boundary.

import os
from glob import glob as glob

import numpy as np
import nibabel as nib

def normalize_volume(volume, brainmask):
    #only handles [n,m,k] data. n m and k can all be the same or different. 
    normBrain = np.zeros(volume.shape) #make a volume of zeros to hold values
    brainValues = volume[np.where(brainmask == 1)]
    for ii in range(0,volume.shape[2]):
        normSlice = volume[:,:,ii]
        normSlice = (normSlice - np.min(brainValues)) / (np.max(brainValues) - np.min(brainValues))
        normSlice = np.multiply(normSlice, brainmask[:,:,ii]) #Make the background back to zero. 
        normBrain[:,:,ii] = normSlice #place the normalized slice by whole volume statistics into normBrain
        
    return normBrain
    
    
PATH = "/Users/brianmccrindle/Documents/Research/TBIFinder_Final"
os.chdir(PATH)

p1 = glob('8*')
p2 = glob('9*')
folders = p1 + p2
names = ['AD', 'FA', 'MD', 'RD']

for f in folders:
    os.chdir(os.path.join(PATH,f))
    for name in names:
        #glob returns a list, so must index at 0
        brainmask = nib.load(glob(name + '*brainmask*')[0]).get_fdata()
        volume = nib.load(os.path.join(PATH,f,glob(name + '*reg*')[0])).get_fdata()
        
        #normalized brain disregarding background. 
        normBrain = normalize_volume(volume, brainmask)
        savepath = "/Users/brianmccrindle/Documents/Research/TBIFinder_Final/Normalized_Brains" 
        
        ni_img = nib.Nifti1Image(normBrain, affine = None)
        nib.save(ni_img,os.path.join(savepath,name + '_' + f + '_' + 'norm.nii.gz'))
        
    os.chdir(PATH) #go back the parent folder

        
        
        
        
        
        
        
        