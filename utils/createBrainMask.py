#Script for making brainmasks for DeepMedic (and potential future processing)
#The output of the function should be a zero background and the entire brain as 1's. 

import os
from glob import glob as glob
import numpy as np
import nibabel as nib

def createBrainMask(volume, brainName):
    
    mask = volume + (volume == 0) #The background should *already* be zero. 
    mask = (np.ones(volume.shape) - mask) > 0 
    brainMask = mask.astype('float') #convert the boolean mask to floats. 
    ni_img = nib.Nifti1Image(brainMask, affine=None)
    nib.save(ni_img, brainName[0:7]+ '_brainmask.nii.gz')
    return brainMask
    
#Utilizing Z-Score normalization. Not sure if this is the best (could try min-max later)
#This does not constrict the rnage from 0-1, but rather ensures that the mean and std of the brain are 0 and 1, respc. 
#The background will not be 0 anymore, but the important assumption is that you run your DL model with an ROI Mask,
    #thus, removing the background from processing. 
def normalizeRegisteredBrain(volume, mask, brainName):
    logicalMask = (mask == 1) #force the mask to be of logical type. 
    mean = volume[logicalMask].mean() #Pull out only the values from the brain, exclude the background. 
    std = volume[logicalMask].std()
    normalized = nib.Nifti1Image((volume - mean) / std, affine = None) #z_score normalization. 
    nib.save(normalized, brainName[0:7]+ '_norm.nii.gz')
 

PATH = "/Users/brianmccrindle/Documents/Research/TBIFinder_Final"
os.chdir(PATH)
patients = sorted(list(filter(('.DS_Store').__ne__, os.listdir()))) 

for patient in patients:
    os.chdir(os.path.join(PATH,patient))
    brains = glob('*reg*')
    
    for brainName in brains: #list of strings
        volume = nib.load(brainName).get_fdata()
        brainMask = createBrainMask(volume,brainName)
        
        #Remove since we do not want to normalize per slice anymore. 
        #Would be interesting to try and use this and compare performance to whole-volume normalization. 
        
        #normalizeRegisteredBrain(volume, brainMask, brainName)




        