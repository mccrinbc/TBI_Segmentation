#create the labels for each of the tensor maps for each brain. 
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib

def createLabel(folderLocation, info, Patient_scores, MZS, Mstd, FZS, Fstd):
    #folderLocation: folder containing all ROI masks with a single complete brain corresponding to the parametric tensor.
    #info: contains information regarding each patient's identified sex and age. 
    #MZS, Mstd, FZSm Fstd: Male / Female Means / Stds for each chosen ROI. 
    
    brainMask = np.zeros([182,218,182])
    
    patientIndex = np.where(int(patient) == info.PatientNum)
    scanType = os.listdir()[0][0:2] #get the scan type through the name of the first file
    
    if not(patientIndex): #If the patient index is empty (ie, patient =! info.PatientNum anywhere)
        print("Patient number does not exist")
        sys.exit("Error Message")
    
    Sex = info.Sex[patientIndex[0]].values[0] #sex of the patient
    if Sex == 'M':
        Z_scores = MZS
        stds = Mstd
    elif Sex == 'F':
        Z_scores = FZS
        stds = Fstd
    else:
        print('Sex specified in excel sheet does not match any Z-Scoring')
        sys.exit()
    
    Age = info.Age[patientIndex[0]].values[0]
    
    #Pull the relevant distribution info.
    Zindex = [ii for ii, x in enumerate(Z_scores.Age == Age) if x][0] #enumerate through a list of True/False to find index
    Z_score_info = Z_scores.iloc[Zindex:Zindex+4,:] #Pull the Z Scores
    std_info     = stds.iloc[Zindex:Zindex+4,:] #Pull stds
    
    #Get info on this patient. 
    Pat_index = [ii for ii, x in enumerate(Patient_scores.Patient_Number == float(patient)) if x][0] #enumerate through a list of True/False to find index
    Patient_info = Patient_scores.iloc[Pat_index:Pat_index+4,:]   #Pull corresponding patient info 
    
    #This determines the scan type location in the list. 
    scan_index = [ii for ii, x in enumerate(Z_score_info.Class == scanType) if x][0] #pull the specific scan out of the subset
    indicator = checkROI(scan_index, Patient_info, Z_score_info, std_info) #indicator to determine which ROIs are OOD
    
    mask_list = sorted(list(filter(('.DS_Store').__ne__, os.listdir())))[:-1]
    indicies = np.where(indicator == 1) #find the indicies where we know those patient ROIs are OOD.
    
    for j in range(0,len(indicies[0])):
        index = indicies[0][j]
        ROI = nib.load(mask_list[index]).get_fdata() > 0 #grab the ROI mask
        brainMask = brainMask + ROI #create the labels for on the mask.
        
    brainMask[brainMask > 0] = 1 #There is some overlap in the ROIs. 
    return brainMask, indicator
        
        
    
            
def checkROI(scan_index, Patient_info, Z_score_info, std_info):
    #utility function to check if the ROI is within / outside distribution set by Z_scores
    patientDist = Patient_info.iloc[scan_index,:].values[2:]
    Z_score_dist = Z_score_info.iloc[scan_index,:].values[2:]
    std_dist = std_info.iloc[scan_index,:].values[2:]
 
    indicator = []
    for ii in range(0,18):
        if patientDist[ii] > (Z_score_dist[ii] + 2*std_dist[ii]) or patientDist[ii] < (Z_score_dist[ii] - 2*std_dist[ii]):
            #print(patientDist[ii], Z_score_dist[ii],std_dist[ii])
            indicator.append(1)
        else:
            indicator.append(0)
    return np.asarray(indicator)



PATH = "/Users/brianmccrindle/Documents/Research/TBIFinder_Reduced/"
os.chdir(PATH)

data = pd.ExcelFile('TBI_Finder_Reporting.xlsx')
info = pd.read_excel(data,data.sheet_names[0]) 
Patient_scores = pd.read_excel(data, data.sheet_names[1])
MZS = pd.read_excel(data,data.sheet_names[2])
Mstd = pd.read_excel(data,data.sheet_names[3])
FZS = pd.read_excel(data,data.sheet_names[4])
Fstd = pd.read_excel(data,data.sheet_names[5])

indicator_array = np.zeros([Patient_scores.shape[0],Patient_scores.shape[1] - 2])

os.chdir(PATH + 'Brains/')
PATH = os.getcwd() #redefine PATH to make it static.
patients = sorted(list(filter(('.DS_Store').__ne__, os.listdir()))) #removes the .DS.Store hidden folder

row = 0
for patient in patients:
    subDirs = [x[0] for x in os.walk(patient)] #recurssively determines the subdirectories within the PATIENT parent directory
    subDirs = sorted(subDirs[1:]) #removes the parent directory name in the set. Sort the same way shown in Finder.
    
    for ii in range(0,4): #there will only ever be 4 tensor maps. 
        print(ii, patient)
        os.chdir(os.path.join(PATH,subDirs[ii]))
        label, indicator = createLabel(os.getcwd(), info, Patient_scores, MZS, Mstd, FZS, Fstd)
        
        indicator_array[row,:] = indicator
        prefix = os.listdir()[0][0:7] #get the name of the file
        
        #if affine = np.eye(4), image is flipped horizontally. 
        ni_img = nib.Nifti1Image(label, affine=None) #Affine: defining the position of the image with respect to some frame of reference
        nib.save(ni_img, prefix + '_label.nii.gz')
        
        row += 1 #increase the row index
        os.chdir(PATH) #Need to do this for subDirs to have more to find
        
        
        
        
        
        
        
        
        
        
        
        
        