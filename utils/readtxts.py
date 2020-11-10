#Grab all values from the txt files in folder 
import os 
from glob import glob as glob 
import numpy as np
import pandas as pd

def readTxts(meanPATH, stdPATH):
    means = []
    stds = []
    
    for ii in range(2): #loop twice
        if ii == 0:
            PATH = meanPATH
        else:
            PATH = stdPATH
            
        os.chdir(PATH)
        files = sorted(glob("*.txt")) #sort the files in alphabetical order
        for f in files:
            file = open(f,"r")
            valString = file.readline()
            num = float(valString.rstrip()) #convert the string with a \n character into int. 
            if ii == 0:
                means.append(num)
            else:
                stds.append(num)
                
    means = np.asarray(means)
    means = means.reshape((1,18))
    stds = np.asarray(stds)
    stds  = stds.reshape((1,18))

    return means,stds

Ts = ['AD','FA','MD','RD'] #tensors
Cases = ["DTI_Male_22_25", "DTI_Male_27","DTI_Male_36_40","DTI_Male_44","DTI_Male_64","DTI_Male_66"]
Cases = ["DTI_Female_22", "DTI_Female_46","DTI_Female_50","DTI_Female_57"]
meanMatrix = np.zeros([4*len(Cases),18]) 
stdMatrix  = np.zeros([4*len(Cases),18]) 

for ii in range(0,len(Cases)):
    #print(ii)
    for j in range(0,len(Ts)):
        #print(j)
        meanPATH = "/Users/brianmccrindle/Desktop/Female_Z_Scores/" + Ts[j] + "/" + Cases[ii] + "/Mean_Final/Full_Mean"
        stdPATH  = "/Users/brianmccrindle/Desktop/Female_Z_Scores/" + Ts[j] + "/" + Cases[ii] + "/Mean_Final/Full_sd"
        means,stds = readTxts(meanPATH,stdPATH)
        meanMatrix[j + 4*(ii),:] = means
        stdMatrix[j  + 4*(ii),:] = stds
