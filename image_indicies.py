import os
import random

def return_image_indicies(images_dir, labels_dir, train_size, dataset_id, seed, random_sampling = True):
    #This function returns a dictionary of indicies that correspond to images in the dataset. 
    #Separate function and called ONCE to ensure that multiple sets are not being created. 
    
    patients = {}
    patients['Female'] = [100307, 102311, 102816, 103515, 104012, 105014, 106521, 107018, 108121, 108323, 111009, 111211, 111413, 112314, 112819, 113215, 113821, 114116, 114217, 117021, 117122, 118023, 118528, 119833, 120010, 120111]
    patients['Male'] = [100206, 100610, 101309, 101410, 102109, 102513, 102715, 103111, 105216, 106319, 110613, 112112, 114621, 116423, 116726, 118225, 118932, 119025, 121416, 122620, 123117, 125222, 129028, 130013, 130114]
    
    #filter and sort the list
    ImageIds = sorted(list(filter(('.DS_Store').__ne__,os.listdir(images_dir)))) 
    LabelIds = sorted(list(filter(('.DS_Store').__ne__,os.listdir(labels_dir))))

    images_fps = [os.path.join(images_dir, image_id) for image_id in ImageIds] #full_paths to slices
    labels_fps = [os.path.join(labels_dir, image_id) for image_id in LabelIds] #full_paths to labels

    if random_sampling == True:
        samples = list(range(0,len(patients[dataset_id]))) #create a list of numbers #create a list of numbers
        random.seed(seed) #set the seed          

        #random sample train_size amount and then do a train/validation split 
        indicies = random.sample(samples,round(train_size*len(patients[dataset_id])))
        val_indicies = indicies[0:round(len(indicies)*0.15)] #15% of the dataset goes towards validation.
        train_indicies = indicies[round(len(indicies)*0.15) : len(indicies)]

        test_indicies = samples
        for j in sorted(indicies, reverse = True): #remove the train/val indicies from test set
            del test_indicies[j]
        
        ## Making the mapping ##
        index_set = [[0]*len(images_fps) for _ in range(3)] #create a set of zeros that are independant lists. 
        flag = -1
        for sets in [train_indicies, val_indicies, test_indicies]: #Iterate through each of the sets. 
            flag = flag + 1
            for z in sets: #iterate each value in a single set
                for ii in range(0,len(images_fps)):
                    if str(patients[dataset_id][z]) in images_fps[ii]:
                        index_set[flag][ii] = 1
                        
        #Where index_set == 1, pull out that index from samples.
        
        samples = list(range(0,len(images_fps)))
        train_mapping = list(compress(samples, index_set[0])) 
        train_mapping = random.sample(train_mapping,len(train_mapping))
        
        val_mapping   = list(compress(samples, index_set[1]))
        val_mapping = random.sample(val_mapping,len(val_mapping))
        
        test_mapping  = list(compress(samples, index_set[2]))
        test_mapping = random.sample(test_mapping,len(test_mapping))
         
        
    mapping = {}
    mapping['train'] = { 'set' : train_mapping}
    mapping['val'] = { 'set' : val_mapping}
    mapping['test'] = { 'set' : test_mapping}
    
    return mapping