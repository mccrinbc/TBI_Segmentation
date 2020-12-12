import os 
import random
import numpy as np
from PIL import Image

from datetime import datetime
import pickle
from tqdm import tqdm #loading bar  

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as ParentDataset

import torchvision
from torchvision import transforms
from torchvision.transforms import Compose

import sklearn.metrics

import segmentation_models_pytorch as smp #model we're using for now. 
import evaluateModel

from matplotlib import pyplot as plt

import batch_testing_script #user defined.
import image_indicies

#Errors associated to potantial randomness / non-deterministic behaviour is a VERY common issue in PT. 
#Look at the following github discussion for more information: 
#https://github.com/pytorch/pytorch/issues/7068
#      sbelharbi commented on Apr 19, 2019

seed = 1010
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True



class TBI_dataset(ParentDataset): #Obtain the attributes of ParentDataset from torch.utils.data
#Finds Image and Label locations, creates random list of indicies for training / val / testing sets to be called
    def __init__(
        self,
        images_dir,
        labels_dir,
        subset="train",
        transform = None, #base transformation is into Tensor. 
        mapping = None #there is no mapping initially. model should stop if no mapping provided. 
        #seed=seed, #We'll get the same thing everytime if we keep using the same seed. 
    ):
        
        #filter and sort the list
        self.ImageIds = sorted(list(filter(('.DS_Store').__ne__,os.listdir(images_dir)))) 
        self.LabelIds = sorted(list(filter(('.DS_Store').__ne__,os.listdir(labels_dir))))
        
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ImageIds] #full_paths to slices
        self.labels_fps = [os.path.join(labels_dir, image_id) for image_id in self.LabelIds] #full_paths to labels

        if mapping == None:
            print("Mapping Required for Model to Run!")

        #We define a mapping to use when calling the Dataset loader based on the parameter "mapping"
        if subset == "train":
            self.mapping = mapping['train']['set']
            print(self.mapping)
        elif subset == "val":
            self.mapping = mapping['val']['set']
            print(self.mapping)
        elif subset == "test":
            self.mapping = mapping['test']['set']
            print(self.mapping)
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
    
    
def datasets(images_dir, labels_dir, train_size, aug_angle, aug_scale, flip_prob, mapping):
    
    #mapping = return_image_indicies(images_dir,labels_dir, train_size, random_sampling = True)
    
    train = TBI_dataset(
        images_dir = images_dir,
        labels_dir = labels_dir,
        subset = "train",
        transform = transform_function(degrees=aug_angle, scale=aug_scale, flip_prob=flip_prob),
        mapping = mapping
    )
    valid = TBI_dataset(
        images_dir = images_dir,
        labels_dir = labels_dir,
        subset = "val",
        transform = transform_function(degrees=aug_angle, scale=aug_scale, flip_prob=flip_prob),
        mapping = mapping
    )
    
    test = TBI_dataset(
        images_dir = images_dir,
        labels_dir = labels_dir,
        subset= "test",
        transform = transform_function(degrees=0, scale = [1,1], flip_prob = 0), #make sure nothing changes. 
        mapping = mapping
    )
    
    return train, valid, test

def transform_function(degrees,scale,flip_prob):
    transform_list = []
    
    transform_list.append(transforms.RandomAffine(degrees, scale = scale))
    transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))
    transform_list.append(transforms.Pad(37)) #all images should be 182x182 before padding. 
    transform_list.append(transforms.ToTensor())
    
    return Compose(transform_list)

def Weights(labels, device):
    #expects an [batch_size,c,n,n] input 
    
    weights = torch.rand(labels.shape) #create a random tensor of weight values. 
    weights = weights.to(device) #put everything onto the GPU. 
    
    for batch_num in range(0,labels.shape[0]):
        num_ones = torch.sum(labels[batch_num,0,:,:]);
        resolution = labels.shape[2] * labels.shape[3]
        num_zeros = resolution - num_ones 
        
        #https://discuss.pytorch.org/t/how-to-apply-a-weighted-bce-loss-to-an-imbalanced-dataset-what-will-the-weight-tensor-contain/56823/2
        #Weight for the positive class
        pos_weight = num_zeros / resolution #should be close to 1.
        neg_weight = 1 - pos_weight 
        
        #create 1s tensor, put to GPU.
        ones = torch.ones(labels.shape[2],labels.shape[3])
        ones = ones.to(device)
        
        weights[batch_num,0,:,:] = ones*neg_weight + labels[batch_num,0,:,:]*pos_weight
        
    #this keeps the clas imbalance in check
    return weights,pos_weight,neg_weight #should be a tensor. 

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

def train_validate(train_dataset, valid_dataset, lr):
    
    earlystop = False 
    
    if torch.cuda.is_available():
        dev ="cuda:2"
    else:
        dev = "cpu"
        
    dev = torch.device(dev)
    
    #this might break, remove worker_init_fn = _init_fn(num_workers)) if so
    train_loader = DataLoader(train_dataset, batch_size, shuffle = True, num_workers = num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle = True, num_workers = num_workers)
    
    model.to(dev) #cast the model onto the device 
    optimizer = optim.Adam(model.parameters(), lr = lr) #learning rate should change 
    
    loss_function = torch.nn.BCELoss() #this takes in a weighted input 
    #loss_function = smp.utils.losses.DiceLoss()
    #loss_function = DiceLoss()
    #metrics = [smp.utils.metrics.IoU(threshold=0.5)]
    
    loss_train = []
    loss_valid = []
    
    loss_train_batchset = []
    loss_valid_batchset = []
    
    epochLoss_train = []
    epochLoss_valid = []
        
    for epoch in range(EPOCHS):
        image_count = 0
        for phase in ["train","val"]:
            print(lr)
            #This determines which portions of the model will have gradients turned off or on. 
            if phase == "train":
                model.train() #put into training mode
                loader = train_loader
            else:
                model.eval() #evaluation mode.
                loader = valid_loader
            
            lr_flag = True #flag is set to False if LR has changed and reset once we go back into training. 
            for ii, data in enumerate(loader): 
                
                brains = data[0] #[batch_size,channels,height,width] 
                labels = data[1]
                
                image_count += len(brains)
                print(epoch, phase, ii, image_count)
                
                brains,labels = brains.to(dev), labels.to(dev) #put the data onto the device
                predictions, single_class = model(brains) #single class is not a useful output. 
                
                predictions = torch.sigmoid(predictions) #using this so that the output is bounded [0,1]
                single_class = torch.sigmoid(single_class)
                
                weights = Weights(labels,dev) #generate the weights for each slice in the batch
                loss_function.pos_weight = weights                    
                
                #Implementing BCE Loss
                loss = loss_function(predictions, labels) #loss changes here. 
                
                if phase == "train":
                    #employ this so we don't get multiples in the same list. 
                    if (loss_valid and ii == 0): #if loss_valid is NOT empty AND it's the first time we see this in the loop
                        #epochLoss_valid.append(loss_valid[-1]) #append the last value in the 
                        epochLoss_valid.append(np.mean(loss_valid_batchset))
                        loss_valid_batchset = []
                        
                        
                    model.zero_grad() #recommended way to perform validation
                    #for p in model.parameters(): p.grad = None #This also sets the gradients to zero. better?
                    
                    loss_train.append(loss.item())
                    loss_train_batchset.append(loss.item()) #append to list of losses in the batch
                    
                    loss.backward()
                    optimizer.step()
                    
                    print(f"Phase: {phase}. Epoch: {epoch}. Loss: {loss.item()}") 
               
                else:
                    if (loss_train and ii == 0):#if loss_valid is NOT empty AND it's the first time we see this in the loop
                        #epochLoss_train.append(loss_train[-1]) #append the last value in the loss_train list.
                        epochLoss_train.append(np.mean(loss_train_batchset))
                        loss_train_batchset = []
                        
                    loss_valid.append(loss.item())
                    loss_valid_batchset.append(loss.item())
                    
                    print(f"Phase: {phase}. Epoch: {epoch}. Loss: {loss.item()}") 
                    
                    #learning rate changes and early stopping
                    #This only occurs during validation.
                    if epoch > 0:
                        if (epoch % 10) == 0: #if the epoch is divisable by 10
                            meanVal = np.mean(loss_valid[epoch - 5 : epoch])
                            print(meanVal, np.abs((meanVal - loss.item()) / meanVal) <= 0.4)
                            if (np.abs((meanVal - loss.item()) / meanVal) <= 0.4 and lr_flag): #if the % difference is small
                            #if epoch == 40 and lr_flag: #doing this for now. 
                                lr = lr * 0.1
                                lr_flag = False
                                for param_group in optimizer.param_groups:
                                    #print(optimizer)
                                    print('Reducing the Learning Rate: ', lr )
                                    param_group['lr'] = lr
                                    #print(optimizer)
                        
                        if (epoch % 100) == 0:
                            meanVal = np.mean(loss_valid[epoch - 50 : epoch])
                            if np.abs((meanVal - loss.item()) / meanVal) <= 0.05:
                                earlystop = True 
               
                #Implementation of early stopping
                if earlystop == True:
                    date = datetime.now()
                    torch.save(model.state_dict(), os.path.join(os.getcwd(), 'results_TBI_model-' + str(date.date()) + '-' + str(date.hour) + '-' + str(date.minute) + '-EARLYSTOP.pt')) #save the model 
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
    #We could potentially make this value an average of the relevant loss_valids
    #epochLoss_valid.append(loss_valid[-1])
    epochLoss_valid.append(np.mean(loss_valid_batchset))
    
    print('Final Learning Rate: ', lr)
    return brains, labels, predictions, single_class, loss_train, loss_valid, epochLoss_train, epochLoss_valid, model.state_dict(), lr

#This function expects singular images. 
#Also biased from high class imbalance. function currently not in use. 
def IoU(prediction, label):
    #Prediction IoU
    intersection = int(torch.sum(torch.mul(prediction,label)))
    union = int(torch.sum(prediction) + torch.sum(label)) - intersection
    IOU_predicted = intersection / (union + 0.0001) #for stability
    mean_IoU = IOU_predicted
    
    #Not including background IoU
    #Background IoU
    #all_zeros = (prediction + label) > 0 #before the inversion
    #intersection = int(torch.sum(~all_zeros))
    #union = int(torch.sum(~ (prediction > 0)) + torch.sum(~ (label > 0)) - intersection)
    #IOU_background = intersection / (union + 0.0001)
    
    #mean_IOU = (IOU_background + IOU_predicted)/2
    return mean_IoU

def testModel(test_dataset, modelPath, threshold): #model = the model class = smp.UNet()

    total_images = 0
    mean_IoUs = []
    loss_set_batch = []
    loss_set = []
    CM_values = [0,0,0,0] #tp, fn, fp, tn
    model.load_state_dict(torch.load(modelPath))
    
    loss_function = torch.nn.BCELoss() #implementing the loss function to show loss for each threshold. 
    
    if torch.cuda.is_available():
        dev ="cuda:2"
    else:
        dev = "cpu"
        
    dev = torch.device(dev)
    model.to(dev) 
    model.eval() #evaluation mode to turn off the gradients / training. 
    
    #turn shuffle off 
    loader = DataLoader(test_dataset, batch_size = 12, shuffle = False, num_workers = num_workers)
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
        
        weights = Weights(labels, dev)
        loss_function.pos_weight = weights                    
                
        #Implementing BCE Loss
        preds = predictions > threshold
        preds = preds.float() #cast the bool tensor into float32 for loss function
        loss = loss_function(preds, labels) #loss changes here. 
        loss_set_batch.append(loss.item()) #append the loss of the batch
        
        predictions_numpy = predictions.cpu().detach().numpy()
        labels_numpy = labels.cpu().detach().numpy()
        for j in range(predictions.shape[0]):
            #labels = [False, True] are needed to make sure we don't have errors with the shape of CM
            #mean_IoUs.append(IoU(predictions[j,0,:,:].cpu() > threshold, labels[j,0,:,:].cpu() > threshold)) #determine the mean IoU
            CM = sklearn.metrics.confusion_matrix(labels_numpy[j,0,:,:].ravel(), predictions_numpy[j,0,:,:].ravel() > threshold, labels = [True,False])
            try: 
                CM_values[0] = CM_values[0] + CM[0][0]
                CM_values[1] = CM_values[1] + CM[0][1]
                CM_values[2] = CM_values[2] + CM[1][0]
                CM_values[3] = CM_values[3] + CM[1][1]
            except:
                print("Error in Appending")
                return CM, CM_values
            
        loss_set.append(np.mean(loss_set_batch)) #append the mean loss of the entire set 
            
    del loader #delete loader, might be wrong to do this
    return np.divide(CM_values , (total_images*(256*256))), np.mean(loss_set) #, np.divide(np.sum(mean_IoUs), len(mean_IoUs))

#######################################################################################################################

## Read the tests from batch_testing_script ##
tests = batch_testing_script.report_tests()
batch_results = []

images_dir = "/home/mccrinbc/Data_Removed_Useless_Slices/normalized_slices"
labels_dir = "/home/mccrinbc/Data_Removed_Useless_Slices/slice_labels"
#Train size is always the same (for now). Implement a better solution later. 
#train_size = tests[ii]['train_size']
train_size = 0.75

#We need to run the train/val/test indicies split before we go into the for loop. 
#For this reason, we've developed a small script to do this sampling for us, and to confirm that it's consistant. 
mapping = image_indicies.return_image_indicies(images_dir, labels_dir, train_size, seed, random_sampling = True)

for ii in tests:
	batch_size = tests[ii]['batch_size']
	EPOCHS = tests[ii]['EPOCHS']
	lr = tests[ii]['lr']
	aug_scale = tests[ii]['aug_scale']
	aug_angle = tests[ii]['aug_angle']
	flip_prob = tests[ii]['flip_prob']
	num_workers = tests[ii]['num_workers']

	#images_dir = "/Users/brianmccrindle/Documents/Research/TBIFinder_Final/Registered_Brains_FA/test_slices"
	#labels_dir = "/Users/brianmccrindle/Documents/Research/TBIFinder_Final/Registered_Brains_FA/test_labels"

	#smp specific variables
	ENCODER = tests[ii]['ENCODER']
	aux_params=dict(
	    pooling='avg',             # one of 'avg', 'max'
	    dropout= tests[ii]['dropout'],  # dropout ratio, default is None
	    #activation='softmax2d',    # activation function, default is None. This is the output activation. softmax2d specifies dim = 1 
	    classes=1,                 # define number of output labels
	)

	#classes = 2 for the softmax transformation.
#	model = getattr(smp, model_arch)
#	setattr(model, 'encoder_name', ENCODER)
#	setattr(model, 'in_channels' , 1)
#	setattr(model, 'classes'     , 1)
#	setattr(model, 'aux_params', aux_params)
	model = smp.Unet(encoder_name = ENCODER, in_channels=1, classes = 1, aux_params = aux_params)

	train_dataset, valid_dataset, test_dataset = datasets(images_dir, labels_dir, train_size, aug_angle, aug_scale, flip_prob, mapping) #now takes in the mapping dictionary for image samples. 

	#Training Cell 
	brains, labels, predictions, single_class, loss_train, loss_valid, epochLoss_train, epochLoss_valid, model_state, lr_final,  = train_validate(train_dataset, valid_dataset, lr)
	    
	date = datetime.now()
	base = "results_TBI_model-End-" + str(date.date()) + '-' + str(date.hour) + '-' + str(date.minute)
	folder_path = r'/home/mccrinbc/' + base
	pkl_name = base + '.pkl'
	model_name = base + '.pt'

	pkl_location = os.path.join(folder_path, pkl_name)

	if not os.path.exists(folder_path):
	    os.makedirs(folder_path)

	torch.save(model_state, os.path.join(folder_path, base +".pt"))
	    
	# Saving the objects:
	with open(pkl_location, 'wb') as f:  # Python 3: open(..., 'wb')
	    pickle.dump([brains, labels, predictions, single_class, loss_train, loss_valid, epochLoss_train, epochLoss_valid, test_dataset], f)

#Look at the train / validation loss 
#Possible variables that might cause the validation loss to jump are:
    # - Learning rate is too high in later epochs
    # - Model could be too big?
    # - Batch size could be too small causing loss in generality between epochs. 
    
	plt.figure()
	plt.plot(np.arange(0, EPOCHS, 1), epochLoss_train)
	plt.plot(np.arange(0, EPOCHS, 1), epochLoss_valid)
	plt.ylabel('Per Epoch Loss')
	plt.xlabel('Epoch')
	plt.title('BCELoss vs Epochs. Initial LR = ' + str(lr) + '. ' + ENCODER + ' : ' + str(EPOCHS))
	plt.legend(['Train Loss: ' + str(epochLoss_train[-1]), 'Valid Loss: ' + str(epochLoss_valid[-1])], loc = "upper right")
	plt.ylim([0,0.6])

	plt.savefig(os.path.join(folder_path, 'Train-Val_Loss.png'))


	#Create names for data to be stored.
	model_name = base + '.pt'
	modelPath = os.path.join(folder_path, model_name)


	#Testing Loop. 
	del model #This removes any confliction with an existing model running on the GPU. 
#	model = getattr(smp, model_arch)
#	setattr(model, 'encoder_name', ENCODER)
#	setattr(model, 'in_channels' , 1)
#	setattr(model, 'classes'     , 1)
#	setattr(model, 'aux_params', aux_params)
	model = smp.Unet(encoder_name = ENCODER, in_channels=1, classes = 1, aux_params = aux_params)
	    
	#read pickle
	if 'pkl_location' not in locals():
	    folder_path = '/home/mccrinbc/results_TBI_model-End-2020-10-05-11'
	    pkl_location = "/home/mccrinbc/results_TBI_model-End-2020-10-05-11/results_TBI_model-End-2020-10-05-11.pkl"
	print(pkl_location)

	with open(pkl_location,'rb') as f:  
	    brains, labels, predictions, single_class, loss_train, loss_valid, epochLoss_train, epochLoss_valid, test_dataset = pickle.load(f)

	if 'modelPath' not in locals():
	    modelPath = 'input something'

	#model = model.load_state_dict(torch.load(modelPath)) #see if this works. 
	#modelPath = "/Users/brianmccrindle/Documents/Research/TBIFinder_Final/Registered_Brains_FA/models_saved/TBI_model-epoch2-2020-08-27-9-55.pt"
	    
	thresholds = np.arange(0,1.05,0.05) #skipping every other element, [[0.   0.05 0.1 ... 1]
	#thresholds = np.arange(0,0.15,0.05) #skipping every other element, [[0.   0.05 0.1 ... 1]
	#thresholds = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
	TPR_list = [] #This is also known as RECALL. 
	FPR_list = []
	IoUs = []
	Dice = []
	total_error = []
	precision = []
	BCE_loss_thresh= []

	for threshold in thresholds:
	    print(threshold)
	    #test the model to capture performance. Reported in the Confusion Matrix values
	    CM_values, BCE_loss = testModel(test_dataset, modelPath, threshold) #tp, fn, fp, tn, [mean_IoUs]
	    
	    BCE_loss_thresh.append(BCE_loss)
	    TPR = CM_values[0] / (CM_values[0] + CM_values[1])
	    FPR = CM_values[2] / (CM_values[2] + CM_values[3])
	    TPR_list.append(TPR)
	    FPR_list.append(FPR)
	    
	    #IoUs, Dice, Total Error, and Precision. 
	    IoUs.append(CM_values[0] / (CM_values[0] + CM_values[1] + CM_values[2]))  #IoU = TP / (TP + FN + FP)
	    Dice.append(2 * CM_values[0] / (2 * CM_values[0] + CM_values[1] + CM_values[2])) #Dice = 2TP / (2TP + FN + FP)
	    total_error.append(CM_values[1] + CM_values[2]) #Error = FP + FN. Weighted equally for now. 
	    precision.append(CM_values[0] / (CM_values[0] + CM_values[2])) #Precision = TP / (TP + FP)

	TPR_list = np.nan_to_num(TPR_list, nan = 0) #Replace any nans with 0.
	FPR_list = np.nan_to_num(FPR_list, nan = 0) #Replace any nans with 0.
	IoUs = np.nan_to_num(IoUs, nan = 0) #Replace any nans with 0. 
	Dice = np.nan_to_num(Dice, nan = 0) #Replace any nans with 0. 
	precision = np.nan_to_num(precision, nan = 0) #Replace any nans with 0.


	#Within the same cell. Save the information from testing. 
	difference_array = np.array(TPR_list) - (1-np.array(FPR_list))
	best_acc_thresh = thresholds[abs(difference_array).argmin()] #Thresholds is already defined.  

	best_IoU_thresh = thresholds[np.where(IoUs == np.max(IoUs))][0]
	best_Dice_thresh = thresholds[np.where(Dice == np.max(Dice))][0]

	results_name = 'test_results.pkl'
	results_location = os.path.join(folder_path, results_name)

	#This is saving the test results into a pkl file
	with open(results_location, 'wb') as f:  # Python 3: open(..., 'wb')
	    pickle.dump([ENCODER, EPOCHS, lr_final, TPR_list, FPR_list, precision, thresholds, best_acc_thresh, IoUs, Dice, BCE_loss_thresh], f)
        #pickle.dump([TPR_list, FPR_list, precision, thresholds, best_acc_thresh, IoUs, Dice], f)
	


	## Looking at how well the data is doing ##
	difference_array = np.array(TPR_list) - (1-np.array(FPR_list))
	#Utility to Plot Relavent Information 
	print(EPOCHS)
	#print('Best IoU Threshold',best_IoU_thresh)
	print('Best Accuracy Threshold:', best_acc_thresh)

	sens = TPR_list[abs(difference_array).argmin()]
	spec = 1 - FPR_list[abs(difference_array).argmin()]

	print('Optimal Accuracy Sens:', sens)
	print('Optimal Accuracy Spec:', spec)
	print('')

	prec_recall_thresh = abs(precision - TPR_list)[0:-2] #remove the last element
	optimal_index = np.where(prec_recall_thresh == np.min(prec_recall_thresh))
	optimal_prec_rec_thresh = thresholds[optimal_index][0]
	prec_thresh = precision[optimal_index][0]
	recall_thresh = TPR_list[optimal_index][0]

	#Here is where we're going to put all of the data into a txt file. 
	#base = base name of the folder, pkl, and pt files. 
	batch_results.append([base, tests[ii]['ENCODER'], EPOCHS, epochLoss_train[-1],epochLoss_valid[-1],best_IoU_thresh,np.max(IoUs),best_Dice_thresh,np.max(Dice), optimal_prec_rec_thresh, prec_thresh, recall_thresh, best_acc_thresh,sens,spec])


#Outside of the for-loop now. 
with open('batch_testing_results.txt', 'w') as f:
    for item in batch_results:
        f.write("%s\n" % item)










