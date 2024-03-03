# Basic Imports.....
from copy import copy
import datetime
from glob import glob
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

# pipeline specific imports..
import constants
import model
import data_loader

# PyTorch related imports....
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2  as transforms
import tensorflow as tf


# Impoving code reproducability...
seed = 10
np.random.seed(seed)
torch.manual_seed(seed)

# setting up gpu details
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype  = torch.float32

# loading image paths & labels....
label_dict = {}
with open(label_path, 'rb') as handle:
    label_dict = pickle.load(handle)

patient_ids = np.array(list(set([i.split("_")[0] for i in label_dict.keys()])))
print("Total number of input images       : ",len(label_dict))
print("Total number of unique patient ids : ",len(patient_ids))

# Details of model weight saving...
print("Model save name :",model_save_name)
if not os.path.exists(save_dir_name + model_name):
    os.makedirs(save_dir_name + model_name, exist_ok = True)
    os.makedirs(save_dir_name + model_name + "/logs/", exist_ok = True)
    os.makedirs(save_dir_name + model_name + "/logs/scalars/", exist_ok = True)

# Setting up tensorboard logging....
logdir = save_dir_name + model_name + "/logs/scalars/" + model_save_name
train_summary_writer = tf.summary.create_file_writer(logdir)

print("Splitting the dataset ....")
train_index, val_index, _, _ = train_test_split(list(range(len(patient_ids))), 
                                                [1]*len(patient_ids), 
                                                test_size=0.2, 
                                                random_state = seed)

# splitting based on patient ids to avoid any data contamination....
train_patient_ids = patient_ids[train_index]
valid_patient_ids = patient_ids[val_index]

# loading image paths : 
train_files = []
valid_files = []
train_cnt = {"healthy" : 0, "suspects" : 0, "glaucoma" : 0}
val_cnt   = {"healthy" : 0, "suspects" : 0, "glaucoma" : 0}
for i in os.listdir(dataset_path):
    pid = i.split("_")[0]
    if pid in train_patient_ids:
        train_cnt[label_dict[i]] += 1
        train_files.append(os.path.join(dataset_path,i))
    elif pid in valid_patient_ids:
        val_cnt[label_dict[i]] += 1
        valid_files.append(os.path.join(dataset_path,i))


print("Label distribution :")
print("Training split   : Healthy : {} | Suspects : {} | Glaucoma : {}".format(train_cnt["healthy"] / len(train_files),
                                                                               train_cnt["suspects"] / len(train_files),
                                                                               train_cnt["glaucoma"] / len(train_files)))

print("Validation split : Healthy : {} | Suspects : {} | Glaucoma : {}".format(val_cnt["healthy"] / len(valid_files),
                                                                             val_cnt["suspects"] / len(valid_files),
                                                                             val_cnt["glaucoma"] / len(valid_files)))

# building base model
pre_proc_func = None
if model_name in ['resnet','mobilenet','densenet']:
    # imagenet pre-processing pipeline...
    pre_proc_func =  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

# building dataloaders....
training_data = GenerateDataset(image_files = train_files,
                                labels_dict = label_dict,
                                img_res = input_shape[0],
                                augment = True,
                                shuffle   = True,
                                transform = pre_proc_func,
                                split_flag = "training")

validation_data = GenerateDataset(image_files = valid_files,
                                  labels_dict = label_dict,
                                  img_res = input_shape[0],
                                  augment = False,
                                  transform = pre_proc_func,
                                  split_flag = "validation")

train_dataloader = DataLoader(training_data, 
                              batch_size = batch_size, 
                              shuffle=True,
                              num_workers=4)

valid_dataloader = DataLoader(validation_data, 
                              batch_size = batch_size, 
                              shuffle=True,
                              num_workers=4)

print("Number of Training   steps : ",len(train_dataloader))
print("Number of Validation steps : ",len(valid_dataloader))

# All model constants are specified in the constants file
# Make experimental changes in it....
base_model = build_model()

# defining loss & lr decay
criterion torch.hub.load('adeelh/pytorch-multi-class-focal-loss',
                          model='FocalLoss',
                          alpha = torch.tensor(list(training_data.class_weights.values())),
                          gamma=2,
                          reduction='mean',
                          force_reload=False)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr = pre_freeze_lr, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

def train_model(model, 
                criterion, 
                optimizer, 
                scheduler, 
                num_epochs=25):

    since = time.time()
   
    # defining metric holding variables....
    best_acc = 0.0

    # Training Epochs....
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Training steps.....
        model.train() 
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_dataloader, position = 0, leave = True):
            
            # moving tensors to gpu....
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # applying computed gradients....
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        # learning rate steps....
        scheduler.step()

        # averaging epoch loss...
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        # Validation step...
        model.eval()   
        for inputs, labels in tqdm(valid_dataloader, position = 0, leave = True):
            # moving tensors to gpu....
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]


        # saving the best validation model....

    
# Performing Training steps....
model_ft = train_model(model = base_model, 
                       criterion  = criterion, 
                       optimizer  = optimizer_ft,
                       scheduler  = exp_lr_scheduler,
                       num_epochs = train_epochs)             
                    
                    
                    

                        
                
               
