# Basic Imports
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import random
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import pickle
import warnings
import pandas as pd
warnings.filterwarnings('always')
import random

# pipeline specific imports..
from constants import *
from model import *
from data_loader import *

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

# tensorboard related imports...
from torch.utils.tensorboard import SummaryWriter

# Impoving code reproducability...
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# setting up gpu details
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype  = torch.float32


# loading image paths & labels....
label_dict = {}
with open(label_path, 'rb') as handle:
    label_dict = pickle.load(handle)

tmp_dict = {}
for i in label_dict.keys():
    name = i.split("_")[0]
    tmp_dict[name] = 0
patient_ids = np.array(list(tmp_dict.keys()))
print("Total number of input images       : ",len(label_dict))
print("Total number of unique patient ids : ",len(patient_ids))



# loading dataframe 
csv_path = "/tscc/nfs/home/vejoshi/oct_fundus_project/oct_fundus_dataset/all_labelled_fundus-rename-table-withgon-20210520.xlsx - Sheet1_Clinical_Demographic_Data_2_14_2024.csv"
df_csv = pd.read_csv(csv_path)

print("#"*30)
print("Splitting the dataset ....")
train_index, val_index, _, _ = train_test_split(list(range(len(patient_ids))),
                                                [1]*len(patient_ids),
                                                test_size=0.1,
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

print("#"*30)
print("Label distribution :")
print("Training split   : Healthy : {} | Suspects : {} | Glaucoma : {}".format(train_cnt["healthy"] / len(train_files),
                                                                               train_cnt["suspects"] / len(train_files),
                                                                               train_cnt["glaucoma"] / len(train_files)))

print("Validation split : Healthy : {} | Suspects : {} | Glaucoma : {}".format(val_cnt["healthy"] / len(valid_files),
                                                                             val_cnt["suspects"] / len(valid_files),
                                                                             val_cnt["glaucoma"] / len(valid_files)))


print("#"*30)
# building base model
pre_proc_func = None
if model_name in ['resnet','mobilenet','densenet']:
    # imagenet pre-processing pipeline...
    pre_proc_func =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
                                         #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

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

# pred order healthy, suspect, glaucoma.....
unique_og_labels = {"Healthy":[0,0,0], "OHT" : [0,0,0], "GON" : [0,0,0], "GON (suspicious)" : [0,0,0], "GVFD":[0,0,0], "GVFD & GON" : [0,0,0]}

# All model constants are specified in the constants file
# Make experimental changes in it....
model = build_model()
model.load_state_dict(torch.load("/tscc/nfs/home/vejoshi/oct_fundus_project/fundus_oct_multi_modal_classifier/expeiments/fundus/resnet/best_f1_score_model.pt"))
model.to(device)
model.eval()

label_dict_cnt = {}
for inputs, labels,file_paths in tqdm(valid_dataloader, position = 0, leave = True):
    # moving tensors to gpu....
    inputs = inputs.to(device)
    labels = labels.to(device)
    og_labels = []
    for f in file_paths:
        patiend_id_val = f.split("/")[-1].split("_")[0]
        eye_type = f.split("/")[-1].split("_")[1]
        diff_val = int(f.split("/")[-1].split("_")[2][1:])
        sub_df_label = df_csv[((df_csv["ran_id"] == patiend_id_val) & (df_csv["eye"] == eye_type) & (df_csv["diff"] == diff_val))]["EyeDX_at_ExamDate"].iloc[0]
        og_labels.append(sub_df_label)
        label_dict_cnt[sub_df_label] = label_dict_cnt.get(sub_df_label,0) + 1

                


    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        _, n_labels = torch.max(labels,1)
        
        for c,p in enumerate(preds.detach().cpu().numpy()):

            unique_og_labels[og_labels[c]][p]+=1

print("Actual label cnt :")
print(label_dict_cnt)

print("Rows are actual labels & column names are predicitons :")
for i in unique_og_labels:
    print("{} : {}".format(i, unique_og_labels[i]))


