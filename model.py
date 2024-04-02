# Basic Imports
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
import numpy as np
from constants import *

# PyTorch related imports....
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import time
from tempfile import TemporaryDirectory
from transformers import Dinov2ForImageClassification, ViTForImageClassification


# Impoving code reproducability...
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
cudnn.benchmark = True

# setting up gpu details
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

print("PyTorch version : {} | Device type : {}".format(torch.__version__, device))

class classificationModel(nn.Module):
    def __init__(self,
                 base_model,
                 hf_flag = False):
        
        super().__init__()

        self.encoder = base_model
        self.hf_flag = hf_flag

        if self.hf_flag == "resnet":
            base_ft_cnt  = self.encoder.fc.in_features
            self.encoder.fc  = nn.Identity(base_ft_cnt) 

            self.layer_1 = nn.Linear(base_ft_cnt, dense_1)
            #self.layer_2 = nn.Linear(dense_1, dense_2)
            #self.layer_3 = nn.Linear(dense_2, dense_3)
            self.final_layer = nn.Linear(dense_1, 1)
            self.activation_func = nn.ReLU()
            #self.final_activation = nn.Softmax(dim=1)
            self.final_activation = nn.Sigmoid()
            self.drop_out_layer = nn.Dropout(p=dropout)

        elif (self.hf_flag == "dino") or (self.hf_flag == "vit"):
            self.encoder.classifier = nn.Linear(768, 1)
            self.final_activation = nn.Sigmoid()
    
        elif self.hf_flag == "mobilenet":
            fc_size = self.encoder.classifier[3].in_features
            self.layer_1 = nn.Linear(in_features=fc_size, out_features=1)
            self.encoder.classifier[3] = self.layer_1
            self.final_activation = nn.Sigmoid()


    def forward(self, x):
        if self.hf_flag == "resnet":
            x = self.encoder(x)
            x = self.drop_out_layer(x)

            # first block.....
            x = self.layer_1(x)
            x = self.activation_func(x)
            x = self.drop_out_layer(x)
            
            """
            # second block
            x = self.layer_2(x)
            x = self.activation_func(x)
            x = self.drop_out_layer(x)
        
            # third block
            x = self.layer_3(x)
            x = self.activation_func(x)
            x = self.drop_out_layer(x)
            """

            # final prediction 
            x = self.final_layer(x)
            x = self.final_activation(x)
        elif (self.hf_flag =="dino") or (self.hf_flag == "vit"):
            x = self.encoder(x).logits
            x = self.final_activation(x)
        
        elif self.hf_flag == "mobilenet":
            x = self.encoder(x)
            x = self.final_activation(x)

        return x


# Using only pre-trained models for our experiments...
# Add the model as the experiments progress....
# Using heavy models for best classification accuracy....
def build_model():
    
    hf_flag = "resnet"
    if model_name == "resnet":
        model_ft = models.resnet50()
    elif model_name == "dinov2":
        model_ft = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-small-imagenet1k-1-layer")
        hf_flag = "dino"
    elif model_name == "mobilenet":
        model_ft = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.IMAGENET1K_V1')
        hf_flag = "mobilenet"
    elif model_name =="vit":
        model_ft = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', 
                                                              num_labels=1, 
                                                              ignore_mismatched_sizes=True, 
                                                              id2label={0 : "glaucoma"}, 
                                                              label2id={"glaucoma" : 0})
        hf_flag = "vit"

    final_model = classificationModel(model_ft, hf_flag)
    

    return final_model

    

