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
import tensorflow as tf

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
                 base_model):
        
        super().__init__()

        self.encoder = base_model
        base_ft_cnt  = self.encoder.fc.in_features
        self.encoder.fc  = nn.Identity(base_ft_cnt) 

        self.layer_1 = nn.Linear(base_ft_cnt, dense_1)
        self.layer_2 = nn.Linear(dense_1, dense_2)
        self.layer_3 = nn.Linear(dense_2, dense_3)
        self.final_layer = nn.Linear(dense_3, num_classes)
        self.activation_func = nn.ReLU()
        self.final_activation = nn.Softmax(dim=1)
        self.drop_out_layer = nn.Dropout(p=dropout)

    def forward(self, x):

        x = self.encoder(x)
        x = self.drop_out_layer(x)

        # first block.....
        x = self.layer_1(x)
        x = self.activation_func(x)
        x = self.drop_out_layer(x)
        
        # second block
        x = self.layer_2(x)
        x = self.activation_func(x)
        x = self.drop_out_layer(x)
        
        # third block
        x = self.layer_3(x)
        x = self.activation_func(x)
        x = self.drop_out_layer(x)

        # final prediction 
        x = self.final_layer(x)
        x = self.final_activation(x)

        return x


# Using only pre-trained models for our experiments...
# Add the model as the experiments progress....
# Using heavy models for best classification accuracy....
def build_model():

    if model_name == "resnet":
        model_ft = models.resnet152(weights='ResNet152_Weights.DEFAULT')
    

    final_model = classificationModel(model_ft)
    

    return final_model

    

