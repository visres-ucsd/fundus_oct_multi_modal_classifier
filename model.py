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
import constants

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

print("PyTorch version : {} | Device type : {}".format(torch.__version, device))


# Using only pre-trained models for our experiments...
# Add the model as the experiments progress....
def build_model():

    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=True, progress=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    

    return model_ft

    

