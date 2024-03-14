# Data loader centric imports....
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from constants import *
import PIL
import numpy as np

# data loader helper class
class GenerateDataset(Dataset):
    def __init__(self,
                 image_files,
                 labels_dict,
                 img_res = 224,
                 augment = False,
                 shuffle = True,
                 transform = None,
                 split_flag = "training"):


        # loading image paths & labels....
        self.image_list = image_files
        # one hot encoded labels......
        self.labels = [label_mapping[labels_dict[i.split("/")[-1]]] for i in self.image_list]
        
        self.labels = np.array(self.labels)
        self.image_list = np.array(self.image_list)

        print("Number of images loaded for {} split : {} images".format(split_flag, len(self.image_list)))
        print("Number of labels loaded for {} split : {} labels".format(split_flag, len(self.labels)))

        # model specific pre-processing function.....
        self.transform = transform

        # other training constants.....
        self.img_res = (img_res, img_res)
        self.augment = augment
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_size = len(self.image_list)

        # inverse of class proportion serves as the class weights 
        # more frequent the class is, less is the associated class weight....
        self.class_weights = {i : (1/len(self.labels[self.labels[:,i] == 1])) * (self.total_size / num_classes)
                              for i in range(len(label_mapping))}

        print("Class weights are : ")
        for ct,i in enumerate(label_mapping):
            freq = len(self.labels[self.labels[:,ct] == 1])
            print("Class Name : {} | Frequency : {} | Weight : {}".format(i, freq, self.class_weights[ct]))


    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):

        # fetching image & label....
        img_inp     = PIL.Image.open(self.image_list[idx]).resize(self.img_res)
        class_label = self.labels[idx]

        # point to experiment further........
        aug_transforms = None
        if self.augment:
            aug_transforms = transforms.Compose([transforms.RandomResizedCrop(size = 224,
                                                                              scale = (0.93,1.0)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomAffine(degrees = (-5,5),
                                                                        translate = (0.05,0.05),
                                                                        interpolation=transforms.InterpolationMode.BILINEAR)])
        
        proc_img = self.transform(img_inp)
        if aug_transforms is not None:
            proc_img = aug_transforms(proc_img)
        
        # one hot encoded labels...
        return proc_img, class_label, self.image_list[idx]

