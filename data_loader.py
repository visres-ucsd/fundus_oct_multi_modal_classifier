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
                 apply_random_prob = 0.2,
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
        self.apply_random_prob = apply_random_prob
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_size = len(self.image_list)

        # inverse of class proportion serves as the class weights 
        # more frequent the class is, less is the associated class weight....
        self.class_weights = {0 : (1/len(self.labels[self.labels[:] == 0]))*(self.total_size/ num_classes),
                              1 : (1/len(self.labels[self.labels[:] == 1]))*(self.total_size/ num_classes)}

        print("Class weights are : ")
        for ct,i in enumerate(label_mapping):
            freq = len(self.labels[self.labels[:] == ct])
            print("Class Name : {} | Frequency : {} | Weight : {}".format(i, freq, self.class_weights[ct]))


    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):

        # fetching image & label....
        img_inp     = PIL.Image.open(self.image_list[idx]).convert('RGB').resize(self.img_res)
        class_label = self.labels[idx]

        # point to experiment further........
        aug_transforms = None
        if self.augment:
            # augmentation params computed for fundus images....
            aug_transforms = transforms.Compose([transforms.RandomResizedCrop(size = 224, scale = (0.7,1.0)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.Pad((50,50,50,50),padding_mode = "reflect"),
                                                 transforms.RandomAffine(degrees = (-10,10), translate = (0.1,0.1), interpolation=transforms.InterpolationMode.BILINEAR),
                                                 transforms.CenterCrop(size = self.img_res[0]),])
                                                 #transforms.ColorJitter(brightness = (1.0,1.2), contrast = (1.0,1.2), saturation = (1.0,1.2)),
                                                 #transforms.GaussianBlur(5, (1.0,3.0))])
                                                                        
        proc_img = img_inp
        if aug_transforms is not None:
            toss = np.random.choice([0,1],p=[1-self.apply_random_prob, self.apply_random_prob])
            if toss == 1:
                proc_img = aug_transforms(img_inp)
        
        # one hot encoded labels... 
        # self transform is model specific compulsory pre-processing
        return self.transform(proc_img), class_label, self.image_list[idx]

