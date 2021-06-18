from torch.utils.data.dataset import Dataset
from torchvision import  transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch

import pandas as pd
import numpy as np
import os
import copy
import math
'''
prelearned info:
   MEAN, STD
   distribution
'''
def calculate_meanstd(loader):
    ct=0
    print(len(loader))
    for i,data in enumerate(loader):
        img,_=data
        m=np.sum(img.numpy(),axis=(0,2,3))
        s=np.sum(img.numpy()**2,axis=(0,2,3))
        if ct>0:
            mean+=m
            std+=s
        else:
            mean=m
            std=s
        ct+=img.shape[0]
        if i%30==0:
            print(ct)
    mean=mean/ct/224/224
    std=std/ct/(224**2)-mean**2
    print(mean,std)
        

def transform():
    MEAN =[0.6097585467]#[0.5893248,0.59196347,0.64798737]
    STD = [0.11923956]#[0.12179819,0.12193635,0.11398414]

    train_transform = transforms.Compose([
        transforms.Resize([116,340]),
        transforms.RandomCrop([112,336], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #AddGaussianNoise(0,1,1),
        transforms.RandomRotation(15),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
      ])

    valid_transform = transforms.Compose(
        [transforms.Resize([116,340]),
         transforms.CenterCrop([112,336]),
         transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD),
        ])
    return train_transform, valid_transform

class flyDataset(Dataset):
    def __init__(self,feed,path='../data/test/test/',is_train=True):
        super().__init__()
        self.path=path
        #not considering bags first, regard each pic individually
        self.ds=[]
        stages=np.array(feed[0])
        c_imgs=np.array(feed[1])
        #dump into ds
        for no in range(stages.shape[0]):
            pics=c_imgs[no].split(',')
            pics[0]=pics[0][1:]
            pics[-1]=pics[-1][:-1]
            for pic in pics:
                self.ds.append(pic)
        #data augmentation
        train_transform,valid_transform=transform()
        self.transform=valid_transform
                
    def __len__(self):
        return len(self.ds)

    def __getitem__(self,index):
        img=self.ds[index]
        img0=img
        img=Image.open(self.path+img)
        img=self.transform(img)
        return img,img0

def get_loader(bs):
    file='../data/test/test_without_label.csv'
    train=pd.read_csv(file)
    stages=train['gene_stage']
    c_imgs=train['imgs']#corresponding imgs to the gene stage
    feed=[stages,c_imgs]
    ds=flyDataset(feed=feed,is_train=True)
    dataloader = DataLoader(ds, batch_size=bs,shuffle=False, num_workers=0)
    return dataloader

