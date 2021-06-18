from torch.utils.data.dataset import Dataset
from torchvision import  transforms
from torch.utils.data import DataLoader
from PIL import Image,ImageFilter
import torch
import random
import pandas as pd
import numpy as np
import os
import copy
import math

class AddGaussianBlur(object):
    def __init__(self,radius=3):
        self.radius=radius
    def __call__(self,img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img
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
    mean=mean/ct/336/112
    std=std/ct/(336*112)-mean**2
    print(mean,std)
        

def transform():
    MEAN =[0.6097585467]
    STD = [0.11923956]
    #two different augmentation method
    train_transform = transforms.Compose([
        transforms.Resize([116,340]),
        transforms.RandomCrop([112,336], padding=4),
        transforms.RandomHorizontalFlip(),
        AddGaussianNoise(0,1,3),
        transforms.RandomRotation(15),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
      ])

    train_transform2 = transforms.Compose([
        transforms.Resize([120,360]),
        transforms.RandomCrop([112,336], padding=4),
        transforms.RandomVerticalFlip(),
        AddGaussianBlur(3),
        transforms.RandomRotation(10),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
      ])
    return train_transform, train_transform2

class flyDataset(Dataset):
    def __init__(self,feed,path='../../data/test/test/',is_train=True):
        super().__init__()
        self.path=path
        #not considering bags first, regard each pic individually
        self.ds=[]
        stages=np.array(feed[0]) #stage
        c_imgs=np.array(feed[1]) #attributes (classes)
        #dump into ds
        for no in range(stages.shape[0]):
            pics=c_imgs[no].split(',')
            pics[0]=pics[0][1:]
            pics[-1]=pics[-1][:-1]
            for pic in pics:
                self.ds.append((pic,stages[no]))
        #data augmentation
        train_transform,valid_transform=transform()
        self.transform=train_transform
        self.transform2=valid_transform
        #bag information
        self.bags={}
        for no,data in enumerate(self.ds):
            stage=data[1]
            if stage not in self.bags:
                self.bags[stage]=[no]
            else:
                self.bags[stage].append(no)
                
    def __len__(self):
        return len(self.ds)

    def __getitem__(self,index):
        img=self.ds[index][0]
        stage=self.ds[index][1]
        try:
            img=Image.open(self.path+img)
        except:
            img=Image.open('../../data/train/train/'+img)
        img1=self.transform(img)
        if len(self.bags[stage])>1:
            while True:
                idx=random.choice(self.bags[stage])
                if not idx==index:
                    break
            img2=self.ds[idx][0]
            try:
                img2=Image.open(self.path+img2)
            except:
                img2=Image.open('../../data/train/train/'+img2)
            assert self.ds[idx][1]==stage
            img2=self.transform2(img2)
        else:
            img2=self.transform2(img)
        return img1,img2,stage

def get_loader(bs):
    file='../../data/test/test_without_label.csv'
    train=pd.read_csv(file)
    stages1=train['gene_stage']
    c_imgs1=train['imgs']#corresponding imgs to the gene stage

    file='../../data/train/train.csv'
    train=pd.read_csv(file)
    stages2=train['gene_stage']
    c_imgs2=train['imgs']#corresponding imgs to the gene stage

    stages=pd.concat([stages1,stages2])
    c_imgs=pd.concat([c_imgs1,c_imgs2])
    feed=[stages,c_imgs]
    ds=flyDataset(feed=feed,is_train=True)
    dataloader = DataLoader(ds, batch_size=bs,shuffle=False, num_workers=0)
    return dataloader
