from torch.utils.data.dataset import Dataset
from torchvision import  transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch

import pandas as pd
import pickle
import numpy as np
import os
import copy
import math
'''
prelearned info:
   MEAN, STD
   distribution
'''
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
        AddGaussianNoise(0,1,1),
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
    def __init__(self,feed,path='../../data/train/train/',is_train=True):
        super().__init__()
        self.path=path
        with open('../../premodel/feature.pickle','rb') as f:
            self.features=pickle.load(f)
        #not considering bags first, regard each pic individually
        self.ds=[]
        stages=np.array(feed[0])
        labels=np.array(feed[1])
        c_imgs=np.array(feed[2])
        distribution=torch.zeros(30)
        #dump into ds
        for no in range(stages.shape[0]):
            lb=torch.zeros(30)
            lbs=labels[no].split(',')
            for l in lbs:
                lb[int(l)]=1
            pics=c_imgs[no].split(',')
            distribution+=lb
            pics[0]=pics[0][1:]
            pics[-1]=pics[-1][:-1]
            self.ds.append((pics,lb,int(stages[no][-1])-2))
        #data augmentation
        train_transform,valid_transform=transform()
        if is_train==True:
            self.transform=train_transform
        else:
            self.transform=valid_transform
        print(distribution)
        self.scalar=torch.zeros(30)
        for i in range(30):
            self.scalar[i]=(len(self.ds)-distribution[i])/distribution[i]                
    def __len__(self):
        return len(self.ds)

    def __getitem__(self,index):
        imgs=self.ds[index][0]
        labels=self.ds[index][1]
        returns=[self.features[i] for i in imgs]
        return returns,labels,self.ds[index][2]

def collate_fn(train_data):
    data_=[]
    labels=[]
    stages=[]
    for t in train_data:
        data_.append(torch.tensor(t[0]))
        labels.append(t[1].numpy().tolist())
        stages.append(t[2])
    data_length = [len(data) for data in data_]
    train_data_ = torch.nn.utils.rnn.pad_sequence(data_, batch_first=True, padding_value=0)
    return train_data_, torch.tensor(labels),stages,data_length

def get_loader(bs):
    file='../../data/train/train.csv'
    train=pd.read_csv(file)
    stages=train['gene_stage']
    labels=train['labels']
    c_imgs=train['imgs']#corresponding imgs to the gene stage
    num=stages.shape[0]
    train_num=int(num*0.95)
    train_feed=[stages[:train_num],labels[:train_num],c_imgs[:train_num]]
    valid_feed=[stages[train_num:],labels[train_num:],c_imgs[train_num:]]
    ds=flyDataset(feed=train_feed,is_train=True)
    valid=flyDataset(feed=valid_feed,is_train=False)
    dataloader = DataLoader(ds, batch_size=bs,shuffle=True, num_workers=0,collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid, batch_size=bs,shuffle=False, num_workers=0,collate_fn=collate_fn)
    return dataloader,valid_dataloader,ds.scalar
