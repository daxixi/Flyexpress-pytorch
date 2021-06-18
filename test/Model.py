import torchvision.models
import torch.nn as nn
import torch
import torch.nn.functional as F
import pycls.models
import math
import copy
from torch.autograd import Variable

def cosdecay(min_,max_,cur,run):
    return min_+0.5*(max_-min_)*(1+math.cos(3.14159*cur/run))

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True,scalar=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.org_scalar =copy.deepcopy(scalar)
        self.scalar=copy.deepcopy(scalar)
        self.cur=0
        self.cur_gamma=0
        self.run=10

    def forward(self, inputs, targets):
        scalar=copy.deepcopy(self.scalar)-1
        weights=scalar.repeat(targets.shape[0],1).cuda()
        weights=torch.mul(weights,targets)+1

        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False,weight=weights)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False,weight=weights)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        F_loss = torch.mean(F_loss,axis=0)
        #print(F_loss)
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

    def step(self):
        self.cur+=1
        for i in range(30):
            self.scalar[i]=cosdecay(1,self.org_scalar[i],self.cur,self.run)

    def step_gamma(self):
        self.cur_gamma+=1
        self.gamma=1/cosdecay(0.5,1,self.cur_gamma,self.run)

class diymodel(nn.Module):
    def __init__(self):
        super(diymodel,self).__init__()
        self.model=pycls.models.regnety("200MF", pretrained=False,
                                 cfg_list=("MODEL.NUM_CLASSES", 30))
        self.upconv=nn.Conv2d(1,3,1)

    def forward(self,x):
        x=self.upconv(x)
        x,p=self.model(x)
        return x,p

def get_model():
    return diymodel()

class avgpool(nn.Module):
    def __init__(self):
        super(avgpool, self).__init__()

    def forward(self,x):
        x=F.avg_pool2d(x, (x.shape[1],1)).squeeze()
        return x

class LSTM(nn.Module):
    def __init__(self, emb_size, hid_size, dropout):
        super(LSTM,self).__init__()
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        self.LSTM=nn.LSTM(self.emb_size, self.hid_size, num_layers=2,
                          batch_first=True, bidirectional=True) #2层双向LSTM
        self.dp=nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.hid_size*2, self.hid_size)
        self.relu=nn.ReLU()
        self.avgpool=avgpool()
        self.fc2 = nn.Linear(self.hid_size, 30)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        x,_=self.LSTM(x)
        x=self.dp(x)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.avgpool(x)
        x=self.fc2(x)
        x=self.sigmoid(x)

        return x
