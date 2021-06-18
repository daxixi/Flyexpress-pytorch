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
        self.cur=5
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
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

    def step(self):
        self.cur+=1
        for i in range(30):
            self.scalar[i]=cosdecay(self.org_scalar[i]*0.8,self.org_scalar[i]*1.2,self.cur,self.run)

    def step_gamma(self):
        self.cur_gamma+=1
        self.gamma=1/cosdecay(0.5,1,self.cur_gamma,self.run)

class identity(nn.Module):
    def __init__(self):
        super(identity,self).__init__()
        pass
    
    def forward(self,x):
        return x

class diymodel(nn.Module):
    def __init__(self):
        super(diymodel,self).__init__()
        self.model= pycls.models.regnety("200MF", pretrained=False,
                                 cfg_list=("MODEL.NUM_CLASSES", 30))
        self.upconv=nn.Conv2d(1,3,1)

    def forward(self,x):
        x=self.upconv(x)
        x=self.model(x)
        return x

def get_model():
    return diymodel()
