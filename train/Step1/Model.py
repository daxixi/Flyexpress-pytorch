import torchvision.models
import torch.nn as nn
import torch
import torch.nn.functional as F
import pycls.models
import math
import copy
from torch.autograd import Variable

class diymodel(nn.Module):
    def __init__(self):
        super(diymodel,self).__init__()
        self.model= pycls.models.regnety("200MF", pretrained=False,
                                 cfg_list=("MODEL.NUM_CLASSES", 30))
        self.upconv=nn.Conv2d(1,3,1)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x=self.upconv(x)
        x=self.model(x)
        x=self.sigmoid(x)
        return x

def get_model():
    return diymodel()
