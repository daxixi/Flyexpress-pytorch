import torchvision.models
import torch.nn as nn
import torch
import torch.nn.functional as F
import pycls.models

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets, weights):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False,weight=weights)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False,weight=weights)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        

class identity(nn.Module):
    def __init__(self):
        super(identity,self).__init__()
        pass
    
    def forward(self,x):
        return x

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        backbone=torchvision.models.resnet50(pretrained=True)
        self.ident=identity()
        self.pre=nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=(2,1), padding=3,bias=False),
                               backbone.bn1,
                               backbone.relu,
                               nn.MaxPool2d(kernel_size=3, stride=(3,2), padding=1))
        self.layer1=backbone.layer1
        self.layer2=backbone.layer2
        self.layer3=backbone.layer3
        self.layer4=backbone.layer4
        #2048*7*7 --> 30*7*7-->avg-->30
        '''
        self.avg=backbone.avgpool
        #2048*1*1
        #直接分30类，过sigmoid
        self.fc=nn.Linear(backbone.fc.in_features,30)
        '''
        self.extraction=nn.Conv2d(2048, 30,1)
        self.avg=backbone.avgpool
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x=self.ident(x)
        x=self.pre(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.extraction(x)
        x=self.avg(x)
        x=torch.flatten(x,1)
        x=self.sigmoid(x)

        return x

class diymodel(nn.Module):
    def __init__(self):
        super(diymodel,self).__init__()
        self.model=model = pycls.models.regnety("200MF", pretrained=False,
                                 cfg_list=("MODEL.NUM_CLASSES", 30))
        self.upconv=nn.Conv2d(1,3,1)

    def forward(self,x):
        x=self.upconv(x)
        x=self.model(x)
        return x

def get_model():
    return diymodel()
