import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import  transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from Dataset_rnn_test import get_loader
from Model import get_model, FocalLoss,LSTM
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='flyexpress', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

onlystage={2:[4],3:[2,3],4:[3],5:[3],6:[4],7:[4],8:[0],9:[4],10:[4],
            12:[4],13:[3],14:[4],15:[4],16:[3],17:[3],18:[2],19:[2],
            20:[2],21:[4],22:[1],23:[2,3],24:[4],25:[1],26:[1],27:[2,3,4],
            28:[4],29:[2]}

def viz(module, input):
    x = input[0][0]
    min_num = np.minimum(4, x.size()[0])
    for i in range(min_num):
        plt.subplot(1, 4, i+1)
        plt.imshow(x[i].cpu())
    plt.show()


def main():
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    model=LSTM(368,1024,0)
    utils.load(model,'../premodel/step3.pt')
    model=model.cuda()
    queue=get_loader(args.batch_size)
    infer(queue, model)

def infer(queue, model):
    model.eval()

    predicts=[]
    stages_all=[]
    for step, (input, stages,length) in enumerate(queue):
        with torch.no_grad():
            input = Variable(input).cuda()
            logits=model(input)
        for index,s in enumerate(stages):
            for key in onlystage:
                if not int(s[-1])-2 in onlystage[key]:
                    logits[index][key]=0
            predicts.append(logits[index].detach().cpu().numpy())
            stages_all.append(s)

    predicts_=np.array(predicts)
    thres=[0.5300000000000002, 0.7400000000000004, 0.9000000000000006, 0.26000000000000006, 0.6200000000000003, 0.19000000000000003, 0.6400000000000003, 0.9500000000000006, 0.8900000000000006, 0.9100000000000006, 0.7400000000000004, 0.6100000000000003, 0.6600000000000004, 0.9700000000000006, 0.8200000000000005, 0.9300000000000006, 0.10999999999999999, 0.21000000000000005, 0.8700000000000006, 0.6600000000000004, 0.8400000000000005, 0.7500000000000004, 0.7900000000000005, 0.8800000000000006, 0.5700000000000003, 0.35000000000000014, 0.9900000000000007, 0.9900000000000007, 0.9700000000000006, 0.9800000000000006]

    
    predicts=copy.deepcopy(predicts_)
    predicts=predicts.T
    for i in range(30):
      predicts[i]=np.where(predicts[i]>thres[i],1,0)
    predicts=predicts.T
    f=open('sample-submission-f1.csv','w')
    f.write('Id,labels\n')
    for key,value in zip(stages_all,predicts):
        f.write(key+',')
        s=''
        for index,v in enumerate(value):
            if v>0:
                s=s+str(index)+' '
        f.write(s[:-1])
        f.write('\n')
    f.close()

    f=open('sample-submission-auc.csv','w')
    f.write('Id,')
    for i in range(29):
        f.write('label'+str(i)+',')
    f.write('label29\n')
    for key,value in zip(stages_all,predicts):
        f.write(key+',')
        s=''
        for index,v in enumerate(value):
            s=s+str(v)+','
        f.write(s[:-1])
        f.write('\n')
    f.close()

main()
