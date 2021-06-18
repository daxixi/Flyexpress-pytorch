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
import pickle

from Datasettest import get_loader
from Model import get_model, FocalLoss

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

    model=get_model()
    utils.load(model,'../premodel/step2.pt')
    model=model.cuda()

    train_queue=get_loader(args.batch_size)
    infer(train_queue, model)
    
def infer(valid_queue, model):
    meter = utils.Meter()
    model.eval()

    features={}
    for step, (input, names) in enumerate(valid_queue):
        with torch.no_grad():
            input = Variable(input).cuda()
            logits,p=model(input)
            s=nn.Sigmoid()
            logits=s(logits)
        for name,feature in zip(names,p):
            features[name]=feature.detach().cpu().numpy().tolist()

    with open('feature_test.pickle','wb' )as f:
        pickle.dump(features,f)

main()
