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

from Dataset import get_loader
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
    utils.load(model,'../../premodel/step2.pt')
    model=model.cuda()

    train_queue,valid_queue,scalar=get_loader(args.batch_size)

    criterion = FocalLoss(scalar=scalar)
    criterion = criterion.cuda()
    
    best_acc = 0.0
    auc,macro,sample = infer(train_queue, model, criterion)

    valid_acc=0.2*auc+0.4*macro+0.4*sample

    logging.info('valid score %f, auc: %f,marco: %f, sample: %f',valid_acc,auc,macro,sample)

def infer(valid_queue, model, criterion):
    meter = utils.Meter()
    model.eval()

    features={}
    for step, (input, target,names) in enumerate(valid_queue):
        with torch.no_grad():
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            logits,p=model(input)
            s=nn.Sigmoid()
            logits=s(logits)
        loss = criterion(logits, target)
        for name,feature in zip(names,p):
            features[name]=feature.detach().cpu().numpy().tolist()

        meter.update(target,logits)
        if step % args.report_freq == 0:
            auc,macro,sample=meter.get()
            logging.info('valid %03d:%f,%f,%f', step, auc,macro,sample)

    with open('feature.pickle','wb' )as f:
        pickle.dump(features,f)
    auc,macro,sample=meter.get()
    meter.get_fp()
    return auc,macro,sample

main()
