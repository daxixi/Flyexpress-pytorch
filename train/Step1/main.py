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

from Dataset import get_loader
from Model import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='flyexpress', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=4.8, help='init learning rate')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=2, help='num of training epochs')
parser.add_argument('--save', type=str, default='step1', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument("--lr-step-size" , type=int, required=False, default=5, help="Number of epochs after which the learning rate will decay")
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay')
args = parser.parse_args()

args.save = 'Fly-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

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
    model=model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)

    queue=get_loader(args.batch_size)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100,eta_min=1e-7,last_epoch=-1)

    criterion = nn.MSELoss()
    criterion = criterion.cuda()

    best_acc = 0.0
    for epoch in range(args.epochs):
        logging.info('epoch%d',epoch)
        obj= train(queue, model, criterion, optimizer)
        scheduler.step()
        logging.info('train loss %f',obj)
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        
def train(train_queue, model, criterion, optimizer):
    objmeter=utils.AverageMeter()
    model.train()

    for step, (input1,input2,_) in enumerate(train_queue):
        input1 = Variable(input1).cuda()
        input2 = Variable(input2).cuda()
        optimizer.zero_grad()
        output1=model(input1)
        output2=model(input2)
        
        loss = criterion(output1,output2)
        loss.backward()
        optimizer.step()
        objmeter.update(loss.data,input1.size(0))

        if step % args.report_freq == 0:
            logging.info('train %03d %e', step,objmeter.avg)
            
    return objmeter.avg

main()
