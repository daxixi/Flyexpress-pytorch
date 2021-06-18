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
from Model import get_model, FocalLoss

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='flyexpress', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
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

#先忽略，到时候注释回来
import warnings
warnings.filterwarnings('ignore')

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
    model=model.cuda()

    criterion = FocalLoss()
    criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)

    train_queue,valid_queue=get_loader(args.batch_size)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,75,eta_min=1e-7,last_epoch=-1)
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        logging.info('epoch%d',epoch)
        obj,auc,macro,sample = train(train_queue, model, criterion, optimizer)
        scheduler.step()
        logging.info('train auc: %f,marco: %f, sample: %f',
                     auc,macro,sample)

        auc,macro,sample = infer(valid_queue, model, criterion)

        valid_acc=0.2*auc+0.4*macro+0.4*sample
        if valid_acc > best_acc:
            best_acc = valid_acc
            utils.save(model, os.path.join(args.save, 'weights.pt'))

        logging.info('valid score %f, auc: %f,marco: %f, sample: %f',
                     valid_acc,auc,macro,sample)
def train(train_queue, model, criterion, optimizer):
    meter = utils.Meter()
    objmeter=utils.AverageMeter()
    model.train()

    for step, (input, target, weights) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda()
        weights=weights.cuda()
        optimizer.zero_grad()
        logits = model(input)
        sigmoid=nn.Sigmoid()
        logits=sigmoid(logits)
        loss = criterion(logits, target,weights)
        meter.update(target,logits)
        loss.backward()
        optimizer.step()
        objmeter.update(loss.data,input.size(0))

        if step % args.report_freq == 0:
            auc,macro,sample=meter.get()
            meter.get_fp()
            logging.info('train %03d %e %f,%f,%f', step,objmeter.avg,auc,macro,sample)

    auc,macro,sample=meter.get()
    meter.get_fp()
    return objmeter.avg,auc,macro,sample

def infer(valid_queue, model, criterion):
        #for visiualize
    #for name,m in model.named_modules():
    #    if name in ['ident','layer1','layer2','layer3','layer4','pre','avg']:
    #        m.register_forward_pre_hook(viz)
    meter = utils.Meter()
    model.eval()


    for step, (input, target,_) in enumerate(valid_queue):
        with torch.no_grad():
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            logits=model(input)
            sigmoid=nn.Sigmoid()
            logits=sigmoid(logits)
        weight=torch.ones(target.shape).cuda()
        loss = criterion(logits, target,weight)

        meter.update(target,logits)
        if step % args.report_freq == 0:
            auc,macro,sample=meter.get()
            logging.info('valid %03d:%f,%f,%f', step, auc,macro,sample)

    auc,macro,sample=meter.get()
    return auc,macro,sample

main()
