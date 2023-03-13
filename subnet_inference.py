from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import models.cifar as models
# from models.cifar.resnet import ResNet18,ResNet20, ResNetEns
from ofa.model_zoo import ofa_net
from ofa.imagenet_classification.elastic_nn.networks import OFAResNets18
from utils import Bar, AverageMeter, accuracy
from utils.class_balanced_loss import CB_loss

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
parser.add_argument('--ens', type=int, default=2, help='number of experts model')
parser.add_argument('--pretrained', type=str, default='checkpoint/0224/ID54_Searched_subnet2_ft300_99991/model_best.pth.tar')

args = parser.parse_args()

def load():
    with open('checkpoint/0224/ID54_Searched_subnet2_ft300_99991/searched_configs2.txt', 'r', encoding='utf-8') as f:
        dict = eval(f.read())  # eval
        print(dict)
    return dict

use_cuda = torch.cuda.is_available()
torch.manual_seed(3407)
if use_cuda:
    torch.cuda.manual_seed_all(3407)

def main():
    
    # Data
    print('==> Preparing dataset')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader = datasets.CIFAR10
    testset = dataloader(root='./data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()

    if args.ens == 1:
        classes = 10
    elif args.ens == 2:
        classes = 6
    elif args.ens == 10:
        classes = 2
    ofa_network = OFAResNets18(
        n_classes=classes, #run_config.data_provider.n_classes
        bn_param=(0.1, 1e-5),
        dropout_rate=0,
        depth_list=[0,1],
        expand_ratio_list=[0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0,1.25,1.5,1.75,2.0],
        width_mult_list=[2.0],
        outputs=args.ens
    )

    # build sub-ensemble model by Configlist
    arch_list = load()
    ensemble_model = ofa_network.get_active_ensembles(arch_list,preserve_weight=False).cuda()
    
    ckpt = torch.load(args.pretrained)['state_dict']
    new_ckpt = {}
    for k, v in ckpt.items():
        new_k = k.replace('module.','') if 'module' in k else k
        new_ckpt[new_k] = v
    ensemble_model.load_state_dict(new_ckpt)
    loss ,losses, test_acc = test(testloader, ensemble_model, criterion, 1, True)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses2 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)

        loss_sub = 0
        if isinstance(outputs, tuple):
            outputs,output_list = outputs
            for i,out in enumerate(output_list):
                temp_target = targets.clone()
                if args.ens == 2:
                    for temp in range(len(temp_target)):
                        if temp_target[temp]>=i*5 and temp_target[temp]<=i*5+4:
                            temp_target[temp] = temp_target[temp]-i*5
                        else:
                            temp_target[temp] = 5
                    loss_sub += CB_loss(temp_target,out,[5000,5000,5000,5000,5000,250000],6,"focal",0.9999,1.0)
                else:
                    for temp in range(len(temp_target)):
                        if temp_target[temp] == i:
                            temp_target[temp] = 0
                        else:
                            temp_target[temp] = 1 
                    loss_sub += CB_loss(temp_target,out,[5000,450000],2,"focal",0.99,0.5)
        
                    
        loss = criterion(outputs, targets)
        losses_cecb = loss+loss_sub/args.ens

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        losses2.update(losses_cecb.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss(Ce+Cb): {losses:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    losses=losses2.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, losses2.avg, top1.avg)


if __name__ == '__main__':
    main()