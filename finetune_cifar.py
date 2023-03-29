'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import models.cifar as models
# from models.cifar.resnet import ResNet18,ResNet20, ResNetEns
from ofa.model_zoo import ofa_net
from ofa.imagenet_classification.elastic_nn.networks import OFAResNets,OFAResNets18,MaskEnsembleResNets18
from ofa.utils.layers import MultiHeadLinearLayer
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.class_balanced_loss import CB_loss

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 80],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint/test', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--beta', type=float, default=0.9999)
parser.add_argument('--gama', type=float, default=0.1)
parser.add_argument('--mode', type=str, default='progressive_shrinkage', choices=['progressive_shrinkage','random','config'])
parser.add_argument('--pretrained', type=str, default='checkpoint/ps_train_[F010203]_lr1_e800/model_best.pth.tar')

# Architecture
parser.add_argument('--branches', type=int, default=2, help='number of experts model')
parser.add_argument('--pruning_ratio_list', type=str, default='0.1,0.2')
parser.add_argument('--ratio', type=float, default=0)

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0
args.config = [[0.1, [0.2, 0.1, 0.0], [0.1, 0.1, 0.0], [0.0, 0.2, 0.0], [0.2, 0.2, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.1, 0.2, 0.0], [0.2, 0.1, 0.0]], [0.2, [0.0, 0.1, 0.0], [0.2, 0.0, 0.0], [0.2, 0.2, 0.0], [0.1, 0.1, 0.0], [0.0, 0.1, 0.0], [0.2, 0.1, 0.0], [0.1, 0.2, 0.0], [0.1, 0.2, 0.0]]]


# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

def load():
    with open('exp/searched_results/searched_configs3.txt', 'r', encoding='utf-8') as f:
        dict = eval(f.read())  # eval
        print(dict)
    return dict

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    criterion = nn.CrossEntropyLoss()

    # Model
    print("==> creating OFA-Resnet ensemble model ")
    args.pruning_ratio_list = sorted(set([float(e) for e in args.pruning_ratio_list.split(',')]), reverse=False)

    if args.branches == 1:
        classes = 10
    elif args.branches == 2:
        classes = 6
    elif args.branches == 10:
        classes = 2
    ofa_network = MaskEnsembleResNets18(
        n_classes=classes, #run_config.data_provider.n_classes
        bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout,
        depth_list=[0],
        expand_ratio_list=[1.0],
        width_mult_list=[2.0],
        branches=args.branches
    ).cuda()
    print(ofa_network)

    # load supernet pretrained weight
    ckpt = torch.load(args.pretrained)['state_dict']
    model = torch.nn.DataParallel(ofa_network).cuda()
    model.load_state_dict(ckpt)
    print("Pretrained OFA-Resnet on cifar10 is loaded")
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(args.epochs), T_mult=1, eta_min=0, last_epoch=-1, verbose=False)

    # Resume
    title = 'cifar-10-resnet' #+ args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.mode == 'progressive_shrinkage':
        print('----- Now training the max network with global pruning rate=0 --------')
        train_epochs(args, logger, model, start_epoch, scheduler, optimizer, trainloader, testloader, criterion,stage='train_max')
        for r in args.pruning_ratio_list:
            print('----- Now training the next step with pruning rate={} --------'.format(r))
            model.module.compute_mask(pruning_rate=r)
            train_epochs(args, logger, model, start_epoch, scheduler, optimizer, trainloader, testloader, criterion,stage='train_ps{}'.format(r))
        print('----- progressive_shrinkage training is Finished ------')

    elif args.mode == 'random':
        print('-----Now training the max network with global pruning rate=0--------')
        train_epochs(args, logger, model, start_epoch, scheduler, optimizer, trainloader, testloader, criterion,stage='train_max')
        print('----- Now training the next step with random pruning rate for every each layer --------')
        train_epochs(args, logger, model, start_epoch, scheduler, optimizer, trainloader, testloader, criterion, random=True,stage='train_random')
    
    elif args.mode == "config":
        print('-----Now training the customized network with given config --------')
        test_loss, test_acc = test(testloader, model, criterion, use_cuda)
        model.module.set_network_from_config(config=args.config)
        train_epochs(args, logger, model, start_epoch, scheduler, optimizer, trainloader, testloader, criterion,stage='train_config')


def train_epochs(args, logger, model, start_epoch, scheduler, optimizer, trainloader, testloader, criterion, random=False, stage='train_max'):
    # Train and val
    best_acc = 0  # best test accuracy
    test_acc = 0
    for epoch in range(start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)
        if epoch != start_epoch:
            scheduler.step()

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, use_cuda)
        if random:
            model.module.set_network_from_config(config=None)

        # append logger file
        logger.append([lr, train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_name = os.path.join(args.checkpoint,stage)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=save_name)

    # logger.close()
    # logger.plot()
    # savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_sub = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss_sub = 0
        if isinstance(outputs, tuple) and args.branches != 1:
            # from class_balanced_loss import CB_loss
            outputs, output_list = outputs
            for i,out in enumerate(output_list):
                temp_target = targets.clone()
                if args.branches == 2:
                    for temp in range(len(temp_target)):
                        if temp_target[temp]>=i*5 and temp_target[temp]<=i*5+4:
                            temp_target[temp] = temp_target[temp]-i*5
                        else:
                            temp_target[temp] = 5
                    loss_sub += CB_loss(temp_target,out,[5000,5000,5000,5000,5000,250000],6,"focal",args.beta,args.gama)
                else:
                    for temp in range(len(temp_target)):
                        if temp_target[temp] == i:
                            temp_target[temp] = 0
                        else:
                            temp_target[temp] = 1 
                    loss_sub += CB_loss(temp_target,out,[5000,450000],2,"focal",args.beta,args.gama)

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        losses_sub.update(loss_sub, inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        loss = loss + loss_sub/args.branches
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} | ETA: {eta:} | Loss: {loss:.2f} {loss_sub:.2f} | top1: {top1: .2f} | top5: {top5: .2f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_sub = losses_sub.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test_sub(testloader, model, criterion, epoch, use_cuda):

    model.eval()
    for i,m in enumerate(model.branches):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        bar = Bar('Processing', max=len(testloader))
        correct = torch.zeros(2)
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            
            temp_target = targets.clone()
            for t in range(len(temp_target)):
                if temp_target[t]>=i*5 and temp_target[t]<=i*5+4:
                    temp_target[t] = temp_target[t]-i*5
                else:
                    temp_target[t] = 5

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, temp_target.data, topk=(1,))
            top1.update(prec1[0], inputs.size(0))
            
            new_output = torch.zeros(outputs.size(0),2,device=outputs.device)
            new_output[:,1] = outputs[:,5]
            new_output[:,0],_ = torch.max(outputs[:,:-1],1)
            temp_target = targets.clone()
            for t in range(len(temp_target)):
                if temp_target[t]>=i*5 and temp_target[t]<=i*5+4:
                    temp_target[t] = 0
                    if new_output[t,0]>new_output[t,1]:
                        correct[0] += 1
                else:
                    temp_target[t] = 1
                    if new_output[t,0]<new_output[t,1]:
                        correct[1] += 1

            # measure accuracy and record loss
            prec5 = accuracy(new_output.data, temp_target.data, topk=(1,))
            top5.update(prec5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Total: {total:} | ETA: {eta:} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
            bar.next()
        bar.finish()
        print(correct)
    return 

def test(testloader, model, criterion, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
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
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs,_ = outputs
        loss = criterion(outputs, targets)
        loss.backward()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    """
    beta 是一个可调节的参数，用于平衡正负样本之间的权重，通常情况下，
    我们可以根据数据集中正负样本的比例来设置 beta 的值。
    例如，如果数据集中负样本比例较高，
    我们可以将 beta 设置为一个较小的值，
    以增加对正样本的关注。

    gamma 是一个可调节的指数参数，
    用于调节易分类样本和难分类样本之间的权重差异。
    当 gamma 设为 0 时,Focal Loss 退化为标准的交叉熵损失。
    一般情况下，我们可以根据数据集中难易分类样本的分布情况来设置 gamma 的值。
    例如，如果数据集中存在一些非常难以分类的样本，
    我们可以将 gamma 设置为一个较大的值，以加强对这些样本的关注。
    """
    main()
