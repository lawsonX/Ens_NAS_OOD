import time
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from utils.class_balanced_loss import CB_loss
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch
import torch.nn as nn

import torch
import torch.nn as nn
BR = 2

class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, branches=2):
        super(MaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        if branches > 1:
            self.mask = torch.ones([branches, out_channels, in_channels, kernel_size, kernel_size]).cuda()
            self.weight_ = nn.Parameter(torch.stack([self.weight for _ in range(branches)],0)).cuda()
            self.weight_.retain_grad()
        else:
            self.mask = torch.ones([out_channels, in_channels, kernel_size, kernel_size]).cuda()

    def get_grad(self,idx):
        self.grad = self.weight_.grad[idx]
        return self.grad
    
    def compute_mask(self, idx, pruning_rate=0.5):
        score = self.weight_[idx] * self.weight_.grad[idx]
        score  = torch.sum(score, dim=tuple(range(1, len(score.shape))))
        score, i = torch.sort(score)
        num_pruning = int(score.numel() * pruning_rate)
        self.mask[idx][i[:num_pruning]] = 0

    def forward(self, x, idx=None):
        if idx is not None:
            weight = self.weight_[idx] * self.mask[idx]
        else:
            weight = self.weight * self.mask
        x = nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

class Net(nn.Module):
    def __init__(self,br=BR):
        super(Net, self).__init__()
        self.conv1 = MaskConv2d(3, 32, 3, padding=1,branches=br)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = MaskConv2d(32, 64, 3, padding=1,branches=br)
        self.conv3 = MaskConv2d(64, 128, 3, padding=1,branches=br)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2_1 = nn.Linear(512, 6)
        self.fc2_2 = nn.Linear(512, 6)

    def compute_mask(self, idx_list=[0,1], pruning_rate=0.2): # TODO：pruning rate 以后改成ofa的channel分配机制
        for i in idx_list:
            # TODO: 之后根据OFA,对block_list进行遍历
            self.conv1.compute_mask(i) 
            self.conv2.compute_mask(i)
            self.conv3.compute_mask(i)
        
    def forward(self, x, idx_list=[0,1]):
        x1= self.conv1(x,idx_list[0])
        x1 = nn.functional.relu(x1)
        x1 = self.pool(x1)
        x1 = self.conv2(x1,idx_list[0])
        x1 = nn.functional.relu(x1)
        x1 = self.pool(x1)
        x1 = self.conv3(x1,idx_list[0])
        x1 = nn.functional.relu(x1)
        x1 = self.pool(x1)
        x1 = x1.view(-1, 128 * 4 * 4)
        x1 = self.fc1(x1)
        x1 = nn.functional.relu(x1)
        x1 = self.fc2_1(x1)

        # 创建一个新的分支
        x2 = self.conv1(x,idx_list[1])
        x2 = nn.functional.relu(x2)
        x2 = self.pool(x2)
        x2 = self.conv2(x2,idx_list[1])
        x2 = nn.functional.relu(x2)
        x2 = self.pool(x2)
        x2 = self.conv3(x2,idx_list[1])
        x2 = nn.functional.relu(x2)
        x2 = self.pool(x2)
        x2 = x2.view(-1, 128 * 4 * 4)
        x2 = self.fc1(x2)
        x2 = nn.functional.relu(x2)
        x2 = self.fc2_2(x2)

        output_list = [x1,x2]
        output = torch.cat(tuple(out[:,:-1] for out in output_list),1)

        return output, output_list

def main():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = datasets.CIFAR10(root='/home/xiaolirui/workspace/ofa-cifar-V3/data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    model = Net().cuda()

    # loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    losses , top1 = test(testloader,model,criterion,True)
    model.compute_mask()
    # for i in range(2):
    #     model.conv1.mask[i]
    #     model.conv2.mask[i]
    #     model.conv3.mask[i]
    losses , top1 = test(testloader,model,criterion,True)

    # torch.save(model.state_dict(), 'checkpoint/demo/model.pth.tar')

def test(testloader, model, criterion, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_sub = AverageMeter()
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
        temp_target = targets.clone()

        # compute output
        outputs = model(inputs)
        loss_ce = criterion(outputs[0], targets)
        loss_cb = 0
        if isinstance(outputs, tuple):
            outputs, output_list = outputs
            for i in range(len(output_list)):
                for temp in range(len(temp_target)):
                    if temp_target[temp]>=i*5 and temp_target[temp]<=i*5+4:
                        temp_target[temp] = temp_target[temp]-i*5
                    else:
                        temp_target[temp] = 5
                loss_cb += CB_loss(temp_target,output_list[i],[5000,5000,5000,5000,5000,250000],6,"focal",0.9999,1.0) 
        
        loss = loss_ce + loss_cb
        loss.backward()

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        loss_sub.update(loss_cb, inputs.size(0))
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
    return losses.avg, top1.avg

if __name__ == '__main__':
    main()
