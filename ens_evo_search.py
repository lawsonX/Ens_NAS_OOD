"""
0313 lirxiao:
This version uses 
    Validation accuracy as the estimator for subnet performance;
    Params as the efficiency constrain;
    (go evolution_finder.py to see implementation detail)
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets
import numpy as np
import time
import random
import math
import copy
import torch.optim as optim
from matplotlib import pyplot as plt
from utils import Bar, Logger, AverageMeter, accuracy

from ofa.model_zoo import ofa_net
from ofa.utils import download_url

from ofa.tutorial import EvolutionFinder, ArchManager
from ofa.tutorial import evaluate_ofa_subnet, evaluate_ofa_specialized
from ofa.imagenet_classification.elastic_nn.networks import OFAResNets, OFAResNets18

# from train_mlp import AccuracyPredictor

parser = argparse.ArgumentParser(description="PyTorch CIFAR10/100 Training")
parser.add_argument("--ens", type=int, default=2, help="number of experts model")
parser.add_argument(
    "--expand_list",
    type=str,
    default="0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 2.0",
)
parser.add_argument(
    "--pretrained",
    type=str,
    default="exp/0206/ID30_lr0075_ResV2_Ens10_cecb_e[0.25~2.0]_w[2]_d[0]/checkpoint/model_best.pth.tar",
)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0
args.ks_list = "3"  # not yet support
args.width_mult_list = "2.0"
args.depth_list = "0,1"

# set random seed
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print(
    "Successfully imported all packages and configured random seed to %d!" % random_seed
)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)
    print("Using GPU.")
else:
    print("Using CPU.")


def save(dict, path):
    if isinstance(dict, str):
        dict = eval(dict)
    with open(path, "a") as f:
        f.write(str(dict))  # dict to str


def load():
    with open("exp/searched_results/searched_configs.txt", "r", encoding="utf-8") as f:
        dict = eval(f.read())  # eval
        print(dict)
    return dict


best_acc = 0  # best test accuracy


def main():
    global best_acc
    criterion = nn.CrossEntropyLoss()
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    # Model
    print("==> creating OFA-Resnet ensemble model ")
    args.width_mult_list = [
        float(width_mult) for width_mult in args.width_mult_list.split(",")
    ]  # 类似slimmable network
    args.ks_list = [int(ks) for ks in args.ks_list.split(",")]
    args.expand_list = [float(e) for e in args.expand_list.split(",")]
    args.depth_list = [int(d) for d in args.depth_list.split(",")]

    args.width_mult_list = (
        args.width_mult_list[0]
        if len(args.width_mult_list) == 1
        else args.width_mult_list
    )

    if args.ens == 1:
        classes = 10
    elif args.ens == 2:
        classes = 6
    elif args.ens == 10:
        classes = 2
    ofa_network = OFAResNets18(
        n_classes=classes,  # run_config.data_provider.n_classes
        bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout,
        depth_list=args.depth_list,
        expand_ratio_list=args.expand_list,
        width_mult_list=args.width_mult_list,
        outputs=args.ens,
    )
    # print(ofa_network)
    ckpt = torch.load(args.pretrained)["state_dict"]
    ofa_network.load_state_dict(ckpt)
    ofa_network = ofa_network.cuda()
    print("Pretrained OFA-Resnet on cifar10 is loaded")

    # ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True).cuda()
    # print('The OFA Network is ready.')

    # if cuda_available:
    #     # path to the dataset
    #     print("Please input the path to the ImageNet dataset.\n")
    #     imagenet_data_path = "/home/datasets/imagenet"
    #     print('The ImageNet dataset files are ready.')
    # else:
    #     print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')

    # if cuda_available:
    #     # The following function build the data transforms for test
    #     def build_val_transform(size):
    #         return transforms.Compose([
    #             transforms.Resize(int(math.ceil(size / 0.875))),
    #             transforms.CenterCrop(size),
    #             transforms.ToTensor(),
    #             transforms.Normalize(
    #                 mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225]
    #             ),
    #         ])

    #     data_loader = torch.utils.data.DataLoader(
    #         datasets.ImageFolder(
    #             root=os.path.join(imagenet_data_path, 'val'),
    #             transform=build_val_transform(224)
    #         ),
    #         batch_size=250,  # test batch size
    #         shuffle=True,
    #         num_workers=16,  # number of workers for the data loader
    #         pin_memory=True,
    #         drop_last=False,
    #     )
    #     print('The ImageNet dataloader is ready.')
    # else:
    #     data_loader = None
    #     print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')

    # accuracy predictor
    # accuracy_predictor = AccuracyPredictor(pretrained=True)
    # print("The accuracy predictor is ready!")
    # print(accuracy_predictor.model)

    P = 100  # The size of population in each generation
    N = 100  # How many generations of population to be searched
    r = 0.25  # The ratio of networks that are used as parents for next generation
    params = {
        "constraint_type": "param",  # Let's do FLOPs-constrained search
        "efficiency_constraint": 7,  # Params constraint (M), suggested range [150, 600]
        "mutate_prob": 0.1,  # The probability of mutation in evolutionary search
        "mutation_ratio": 0.5,  # The ratio of networks that are generated through mutation in generation n >= 2.
        # 'efficiency_predictor': params_counter, # To use a predefined efficiency predictor.
        # "accuracy_predictor": accuracy_predictor,  # To use a predefined accuracy_predictor predictor.
        "population_size": P,
        "max_time_budget": N,
        "parent_ratio": r,
        "ofa_network": ofa_network,
        "data_loader": testloader,
    }

    # build the evolution finder
    finder = EvolutionFinder(**params) # This version using validator as the estimator for subnes' accuracy 
    manager = ArchManager()

    # start searching
    print("start searching....")
    # finder.set_efficiency_constraint(11)
    best_valids, best_info = finder.run_evolution_search(ens=args.ens)
    print(best_info)
    arch_list = [a[1] for a in best_info]
    save(arch_list, path="exp/searched_results/searched_configs3.txt")

    ## get ensemble model
    arch_list = load()
    _, test_acc = test(
        testloader, ofa_network, criterion, 1, True
    )  # use loss.backward() to get grad for importance calculation

    ensemble_model = ofa_network.get_active_ensembles(arch_list, preserve_weight=True)
    ensemble_model = ensemble_model.cuda()
    _, test_acc = test(testloader, ensemble_model, criterion, 1, True)


def test(testloader, model, criterion, epoch, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar("Processing", max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx == 0:
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(
                inputs, volatile=True
            ), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs, _ = outputs
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

            print(
                "top1: {top1: .4f} | top5: {top5: .4f}".format(
                    top1=top1.avg,
                    top5=top5.avg,
                )
            )
            # plot progress
            # bar.suffix  = '({batch}/{size}) Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            #             batch=batch_idx + 1,
            #             size=len(testloader),
            #             total=bar.elapsed_td,
            #             eta=bar.eta_td,
            #             loss=losses.avg,
            #             top1=top1.avg,
            #             top5=top5.avg,
            #             )
            # bar.next()
        # bar.finish()
    return (losses.avg, top1.avg)


if __name__ == "__main__":
    main()
