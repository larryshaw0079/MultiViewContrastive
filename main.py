"""
@Time    : 2020/11/9 16:52
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : main.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mvc.data import get_training_dataset, get_evaluation_dataset
from mvc.model import Moco, DCC, DPC, CoCLR


def setup_seed(seed):
    warnings.warn(f'You have chosen to seed ({seed}) training. This will turn on the CUDNN deterministic setting, '
                  f'which can slow down your training considerably! You may see unexpected behavior when restarting '
                  f'from checkpoints.')

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    # Dataset & saving & loading
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--data-name', type=str, default='sleepedf', choices=['sleepedf', 'isruc'])
    parser.add_argument('--channels', type=int, default=2)
    parser.add_argument('--classes', type=int, default=5)

    # Model
    parser.add_argument('--backend', type=str, default='dpc', choices=['moco', 'dcc', 'dpc', 'coclr'])
    parser.add_argument('--feature-dim', type=int, default=128)

    # Training
    parser.add_argument('--devices', type=int, nargs='+', default=None)
    parser.add_argument('--pretrain-epochs', type=int, default=200)
    parser.add_argument('--finetune-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)

    # Optimization
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-3)
    parser.add_argument('--wd', dest='weight_decay', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9, help='Only valid for SGD optimizer')

    # Misc
    parser.add_argument('--disp-interval', type=int, default=20)
    parser.add_argument('--seed', type=int, default=None)

    args_parsed = parser.parse_args()

    if verbose:
        message = ''
        message += '-------------------------------- Args ------------------------------\n'
        for k, v in sorted(vars(args_parsed).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-------------------------------- End ----------------------------------'
        print(message)

    return args_parsed


def pretrain(model, dataset, device, args):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError('Invalid optimizer!')

    criterion = nn.CrossEntropyLoss().cuda(device)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, pin_memory=True, drop_last=True)

    model.train()
    for epoch in range(args.pretrain_epochs):
        for x in data_loader:
            x = x.cuda(device, non_blocking=True)


def finetune(model, dataset, device, args):
    classifier = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Linear(args.feature_dim, args.classes)
    )

    classifier.cuda(device)

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    classifier.train()


def evaluate(model, dataset, device, args):
    pass


def main_worker(device, args):
    if args.backend == 'moco':
        model = Moco()
    elif args.backend == 'dcc':
        model = DCC()
    elif args.backend == 'dpc':
        model = DPC()
    elif args.backend == 'coclr':
        model = CoCLR()
    else:
        raise ValueError('Invalid backend!')

    model.cuda(device)

    train_dataset = get_training_dataset()

    pretrain(model, train_dataset, device, args)

    finetune(model, train_dataset, device, args)

    test_dataset = get_evaluation_dataset()

    evaluate(model, test_dataset, device, args)


if __name__ == '__main__':
    args = parse_args()

    if args.seed is not None:
        setup_seed(args.seed)

    devices = args.devices
    if devices is None:
        devices = list(range(torch.cuda.device_count()))

    print(f'[INFO] Using devices {devices}...')

    main_worker(devices[0], args)
