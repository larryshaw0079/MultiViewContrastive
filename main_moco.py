"""
@Time    : 2020/11/9 16:52
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : main_moco.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import pickle
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.std import tqdm

from mvc.data import LmdbDatasetWithEdges, transformation
from mvc.model import Moco, MocoClassifier
from mvc.utils import logits_accuracy, adjust_learning_rate, get_performance

AVAILABLE_1D_TRANSFORMATIONS = [
    'perturbation',
    'jittering',
    'flipping',
    'negating',
    'scaling',
    'mwarping',
    'twarping',
    'cshuffling',
    'cropping'
]

AVAILABLE_2D_TRANSFORMATIONS = [
    'jittering2d',
    'flipping2d',
    'negating2d',
    'scaling2d',
    'mwarping2d',
    'cshuffling2d',
    'cropping2d'
]


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
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--data-name', type=str, default='sleepedf', choices=['sleepedf', 'isruc'])
    parser.add_argument('--save-path', type=str, default='cache/')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--meta-file', type=str, required=True)
    parser.add_argument('--channels', type=int, default=2)
    parser.add_argument('--time-len', type=int, default=3000)
    parser.add_argument('--freq-len', type=int, default=None)
    parser.add_argument('--num-extend', type=int, default=500)
    parser.add_argument('--classes', type=int, default=5)

    # Model
    parser.add_argument('--network', type=str, default='r1d', choices=['r1d', 'r2d'])
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--aug', dest='augmentation', type=str, nargs='+', default=None)

    # Training
    parser.add_argument('--devices', type=int, nargs='+', default=None)
    parser.add_argument('--fold', type=int, default=20)
    parser.add_argument('--pretrain-epochs', type=int, default=200)
    parser.add_argument('--finetune-epochs', type=int, default=10)
    parser.add_argument('--finetune-ratio', type=float, default=0.1)
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--lr-schedule', type=int, nargs='*', default=[120, 160])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)

    # Optimization
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9, help='Only valid for SGD optimizer')

    # MOCO specific configs:
    parser.add_argument('--moco-k', default=2048, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')

    # Misc
    parser.add_argument('--tensorboard', action='store_true')
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


def get_augmentations(augmentation_list, two_crop=False):
    augmentation = []

    for aug_param in augmentation_list:
        if aug_param == 'perturbation':
            augmentation.append(transformation.Perturbation(min_perturbation=10, max_perturbation=300))
        elif aug_param == 'jittering':
            augmentation.append(transformation.Jittering())
        elif aug_param == 'flipping':
            augmentation.append(transformation.Flipping(randomize=True))
        elif aug_param == 'negating':
            augmentation.append(transformation.Negating(randomize=True))
        elif aug_param == 'scaling':
            augmentation.append(transformation.Scaling(randomize=True))
        elif aug_param == 'mwarping':
            augmentation.append(transformation.MagnitudeWarping())
        elif aug_param == 'twarping':
            augmentation.append(transformation.TimeWarping())
        elif aug_param == 'cshuffling':
            augmentation.append(transformation.ChannelShuffling())
        elif aug_param == 'cropping':
            augmentation.append(transformation.RandomCropping(size=2000))
        elif aug_param == 'jittering2d':
            augmentation.append(transformation.Jittering2d())
        elif aug_param == 'flipping2d':
            augmentation.append(transformation.Flipping2d(axis='both', randomize=True))
        elif aug_param == 'negating2d':
            augmentation.append(transformation.Negating2d(randomize=True))
        elif aug_param == 'scaling2d':
            augmentation.append(transformation.Scaling2d(randomize=True))
        elif aug_param == 'mwarping2d':
            augmentation.append(transformation.MagnitudeWarping2d())
        elif aug_param == 'cshuffling2d':
            augmentation.append(transformation.ChannelShuffling2d())
        elif aug_param == 'cropping2d':
            augmentation.append(transformation.RandomCropping2d(size=(80, 20)))
        else:
            raise ValueError(f'Invalid augmentation `{aug_param}`!')

    if two_crop:
        return transformation.TwoCropsTransform(transformation.Compose(augmentation))
    else:
        return transformation.Compose(augmentation)


def pretrain(model, dataset, device, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError('Invalid optimizer!')

    criterion = nn.CrossEntropyLoss().cuda(device)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, pin_memory=True, drop_last=True)

    model.train()
    for epoch in range(args.pretrain_epochs):
        losses = []
        accuracies = []
        adjust_learning_rate(optimizer, args.lr, epoch, args.pretrain_epochs, args)
        with tqdm(data_loader, desc=f'EPOCH [{epoch + 1}/{args.pretrain_epochs}]') as progress_bar:
            for x, _ in progress_bar:
                q, k = x[0], x[1]
                q, k = k.cuda(device, non_blocking=True), q.cuda(device, non_blocking=True)

                output, target = model(q, k)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                accuracies.append(logits_accuracy(output, target, topk=(1,))[0])

                progress_bar.set_postfix({'Loss': np.mean(losses), 'Acc': np.mean(accuracies)})

        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, f'model_pretrain_{epoch}.pth.tar'))


def finetune(classifier, dataset, device, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(classifier.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError('Invalid optimizer!')

    criterion = nn.CrossEntropyLoss().cuda(device)

    sampled_indices = np.arange(len(dataset))
    np.random.shuffle(sampled_indices)
    sampled_indices = sampled_indices[:int(len(sampled_indices) * args.finetune_ratio)]
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=False, pin_memory=True, drop_last=True,
                             sampler=SubsetRandomSampler(sampled_indices))

    classifier.train()
    for epoch in range(args.finetune_epochs):
        losses = []
        accuracies = []
        with tqdm(data_loader, desc=f'EPOCH [{epoch + 1}/{args.finetune_epochs}]') as progress_bar:
            for x, y in progress_bar:
                x, y = x.cuda(device, non_blocking=True), y.cuda(device, non_blocking=True)

                out = classifier(x)
                loss = criterion(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                accuracies.append(logits_accuracy(out, y, topk=(1,))[0])

                progress_bar.set_postfix({'Loss': np.mean(losses), 'Acc': np.mean(accuracies)})


def evaluate(classifier, dataset, device, args):
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, pin_memory=True, drop_last=True)

    targets = []
    scores = []

    classifier.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.cuda(device, non_blocking=True)

            out = classifier(x)
            scores.append(out.cpu().numpy())
            targets.append(y.numpy())

    scores = np.concatenate(scores, axis=0)
    targets = np.concatenate(targets, axis=0)

    return scores, targets


def main_worker(run_id, device, train_patients, test_patients, args):
    if args.tensorboard:
        writer = SummaryWriter(os.path.join(args.save_path, f'runs/run_{run_id}/'))

    # Pretraining
    model = Moco(network=args.network, device=device, in_channel=args.channels, mid_channel=16, dim=args.feature_dim,
                 K=args.moco_k, m=args.moco_m, T=args.moco_t)
    model.cuda(device)

    if args.tensorboard:
        writer.add_graph(model, [torch.randn(args.batch_size, args.channels, args.size).cuda(device),
                                 torch.randn(args.batch_size, args.channels, args.size).cuda(device)])

    # for name, param in model.named_parameters():
    #     print(name, param.shape)
    # model_summary(model, input_size=[(2, 100, 30), (2, 100, 30)], device=f'cuda:{device}')
    if args.augmentation is None:
        warnings.warn('Using all augmentations defaultly...')
        if args.network == 'r1d':
            train_augmentation = get_augmentations(AVAILABLE_1D_TRANSFORMATIONS, two_crop=True)
        else:
            train_augmentation = get_augmentations(AVAILABLE_2D_TRANSFORMATIONS, two_crop=True)
    else:
        train_augmentation = get_augmentations(args.augmentation, two_crop=True)
    train_dataset = LmdbDatasetWithEdges(lmdb_path=args.data_path, meta_file=args.meta_file, num_channel=args.channels,
                                         size=args.time_len if args.freq_len is None else (
                                             args.freq_len, args.time_len),
                                         num_extend=args.num_extend, patients=train_patients,
                                         transform=train_augmentation)
    print(train_dataset)

    pretrain(model, train_dataset, device, args)
    torch.save(model.state_dict(), os.path.join(args.save_path, 'model_pretrained.pth.tar'))

    # Finetuning
    classifier = MocoClassifier(network=args.network, device=device, in_channel=args.channels, mid_channel=16,
                                dim=args.feature_dim,
                                num_class=5,
                                dropout=0.5,
                                use_dropout=False,
                                use_l2_norm=True,
                                use_final_bn=True)
    classifier.cuda(device)

    state_dict = model.state_dict()
    new_dict = {}
    for k, v in state_dict.items():
        k = k.replace('encoder_q.0.', 'backbone.')
        new_dict[k] = v
    state_dict = new_dict

    classifier.load_state_dict(state_dict, strict=False)

    if args.network == 'r1d':
        finetune_augmentation = get_augmentations(['jittering'], two_crop=False)
    else:
        finetune_augmentation = get_augmentations(['jittering2d'], two_crop=False)
    finetune_dataset = LmdbDatasetWithEdges(lmdb_path=args.data_path, meta_file=args.meta_file,
                                            num_channel=args.channels,
                                            size=args.time_len if args.freq_len is None else (
                                                args.freq_len, args.time_len), num_extend=args.num_extend,
                                            patients=train_patients, transform=finetune_augmentation)
    finetune(classifier, finetune_dataset, device, args)
    torch.save(classifier.state_dict(), os.path.join(args.save_path, 'classifier_finetuned.pth.tar'))

    # Evaluation
    if args.network == 'r1d':
        test_augmentation = get_augmentations(['jittering'], two_crop=False)
    else:
        test_augmentation = get_augmentations(['jittering2d'], two_crop=False)
    test_dataset = LmdbDatasetWithEdges(lmdb_path=args.data_path, meta_file=args.meta_file,
                                        num_channel=args.channels,
                                        size=args.time_len if args.freq_len is None else (args.freq_len, args.time_len),
                                        num_extend=args.num_extend,
                                        patients=test_patients, transform=test_augmentation)

    scores, targets = evaluate(classifier, test_dataset, device, args)
    performance = get_performance(scores, targets)
    with open(os.path.join(args.save_path, f'performance_{run_id}.pkl'), 'wb') as f:
        pickle.dump(performance, f)
    print(performance)


if __name__ == '__main__':
    args = parse_args()

    if args.seed is not None:
        setup_seed(args.seed)

    devices = args.devices
    if devices is None:
        devices = list(range(torch.cuda.device_count()))

    if not os.path.exists(args.save_path):
        warnings.warn(f'The path {args.save_path} dost not existed, created...')
        os.makedirs(args.save_path)

    print(f'[INFO] Using devices {devices}...')

    with open(args.meta_file, 'rb') as f:
        meta_info = pickle.load(f)
        patients = np.unique(meta_info['patient'])

    assert args.fold <= len(patients)
    kf = KFold(n_splits=args.fold)
    for i, (train_index, test_index) in enumerate(kf.split(patients)):
        print(f'[INFO] Running cross validation for {i + 1}/{args.fold} fold...')
        train_patients, test_patients = patients[train_index], patients[test_index]
        main_worker(i, devices[0], train_patients.tolist(), test_patients.tolist(), args)

        # TODO only run 1 fold now
        break
