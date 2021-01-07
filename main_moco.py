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
import shutil
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.std import tqdm

from mvc.data import LmdbDatasetWithEdges, transformation
from mvc.model import Moco, MocoClassifier
from mvc.utils import (
    logits_accuracy,
    adjust_learning_rate,
    get_performance,
    representation_to_tsv
)

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
    parser.add_argument('--save-path', type=str, default='cache/tmp')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load-path', type=str, default=None)
    parser.add_argument('--meta-file', type=str, required=True)
    parser.add_argument('--channels', type=int, default=2)
    parser.add_argument('--time-len', type=int, default=3000)
    parser.add_argument('--freq-len', type=int, default=None)
    parser.add_argument('--num-extend', type=int, default=500)
    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--write-embedding', action='store_true')

    # Model
    parser.add_argument('--network', type=str, default='r1d', choices=['r1d', 'r2d'])
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--aug', dest='augmentation', type=str, nargs='+', default=None)

    # Training
    parser.add_argument('--only-pretrain', action='store_true')
    parser.add_argument('--devices', type=int, nargs='+', default=None)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--kfold', type=int, default=10)
    parser.add_argument('--pretrain-epochs', type=int, default=200)
    parser.add_argument('--finetune-epochs', type=int, default=10)
    parser.add_argument('--finetune-ratio', type=float, default=0.1)
    parser.add_argument('--finetune-mode', type=str, default='freeze', choices=['freeze', 'smaller', 'all'])
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--lr-schedule', type=int, nargs='*', default=[120, 160])
    parser.add_argument('--batch-size', type=int, default=128)
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
    parser.add_argument('--wandb', action='store_true')
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


def pretrain(model, dataset, device, run_id, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98), eps=1e-09,
                               amsgrad=True)
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
                q, k = q.cuda(device, non_blocking=True), k.cuda(device, non_blocking=True)

                output, target = model(q, k)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                acc = logits_accuracy(output, target, topk=(1,))[0]
                accuracies.append(acc)

                progress_bar.set_postfix({'Loss': np.mean(losses), 'Acc': np.mean(accuracies)})

        if args.wandb:
            wandb.log({
                "pretrain_loss": np.mean(losses),
                "pretrain_acc": np.mean(accuracies)
            })

        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, f'moco_run_{run_id}_pretrain_{epoch}.pth.tar'))


def finetune(classifier, dataset, device, args):
    params = []
    if args.finetune_mode == 'freeze':
        print('[INFO] Finetune classifier only for the last layer...')
        for name, param in classifier.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            else:
                params.append({'params': param})
    elif args.finetune_mode == 'smaller':
        print('[INFO] Finetune the whole classifier where the backbone have a smaller lr...')
        for name, param in classifier.named_parameters():
            if 'backbone' in name:
                params.append({'params': param, 'lr': args.lr / 10})
            else:
                params.append({'params': param})
    else:
        print('[INFO] Finetune the whole classifier...')
        for name, param in classifier.named_parameters():
            params.append({'params': param})

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98), eps=1e-09,
                               amsgrad=True)
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

        if args.wandb:
            wandb.log({
                "finetune_loss": np.mean(losses),
                "finetune_acc": np.mean(accuracies)
            })


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


def write_embedding(model, device, args):
    if args.network == 'r1d':
        emb_augmentation = get_augmentations(['jittering'], two_crop=False)
    else:
        emb_augmentation = get_augmentations(['jittering2d'], two_crop=False)
    emb_dataset = LmdbDatasetWithEdges(lmdb_path=args.data_path, meta_file=args.meta_file,
                                       num_channel=args.channels,
                                       size=args.time_len if args.freq_len is None else (
                                           args.freq_len, args.time_len), num_extend=args.num_extend,
                                       patients=train_patients, transform=emb_augmentation)
    data_loader = DataLoader(emb_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, pin_memory=True, drop_last=True)
    encoder = model.encoder_q
    embeddings = []
    labels = []
    encoder.eval()
    print('[INFO] Writing embeddings...')
    for x, y in tqdm(data_loader, desc='EMBEDDING'):
        x = x.cuda(device, non_blocking=True)
        with torch.no_grad():
            z = encoder(x)
            z = z.squeeze()
            embeddings.append(z.cpu().numpy())
            labels.append(y.numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    print(embeddings.shape)

    representation_to_tsv(embeddings, dest_path=args.save_path, labels=labels)


def main_worker(run_id, device, train_patients, test_patients, args):
    # Pretraining
    model = Moco(network=args.network, device=device, in_channel=args.channels, mid_channel=16, dim=args.feature_dim,
                 K=args.moco_k, m=args.moco_m, T=args.moco_t)
    model.cuda(device)

    if args.wandb:
        wandb.watch(model)

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

    if args.resume:
        assert args.load_path is not None
        print('[INFO] Loading from pretrained weights...')
        model.load_state_dict(torch.load(args.load_path))
    else:
        pretrain(model, train_dataset, device, run_id, args)
    torch.save(model.state_dict(), os.path.join(args.save_path, f'moco_run_{run_id}_pretrained.pth.tar'))

    if args.write_embedding:
        write_embedding(model, device, args)

    if args.only_pretrain:
        return

    # Finetuning
    if args.finetune_mode == 'freeze':
        use_dropout = False
        use_l2_norm = True
        use_final_bn = True
    else:
        use_dropout = True
        use_l2_norm = False
        use_final_bn = False

    classifier = MocoClassifier(network=args.network, device=device, in_channel=args.channels, mid_channel=16,
                                dim=args.feature_dim,
                                num_class=5,
                                dropout=0.5,
                                use_dropout=use_dropout,
                                use_l2_norm=use_l2_norm,
                                use_final_bn=use_final_bn)
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
    torch.save(classifier.state_dict(), os.path.join(args.save_path, f'moco_run_{run_id}_finetuned.pth.tar'))

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

    if args.wandb:
        wandb.log({
            'accuracy': performance['accuracy'],
            'f1_micro': performance['f1_micro'],
            'f1_macro': performance['f1_macro'],
            'accuracy_per_class': performance['accuracy_per_class']
        })

    with open(os.path.join(args.save_path, f'statistics_{run_id}.pkl'), 'wb') as f:
        pickle.dump({'performance': performance, 'args': vars(args), 'cmd': sys.argv}, f)
    print(performance)


if __name__ == '__main__':
    args = parse_args()

    if args.wandb:
        with open('./data/wandb.txt', 'r') as f:
            os.environ['WANDB_API_KEY'] = f.readlines()[0]
        wandb.init(project='MVC', group=f'MOCO_{args.network}', config=args)

    if args.seed is not None:
        setup_seed(args.seed)
    else:
        torch.backends.cudnn.deterministic = True  # makes conv1d faster

    devices = args.devices
    if devices is None:
        devices = list(range(torch.cuda.device_count()))

    if not os.path.exists(args.save_path):
        warnings.warn(f'The path {args.save_path} dost not existed, created...')
        os.makedirs(args.save_path)
    elif not args.resume:
        warnings.warn(f'The path {args.save_path} already exists, deleted...')
        shutil.rmtree(args.save_path)
        os.makedirs(args.save_path)

    print(f'[INFO] Using devices {devices}...')

    with open(args.meta_file, 'rb') as f:
        meta_info = pickle.load(f)
        patients = np.unique(meta_info['patient'])

    patients = sorted(patients)
    patients = np.array(patients)

    assert args.kfold <= len(patients)
    assert args.fold < args.kfold
    kf = KFold(n_splits=args.kfold)
    for i, (train_index, test_index) in enumerate(kf.split(patients)):
        if i == args.fold:
            print(f'[INFO] Running cross validation for {i + 1}/{args.kfold} fold...')
            train_patients, test_patients = patients[train_index], patients[test_index]
            main_worker(i, devices[0], train_patients.tolist(), test_patients.tolist(), args)
            break
