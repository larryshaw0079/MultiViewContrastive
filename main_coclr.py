"""
@Time    : 2020/12/15 22:56
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : main_coclr.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import copy
import os
import pickle
import random
import shutil
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.std import tqdm

from mvc.data import LmdbDatasetWithEdges, TwoDataset, transformation
from mvc.model import CoCLR, MocoClassifier
from mvc.utils import multi_nce_loss, logits_accuracy, get_performance

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
    parser.add_argument('--data-name', type=str, default='sleepedf', choices=['sleepedf', 'isruc'])
    parser.add_argument('--data-path-v1', type=str, required=True)
    parser.add_argument('--data-path-v2', type=str, required=True)
    parser.add_argument('--meta-file-v1', type=str, required=True)
    parser.add_argument('--meta-file-v2', type=str, required=True)
    parser.add_argument('--load-path-v1', type=str, required=True)
    parser.add_argument('--load-path-v2', type=str, required=True)
    parser.add_argument('--channels', type=int, default=2)
    parser.add_argument('--time-len-v1', type=int, default=3000)
    parser.add_argument('--time-len-v2', type=int, default=30)
    parser.add_argument('--freq-len-v1', type=int, default=None)
    parser.add_argument('--freq-len-v2', type=int, default=100)
    parser.add_argument('--num-extend-v1', type=int, default=500)
    parser.add_argument('--num-extend-v2', type=int, default=10)
    parser.add_argument('--save-path', type=str, default='cache/')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--classes', type=int, default=5)

    # Model
    parser.add_argument('--network', type=str, default='r1d', choices=['r1d', 'r2d'])
    parser.add_argument('--second-network', type=str, default='r2d', choices=['r1d', 'r2d'])
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--aug-v1', dest='augmentation_v1', type=str, nargs='+', default=None)
    parser.add_argument('--aug-v2', dest='augmentation_v2', type=str, nargs='+', default=None)

    # Training
    parser.add_argument('--devices', type=int, nargs='+', default=None)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--kfold', type=int, default=10)
    parser.add_argument('--iter', dest='iteration', type=int, default=5)
    parser.add_argument('--pretrain-epochs', type=int, default=10)
    parser.add_argument('--finetune-epochs', type=int, default=10)
    parser.add_argument('--finetune-ratio', type=float, default=0.1)
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--lr-schedule', type=int, nargs='*', default=[120, 160])
    parser.add_argument('--batch-size', type=int, default=256)
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


def pretrain(model, train_dataset_v1, train_dataset_v2, device, run_id, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError('Invalid optimizer!')

    assert len(train_dataset_v1) == len(train_dataset_v2)
    # sampler = ShuffleSampler(train_dataset_v1, len(train_dataset_v1))
    dataset = TwoDataset(train_dataset_v1, train_dataset_v2)
    # data_loader_v1 = DataLoader(train_dataset_v1, batch_size=args.batch_size, num_workers=args.num_workers,
    #                             shuffle=False, pin_memory=True, drop_last=True)
    # data_loader_v2 = DataLoader(train_dataset_v2, batch_size=args.batch_size, num_workers=args.num_workers,
    #                             shuffle=False, pin_memory=True, drop_last=True)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, pin_memory=True, drop_last=True)

    model.train()
    for epoch in range(args.pretrain_epochs):
        losses = []
        accuracies = []
        with tqdm(data_loader, desc=f'EPOCH [{epoch + 1}/{args.pretrain_epochs}]',
                  total=len(train_dataset_v1) // args.batch_size) as progress_bar:
            for x, _, idx1, z, __, idx2 in progress_bar:
                x0, x1 = x[0].cuda(device, non_blocking=True), x[1].cuda(device, non_blocking=True)
                z0, z1 = z[0].cuda(device, non_blocking=True), z[1].cuda(device, non_blocking=True)
                idx = idx1.cuda(device, non_blocking=True)

                assert (idx1 == idx2).all()

                logits, mask = model(x0, x1, z0, z1, idx)
                mask_sum = mask.sum(1)

                if random.random() < 0.9:
                    # because model has been pretrained with infoNCE,
                    # in this stage, self-similarity is already very high,
                    # randomly mask out the self-similarity for optimization efficiency,
                    mask_clone = mask.clone()
                    mask_clone[mask_sum != 1, 0] = 0  # mask out self-similarity
                    loss = multi_nce_loss(logits, mask_clone)
                else:
                    loss = multi_nce_loss(logits, mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                progress_bar.set_postfix({'Loss': np.mean(losses), 'Acc': np.mean(accuracies)})


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
    assert os.path.isfile(args.load_path_v1), f'Invalid file path {args.load_path_v1}!'
    assert os.path.isfile(args.load_path_v2), f'Invalid file path {args.load_path_v2}!'

    state_dict_v1 = torch.load(args.load_path_v1)
    state_dict_v2 = torch.load(args.load_path_v2)

    if args.augmentation_v1 is None:
        warnings.warn('Using all augmentations defaultly...')
        if args.network == 'r1d':
            train_augmentation_v1 = get_augmentations(AVAILABLE_1D_TRANSFORMATIONS, two_crop=True)
        else:
            train_augmentation_v1 = get_augmentations(AVAILABLE_2D_TRANSFORMATIONS, two_crop=True)
    else:
        train_augmentation_v1 = get_augmentations(args.augmentation_v1, two_crop=True)

    if args.augmentation_v2 is None:
        warnings.warn('Using all augmentations defaultly...')
        if args.second_network == 'r1d':
            train_augmentation_v2 = get_augmentations(AVAILABLE_1D_TRANSFORMATIONS, two_crop=True)
        else:
            train_augmentation_v2 = get_augmentations(AVAILABLE_2D_TRANSFORMATIONS, two_crop=True)
    else:
        train_augmentation_v2 = get_augmentations(args.augmentation_v2, two_crop=True)

    train_dataset_v1 = LmdbDatasetWithEdges(lmdb_path=args.data_path_v1, meta_file=args.meta_file_v1,
                                            num_channel=args.channels,
                                            size=args.time_len_v1 if args.freq_len_v1 is None else (
                                                args.freq_len_v1, args.time_len_v1),
                                            num_extend=args.num_extend_v1, patients=train_patients,
                                            transform=train_augmentation_v1, return_idx=True)
    print(train_dataset_v1)

    train_dataset_v2 = LmdbDatasetWithEdges(lmdb_path=args.data_path_v2, meta_file=args.meta_file_v2,
                                            num_channel=args.channels,
                                            size=args.time_len_v2 if args.freq_len_v2 is None else (
                                                args.freq_len_v2, args.time_len_v2),
                                            num_extend=args.num_extend_v2, patients=train_patients,
                                            transform=train_augmentation_v2, return_idx=True)
    print(train_dataset_v2)

    # The last iteration should train the first view
    assert args.iteration % 2 == 1

    # Refine pretraining
    for it in range(args.iteration):
        reverse = False
        if it % 2 == 1:
            reverse = True

        if reverse:
            print(f'[INFO] Iteration {it}, train the second view...')
        else:
            print(f'[INFO] Iteration {it}, train the first view...')

        if reverse:
            model = CoCLR(network=args.second_network, second_network=args.network, device=device,
                          in_channel=args.channels, mid_channel=16, dim=args.feature_dim, K=args.moco_k, m=args.moco_m,
                          T=args.moco_t)
        else:
            model = CoCLR(network=args.network, second_network=args.second_network, device=device,
                          in_channel=args.channels, mid_channel=16, dim=args.feature_dim, K=args.moco_k, m=args.moco_m,
                          T=args.moco_t)
        model.cuda(device)

        # Second view as sampler
        new_dict = {}
        new_state_dict_v2 = copy.deepcopy(state_dict_v2)
        for k, v in new_state_dict_v2.items():
            if 'encoder_q.' in k:
                k = k.replace('encoder_q.', 'sampler.')
                new_dict[k] = v
        new_state_dict_v2 = new_dict
        new_dict = {}
        # Remove queue
        for k, v in new_state_dict_v2.items():
            if 'queue' not in k:
                new_dict[k] = v
        new_state_dict_v2 = new_dict

        # First view as encoder k
        new_dict = {}  # remove queue, queue_ptr
        new_state_dict_v1 = copy.deepcopy(state_dict_v1)
        for k, v in new_state_dict_v1.items():
            if 'queue' not in k:
                new_dict[k] = v
        new_state_dict_v1 = new_dict
        new_dict = {}
        for k, v in new_state_dict_v1.items():
            if 'encoder_q.' in k:
                k = k.replace('encoder_q.', 'encoder_k.')
                new_dict[k] = v
        new_state_dict_v1 = new_dict

        state_dict = {**new_state_dict_v1, **new_state_dict_v2}
        model.load_state_dict(state_dict, strict=False)

        if reverse:
            pretrain(model, train_dataset_v2, train_dataset_v1, device, run_id, args)
        else:
            pretrain(model, train_dataset_v1, train_dataset_v2, device, run_id, args)

        # Update the state dict
        state_dict_v1 = model.state_dict()
        state_dict_v1, state_dict_v2 = state_dict_v2, state_dict_v1

        # Saving
        if reverse:
            torch.save(model.state_dict(), os.path.join(args.save_path, f'coclr_second_run_{run_id}_iter_{it}.pth.tar'))
        else:
            torch.save(model.state_dict(), os.path.join(args.save_path, f'coclr_first_run_{run_id}_iter_{it}.pth.tar'))

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
    finetune_dataset = LmdbDatasetWithEdges(lmdb_path=args.data_path_v1, meta_file=args.meta_file_v1,
                                            num_channel=args.channels,
                                            size=args.time_len_v1 if args.freq_len_v1 is None else (
                                                args.freq_len_v1, args.time_len_v1), num_extend=args.num_extend_v1,
                                            patients=train_patients, transform=finetune_augmentation)
    finetune(classifier, finetune_dataset, device, args)
    torch.save(classifier.state_dict(), os.path.join(args.save_path, f'coclr_run_{run_id}_finetuned.pth.tar'))

    # Evaluation
    if args.network == 'r1d':
        test_augmentation = get_augmentations(['jittering'], two_crop=False)
    else:
        test_augmentation = get_augmentations(['jittering2d'], two_crop=False)
    test_dataset = LmdbDatasetWithEdges(lmdb_path=args.data_path_v1, meta_file=args.meta_file_v1,
                                        num_channel=args.channels,
                                        size=args.time_len_v1 if args.freq_len_v1 is None else (
                                            args.freq_len_v1, args.time_len_v1),
                                        num_extend=args.num_extend_v1,
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
    else:
        warnings.warn(f'The path {args.save_path} already exists, deleted...')
        shutil.rmtree(args.save_path)
        os.makedirs(args.save_path)

    print(f'[INFO] Using devices {devices}...')

    with open(args.meta_file_v1, 'rb') as f:
        meta_info = pickle.load(f)
        patients = np.unique(meta_info['patient'])

    assert args.kfold <= len(patients)
    assert args.fold < args.kfold
    kf = KFold(n_splits=args.kfold)
    for i, (train_index, test_index) in enumerate(kf.split(patients)):
        if i == args.fold:
            print(f'[INFO] Running cross validation for {i + 1}/{args.kfold} fold...')
            train_patients, test_patients = patients[train_index], patients[test_index]
            main_worker(i, devices[0], train_patients.tolist(), test_patients.tolist(), args)
            break
