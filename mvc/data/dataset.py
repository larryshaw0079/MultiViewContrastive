"""
@Time    : 2020/9/17 19:09
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import pickle
from typing import List, Union, Tuple

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from .transformation import Transformation


def get_training_dataset(lmdb_path, meta_file, num_channel, length, num_extend, transform):
    pass


def get_evaluation_dataset():
    pass


class LmdbDataset(Dataset):
    def __init__(self, lmdb_path, meta_file, num_channel, length):
        self.lmdb_path = lmdb_path
        with open(meta_file, 'rb') as f:
            self.meta_info = pickle.load(f)
        self.num_channel = num_channel
        self.length = length

        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.keys = self.meta_info['path']
        self.labels = torch.from_numpy(np.asarray(self.meta_info['class'], dtype=np.long))
        self.size = len(self.keys)

    def __getitem__(self, item):
        with self.env.begin(write=False) as txn:
            buffer = txn.get(self.keys[item].encode('ascii'))
        data = np.frombuffer(buffer, dtype=np.float32).copy().reshape(self.num_channel, self.length)
        data = torch.from_numpy(data)
        label = self.labels[item]

        return data, label

    def __len__(self):
        return self.size


class LmdbDatasetWithEdges(Dataset):
    def __init__(self, lmdb_path, meta_file, num_channel, size: Union[int, Tuple[int]], num_extend,
                 patients: List = None,
                 transform: Transformation = None):
        self.lmdb_path = lmdb_path
        with open(meta_file, 'rb') as f:
            self.meta_info = pickle.load(f)
        self.num_channel = num_channel
        if isinstance(size, int):
            size = (size,)
            self.full_shape = (num_channel, size)
        elif isinstance(size, tuple):
            assert len(size) == 2
            self.full_shape = (num_channel, *size)
        else:
            raise ValueError('Invalid` length`!')
        self.size = size
        self.num_extend = num_extend
        self.transform = transform
        self.patients = patients

        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        if self.patients is None:
            self.keys = self.meta_info['path']
        else:
            self.keys = []
            for i, p in enumerate(self.meta_info['path']):
                if self.meta_info['patient'][i] in self.patients:
                    self.keys.append(p)

        self.labels = torch.from_numpy(np.asarray(self.meta_info['class'], dtype=np.long))
        self.len = len(self.keys)

    def __getitem__(self, item):
        with self.env.begin(write=False) as txn:
            buffer = txn.get(self.keys[item].encode('ascii'))
        data = np.frombuffer(buffer, dtype=np.float32).copy().reshape(*self.full_shape[:-1],
                                                                      self.full_shape[-1] + 2 * self.num_extend)
        data = {'head': data[..., :self.num_extend],
                'mid': data[..., self.num_extend:-self.num_extend],
                'tail': data[..., -self.num_extend:]}

        if self.transform is not None:
            data = self.transform(data)

        if isinstance(data, list):
            data = [torch.from_numpy(data[0]['mid'].astype(np.float32)),
                    torch.from_numpy(data[1]['mid'].astype(np.float32))]
        else:
            data = torch.from_numpy(data['mid'].astype(np.float32))

        label = self.labels[item]

        return data, label

    def __len__(self):
        return self.len

    def __repr__(self):
        return f"""
            ***********************************
            Dataset Summary:
            # Instance: {self.len}
            Shape: {self.full_shape}
            ***********************************
        """
