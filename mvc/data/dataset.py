"""
@Time    : 2020/9/17 19:09
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import pickle
from typing import Tuple

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from .transformation import Transformation


class LmdbDataset(Dataset):
    def __init__(self, lmdb_path, meta_file, num_channel):
        self.lmdb_path = lmdb_path
        with open(meta_file, 'rb') as f:
            self.meta_info = pickle.load(f)
        self.num_channel = num_channel

        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.keys = self.meta_info['path']
        self.labels = torch.from_numpy(np.asarray(self.meta_info['class'], dtype=np.long))
        self.length = len(self.keys)

    def fetch_data(self, item) -> Tuple[np.ndarray, torch.Tensor]:
        with self.env.begin(write=False) as txn:
            buffer = txn.get(self.keys[item].encode('ascii'))
        data = np.frombuffer(buffer, dtype=np.float32).copy().reshape(self.num_channel, -1)
        label = self.labels[item]

        return data, label

    def __getitem__(self, item):
        data, label = self.fetch_data(item)
        data = torch.from_numpy(data)

        return data, label

    def __len__(self):
        return self.length


class SleepDataset(LmdbDataset):
    def __init__(self, lmdb_path, meta_file, num_channel, transform: Transformation):
        super(SleepDataset, self).__init__(lmdb_path, meta_file, num_channel)

        self.transform = transform

    def __getitem__(self, item):
        data, label = self.fetch_data(item)
        data = self.transform(data)
        data = torch.from_numpy(data)

        return data, label

# class SleepDataset(Dataset):
#     def __init__(self, x, y, return_label=False):
#         self.return_label = return_label
#
#         self.data = x
#         self.targets = y
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, item):
#         if self.return_label:
#             return (
#                 torch.from_numpy(self.data[item].astype(np.float32)),
#                 torch.from_numpy(self.targets[item].astype(np.long))
#             )
#         else:
#             return torch.from_numpy(self.data[item].astype(np.float32))
#
#     def __repr__(self):
#         return f"""
#                ****************************************
#                Model  : {self.__class__.__name__}
#                Length : {len(self)}
#                ****************************************
#                 """
