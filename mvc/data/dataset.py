"""
@Time    : 2020/9/17 19:09
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import os

import lmdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# SLEEPEDF_SUBJECTS = ['SC4042E0.npz',
#                      'SC4061E0.npz',
#                      'SC4051E0.npz',
#                      'SC4062E0.npz',
#                      'SC4022E0.npz',
#                      'SC4072E0.npz',
#                      'SC4041E0.npz',
#                      'SC4052E0.npz',
#                      'SC4011E0.npz',
#                      'SC4012E0.npz',
#                      'SC4002E0.npz',
#                      'SC4032E0.npz',
#                      'SC4021E0.npz',
#                      'SC4001E0.npz',
#                      'SC4091E0.npz',
#                      'SC4031E0.npz',
#                      'SC4082E0.npz',
#                      'SC4081E0.npz',
#                      'SC4071E0.npz',
#                      'SC4092E0.npz']
#
# ISRUC_SUBJECTS = [f'subject{i}.npz' for i in range(1, 11)]


def get_sleepedf_instance():
    pass


def get_sleepedf_sequence():
    pass


def get_isruc_instance():
    pass


def get_isruc_sequence():
    pass


def get_training_dataset(path, data_category, backend):
    assert os.path.exists(path)


def get_evaluation_dataset(path, data_category, backend):
    assert os.path.exists(path)


# def get_training_dataset(path, seq_len, data_category, patients: List = None):
#     assert os.path.exists(path)
#     assert data_category in ['sleepedf', 'isruc']
#     file_names = patients
#
#     data_list = []
#     target_list = []
#
#     if isinstance(patients, list):
#         candidate_files = list(map(lambda p: os.path.join(path, p), patients))
#     else:
#         raise ValueError('Invalid patients param!')
#
#     for filename in candidate_files:
#         tmp = np.load(filename)
#
#         if data_category == 'sleepedf':
#             current_data = np.concatenate(
#                 (tmp['eeg_fpz_cz'].reshape(-1, 1, tmp['eeg_fpz_cz'].shape[-1]),
#                  tmp['eeg_pz_oz'].reshape(-1, 1, tmp['eeg_pz_oz'].shape[-1])),
#                 axis=1)
#             current_target = tmp['annotation']
#         else:
#             current_data = []
#             for channel in ['F3_A2', 'C3_A2', 'F4_A1', 'C4_A1', 'O1_A2', 'O2_A1']:
#                 current_data.append(np.expand_dims(tmp[channel], 1))
#             current_data = np.concatenate(current_data, axis=1)
#             current_target = tmp['label']
#
#         for i in range(0, len(current_data), seq_len):
#             if i + seq_len > len(current_data):
#                 break
#             data_list.append(np.expand_dims(current_data[i:i + seq_len], axis=0))
#             target_list.append(np.expand_dims(current_target[i:i + seq_len], axis=0))
#
#     data_list = np.concatenate(data_list)
#     target_list = np.concatenate(target_list)
#
#     return data_list, target_list
#
#
# def get_evaluation_dataset(path, seq_len, data_category, patients: List, sample_ratio=1.0):
#     assert os.path.exists(path)
#     assert data_category in ['sleepedf', 'isruc']
#     file_names = patients
#
#     data_list = []
#     target_list = []
#
#     if isinstance(patients, list):
#         candidate_files = list(map(lambda p: os.path.join(path, p), patients))
#     else:
#         raise ValueError('Invalid patients param!')
#
#     for filename in candidate_files:
#         tmp = np.load(filename)
#
#         if data_category == 'sleepedf':
#             current_data = np.concatenate(
#                 (tmp['eeg_fpz_cz'].reshape(-1, 1, tmp['eeg_fpz_cz'].shape[-1]),
#                  tmp['eeg_pz_oz'].reshape(-1, 1, tmp['eeg_pz_oz'].shape[-1])),
#                 axis=1)
#             current_target = tmp['annotation']
#         else:
#             current_data = []
#             for channel in ['F3_A2', 'C3_A2', 'F4_A1', 'C4_A1', 'O1_A2', 'O2_A1']:
#                 current_data.append(np.expand_dims(tmp[channel], 1))
#             current_data = np.concatenate(current_data, axis=1)
#             current_target = tmp['label']
#         if sample_ratio == 1.0:
#             for i in range(len(current_data)):
#                 if i + seq_len > len(current_data):
#                     break
#                 data_list.append(np.expand_dims(current_data[i:i + seq_len], axis=0))
#                 target_list.append(np.expand_dims(current_target[i:i + seq_len], axis=0))
#         else:
#             idx = np.arange(seq_len - 1, len(current_data))
#             selected_idx = np.random.choice(idx, size=int(len(idx) * sample_ratio), replace=False)
#             for i in selected_idx:
#                 data_list.append(np.expand_dims(current_data[i - seq_len + 1:i + 1], axis=0))
#                 target_list.append(np.expand_dims(current_target[i - seq_len + 1:i + 1], axis=0))
#
#     data_list = np.concatenate(data_list)
#     target_list = np.concatenate(target_list)
#
#     return data_list, target_list


class LmdbDataset(Dataset):
    def __init__(self, lmdb_path, meta_file):
        self.lmdb_path = lmdb_path
        self.meta_info = pd.read_csv(meta_file)

        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.keys = self.meta_info['path'].values.tolist()
        self.labels = torch.from_numpy(self.meta_info['class'].values.astype(np.long))
        self.length = len(self.keys)

    def __getitem__(self, item):
        with self.env.begin(write=False) as txn:
            buffer = txn.get(self.keys[item].encode('ascii'))
        data = np.frombuffer(buffer, dtype=np.float32)
        data = torch.from_numpy(data)
        label = self.labels[item]

        return data, label

    def __len__(self):
        return self.length


class SleepDataset(Dataset):
    def __init__(self, x, y, return_label=False):
        self.return_label = return_label

        self.data = x
        self.targets = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.return_label:
            return (
                torch.from_numpy(self.data[item].astype(np.float32)),
                torch.from_numpy(self.targets[item].astype(np.long))
            )
        else:
            return torch.from_numpy(self.data[item].astype(np.float32))

    def __repr__(self):
        return f"""
               ****************************************
               Model  : {self.__class__.__name__}
               Length : {len(self)}
               ****************************************
                """
