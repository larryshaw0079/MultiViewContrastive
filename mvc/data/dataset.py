"""
@Time    : 2020/9/17 19:09
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import os
import pickle
from typing import List, Union, Tuple

import lmdb
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import Dataset, Sampler

from .transformation import Transformation

EPS = 1e-8


def tackle_denominator(x: np.ndarray):
    x[x == 0.0] = EPS
    return x


def tensor_standardize(x: np.ndarray, dim=-1):
    x_mean = np.expand_dims(x.mean(axis=dim), axis=dim)
    x_std = np.expand_dims(x.std(axis=dim), axis=dim)
    return (x - x_mean) / tackle_denominator(x_std)


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


class SleepDataset(Dataset):
    def __init__(self, data_path, data_name, num_epoch, patients: List = None, preprocessing: str = 'none',
                 return_idx=False, verbose=True):
        assert isinstance(patients, list)

        self.data_path = data_path
        self.data_name = data_name
        self.patients = patients
        self.preprocessing = preprocessing
        self.return_idx = return_idx

        assert preprocessing in ['none', 'quantile', 'standard']

        self.data = []
        self.labels = []

        for i, patient in enumerate(patients):
            if verbose:
                print(f'[INFO] Processing the {i + 1}-th patient {patient}...')
            data = np.load(os.path.join(data_path, patient))
            if data_name == 'sleepedf':
                recordings = np.stack([data['eeg_fpz_cz'], data['eeg_pz_oz']], axis=1)
                annotations = data['annotation']
            elif data_name == 'isruc':
                recordings = np.stack([data['F3_A2'], data['C3_A2'], data['F4_A1'], data['C4_A1'],
                                       data['O1_A2'], data['O2_A1']], axis=1)
                annotations = data['label'].flatten()
            else:
                raise ValueError

            if preprocessing == 'standard':
                print(f'[INFO] Applying standard scaler...')
                # scaler = StandardScaler()
                # recordings_old = recordings
                # recordings = []
                # for j in range(recordings_old.shape[0]):
                #     recordings.append(scaler.fit_transform(recordings_old[j].transpose()).transpose())
                # recordings = np.stack(recordings, axis=0)

                recordings = tensor_standardize(recordings, dim=-1)
            elif preprocessing == 'quantile':
                print(f'[INFO] Applying quantile scaler...')
                scaler = QuantileTransformer(output_distribution='normal')
                recordings_old = recordings
                recordings = []
                for j in range(recordings_old.shape[0]):
                    recordings.append(scaler.fit_transform(recordings_old[j].transpose()).transpose())
                recordings = np.stack(recordings, axis=0)
            else:
                print(f'[INFO] Convert the unit from V to uV...')
                recordings *= 1e6

            if verbose:
                print(f'[INFO] The shape of the {i + 1}-th patient: {recordings.shape}...')
            recordings = recordings[:(recordings.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch,
                                                                                             *recordings.shape[1:])
            annotations = annotations[:(annotations.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch)

            assert recordings.shape[:2] == annotations.shape[:2]

            self.data.append(recordings)
            self.labels.append(annotations)

        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)
        self.idx = np.arange(self.data.shape[0] * self.data.shape[1]).reshape(-1, self.data.shape[1])
        self.full_shape = self.data[0].shape

    def __getitem__(self, item):
        x = self.data[item]
        y = self.labels[item]

        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.long))

        if self.return_idx:
            return x, y, torch.from_numpy(self.idx[item].astype(np.long))
        else:
            return x, y

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return """
**********************************************************************
Dataset Summary:
Preprocessing: {}
# Instance: {}
Shape of an Instance: {}
Selected patients: {}
**********************************************************************
            """.format(self.preprocessing, len(self.data), self.full_shape, self.patients)


class SleepDataset2d(Dataset):
    def __init__(self, data_path, data_name, num_epoch, patients: List = None, preprocessing: str = 'none',
                 verbose=True):
        assert isinstance(patients, list)

        self.data_path = data_path
        self.data_name = data_name
        self.patients = patients
        self.preprocessing = preprocessing

        assert preprocessing in ['none', 'quantile', 'standard']

        self.data = []
        self.labels = []

        for i, patient in enumerate(patients):
            if verbose:
                print(f'[INFO] Processing the {i + 1}-th patient {patient}...')
            data = np.load(os.path.join(data_path, patient))
            if data_name == 'sleepedf':
                recordings = data['data']
                annotations = data['annotation']
            elif data_name == 'isruc':
                recordings = np.stack([data['F3_A2'], data['C3_A2'], data['F4_A1'], data['C4_A1'],
                                       data['O1_A2'], data['O2_A1']], axis=1)
                annotations = data['label'].flatten()
            else:
                raise ValueError

            assert preprocessing == 'none'

            if verbose:
                print(f'[INFO] The shape of the {i + 1}-th patient: {recordings.shape}...')
            recordings = recordings[:(recordings.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch,
                                                                                             *recordings.shape[1:])
            annotations = annotations[:(annotations.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch)

            assert recordings.shape[:2] == annotations.shape[:2]

            self.data.append(recordings)
            self.labels.append(annotations)

        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)
        self.full_shape = self.data[0].shape

    def __getitem__(self, item):
        x = self.data[item]
        y = self.labels[item]

        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.long))

        return x, y

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return """
    **********************************************************************
    Dataset Summary:
    Preprocessing: {}
    # Instance: {}
    Shape of an Instance: {}
    Selected patients: {}
    **********************************************************************
                """.format(self.preprocessing, len(self.data), self.full_shape, self.patients)


class SleepDatasetImg(Dataset):
    def __init__(self, data_path, data_name, num_epoch, transform, patients: List = None, return_idx=False,
                 verbose=True):
        assert isinstance(patients, list)

        self.data_path = data_path
        self.data_name = data_name
        self.num_epoch = num_epoch
        self.transform = transform
        self.patients = patients
        self.return_idx = return_idx

        self.data = []
        self.labels = []

        for i, patient in enumerate(patients):
            if verbose:
                print(f'[INFO] Processing the {i + 1}-th patient {patient}...')
            data = np.load(os.path.join(data_path, patient))
            recordings = data['data']
            annotations = data['label']

            if verbose:
                print(f'[INFO] The shape of the {i + 1}-th patient: {recordings.shape}...')
            recordings = recordings[:(recordings.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch,
                                                                                             *recordings.shape[1:])
            annotations = annotations[:(annotations.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch)

            assert recordings.shape[:2] == annotations.shape[:2]

            self.data.append(recordings)
            self.labels.append(annotations)

        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)
        self.idx = np.arange(self.data.shape[0] * self.data.shape[1]).reshape(-1, self.data.shape[1])
        self.full_shape = self.data[0].shape

    def __getitem__(self, item):
        x = self.data[item]
        y = self.labels[item]

        # x = torch.stack([self.transform(x[0]), self.transform(x[1])], dim=0)
        # print(x.shape, x[:, 0].shape, '-------------')
        x1 = torch.stack([self.transform(Image.fromarray(x[i][0])) for i in range(x.shape[0])], dim=0)  # TODO for temp
        x2 = torch.stack([self.transform(Image.fromarray(x[i][1])) for i in range(x.shape[0])], dim=0)
        x = torch.cat([x1, x2], dim=1)
        y = torch.from_numpy(y.astype(np.long))

        if self.return_idx:
            return x, y, torch.from_numpy(self.idx[item].astype(np.long))
        else:
            return x, y

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return """
    **********************************************************************
    Dataset Summary:
    # Instance: {}
    Shape of an Instance: {}
    Selected patients: {}
    **********************************************************************
                """.format(len(self.data), self.full_shape, self.patients)


class LmdbDatasetWithEdges(Dataset):
    def __init__(self, lmdb_path, meta_file, num_channel, size: Union[int, Tuple[int]], num_extend,
                 patients: List = None,
                 transform: Transformation = None, return_idx: bool = False):
        self.lmdb_path = lmdb_path
        with open(meta_file, 'rb') as f:
            self.meta_info = pickle.load(f)
        self.num_channel = num_channel
        if isinstance(size, int):
            size = (size,)
            self.full_shape = (num_channel, *size)
        elif isinstance(size, tuple):
            assert len(size) == 2
            self.full_shape = (num_channel, *size)
        else:
            raise ValueError('Invalid` length`!')
        self.size = size
        self.num_extend = num_extend
        self.transform = transform
        self.patients = patients
        self.return_idx = return_idx

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

        if self.return_idx:
            return data, label, item
        else:
            return data, label

    def __len__(self):
        return self.len

    def __repr__(self):
        return """
**********************************************************************
Dataset Summary:
# Instance: {}
Shape of an Instance: {}
Selected patients: {}
**********************************************************************
        """.format(self.len, self.full_shape, self.patients)


class TwoDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2)

        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, item):
        return (*self.dataset1[item], *self.dataset2[item])

    def __len__(self):
        return len(self.dataset1)


class ShuffleSampler(Sampler):
    def __init__(self, data_source, total_len):
        super(ShuffleSampler, self).__init__(data_source)

        self.total_len = total_len

    def __iter__(self):
        yield from torch.randperm(self.total_len)

    def __len__(self):
        return self.total_len
