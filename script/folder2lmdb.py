"""
@Time    : 2020/12/8 12:22
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : folder2lmdb.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.append('../')
from mvc.utils.data import folder_to_lmdb, LmdbDataset


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--meta-file', type=str, required=True)
    parser.add_argument('--dest-path', type=str, required=True)
    parser.add_argument('--commit-interval', type=int, default=100)
    parser.add_argument('--test', action='store_true')

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


if __name__ == '__main__':
    args = parse_args()

    folder_to_lmdb(args.meta_file, args.dest_path, args.commit_interval)

    if args.test:
        dataset = LmdbDataset(args.dest_path, args.meta_file)
        meta_df = pd.read_csv(args.meta_file)
        data = np.load(meta_df.loc[0, 'path'])
        print(np.concatenate([np.expand_dims(data['data_q'], axis=0), data['data_k']], axis=0))
        print(dataset[0][0].numpy())
