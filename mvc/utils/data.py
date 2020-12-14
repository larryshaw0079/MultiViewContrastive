"""
@Time    : 2020/12/7 16:36
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
from tqdm.std import tqdm


def folder_to_lmdb(data_path, dest_file, commit_interval):
    assert dest_file.endswith('.lmdb')

    print('Start...')
    meta_df = pd.read_csv(os.path.join(data_path, 'meta.csv'))
    files = [os.path.join(data_path, p) for p in meta_df['path'].values.tolist()]
    file_size = np.load(files[0])['data_q'].nbytes + np.load(files[0])['data_k'].nbytes
    dataset_size = file_size * len(files)
    print(f'Estimated dataset size: {dataset_size} bytes')

    env = lmdb.open(dest_file, map_size=dataset_size * 10)
    txn = env.begin(write=True)

    for idx, file in tqdm(enumerate(files), total=len(files), desc='Writing LMDB'):
        data = np.load(file)
        data = np.concatenate([np.expand_dims(data['data_q'], axis=0), data['data_k']], axis=0)
        key = file.encode('ascii')
        txn.put(key, data)

        if (idx + 1) % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
