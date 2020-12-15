"""
@Time    : 2020/12/7 16:36
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import os
import pickle

import lmdb
import numpy as np
from tqdm.std import tqdm


def folder_to_lmdb(data_path, dest_file, commit_interval):
    assert dest_file.endswith('.lmdb')

    print('[INFO] Start...')
    with open(os.path.join(data_path, 'meta.pkl'), 'rb') as f:
        meta_info = pickle.load(f)
    files = meta_info['path']

    file_size = np.load(os.path.join(data_path, files[0]))['data'].nbytes
    dataset_size = file_size * len(files)
    print(f'[INFO] Estimated dataset size: {dataset_size} bytes')

    env = lmdb.open(dest_file, map_size=dataset_size * 10)
    txn = env.begin(write=True)

    for idx, file in tqdm(enumerate(files), total=len(files), desc='Writing LMDB'):
        data = np.load(os.path.join(data_path, file))
        value = data['data'].astype(np.float32)
        value = value.tobytes()
        # value = pa.serialize(value).to_buffer()
        key = file.encode('ascii')
        txn.put(key, value)

        if (idx + 1) % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
    print('[INFO] Finished...')
