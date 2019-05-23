import os
import random
import numpy as np
import scipy.io as spio
from torch.utils.data import Dataset, DataLoader
import torch
from spdnet.utils import untangent_space, symmetric, tangent_space, is_pos_def
from tqdm import tqdm
from dataset import *
from pathlib import Path


def augment(cov, tangent):
    rnd = tangent + 0.2 * symmetric(torch.rand(cov.size(1), cov.size(1)))
    out = untangent_space(rnd, cov)
    return out

train_dataset = AfewDataset(train=True)
valid_dataset = AfewDataset(train=False)

train_dir = './data/augmented/train'
valid_dir = './data/augmented/valid'

Path(train_dir).mkdir(parents=True, exist_ok=True)
Path(valid_dir).mkdir(parents=True, exist_ok=True)

print('Generating train dataset ...')
for idx in tqdm(range(train_dataset.nSamples)):
    index = train_dataset.data_index[idx]
    data_path = os.path.join(train_dataset.base_path,'spdface_400_inter_histeq', train_dataset.spd_path[index])
    data = loadmat(data_path)
    data = torch.from_numpy(data['Y1'])
    label = np.asarray([train_dataset.labels[index] - 1]).astype(np.long)

    tangent = tangent_space(data, data)
    for i in range(5):
        sample = {'data': augment(data, tangent), 'label': torch.from_numpy(label)}
        path = os.path.join(train_dir, '%d_%d.pth' % (idx, i))
        torch.save(sample, path)

print('Saving valid dataset ...')
for idx in tqdm(range(valid_dataset.nSamples)):
    index = valid_dataset.data_index[idx]
    data_path = os.path.join(valid_dataset.base_path, valid_dataset.spd_path[index])
    data = loadmat(data_path)
    data = torch.from_numpy(data['Y1'])
    label = np.asarray([valid_dataset.labels[index] - 1]).astype(np.long)
    sample = {'data': data, 'label': torch.from_numpy(label)}
    path = os.path.join(valid_dir, '%d_0.pth' % (idx))
    torch.save(sample, path)

