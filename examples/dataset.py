from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import scipy.io as spio
from tqdm import tqdm
from glob import glob
import logging
import sys
from six.moves import urllib
import zipfile

import torch
from torch.utils.data import Dataset, DataLoader


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

downloads = [
    {'test': False, 'url': 'http://www.vision.ee.ethz.ch/~zzhiwu/codes/AFEW_SPD_data.zip',
        'dst': './data/AFEW', 'zip': True, 'filename': 'data.zip'},
    {'test': False, 'url': 'https://github.com/zzhiwu/SPDNet/raw/master/data/afew/spddb_afew_train_spd400_int_histeq.mat',
        'dst': './data/AFEW', 'zip': False, 'filename': 'spddb_afew_train_spd400_int_histeq.mat'}
]


def download(url, dest_directory, filename, is_zip):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    filepath = os.path.join(dest_directory, filename)

    logging.info("Download URL: {}".format(url))
    logging.info("Download DIR: {}".format(dest_directory))

    def _progress(count, block_size, total_size):
        prog = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write('\r>> Downloading %s %.1f%%' %
                         (filename, prog))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, filepath,
                                             reporthook=_progress)
    print()

    if is_zip:
        # Extract data
        logging.info("Extracting " + filename + " ...")
        zipfile.ZipFile(filepath, 'r').extractall(dest_directory)

    return filepath


def check_files(only_test):
    for data in downloads:
        if only_test and not data['test']:
            continue
        filepath = os.path.join(data['dst'], data['filename'])
        if not os.path.exists(filepath):
            return False
    return True


def download_dataset(only_test=False):

    if check_files(only_test):
        return

    logging.info("Downloading data ...")
    for data in downloads:
        if only_test and not data['test']:
            continue
        download(data['url'], data['dst'], data['filename'], data['zip'])
    logging.info("All data have been downloaded successfully.")


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


class AfewDataset(Dataset):
    def __init__(self, shuffle=True, train=False, augmentation=False):
        super(AfewDataset, self).__init__()

        download_dataset()

        self.train = train
        self.base_path = downloads[0]['dst']
        self.dataset_path = os.path.join(self.base_path, downloads[1]['filename'])

        dataset = loadmat(self.dataset_path)
        self.spd_path = [path.replace('\\', '/') for path in dataset['spd_train']['spd']['name']]
        self.labels = dataset['spd_train']['spd']['label']
        
        spd_set = dataset['spd_train']['spd']['set']
        if train:
            self.data_index = np.argwhere(spd_set == 1).squeeze()
        else:
            self.data_index = np.argwhere(spd_set == 2).squeeze()

        if shuffle:
            random.shuffle(self.data_index)

        self.nSamples = len(self.data_index) 
        print(self.nSamples)
        self.nClasses = 7
        self.augmentation = augmentation
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):
        index = self.data_index[idx]
        data_path = os.path.join(self.base_path,'spdface_400_inter_histeq', self.spd_path[index])
        data = loadmat(data_path)
        data = torch.from_numpy(data['Y1'])
        label = np.asarray([self.labels[index] - 1]).astype(np.long)

        sample = {'data': data, 'label': torch.from_numpy(label)}
        return sample


class AfewDatasetAugmented(Dataset):
    def __init__(self, train=False, augmentation=False):
        super(AfewDatasetAugmented, self).__init__()

        self.train = train
        if train:
            self.base_path = './data/augmented/train'
        else:
            self.base_path = './data/augmented/valid'

        self.paths = glob(os.path.join(self.base_path, '*.pth'))
        
        self.nSamples = len(self.paths) 
        self.nClasses = 7
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        sample = torch.load(self.paths[index])
        return sample
