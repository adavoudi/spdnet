import os
import random
import numpy as np
import scipy.io as spio
from torch.utils.data import Dataset, DataLoader
import torch

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
    def __init__(self, base_path, dataset_path, shuffle=True, train=False):
        super(AfewDataset, self).__init__()

        self.train = train
        self.base_path = base_path
        self.dataset_path = dataset_path

        dataset = loadmat(dataset_path)
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
        self.nClasses = 7

    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):
        index = self.data_index[idx]
        data_path = os.path.join(self.base_path, self.spd_path[index])
        data = loadmat(data_path)
        data = data['Y1']
        label = np.asarray(self.labels[index] - 1).astype(np.long)

        sample = {'data': torch.from_numpy(data), 'label': torch.from_numpy(label)}

        return sample


if __name__ == '__main__':

    transformed_dataset = AfewDataset('./data/AFEW', '/home/alireza/projects/SPDNet/data/afew/spddb_afew_train_spd400_int_histeq.mat')
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=False, num_workers=4)

    
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['data'].size(),
          sample_batched['label'].size())


