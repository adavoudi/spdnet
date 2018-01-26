import torch
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer, required
from torch import nn
from torch.autograd import Function
import math
import numpy as np

from dataset import *
from SPDNet import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.gkernel = GaussianKernel(2)
        self.trans1 = SPDTransform(400, 200)
        self.trans2 = SPDTransform(200, 100)
        self.trans3 = SPDTransform(100, 50)
        self.rect1  = SPDRectified()
        self.rect2  = SPDRectified()
        self.rect3  = SPDRectified()
        self.tangent = SPDTangentSpace() 
        self.linear = nn.Linear(1275, 7, bias=False)

    def forward(self, x):
        # x = self.gkernel(x)
        # print('-1: ', is_pos_def(x))
        x = self.trans1(x)
        # print('0: ', is_pos_def(x))
        x = self.rect1(x)
        x = self.trans2(x)
        # print('1: ', is_pos_def(x))
        x = self.rect2(x)
        x = self.trans3(x)
        # print('2: ', is_pos_def(x))
        x = self.rect3(x)
        x = self.tangent(x)
        x = self.linear(x)
        # print('3: ', x)
        return x


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


from torch.autograd import Variable

model = Net()
model.train()
# print([p.type for p in model.parameters()])
loss_fn = loss = nn.CrossEntropyLoss()
optimizer = StiefelMetaOptimizer(torch.optim.Adam(model.parameters()))

transformed_dataset = AfewDataset('./data/AFEW', '/home/alireza/projects/SPDNet/data/afew/spddb_afew_train_spd400_int_histeq.mat', train=True)
dataloader = DataLoader(transformed_dataset, batch_size=30,
                    shuffle=True, num_workers=4)

transformed_dataset_val = AfewDataset('./data/AFEW', '/home/alireza/projects/SPDNet/data/afew/spddb_afew_train_spd400_int_histeq.mat', train=False)
dataloader_val = DataLoader(transformed_dataset_val, batch_size=30,
                    shuffle=False, num_workers=4)

from tqdm import tqdm

for e in range(200):
    print('Epoch: ', e)
    bar = tqdm(enumerate(dataloader))
    mean_acc = 0
    total_batch = 0
    for i_batch, sample_batched in bar:
        total_batch += 1
        x = Variable(sample_batched['data'])
        y = Variable(sample_batched['label']).squeeze()
        
        y_pred = model(x)
        
        loss = loss_fn(y_pred, y)
        
        acc = accuracy(y_pred.data, y.data)[0][0]
        mean_acc += acc
        bar.set_description('{:0.3f}, acc: {:0.3f}'.format(loss.data[0], acc))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_acc /= total_batch
    print('Mean acc: ', mean_acc)

    if e % 5 == 0:
        print('Performing validation:')
        mean_acc = 0
        mean_loss = 0
        total_samples = 0
        for i_batch, sample_batched in enumerate(dataloader_val):
            x = Variable(sample_batched['data'])
            y = Variable(sample_batched['label']).squeeze()
                        
            y_pred = model(x)
            loss = loss_fn(y_pred, y).data[0]
            acc = accuracy(y_pred.data, y.data)[0][0]
            
            n_samples = x.size(0)
            total_samples += n_samples
            mean_acc += acc * n_samples
            mean_loss += loss * n_samples

        print('Mean loss: {:0.3f}, mean acc: {:0.3f}'.format(mean_loss/total_samples, mean_acc/total_samples))


