import torch
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer, required
from torch import nn
from torch.autograd import Function
import math
import numpy as np
from tqdm import tqdm

from dataset import *
from SPDNet import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.trans1 = SPDTransform(400, 50)
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
        # x = self.trans2(x)
        # print('1: ', is_pos_def(x))
        # x = self.rect2(x)
        # x = self.trans3(x)
        # print('2: ', is_pos_def(x))
        # x = self.rect3(x)
        x = self.tangent(x)
        x = self.linear(x)
        # print('3: ', x)
        return x




transformed_dataset = AfewDataset('/home/alireza/projects/TF_SPDNet/data/AFEW', '/home/alireza/projects/SPDNet/data/afew/spddb_afew_train_spd400_int_histeq.mat', train=True)
dataloader = DataLoader(transformed_dataset, batch_size=30,
                    shuffle=True, num_workers=4)

transformed_dataset_val = AfewDataset('/home/alireza/projects/TF_SPDNet/data/AFEW', '/home/alireza/projects/SPDNet/data/afew/spddb_afew_train_spd400_int_histeq.mat', train=False)
dataloader_val = DataLoader(transformed_dataset_val, batch_size=30,
                    shuffle=False, num_workers=4)


model = Net()
# model = model.double()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = StiefelMetaOptimizer(optimizer)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    bar = tqdm(enumerate(dataloader))
    for batch_idx, sample_batched in bar:
        inputs = Variable(sample_batched['data'])
        targets = Variable(sample_batched['label']).squeeze()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

best_acc = 0
def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    bar = tqdm(enumerate(dataloader_val))
    for batch_idx, sample_batched in bar:
        inputs = Variable(sample_batched['data'])
        targets = Variable(sample_batched['label']).squeeze()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

start_epoch = 1
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)