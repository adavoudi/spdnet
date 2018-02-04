from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import os

import torchvision
import torchvision.transforms as transforms

from MKNet import *
from SPDNet import *

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, relu=True, bn=True):
        super(BasicBlock, self).__init__()
        self.use_relu = relu
        self.use_bn = bn
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)
        if relu:
            self.relu = torch.nn.ReLU()
        if bn:
            self.bn =  nn.BatchNorm2d(out_channels)

    def forward(self, input):
        out = self.conv(input)
        if self.use_relu:
            out = self.relu(out)
        if self.use_bn:
            out = self.bn(out)
        return out


class CifarNet_1_Base(nn.Module):
    def __init__(self):
        super(CifarNet_1_Base, self).__init__()
        self.block_1 = BasicBlock(3, 16, 2) # 16
        self.block_2 = BasicBlock(16, 16, 1) # 16
        self.block_3 = BasicBlock(16, 16, 1) # 16
        self.block_4 = BasicBlock(16, 32, 2) # 8
        self.block_5 = BasicBlock(32, 32, 1) # 8
        self.block_6 = BasicBlock(32, 64, 2, bn=False, relu=False) # 4
        self.linear = nn.Linear(1024, 10, bias=False)

    def forward(self, input, out_conv=False):
        
        out = self.block_1(input)
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)
        out = self.block_5(out)
        out = self.block_6(out)

        if not out_conv:
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        else:
            out = out.view(out.size(0), out.size(1), -1)

        return out


class CifarNet_1(nn.Module):
    def __init__(self):
        super(CifarNet_1, self).__init__()
        self.gkernel_1 = GaussianKernel(64, subtract_mean=True, use_center=False, kernel_width=0.1, laplacian_kernel=False, center_init_scale=1)
        # self.gkernel_2 = GaussianKernel(64, use_center=False, kernel_width=None, laplacian_kernel=False, center_init_scale=3)
        # self.mix = MixKernel(use_weight_for_a=True, use_weight_for_b=True)
        self.trans = SPDTransform(64, 20)
        self.rect = SPDRectified()
        self.tangent = SPDTangentSpace() 
        self.linear = nn.Linear(210, 10, bias=True)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input):
        
        out = self.gkernel_1(input)
        # out2 = self.gkernel_2(input)
        # out = self.mix(out1, out2)
        out = self.trans(out)
        out = self.rect(out)
        out = self.tangent(out)
        out = self.dropout(out)
        out = self.linear(out)

        return out


spdnet = CifarNet_1()
model_base = CifarNet_1_Base()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

use_cuda = False
criterion = nn.CrossEntropyLoss()

optimizer_base = optim.Adam(model_base.parameters(), lr=0.01)
optimizer_spdnet = optim.Adam(spdnet.parameters(), lr=0.0001)
optimizer_spdnet = StiefelMetaOptimizer(optimizer_spdnet)

# Training
def train(epoch, train_spdnet=False):
    print('\nEpoch: %d' % epoch)
    spdnet.train()
    model_base.train()
    train_loss = 0
    correct = 0
    total = 0
    bar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets) in bar:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets
        
        optimizer_base.zero_grad()

        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model_base(inputs)
        loss = criterion(outputs, targets)        
        loss.backward()
        optimizer_base.step()

        if train_spdnet:
            optimizer_spdnet.zero_grad()
            outputs = model_base(inputs, True)
            outputs = spdnet(outputs)
            loss = criterion(outputs, targets)        
            loss.backward()
            optimizer_spdnet.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, test_spdnet=False):
    global best_acc
    model_base.eval()
    spdnet.eval()
    test_loss = 0
    correct = 0
    total = 0
    bar = tqdm(enumerate(testloader))
    for batch_idx, (inputs, targets) in bar:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        
        if test_spdnet:
            outputs = model_base(inputs, True)
            outputs = spdnet(outputs)
        else:
            outputs = model_base(inputs, False)

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
            'base': model_base,
            'net': spdnet,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/kspd-ckpt.t7')
        best_acc = acc


for epoch in range(1):
    train(epoch)
    test(epoch)

print('Training SPDNet ... ')

for epoch in range(20):
    train(epoch, True)
    test(epoch, True)