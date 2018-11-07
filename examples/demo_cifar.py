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

from spdnet.spd import SPDTransform, ParametricVectorize
from spdnet.optimizer import StiefelMetaOptimizer
from spdnet.kernel import Covariance


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


class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.block_1 = BasicBlock(3, 16, 2) # 16
        self.block_2 = BasicBlock(16, 16, 1) # 16
        self.block_3 = BasicBlock(16, 16, 1) # 16
        self.block_4 = BasicBlock(16, 32, 2) # 8
        self.block_5 = BasicBlock(32, 32, 1) # 8
        self.block_6 = BasicBlock(32, 64, 2, bn=False, relu=False)
        self.gkernel = Covariance()#GaussianKernel(64, kernel_width=0.1, laplacian_kernel=False)
        self.trans = SPDTransform(65, 32)
        self.relu = nn.ReLU()
        self.vec = ParametricVectorize(32, 20)
        self.linear = nn.Linear(20, 10, bias=True)

    def forward(self, input, out_conv=False):
        
        out = self.block_1(input)
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)
        out = self.block_5(out)
        out = self.block_6(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = self.gkernel(out)
        out = self.trans(out)
        out = self.relu(out)
        out = self.vec(out)
        out = self.linear(out)
        
        return out



net = CifarNet()

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

use_cuda = True

if use_cuda:
    net = net.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)
optimizer = StiefelMetaOptimizer(optimizer)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0.0
    total = 0.0
    bar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets) in bar:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)        
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().data.item()

        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0.0
    bar = tqdm(enumerate(testloader))
    for batch_idx, (inputs, targets) in bar:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().data.item()

        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'base': model_base,
    #         'net': spdnet,
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/kspd-ckpt.t7')
    #     best_acc = acc

for epoch in range(200):
    train(epoch)
    test(epoch)