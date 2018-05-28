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

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.trans1 = SPDTransform(400, 200)
        self.trans2 = SPDTransform(200, 100)
        self.trans3 = SPDTransform(100, 50)
        self.rect1  = SPDRectified()
        self.rect2  = SPDRectified()
        self.tangent = SPDTangentSpace()

    def forward(self, x):
        x = self.trans1(x)
        x = self.rect1(x)
        x = self.trans2(x)
        x = self.rect2(x)
        x = self.trans3(x)
        x = self.tangent(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.base = BaseNet()
        self.linear = nn.Linear(1275, 7, bias=False)

    def forward(self, x):
        x = self.base(x)
        x = self.linear(x)
        return x

transformed_dataset = AfewDataset('./data/AFEW', './data/AFEW/spddb_afew_train_spd400_int_histeq.mat', train=True)
dataloader = DataLoader(transformed_dataset, batch_size=30,
                    shuffle=True, num_workers=4)

transformed_dataset_val = AfewDataset('./data/AFEW', './data/AFEW/spddb_afew_train_spd400_int_histeq.mat', train=False)
dataloader_val = DataLoader(transformed_dataset_val, batch_size=30,
                    shuffle=False, num_workers=4)


def test_net():
    for batch_idx, sample_batched in enumerate(dataloader):
        inputs = Variable(sample_batched['data'])
        targets = Variable(sample_batched['label']).squeeze()
        break

    criterion = nn.CrossEntropyLoss()
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0005)
    optimizer = StiefelMetaOptimizer(optimizer)
    before = {key: value.clone() for key,value in model.state_dict().items()}

    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    after = model.state_dict()

    for key, value in before.items():
        if 'epsilon' not in key and (value == after[key]).all():
            return False
    return True

print('Test result: ', test_net())

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
                {'params': model.base.parameters(), 'lr': 1e-4},
                {'params': model.linear.parameters(), 'lr': 1e-7}
            ], weight_decay=0.0005)
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

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1.0), 100.*correct/total, correct, total))

    return (train_loss/(batch_idx+1), 100.*correct/total)

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

        test_loss += loss.data.item()
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

    return (test_loss/(batch_idx+1), 100.*correct/total)

log_file = open('log.txt', 'a')

start_epoch = 1
for epoch in range(start_epoch, start_epoch+500):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)

    log_file.write('%d,%f,%f,%f,%f\n' % (epoch, train_loss, train_acc, test_loss, test_acc))
    log_file.flush()

log_file.close()