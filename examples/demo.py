import torch
from torch import nn
from tqdm import tqdm

from dataset import *
from spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified
from spdnet.optimizer import StiefelMetaOptimizer

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.trans1 = SPDTransform(400, 200)
        self.trans2 = SPDTransform(200, 100)
        self.trans3 = SPDTransform(100, 50)
        self.rect1  = SPDRectified()
        self.rect2  = SPDRectified()
        self.rect3  = SPDRectified()
        self.tangent = SPDTangentSpace(50)
        self.linear = nn.Linear(1275, 7, bias=True)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.trans1(x)
        x = self.rect1(x)
        x = self.trans2(x)
        x = self.rect2(x)
        x = self.trans3(x)
        x = self.rect3(x)
        x = self.tangent(x)
        # x = self.dropout(x)
        x = self.linear(x)
        return x

transformed_dataset = AfewDataset(train=True)
dataloader = DataLoader(transformed_dataset, batch_size=30,
                    shuffle=True, num_workers=4)

transformed_dataset_val = AfewDataset(train=False)
dataloader_val = DataLoader(transformed_dataset_val, batch_size=30,
                    shuffle=False, num_workers=4)

use_cuda = True
model = Net()
if use_cuda:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adadelta(model.parameters())
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer = StiefelMetaOptimizer(optimizer)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0.0
    total = 0.0
    bar = tqdm(enumerate(dataloader))
    for batch_idx, sample_batched in bar:
        inputs = sample_batched['data']
        targets = sample_batched['label'].squeeze()

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().data.item()

        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1.0), 100.*correct/total, correct, total))

    return (train_loss/(batch_idx+1), 100.*correct/total)

best_acc = 0
def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0.0
    total = 0.0
    bar = tqdm(enumerate(dataloader_val))
    for batch_idx, sample_batched in bar:
        inputs = sample_batched['data']
        targets = sample_batched['label'].squeeze()

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().data.item()

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