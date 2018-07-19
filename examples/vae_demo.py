import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
from tqdm import tqdm
from dataset import *
from SPDNet import *
from MKNet import *

transformed_dataset = AfewDataset('/home/alireza/projects/TF_SPDNet/data/AFEW', '/home/alireza/projects/SPDNet/data/afew/spddb_afew_train_spd400_int_histeq.mat', train=True)
dataloader = DataLoader(transformed_dataset, batch_size=30,
                    shuffle=True, num_workers=4)

transformed_dataset_val = AfewDataset('/home/alireza/projects/TF_SPDNet/data/AFEW', '/home/alireza/projects/SPDNet/data/afew/spddb_afew_train_spd400_int_histeq.mat', train=False)
dataloader_val = DataLoader(transformed_dataset_val, batch_size=30,
                    shuffle=False, num_workers=4)

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.trans1 = SPDTransform(400, 200)
        self.trans2 = SPDTransform(200, 100)
        self.trans3 = SPDTransform(100, 50)
        # self.rec1 = SPDRectified()
        # self.rec2 = SPDRectified()
        self.tangent = SPDTangentSpace(True) 
        self.bn =  nn.BatchNorm1d(1275)
        # self.bn0 =  nn.BatchNorm1d(1275)
        self.fc21 = nn.Linear(1275, 40, bias=True)
        self.fc22 = nn.Linear(1275, 40, bias=True)
        self.fc3 = nn.Linear(40, 400)
        self.fc4 = nn.Linear(400, 1275)
        self.fc5 = nn.Linear(1275, 80200)
        self.linear = nn.Linear(1275, 7, bias=True)

    def encode(self, x):
        x = self.trans1(x)
        # x = self.rec1(x)
        # x = self.trans2(x)
        # x = self.rec2(x)
        # x = self.trans3(x)
        x = self.tangent(x)
        x = self.bn(x)
        return self.fc21(x), self.fc22(x), x

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        return self.fc5(h4)

    def forward(self, x):
        mu, logvar, clsi = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar, clsi

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(1275, 7, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # x = self.dropout(x)
        x = self.linear(x)
        return x


model = VAE()
model.load_state_dict(torch.load('vae.pth'))
model2 = Net()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model2.parameters())
# optimizer = StiefelMetaOptimizer(optimizer)

# reconstruction_function = nn.MSELoss(size_average=False)
# tangent_space = SPDTangentSpace(True)

# def loss_function(recon_x, x, mu, logvar):
#     """
#     recon_x: generating images
#     x: origin images
#     mu: latent mean
#     logvar: latent log variance
#     """
#     BCE = reconstruction_function(recon_x, tangent_space(x))  # mse loss
#     # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
#     KLD = torch.sum(KLD_element).mul_(-0.5)
#     # KL divergence
#     return BCE + KLD

# try:
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0
#         for batch_idx, data in enumerate(dataloader):
#             img = Variable(data['data'])
#             optimizer.zero_grad()
#             recon_batch, mu, logvar = model(img)
#             loss = loss_function(recon_batch, img, mu, logvar)
#             loss.backward()
#             train_loss += loss.data[0]
#             optimizer.step()
#             if batch_idx % 10 == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     epoch,
#                     batch_idx * len(img),
#                     len(dataloader.dataset), 100. * batch_idx / len(dataloader),
#                     loss.data[0] / len(img)))

#         print('====> Epoch: {} Average loss: {:.4f}'.format(
#             epoch, train_loss / len(dataloader.dataset)))
# except KeyboardInterrupt:
#     pass

# torch.save(model.state_dict(), './vae.pth')

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
        _,_,out = model.encode(inputs)
        outputs = model2(out)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # print(model.pow.weight.view(1, -1))

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

        _,_,out = model.encode(inputs)
        outputs = model2(out)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': model,
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.t7')
    #     best_acc = acc

start_epoch = 1
for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    test(epoch)