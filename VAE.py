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

        self.trans1 = SPDTransform(400, 50)
        # self.trans2 = SPDTransform(200, 100)
        # self.trans3 = SPDTransform(100, 20)
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
        return self.fc21(x), self.fc22(x), self.linear(x)

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

model = VAE()
model.load_state_dict(torch.load('./vae.pth'))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
optimizer = StiefelMetaOptimizer(optimizer)

reconstruction_function = nn.MSELoss(size_average=False)
tangent_space = SPDTangentSpace(True)
criterion_cls = nn.CrossEntropyLoss()

def loss_function(recon_x, x, mu, logvar, clsi, clsorg):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, tangent_space(x))  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    loss_cls = criterion(clsi, clsorg)
    # KL divergence
    return BCE + KLD + loss_cls

try:
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):
            img = Variable(data['data'])
            target = Variable(data['label']).squeeze()
            optimizer.zero_grad()
            recon_batch, mu, logvar, clsi = model(img)
            loss = loss_function(recon_batch, img, mu, logvar, clsi, target)
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(img),
                    len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                    loss.data[0] / len(img)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(dataloader.dataset)))
except KeyboardInterrupt:
    pass

torch.save(model.state_dict(), './vae.pth')