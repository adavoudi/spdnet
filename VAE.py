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

# def determinant(A):
#     """
#         returns the determinant of spd matrix A
#     """
#     # try:
#     output = torch.potrf(A).diag().prod()
#     # except:
#         # print(is_pos_def(A.data))
#     return output

# def distance_kullback(A, B):
#     """Kullback leibler divergence between two covariance matrices A and B.
#     :param A: First covariance matrix
#     :param B: Second covariance matrix
#     :returns: Kullback leibler divergence between A and B
#     """
#     dim = A.size(0)
#     logdet = torch.log(determinant(A) / determinant(B))
#     kl = torch.mm(B.inverse(), A).trace() - dim + logdet
#     return 0.5 * kl

num_epochs = 100
batch_size = 100
learning_rate = 1e-3

transformed_dataset = AfewDataset('./data/AFEW', './data/AFEW/spddb_afew_train_spd400_int_histeq.mat', train=True)
dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                    shuffle=True, num_workers=4)

transformed_dataset_val = AfewDataset('./data/AFEW', './data/AFEW/spddb_afew_train_spd400_int_histeq.mat', train=False)
dataloader_val = DataLoader(transformed_dataset_val, batch_size=batch_size,
                    shuffle=False, num_workers=4)


class KullbackDistance(nn.Module):

    def __init__(self):
        super(KullbackDistance, self).__init__()

    def forward(self, A, B):
        dim = A.size(0)
        detA = torch.potrf(A).diag().prod() + 1e-6
        detB = torch.potrf(B).diag().prod() + 1e-6
        logdet = torch.log(detA / detB)
        kl = torch.mm(B.inverse(), A).trace() - dim + logdet
        return 0.5 * kl


class VAELoss(nn.Module):

    def __init__(self, encoded_size):
        super(VAELoss, self).__init__()
        self.register_buffer('identity', torch.eye(encoded_size, encoded_size))
        self.kldist = KullbackDistance()

    def forward(self, recon_x, orig_x, encoded_x):
        # print("recon_x: ", recon_x)
        # print("encoded_x: ", encoded_x)
        identity = Variable(self.identity, requires_grad=False)
        kl_loss_reconstruction = 0
        for index in range(recon_x.size(0)):
            A = orig_x[index]
            B = recon_x[index]
            # print("A: ", is_nan_or_inf(A.data))
            kl_loss_reconstruction -= self.kldist(A, B)
            # print(kl_loss_reconstruction)

        kl_loss_encoding = 0
        for index in range(encoded_x.size(0)):
            kl_loss_encoding -= self.kldist(encoded_x[index], identity)
        
        loss = kl_loss_encoding + kl_loss_reconstruction
        return loss, kl_loss_reconstruction, kl_loss_encoding

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.enc_trans1 = SPDTransform(400, 200)
        self.enc_trans2 = SPDTransform(200, 100)
        self.enc_trans3 = SPDTransform(100, 50)
        self.enc_rec1 = SPDRectified()
        self.enc_rec2 = SPDRectified()
        self.enc_rec3 = SPDRectified()

        self.dec_trans1 = SPDTransform(50, 100)
        self.dec_trans2 = SPDTransform(100, 200)
        self.dec_trans3 = SPDTransform(200, 400)
        self.dec_rec1 = SPDRectified()
        self.dec_rec2 = SPDRectified()
        self.dec_rec3 = SPDRectified()

    def encode(self, x):
        x = self.enc_trans1(x)
        x = self.enc_rec1(x)
        x = self.enc_trans2(x)
        x = self.enc_rec2(x)
        x = self.enc_trans3(x)
        x = self.enc_rec3(x)
        return x

    def decode(self, x):
        x = self.dec_trans1(x)
        x = self.dec_rec1(x)
        x = self.dec_trans2(x)
        x = self.dec_rec2(x)
        x = self.dec_trans3(x)
        x = self.dec_rec3(x)

        return x

    def forward(self, x):
        enc = self.encode(x)
        dec = self.decode(enc)
        return dec, enc

model = VAE()
# model.load_state_dict(torch.load('./vae.pth'))

optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad])
optimizer = StiefelMetaOptimizer(optimizer)



# def loss_function(recon_x, orig_x, encoded_x):

#     identity = Variable(torch.eye(encoded_x.size(1),encoded_x.size(1)), requires_grad=False)

#     kl_loss_reconstruction = 0
#     for index in range(recon_x.size(0)):
#         A = orig_x[index]
#         B = recon_x[index]
#         # print('A:', is_pos_def(A.data))
#         # print('B:', is_pos_def(B.data))
#         kl_loss_reconstruction -= distance_kullback(A, B)

#     kl_loss_encoding = 0
#     # for index in range(encoded_x.size(0)):
#     #     kl_loss_encoding -= distance_kullback(encoded_x[index], identity)
    
#     loss = kl_loss_encoding + kl_loss_reconstruction
    
#     return loss

try:
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_loss_reconstruction = 0
        train_loss_encoding = 0
        for batch_idx, data in enumerate(dataloader):
            inputs = Variable(data['data'], requires_grad=False)
            # targets = Variable(data['label']).squeeze()
            
            dec, enc = model(inputs)
            loss_func = VAELoss(enc.size(1))
            loss, loss_reconstruction, loss_encoding  = loss_func(dec, inputs, enc)
            
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.data[0]
            train_loss_encoding += loss_encoding.data[0]
            train_loss_reconstruction += loss_reconstruction.data[0]
            optimizer.step()

            if batch_idx % 5 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tReconstruction: {:.6f}\tEncoding: {:.6f}'.format(
                    epoch,
                    batch_idx * len(inputs),
                    len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                    train_loss / ((batch_idx+1) * len(inputs)),
                    train_loss_reconstruction / ((batch_idx+1) * len(inputs)),
                    train_loss_encoding / ((batch_idx+1) * len(inputs))))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(dataloader.dataset)))

except KeyboardInterrupt:
    pass

torch.save(model.state_dict(), './vae.pth')