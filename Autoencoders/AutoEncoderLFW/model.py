import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from ...constants import Constants

img_size = 32
n_c = 3

def reconstruct_and_generate(model, epoch, test_loader):
    model.eval()
    for i, (data, _) in enumerate(test_loader):
        batch_size = data.shape[0]
        data = data.to(Constants.pDevice)
        data.volatile=True
        import copy
        manipulatedData = copy.deepcopy(data)
        manipulatedData = manipulateData(manipulatedData)
        dataToUse = manipulatedData

        recon_batch, mu, logvar = model(dataToUse)
        recon_cpu = torch.cat([manipulatedData[:10], recon_batch.view(batch_size, n_c, img_size, img_size)[:10]])
        save_image(recon_cpu.data.cpu(), Constants.savesFolder+'results_Q2/reconstruction_' + str(epoch) + '.png', nrow=10)
        break

    sample = torch.randn(10, 128)
    sample = sample.to(Constants.pDevice)
    sample = model.decode(sample).cpu()
    n_samples = 5
    catTensor = torch.cat((data[:10].data.cpu()[:n_samples], sample.data.view(10, n_c, img_size, img_size).cpu()[:n_samples]))
    save_image(catTensor, Constants.savesFolder+'results_Q2/sample_' + str(epoch) + '.png', nrow=5)


def manipulateData(data):
    data[:,:,13:19,13:19] = 0.0
    return data

def loss_function(recon_x, x, mu, logvar):
    batch_size = logvar.shape[0]
    MSE = F.mse_loss(recon_x, x.view(-1, img_size * img_size * n_c))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size * img_size * img_size * n_c
    return MSE + KLD

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.conv1 = nn.Conv2d(n_c, 3, 2, stride=1)
        self.conv2 = nn.Conv2d(3, 32, 2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 2, stride=1)
        self.conv4 = nn.Conv2d(32, 32, 2, stride=1)

        self.mu = nn.Linear(5408, 128)
        self.logvar = nn.Linear(5408, 128)

        if Constants.useMNISTInput and Constants.useEncoderInput:
            self.fc1 = nn.Linear(128+mnistInput.shape[1], 128)
        elif Constants.useMNISTInput and not Constants.useEncoderInput:
            self.fc1 = nn.Linear(mnistInput.shape[1], 128)
        elif not Constants.useMNISTInput and Constants.useEncoderInput:
            self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 8192)
        self.deconv1=nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2,             padding=0, output_padding=0)
        self.deconv2=nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2,             padding=1, output_padding=0)
        self.deconv3=nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2,   padding=0, output_padding=0)
        self.deconv4=nn.ConvTranspose2d(in_channels=32, out_channels=3,  kernel_size=2,             padding=1, output_padding=0)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        h1 = self.relu(self.conv1(x))
        h1 = self.relu(self.conv2(h1))
        h1 = self.relu(self.conv3(h1))
        h1 = self.relu(self.conv4(h1))
        h1 = h1.view(-1, 5408)
        return self.mu(h1), self.logvar(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc1(z))
        h3 = self.relu(self.fc2(h3))
        h3 = h3.view(-1, 32, 16, 16)
        h3 = self.deconv1(h3)
        h3 = self.deconv2(h3)
        h3 = self.deconv3(h3)
        h3 = self.deconv4(h3)
        return h3

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        return z.view(-1, 3*32*32), mu, logvar
