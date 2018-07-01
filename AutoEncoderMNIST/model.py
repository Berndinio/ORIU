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
from ..constants import Constants

img_size = 32
n_c = 3

def reconstruct_and_generate(model, mnistNet, epoch, test_loader):
    model.eval()
    for i, (data, _) in enumerate(test_loader):
        batch_size = data.shape[0]
        data = data.to(Constants.pDevice)
        data.volatile=True
        import copy
        manipulatedData = copy.deepcopy(data)
        manipulatedData = manipulateData(manipulatedData)
        dataToUse = manipulatedData

        mnistOut = mnistNet(dataToUse, Constants.VAERepresentationMode)
        if Constants.VAERepresentationMode==0:
            _, mnistOut = torch.max(mnistOut, 1, keepdim=True)
            mnistOut = mnistOut.type(torch.FloatTensor).to(Constants.pDevice)

        recon_batch, mu, logvar = model(dataToUse, mnistOut)
        recon_cpu = torch.cat([manipulatedData[:10], recon_batch.view(batch_size, n_c, img_size, img_size)[:10]])
        save_image(recon_cpu.data.cpu(), Constants.savesFolder+'results_Q2/reconstruction_' + str(epoch) + '.png', nrow=10)
        break

    sample = torch.randn(10, 128)
    sample = sample.to(Constants.pDevice)
    sample = model.decode(sample, mnistOut[:10]).cpu()
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
    def __init__(self, mnistInput):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(img_size * img_size * n_c, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.mu = nn.Linear(256, 128)
        self.logvar = nn.Linear(256, 128)

        if Constants.useMNISTInput and Constants.useEncoderInput:
            self.fc4 = nn.Linear(128+mnistInput.shape[1], 256)
        elif Constants.useMNISTInput and not Constants.useEncoderInput:
            self.fc4 = nn.Linear(mnistInput.shape[1], 256)
        elif not Constants.useMNISTInput and Constants.useEncoderInput:
            self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 1024)
        self.fc7 = nn.Linear(1024, img_size * img_size * n_c)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h1 = self.relu(self.fc2(h1))
        h1 = self.relu(self.fc3(h1))
        return self.mu(h1), self.logvar(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, mnistInput):
        if Constants.useMNISTInput and Constants.useEncoderInput:
            z = torch.cat((z, mnistInput), dim=1)
        elif Constants.useMNISTInput and not Constants.useEncoderInput:
            z = mnistInput
        elif not Constants.useMNISTInput and Constants.useEncoderInput:
            z = z
        h3 = self.relu(self.fc4(z))
        h3 = self.relu(self.fc5(h3))
        h3 = self.relu(self.fc6(h3))
        return self.tanh(self.fc7(h3))

    def forward(self, x, mnistInput):
        mu, logvar = self.encode(x.view(-1, img_size * img_size * n_c))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, mnistInput), mu, logvar
