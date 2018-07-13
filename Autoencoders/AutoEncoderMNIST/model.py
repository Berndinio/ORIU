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

img_size = 28
n_c = 1

def reconstruct_and_generate(model, mnistNet, epoch, test_loader):
    model.eval()
    for i, (data, _) in enumerate(test_loader):
        data = data.to(Constants.pDevice)
        data.volatile=True
        import copy
        manipulatedData = copy.deepcopy(data)
        manipulatedData = manipulateData(manipulatedData)

        mnistOut = mnistNet(manipulatedData[:10], Constants.VAERepresentationMode)
        if Constants.VAERepresentationMode==0:
            _, mnistOut = torch.max(mnistOut, 1, keepdim=True)
            mnistOut = mnistOut.type(torch.FloatTensor).to(Constants.pDevice)

        if not Constants.useRandom:
            recon_batch, mu, logvar = model(manipulatedData[:10], mnistOut[:10])
            recon_batch = recon_batch.cpu()
        elif Constants.useRandom:
            generatedRandom = torch.randn(10, 64).float().to(Constants.pDevice)
            recon_batch = model.decode(generatedRandom, mnistOut[:10]).cpu()
        filledData = copy.deepcopy(manipulatedData[:10])
        filledData[:10,:,13:19,13:19] = recon_batch.view(10, n_c, img_size, img_size)[:10,:,13:19,13:19]
        recon_cpu = torch.cat([manipulatedData[:10].cpu(), recon_batch.view(10, n_c, img_size, img_size)[:10]])
        recon_cpu = torch.cat([recon_cpu, filledData.cpu()])
        save_image(recon_cpu.data.cpu(), Constants.savesFolder+'results_Q2/reconstruction_' + str(epoch) + '.png', nrow=10)

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

        self.fc1 = nn.Linear(img_size * img_size * n_c, 512)
        self.fc2 = nn.Linear(512, 256)
        self.mu = nn.Linear(256, 64)
        self.logvar = nn.Linear(256, 64)

        if Constants.useMNISTInput and Constants.useEncoderInput:
            self.fc4 = nn.Linear(64+mnistInput.shape[1], 256)
        elif Constants.useMNISTInput and not Constants.useEncoderInput:
            self.fc4 = nn.Linear(mnistInput.shape[1], 256)
        elif not Constants.useMNISTInput and Constants.useEncoderInput:
            self.fc4 = nn.Linear(64, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, img_size * img_size * n_c)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h1 = self.relu(self.fc2(h1))
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
        h3 = self.tanh(self.fc6(h3))
        return h3

    def forward(self, x, mnistInput):
        mu, logvar = self.encode(x.view(-1, img_size * img_size * n_c))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, mnistInput), mu, logvar
