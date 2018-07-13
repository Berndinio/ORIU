import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision

from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from .model import VAE,loss_function, reconstruct_and_generate, manipulateData
from ...constants import Constants
from ..MNISTnet.model import Net


parser = argparse.ArgumentParser(description='General parameters')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 0.01)')
args = parser.parse_args()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root=Constants.savesFolder+'MNIST-data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.MNIST(root=Constants.savesFolder+'MNIST-data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                         shuffle=True, num_workers=4)

#load models
mnistNet = torch.load(Constants.savesFolder+'trainedMNIST-NN.pt')
mnistNet = mnistNet.to(Constants.pDevice)


dataiter = iter(test_loader)
images, labels = dataiter.next()
mnistOut = mnistNet(images.to(Constants.pDevice), Constants.VAERepresentationMode)
if Constants.VAERepresentationMode==0:
    _, mnistOut = torch.max(mnistOut, 1, keepdim=True)
    mnistOut = mnistOut.type(torch.FloatTensor).to(Constants.pDevice)
model = VAE(mnistOut)
model.to(Constants.pDevice)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(1, args.epochs + 1):
    model.train()
    for i, (data, _) in enumerate(train_loader):
        data = data.to(Constants.pDevice)
        import copy
        manipulatedData = copy.deepcopy(data)
        manipulatedData = manipulateData(manipulatedData)
        dataToTrainAt = data

        mnistOut = mnistNet(dataToTrainAt, Constants.VAERepresentationMode)
        if Constants.VAERepresentationMode==0:
            _, mnistOut = torch.max(mnistOut, 1, keepdim=True)
            mnistOut = mnistOut.type(torch.FloatTensor).to(Constants.pDevice)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(dataToTrainAt, mnistOut)
        loss_batch = loss_function(recon_batch, data, mu, logvar)
        loss_batch.backward()
        optimizer.step()

        print('Epoch: {} Iter: {}/{} \tLoss: {}'.format(epoch, i * len(data), len(train_loader.dataset),
        loss_batch.data[0] / len(data)))
    reconstruct_and_generate(model, mnistNet, epoch, test_loader)
torch.save(model, Constants.savesFolder+'trainedMNIST-AutoEncoder.pt')
