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

from .model import VAE,loss_function, reconstruct_and_generate
from ...constants import Constants
from ..MNISTnet.model import Net
from ...dataset.lfwDataset import lfwDataset as LFW


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
    [transforms.Resize(32),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = LFW("lfwCropped", transform=transform, train=True)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=4)
testset = LFW("lfwCropped", transform=transform, train=False)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=4)

trainset_rectangle = LFW("lfwCroppedRectangle", transform=transform, train=True)
train_loader_rectangle = torch.utils.data.DataLoader(trainset_rectangle, batch_size=args.batch_size,
                                          shuffle=False, num_workers=4)
testset_rectangle = LFW("lfwCroppedRectangle", transform=transform, train=False)
test_loader_rectangle = torch.utils.data.DataLoader(testset_rectangle, batch_size=args.batch_size,
                                          shuffle=False, num_workers=4)



model = VAE()
model.to(Constants.pDevice)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(1, args.epochs + 1):
    model.train()
    for i, ((data, _),(data_rectangle, _)) in enumerate(zip(train_loader, train_loader_rectangle)):
        data = data.to(Constants.pDevice)
        dataToTrainAt = data_rectangle.to(Constants.pDevice)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(dataToTrainAt)
        loss_batch = loss_function(recon_batch, data, mu, logvar)
        loss_batch.backward()
        optimizer.step()

        print('Epoch: {} Iter: {}/{} \tLoss: {}'.format(epoch, i * len(data), len(train_loader.dataset),
        loss_batch.data[0] / len(data)))

    reconstruct_and_generate(model, epoch, test_loader, test_loader_rectangle)

torch.save(model, Constants.savesFolder+'trainedLFW-AutoEncoder.pt')
