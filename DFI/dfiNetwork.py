import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import hickle as hkl
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from ..constants import Constants
import h5py
import sys
from .openFace.loadOpenFace import prepareOpenFace
from ..Autoencoders.MNISTnet.model import Net as MNISTNet

class VGG19Net(nn.ModuleList):
    def __init__(self, weightsToLoad=None):
        print("Loading network weights...")
        super(VGG19Net, self).__init__()

        if(Constants.cudaAvailable):
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        self.cfg = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M', (25088,4096), (4096,4096), (4096,1000)],
        }
        #save the output dimensions for later reconstruction
        self.reconstructionDimensions = []
        self.reconstructionSaved = False
        #build VGG19
        self.layers = []
        self.paramLayers = []
        lastNum = 3
        weightCounter = 0

        for num,i in enumerate(self.cfg['E']):
            if type(i) == int or type(i) == tuple:
                if num<(len(self.cfg['E'])-3):
                    self.layers.append(nn.Conv2d(lastNum, i, 3, padding=1))
                else:
                    self.layers.append(nn.Linear(i[0],i[1]))
                if not(weightsToLoad is None):
                    #load our weights, if file is given
                    #print(self.layers[-1].weight.shape, weightsToLoad[weightCounter].shape)
                    self.layers[-1].weight = nn.Parameter(weightsToLoad[weightCounter])
                    weightCounter += 1
                    #print(self.layers[-1].bias.shape, weightsToLoad[weightCounter].shape)
                    self.layers[-1].bias = nn.Parameter(weightsToLoad[weightCounter])
                    weightCounter += 1
                self.paramLayers.append(self.layers[-1].weight)
                self.paramLayers.append(self.layers[-1].bias)
                lastNum = i
            if type(i) == str:
                self.layers.append(nn.MaxPool2d(2, stride=2))

        #register parameters
        self.params = nn.ParameterList(self.paramLayers)
        #self.params = nn.ParameterList([])
        print("Network loaded...")

    def flattenConcat(self, convOut, alreadyFlat, pMode=0):
        """
        Flattens our inputs for the first FC layer.
        """
        shape = convOut.data.shape
        shape0 = shape[0]
        #shape1 = shape[1] * shape[2] * shape[3]
        flattened = convOut.view(shape0,-1)
        #print(convOut.shape, flattened.shape)
        if(len(alreadyFlat.data.shape) == 0):
            return flattened
        return torch.cat((alreadyFlat.to(Constants.pDevice), flattened ), 1)


    def forward(self, x, mode=0):
        """Forwards our input and extracts our flattened wanted features.

        Parameters
        ----------
        x : type Tensor
            Tensor of input images.

        Returns
        -------
        type Tensor
            Output of the network.

        """
        outs = Variable(torch.FloatTensor())
        if(mode==0):
            #layers like in the paper
            layers = [6, 11, 16]
        elif mode==1:
            #only the first conv layer of the paper
            layers = [6]
        elif mode==2:
            #only the first conv layer of the paper
            layers = [9]
        elif mode==3:
            #only the second conv layer of the paper
            layers = [11]
        elif mode==4:
            #only the second conv layer of the paper
            layers = [14]
        elif mode==5:
            #only the third conv layer of the paper
            layers = [16]
        elif mode==6:
            #only the third conv layer of the paper
            layers = [19]
        elif mode==7:
            #first FC layer
            layers = [21]
        elif mode==8:
            #second FC layer
            layers = [22]
        elif mode==9:
            #thrid FC layer
            layers = [23]

        for idx,i in enumerate(self.cfg['E']):
            if type(i) == int:
                x = F.relu(self.layers[idx](x))
            if type(i) == str:
                x = self.layers[idx](x)
            #3 FC layers
            if type(i) == tuple and idx==len(self.cfg['E'])-3:
                try:
                    x = x.view(x.size(0), -1)
                    x = F.relu(self.layers[idx](x))
                except Exception as e:
                    print("Size should be: "+str(x.shape)+".\nLayer is: "+str(self.layers[idx]))
            if type(i) == tuple and idx==len(self.cfg['E'])-2:
                x = F.relu(self.layers[idx](x))
            if type(i) == tuple and idx==len(self.cfg['E'])-1:
                x = F.softmax(self.layers[idx](x), dim=0)
            if(idx in layers):
                if not self.reconstructionSaved:
                    shape = x.data.shape
                    self.reconstructionDimensions.append(shape)
                outs = self.flattenConcat(x,outs)
        self.reconstructionSaved = True
        return outs


class dfiNetwork:
    def __init__(self ,pFile=None, pMaxImagesCount= 80, netToLoad="VGG19"):
        print("Beginning to load "+netToLoad)
        self.netType = netToLoad
        self.resize = 0
        #this if else is from pytorch version<4.0
        if(Constants.cudaAvailable):
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if(self.netType=="VGG19" or self.netType==None):
                self.net = VGG19Net(self.loadWeightsProcess(pFile)).to(Constants.pDevice)
                self.resize = 224
            elif(self.netType=="OpenFace"):
                self.net = prepareOpenFace(useCuda=True, gpuDevice=0, useMultiGPU=False).eval()
                self.resize = 96
            elif(self.netType=="MNISTNet"):
                self.net = torch.load(Constants.savesFolder+"pretrainedMNIST-NN.pt").to(Constants.pDevice)
                self.resize = 28
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            if(self.netType=="VGG19" or self.netType==None):
                self.net = VGG19Net(self.loadWeightsProcess(pFile))
                self.resize = 224
            elif(self.netType=="OpenFace"):
                self.net = prepareOpenFace(useCuda=False, gpuDevice=0, useMultiGPU=False).eval()
                self.resize = 96
            elif(self.netType=="MNISTNet"):
                self.net = torch.load(Constants.savesFolder+"pretrainedMNIST-NN.pt")
                self.resize = 28


        self.trans = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

        self.invTrans = transforms.Compose([
            transforms.Normalize(mean=[ 0., 0., 0. ],
                                  std=[ 1/0.229, 1/0.224, 1/0.225 ]),
            transforms.Normalize(mean=[ -0.485, -0.456, -0.406 ],
                                  std=[ 1., 1., 1. ]),])

        self.trans = transforms.Compose([
            transforms.Resize((self.resize,self.resize)),
            transforms.ToTensor(),])

        self.images = None
        self.imageCounter = 0
        self.maxImagesCount = pMaxImagesCount

        self.input = None
        self.output = None


    def appendImages(self, images):
        """Append a list of images to que

        Parameters
        ----------
        images : type List
            List of images (PIL).
        """
        for img in images:
            self.appendImage(img)

    def appendImage(self,image):
        """Appends an image into the list, which can be forwarded through the
        network with self.forwardAll.

        Parameters
        ----------
        image : type Image (PIL)
            The image to append into que.
        """
        img = self.trans(image).unsqueeze_(0)
        if(self.imageCounter == 0):
            self.images = img
        else:
            self.images = torch.cat((self.images, img),0)

        if(self.images.size()[0]>self.maxImagesCount):
            self.images = self.images[:-1]
            raise Exception("DfiNetwork", "Tried to append images too large for usable memory.")
        self.imageCounter += 1

    def flushImages(self):
        """
        Deletes all images from the que.
        """
        #del self.input
        self.input = None
        self.output = None
        self.images = None
        self.imageCounter = 0

    def forwardAll(self, pMode=0):
        """Forwards all images from the que.

        Returns
        -------
        type Tensor
            Tensor of output of the network

        """
        if(Constants.cudaAvailable):
            self.input = Variable(self.images.to(Constants.pDevice), requires_grad=True).to(Constants.pDevice)
        else:
            self.input = Variable(self.images, requires_grad=True)
        self.output = self.net(self.input, mode=pMode)
        return self.output

    def loadWeightsProcess(self, pfile):
        """Preprocesses our h5 weight file.
        Parameters
        ----------
        pfile : type String
            Filepath of source weight file.

        Returns
        -------
        type list
            List of weights (weight & bias).

        """

        if(pfile==None):
            return None
        h5_file = h5py.File(pfile)
        keys = list(h5_file.keys())

        endParams = []
        # convert numpy arrays to torch Variables
        for i in range(len(keys)):
            idx = "/layer_"+str(i)
            for k in list(h5_file[idx].keys()):
                idxx = idx+"/"+k
                #print(h5_file[idxx])
                #skip 3 bias + 3 linear weight layer
                if(i<len(keys)-6):
                    endParams.append(torch.from_numpy(np.array(h5_file[idxx])))
                else:
                    x = torch.from_numpy(np.array(h5_file[idxx]))
                    if(len(x.shape)>1):
                        x = torch.t(x)
                    endParams.append(x)
        return endParams

    #lam is the smoothning factor
    def reconstructImage(self, targetVector, NUM_ITER=1, saveIt=False, lam=0.01, LR=2.0):

        finalImages = []
        #if it crashes due to gradient escalation
        try:
            from PIL import Image
            self.flushImages()
            image = Image.new('RGB', (224, 224), color=0)
            self.appendImage(image)
            self.forwardAll(Constants.reconstructionMode)

            ##best
            #LR = 2.0
            #diffClamp = 20
            #gradClamp = 0.2
            #lam = 0.01
            diffClamp = 20
            gradClamp = 0.1

            optimizer = torch.optim.LBFGS([self.input], lr=LR)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,350,450,500], gamma=0.5)
            for i in range(NUM_ITER):
                print("Iteration "+str(i)+"/"+str(NUM_ITER))

                #append image
                if(i%int(NUM_ITER/20)==0):
                    finalImages.append(self.input[0].cpu().data)
                def closure():
                    optimizer.zero_grad()
                    act_value = self.net(self.input)
                    #euclidean loss part
                    diff = act_value-targetVector
                    diff = torch.clamp(diff,-diffClamp ,diffClamp)
                    loss = torch.norm(diff, p=2)
                    #neighbor pixel loss
                    squaredX = torch.sum(lam*(self.input[0,:,1:] - self.input[0,:,:-1])*(self.input[0,:,1:] - self.input[0,:,:-1]))
                    squaredY = torch.sum(lam*(self.input[0,:,:,1:] - self.input[0,:,:,:-1])*(self.input[0,:,:,1:] - self.input[0,:,:,:-1]))
                    loss = loss + squaredX + squaredY
                    #print(self.input)
                    loss.backward(retain_graph=True)
                    self.input.grad = torch.clamp(self.input.grad, -gradClamp, gradClamp)
                    #print(self.input.grad)
                    return loss
                scheduler.step()
                optimizer.step(closure)
            #append last time
            finalImages.append(self.input[0].cpu().data)
            return finalImages
        except Exception as e:
            print(e)
            return finalImages

if __name__ == "__main__":
    pass
