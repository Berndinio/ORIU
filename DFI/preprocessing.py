import os
import faiss
import numpy as np
from .dfiNetwork import dfiNetwork
from ..constants import Constants
from PIL import Image
import os.path
import torch
import pickle
import klepto
import matplotlib.pyplot as plt
from ..dataset.lfwDataset import lfwDataset as LFW
from torchvision import datasets, transforms


class Preprocessing:
    def __init__(self, netToLoad="VGG19"):
        self.net = dfiNetwork(Constants.weightsFile, netToLoad=netToLoad)

    #Neg = source set
    def getKNN(self, faissPositive, faissNegative, imageToManipulate, k=100):
        #get DFR and NORMALIZE it
        self.net.flushImages()
        self.net.appendImage(imageToManipulate)
        output = self.net.forwardAll(Constants.KNNIndexMode)
        self.net.flushImages()
        findKNNTo = output.cpu().data.numpy()
        normsPos = torch.unsqueeze(torch.norm(output, 2, 1),1)
        normalizedPosOutput = output/normsPos.expand_as(output)
        findKNNToNormalized = normalizedPosOutput.cpu().data.numpy()

        #search for neighbors
        distPos, indexPos = faissPositive.search(findKNNToNormalized, k)
        distNeg, indexNeg = faissNegative.search(findKNNToNormalized, k)
        #load images and get the DFR for reconstruction
        indexPos = indexPos[0]
        indexNeg = indexNeg[0]
        return indexPos, indexNeg

    def getManipulatedVector(self, imageToManipulate, listPos, listNeg, indexPos, indexNeg, datasetPos, datasetNeg, factorManipulation=4.0):
        k = len(indexPos)
        selectedPos = None
        selectedNeg = None
        for i, (iPos, iNeg) in enumerate(zip(indexPos, indexNeg)):
            print(str(i)+"/"+str(k))
            listPos[iPos]
            imgPos, _ = datasetPos[listPos[iPos]]
            imgNeg, _ = datasetNeg[listNeg[iNeg]]
            self.net.appendImage(imgPos)
            self.net.appendImage(imgNeg)
            out = self.net.forwardAll(Constants.reconstructionMode).cpu().data
            self.net.flushImages()
            if(selectedPos is None):
                 selectedPos = out[0].unsqueeze(dim=0)
                 selectedNeg = out[1].unsqueeze(dim=0)
            else:
                 selectedPos = torch.cat((selectedPos, out[0].unsqueeze(dim=0)), 0)
                 selectedNeg = torch.cat((selectedNeg, out[1].unsqueeze(dim=0)), 0)
        #getting the difference
        sumPos = torch.sum(selectedPos, dim=0)
        sumNeg = torch.sum(selectedNeg, dim=0)
        sumPos = sumPos/k
        sumNeg = sumNeg/k
        w = sumPos-sumNeg
        #getting original image in reconstruction mode and manipulate it
        self.net.flushImages()
        self.net.appendImage(imageToManipulate)
        toManipulate = self.net.forwardAll(Constants.reconstructionMode).cpu()
        self.net.flushImages()
        alpha = 0.4/(1/toManipulate.shape[1]*torch.sum(w**2)) * factorManipulation
        return toManipulate + alpha * (sumPos-sumNeg)


    def getFaissFeatures(self, numIterations=1, filePrefix="1dummy", datasetPath=None, packageSize=10, posNeg=None, saveIt=True):
        filePrefix = filePrefix+"-packageSize_"+str(packageSize)
        print("Building Faiss tree on path " + datasetPath + " with numIterations "+str(numIterations)+" and filePrefix "+filePrefix)
        imageBatchSize = 10*2
        realFileIndex = list(range(len(posNeg)))

        #get usual image size
        self.net.flushImages()
        self.net.appendImage(Image.open(Constants.datasetRootPath + "dummyImage.jpg"))
        output = self.net.forwardAll(Constants.KNNIndexMode)
        self.net.flushImages()

        fileList = []
        vectors = None
        filesNotFound = 0

        for i in range(numIterations):
            if(i%10==0):
                print("Faiss epoch: "+str(i)+"/"+str(numIterations-1))
            if(saveIt and i%packageSize == 0 and i!=0):
                filePath = Constants.targetDataObjectsPath+"/faiss/"+filePrefix+".faiss"
                d = klepto.archives.dir_archive(filePath, cached=True, serialized=True, compression=0)
                d["fileList"+str(int(i/packageSize))] = fileList
                d["vectors"+str(int(i/packageSize))] = vectors
                d.dump()
                d.clear()
                fileList = []
                vectors = None

            #get random images
            lenPosNeg = 0
            while(self.net.imageCounter<imageBatchSize and len(posNeg) > 0):
                randPosNeg = np.random.randint(len(posNeg),size=1)
                #append image
                img, _ = posNeg[randPosNeg[0]]
                self.net.appendImage(img)
                #append index
                realIdx = realFileIndex[randPosNeg[0]]
                fileList.append(realIdx)
                lenPosNeg = lenPosNeg + 1

                realFileIndex.pop(randPosNeg[0])
                posNeg.pop(randPosNeg[0])

            if(lenPosNeg>0):
                netOutput = self.net.forwardAll(Constants.KNNIndexMode)
            #break if nothing to do anymore
            if(lenPosNeg==0):
                break

            if(lenPosNeg>0):
                netOutput2 = netOutput.cpu().data
                if(vectors is None):
                    vectors = netOutput2
                else:
                    vectors = torch.cat((vectors, netOutput2))
            self.net.flushImages()
        #export last time
        if(saveIt):
            filePath = Constants.targetDataObjectsPath+"/faiss/"+filePrefix+".faiss"
            d = klepto.archives.dir_archive(filePath, cached=True, serialized=True, compression=0)
            d["fileList"+str(int(i/50)+1)] = fileList
            d["vectors"+str(int(i/50)+1)] = vectors
            d.dump()
            d.clear()
        return fileList, vectors

    def buildFaissFromFile(self, pFileName, numPacks=10):
        d = klepto.archives.dir_archive(Constants.targetDataObjectsPath+"faiss/"+pFileName, cached=False, serialized=True, compression=0)
        print(Constants.targetDataObjectsPath+"faiss/"+pFileName)
        import os
        allIndices = []
        d.load('vectors1')
        vectors = d["vectors1"]
        #now build the faiss index
        #yep, it is dim 0
        faissP = faiss.IndexFlatL2(vectors[0].shape[0])

        numDirs = sum(True for i in os.listdir(Constants.targetDataObjectsPath+"faiss/"+pFileName))
        if numPacks>numDirs/4:
            numPacks = int(numDirs/2)

        for i in range(1, numPacks+1):
            print("Loading vectors"+str(i))
            d.load('vectors'+str(i))
            print("Loading fileList"+str(i))
            d.load('fileList'+str(i))
            print("Loading ended")

            indices = d["fileList"+str(i)]
            vectors = d["vectors"+str(i)]
            allIndices = allIndices+indices

            #get the splitted sets
            vectorsNP = vectors.numpy()
            del vectors

            #normalize to get cosine similarity
            norms = np.expand_dims(np.linalg.norm(vectorsNP, 2, 1), 1)
            normalizedOutput = vectorsNP/np.broadcast_to(norms, vectorsNP.shape)
            del vectorsNP
            del norms
            faissP.add(normalizedOutput)
            del normalizedOutput
        return allIndices, faissP


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.ToPILImage()
         ])
    trainset = LFW("lfwCropped", transform=transform, train=True)
    testset = LFW("lfwCropped", transform=transform, train=False)
    trainset_rectangle = LFW("lfwCroppedRectangle", transform=transform, train=True)
    testset_rectangle = LFW("lfwCroppedRectangle", transform=transform, train=False)
    prep = Preprocessing()

    if True:
        imageToManipulate, _ = testset_rectangle[0]
        imageToManipulate.show()

        name = "IndexMode7-noBox.faiss-packageSize_10.faiss"
        listPos, faissPos = prep.buildFaissFromFile(name, numPacks=100)
        name = "IndexMode7-box.faiss-packageSize_10.faiss"
        listNeg, faissNeg = prep.buildFaissFromFile(name, numPacks=100)
        indexPos, indexNeg = prep.getKNN(faissPos, faissNeg, imageToManipulate)

        manipulated = prep.getManipulatedVector(imageToManipulate, listPos, listNeg, indexPos, indexNeg, trainset, trainset_rectangle, factorManipulation=15.0)

        #reconstruct
        reconstructed = prep.net.reconstructImage(manipulated.to(Constants.pDevice), 200, saveIt=False, lam=0.04, LR=1.7)
        #convert image back to Height,Width,Channels
        img = np.transpose(reconstructed[-1].numpy(), (1,2,0))
        img = np.clip(img, 0, 1)
        #show the image
        plt.imshow(img)
        plt.show()

    if False:
        #no Normalize
        name = "IndexMode"+str(Constants.KNNIndexMode)+"-noBox.faiss"
        prep.getFaissFeatures(numIterations=99999999, filePrefix=name, datasetPath=Constants.datasetRootPath+"lfwCropped/", posNeg=trainset)
        name = "IndexMode"+str(Constants.KNNIndexMode)+"-box.faiss"
        prep.getFaissFeatures(numIterations=99999999, filePrefix=name, datasetPath=Constants.datasetRootPath+"lfwCroppedRectangle/", posNeg=trainset_rectangle)
