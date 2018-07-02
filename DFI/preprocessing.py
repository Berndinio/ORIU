import os
import faiss
import numpy as np
from .dfiNetwork import dfiNetwork
from ...constants import Constants
from PIL import Image
import os.path
import torch
import pickle
import klepto
import matplotlib.pyplot as plt


class Preprocessing:
    def __init__(self, netToLoad="VGG19"):
        self.net = dfiNetwork()

    #Neg = source set
    def getKNN(self, listPos, faissPositive, listNeg, faissNegative, imageToManipulate, k=100, save=False, fName="dummy.jpg"):
        if(save==True):
            fig=plt.figure(figsize=(12, 20))
            #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            fig.add_subplot(12, 20, 1)
            plt.imshow(imageToManipulate)
            plt.title("Source", fontsize=6)
            plt.axis('off')
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

        if(save==True):
            for i, (iPos, iNeg) in enumerate(zip(indexPos, indexNeg)):
                imgPos = Image.open(listPos[iPos])
                imgNeg = Image.open(listNeg[iNeg])
                if(save==True):
                    #positive
                    fig.add_subplot(12, 20, 21+i)
                    plt.imshow(imgPos)
                    plt.title(str(round(distPos[0,i],4)), fontsize=6)
                    plt.axis('off')
                    #negative
                    fig.add_subplot(12, 20, 141+i)
                    plt.imshow(imgNeg)
                    plt.title(str(round(distNeg[0,i],4)), fontsize=6)
                    plt.axis('off')
            plt.savefig(Constants.targetDataGraphicsPath+"neighborSearch/"+fName, dpi=500)
        return listPos, indexPos, listNeg, indexNeg

    def getManipulatedVector(self, imageToManipulate, listPos, listNeg, indexPos, indexNeg, factorManipulation=5.0):
        k = len(indexPos)
        selectedPos = None
        selectedNeg = None
        for i, (iPos, iNeg) in enumerate(zip(indexPos, indexNeg)):
            print(str(i)+"/"+str(k))
            imgPos = Image.open(listPos[iPos])
            imgNeg = Image.open(listNeg[iNeg])
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


    def getFaissFeatures(self, numIterations=1, filePrefix="1dummy", datasetPath=Constants.datasetCelebaAligned, packageSize=10, posNeg=None, saveIt=True):
        filePrefix = filePrefix+"-packageSize_"+str(packageSize)
        print("Building Faiss tree on path " + datasetPath + " with numIterations "+str(numIterations)+" and filePrefix "+filePrefix)
        imageBatchSize = 10*2

        #get usual image size
        self.net.flushImages()
        self.net.appendImage(Image.open(Constants.datasetRootPath + "dummyImage.jpg"))
        output = self.net.forwardAll(Constants.KNNIndexMode)
        self.net.flushImages()

        fileList = []
        vectors = None
        filesNotFound = 0

        if posNeg is None:
            #just catch ANY feature
            positive, negative = self.getAttributeSets("No_Beard")
            posNeg = positive + negative
            from random import shuffle
            shuffle(posNeg)

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
                fName = posNeg[randPosNeg[0]]

                if(os.path.isfile(datasetPath + fName)):
                    img = Image.open(datasetPath + fName)
                    self.net.appendImage(img)
                    fileList.append(fName)
                    lenPosNeg = lenPosNeg + 1
                else:
                    filesNotFound = filesNotFound + 1
                    print("File "+fName+" not existent. Already:" + str(filesNotFound))
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

    def buildFaissFromFile(self, pFileName, positive, negative, datasetPath, numPacks=1):
        d = klepto.archives.dir_archive(Constants.targetDataObjectsPath+"faiss/"+pFileName, cached=False, serialized=True, compression=0)
        print(Constants.targetDataObjectsPath+"faiss/"+pFileName)
        import os
        endListPos = []
        endListNeg = []

        d.load('vectors1')
        vectors = d["vectors1"]
        #now build the faiss index
        #yep, it is dim 0
        faissPositive = faiss.IndexFlatL2(vectors[0].shape[0])
        faissNegative = faiss.IndexFlatL2(vectors[0].shape[0])

        numDirs = sum(True for i in os.listdir(Constants.targetDataObjectsPath+"faiss/"+pFileName))
        if numPacks>numDirs/4:
            numPacks = int(numDirs/2)

        for i in range(1, numPacks+1):
            print("Loading vectors"+str(i))
            d.load('vectors'+str(i))
            print("Loading fileList"+str(i))
            d.load('fileList'+str(i))
            print("Loading ended")

            fileList = d["fileList"+str(i)]
            vectors = d["vectors"+str(i)]


            #get the splitted sets
            indicesPos, indicesNeg = [], []
            for idx, fil in enumerate(fileList):
                if(fil in positive):
                    indicesPos.append(idx)
                elif(fil in negative):
                    indicesNeg.append(idx)

            listPos, vectorsPos, listNeg, vectorsNeg = [fileList[i] for i in indicesPos], vectors[indicesPos].numpy(), [fileList[i] for i in indicesNeg], vectors[indicesNeg].numpy()
            del vectors

            #normalize to get cosine similarity
            normsPos = np.expand_dims(np.linalg.norm(vectorsPos, 2, 1), 1)
            normalizedPosOutput = vectorsPos/np.broadcast_to(normsPos, vectorsPos.shape)
            del vectorsPos
            del normsPos
            faissPositive.add(normalizedPosOutput)
            del normalizedPosOutput

            #normalize to get cosine similarity
            normsNeg = np.expand_dims(np.linalg.norm(vectorsNeg, 2, 1), 1)
            normalizedNegOutput = vectorsNeg/np.broadcast_to(normsNeg, vectorsNeg.shape)
            del vectorsNeg
            del normsNeg
            faissNegative.add(normalizedNegOutput)
            del normalizedNegOutput

            endListPos = endListPos + [datasetPath+i for i in listPos]
            endListNeg = endListNeg + [datasetPath+i for i in listNeg]
        return endListPos, faissPositive, endListNeg, faissNegative



    def getAttributeSets(self, attr):
        attributeFilename = Constants.celebaAttributesFilePath
        f = open(attributeFilename, 'r')
        line = f.readline()
        line = f.readline()
        attributes = line.split(" ")
        attrIdx = attributes.index(attr) + 1
        line = f.readline()
        # 1 = positive, -1 negative
        sets = [[],[],[]]
        while line:
            features = line[:-1].split(" ")
            features = list(filter(None, features))
            sets[int(features[attrIdx])].append(features[0])
            line = f.readline()
        return sets[1], sets[-1]

    def filterAttributeSet(self, toFilter, toRemove):
        newList = [it for it in toFilter if it not in toRemove]
        return newList

def main():
    print("\033[94mRunning Preprocessing Test \033[0m")
    prep = Preprocessing()
    #test dataset selection
    positive, negative = prep.getAttributeSets("No_Beard")
    if(len(positive)+len(negative) != 202599):
        raise Exception("Preprocessing", "Some data went missing while getting Attribute Sets")

    #test building faiss tree
    listPos, vectorsPos, listNeg, vectorsNeg = prep.getFaissFeatures_splitted(positive, negative)
    listPos, vectorsPos, faissPositive, listNeg, vectorsNeg, faissNegative = prep.buildFaissFromFile_splitted("faissTree-epoch_500.faiss")
    img = Image.open(Constants.datasetCelebaAligned + "023904.jpg")
    selectedPos, selectedNeg = getKNN(listNeg, faissNegative, listPos, faissPositive, copy.deepcopy(img))
    manipulated = getManipulatedVector(copy.deepcopy(img), listPos, listNeg, selectedPos, selectedNeg, factorManipulation)
    print(manipulated, manipulated.shape)
    print("\033[92mPreprocessing Test Passed \033[0m \n")

if __name__ == '__main__':
    import copy
    prep = Preprocessing(netToLoad="OpenFace")

    #just catch ANY feature
    positive, negative = prep.getAttributeSets("No_Beard")
    pposNeg = positive + negative
    np.random.seed(42)
    randoms = np.random.choice(len(pposNeg), size=len(pposNeg), replace=False)
    pposNeg = np.asarray(pposNeg)[randoms].tolist()

    Constants.KNNIndexMode = 1
    name = "OpenFace-AlignmentMode1-IndexMode"+str(Constants.KNNIndexMode)+".faiss"
    prep.getFaissFeatures(numIterations=99999999, filePrefix=name, datasetPath=Constants.datasetCelebaAlignedEyes, posNeg=copy.deepcopy(pposNeg))
