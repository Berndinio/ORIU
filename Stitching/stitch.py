from ..dataset.lfwDataset import lfwDataset as LFW
from torchvision import datasets, transforms
from ..DFI.preprocessing import Preprocessing
import dlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
import argparse
from ..dataset.MNISTDataset import MNISTDataset as MNIST

class Stitching:
    def __init__(self, pDSet):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("ORIU-project/shape_predictor_68_face_landmarks.dat")
        self.preprocessing = Preprocessing()
        self.dSet = pDSet
        pass

    def getLandmarks(self, imgToAnalyze, draw=False):
        #get a cv2 representation ... dlib needs cv2 image
        img = np.array(imgToAnalyze)
        box = [(75,75), (115, 115)]
        imgCV2 = img[:, :, ::-1].copy()

        #detect landmarks
        faces = self.detector(imgCV2, 1)
        shape = self.predictor(imgCV2, faces[0])
        if draw:
            for i in range(0, 68):
                cv2.circle(imgCV2,(shape.part(i).x, shape.part(i).y),2,(0,0,255),2)
                cv2.putText(imgCV2,str(i), (shape.part(i).x,shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            cv2.imshow("image", imgCV2)
        landmarks = np.zeros((68,2))
        for i in range(0, 68):
            landmarks[i,0] = shape.part(i).x
            landmarks[i,1] = shape.part(i).y
        return landmarks

    def start(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage()
             ])

        if(self.dSet == "LFw"):
            #get data
            trainset = LFW("lfwCropped", transform=transform, train=True)
            testset = LFW("lfwCropped", transform=transform, train=False)
            trainset_rectangle = LFW("lfwCroppedRectangle", transform=transform, train=True)
            testset_rectangle = LFW("lfwCroppedRectangle", transform=transform, train=False)

            #find nearest neighbors by FAISS Index
            name = "LFW-IndexMode7-noBox.faiss-packageSize_10.faiss"
            listPos, faissPos = self.preprocessing.buildFaissFromFile(name, numPacks=100)
            name = "LFW-IndexMode7-box.faiss-packageSize_10.faiss"
            listNeg, faissNeg = self.preprocessing.buildFaissFromFile(name, numPacks=100)
        else:
            #load the dataset properly
            trainset = MNIST("MNIST", transform=transform, train=True)
            testset = MNIST("MNIST", transform=transform, train=False)
            trainset_rectangle = MNIST("MNIST-occluded", transform=transform, train=True)
            testset_rectangle = MNIST("MNIST-occluded", transform=transform, train=False)

            #find nearest neighbors by FAISS Index
            name = "MNIST-IndexMode7-noBox.faiss-packageSize_10.faiss"
            listPos, faissPos = self.preprocessing.buildFaissFromFile(name, numPacks=100)
            name = "MNIST-IndexMode7-box.faiss-packageSize_10.faiss"
            listNeg, faissNeg = self.preprocessing.buildFaissFromFile(name, numPacks=100)

        for x in range(100):
            if(self.dSet == "MNIST"):
                #additionally make RGB image
                transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
                     transforms.ToPILImage()
                     ])

                print("Datasetindex: "+str(x))
                imageToManipulateT, _ = trainset_rectangle[x]
                imageToManipulate = transform(imageToManipulateT)
                #just the numpy version of the imageToManipulate
                targetImage = np.array(imageToManipulate)
                print(targetImage.shape)
                #find NN
                indexPos, indexNeg = self.preprocessing.getKNN(faissPos, faissNeg, imageToManipulate, k=1000)
                box = [(13,13), (19, 19)]
                distance = 9999999999999
                idx = None
                for i,neighborIdx in enumerate(indexPos):
                    distance_temp = 0
                    print(str(i)+"/"+str(len(indexPos)))
                    dSetIdx = listPos[neighborIdx]
                    img, _ = trainset[dSetIdx]
                    imageToManipulate = transform(imageToManipulate)

                    #compute smoothening/pixel loss
                    npImg = np.array(img)
                    npImg[box[0][0]:box[1][0], box[0][1]:box[1][1]] = 0.0
                    pixelLoss = np.linalg.norm(targetImage - npImg)

                    distance_temp += pixelLoss
                    if distance_temp<distance:
                        distance = distance_temp
                        idx = dSetIdx
                img, _ = trainset[dSetIdx]
                optimalImage = np.array(img)
                targetImage[box[0][0]:box[1][0], box[0][1]:box[1][1], :] = optimalImage[box[0][0]:box[1][0], box[0][1]:box[1][1], :]

                #show some things
                finalImage = Image.fromarray(targetImage, 'RGB')

                plt.clf()
                plt.figure(1)
                plt.subplot(131)
                plt.imshow(imageToManipulateT)
                plt.subplot(132)
                plt.imshow(optimalImage)
                plt.subplot(133)
                plt.imshow(finalImage)
                plt.show()


            if(self.dSet == "LFW"):
                print("Datasetindex: "+str(x))
                imageToManipulate, _ = trainset_rectangle[x]
                #just the numpy version of the imageToManipulate
                targetImage = np.array(imageToManipulate)
                #get the absolutely nearest neighbor by landmark loss
                try:
                    toManipulateLandmarks = self.getLandmarks(imageToManipulate)
                except:
                    print("Continued")
                    continue
                #find NN
                indexPos, indexNeg = self.preprocessing.getKNN(faissPos, faissNeg, imageToManipulate)
                #find absolutely NN
                box = [(75,75), (115+1, 115+1)]
                distance = 9999999999999
                idx = None
                for i,neighborIdx in enumerate(indexPos):
                    print(str(i)+"/"+str(len(indexPos)))
                    dSetIdx = listPos[neighborIdx]
                    img, _ = trainset[dSetIdx]
                    #compute landmark loss
                    try:
                        landmarks = self.getLandmarks(img)
                    except:
                        continue
                    distance_temp = np.linalg.norm(landmarks-toManipulateLandmarks)
                    #compute smoothening/pixel loss
                    npImg = np.array(img)
                    offset = 2
                    pixelLoss = np.linalg.norm(targetImage[box[0][0]:box[1][0], box[0][1]-offset] - npImg[box[0][0]:box[1][0], box[0][1]])
                    pixelLoss += np.linalg.norm(targetImage[box[0][0]:box[1][0], box[1][1]+offset] - npImg[box[0][0]:box[1][0], box[1][1]])

                    pixelLoss += np.linalg.norm(targetImage[box[0][0]-offset, box[0][1]:box[1][1]] - npImg[box[0][0], box[0][1]:box[1][1]])
                    pixelLoss += np.linalg.norm(targetImage[box[1][0]+offset, box[0][1]:box[1][1]] - npImg[box[1][0], box[0][1]:box[1][1]])

                    distance_temp += pixelLoss*3
                    if distance_temp<distance:
                        distance = distance_temp
                        idx = dSetIdx
                print(dSetIdx)
                print(distance, idx)
                img, _ = trainset[dSetIdx]
                optimalImage = np.array(img)

                #transform the optimal image
                from skimage.transform import warp, AffineTransform, SimilarityTransform
                import copy
                from skimage.measure import ransac
                landmarksOptimal = self.getLandmarks(img)
                toManipulateLandmarks
                model_robust,_ = ransac((landmarksOptimal, toManipulateLandmarks), SimilarityTransform, min_samples=40, residual_threshold=2, max_trials=1000)
                new_optimalImage = warp(copy.deepcopy(optimalImage), model_robust.inverse)
                #cut nose out and put it into target image
                targetImage[box[0][0]:box[1][0], box[0][1]:box[1][1], :] = optimalImage[box[0][0]:box[1][0], box[0][1]:box[1][1], :]
                #targetImage[box[0][0]:box[1][0], box[0][1]:box[1][1], :] = new_optimalImage[box[0][0]:box[1][0], box[0][1]:box[1][1], :]*255

                #show some things
                finalImage = Image.fromarray(targetImage, 'RGB')

                plt.clf()
                plt.figure(1)
                plt.subplot(141)
                plt.imshow(imageToManipulate)
                plt.subplot(142)
                plt.imshow(optimalImage)
                plt.subplot(143)
                plt.imshow(new_optimalImage)
                plt.subplot(144)
                plt.imshow(finalImage)
                plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='General parameters')
    parser.add_argument('--dSet', type=str, default="LFW", metavar='N',
                        help='Dataset to take for manipulation')
    args = parser.parse_args()

    stitch = Stitching(args.dSet)
    stitch.start()
