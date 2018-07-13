import torch

class Constants:
    savesFolder = "ORIU-project/saves/"
    cudaAvailable = torch.cuda.is_available()
    pDevice = torch.device("cuda" if cudaAvailable else "cpu")

    #Mode which will be taken for representation of the sampled image.
    #mode  0 ===> one hot encoded output
    #mode  1 ===> see Net
    #mode  2 ===> see Net
    #mode>=3 ===> softmax output
    VAERepresentationMode = 3

    useMNISTInput = True
    useEncoderInput = True

    #wheter to use random generated samples or exploit the VAE as deep
    #learning neural net instantly reconstructing the image.
    useRandom = True
    #only for LFW dataset available
    useOccludedForTraining = False

    #VGG weights to load
    weightsFile = "ORIU-project/DFI/vgg19_weights.h5"
    targetDataGraphicsPath = savesFolder + "KNNresult"
    targetDataObjectsPath = savesFolder

    KNNIndexMode = 7
    reconstructionMode = 0

    datasetRootPath = "ORIU-project/dataset/"
