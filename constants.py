import torch

class Constants:
    savesFolder = "ORIU-project/saves/"
    cudaAvailable = torch.cuda.is_available()
    pDevice = torch.device("cuda" if cudaAvailable else "cpu")

    #Mode which will be taken for representation of the sampled image.
    # ==0 means it will take the argmax
    VAERepresentationMode = 4

    useMNISTInput = False
    useEncoderInput = True



    #VGG weights to load
    weightsFile = "ORIU-project/DFI/vgg19_weights.h5"
    targetDataGraphicsPath = savesFolder + "KNNresult"
    targetDataObjectsPath = savesFolder

    KNNIndexMode = 7
    reconstructionMode = 0

    datasetRootPath = "ORIU-project/dataset/"
