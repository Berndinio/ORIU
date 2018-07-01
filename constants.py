import torch

class Constants:
    savesFolder = "ORIU-project/saves/"
    pDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Mode which will be taken for representation of the sampled image.
    # ==0 means it will take the argmax
    VAERepresentationMode = 4

    useMNISTInput = False
    useEncoderInput = True
