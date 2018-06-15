import torch

class Constants:
    savesFolder = "ORIU-project/saves/"
    pDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Mode which will be taken for representation of the sampled image.
    VAERepresentationMode = 0

    useMNISTInput = True
    useEncoderInput = True
