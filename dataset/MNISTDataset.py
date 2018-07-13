from torch.utils.data import Dataset
from os import walk
import os
import io
from PIL import Image
import numpy as np

class MNISTDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, directory, train=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert (not train is None)
        self.files = []
        self.root_dir = "/home/berndinio/Schreibtisch/ORIU-project/dataset/"+directory+"/"

        for (dirpath, dirnames, filenames) in walk(self.root_dir):
            for fileP in filenames:
                fileP = self.root_dir+fileP
                self.files.append(fileP)
            break
        self.root_dir = "/home/berndinio/Schreibtisch/ORIU-project/dataset/"+directory+"/"
        self.transform = transform

        if(train):
            self.files = self.files[:-100]
        else:
            self.files = self.files[-100:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.files[idx])
        label = int(img_name[-5])
        image = Image.open(img_name)
        if self.transform:
            sample = self.transform(image)
        return sample, 1

    def pop(self, idx):
        self.files.pop(idx)
