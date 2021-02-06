"""
Custom Dataset for Training
"""
#!/usr/bin/env python
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms
import torchvision.transforms.functional as TF
import glob
import random
import numpy as np
import os
import torch


def fetch_loaders(paths_dict, batch_size=32, shuffle=True):
    """ Function to fetch dataLoaders for the Training / Validation

    Args:
        processed_dir(str): Directory with the processed data
        batch_size(int): The size of each batch during training. Defaults to 32.

    Return:
        Returns train and val dataloaders

    """
    loaders = {}
    for split, paths in paths_dict.items():
        ds = GlacierDataset(paths["x"], paths["y"])
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loaders

class Rotate:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class GlacierDataset(Dataset):
    """Custom Dataset for Glacier Data

    Indexing the i^th element returns the underlying image and the associated
    binary y
    """
    def __init__(self, x_paths, y_paths, imsize=512):
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.imsize = imsize

    def __getitem__(self, index):
        x_path = self.x_paths[index]
        y_path = self.y_paths[index]
        z = [np.load(p) for p in [x_path, y_path]]
        z = [TF.to_tensor(z_).float() for z_ in z]

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(z[0], output_size=(self.imsize, self.imsize))
        z = [TF.crop(z_, i, j, h, w) for z_ in z]

        # Random flipping
        if random.random() > 0.5:
            z = [TF.hflip(z_) for z_ in z]

        if random.random() > 0.5:
            z = [TF.vflip(z_) for z_ in z]

        return z


    def __len__(self):
        return len(self.x_paths)
