"""
Custom Dataset for Training
"""
#!/usr/bin/env python
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms
import glob
import numpy as np
import os
import random
import shutil
import tarfile
import torch
import torchvision.transforms.functional as TF
import urllib.request


def create_dir(p):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)


def download_data(link, p, unzip=True):
  urllib.request.urlretrieve(link, p)
  if unzip:
    tar = tarfile.open(p)
    tar.extractall(p.parent)
    tar.close()


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
