#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFT6135: Representation Learning
Assignment 4: Generative Models

Authors:
	Samuel Laferriere <samuel.laferriere.cyr@umontreal.ca>
	Joey Litalien <joey.litalien@umontreal.ca>
"""

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np
import datetime
from utils import *


def compute_mean_std(data_loader):
    """Compute mean and standard deviation for a given dataset"""

    means, stds = torch.zeros(3), torch.zeros(3)
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.squeeze(0)
        means += torch.Tensor([torch.mean(x[i]) for i in range(3)])
        stds += torch.Tensor([torch.std(x[i]) for i in range(3)])
        if batch_idx % 1000 == 0 and batch_idx:
            print("{:d} images processed".format(batch_idx))

    mean = torch.div(means, len(data_loader.dataset))
    std = torch.div(stds, len(data_loader.dataset))
    print("Mean = {}\nStd = {}".format(mean.tolist(), std.tolist()))
    return mean, std


def load_dataset(root_dir, batch_size, normalize=True, redux=True):
    """Load data from image folder"""

    if redux: # Redux dataset (10k examples)
        mean, std = [0.5066, 0.4261, 0.3836], [0.2589, 0.2380, 0.2340]
    else: # Full dataset (~202k examples)
        mean, std = [0.5066, 0.4261, 0.3836], [0.2589, 0.2380, 0.2340]
    normalize = transforms.Normalize(mean=mean, std=std)

    if normalize:
        train_data = ImageFolder(root=root_dir,
            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    else:
        train_data = ImageFolder(root=root_dir, transform=transforms.ToTensor())

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader


if __name__ == "__main__":
    root_dir = "../data/celebA_all"
    train_loader = load_dataset(root_dir, 1)
    compute_mean_std(train_loader, normalize=False)
