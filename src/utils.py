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
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np


def progress_bar(batch_idx, report_interval, last_loss):
    """Neat progress bar to track training"""

    bar_size = 25
    progress = (((batch_idx - 1) % report_interval) + 1) / report_interval
    fill = int(progress * bar_size)
    print("\rBatch {:>5d} [{}{}] Loss: {:.4f}".format(batch_idx,
        u"\u25A0" * fill, " " * (bar_size - fill), last_loss), end="")


def show_learning_stats(epoch, nb_epochs, loss_avg, error, elapsed):
    """Format printing"""

    dec = str(int(np.ceil(np.log10(num_batches))))
    print("Batch {:>{dec}d} / {:d} | Avg loss: {:.4f} | Avg error bits: {:.2f} | Avg time/seq: {:>3d} ms".format(batch_idx, num_batches, loss_avg, error, elapsed, dec=dec))


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
    else: # Full dataset (~202k examples), to be computed
        mean, std = [0.5066, 0.4261, 0.3836], [0.2589, 0.2380, 0.2340]
    normalize = transforms.Normalize(mean=mean, std=std)

    if normalize:
        train_data = ImageFolder(root=root_dir,
            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    else:
        train_data = ImageFolder(root=root_dir, transform=transforms.ToTensor())

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return data_loader



class AverageMeter(object):
    """Compute and store the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
