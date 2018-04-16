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
import datetime


def clear_line():
    """Clear line from any characters"""
    print('\r{}'.format(' ' * 80), end='\r')


def format_hdr(gan, root_dir, training_len):
    """Print type of GAN with number of parameters"""

    num_params_D, num_params_G = gan.get_num_params()
    loss = gan.loss
    if gan.loss is 'og':
        gan_type = 'Deep convolutional GAN (DCGAN)'
        gan_loss = 'min_G max_D  E_x[log D(x)] + E_z[1 - log D(G(x))]'
    elif gan.loss is 'wasserstein':
        gan_type = 'Wasserstein GAN (WGAN)'
        gan_loss = 'min_G max_D  E_x[D(x)] - E_z[D(G(x))]'
    else:
        gan_type = 'Unknown'
        gan_loss = 'Unknown'
    title = 'Generative Adversarial Network (GAN)'.center(80)
    sep, sep_ = 80 * '-', 80 * '='
    type_str = 'Type: {}'.format(gan_type)
    loss_str = 'Loss: {}'.format(gan_loss)
    param_D_str = 'Nb of generator params: {:,}'.format(num_params_D)
    param_G_str = 'Nb of discriminator params: {:,}'.format(num_params_G)
    dataset = 'Training on CelebA dataset ({}) with {:,} faces'.format(root_dir, training_len)
    hdr = '\n'.join([sep_, title, sep, dataset, type_str, loss_str, param_D_str, param_G_str, sep_])
    print(hdr)


def time_elapsed_since(start):
    """Compute elapsed time since start"""

    end = datetime.datetime.now()
    return str(end - start)[:-7]


def progress_bar(batch_idx, report_interval, G_loss, D_loss):
    """Neat progress bar to track training"""

    bar_size = 24
    progress = (((batch_idx - 1) % report_interval) + 1) / report_interval
    fill = int(progress * bar_size)
    print('\rBatch {:>4d} [{}{}] G loss: {:>7.4f} | D loss: {:>7.4f}'.format(batch_idx, '=' * fill, ' ' * (bar_size - fill), G_loss, D_loss), end='')


def show_learning_stats(batch_idx, num_batches, g_loss, d_loss, elapsed):
    """Format stats"""

    clear_line()
    dec = str(int(np.ceil(np.log10(num_batches))))
    print('Batch {:>{dec}d} / {:d} | G loss: {:>7.4f} | D loss: {:>7.4f} | Avg time / batch: {:d} ms'.format(batch_idx, num_batches, g_loss, d_loss, int(elapsed), dec=dec))


def compute_mean_std(data_loader):
    """Compute mean and standard deviation for a given dataset"""

    means, stds = torch.zeros(3), torch.zeros(3)
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.squeeze(0)
        means += torch.Tensor([torch.mean(x[i]) for i in range(3)])
        stds += torch.Tensor([torch.std(x[i]) for i in range(3)])
        if batch_idx % 1000 == 0 and batch_idx:
            print('{:d} images processed'.format(batch_idx))

    mean = torch.div(means, len(data_loader.dataset))
    std = torch.div(stds, len(data_loader.dataset))
    print('Mean = {}\nStd = {}'.format(mean.tolist(), std.tolist()))
    return mean, std


def load_dataset(root_dir, batch_size):
    """Load data from image folder"""

    # mean, std = [0.5066, 0.4261, 0.3836], [0.2589, 0.2380, 0.2340]
    mean, std = [0.5] * 3, [0.5] * 3
    normalize = transforms.Normalize(mean=mean, std=std)
    train_data = ImageFolder(root=root_dir,
       transform=transforms.Compose([transforms.ToTensor(), normalize]))
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return data_loader


def unnormalize(img):
    """Unnormalize image"""

    # mean, std = [0.5066, 0.4261, 0.3836], [0.2589, 0.2380, 0.2340]
    mean, std = [0.5] * 3, [0.5] * 3
    return img.cpu().data * torch.Tensor(std).view(-1, 1, 1) + \
        torch.Tensor(mean).view(-1, 1, 1)


class AvgMeter(object):
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
