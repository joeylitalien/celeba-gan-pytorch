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
import numpy as np


class DCGAN(nn.Module):
    """
    Implementation of DCGAN
    'Unsupervised Representation Learning with
    Deep Convolutional Generative Adversarial Networks'
    A. Radford, L. Metz & S. Chintala
    arXiv:1511.06434v2

    This is only a container for the generator/discriminator architecture
    and weight initialization schemes. No optimizers are attached.
    """

    def __init__(self, loss="og", latent_dim=100, batch_size=128):
        super(DCGAN, self).__init__()
        self.G = Generator()
        self.D = Discriminator()
        self.init_weights(self.G)
        self.init_weights(self.D)


    def init_weights(self, model):
        """Initialize weights and biases (according to paper)"""

        for m in model.parameters():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()


    def og_loss(self, y_real, y_fake):
        """Original loss from Goodfellow's GAN paper"""
        raise NotImplementedError


    def wasserstein_loss(self, y_real, y_fake):
        """Loss from Arjovsky's WGAN paper"""
        raise NotImplementedError


class Generator(nn.Module):
    """DCGAN Generator G(z)"""

    def __init__(self, latent_dim=100, batch_size=128):
        super(Generator, self).__init__()
        self.features = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, batch_size * 8, 4, 1, 0),
            nn.BatchNorm2d(batch_size * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(batch_size * 8, batch_size * 4, 4, 2, 1),
            nn.BatchNorm2d(batch_size * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(batch_size * 4, batch_size * 2, 4, 2, 1),
            nn.BatchNorm2d(batch_size * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(batch_size * 2, batch_size , 4, 2, 1),
            nn.BatchNorm2d(batch_size),
            nn.ReLU(),
            nn.ConvTranspose2d(batch_size, 3, 4, 2, 1),
            nn.Tanh())

    def forward(self, x):
        return self.features(x)


class Discriminator(nn.Module):
    """DCGAN Discriminator D(z)"""

    def __init__(self, batch_size=128):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, batch_size, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(batch_size, batch_size * 2, 4, 2, 1),
            nn.BatchNorm2d(batch_size * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(batch_size * 2, batch_size * 4, 4, 2, 1),
            nn.BatchNorm2d(batch_size * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(batch_size * 4, batch_size * 8, 4, 2, 1),
            nn.BatchNorm2d(batch_size * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(batch_size * 8, 1, 4, 1, 0),
            nn.Sigmoid())

    def forward(self, x):
        return self.features(x)
