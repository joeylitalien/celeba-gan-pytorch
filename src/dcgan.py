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
import torch.nn.functional as F
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

    def __init__(self, loss="og", latent_dim=100, batch_size=128,
            use_cuda=True):
        super(DCGAN, self).__init__()
        self.loss = loss
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.use_cuda = use_cuda

        self.G = Generator()
        self.D = Discriminator()
        self.init_weights(self.G)
        self.init_weights(self.D)

        self.y_real = Variable(torch.ones(batch_size))
        self.y_fake = Variable(torch.zeros(batch_size))
        if torch.cuda.is_available() and self.use_cuda:
            self.y_real = self.y_real.cuda()
            self.y_fake = self.y_fake.cuda()


    def init_weights(self, model):
        """Initialize weights and biases (according to paper)"""

        for m in model.parameters():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()


    def get_num_params(self):
        """Compute the total number of parameters in model"""

        num_params_D, num_params_G = 0, 0
        for p in self.D.parameters():
            num_params_D += p.data.view(-1).size(0)
        for p in self.G.parameters():
            num_params_G += p.data.view(-1).size(0)
        return num_params_D, num_params_G



    def create_latent_var(self, batch_size):
        """Create latent variable z"""

        z = torch.randn(batch_size, self.latent_dim)
        z = Variable(z.unsqueeze(-1).unsqueeze(-1))
        if torch.cuda.is_available() and self.use_cuda:
            z = z.cuda()
        return z


    def train_G(self, G_optimizer, batch_size):
        """Update generator parameters"""

        if self.loss is "og":
            self.G.zero_grad()

            z = self.create_latent_var(self.batch_size)
            G_out = self.G(z)
            D_out = self.D(G_out).squeeze()

            #TODO: understand why this is doing the right thing
            G_train_loss = F.binary_cross_entropy(D_out, self.y_real)
            G_train_loss.backward()
            G_optimizer.step()

            #  Update generator loss
            G_loss = G_train_loss.data[0]

        else:
            raise NotImplementedError

        return G_loss


    def train_D(self, x, D_optimizer, batch_size):
        """Update discriminator parameters"""

        if self.loss is "og":
            self.D.zero_grad()

            x = Variable(x)
            # CUDA support
            if torch.cuda.is_available() and self.use_cuda:
                x = x.cuda()

            # Pass image through discriminator
            D_out = self.D(x).squeeze()
            D_real_loss = F.binary_cross_entropy(D_out, self.y_real)

            # Pass latent variable through GAN
            z = self.create_latent_var(self.batch_size)
            G_out = self.G(z)
            D_out = self.D(G_out).squeeze()
            D_fake_loss = F.binary_cross_entropy(D_out, self.y_fake)
            D_fake_score = D_out.data.mean()

            # Total train loss
            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            D_optimizer.step()

            # Update discriminator loss
            D_loss = D_train_loss.data[0]

        else:
            raise NotImplementedError

        return D_loss


    def generate_img(self):
        """Sample random image from GAN"""

        z = self.create_latent_var(1)
        return self.G(z).squeeze()


class Generator(nn.Module):
    """DCGAN Generator G(z)"""
    #TODO: Use torch.nn.Upsample

    def __init__(self, latent_dim=100, batch_size=128):
        super(Generator, self).__init__()
        self.features = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, batch_size * 8,
                kernel_size=4, stride=1, padding=0),
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
            nn.Conv2d(batch_size, batch_size * 2,
                kernel_size=4, stride=2, padding=1),
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
