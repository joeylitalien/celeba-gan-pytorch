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
from torch.autograd import Variable, grad
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

    def __init__(self, gan_type='gan', latent_dim=100, batch_size=128,
            use_cuda=True):
        super(DCGAN, self).__init__()
        self.gan_type = gan_type
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


    def load_model(self, filename, use_cuda=True):
        """Load PyTorch model"""

        print('Loading generator checkpoint from: {}'.format(filename))
        if use_cuda:
            self.G.load_state_dict(torch.load(filename))
        else:
            self.G.load_state_dict(torch.load(filename, map_location='cpu'))


    def save_model(self, ckpt_path, epoch, override=True):
        """Save model"""

        if override:
            fname_gen_pt = '{}/{}-gen.pt'.format(ckpt_path, self.gan_type)
            fname_disc_pt = '{}/{}-disc.pt'.format(ckpt_path, self.gan_type)
        else:
            fname_gen_pt = '{}/{}-gen-epoch-{}.pt'.format(ckpt_path, self.gan_type, epoch + 1)
            fname_disc_pt = '{}/{}-disc-epoch-{}.pt'.format(ckpt_path, self.gan_type, epoch + 1)

        print('Saving generator checkpoint to: {}'.format(fname_gen_pt))
        torch.save(self.G.state_dict(), fname_gen_pt)
        sep = '\n' + 80 * '-'
        print('Saving discriminator checkpoint to: {}{}'.format(fname_disc_pt, sep))
        torch.save(self.D.state_dict(), fname_disc_pt)


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



    def create_latent_var(self, batch_size, seed=None):
        """Create latent variable z"""

        if seed:
            torch.manual_seed(seed)
        z = torch.randn(batch_size, self.latent_dim)
        z = Variable(z.unsqueeze(-1).unsqueeze(-1))
        if torch.cuda.is_available() and self.use_cuda:
            z = z.cuda()
        return z


    def train_G(self, G_optimizer, batch_size):
        """Update generator parameters"""

        self.G.zero_grad()

        # Through generator, then discriminator
        z = self.create_latent_var(self.batch_size)
        fake_imgs = self.G(z)
        D_out_fake = self.D(fake_imgs)

        if self.gan_type == 'gan':
            # Evaluate loss and backpropagate
            G_train_loss = F.binary_cross_entropy(D_out_fake, self.y_real)
            G_train_loss.backward()
            G_optimizer.step()

            #  Update generator loss
            G_loss = G_train_loss.data[0]

        elif self.gan_type == 'wgan':
            # Evaluate loss and backpropagate (negative since we minimize)
            G_train_loss = -D_out_fake.mean()
            G_train_loss.backward()
            G_optimizer.step()

            #  Update generator loss
            G_loss = G_train_loss.data[0]

        elif self.gan_type == 'lsgan':
            # Evaluate loss and backpropagate (negative since we minimize)
            G_train_loss = torch.mean((D_out_fake - 1) ** 2)
            G_train_loss.backward()
            G_optimizer.step()

            #  Update generator loss
            G_loss = G_train_loss.data[0]

        else:
            raise NotImplementedError

        return G_loss


    def train_D(self, x, D_optimizer, batch_size):
        """Update discriminator parameters"""

        self.D.zero_grad()

        # Through generator, then discriminator
        D_out_real = self.D(x)
        z = self.create_latent_var(self.batch_size)
        fake_imgs = self.G(z)
        D_out_fake = self.D(fake_imgs)

        if self.gan_type == 'gan':
            D_real_loss = F.binary_cross_entropy(D_out_real, self.y_real)
            D_fake_loss = F.binary_cross_entropy(D_out_fake, self.y_fake)

            # Update discriminator
            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            D_optimizer.step()

            # Update discriminator loss
            D_loss = D_train_loss.data[0]

        elif self.gan_type == 'wgan':
            # Update discriminator (negative since we minimize)
            D_train_loss = -(D_out_real.mean() - D_out_fake.mean())
            D_train_loss.backward()
            D_optimizer.step()

            # Clip weights
            self.D.clip()

            # Update discriminator loss
            D_loss = D_train_loss.data[0]

        elif self.gan_type == 'lsgan':
            # Update discriminator
            D_real_loss = torch.mean((D_out_real - 1) ** 2)
            D_fake_loss = torch.mean(D_out_fake ** 2)
            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            D_optimizer.step()

            # Update discriminator loss
            D_loss = D_train_loss.data[0]

        else:
            raise NotImplementedError

        return D_loss


    def generate_img(self, z=None, seed=None):
        """Sample random image from GAN"""

        # Nothing was provided, sample
        if z is None and seed is None:
            z = self.create_latent_var(1)
        # Seed was provided, use it to sample
        elif z is None and seed:
            z = self.create_latent_var(1, seed)
        # Either z was passed, or it was created above
        return self.G(z).squeeze()


class Generator(nn.Module):
    """DCGAN Generator G(z)"""
    #TODO: Use torch.nn.Upsample

    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        """
        self.features = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024,
                kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh())
            """
        self.features = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh())

    def forward(self, x):
        return self.features(x)


class Discriminator(nn.Module):
    """DCGAN Discriminator D(z)"""

    def __init__(self):
        super(Discriminator, self).__init__()
        """
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            #nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid())
            """
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        return self.features(x).squeeze()


    def clip(self, c=0.05):
        """Weight clipping in (-c, c)"""

        for p in self.parameters():
            p.data.clamp_(-c, c)
