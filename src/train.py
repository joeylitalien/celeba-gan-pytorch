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
from dcgan import DCGAN, Generator, Discriminator

import numpy as np
import os, sys
import datetime
import utils


class CelebA(object):
    """Implement DCGAN for CelebA dataset"""

    def __init__(self, train_params, ckpt_params, gan_params):
        # Training parameters
        self.gen_dir = train_params["gen_dir"]
        self.batch_size = train_params["batch_size"]
        self.train_len = train_params["train_len"]
        self.learning_rate = train_params["learning_rate"]
        self.momentum = train_params["momentum"]
        self.optim = train_params["optim"]
        self.use_cuda = train_params["use_cuda"]

        # Checkpoint parameters (when, where)
        self.batch_report_interval = ckpt_params["batch_report_interval"]
        self.stats_path = ckpt_params["stats_path"]
        self.ckpts_path = ckpt_params["ckpts_path"]
        self.epoch_ckpt_interval = ckpt_params["epoch_ckpt_interval"]

        # Create directories if they don't exist
        if not os.path.isdir(self.stats_path):
            os.mkdir(self.stats_path)
        if not os.path.isdir(self.ckpts_path):
            os.mkdir(self.ckpts_path)
        if not os.path.isdir(self.gen_dir):
            os.mkdir(self.gen_dir)

        # GAN parameters
        self.gan_loss = gan_params["gan_loss"]
        self.latent_dim = gan_params["latent_dim"]

        # Get ready to ruuummmmmmble
        self.compile()


    def compile(self):
        """Compile model (loss function, optimizers, etc.)"""

        self.gan = DCGAN(self.gan_loss, self.latent_dim, self.batch_size)

        #TODO: Pass a functional to self.loss_fn that matches the loss
        # type in DCGAN
        self.loss_fn = nn.BCELoss()

        if self.optim is "adam":
            # Generator
            self.G_optimizer = optim.Adam(self.gan.G.parameters(),
                lr=self.learning_rate,
                betas=self.momentum)
            # Discriminator
            self.D_optimizer = optim.Adam(self.gan.D.parameters(),
                lr=self.learning_rate,
                betas=self.momentum)

        else:
            raise NotImplementedError

        # CUDA support
        if torch.cuda.is_available() and self.use_cuda:
            self.gan = self.gan.cuda()
            self.loss_fn = self.loss_fn.cuda()


    def save_model(self, epoch, stats):
        """Save model (both PyTorch parameters and tracked stats)"""
        raise NotImplementedError


    def load_model(self, filename):
        """Load PyTorch model"""
        raise NotImplementedError


    def train(self, nb_epochs, data_loader):
        """Train model on data"""

        # Initialize tracked quantities
        G_loss, D_loss, times = [], [], AverageMeter()

        # Train
        start = datetime.datetime.now()
        for epoch in range(nb_epochs):
            print("Epoch {:d} | {:d}".format(epoch + 1, nb_epochs))
            G_losses, D_losses = AverageMeter(), AverageMeter()

            # Mini-batch SGD
            for batch_idx, (x, _) in enumerate(data_loader):
                #TODO: Implement generator/discriminator training
                continue

        # Print elapsed time
        end = datetime.datetime.now()
        elapsed = str(end - start)[:-7]
        print("Training done! Total elapsed time: {}\n".format(elapsed))

        return G_loss, D_loss


if __name__ == "__main__":

    train_params = {
        "root_dir": "./../data/celebA_all",
        "gen_dir": "./../generated",
        "batch_size": 128,
        "train_len": 10000,
        "learning_rate": 0.0002,
        "momentum": (0.5, 0.999),
        "optim": "adam",
        "use_cuda": True
    }

    ckpt_params = {
        "batch_report_interval": 200,
        "stats_path": "./stats",
        "ckpts_path": "./checkpoints",
        "epoch_ckpt_interval": 1000,
    }

    gan_params = {
        "gan_loss": "og", # "wasserstein" also available
        "latent_dim": 100
    }

    gan = CelebA(train_params, ckpt_params, gan_params)
    data_loader = utils.load_dataset(train_params["root_dir"],
        train_params["batch_size"])
