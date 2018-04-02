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
import datetime
import utils


class CelebA(object):
    """Implement DCGAN for CelebA dataset"""

    def __init__(self, train_params, ckpt_params, gan_params):
        # Training parameters
        self.batch_size = train_params["batch_size"]
        self.train_len = train_params["train_len"]
        self.learning_rate = train_params["learning_rate"]
        self.momentum = train_params["momentum"]
        self.optim = train_params["optim"]
        self.use_cuda = train_params["use_cuda"]

        # Checkpoint parameters (when, where)
        self.batch_report_interval = ckpt_params["batch_report_interval"]
        self.stats_path = ckpt_params["stats_path"]
        self.ckpt_path = ckpt_params["ckpt_path"]
        self.epoch_ckpt_interval = ckpt_params["epoch_ckpt_interval"]

        # GAN parameters
        self.latent_dim = gan_params["latent_dim"]

        # Get ready to ruuummmmmmble
        self.compile()


    def compile(self):
        """Compile model (loss function, optimizers, etc.)"""

        self.gan = DCGAN(self.latent_dim, self.batch_size)
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


    def train(self, nb_epochs, data_loader):
        """Train model on data"""

        # Set learning phase
        self.model.train(True)

        # Initialize tracked quantities
        G_loss, D_loss, times = [], [], AverageMeter()

        # Train
        start = datetime.datetime.now()
        for epoch in range(nb_epochs):
            print("Epoch {:d} | {:d}".format(epoch + 1, nb_epochs))
            losses = AverageMeter()

            # Mini-batch SGD
            for batch_idx, (x, _) in enumerate(data_loader):

                # Forward pass
                x, y = Variable(x), Variable(y)

                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                # Predict
                y_pred = self.model(x)

                # Compute loss
                loss = self.loss_fn(y_pred, y)
                losses.update(loss.data[0], x.size(0))

                # Zero gradients, perform a backward pass, and update the weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Print elapsed time
        end = datetime.datetime.now()
        elapsed = str(end - start)[:-7]
        print("Training done! Total elapsed time: {}\n".format(elapsed))

        return train_loss, train_acc, valid_acc, test_acc


if __name__ == "__main__":

    train_params = {
        "root_dir": "../data/celebA_all",
        "batch_size": 128,
        "train_len": 10000,
        "learning_rate": 0.0002,
        "momentum": (0.5, 0.999),
        "optim": "adam",
        "use_cuda": True
    }

    ckpt_params = {
        "batch_report_interval": 200,
        "stats_path": "stats",
        "ckpt_path": "checkpoints",
        "epoch_ckpt_interval": 1000,
    }

    gan_params = {
        "latent_dim": 100
    }

    gan = CelebA(train_params, ckpt_params, gan_params)
    data_loader = utils.load_dataset(train_params["root_dir"],
        train_params["batch_size"])
