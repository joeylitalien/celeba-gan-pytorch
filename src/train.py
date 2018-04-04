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
import torchvision.utils

import numpy as np
import os, sys
import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
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

        #TODO: Pass a functional to self.criterion that matches the loss
        # type in DCGAN
        self.criterion = nn.BCELoss()

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
            self.criterion = self.criterion.cuda()


    def save_model(self, epoch):
        """Save model (both PyTorch parameters and tracked stats)"""

        filename_pt = "{}/gan-epoch-{}".format(self.ckpts_path, epoch + 1)
        filename_pt += ".pt"
        torch.save(self.gan.G.state_dict(), filename_pt)
        print("Saving generator model checkpoint to: {}".format(filename_pt))


    def load_model(self, filename):
        """Load PyTorch model"""

        ckpt_filename = self.ckpts_path + "/" + filename + ".pt"
        print("Loading generator model checkpoint from: {}".format(ckpt_filename))
        self.gan.G.load_state_dict(torch.load(ckpt_filename))


    def train(self, nb_epochs, data_loader):
        """Train model on data"""

        # Initialize tracked quantities
        G_loss, D_loss, times = [], [], utils.AvgMeter()
        num_batches = self.train_len // self.batch_size

        # Train
        #start = datetime.datetime.now()
        for epoch in range(nb_epochs):
            print("[Epoch {:d} / {:d}]".format(epoch + 1, nb_epochs))
            G_losses, D_losses = utils.AvgMeter(), utils.AvgMeter()

            # Mini-batch SGD
            for batch_idx, (x, _) in enumerate(data_loader):

                # Print progress bar
                utils.progress_bar(batch_idx, self.batch_report_interval,
                    G_losses.avg, D_losses.avg)
                start = datetime.datetime.now()

                # Update generator every 3x we update discriminator
                D_loss = self.gan.train_D(x, self.D_optimizer, self.batch_size)
                if batch_idx % 3 == 0:
                    G_loss = self.gan.train_G(self.G_optimizer, self.batch_size)
                D_losses.update(D_loss, self.batch_size)
                G_losses.update(G_loss, self.batch_size)

                # Pretty print
                if batch_idx % self.batch_report_interval == 0 and batch_idx:
                    # Save losses and accuracies
                    #G_loss.append(G_losses.avg)
                    #D_loss.append(D_losses.avg)
                    print("\r{}".format(" " * 80), end="\r")
                    end = datetime.datetime.now()
                    elapsed = int((end - start).total_seconds() * 1000)
                    utils.show_learning_stats(batch_idx, num_batches, G_losses.avg, D_losses.avg, elapsed)
                    G_losses.reset()
                    D_losses.reset()

            self.save_model(epoch)

        # Print elapsed time
        #end = datetime.datetime.now()
        #elapsed = str(end - start)[:-7]
        #print("Training done! Total elapsed time: {}\n".format(elapsed))

        return G_loss, D_loss


if __name__ == "__main__":

    train_params = {
        "root_dir": "./../data/celebA_redux",
        "gen_dir": "./../generated",
        "batch_size": 128,
        "train_len": 12800,
        "learning_rate": 0.0002,
        "momentum": (0.5, 0.999),
        "optim": "adam",
        "use_cuda": True
    }

    ckpt_params = {
        "batch_report_interval": 5,
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

    #print(gan.gan.get_num_params())
    #gan.train(5, data_loader)
    gan.load_model("gan-epoch-100")
    img = gan.gan.generate_img()
    img = utils.unnormalize(img)
    torchvision.utils.save_image(img, "./../generated/test.png")
    #plt.imsave("./../generated/test.png", np.array(img))
