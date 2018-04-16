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
import pickle
import os, sys
import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import utils
from score import inception_score


class CelebA(object):
    """Implement DCGAN for CelebA dataset"""

    def __init__(self, train_params, ckpt_params, gan_params):
        # Training parameters
        self.root_dir = train_params["root_dir"]
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
        self.save_stats_interval = ckpt_params["save_stats_interval"]

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

        # Make sure report interval divides total num of batches
        self.num_batches = self.train_len // self.batch_size
        #assert self.num_batches % self.batch_report_interval == 0, \
        #    "Batch report interval must divide total number of batches per epoch"

        # Get ready to ruuummmmmmble
        self.compile()


    def compile(self):
        """Compile model (loss function, optimizers, etc.)"""

        # Create new GAN
        self.gan = DCGAN(self.gan_loss, self.latent_dim, self.batch_size,
            self.use_cuda)

        # Set optimizers for generator and discriminator
        if self.optim is "adam":
            self.G_optimizer = optim.Adam(self.gan.G.parameters(),
                lr=self.learning_rate,
                betas=self.momentum)
            self.D_optimizer = optim.Adam(self.gan.D.parameters(),
                lr=self.learning_rate,
                betas=self.momentum)

        else:
            raise NotImplementedError

        # CUDA support
        if torch.cuda.is_available() and self.use_cuda:
            self.gan = self.gan.cuda()


    def save_model(self, epoch, override=True):
        """Save model"""

        if override:
            fname_gen_pt = "{}/dcgan-gen.pt".format(self.ckpts_path)
            fname_disc_pt = "{}/dcgan-disc.pt".format(self.ckpts_path)
        else:
            fname_gen_pt = "{}/dcgan-gen-epoch-{}.pt".format(self.ckpts_path, epoch + 1)
            fname_disc_pt = "{}/dcgan-disc-epoch-{}.pt".format(self.ckpts_path, epoch + 1)

        print("Saving generator checkpoint to: {}".format(fname_gen_pt))
        torch.save(self.gan.G.state_dict(), fname_gen_pt)
        sep = "\n" + 80 * "-"
        print("Saving discriminator checkpoint to: {}{}".format(fname_disc_pt, sep))
        torch.save(self.gan.D.state_dict(), fname_disc_pt)


    def save_stats(self, stats):
        """Save model statistics"""

        fname_pkl = "{}/dcgan-stats.pkl".format(self.stats_path)
        print("Saving model statistics to: {}".format(fname_pkl))
        with open(fname_pkl, "wb") as fp:
            pickle.dump(stats, fp)


    def load_model(self, filename):
        """Load PyTorch model"""

        ckpt_filename = self.ckpts_path + "/" + filename + ".pt"
        print("Loading generator checkpoint from: {}".format(ckpt_filename))
        self.gan.G.load_state_dict(torch.load(ckpt_filename))


    def train(self, nb_epochs, data_loader):
        """Train model on data"""

        # Initialize tracked quantities and prepare everything
        G_all_losses, D_all_losses, times = [], [], utils.AvgMeter()
        utils.format_hdr(self.gan, self.root_dir, self.train_len)
        start = datetime.datetime.now()

        # Train
        for epoch in range(nb_epochs):
            print("EPOCH {:d} / {:d}".format(epoch + 1, nb_epochs))
            G_losses, D_losses = utils.AvgMeter(), utils.AvgMeter()
            start_epoch = datetime.datetime.now()

            avg_time_per_batch = utils.AvgMeter()
            # Mini-batch SGD
            for batch_idx, (x, _) in enumerate(data_loader):
                if x.shape[0] != self.batch_size:
                    break
                batch_start = datetime.datetime.now()
                # Print progress bar
                utils.progress_bar(batch_idx, self.batch_report_interval,
                    G_losses.avg, D_losses.avg)

                x = Variable(x)
                if torch.cuda.is_available() and self.use_cuda:
                    x = x.cuda()

                # Update generator every 3x we update discriminator
                D_loss = self.gan.train_D(x, self.D_optimizer, self.batch_size)
                if batch_idx % 3 == 0:
                    G_loss = self.gan.train_G(self.G_optimizer, self.batch_size)
                D_losses.update(D_loss, self.batch_size)
                G_losses.update(G_loss, self.batch_size)

                batch_end = datetime.datetime.now()
                batch_time = int((batch_end - batch_start).total_seconds() * 1000)
                avg_time_per_batch.update(batch_time)

                # Report model statistics
                if batch_idx % self.batch_report_interval == 0 and batch_idx:
                    G_all_losses.append(G_losses.avg)
                    D_all_losses.append(D_losses.avg)
                    utils.show_learning_stats(batch_idx, self.num_batches, G_losses.avg, D_losses.avg, avg_time_per_batch.avg)
                    [k.reset() for k in [G_losses, D_losses, avg_time_per_batch]]

                # Save stats
                if batch_idx % self.save_stats_interval == 0 and batch_idx:
                    stats = dict(G_loss=G_all_losses, D_loss=D_all_losses)
                    self.save_stats(stats)

            # Save model
            utils.clear_line()
            print("Elapsed time for epoch: {}".format(utils.time_elapsed_since(start_epoch)))
            self.save_model(epoch)

        # Print elapsed time
        elapsed = utils.time_elapsed_since(start)
        print("Training done! Total elapsed time: {}\n".format(elapsed))

        return G_loss, D_loss


if __name__ == "__main__":

    train_params = {
        "root_dir": "./../data/celebA_all",
        "gen_dir": "./../generated",
        "batch_size": 128,
        "train_len": 202599,
        "learning_rate": 0.0002,
        "momentum": (0.5, 0.999),
        "optim": "adam",
        "use_cuda": True
    }

    ckpt_params = {
        "batch_report_interval": 100,
        "stats_path": "./stats/wgan",
        "ckpts_path": "./checkpoints/wgan",
        "save_stats_interval": 500
    }

    gan_params = {
        "gan_loss": "wasserstein", # "og" or "wasserstein"
        "latent_dim": 100
    }

    gan = CelebA(train_params, ckpt_params, gan_params)
    data_loader = utils.load_dataset(train_params["root_dir"],
        train_params["batch_size"])

    gan.train(10, data_loader)
    #gan.load_model("dcgan-gen")

    # torch.manual_seed(0)
    # z0 = gan.gan.create_latent_var(1)
    # torch.manual_seed(11)
    # z1 = gan.gan.create_latent_var(1)
    # imgs = gan.gan.interpolate(z0,z1)
    # for i, img in enumerate(imgs):
    #     img = utils.unnormalize(img)
    #     fname = "../interpolated/test{:.1f}.png".format(i/10)
    #     torchvision.utils.save_image(img, fname)

    #print(inception_score(gan.gan))

    # for i in range(50):
    #     img = gan.gan.generate_img()
    #     img = utils.unnormalize(img)
    #     fname = "../generated/test{:d}.png".format(i)
    #     torchvision.utils.save_image(img, fname)
