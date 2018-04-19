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
from torchvision.models.inception import inception_v3
from torch.utils.data import TensorDataset

import numpy as np
from scipy.stats import entropy
import pickle
import glob, os, sys
import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import utils
import subprocess as sp
import argparse
import torch.nn.functional as F
import itertools

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x,dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, (batch,_) in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def mode_score(imgs, real_imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x,dim=1).data.cpu().numpy()

    # Get predictions
    preds_gen = np.zeros((N, 1000))
    preds_real = np.zeros((N, 1000))

    for i, (batch,_) in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds_gen[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    for i, (batch,_) in enumerate(real_imgs, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds_real[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part_gen = preds_gen[k * (N // splits): (k+1) * (N // splits), :]
        part_real = preds_real[k * (N // splits): (k+1) * (N // splits), :]
        py_gen = np.mean(part_gen, axis=0)
        py_real = np.mean(part_real, axis=0)
        KL_gen_real = entropy(py_gen, py_real)
        scores = []
        for i in range(part_gen.shape[0]):
            pyx = part_gen[i, :]
            scores.append(entropy(pyx, py_gen))
        split_scores.append(np.exp(np.mean(scores) - KL_gen_real))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == "__main__":
    gan = DCGAN().cuda()
    gan.load_model("checkpoints/test/wgan/wgan-gen.pt")
    gan.eval()
    # gen_imgs = gan.generate_img(n=32*20)
    # gen_imgs = TensorDataset(gen_imgs.data, gen_imgs.data)
    # print(inception_score(gen_imgs, resize=True))
    #
    # real_imgs = utils.load_dataset("../data/celebA_all", 32)
    # real_imgs = itertools.islice(real_imgs, 20)
    # print(mode_score(gen_imgs, real_imgs, resize=True))

    # 6-> 25
    # 16-> 42
    # 57 -> 85
    # 95 -> 111
    for i in range(80,120):
        img = gan.generate_img(seed=i).squeeze()
        img = utils.unnormalize(img)
        torchvision.utils.save_image(img, "seed={}.png".format(i))
