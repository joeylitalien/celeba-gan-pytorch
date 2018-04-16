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
import torchvision.utils
import numpy as np
import argparse
import os
import utils
from dcgan import DCGAN


"""
Perform linear interpolation in latent or screen space
Good random seeds for GAN:
    Women: 442, 491, 625
    Men:   268, 296, 573
"""

if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description='Lerp in latent/screen space')
    parser.add_argument('-gpu', '--cuda', help='use cuda', action='store_true')
    parser.add_argument('-t', '--type', help='model type (gan or wgan)')
    parser.add_argument('-p', '--pretrained',
        help='load pretrained model (generator only)', metavar='PATH')
    parser.add_argument('-d', '--dir', help='output directory for interpolation',
        default='./interpolated')
    parser.add_argument('-f', '--nb-frames', help='number of frames', metavar='N', default=10, type=int)
    excl = parser.add_mutually_exclusive_group()
    excl.add_argument('-ll', '--latent-lerp', metavar=('SO', 'S1'),
        help='interpolate in latent space (random seeds s0 & s1)', nargs=2, type=int)
    excl.add_argument('-sl', '--screen-lerp', metavar=('SO', 'S1'),
        help='interpolate in screen space (random seeds s0 & s1)', nargs=2, type=int)
    args = parser.parse_args()

    # Compile GAN and load model (either on CPU or GPU)
    gan = DCGAN(gan_type=args.type, use_cuda=args.cuda)
    if torch.cuda.is_available() and args.cuda:
        gan = gan.cuda()
    gan.load_model(filename=args.pretrained, use_cuda=args.cuda)

    # Make directory if it doesn't exist yet
    if not os.path.isdir(args.dir):
        os.mkdir(args.dir)

    # Create random tensors from seeds
    s0, s1 = args.latent_lerp if args.latent_lerp else args.screen_lerp
    space = 'latent' if args.latent_lerp else 'screen'
    print('Interpolating random seeds {:d} & {:d} in {} space...'.format(s0, s1, space))
    torch.manual_seed(s0)
    z0 = gan.create_latent_var(1)
    torch.manual_seed(s1)
    z1 = gan.create_latent_var(1)

    # Interpolate
    if args.latent_lerp:
        imgs = gan.latent_lerp(z0, z1, args.nb_frames)
    else:
        x0 = gan.generate_img(z0)
        x1 = gan.generate_img(z1)
        imgs = gan.screen_lerp(x0, x1, args.nb_frames)

    # Save files
    for i, img in enumerate(imgs):
        img = utils.unnormalize(img)
        fname_in = '{}/frame{:d}.png'.format(args.dir, i)
        torchvision.utils.save_image(img, fname_in)
    print("Interpolated {} images saved in {}".format(args.nb_frames, args.dir))
