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
import subprocess as sp
import utils
from dcgan import DCGAN


"""
Perform linear interpolation in latent or screen space
Good random seeds for GAN:
    Women: 442, 491, 625
    Men:   268, 296, 573
"""


def latent_lerp(gan, z0, z1, nb_frames):
    """Interpolate between two images in latent space"""

    imgs = []
    for i in range(nb_frames):
        alpha = i / nb_frames
        z = (1 - alpha) * z0 + alpha * z1
        imgs.append(gan.generate_img(z))
    return imgs


def screen_lerp(x0, x1, nb_frames):
    """Interpolate between two images in latent space"""

    imgs = []
    for i in range(nb_frames):
        alpha = i / nb_frames
        x = (1 - alpha) * x0 + alpha * x1
        imgs.append(x)
    return imgs


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description='Lerp in latent/screen space')
    parser.add_argument('-gpu', '--cuda', help='use cuda', action='store_true')
    parser.add_argument('-t', '--type', help='model type: gan | wgan)',
        action='store', choices=['gan', 'wgan'], default='gan', type=str)
    parser.add_argument('-p', '--pretrained',
        help='load pretrained model (generator only)', metavar='PATH')
    parser.add_argument('-d', '--dir', help='output directory for interpolation',
        default='./interpolated')
    parser.add_argument('-f', '--nb-frames', help='number of frames', metavar='N', default=10, type=int)
    parser.add_argument('-v', '--video', help='turn frames into video/gif',
        action='store_true')
    excl = parser.add_mutually_exclusive_group()
    excl.add_argument('-l', '--latent', metavar=('SO', 'S1'),
        help='interpolate in latent space (random seeds s0 & s1)', nargs=2, type=int)
    excl.add_argument('-s', '--screen', metavar=('SO', 'S1'),
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
    s0, s1 = args.latent if args.latent else args.screen
    space = 'latent' if args.latent else 'screen'
    print('Interpolating random seeds {:d} & {:d} in {} space...'.format(s0, s1, space))
    z0 = gan.create_latent_var(1, s0)
    z1 = gan.create_latent_var(1, s1)

    # Interpolate
    if args.latent:
        imgs = latent_lerp(gan, z0, z1, args.nb_frames)
    else:
        x0 = gan.generate_img(z0)
        x1 = gan.generate_img(z1)
        imgs = screen_lerp(x0, x1, args.nb_frames)

    # Save files
    for i, img in enumerate(imgs):
        img = utils.unnormalize(img)
        fname_in = '{}/frame{:d}.png'.format(args.dir, i)
        torchvision.utils.save_image(img, fname_in)
        # Generate frames for perfect looping
        if args.video:
            fname_out = "{}/frame{:d}.png".format(args.dir, 2*args.nb_frames - i - 1)
            torchvision.utils.save_image(img, fname_out)
    print("Interpolated {} images saved in {}".format(args.nb_frames, args.dir))

    # Make video
    if args.video:
        mk_video = './make_anim.sh {} {}'.format(args.dir, args.nb_frames)
        sp.call(mk_video.split(), stderr=sp.DEVNULL, stdout=sp.DEVNULL)
        print("Interpolation video saved in {}".format(os.path.join(args.dir, 'video')))

"""
    if args.latent_play:
        gan.load_model('dcgan-gen', cpu=True)
        torch.manual_seed(442)
        z0 = gan.gan.create_latent_var(1)
        img = gan.gan.generate_img(z0)
        fname_in = '../play/dim_og.png'
        img = utils.unnormalize(img)
        torchvision.utils.save_image(img, fname_in)
        for i in range(100):
            z1 = z0.clone()
            z = z1[0, i, :, :].data[0][0]
            z1[0, i, :, :] = -np.sign(z) * 3
            print('i={:2d}, z={:2.4f}'.format(i, z))
            img = gan.gan.generate_img(z1)
            img = utils.unnormalize(img)
            fname_in = '../play/dim{:d}.png'.format(i)
            torchvision.utils.save_image(img, fname_in)
            #torchvision.utils.save_image(img, fname_out)
        """
