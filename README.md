# Generative Adversarial Networks
### IFT6135 Representation Learning --- Assignment 4

### Dependencies
Tested on Python 2.7.x / 3.6.x.
* [PyTorch](http://pytorch.org/) (0.3.0)
* [NumPy](http://www.numpy.org/) (1.13.3)
* [Pickle](https://docs.python.org/3/library/pickle.html) (0.7.4)


### Small dataset
Available [here](https://drive.google.com/open?id=1p6WtrxprsjsiedQJkKVoiqvdrP1m9BuF).

## Training

![gantraining](src/checkpoints/trained_gan/gan_anim.gif) ![gantraining](src/checkpoints/trained_wgan/wgan_anim.gif)
![gantraining](src/checkpoints/trained_lsgan/lsgan_anim.gif)


## Latent space exploration

![latentexplore-gan](explore/latent_dims/gan/gan_men_latent_play.png)
![latentexplore-gan](explore/latent_dims/gan/gan_women_latent_play.png)

![latentexplore-wgan](explore/latent_dims/wgan/wgan_men_latent_play.png)
![latentexplore-wgan](explore/latent_dims/wgan/wgan_women_latent_play.png)


## Interpolation in latent space
`./lerp.py -p checkpoints/gan/dcgan-gen.pt -ll 560 580` will linearly interpolate between two random tensors generated from seeds 560 and 580. Use `./lerp.py --h` for more details how to use it.

![latentlerpgan](explore/latent_space/gan/1_gan_latent_lerp.gif) ![latentlerpgan](explore/latent_space/gan/2_gan_latent_lerp.gif)
![latentlerpgan](explore/latent_space/gan/3_gan_latent_lerp.gif)
![latentlerpgan](explore/latent_space/gan/4_gan_latent_lerp.gif)

![latentlerpwgan](explore/latent_space/wgan/1_wgan_latent_lerp.gif) ![latentlerpwgan](explore/latent_space/wgan/2_wgan_latent_lerp.gif)
![latentlerpwgan](explore/latent_space/wgan/3_wgan_latent_lerp.gif)
![latentlerpwgan](explore/latent_space/wgan/4_wgan_latent_lerp.gif)

## Interpolation in screen space

![screenlerpgan](explore/screen_space/gan/1_gan_screen_lerp.gif) ![screenlerpgan](explore/screen_space/gan/2_gan_screen_lerp.gif)
![screenlerpgan](explore/screen_space/gan/3_gan_screen_lerp.gif)
![screenlerpgan](explore/screen_space/gan/4_gan_screen_lerp.gif)

![screenlerpwgan](explore/screen_space/wgan/1_wgan_screen_lerp.gif) ![screenlerpwgan](explore/screen_space/wgan/2_wgan_screen_lerp.gif)
![screenlerpwgan](explore/screen_space/wgan/3_wgan_screen_lerp.gif)
![screenlerpwgan](explore/screen_space/wgan/4_wgan_screen_lerp.gif)
