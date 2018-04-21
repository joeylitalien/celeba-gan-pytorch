# Generative Adversarial Networks
### IFT6135 Representation Learning --- Assignment 4

### Dependencies
Tested on 3.6.x.
* [PyTorch](http://pytorch.org/) (0.3.0)
* [NumPy](http://www.numpy.org/) (1.13.3)
* [ffmpeg](https://www.ffmpeg.org)


### Small dataset
Available [here](https://drive.google.com/open?id=1p6WtrxprsjsiedQJkKVoiqvdrP1m9BuF).

## Training
GAN | WGAN | LSGAN 
:--------------------------------------------:|:------------------------------------------------:|:------------------------------------------------:
![](src/checkpoints/trained_gan/gan_anim.gif) | ![](src/checkpoints/trained_wgan/wgan_anim.gif) | ![](src/checkpoints/trained_lsgan/lsgan_anim.gif)

## Latent space exploration

GAN | WGAN
:--------------------------------------------:|:------------------------------------------------:|
![latentexplore-gan](report/imgs/gan_latent_play.png) | ![latentexplore-gan](report/imgs/wgan_latent_play.png)



## Interpolation in latent space
To perform linear interpolation in *latent space*, run

```
./lerp.py --pretrained ./checkpoints/gan/dcgan-gen.pt \
          --dir ./out \
          --latent 140 180 \
          --nb-frames 50 \
          --video \
          --cuda
``` 
This will linearly interpolate between two random *tensors* generated from seeds 140 and 180 and create a GIF/MP4 videos of the sequence. The frames and videos will be stored in `./out`.


<table align="center">
  <tr align="center">
    <td colspan=4>GAN</td>
    <td colspan=4>WGAN</td>
  </tr>
  <tr align="center">
    <td colspan=4><img src="report/imgs/gan_latent_lerp.png"></td>
    <td colspan=4><img src="report/imgs/wgan_latent_lerp.png"></td>
  </tr>  
  <tr align="center">
    <td><img src="explore/latent_space/gan/1_gan_latent_lerp.gif"></td>
    <td><img src="explore/latent_space/gan/2_gan_latent_lerp.gif"></td>    
    <td><img src="explore/latent_space/gan/3_gan_latent_lerp.gif"></td>
    <td><img src="explore/latent_space/gan/4_gan_latent_lerp.gif"></td>
    <td><img src="explore/latent_space/wgan/1_wgan_latent_lerp.gif"></td>
    <td><img src="explore/latent_space/wgan/2_wgan_latent_lerp.gif"></td>    
    <td><img src="explore/latent_space/wgan/3_wgan_latent_lerp.gif"></td>
    <td><img src="explore/latent_space/wgan/4_wgan_latent_lerp.gif"></td>
  </tr>
</table>

  
## Interpolation in screen space
To perform linear interpolation in *screen space*, run

```
./lerp.py --pretrained ./checkpoints/gan/dcgan-gen.pt \
          --dir ./out \
          --screen 140 180 \
          --nb-frames 50 \
          --video \
          --cuda
``` 
This will linearly interpolate between two random *images* generated from seeds 140 and 180 and create a GIF/MP4 videos of the sequence.

<table align="center">
  <tr align="center">
    <td colspan=4>GAN</td>
    <td colspan=4>WGAN</td>
  </tr>
  <tr align="center">
    <td colspan=4><img src="report/imgs/gan_screen_lerp.png"></td>
    <td colspan=4><img src="report/imgs/wgan_screen_lerp.png"></td>
  </tr>  
  <tr align="center">
    <td><img src="explore/screen_space/gan/1_gan_screen_lerp.gif"></td>
    <td><img src="explore/screen_space/gan/2_gan_screen_lerp.gif"></td>    
    <td><img src="explore/screen_space/gan/3_gan_screen_lerp.gif"></td>
    <td><img src="explore/screen_space/gan/4_gan_screen_lerp.gif"></td>
    <td><img src="explore/screen_space/wgan/1_wgan_screen_lerp.gif"></td>
    <td><img src="explore/screen_space/wgan/2_wgan_screen_lerp.gif"></td>    
    <td><img src="explore/screen_space/wgan/3_wgan_screen_lerp.gif"></td>
    <td><img src="explore/screen_space/wgan/4_wgan_screen_lerp.gif"></td>
  </tr>
</table>
