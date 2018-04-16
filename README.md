# IFT6135 Representation Learning
## Assignment 4: Generative Models

### Dependencies
Tested on Python 2.7.x / 3.6.x.
* [PyTorch](http://pytorch.org/) (0.3.0)
* [NumPy](http://www.numpy.org/) (1.13.3)
* [Pickle](https://docs.python.org/3/library/pickle.html) (0.7.4)


### Small dataset
Available [here](https://drive.google.com/open?id=1p6WtrxprsjsiedQJkKVoiqvdrP1m9BuF).

### Linear interpolation in latent/screen space
`./lerp.py -p checkpoints/gan/dcgan-gen.pt -ll 560 580` will linearly interpolate between two random tensors generated from seeds 560 and 580. Use `./lerp.py --h` for more details how to use it.
