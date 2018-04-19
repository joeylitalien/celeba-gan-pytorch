# Make directories
mkdir -p checkpoints/trained_gan
mkdir -p checkpoints/trained_lsgan
mkdir -p checkpoints/trained_wgan

# Deep convolutional GAN
./train.py --type gan \
           --nb-epochs 50 \
           --ckpt ../checkpoints/trained_gan \
           --cuda

# Least squares GAN
./train.py --type lsgan \
           --nb-epochs 50 \
           --ckpt ../checkpoints/trained_lsgan \
           --cuda

# Wasserstein GAN
./train.py --type wgan \
           --nb-epochs 50 \
           --learning-rate 0.00005 \
           --optimizer rmsprop \
           --critic 5 \
           --ckpt ../checkpoints/trained_wgan \
           --cuda
