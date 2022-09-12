# PG-Alignments-GAN
This repository contains scripts, configuration files, and examples for a Generative Adversarial Network (GAN) of Population Genetic Alignments

## Overview
The scripts here function to train and evaluate a GAN that learns the distribution of and mimicks population genetic alignments. Within layers.py there are several generator and discriminator architectures you can use, but the best performing is a Deep-Convolutional GAN using a Wasserstein loss with gradient penalty (DCWGAN-GP). Full details of the architecture are depicted below.
![GAN Architecture](https://github.com/SchriderLab/PG-Alignments-GAN/blob/main/img/Architecture_extended.png)

## Implementation
PG-Alignments-GAN is implemented in python (3.9.7) using pytorch (1.9.1) libraries

All dependencies and libraries are best installed using Conda using the provided environment file. To do so:

```{bash}
git clone git@github.com:SchriderLab/PG-Alignments-GAN.git

cd PG-Alignments-GAN

conda env create -f PG-Alignments-GAN.yml

conda activate PGA-GAN
```

To train the GAN:

```{bash}
python3 src/train_wgan_v2.py --odir ODIR --idir IDIR --use_cuda --plot
```
Optional arguments:

```
  -h, --help            show this help message and exit
  --latent_size LATENT_SIZE
                        size of latent/noise vector
  --idir IDIR           input directory
  --odir ODIR           output directory
  --plot                plot summaries in output
  --gen GEN             set what type of generator to be used. Options: sigGen tanGen tanNorm
  --loss LOSS           whether to use gp or div to make the loss 1-Lipschitz compatible
  --gen_lr GEN_LR       generator learning rate
  --disc_lr DISC_LR     discriminator learning rate
  --num_in NUM_IN       number of input alignments
  --use_cuda            use cuda?
  --save_freq SAVE_FREQ
                        save model every save_freq epochs
  --batch_size BATCH_SIZE
                        set batch size
  --epochs EPOCHS       total number of epochs
  --critic_iter CRITIC_ITER
                        number of generator iterations per critic iteration
  --gp_lambda GP_LAMBDA
                        lambda for gradient penalty
  --use_buffer          use a buffer for fake data sampling
  --buffer_n BUFFER_N   the buffer size will be this many batches large (integer)
  --permute             permute real data along the individual axis
  --label_smooth        label smooth both real and fake data
  --label_noise LABEL_NOISE
                        upper bound of the uniform distribution used to label smooth
  --mono_switch         switch some input sites to monomorphic for training
  --normalize           normalize inputs for tanh activation
  --shuffle_inds        shuffle individuals in each input alignment
  --verbose             verbose output to log
  ```

![GAN Example images](https://github.com/SchriderLab/PG-Alignments-GAN/blob/main/img/Alignments_fig-01.png)
