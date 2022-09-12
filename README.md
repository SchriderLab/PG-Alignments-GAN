# PG-Alignments-GAN
This repository contains scripts, configuration files, and examples for a Generative Adversarial Network (GAN) of Population Genetic Alignments. This paper associated with this work can be found here: PAPER

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
## Output

Below are some example input and generated alignments, evaluated at the point where the 2D Sliced Wasserstein Distance (2DSWD, see Evaluation) between input and generated alignments is minimized 
![GAN Example images](https://github.com/SchriderLab/PG-Alignments-GAN/blob/main/img/Alignments_fig-01.png)

## Evaluation

The GAN can be evaluated in a number of ways. One way we did so is to calculate the 2D Sliced Wasserstein Distance, as calculated from the site-frequency-spectrum (SFS), between the input and generated alignments. This measurement is essentially the difference between the input and generated data distributions in multidimensional space. This measurement is calculated at every save frequency (SAVE_FREQ) and an example is shown below. Here, the minimum is reached relatively soon and is stably maintained. In other examples where the GAN struggles this line may be more erratic or increase after reaching a minimum. 

<p align="center">
<img src="https://github.com/SchriderLab/PG-Alignments-GAN/blob/main/img/example_2dswd.png" alt="example 2dswd" width="500" align="center"/>
</p>

Another way to evaluate the GAN is to calculate the Adversarial Accuracy. This measurement is used to determine the level of overfitting or underfitting of the network, where an ideal value of all AA values is 0.5. Essentially this measurement looks at how how often the nearest neighbor alignment to a generated or input alignment is another generated (AAsynth) or input alignment (AAtruth) , respectively, in some multidimensional space. For a perfectly fit model, generated alignments would be next to other generated alignments 50% of the time and similarly for input alignments, resulting in an AAts score of 0.5. For more information see Yelmen et al. (2021). Below, AAts is above 0.5 indicating the model is underfitting, but it is closely tracking with the AAtruth and AAsynth values, meaning the underfitting isn't from the model focusing on some smaller part of the input alignment distribution.

<p align="center">
<img src="https://github.com/SchriderLab/PG-Alignments-GAN/blob/main/img/example_aa.png" alt="example AA" width="500"/>
</p>

Additional ways to evaluate include investigating the output of the GAN in more detail and looking at metrics relevant to population genetics, such as the SFS. Enabling plotting (--plot) will automatically generate these (and the above 2DSWD and AA plots) in your output directory. 

## References

Yelmen, Burak, et al. "Creating artificial human genomes using generative neural networks." PLoS genetics 17.2 (2021): e1009303.
