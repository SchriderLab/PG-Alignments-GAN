#!/bin/bash

# Launchs job to train Wasserstein GAN
# Example: slurm/run_wgan.sh input_data wgan_output 100 10000 5 0.1 256 0.0002 10 sigGen [indir outdir save_freq num_inputs neg_slope batch_size learn_rate gp_lambda gen_type]

#SBATCH --job-name=GAN_training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=10-00:00:00
#SBATCH --partition=volta-gpu
#SBATCH --output=GAN-%j.log
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --mail-type=end
#SBATCH --mail-user=wbooker14@gmail.com

source /nas/longleaf/home/wwbooker/miniconda3/etc/profile.d/conda.sh
unset OMP_NUM_THREADS
conda deactivate
conda activate torchLearn

IDIR=$1
ODIR=${2}
SAVE_FREQ=$3
NUM_IN=$4
CRITIC_ITER=$5
BATCH_SIZE=$6
LEARN_RATE=$7
GP_LAMBDA=$8
ODIR="${2}${NUM_IN}-nIn_${CRITIC_ITER}-cIter_${BATCH_SIZE}-bSize_-gplambda_${GP_LAMBDA}-lr_${LEARN_RATE}/"
#if ["$9" = "T"]; then
#    LEAKY_GEN="--leaky_gen"
#fi
#if ["$10" = "T"]; then
#    INPUT_NOISE="--input_noise"
#fi

python3 src/train_wgan_v2.py --odir $ODIR --idir $IDIR --save_freq $SAVE_FREQ --num_in $NUM_IN --critic_iter $CRITIC_ITER --batch_size $BATCH_SIZE --gen_lr $LEARN_RATE --disc_lr $LEARN_RATE --gp_lambda $GP_LAMBDA --use_cuda --plot --shuffle_inds
