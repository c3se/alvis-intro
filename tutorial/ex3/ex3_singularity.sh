#!/bin/env bash
#SBATCH -A C3SE-STAFF -p alvis # find your project with the "projinfo" command
#SBATCH -t 0-00:10:00
#SBATCH -J pytorch_MNIST
#SBATCH --gpus-per-node=T4:1

CONTAINER=/apps/nvidia-ngc-containers/tensorflow/20.11-tf2-py3.sif

# Don't forget the --nv flag, else your containers won't see the GPUs!
singularity exec --nv $CONTAINER python ./ex3_main.py > results_ex3.out

