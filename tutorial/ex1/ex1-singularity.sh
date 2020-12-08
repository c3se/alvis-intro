#!/bin/sh

# This example shows how to prepare a job script and use the container images to
# run a training example with TensorFlow

#SBATCH -A C3SE-STAFF
#SBATCH -p alvis
#SBATCH -t 00-00:05:00

#SBATCH -n 1
#SBATCH --gpus-per-node=V100:1

singularity exec --nv /apps/hpc-ai-containers/TensorFlow/TensorFlow_v2.1.0-tf2-py3-NGC-R20.03.sif python cnn_with_own_data_ex1.py > res_ex1_sing.out
