#!/bin/env bash


# This script is used for the Open Ondemand portal.
# You can use it as a reference for creating a custom ~/portal/jupyter/my_jupyter_env.sh file

#SBATCH -A NAISS2024-22-219    # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-00:01:00          # how long time it will take to run
#SBATCH --gpus-per-node=T4:1   # choosing no. GPUs and their type
#SBATCH -J modules             # the jobname (not necessary)

module purge

module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load matplotlib/3.5.2-foss-2022a
