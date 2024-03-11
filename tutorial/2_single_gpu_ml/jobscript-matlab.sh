#!/bin/env bash

#SBATCH -A NAISS2024-22-219    # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-00:15:00          # how long time it will take to run
#SBATCH --gpus-per-node=A40:1  # choosing no. GPUs and their type
#SBATCH -J ml-matlab           # the jobname (not necessary)

ml purge
ml MATLAB

echo "Running MATLAB from $HOSTNAME"
matlab -batch regression
