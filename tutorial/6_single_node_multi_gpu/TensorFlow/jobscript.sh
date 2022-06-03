#!/bin/env bash
#SBATCH -A SNIC2021-7-120  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=T4:2
#SBATCH -J "SNMG TensorFlow"  # Single node, multiple GPUs

# Set-up environment
module purge
ml TensorFlow/2.5.0-fosscuda-2020b

# Run DataParallel
python mirrored_strategy.py
