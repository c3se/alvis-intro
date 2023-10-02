#!/bin/env bash
#SBATCH -A SNIC2022-22-1064
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=T4:2
#SBATCH -J "SNMG TensorFlow"  # Single node, multiple GPUs

# Set-up environment
module purge
ml TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1

# Run DataParallel
python mirrored_strategy.py
