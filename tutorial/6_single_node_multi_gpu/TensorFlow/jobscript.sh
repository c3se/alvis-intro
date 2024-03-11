#!/bin/env bash
#SBATCH -A NAISS2024-22-219
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=T4:2
#SBATCH -J "SNMG TensorFlow"  # Single node, multiple GPUs

# Set-up environment
module purge
ml TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

# Run DataParallel
python mirrored_strategy.py
