#!/bin/env bash
#SBATCH -A NAISS2024-22-219
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=T4:2
#SBATCH -J "SNMG TensorFlow"  # Single node, multiple GPUs

# Set-up environment
module purge
ml TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1

# Run DataParallel
python mirrored_strategy.py
