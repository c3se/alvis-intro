#!/bin/env bash

#SBATCH -A SNIC2021-7-119  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=T4:2
#SBATCH -J "SNMG PyTorch"  # Single node, multiple GPUs

# Set-up environment
flat_modules
ml PyTorch/1.8.1-fosscuda-2020b

# Run DataParallel
#python dp_pytorch.py

# Run DistributedDataParallel with torch.multiprocessing
python ddp_pytorch_mp.py
