#!/bin/env bash
#SBATCH -A C3SE-STAFF # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=T4:8
#SBATCH -N 2
#SBATCH -J "MNMG PyTorch"  # multi node, multi GPU

# Set-up environment
#flat_modules
module purge
ml Horovod/0.21.1-fosscuda-2019b-PyTorch-1.7.1-Python-3.7.4

ngpus=$(nvidia-smi -L | wc -l)

# Run DistributedDataParallel with srun (MPI backend)
srun --ntasks-per-node=$ngpus python hvd_distributed_optimizer.py
