#!/bin/env bash
#SBATCH -A SNIC2021-7-120  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=T4:8
#SBATCH --ntasks-per-node=8
#SBATCH -N 2
#SBATCH -J "MNMG PyTorch"  # multi node, multi GPU

# Set-up environment
flat_modules
ml PyTorch/1.9.0-fosscuda-2020b

# Run DistributedDataParallel with srun (MPI)
srun python ddp_mpi.py
