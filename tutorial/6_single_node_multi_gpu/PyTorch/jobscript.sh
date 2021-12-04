#!/bin/env bash
#SBATCH -A SNIC2021-7-120  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 00:20:00
#SBATCH --gpus-per-node=T4:2
#SBATCH -J "SNMG PyTorch"  # Single node, multiple GPUs

# Set-up environment
flat_modules
ml PyTorch/1.9.0-fosscuda-2020b TensorFlow/2.5.0-fosscuda-2020b

# Run DataParallel
#python dp.py

# Set up for the different multiprocessing alternatives
export MASTER_ADDR="$HOSTNAME"
export MASTER_PORT="12345"
ngpus=$(nvidia-smi -L | wc -l)
export WORLD_SIZE=$ngpus

# Run DistributedDataParallel with torch.multiprocessing
python ddp.py --spawn

# Run DistributedDataParallel with run
# (elastic version of predecessor "python -m torch.distributed.launch --use-env" and
# equivalent to "torchrun" in newer versions)
#TORCHELASTIC_MAX_RESTARTS=1
#python -m torch.distributed.run \
#    --standalone \
#    --nnodes=1 \
#    --nproc_per_node=$ngpus \
#    ddp.py

# Run DistributedDataParallel with srun (MPI)
#srun --ntasks=$ngpus python ddp.py --backend=mpi
