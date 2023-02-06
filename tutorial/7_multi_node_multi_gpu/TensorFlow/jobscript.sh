#!/bin/env bash
#SBATCH -A SNIC2022-22-1064  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=A100:4
#SBATCH -N 2
#SBATCH -J "MNMG TensorFlow"  # Multi node, multiple GPUs

#=============================================================================
#                              TensorFlow
#=============================================================================
module purge
ml TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1

# Run with MultiWorkerMirroredStrategy
srun -N $SLURM_NNODES --ntasks-per-node=$SLURM_GPUS_ON_NODE python mwms.py --communicator=NCCL
#srun -N $SLURM_NNODES --ntasks-per-node=$SLURM_GPUS_ON_NODE python mwms.py --communicator=RING


#=============================================================================
#                       TensorFlow with Horovod
#=============================================================================
#module purge
#ml Horovod/0.21.1-fosscuda-2020b-TensorFlow-2.4.1
#
## Run with Horovod
#srun -N $SLURM_NNODES --ntasks-per-node=$SLURM_GPUS_ON_NODE python hvd.py
