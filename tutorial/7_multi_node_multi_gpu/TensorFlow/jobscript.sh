#!/bin/env bash
#SBATCH -A NAISS2024-22-219
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=A100:4
#SBATCH -N 2
#SBATCH -J "MNMG TensorFlow"  # Multi node, multiple GPUs


#=============================================================================
#                              TensorFlow
#=============================================================================
module purge
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

# Run with MultiWorkerMirroredStrategy
srun python mwms.py


##=============================================================================
##                       TensorFlow with Horovod
##=============================================================================
#module purge
#ml Horovod/0.28.1-foss-2022a-CUDA-11.7.0-TensorFlow-2.11.0
#
## Run with Horovod
#srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node=$SLURM_GPUS_ON_NODE python hvd.py
