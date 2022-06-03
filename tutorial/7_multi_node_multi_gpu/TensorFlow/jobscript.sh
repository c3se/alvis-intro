#!/bin/env bash
#SBATCH -A SNIC2021-7-120  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=T4:2
#SBATCH --ntasks-per-node=2
#SBATCH -J "MNMG TensorFlow"  # Single node, multiple GPUs

#=============================================================================
#                              TensorFlow
#=============================================================================
module purge
ml TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1

# Run with MultiWorkerMirroredStrategy
#srun python mwms.py --communicator=NCCL
srun python mwms.py --communicator=RING


#=============================================================================
#                       TensorFlow with Horovod
#=============================================================================
#module purge
#ml Horovod/0.21.1-fosscuda-2020b-TensorFlow-2.4.1

# Run with Horovod
#srun python hvd.py