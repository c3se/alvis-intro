#!/usr/bin/env bash
#SBATCH -A C3SE-STAFF
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 5:00:00

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 HF-Datasets/2.18.0-gfbf-2023a

python train.py
