#!/usr/bin/env bash
#SBATCH -A C3SE-STAFF
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 6:00:00

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 HF-Datasets/2.18.0-gfbf-2023a

python train.py --batch-size=64 --num-epochs=3 --max-steps-per-epoch=1000 "$@"
