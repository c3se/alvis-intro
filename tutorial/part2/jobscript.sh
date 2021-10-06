#!/bin/env bash
#SBATCH -A C3SE-STAFF         # find your project with the "projinfo" command
#SBATCH -p alvis              # what partition to use (usually not needed)
#SBATCH -t 0-00:30:00         # how long time it will take to run
#SBATCH --gpus-per-node=T4:1  # choosing no. GPUs and their type
#SBATCH -J regression         # the jobname (not needed)
#SBATCH -o regression.out     # name of the output file

# Load modules
flat_modules  # includes a `module purge` call
ml PyTorch/1.9.0-fosscuda-2020b matplotlib/3.3.3-fosscuda-2020b JupyterLab/2.2.8-GCCcore-10.2.0

# Interactive
#jupyter lab
# or you can also use
#jupyter notebook

# Non-interactive
python regression.py
#ipython -c "%run regression.ipynb"