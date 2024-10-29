#!/bin/env bash


# This script is used for the Open Ondemand portal.
# You can use it as a reference for creating a custom ~/portal/jupyter/my_jupyter_env.sh file

#SBATCH -A NAISS2024-22-219    # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-00:01:00          # how long time it will take to run
#SBATCH --gpus-per-node=T4:1   # choosing no. GPUs and their type
#SBATCH -J modules             # the jobname (not necessary)

module purge

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a

# You need to have a version of jupyter (e.g. JupyterLab or IPython  from the module system)
module load JupyterLab/4.0.5-GCCcore-12.3.0

# You can launch jupyter notebook or lab, but you must specify the config file as below: 
jupyter lab --config="${CONFIG_FILE}"
