#!/bin/env bash

#SBATCH -A NAISS2024-22-219    # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-00:01:00          # how long time it will take to run
#SBATCH --gpus-per-node=A40:1  # choosing no. GPUs and their type
#SBATCH -J ml-matlab           # the jobname (not necessary)

ml purge
ml MATLAB

# Create a matlab file tmp.m with content as below
# This is just an example, usually you would write tmp.m directly in a sperate
# file, i.e. not as part of the jobscipt.
cat <<- EOF > tmp.m
    settings;
    a = 1;
    b = 2;
    a + b
EOF

# Run above matlab script headless
matlab -batch tmp

# Remove above generated matlab script
rm tmp.m

