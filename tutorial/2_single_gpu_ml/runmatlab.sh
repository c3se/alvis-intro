#!/bin/env bash

#SBATCH -A SNIC2021-7-120      # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-00:01:00          # how long time it will take to run
#SBATCH --gpus-per-node=A40:1  # choosing no. GPUs and their type
#SBATCH -J ml-matlab           # the jobname (not necessary)

ml purge
ml MATLAB

cat <<- EOF > tmp.m
	settings;
	a = 1;
	b = 2;
	a + b
EOF

matlab -batch tmp

rm tmp.m

