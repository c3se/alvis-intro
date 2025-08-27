#!/usr/bin/env bash
#SBATCH -A C3SE-STAFF  # <-- add your project here
#SBATCH -C NOGPU -c 4
#SBATCH -t 360
#SBATCH -J MSA-2pv7

module purge

apptainer run /apps/containers/AlphaFold/AlphaFold-3.0.1.sif \
    --db_dir=/mimer/NOBACKUP/Datasets/AlphafoldDatasets/v3.0.1/ \
    --run_data_pipeline \
    --norun_inference \
    --output_dir "msa" \
    --json_path 2pv7_input.json \
    "$@"
