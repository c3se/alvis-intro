#!/usr/bin/env bash
#SBATCH -A C3SE-STAFF  # <-- add your project here
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 120
#SBATCH -J fold-2pv7

module purge

# You need to supply your own model weights, see
# https://github.com/google-deepmind/alphafold3/tree/v3.0.1?tab=readme-ov-file#obtaining-model-parameters
AF3_MODEL_DIR="/mimer/NOBACKUP/groups/c3-staff/vikren/AlphaFoldv3.0"

apptainer run /apps/containers/AlphaFold/AlphaFold-3.0.1.sif \
    --db_dir="/mimer/NOBACKUP/Datasets/AlphafoldDatasets/v3.0.1/" \
    --model_dir="${AF3_MODEL_DIR}" \
    --norun_data_pipeline \
    --run_inference \
    --output_dir="fold" \
    --json_path msa/2pv7/2pv7_data.json \
    "$@"

# If you run out of VRAM, see
# https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md#nvidia-a100-40-gb
# for guidance contact support
# https://supr.naiss.se/support/
