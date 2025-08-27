#!/usr/bin/env bash
#SBATCH -A C3SE-STAFF  # <-- add your project here
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 120
#SBATCH -J fold-8D6M

module purge
module load AlphaFold/2.3.2-foss-2023a-CUDA-12.1.1

identifier="${IDENTIFIER:-8D6M}"
fasta_path="${identifier}.fasta"
output_dir="${SLURM_SUBMIT_DIR}"
export ALPHAFOLD_DATA_DIR="/mimer/NOBACKUP/Datasets/AlphafoldDatasets/2022_12"

if [ ! -f "${output_dir}/features.pkl" ]; then
    echo Could not find "features.pkl", run MSA on CPU first. >> /dev/stderr
    exit 1
fi

# Preciction
alphafold
    --fasta_paths="${fasta_path}" \
    --max_template_date=2022-11-01 \
    --output_dir="${output_dir}" \
    "$@"

# if you want to run the prediction with a job-array in parallel you could do:
# sbatch --array=1-5 ...
# alphafold ... --only_model_pred="{SLURM_ARRAY_TASK_ID}"

# Relaxation
alphafold
    --fasta_paths="${fasta_path}" \
    --max_template_date=2022-11-01 \
    --output_dir="${output_dir}" \
    --models_to_relax=BEST \
    "$@"

# to run relaxation in a CPU-only job use --nouse_gpu_relax
