#!/usr/bin/env bash
#SBATCH -A C3SE-STAFF  # <-- add your project here
#SBATCH -C NOGPU -c 4
#SBATCH -t 360
#SBATCH -J MSA-8D6M

# Get the fasta file from which we will predict a shape
identifier="${IDENTIFIER:-8D6M}"
fasta_path="${identifier}.fasta"
if [ ! -f "$fasta_path" ]; then
    # It is **not** recommended to download in the job, this is an exception
    # to make the example easier to follow. Remember that alvis2 is the
    dedicated data transfer node.
    wget "https://www.rcsb.org/fasta/entry/${identifier}" -O "$fasta_path"
    # you can find this structure at
    # https://www.rcsb.org/structure/$identifier
    # e.g.
    # https://www.rcsb.org/structure/6OSN
    # This is where you will find  release date of the structure
fi

module purge
module load AlphaFold/2.3.2-foss-2023a-CUDA-12.1.1

export ALPHAFOLD_DATA_DIR="/mimer/NOBACKUP/Datasets/AlphafoldDatasets/2022_12"
export ALPHAFOLD_HHBLITS_N_CPU="${SLURM_CPUS_ON_NODE}"
output_dir="${SLURM_SUBMIT_DIR}"

# Will create a features.pkl file from the MSA and then stop
# https://www.c3se.chalmers.se/documentation/software/machine_learning/alphafold/#patch
alphafold \
    --fasta_paths="${fasta_path}" \
    --max_template_date=2022-11-01 \
    --output_dir="${output_dir}" \
    "$@"
