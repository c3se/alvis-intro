#!/usr/bin/env bash

# Obviously change project to your project
#SBATCH -A C3SE-STAFF

# This job usually only takes around 1 hour, however if a job runs out of
# memory or if there is a huge load on the filesystem it might take much
# longer. Remember to check your jobs with `job_stats.py <JOBID>`.
#SBATCH -t 0-06:00:00

# The A100 nodes have the fastest connection to Mimer and from testing are
# usually slightly faster for most tests. However, the difference is not very
# large. Note that Alphafold only can use 1 GPU.
#SBATCH --gpus-per-node=A100:1

#SBATCH -J AlphaFold


# This example is derived from https://www.youtube.com/watch?v=aXxa5G8Ir70
# by Pedro Ojedo May at HPC2N


# 1. Get the fasta file from which we will predict a shape
identifier=8D6M
fasta_path=${identifier}.fasta
if [ ! -f $fasta_path ]; then
    # It is usually recommended that you should download data on alvis2 but
    # this is only one very small file that shouldn't take more than a second
    # so we will make an exception and download it directly on the compute node
    wget https://www.rcsb.org/fasta/entry/${identifier} -O $fasta_path
    # you can find this structure at
    # https://www.rcsb.org/structure/$identifier
    # e.g.
    # https://www.rcsb.org/structure/6OSN
    # This is where you will find  release date of the structure
fi

# 2. Set-up software environment
ml purge
ml AlphaFold/2.2.2-foss-2021a-CUDA-11.3.1

# 3. Optionally set #cores HHblits will be using. But usually has little impact
# on performance
#export ALPHAFOLD_HHBLITS_N_CPU=$SLURM_CPUS_ON_NODE 

# 4. Set path to where dataset is located
export ALPHAFOLD_DATA_DIR=/mimer/NOBACKUP/Datasets/AlphafoldDatasets/2022_03

# 5. Run simulation
# --fasta_paths        path to *.fasta file to predict structure of
# --max_template_date  should be before release date of structure to avoid that
#                      it is part of look-up dataset
# --output_dir         base directory for predicted structure files
# see alphafold --help to see a complete list of flags
alphafold \
 --fasta_paths=$fasta_path \
 --max_template_date=2022-11-01 \
 --output_dir=$SLURM_SUBMIT_DIR

