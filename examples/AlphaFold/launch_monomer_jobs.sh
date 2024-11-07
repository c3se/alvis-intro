#!/bin/bash

# Author: Viktor Rehnberg (Chalmers University of Technology - C3SE)
# Questions answered through https://supr.naiss.se/support/?centre_resource=c6

set -e

# 1. Get the fasta file from which we will predict a shape
identifier="${IDENTIFIER:-8D6M}"
fasta_path=${identifier}.fasta
if [ ! -f $fasta_path ]; then
    # It is usually recommended that you should download data on alvis2 but
    # this is only one very small file that shouldn't take more than a second
    # so we will make an exception and download it directly on the compute node
    # in order to simplify this example
    wget https://www.rcsb.org/fasta/entry/${identifier} -O $fasta_path
    # you can find this structure at
    # https://www.rcsb.org/structure/$identifier
    # e.g.
    # https://www.rcsb.org/structure/6OSN
    # This is where you will find  release date of the structure
fi

# 2. Set-up environment
module purge

module load "AlphaFold/2.3.2-foss-2023a-CUDA-12.1.1"
export ALPHAFOLD_DATA_DIR=/mimer/NOBACKUP/Datasets/AlphafoldDatasets/2022_12

# 3. Set common arguments
export SBATCH_ACCOUNT=C3SE-STAFF
export SBATCH_PARTITION=alvis
alphafold_args=(\
    --fasta_paths="$fasta_path" \
    --max_template_date=2022-11-01 \
    --output_dir="$PWD" \
)

# 4. Launch multi-sequence alignment on a CPU only node
if [ ! -f $identifier/features.pkl ]; then
    msa_jobid=$(
       sbatch \
            --parsable \
            -t 360 \
            -C "NOGPU" \
            -c 4 \
            -J "MSA-$identifier" \
            --wrap 'ALPHAFOLD_HHBLITS_N_CPU=$SLURM_CPUS_ON_NODE alphafold '"${alphafold_args[*]}" \
    )
else
    echo "Features exist, skipping MSA job"
fi

# 5. Launch prediction tasks on GPU nodes
models_to_run=""
for model in {1..5}; do
    if [ ! -f "${identifier}/result_model_${model}_pred_0.pkl" ]; then
        models_to_run+=$model,
    fi
done

if [ ${#models_to_run} -gt 0 ]; then
    prediction_arrayid=$(
        sbatch \
            --parsable \
            ${msa_jobid:+--dependency=afterok:${msa_jobid}} \
            -t 60 \
            --gpus-per-node=A40:1 \
            --array=$models_to_run \
            -J "AF-$identifier" \
            --wrap "alphafold ${alphafold_args[*]} --only_model_pred="'"${SLURM_ARRAY_TASK_ID}"' \
    )
else
    echo "No models to run, skipping prediction jobs"
fi

# 6. Launch relaxation on CPU node
# can also run on GPUs by instead specifying --use_gpu_relax to alphafold
# but this one only takes a couple minutes, so no need for that
relax_jobid=$(
    sbatch \
        --parsable \
        ${prediction_arrayid:+--dependency=afterok:${prediction_arrayid}} \
        -t 60 \
        -C NOGPU \
        -J "relax-$identifier" \
        --wrap "alphafold ${alphafold_args[*]} --models_to_relax=BEST --nouse_gpu_relax" \
)

echo Succesfully launched jobs $msa_jobid $prediction_arrayid $relax_jobid
