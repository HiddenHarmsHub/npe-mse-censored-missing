#!/bin/bash
#SBATCH --job-name=ms_train
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_output/%x_%j.out
## Add your cluster's --partition / --account lines here if it needs them.
## Submit from the repository root (submit_model_selection.sh does this).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
## Load Julia 1.10 (the Manifest version) as your cluster requires, e.g.
# module load julia/1.10
export JULIA_NUM_THREADS=1

## One worker per network: {A, B} × {classifier, conditional NPE} × 3 replicates.
## Finished networks are skipped, so resubmitting after a time-out continues where it stopped.
julia --project src/model_selection_train.jl
