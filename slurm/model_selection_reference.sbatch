#!/bin/bash
#SBATCH --job-name=ms_reference
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_output/%x_%j.out
## Add your cluster's --partition / --account lines here if it needs them.
## Submit from the repository root (submit_model_selection.sh does this).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
## Load Julia 1.10 (the Manifest version) as your cluster requires, e.g.
# module load julia/1.10
export JULIA_NUM_THREADS=1

## 500 datasets (A: 200, B: 300), all structures enumerated; finished datasets are skipped on resubmission
julia --project src/model_selection_reference_study.jl
