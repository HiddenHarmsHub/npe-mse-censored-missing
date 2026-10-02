#!/bin/bash
#SBATCH --job-name=ms_postprocess
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output/%x_%j.out
## Add your cluster's --partition / --account lines here if it needs them.
## Submit from the repository root (submit_model_selection.sh does this).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
## Load Julia 1.10 (the Manifest version) as your cluster requires, e.g.
# module load julia/1.10
export JULIA_NUM_THREADS=1

## Single-process steps: domain checks (with the embedding check now that classifiers exist),
## summary tables, the real-data reanalysis, and figures if R is available
julia --project src/model_selection_prior_predictive.jl
julia --project src/model_selection_combine.jl
julia --project src/model_selection_real_data.jl
if command -v Rscript > /dev/null; then
    Rscript src/model_selection_plots.R
else
    echo "Rscript not found; run 'Rscript src/model_selection_plots.R' locally."
fi
