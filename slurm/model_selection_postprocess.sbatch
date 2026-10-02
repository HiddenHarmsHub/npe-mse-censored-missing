#!/bin/bash
#SBATCH --job-name=ms_postprocess
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module purge; module load bluebear
module load bear-apps/2022b/live
module load Julia/1.10.0-linux-x86_64
module load R/4.3.1-foss-2022b
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
