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
module purge
module load bear-apps/2022b/live
module load Julia/1.10.0-linux-x86_64
module load R/4.3.1-foss-2022b
export JULIA_NUM_THREADS=1
## The job was submitted with --export=MS_RUN only (submit_model_selection.sh), and srun steps inherit that
## setting, so SlurmClusterManager's workers would start without the module-loaded PATH. Pass the job's
## full environment, as set up above, to the worker steps.
export SLURM_EXPORT_ENV=ALL
echo "Model-selection run: ${MS_RUN:-b4_g4 (default; MS_RUN not set)}"

## Single-process steps for the run in MS_RUN: domain checks (with the embedding check now that classifiers
## exist), summary tables, the real-data reanalysis, the comparison across every run with results so far,
## and figures if R is available
julia --project src/model_selection_prior_predictive.jl
julia --project src/model_selection_combine.jl
julia --project src/model_selection_real_data.jl
julia --project src/model_selection_compare_runs.jl
if command -v Rscript > /dev/null; then
    Rscript src/model_selection_plots.R
else
    echo "Rscript not found; run 'Rscript src/model_selection_plots.R' locally."
fi
