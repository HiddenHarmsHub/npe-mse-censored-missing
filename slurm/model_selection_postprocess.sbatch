#!/bin/bash
#SBATCH --job-name=ms_postprocess
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module purge
module load bear-apps/2022b/live
module load Julia/1.10.0-linux-x86_64
module load R/4.3.1-foss-2022b
export JULIA_NUM_THREADS=1
## srun steps (SlurmClusterManager's worker processes) need the module-loaded PATH above
export SLURM_EXPORT_ENV=ALL
## The model-selection run is the script's first argument (sbatch <script> <run>), default b4_g4.
export MS_RUN="${1:-b4_g4}"
echo "Model-selection run: ${MS_RUN}"

## Single-process steps for the run in MS_RUN
julia --project src/model_selection_prior_predictive.jl
julia --project src/model_selection_combine.jl
julia --project src/model_selection_parameter_calibration.jl
julia --project src/model_selection_real_data.jl
julia --project src/model_selection_local_checks.jl
julia --project src/model_selection_compare_runs.jl
if command -v Rscript > /dev/null; then
    Rscript src/model_selection_plots.R
else
    echo "Rscript not found; run 'Rscript src/model_selection_plots.R' locally."
fi
