#!/bin/bash
#SBATCH --job-name=rb_postprocess
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module purge
module load bear-apps/2022b/live
module load Julia/1.10.0-linux-x86_64
export JULIA_NUM_THREADS=1
export SLURM_EXPORT_ENV=ALL
module load R/4.3.1-foss-2022b

julia --project src/robustness_combine.jl
julia --project src/robustness_real_data.jl
if command -v Rscript > /dev/null; then
    Rscript src/robustness_plots.R
else
    echo "Rscript not found; run 'Rscript src/robustness_plots.R' locally."
fi
