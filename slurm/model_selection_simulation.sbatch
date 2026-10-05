#!/bin/bash
#SBATCH --job-name=ms_simulation
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_output/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module purge
module load bear-apps/2022b/live
module load Julia/1.10.0-linux-x86_64
export JULIA_NUM_THREADS=1
## The job was submitted with --export=MS_RUN only (submit_model_selection.sh), and srun steps inherit that
## setting, so SlurmClusterManager's workers would start without the module-loaded PATH. Pass the job's
## full environment, as set up above, to the worker steps.
export SLURM_EXPORT_ENV=ALL
echo "Model-selection run: ${MS_RUN:-b4_g4 (default; MS_RUN not set)}"

## 280 chunks of 100 datasets; finished chunks are skipped on resubmission
julia --project src/model_selection_simulation.jl
