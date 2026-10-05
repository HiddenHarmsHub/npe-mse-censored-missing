#!/bin/bash
#SBATCH --job-name=ms_architecture
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
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

## One worker per architecture (24 per system); the master process starts them with SlurmManager
julia --project src/architecture_selection.jl
