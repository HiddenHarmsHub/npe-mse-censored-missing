#!/bin/bash
#SBATCH --job-name=ms_train
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_output/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module purge; module load bluebear
module load bear-apps/2022b/live
module load Julia/1.10.0-linux-x86_64
export JULIA_NUM_THREADS=1
echo "Model-selection run: ${MS_RUN:-b4_g4 (default; MS_RUN not set)}"

## One worker per network: {A, B} × {classifier, conditional NPE} × 3 replicates.
## Finished networks are skipped, so resubmitting after a time-out continues where it stopped.
julia --project src/model_selection_train.jl
