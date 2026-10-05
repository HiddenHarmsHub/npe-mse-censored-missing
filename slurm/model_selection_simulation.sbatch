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
echo "Model-selection run: ${MS_RUN:-b4_g4 (default; MS_RUN not set)}"

## 280 chunks of 100 datasets; finished chunks are skipped on resubmission
julia --project src/model_selection_simulation.jl
