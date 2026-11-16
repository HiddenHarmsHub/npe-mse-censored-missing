#!/bin/bash
#SBATCH --job-name=real_data_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk

## Fixed-structure real-data analysis
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module purge
module load bear-apps/2022b/live
module load Julia/1.10.0-linux-x86_64
export JULIA_NUM_THREADS=1

julia --project src/real_data_analysis.jl
