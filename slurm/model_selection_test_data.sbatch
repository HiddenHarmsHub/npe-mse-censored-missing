#!/bin/bash
#SBATCH --job-name=ms_test_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_output/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module purge; module load bluebear
module load bear-apps/2022b/live
module load Julia/1.10.0-linux-x86_64
export JULIA_NUM_THREADS=1

julia --project src/model_selection_test_data.jl
