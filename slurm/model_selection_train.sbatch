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
module purge
module load bear-apps/2022b/live
module load Julia/1.10.0-linux-x86_64
export JULIA_NUM_THREADS=1
## The model-selection run is the script's first argument (sbatch <script> <run>), default b4_g4. It is not
## passed with --export: BlueBEAR jobs start from a clean environment, and changing what is exported
## breaks either the module loads or the PATH that SlurmClusterManager's worker processes need.
export MS_RUN="${1:-b4_g4}"
echo "Model-selection run: ${MS_RUN}"

## One worker per network: {A, B} × {classifier, conditional NPE} × 3 replicates.
## Finished networks are skipped, so resubmitting after a time-out continues where it stopped.
julia --project src/model_selection_train.jl
