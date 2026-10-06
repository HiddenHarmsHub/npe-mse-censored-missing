#!/bin/bash
#SBATCH --job-name=ms_test_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module purge
module load bear-apps/2022b/live
module load Julia/1.10.0-linux-x86_64
export JULIA_NUM_THREADS=1
## srun steps (SlurmClusterManager's worker processes) need the module-loaded PATH above; on BlueBEAR they
## only inherit it with SLURM_EXPORT_ENV=ALL (previously set by the bluebear module, which no longer loads)
export SLURM_EXPORT_ENV=ALL
## The model-selection run is the script's first argument (sbatch <script> <run>), default b4_g4. It is not
## passed with --export, which would carry the login session's module state into the job.
export MS_RUN="${1:-b4_g4}"
echo "Model-selection run: ${MS_RUN}"

julia --project src/model_selection_test_data.jl
