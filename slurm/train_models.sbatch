#!/bin/bash
#SBATCH --job-name=train_models
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
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
## srun does not inherit --cpus-per-task from sbatch, so pass it on explicitly; each worker then runs
## with that many Julia and BLAS threads (see train_models.jl)
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

## One worker per model: 93 NBE (point and interval estimator) and 93 NPE models at the full training
## budget in mse_functions.jl. Finished models are skipped, so resubmitting after a time-out continues
## where it stopped; a model that was mid-training when the job ended starts again.
julia --project src/train_models.jl
