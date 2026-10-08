#!/bin/bash
#SBATCH --job-name=speed_training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module purge
## Same Julia as training (juliaup 1.10.11) when installed; see train_models.sbatch
if [[ -x "${HOME}/.juliaup/bin/julia" ]]; then
    export PATH="${HOME}/.juliaup/bin:${PATH}"
    JULIA=(julia +1.10.11)
else
    module load bear-apps/2022b/live
    module load Julia/1.10.0-linux-x86_64
    JULIA=(julia)
fi
"${JULIA[@]}" --version

## One process with the same cores as a training worker in train_models.sbatch, so the times match how the
## models were actually trained; BLAS uses the same number of threads
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

## Times one full training run of the K=5 [0, 10] 256x3 NBE and NPE at the training budget in mse_functions.jl
"${JULIA[@]}" --project --threads "${SLURM_CPUS_PER_TASK}" src/speed_comparisons.jl training
