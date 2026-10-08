#!/bin/bash
#SBATCH --job-name=speed_inference
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
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

export JULIA_NUM_THREADS=1
## srun steps (SlurmClusterManager's worker processes) need the PATH set above
export SLURM_EXPORT_ENV=ALL

## Inference times on 100 K=5 test datasets: NBE, NPE sampling and a short NUTS sweep, one dataset per worker.
## Datasets with a results file are skipped, so resubmitting continues where it stopped.
"${JULIA[@]}" --project src/speed_comparisons.jl inference
