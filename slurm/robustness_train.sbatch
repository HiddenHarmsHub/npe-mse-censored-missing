#!/bin/bash
#SBATCH --job-name=rb_train
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module purge
## Multi-threaded workers need a later 1.10 release (see train_models.sbatch)
if [[ -x "${HOME}/.juliaup/bin/julia" ]]; then
    export PATH="${HOME}/.juliaup/bin:${PATH}"
    JULIA=(julia +1.10.11)
else
    module load bear-apps/2022b/live
    module load Julia/1.10.0-linux-x86_64
    JULIA=(julia)
fi
export JULIA_NUM_THREADS=1
export SLURM_EXPORT_ENV=ALL
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

"${JULIA[@]}" --project src/robustness_train.jl
