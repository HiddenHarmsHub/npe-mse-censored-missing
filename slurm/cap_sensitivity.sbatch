#!/bin/bash
#SBATCH --job-name=cap_sensitivity
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.s.marsh@bham.ac.uk

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module purge
## Multi-threaded workers segfaulted under the Julia 1.10.0 module
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
export SLURM_EXPORT_ENV=ALL
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

## One worker per network: NBE and NPE, each with capped pairs redrawn and with a new seed
"${JULIA[@]}" --project src/cap_sensitivity_train.jl
