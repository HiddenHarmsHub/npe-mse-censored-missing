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
## Multi-threaded workers segfaulted under the Julia 1.10.0 module, so use a later 1.10 release from juliaup
## when it is installed (juliaup add 1.10.11). The module is not loaded alongside it, as its library paths
## point at 1.10.0. Workers are started by srun with the same Julia binary as this process.
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
## srun steps (SlurmClusterManager's worker processes) need the PATH set above; on BlueBEAR they
## only inherit it with SLURM_EXPORT_ENV=ALL (previously set by the bluebear module, which no longer loads)
export SLURM_EXPORT_ENV=ALL
## srun does not inherit --cpus-per-task from sbatch, so pass it on explicitly; each worker then runs
## with that many Julia and BLAS threads (see train_models.jl)
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

## One worker per model: 93 NBE (point and interval estimator) and 93 NPE models at the full training
## budget in mse_functions.jl. Finished models are skipped, so resubmitting after a time-out continues
## where it stopped; a model that was mid-training when the job ended starts again.
"${JULIA[@]}" --project src/train_models.jl
