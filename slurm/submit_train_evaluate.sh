#!/bin/bash
## Submit NBE/NPE training and the evaluation that depends on it.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

train=$(sbatch --parsable slurm/train_models.sbatch)
evaluate=$(sbatch --parsable --dependency=afterok:${train} slurm/evaluate_models.sbatch)

echo "training:       ${train}"
echo "evaluation:     ${evaluate} (after ${train})"
