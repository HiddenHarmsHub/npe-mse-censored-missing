#!/bin/bash
## Submit NBE/NPE training and the evaluation that depends on it. From the repository root:
##   bash slurm/submit_train_evaluate.sh
## Evaluation (intercept estimates, NPE samples for the K=5 test set, combined results) starts only if
## training ends successfully, which train_models.jl reports only when every model file exists. If training
## fails or times out, the evaluation job stays pending with reason DependencyNeverSatisfied: cancel it
## (scancel <id>) and run this script again, which trains only the missing models.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

train=$(sbatch --parsable slurm/train_models.sbatch)
evaluate=$(sbatch --parsable --dependency=afterok:${train} slurm/evaluate_models.sbatch)

echo "training:       ${train}"
echo "evaluation:     ${evaluate} (after ${train})"
