#!/bin/bash
## Submit the robustness study with dependencies, from the repository root: bash slurm/submit_robustness.sh
## Every stage skips finished work, so resubmitting only fills in what is missing.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

test_data=$(sbatch --parsable slurm/robustness_test_data.sbatch)
train=$(sbatch --parsable slurm/robustness_train.sbatch)
simulation=$(sbatch --parsable --dependency=afterok:${train}:${test_data} slurm/robustness_simulation.sbatch)
postprocess=$(sbatch --parsable --dependency=afterok:${simulation} slurm/robustness_postprocess.sbatch)

echo "test data:      ${test_data}"
echo "training:       ${train}"
echo "simulation:     ${simulation} (after ${train}, ${test_data})"
echo "postprocessing: ${postprocess} (after ${simulation})"
