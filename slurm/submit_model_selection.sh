#!/bin/bash
## Submit the model-selection pipeline for one run, with dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MS_RUN="${1:-b4_g4}"
## The run name goes to each job as the sbatch script's argument
submit() { sbatch --parsable --job-name="$1_${MS_RUN}" "${@:2}" "${MS_RUN}"; }

test_data=$(submit ms_test_data slurm/model_selection_test_data.sbatch)
if [[ -f output/architecture_selection/selected.csv ]]; then
    echo "architecture:   already frozen (output/architecture_selection/selected.csv)"
    train=$(submit ms_train slurm/model_selection_train.sbatch)
else
    architecture=$(submit ms_architecture slurm/architecture_selection.sbatch)
    echo "architecture:   ${architecture}"
    train=$(submit ms_train --dependency=afterok:${architecture} slurm/model_selection_train.sbatch)
fi
reference=$(submit ms_reference --dependency=afterok:${test_data} slurm/model_selection_reference.sbatch)
simulation=$(submit ms_simulation --dependency=afterok:${train}:${test_data} slurm/model_selection_simulation.sbatch)
postprocess=$(submit ms_postprocess --dependency=afterok:${simulation}:${reference} slurm/model_selection_postprocess.sbatch)

echo "run:            ${MS_RUN}"
echo "test data:      ${test_data}"
echo "training:       ${train}"
echo "reference:      ${reference} (after ${test_data})"
echo "simulation:     ${simulation} (after ${train}, ${test_data})"
echo "postprocessing: ${postprocess} (after ${simulation}, ${reference})"
