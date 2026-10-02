#!/bin/bash
## Submit the model-selection pipeline with dependencies. Run from the repository root:
##   bash slurm/submit_model_selection.sh
## Order: test data and architecture selection start at once; training waits for architecture
## selection; the reference study needs only the test data, so it runs alongside training;
## the simulation study waits for training and test data; post-processing waits for both studies.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm_output

submit() { sbatch --parsable "$@"; }

test_data=$(submit slurm/model_selection_test_data.sbatch)
architecture=$(submit slurm/architecture_selection.sbatch)
train=$(submit --dependency=afterok:${architecture} slurm/model_selection_train.sbatch)
reference=$(submit --dependency=afterok:${test_data} slurm/model_selection_reference.sbatch)
simulation=$(submit --dependency=afterok:${train}:${test_data} slurm/model_selection_simulation.sbatch)
postprocess=$(submit --dependency=afterok:${simulation}:${reference} slurm/model_selection_postprocess.sbatch)

echo "test data:      ${test_data}"
echo "architecture:   ${architecture}"
echo "training:       ${train} (after ${architecture})"
echo "reference:      ${reference} (after ${test_data})"
echo "simulation:     ${simulation} (after ${train}, ${test_data})"
echo "postprocessing: ${postprocess} (after ${simulation}, ${reference})"
