#!/bin/bash
## Submit the model-selection pipeline for one run, with dependencies. From the repository root:
##   bash slurm/submit_model_selection.sh            # original run, b4_g4: β, γ ~ N(0, 4²)
##   bash slurm/submit_model_selection.sh b4_g4r     # original priors retrained with the new settings
##   bash slurm/submit_model_selection.sh b2_g2      # β, γ ~ N(0, 2²), new settings
##   bash slurm/submit_model_selection.sh b1_g1      # β, γ ~ N(0, 1), new settings
## Runs are defined in model_selection_runs (src/model_selection_functions.jl). The run name reaches the
## Julia scripts through MS_RUN, which sbatch exports to the jobs. Every stage skips work already done,
## so resubmitting a run only does what is missing.
## Order: test data (and architecture selection, if not yet frozen) start at once; training waits for the
## architecture; the reference study needs only the test data, so it runs alongside training; the
## simulation study waits for training and test data; post-processing waits for both studies.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm_output

export MS_RUN="${1:-b4_g4}"
## BlueBEAR does not pass the submitting shell's environment to jobs, so MS_RUN is exported explicitly;
## without it every job silently falls back to the default run b4_g4. Only MS_RUN is passed (not ALL):
## carrying the login session's module state into a job breaks the module loads in the sbatch files.
submit() { sbatch --parsable --export=MS_RUN="${MS_RUN}" --job-name="$1_${MS_RUN}" "${@:2}"; }

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
