#!/bin/bash
## Copy results from BlueBEAR: the model-selection runs (b4_g4, b4_g4r, b2_g2, b1_g1, ...) and the
## reference-inference benchmark (NUTS, IRLS-MH and importance sampling on the K=5 benchmark datasets).
## Run julia --project src/combine_results.jl on the cluster first, so the combined CSVs and SBC ranks
## exist there and the raw NPE posterior samples never need copying.
## Run from the repository root:
##   ./transfer_bluebear.sh                dry run: list what would be copied
##   ./transfer_bluebear.sh --copy         copy
##   ./transfer_bluebear.sh --copy --all   also copy the large per-chunk simulation and per-dataset
##                                         reference folders, the test sets and the NUTS/IRLS draws
##                                         (needed for the trace plots in plots.R)
set -euo pipefail
cd "$(dirname "$0")"

REMOTE=marshjs@bluebear.bham.ac.uk:/rds/projects/s/seymourg-bsbt/npe-mse-censored-missing
## Fail fast when BlueBEAR is unreachable (e.g. the university VPN is not connected)
OPTS=(-avz --prune-empty-dirs -e "ssh -o ConnectTimeout=20")
[[ " $* " == *" --copy "* ]] || OPTS+=(--dry-run)

## rsync uses the first rule that matches, relative to output/. Patterns rather than folder names, so runs
## that do not exist on the cluster yet are simply skipped and new runs are picked up automatically:
##   model_selection*        each run's summaries, real-data and prior-predictive results, and the
##                           cross-run comparison (model_selection_comparison)
##   models_model_selection* each run's trained networks and training curves
##   /*.csv                  combined results from combine_results.jl (summaries, diagnostics, SBC ranks)
##   {mcmc,irls}_*, is_*     per-dataset reference-inference summaries and diagnostics
##   real_data_analysis      fixed-structure real-data analysis (real_data_analysis.jl)
##   robustness, models_robustness  robustness study summaries, figures and networks
##   models_cap_sensitivity  baseline networks retrained for the simulator-cap check (cap_sensitivity_train.jl)
LARGE=()
[[ " $* " == *" --all "* ]] || LARGE=(--exclude '/model_selection*/simulation/' --exclude '/model_selection*/reference/'
                                      --exclude '/model_selection*/test_data_*.bson'
                                      --exclude '/mcmc_samples/' --exclude '/irls_samples/'
                                      --exclude '/robustness/simulation/' --exclude '/robustness/test_data/'
                                      --exclude '/robustness/posteriors.csv')
FILTERS=("${LARGE[@]}"
    --include '/architecture_selection/***'
    --include '/models_model_selection*/***'
    --include '/model_selection*/***'
    --include '/figures/***'
    --include '/*.csv'
    --include '/mcmc_samples/***'
    --include '/mcmc_summary/***'
    --include '/mcmc_diagnostics/***'
    --include '/irls_samples/***'
    --include '/irls_summary/***'
    --include '/irls_diagnostics/***'
    --include '/is_summary/***'
    --include '/reference_validation/***'
    --include '/real_data_analysis/***'
    --include '/speed_comparisons/***'
    --include '/robustness/***'
    --include '/models_robustness/***'
    --include '/models_cap_sensitivity/***'
    --exclude '*')

mkdir -p output logs
rsync "${OPTS[@]}" "${FILTERS[@]}" "$REMOTE/output/" output/
rsync "${OPTS[@]}" "$REMOTE/logs/" logs/
