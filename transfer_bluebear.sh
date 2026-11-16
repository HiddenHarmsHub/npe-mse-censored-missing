#!/bin/bash
## Copy results from BlueBEAR. Run combine_results.jl on the cluster first.
##   ./transfer_bluebear.sh                dry run
##   ./transfer_bluebear.sh --copy         copy
##   ./transfer_bluebear.sh --copy --all   also copy the large per-chunk folders, test sets and sampler draws
set -euo pipefail
cd "$(dirname "$0")"

REMOTE=marshjs@bluebear.bham.ac.uk:/rds/projects/s/seymourg-bsbt/npe-mse-censored-missing
## Fail fast when BlueBEAR is unreachable
OPTS=(-avz --prune-empty-dirs -e "ssh -o ConnectTimeout=20")
[[ " $* " == *" --copy "* ]] || OPTS+=(--dry-run)

## rsync uses the first matching rule, relative to output/
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
