# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
Research code for a paper comparing Neural Bayes Estimators (NBE, posterior quantiles), Neural Posterior Estimation (NPE, normalising flow) and MCMC (Turing NUTS) for Multiple Systems Estimation (log-linear Poisson capture–recapture models) when some cells are censored or missing. Julia 1.10 (`Project.toml`/`Manifest.toml`), built on `NeuralEstimators.jl` + Flux; R (`src/plots.R`, tidyverse via `pacman`) for figures. Unit tests for the likelihood and reference samplers run with `julia --project test/runtests.jl`; there is no build step.

## Running
- Run everything **from the repo root**. Data/output paths are written as `joinpath("output", ...)` / `joinpath("data", ...)` relative to the working directory, while scripts `include("mse_functions.jl")` relative to `src/`.
- Instantiate once: `julia --project -e 'using Pkg; Pkg.instantiate()'`
- `train_models.jl`, `simulation_study.jl`, `mcmc_simulation_study.jl` and `speed_comparisons.jl` are written for a **Slurm cluster**: they call `addprocs(SlurmManager())` and use `@everywhere`/`pmap`. To run them locally, remove the `Distributed`/`SlurmClusterManager` lines and the `addprocs` call, drop the `@everywhere` macros, and replace `pmap` with `map`.
- `real_data_analysis.jl`, `combine_results.jl` and `usage_example.jl` are not distributed and run directly with `julia --project src/<script>.jl`.
- Figures: `Rscript src/plots.R`. It reads the CSVs in `output/` and writes PNGs to `output/figures/`.
- Long-running scripts skip any output file that already exists (`overwrite_models = false`, `overwrite_files = false`, and the `isfile` checks). Delete the output files to force a rerun.

## Pipeline (in order)
1. `train_models.jl` builds hyperparameter grids of `(n_lists, width, n_hidden, train_size, censoring_lower, censoring_upper, m)` and trains NBE models (`train_model_mlp`) and NPE models (`train_npe`), saving them to `output/models_nbe/` and `output/models_npe/`.
2. `simulation_study.jl` generates fixed test sets (`output/test_data/test_data_K.bson`, seed 42, uncensored log-counts), evaluates every NBE/NPE pair on the intercept and writes `output/intercept_estimates/`. It also writes NPE samples/summaries for K=5.
3. `mcmc_simulation_study.jl` fits a proportional stratified subset of 1,000 K=5 test datasets (censoring 0–10, `output/benchmark_datasets.csv`) with the NUTS remediation ladder (`run_reference_fit`), IRLS-MH (`fit_irls`) and importance sampling (`importance_reference`), writing `output/{mcmc,irls}_{samples,summary,diagnostics}/` and `output/is_summary/`. `reference_validation.jl` is the local software check (cross-method agreement and K=3 prior-predictive coverage).
4. `combine_results.jl` concatenates the per-file outputs into `output/intercept_estimate_comparison.csv`, `output/{mcmc,irls,is,npe}_summary.csv`, `output/{mcmc,irls}_diagnostics.csv` and `output/sbc_ranks.csv` (fractional rank of the truth; central-interval coverage at level L is `abs(rank - 0.5) <= L / 2`).
5. `real_data_analysis.jl` uses the pre-specified architectures in `output/architecture_selection/selected.csv` (system A for Silverman, B for King, for both NBE and NPE). It fits the Silverman (5-list) and King (4-list, with censored cells) data with NBE, NPE, MCMC and MLE, writing to `output/real_data_analysis/`. MCMC uses the validated `run_reference_fit` stages (diagnostics in `*_mcmc_diagnostics.csv`), and the MLE is a multi-start BFGS fit with Wald intervals (`mle_estimates`).
6. `speed_comparisons.jl` benchmarks training and inference times and writes to `output/speed_comparisons/`.

### Model selection, after step 1
Two application-matched systems are defined in `model_selection_systems`: A (K=5, uncensored, Silverman) and B (K=4, censored on [1, 4], King).

Named **runs** in `model_selection_runs` fix the coefficient priors and training settings:
- `b4_g4` is the original run, with β, γ ~ N(0, 4²). It keeps the folders below.
- `b4_g4r` keeps the original priors but uses the new training settings (`retraining_settings`), separating prior effects from training effects.
- `b2_g2` and `b1_g1` use β, γ ~ N(0, 2²) and N(0, 1) with the new settings.
- `b4_g4x` is `b4_g4r` with a longer schedule (800 classifier and 600 conditional NPE epochs), trained from scratch. Warm restarts from trained networks were tried and rejected: a fresh Adam state knocks a converged flow off its optimum.

**`b4_g4x` is the main analysis** (main text of `docs/model_selection_summary.Rmd`); the other runs and the g-prior analysis are supplementary.

Every run except `b4_g4` writes to `output/model_selection_<run>/` and `output/models_model_selection_<run>/`. `model_selection_compare_runs.jl` tabulates every run with results in `output/model_selection_comparison/`.

Scripts pick the run from the `MS_RUN` environment variable (default `b4_g4`), for example `MS_RUN=b1_g1 julia --project src/model_selection_combine.jl`.
1. `architecture_selection.jl` (Slurm) scores the `grid_5` and `grid_4` NPE models on fresh validation sets. It applies the pre-specified rule in `selection_rule` and writes `output/architecture_selection/selected.csv`, which every later step reads.
2. `model_selection_train.jl` (Slurm) trains a structure classifier and a conditional NPE per system, with three replicates, into `output/models_model_selection/`.
3. `model_selection_test_data.jl` (single process) writes the test sets `output/model_selection/test_data_{A,B}.bson`. `model_selection_simulation.jl` (Slurm) evaluates the oracle, all-interactions, MAP and model-averaged posteriors, plus the fixed-model NPE, in chunks under `output/model_selection/simulation/`.
4. `model_selection_reference_study.jl` (Slurm) enumerates every structure on a stratified subset, estimating each marginal likelihood by importance sampling, and writes to `output/model_selection/reference/`.
5. `bash slurm/submit_model_selection.sh [run]` submits all of these for one run, with dependencies (`slurm/*.sbatch`, logs in `logs/`). It skips architecture selection once `selected.csv` exists and ends with a single-process job for the local scripts. Every stage skips completed work, so resubmitting a run only fills in what is missing.
6. Local scripts: `model_selection_prior_predictive.jl` (prior tables and real-data domain checks), `model_selection_combine.jl` (summary CSVs), `model_selection_parameter_calibration.jl` (SBC ranks and coverage of every parameter of the conditional NPE given the true structure, per replicate), `model_selection_real_data.jl` (reanalysis, including MLE/AIC/BIC of every structure in `{A,B}_mle_all_structures.csv`), `model_selection_local_checks.jl` (conditional NPE against the exact posterior on the real data and on observed-like simulations, with neural importance sampling; writes `local_checks/`), `model_selection_gprior.jl` (run-independent exact model averaging under conting's unit-information and hyper-g priors, `julia -t auto`, writes `output/model_selection_gprior/`) and `Rscript src/model_selection_plots.R`.
   - Published structures for the reanalysis are in `literature_models` in `model_selection_real_data.jl`, with their sources and published estimates. King et al. (2021) compute BIC with n = 15 cells (`BIC_cells`); Silverman (2020) with n = individuals (`BIC_individuals`).
   - g-priors use sum-to-zero coding (`sum_to_zero_map`): iid N(0, σ²) on sum-to-zero terms, which is γ ~ N(0, 16σ²) in corner-point terms with correlated main effects. Pass `X = one_hot_encode_parameters(K) * M` and `M` to `log_evidence`.
   - The test sets are not copied from the cluster by default. Regenerate them exactly with `MS_RUN=<run> julia +1.10 --project src/model_selection_test_data.jl` (Julia 1.10, as on the cluster, so the random streams match). The combine script needs them for the interaction-strength tables (`inclusion_by_strength.csv`, `structure_by_strength.csv`, and the `weakest_gamma` / `strongest_gamma` strata). Without the per-chunk folders it reuses the combined simulation CSVs.

### Robustness study, after the model-selection runs
`robustness_functions.jl` defines scenarios (`robustness_scenarios`: in distribution, parameter shifts, negative binomial overdispersion, omitted three-way interactions, and for system B censoring-rule shifts) and experiments (`robustness_experiments`: `main` with the N(0, 4²) networks and `b4_g4x` model averaging on every scenario, `prior` with networks trained under N(0, 2²)/N(0, 1) on S4a/S4b, `ablation` with uncensored system B). Every dataset is fitted by the multi-quantile NBE (`robustness_probs`, `MonotoneQuantilesQ`), the NPE, the NPE reweighted by neural importance sampling (its weights degenerate on most simulated datasets, so the figures leave it out and use its ESS only as a diagnostic), model averaging and the exact posterior (`exact_posterior`, Laplace-t importance sampling), with applicability diagnostics (NIS ESS, prior-predictive summary tails, NPE-embedding nearest-neighbour percentile).
1. `bash slurm/submit_robustness.sh` runs `robustness_test_data.jl` (single process, Julia 1.10; writes `output/robustness/test_data/` and `scenarios.csv`) and `robustness_train.jl` (networks in `output/models_robustness/{nbe,npe}/`), then `robustness_simulation.jl` (chunks in `output/robustness/simulation/`), then the post-processing job.
2. Local: `robustness_combine.jl` (summary CSVs with Monte Carlo SEs in `output/robustness/`, plus `grid_summary.csv` from `intercept_estimate_comparison.csv`), `robustness_real_data.jl` (real data under each prior, with diagnostics) and `Rscript src/robustness_plots.R`.
- The N(0, 4²) NPE of systems A and B is the original fixed-structure model in `output/models_npe/`. Filenames: `nbe_{K}_{w}_{d}_{ts}_{cl}_{cu}_{sb}_{sg}_{nq}.bson` and `npe_{K}_{w}_{d}_{enc}_{ts}_{cl}_{cu}_{sb}_{sg}.bson`.
- Datasets reaching the simulator cap are excluded from every summary; approximation summaries use only datasets whose exact importance ESS is at least 1,000.

Intermediate directories (`test_data`, `intercept_estimates`, `mcmc_*`, `npe_*`, `speed_comparisons`) are gitignored. Trained models, final summary CSVs and figures are committed.

## Core conventions (`src/mse_functions.jl`, `src/mcmc_functions.jl`)
- **Parameter vector** for K lists: `[intercept; β_1..β_K; γ_ij for i<j]`, length `1 + K + binomial(K,2)`, in the order of `enumerate_two_digit_numbers(K)`. `simulate_data` recovers K from `length(pars)`. Priors: intercept ~ Uniform(1,10), β, γ ~ Normal(0,4) everywhere, including the reference samplers.
- **Cells**: `2^K - 1` observable capture histories, ordered by `enumerate_all_combinations(K)` (strings such as `"1,3"`). In data CSVs the `group` column joins the list digits without separators (`"13"`). `one_hot_encode_parameters(K)` gives the cell × parameter design matrix used by the MCMC likelihood.
- **Network input encoding**: counts are transformed with `log(Z + 1)`. With censoring (`censoring_upper > 0`) the input is `vcat(U, W)` and has length `2 * (2^K - 1)`. `U` is the log-count, or `-1` for a censored/missing cell, and `W` is a 0/1 indicator. The MCMC code takes raw integer counts with `-1` marking censored cells.
- **Model filenames encode the hyperparameters**:
  - NBE: `model_{K}_{width}_{n_hidden}_{train_size}_{cl}_{cu}_{m}.bson` (key `:estimator`)
  - NBE is one network per file (a `PointEstimator`, key `:estimator`) whose output is the stacked 0.025, 0.5 and 0.975 quantiles of every parameter, trained on the summed pinball loss (`quantileloss` with `nbe_probs`). The final layer `MonotoneQuantiles` (`nbe_layers.jl`, needed to load the models) builds each quantile as a positive increment on the one below and maps the intercept's quantiles into its prior support. `load_model_nbe` returns the median, or `[lower; upper]` with `ci = true` (or an old `model_ci_...` name).
  - NPE: `model_{K}_{width}_{n_hidden}_{encoding_dim}_{train_size}_{cl}_{cu}_{m}.bson` (`encoding_dim` = 128)

  Scripts parse these names with `\d+` regexes, so keep the format when adding models. Use the `load_model_nbe` / `load_model_npe` helpers.
- Training budget: `train_size_default` (200,000 fresh simulations per epoch, the `train_size` in model filenames) and `training_settings` in `mse_functions.jl` apply to all NBE and NPE models; NPE uses `npe_num_coupling_layers` (10). The original 10,000-simulation models stopped after 5 epochs without improvement and were under-trained (over-wide NPE posteriors); `npe_pilot.jl` documents the comparison. `slurm/train_models.sbatch` retrains everything with multi-threaded workers.
- NBE networks with a Uniform intercept prior squash the intercept output into its support using `Compress`. NPE posterior samples are restricted to the intercept prior support by rejection with `boundedsampleposterior` / `bounded_sample` (bounds 1 and 10).
- `rpois` caps `logλ` at 43 to avoid Poisson overflow; `is_capped` flags datasets that hit it, and they are excluded from calibration checks.
- Test-set BSON files hold `Z_test` (log-counts), `Z_counts` (raw counts, K ≤ 10 only because of the BSON 2 GB array limit) and `params`. The samplers take raw counts from `load_test_counts`; never recover counts from Float32 log-counts.
- Reference inference (`mcmc_functions.jl`, `irls_functions.jl`): the likelihood is written in deviance form (`poisson_kernel`), and censored cells contribute `log P(a ≤ N ≤ b)`. NUTS stage 0 is the original Turing setup (logit-transformed intercept, diagonal metric). The remediation stages sample `z` with θ = mode + L z (`laplace_approximation`, a Newton mode from a deterministic log-count start), and enforce the intercept support by rejection. Non-finite energy errors are counted as `n_rejected`, not as divergences. The pass criteria are in `diagnostic_thresholds`.
- Model selection (`model_selection_functions.jl`, `model_selection_reference.jl`):
  - **Structures.** Structure class `c` is the bit pattern of `c - 1`, where bit `j` is the j-th pair of `enumerate_two_digit_numbers(K)`. Use `model_mask` / `model_index`.
  - **Latent coefficients.** Every γ keeps a latent coefficient, and `effective_parameters` zeroes the inactive ones before simulating.
  - **Conditional NPE input.** The conditional NPE input is `vcat(encoded y, mask)`.
  - **Model prior.** The primary model prior is ρ ~ Beta(1,1). Sensitivity priors in `sensitivity_model_priors` are obtained exactly by `reweight_model_probs`, with no retraining. In the simulation study, coverage under them reweights test sets by the `logw_*` columns.
  - **Filenames.** They add `a`, `b` and the γ prior sd `sg` plus a replicate index:
    - `classifier_{K}_{width}_{n_hidden}_{train_size}_{cl}_{cu}_{a}_{b}_{sg}_{rep}.bson`
    - `cnpe_{K}_{width}_{n_hidden}_{enc}_{train_size}_{cl}_{cu}_{sg}_{rep}.bson`

    The β prior sd is not in the names; it is fixed by the run's folder. A run may give the classifier its own width (`b1_g1` uses 256 for system A).
  - **Ensembles.** Replicate 0 means the ensemble of replicates 1–3. `load_classifier` / `load_cnpe` with `rep = 0` return vectors; `model_probabilities` averages the members' probabilities, and `raw_draws` splits draws evenly across members.
  - **Intercept support.** Model-averaged draws keep the structure drawn from the classifier and redraw θ given m until α lies in the intercept support (`bounded_draws`), so the structure probabilities are the classifier's. Rejecting (m, θ) jointly, as before November 2026, multiplied them by each structure's acceptance rate; `model_selection_support_acceptance.jl` measures the difference. A structure or dataset with no draws inside the support gives empty draws and NaN metrics (`n_draws = 0`), which `model_selection_combine.jl` counts in `datasets_without_draws.csv` and leaves out of the summaries.
  - **Training.** `training_options` builds the optimiser from a run's settings (Adam, optionally after `ClipNorm`). A small `NeuralEstimators.findlr` method lets NeuralEstimators read the learning rate from an `OptimiserChain`.
  - **Population quantities.** The primary quantity is the realised N0 ~ Poisson(e^α). N = N_obs + N0, where censored cells are imputed from their truncated Poisson, so N_obs is uncertain for censored data. Totals are Float64, because integer sums of capped cells overflow.
  - **Reference evidence.** It integrates over the active parameters only: `X[:, active]`, with the `K` keyword on `log_posterior` / `laplace_approximation` / `importance_reference`.
- `usage_example.jl` shows end-to-end inference on user-supplied counts with the shipped K=5, width 256, 3 hidden layers models.
