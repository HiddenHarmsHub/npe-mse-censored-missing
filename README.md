# npe-mse-censored-missing
Neural methods for Multiple Systems Estimation with censored or missing data

## Source Files

The `src/` directory contains the following files:

### Core Functions

- **`mse_functions.jl`**: Core functions for Multiple Systems Estimation (MSE) including parameter sampling, data simulation with censoring, likelihood computation, model training (both Neural Bayes Estimators and Neural Posterior Estimation), and utility functions for data manipulation and model evaluation.

- **`mcmc_functions.jl`**: Implements MCMC inference using Turing.jl, including model specifications for censored and uncensored data, and functions to run MCMC sampling with diagnostics (Rhat, ESS) for MSE models.

### Model Training and Evaluation

- **`train_models.jl`**: Main script for training Neural Bayes Estimators (NBE) and Neural Posterior Estimation (NPE) models across various hyperparameter grids (number of lists, network architecture, censoring levels, training size).

- **`simulation_study.jl`**: Evaluates trained NBE and NPE models on fixed test datasets across different list sizes (3-15 lists) and censoring configurations. Generates intercept estimates comparing NPE, NBE, and true values.

- **`combine_results.jl`**: Aggregates the per-file simulation study outputs (intercept estimates, MCMC summaries and NPE summaries) into single consolidated CSVs in `output/` for analysis.

### Analysis Scripts

- **`mcmc_simulation_study.jl`**: Runs MCMC inference on test datasets to obtain posterior samples for comparison with neural estimation methods. Parallelized using distributed computing.


### Performance Analysis

- **`speed_comparisons.jl`**: Benchmarks inference speed comparing NBE, NPE, and MCMC methods with various iteration counts. Performs multiple timed runs to assess computational efficiency.


### Real Data and Visualization

- **`real_data_analysis.jl`**: Applies trained models to real-world datasets, which are the silverman modern slavery data and the King data. Selects best-performing architectures based on Mean Absolute Error and generates estimates with confidence intervals.

- **`plots.R`**: Contains all the code to produce the plots in the paper.

### Usage Example

- **`usage_example.jl`**: Shows how to run inference on your own count data with the trained 5-list NBE (point estimates and 95% intervals) and NPE (posterior samples) models in `output/models_nbe/` and `output/models_npe/`.

## Data

The `data/` directory contains:

- **`silverman_5.csv`**: The modern slavery dataset with 5 lists.
- **`king.csv`**: The King dataset used in MSE studies.

Please see the references in the paper for more details on these datasets.


## Notes

- This code is intended to be run on a slurm cluster, hence the various distributed computing setups. If you wish to run the code locally, you may need to modify the distributed computing parts accordingly (i.e. remove the `using Distributed, SlurmClusterManager` and `addprocs(SlurmManager(); ...)` lines, drop the `@everywhere` macros, and replace `pmap` calls with standard `map`).