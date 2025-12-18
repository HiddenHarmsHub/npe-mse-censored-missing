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

- **`combine_simulation_study_results.jl`**: Aggregates simulation study results from multiple intercept estimate CSV files into a single consolidated dataset for analysis.

### Analysis Scripts

- **`analyse_mcmc_samples.R`**: R script for analyzing MCMC samples, including diagnostic checks (Rhat values), visualization of trace plots, and comparison of MCMC estimates with true parameter values.

- **`mcmc_simulation_study.jl`**: Runs MCMC inference on test datasets to obtain posterior samples for comparison with neural estimation methods. Parallelized using distributed computing.


### Performance Analysis

- **`speed_comparisons.jl`**: Benchmarks inference speed comparing NBE, NPE, and MCMC methods with various iteration counts. Performs multiple timed runs to assess computational efficiency.


### Real Data and Visualization

- **`real_data_analysis.jl`**: Applies trained models to real-world datasets, which are the silverman modern slavery data and the King data. Selects best-performing architectures based on Mean Absolute Error and generates estimates with confidence intervals.

- **`plots.R`**: Contains all the code to produce the plots in the paper.

## Data

The `data/` directory contains:

- **`silverman_5.csv`**: The modern slavery dataset with 5 lists.
- **`king_data.csv`**: The King dataset used in MSE studies.

Please see the references in the paper for more details on these datasets.


## Notes

- This code is intended to be run on a slurm cluster, hence the various distributed computing setups. If you wish to run the code locally, you may need to modify the distributed computing parts accordingly (i.e. remove `using Distributed` and `using SlurmClusterManager` lines, and replace `@distributed` loops with standard `for` loops).