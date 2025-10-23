## Here we run use MCMC to infer model parameters and compare those
## with that obtained through NBE
using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")
@everywhere include("mcmc_functions.jl")


test_path = joinpath("output", "test_data")
n_lists = 5
test_data, test_pars = load_test_data(test_path, n_lists, 0, 10)

mcmc_samples_path = joinpath("output", "mcmc_samples")
mcmc_summary_path = joinpath("output", "mcmc_summary")
mkpath(mcmc_samples_path)
mkpath(mcmc_summary_path)

overwrite_files = false
test_idx = 1:size(test_data, 2)

if !overwrite_files
    test_idx = filter(
        slice_idx ->  begin
            samples_file = joinpath(mcmc_samples_path, "mcmc_test_results_$slice_idx.csv")
            summary_file = joinpath(mcmc_summary_path, "mcmc_test_summary_$slice_idx.csv")
            !(isfile(samples_file) && isfile(summary_file))
        end,
        collect(test_idx)
    )
end


pmap(
    slice_idx ->  begin
        wid = myid()
        println("Worker $wid running ")
        run_mcmc_test_slice(
            slice_idx,
            n_lists, 
            test_data, 
            test_pars; 
            num_chains = 4, 
            samples_path = mcmc_samples_path, 
            summary_path = mcmc_summary_path
        )
    end,
    test_idx
)