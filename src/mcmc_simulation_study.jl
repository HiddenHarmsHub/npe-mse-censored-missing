## Reference posteriors (NUTS with remediation, and IRLS-MH) for a stratified
## subset of the K=5 test set, used to validate NBE and NPE
using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")
@everywhere include("mcmc_functions.jl")
@everywhere using Logging
@everywhere disable_logging(Logging.Info)  # Turing logs every initial step size


test_path = joinpath("output", "test_data")
n_lists = 5
censoring_lower = 0
censoring_upper = 10
test_counts, test_pars = load_test_counts(test_path, n_lists, censoring_lower, censoring_upper)

benchmark_idx = select_benchmark_datasets(test_pars, test_counts; n = 1000)
CSV.write(
    joinpath("output", "benchmark_datasets.csv"),
    DataFrame(
        dataset = benchmark_idx,
        intercept = test_pars[1, benchmark_idx],
        n_censored = vec(sum(test_counts[:, benchmark_idx] .== -1, dims = 1)),
        capped = is_capped.(eachcol(test_pars[:, benchmark_idx]))
    )
)

paths = (
    mcmc_samples = joinpath("output", "mcmc_samples"),
    mcmc_summary = joinpath("output", "mcmc_summary"),
    mcmc_diagnostics = joinpath("output", "mcmc_diagnostics"),
    irls_samples = joinpath("output", "irls_samples"),
    irls_summary = joinpath("output", "irls_summary"),
    irls_diagnostics = joinpath("output", "irls_diagnostics"),
    is_summary = joinpath("output", "is_summary")
)
mkpath.(values(paths))

overwrite_files = false
test_idx = benchmark_idx

if !overwrite_files
    test_idx = filter(
        slice_idx ->  begin
            !isfile(joinpath(paths.is_summary, "is_test_summary_$slice_idx.csv"))
        end,
        test_idx
    )
end


## A failing dataset is reported and left for the next run, which skips completed datasets
pmap(
    slice_idx ->  begin
        wid = myid()
        println("Worker $wid running ")
        try
            run_mcmc_test_slice(
                slice_idx,
                test_counts,
                test_pars;
                censoring_lower = censoring_lower,
                censoring_upper = censoring_upper,
                paths = paths
            )
        catch e
            println("Dataset $slice_idx failed: ", sprint(showerror, e))
        end
    end,
    test_idx
)
