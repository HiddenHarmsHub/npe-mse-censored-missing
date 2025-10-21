## Here we run use MCMC to infer model parameters and compare those
## with that obtained through NBE
using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")

@everywhere using Turing

@everywhere @model function mse_model_censored(y, X, intercept_dist, beta_dist, gamma_dist, censoring_lower, censoring_threshold)
    n_pars = size(X, 2)
    K = Int(-0.5 + (sqrt(8 * (n_pars - 1) + 1) / 2))  # Solve for K given length of params
    intercept ~ intercept_dist
    betas ~ filldist(beta_dist, K)
    gammas ~ filldist(gamma_dist, binomial(K, 2))
    
    params = vcat(intercept, betas, gammas)
    
    Turing.@addlogprob!(likelihood_censored(y, params, X, censoring_lower, censoring_threshold))
end

@everywhere function run_mcmc_test_slice(slice_idx, n_lists, test_data, test_pars; num_chains = 4, samples_path = nothing, summary_path = nothing, n_iterations = 5000)

    println("Running MCMC for slice $slice_idx...")

    samples_file = joinpath(samples_path, "mcmc_test_results_$slice_idx.csv")
    summary_file = joinpath(summary_path, "mcmc_test_summary_$slice_idx.csv")

    if isfile(samples_file) && isfile(summary_file)
        println("MCMC results and summary for slice $slice_idx already exist. Skipping...")
        return
    end

    n_data = 2^n_lists - 1
    mcmc_test_data = test_data[1:n_data, slice_idx]

    ## Conver back to integer for Poisson sampling
    input_counts = Int.(ifelse.(mcmc_test_data .== -1.0, -1.0, floor.(exp.(mcmc_test_data) .- 1)))
    X = one_hot_encode_parameters(n_lists)

    intercept_dist = Uniform(1, 10)
    beta_dist = Normal(0, 4)
    gamma_dist = Normal(0, 1/5)
    censoring_lower = 1
    censoring_threshold = 10

    m = mse_model_censored(input_counts, X, intercept_dist, beta_dist, gamma_dist, censoring_lower, censoring_threshold)
    chains = sample(
        m, 
        NUTS(), 
        MCMCSerial(), 
        n_iterations, 
        num_chains, 
        progress = false,
        parallel = false
    )

    res_df = DataFrame(chains)
    res_df = select(res_df, r"intercept|betas|gammas")

    if !isnothing(samples_path)
        CSV.write(samples_file, res_df)
    end

    summstats = summarystats(chains)

    summary_df = DataFrame(
        parameters = summstats.nt[:parameters],
        true_values = vec(test_pars[:, slice_idx]),
        estimated_means = mean.(eachcol(res_df)),
        estimated_medians = median.(eachcol(res_df)),
        estimated_std = std.(eachcol(res_df)),
        lower_95ci = quantile.(eachcol(res_df), 0.025),
        upper_95ci = quantile.(eachcol(res_df), 0.975),
        rhat = summstats.nt[:rhat]
    )

    if !isnothing(summary_path)
        CSV.write(summary_file, summary_df)
    end
end


test_path = joinpath("output", "test_data")
n_lists = 5
test_data, test_pars = load_test_data(test_path, n_lists, 0, 10)

mcmc_samples_path = joinpath("output", "mcmc_samples")
mcmc_summary_path = joinpath("output", "mcmc_summary")
mkpath(mcmc_samples_path)
mkpath(mcmc_summary_path)

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
    #1:size(test_data, 2)
    1:96
)