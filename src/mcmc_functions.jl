using Turing

@model function mse_model_censored(y, X, intercept_dist, beta_dist, gamma_dist, censoring_lower, censoring_threshold)
    n_pars = size(X, 2)
    K = Int(-0.5 + (sqrt(8 * (n_pars - 1) + 1) / 2))  # Solve for K given length of params
    intercept ~ intercept_dist
    betas ~ filldist(beta_dist, K)
    gammas ~ filldist(gamma_dist, binomial(K, 2))
    
    params = vcat(intercept, betas, gammas)
    
    Turing.@addlogprob!(likelihood_censored(y, params, X, censoring_lower, censoring_threshold))
end

function run_mcmc_test_slice(slice_idx, n_lists, test_data, test_pars; num_chains = 4, samples_path = nothing, summary_path = nothing, n_iterations = 5000)
    println("Running MCMC for slice $slice_idx...")

    samples_file = joinpath(samples_path, "mcmc_test_results_$slice_idx.csv")
    summary_file = joinpath(summary_path, "mcmc_test_summary_$slice_idx.csv")

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

function likelihood_censored(counts::Vector{Int64}, pars::Vector, X::Matrix{Int64}, censoring_lower::Int, censoring_threshold::Int)
    rates = exp.(X * pars)
    ll = 0.0
    for (rate, count) in zip(rates, counts)
        if count == -1
            # P(0 <= X <= censoring_threshold) = F(censoring_threshold; λ) - F(0; λ)
            for k in censoring_lower:censoring_threshold
                ll += k * log.(rate) .- rate 
            end
        else
            ll += count .* log.(rate) .- rate
        end
    end
    return ll
end