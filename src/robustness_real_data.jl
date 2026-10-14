## Robustness study: real-data estimates under each coefficient prior, with the applicability diagnostics.
include("mse_functions.jl")
include("mcmc_functions.jl")
include("model_selection_functions.jl")
include("model_selection_reference.jl")
include("robustness_functions.jl")

n_draws = 20_000
real_data = Dict("A" => load_silverman_data(), "B" => load_king_counts())
q(x) = quantile(x, [0.5, 0.025, 0.975])

rows, diagnostic_rows = NamedTuple[], NamedTuple[]
for s in ["A", "B"]
    config = robustness_config(s)
    K = config.K
    counts = real_data[s]
    y = encode_counts(counts; censored = config.censoring_upper > 0)
    fit = (K = K, censoring_lower = config.censoring_lower, censoring_upper = config.censoring_upper)
    for sigma in robustness_sigmas
        priors = coefficient_priors(beta_sd = sigma, gamma_sd = sigma)
        Random.seed!(stable_seed("$(s)_real_$(sigma)"))
        population(θ) = population_draws(θ, counts; fit.censoring_lower, fit.censoring_upper)
        npe = BSON.load(robustness_npe_file(config, sigma))[:estimator]
        nis = npe_importance_sampling(npe, y, counts, fit, priors, n_draws)
        exact = exact_posterior(counts, fit, priors, n_draws; n_draws = 200_000)
        draws = Dict("npe" => population(bounded_draws(npe, reshape(y, :, 1), n_draws)), "npe_nis" => population(nis.θ),
                     "exact" => population(exact.θ))
        for (method, d) in draws, quantity in (:lambda0, :N0, :N)
            v = q(getproperty(d, quantity))
            push!(rows, (system = s, sigma = sigma, method = method, quantity = String(quantity), median = v[1], lower_95 = v[2], upper_95 = v[3]))
        end
        nbe_file = robustness_nbe_file(config, sigma)
        if isfile(nbe_file)
            α = nbe_quantiles(BSON.load(nbe_file)[:estimator], y)[1, :]
            at(p) = exp(α[findfirst(≈(p), robustness_probs)])
            push!(rows, (system = s, sigma = sigma, method = "nbe", quantity = "lambda0", median = at(0.5), lower_95 = at(0.025), upper_95 = at(0.975)))
        end
        if sigma == 4
            reference = BSON.load(robustness_reference_file(s))
            push!(diagnostic_rows, (system = s, nis_ess = nis.ess, exact_ess = exact.ess,
                summary_tail = summary_tail(counts, K, reference[:summaries]),
                embedding_percentile = embedding_percentile(embedding_reference(npe, reference[:y]), y)))
        end
    end
end
CSV.write(joinpath(robustness_path, "real_data.csv"), sort!(DataFrame(rows), [:system, :quantity, :method, :sigma]))
CSV.write(joinpath(robustness_path, "real_data_diagnostics.csv"), DataFrame(diagnostic_rows))
println(DataFrame(diagnostic_rows))
