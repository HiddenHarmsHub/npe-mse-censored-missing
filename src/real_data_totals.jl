## Expected hidden count λ0 = e^α, realised hidden count N0 and total population N for the
## fixed-structure (all interactions) real-data fits, for the tables of the real-data section.
## MCMC and NPE: N0 ~ Poisson(e^α) and censored cells imputed from their truncated Poisson, per draw.
## NBE and MLE give intervals for α only, so their totals are N_obs + λ0 at the interval endpoints.
## Run from the repo root: julia --project src/real_data_totals.jl
include("mse_functions.jl")
include("model_selection_functions.jl")
using CSV, DataFrames, Random, Statistics

output_path = joinpath("output", "real_data_analysis")
Random.seed!(2024)

datasets = [
    (name = "silverman", counts = load_silverman_data(), censoring_lower = 0, censoring_upper = 0),
    (name = "king", counts = load_king_counts(), censoring_lower = 1, censoring_upper = 4),
]

q(x) = (median = median(x), q025 = quantile(x, 0.025), q975 = quantile(x, 0.975))

rows = DataFrame()
for d in datasets
    counts = vec(Int.(d.counts))
    exact = sum(counts[counts .>= 0])
    n_censored = count(==(-1), counts)
    imputed_censored = nothing
    for method in ["mcmc", "npe"]
        samples = CSV.read(joinpath(output_path, "$(d.name)_$(method)_posterior_samples.csv"), DataFrame)
        θ = Matrix(select(samples, Not(intersect(names(samples), ["iteration", "chain"]))))'
        draws = population_draws(θ, counts; d.censoring_lower, d.censoring_upper)
        method == "mcmc" && (imputed_censored = median(draws.N_obs) - exact)
        for quantity in (:lambda0, :N0, :N_obs, :N)
            push!(rows, merge((data = d.name, method = uppercase(method), quantity = String(quantity), n_draws = size(θ, 2)), q(getfield(draws, quantity))))
        end
    end
    ## N_obs for the NBE and MLE totals: exact cells plus the MCMC posterior median of the censored cells
    n_obs = exact + (n_censored > 0 ? imputed_censored : 0)
    for method in ["nbe", "mle"]
        est = CSV.read(joinpath(output_path, "$(d.name)_$(method)_parameter_estimates.csv"), DataFrame)
        a = only(eachrow(filter(r -> r.parameter == "alpha", est)))
        λ = (median = exp(a.estimate), q025 = exp(a.lower_ci), q975 = exp(a.upper_ci))
        push!(rows, merge((data = d.name, method = uppercase(method), quantity = "lambda0", n_draws = 0), λ))
        push!(rows, merge((data = d.name, method = uppercase(method), quantity = "N_obs_plus_lambda0", n_draws = 0),
            (median = λ.median + n_obs, q025 = λ.q025 + n_obs, q975 = λ.q975 + n_obs)))
    end
    println("$(d.name): $exact exact individuals, $n_censored censored cells, imputed censored total (MCMC median) $(imputed_censored)")
end

CSV.write(joinpath(output_path, "totals.csv"), rows)
show(rows, allrows = true)
