using Pkg; Pkg.activate(".")
using Distributions, Random, NeuralEstimators, Flux, StatsPlots, DataFrames, Optim
using BSON: @save, @load

include("mse_functions.jl")


K = 5
intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)

sample_nbe(n_reps) = hcat([generate_parameters_nbe(K, intercept_dist, beta_dist, gamma_dist) for _ in 1:n_reps]...)
simulate_nbe(θ, m) = [generate_data_nbe(params, m) for params in eachcol(θ)]

n_data = K + binomial(K, 2)
n_pars = 1 + n_data

load_estimator = true

if load_estimator
    @load joinpath("output", "nbe_model_final.bson") estimator
    println("Loaded existing estimator")
else
    w = 128  # width of each hidden layer 

    # Inner and outer networks
    ψ = Chain(Dense(n_data, w, relu), Dense(w, n_pars, relu))
    ϕ = Chain(Dense(n_pars, w, relu), Dense(w, n_pars))

    # Combine into a DeepSet
    network = DeepSet(ψ, ϕ)

    estimator = PointEstimator(network)

    m = 50
    estimator = train(
        estimator, 
        sample_nbe, 
        simulate_nbe, 
        m = m
    )

    @save joinpath("output", "nbe_model_final.bson") estimator
end


## validation checking
θ_test = sample_nbe(1000)
Z_test = simulate_nbe(θ_test, m)
assessment = assess(estimator, θ_test, Z_test, probs = [0.025, 0.975])


## check on a fixed parameter set
θ_fixed = sample_nbe(1)
n_reps = 1000
nbe_estimates = map(1:n_reps) do i
    Z_fixed = simulate_nbe(θ_fixed, 1)
    return vec(NeuralEstimators.estimate(estimator, Z_fixed))
end |> (y -> hcat(y...))

plot_nbe_estimates(nbe_estimates, θ_fixed, param_max = 6)

n_sims = 1000
par_names = ["intercept"; ["beta[$i]" for i in 1:K]; ["gamma[$i,$j]" for i in 1:K-1 for j in i+1:K]]

nbe_coverage_df = map(1:n_sims) do i
    θ_fixed = sample_nbe(1)
    Z_fixed = simulate_nbe(θ_fixed, 1)
    θ_hat = vec(NeuralEstimators.estimate(estimator, Z_fixed))
    return DataFrame(
        par = par_names,
        estimate = θ_hat,
        error = θ_hat - vec(θ_fixed),
        relative_error = (θ_hat - vec(θ_fixed)) ./ abs.(vec(θ_fixed))
    )
end |> (x -> vcat(x...))

intercept_df = filter(x -> x.par == "intercept", nbe_coverage_df)
scatter(intercept_df.estimate, intercept_df.error, xlabel = "Estimate", ylabel = "Error", title = "NBE Intercept Estimates")

beta_df = filter(x -> x.par == "beta[1]", nbe_coverage_df)
scatter(beta_df.estimate, beta_df.error, xlabel = "Estimate", ylabel = "Error", title = "NBE Intercept Estimates")

