using Pkg; Pkg.activate(".")
using Distributions, Random, Optim, DataFrames, Plots, CSV, Combinatorics, Turing, LinearAlgebra, StatsPlots, NeuralEstimators, Flux, Folds, BSON

include("mse_functions.jl")

function train_model(K, m, n_param_validation, intercept_dist, beta_dist, gamma_dist, i, models_path)
    model_save_path = joinpath(models_path, "nbe_model_$i.bson")
    if isfile(model_save_path)
        println("Model $i already exists, skipping training.")
        return
    end
    ## Neural estimation wrappers
    sample_nbe(n_reps) = hcat([generate_parameters_nbe(K, intercept_dist, beta_dist, gamma_dist) for _ in 1:n_reps]...)
    simulate_nbe(θ, m) = [generate_data_nbe(params, m) for params in eachcol(θ)]

    ## First we train a neural network to try and infer parameters
    n_pars = 1 + K + binomial(K, 2)
    n_data = K + binomial(K, 2)

    w = 128  # width of each hidden layer 

    # Inner and outer networks
    ψ = Chain(Dense(n_data, w, relu), Dense(w, n_pars, relu))    
    ϕ = Chain(Dense(n_pars, w, relu), Dense(w, n_pars, identity))          

    # Combine into a DeepSet
    network = DeepSet(ψ, ϕ)

    estimator = PointEstimator(network)

    estimator = train(
        estimator, 
        sample_nbe, 
        simulate_nbe, 
        m = m,
        K = n_param_validation
    )
    BSON.@save joinpath(models_path, "nbe_model_$i.bson") estimator
end

intercept_dists = [Uniform(1, 10)]
beta_dists = [Normal(0, 4)]
gamma_dists = [Normal(0, 1/5)]

ms = [1, 10, 50, 100, 500]
Ks = [3, 4, 5, 6]
n_params_validation = [10_000, 100_000, 1_000_000, 10_000_000]

# Create all combinations of Ks and ms
grid = collect(Base.product(Ks, ms, n_params_validation, intercept_dists, beta_dists, gamma_dists))[:]
model_list = DataFrame(grid, [:K, :m, :n_params_validation, :intercept_dist, :beta_dist, :gamma_dist])

output_path = joinpath("output")
models_path = joinpath(output_path, "models")
mkpath(models_path)

model_list_file = joinpath(output_path, "model_list.BSON")
append_model_list(model_list_file, model_list)

Folds.map(
    model -> train_model(
        model.K, 
        model.m,
        model.n_params_validation, 
        model.intercept_dist, 
        model.beta_dist, 
        model.gamma_dist, 
        model.i, 
        models_path
    ),
    eachrow(model_list)
)
