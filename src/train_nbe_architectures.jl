using Pkg; Pkg.activate(".")
using Distributions, Random, NeuralEstimators, Flux, StatsPlots, DataFrames, Optim, Folds


include("mse_functions.jl")

K = 4
intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)

sample_nbe(n_reps) = hcat([generate_parameters_nbe(K, intercept_dist, beta_dist, gamma_dist) for _ in 1:n_reps]...)
simulate_nbe(θ, m) = [generate_data_nbe(params, m) for params in eachcol(θ)]

n_data = K + binomial(K, 2)
n_pars = 1 + n_data

n_inner_hidden_layers = collect(1:3)
w_encoder_values = [8, 16, 32, 64, 128, 256]
n_outer_hidden_layers = collect(1:3)
hidden_layer_widths = [8, 16, 32, 64, 128, 256]
ms = [1, 5, 10, 20, 50, 100]


grid = collect(Base.product(
    n_inner_hidden_layers, 
    w_encoder_values, 
    n_outer_hidden_layers, 
    hidden_layer_widths, 
    ms
))[:]

architecture_list = DataFrame(
    grid, 
    [
        :n_inner_hidden_layers, 
        :w_encoder, 
        :n_outer_hidden_layers,
        :hidden_layer_width,
        :m    
    ]
)

architecture_list.i = 1:nrow(architecture_list)

output_path = joinpath("output")
architectures_path = joinpath(output_path, "architectures")
mkpath(architectures_path)

BSON.@save joinpath(output_path, "architecture_list.bson") architecture_list

function train_model(i, n_inner_hidden_layers, w_encoder, n_outer_hidden_layers, hidden_layer_width, m)
    # Inner and outer networks
    ψ = Chain(
        Dense(n_data, hidden_layer_width, relu),
        [Dense(hidden_layer_width, hidden_layer_width, relu) for _ in 1:n_inner_hidden_layers]...,
        Dense(hidden_layer_width, w_encoder, relu)
    )
    ϕ = Chain(
        Dense(w_encoder, hidden_layer_width, relu), 
        [Dense(hidden_layer_width, hidden_layer_width, relu) for _ in 1:n_outer_hidden_layers]...,
        Dense(hidden_layer_width, n_pars)
    )

    # Combine into a DeepSet
    network = DeepSet(ψ, ϕ)

    estimator = PointEstimator(network)

    estimator = train(
        estimator, 
        sample_nbe, 
        simulate_nbe, 
        m = m
    )

    BSON.@save joinpath(architectures_path, "architecture_$i.bson") estimator
end

Folds.map(
    model -> train_model(
        model.i,
        model.n_inner_hidden_layers, 
        model.w_encoder, 
        model.n_outer_hidden_layers, 
        model.hidden_layer_width,
        model.m
    ),
    eachrow(architecture_list)
)