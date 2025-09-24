using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere using Distributions, Random, NeuralEstimators, Flux, StatsPlots, DataFrames, Optim, BSON

@everywhere include("mse_functions.jl")


n_inner_hidden_layers = [2]
w_encoder_values = [32]
n_outer_hidden_layers = [1]
hidden_layer_widths = [256]
ms = [1, 5, 10, 25]
n_training_samples = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]


grid = collect(Base.product(
    n_inner_hidden_layers, 
    w_encoder_values, 
    n_outer_hidden_layers, 
    hidden_layer_widths, 
    ms,
    n_training_samples
))[:]

architecture_list = DataFrame(
    grid, 
    [
        :n_inner_hidden_layers, 
        :w_encoder, 
        :n_outer_hidden_layers,
        :hidden_layer_width,
        :m,
        :n_training_samples
    ]
)

architecture_list.i = 1:nrow(architecture_list)
architecture_list.K .= 4

output_path = joinpath("output")
architectures_path = joinpath(output_path, "architectures_sample_size_censored")
mkpath(architectures_path)

BSON.@save joinpath(output_path, "architecture_list_sample_size_censored.bson") architecture_list


@everywhere function train_model(i, K, n_inner_hidden_layers, w_encoder, n_outer_hidden_layers, hidden_layer_width, m, savepath, n_training_samples)
    intercept_dist = Uniform(1, 10)
    beta_dist = Normal(0, 4)
    gamma_dist = Normal(0, 1/5)


    n_data = K + binomial(K, 2)
    n_pars = 1 + n_data

    sample_nbe(n_reps) = hcat([generate_parameters_nbe(K, intercept_dist, beta_dist, gamma_dist) for _ in 1:n_reps]...)
    simulate_nbe(θ, m) = [generate_data_nbe(params, m) for params in eachcol(θ)]

    function simulatecensored_nbe_2(θ, m; kwargs...)
        Z = simulate_nbe(θ, m)
        UW = map(Z) do Zₖ
            mapslices(Z -> censorandaugment(Z; kwargs...), Zₖ, dims = 1)
        end
        return UW
    end

    censor_threshold = 4
    simulator_censored(θ, m) = simulatecensored_nbe_2(θ, m, c = log(censor_threshold))

    # Inner and outer networks
    ψ = Chain(
        Dense(n_data * 2, hidden_layer_width, relu),
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
        simulator_censored, 
        m = m,
        epochs = 200,
        K = n_training_samples
    )

    BSON.@save joinpath(savepath, "architecture_$i.bson") estimator
end

pmap(
    model -> train_model(
        model.i,
        model.K,
        model.n_inner_hidden_layers, 
        model.w_encoder, 
        model.n_outer_hidden_layers, 
        model.hidden_layer_width,
        model.m,
        architectures_path,
        model.n_training_samples
    ),
    eachrow(architecture_list)
)
