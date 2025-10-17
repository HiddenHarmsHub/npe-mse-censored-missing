## Here we train a model that takes the censoring level as an input
using CUDA, cuDNN, Distributions, NeuralEstimators, Flux, BSON

include("mse_functions.jl")


intercept_dist = Uniform(1, 10)
intercept_support = ifelse(typeof(intercept_dist) <: Uniform, params(intercept_dist), nothing)

sample_nbe(n_reps) = hcat([sample_parameters(n_lists, intercept_dist = intercept_dist) for _ in 1:n_reps]...)
function simulate_nbe(θ, m)
    censoring_threshold = rand(DiscreteUniform(0, 16))
    if censoring_threshold == 0
        simulated_data = hcat([simulate_data(params, m, censoring_threshold = censoring_threshold) for params in eachcol(θ)]...)
        n_data, n_rep = size(simulated_data)
        return vcat(simulated_data, zeros(Float32, n_data + 1, n_rep))
    end
    return hcat([[simulate_data(params, m, censoring_threshold = censoring_threshold); censoring_threshold] for params in eachcol(θ)]...)
end

n_hidden = 3
width = 256
n_lists = 4
train_size = 10_000

network = construct_MLP_c(width, n_hidden, n_lists, intercept_support)
estimator = PointEstimator(network)

estimator = train(
    estimator, 
    sample_nbe, 
    simulate_nbe, 
    K = train_size,
    m = 1,
    epochs = 10,
    stopping_epochs = 10,
    use_gpu = true
)

savepath = joinpath("output")

## -1 to indicate variable censoring
mdl_str = "model_$(n_lists)_$(width)_$(n_hidden)_-1_$(train_size).bson"
BSON.@save joinpath(savepath, mdl_str) estimator