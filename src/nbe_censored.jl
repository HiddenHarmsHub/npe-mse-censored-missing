using Pkg; Pkg.activate(".")
using Distributions, Random, NeuralEstimators, Flux, StatsPlots, DataFrames, Optim, Folds
using BSON: @save, @load

include("mse_functions.jl")

K = 5
intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)

sample_nbe(n_reps) = hcat([generate_parameters_nbe(K, intercept_dist, beta_dist, gamma_dist) for _ in 1:n_reps]...)
simulate_nbe(θ, m) = [generate_data_nbe(params, m) for params in eachcol(θ)]

function censorandaugment(Z; c, v = -1.0)
    W = 1 * (Z .<= c)
    U = ifelse.(Z .<= c, v, Z)
    return vcat(U, W)
end

Z = simulate_nbe(sample_nbe(1), 1)[1]
censorandaugment(Z, c = 6)

function simulatecensored_nbe(θ, m; kwargs...)
    Z = simulate_nbe(θ, m)
    UW = Folds.map(Z) do Zₖ
		mapslices(Z -> censorandaugment(Z; kwargs...), Zₖ, dims = 1)
	end
end

n = 2    # dimension of each data replicate (bivariate)
w = 128  # width of each hidden layer

# Final layer has output dimension d=2 and enforces parameter constraints
final_layer = Parallel(
    vcat,
    Dense(w, 1, tanh),    # ρ ∈ (-1,1)
    Dense(w, 1, sigmoid)  # δ ∈ (0,1)
)
ψ = Chain(Dense(n * 2, w, relu), Dense(w, w, relu))    
ϕ = Chain(Dense(w, w, relu), final_layer)           
network = DeepSet(ψ, ϕ)

# Initialise the estimator
estimator = PointEstimator(network)


# Number of independent replicates in each data set
m = 200 

# Train an estimator with no censoring
simulator1(θ, m) = simulatecensored_nbe(θ, m; c = 10) 
estimator1 = train(estimator, sample_nbe, simulator1, m = m) 