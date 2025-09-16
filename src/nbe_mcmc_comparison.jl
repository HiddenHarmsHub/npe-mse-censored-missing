using Pkg; Pkg.activate(".")
using Distributions, Random, Optim, DataFrames, Plots, CSV, Combinatorics, Turing, LinearAlgebra, StatsPlots, NeuralEstimators, Flux

include("mse_functions.jl")

## First we train a neural network to try and infer parameters

K = 3
intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)


n_pars = 1 + K + binomial(K, 2)
n_data = K + binomial(K, 2)

w = 128  # width of each hidden layer 

# Inner and outer networks
ψ = Chain(Dense(n_data, w, relu), Dense(w, n_pars, relu))    
ϕ = Chain(Dense(n_pars, w, relu), Dense(w, n_pars, identity))          

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

θ_test = sample_nbe(1000)
Z_test = simulate_nbe(θ_test, 1)
assessment = assess(estimator, θ_test, Z_test, probs = [0.025, 0.975])
output_df = select(assessment[1], [:k, :parameter, :estimate, :truth])



## Now use MCMC to infer parameters


@model function mse_model(y, X)
    n_pars = ncol(X)
    K = Int(-0.5 + (sqrt(8 * (n_pars - 1) + 1) / 2))  # Solve for K given length of params
    intercept ~ Uniform(1, 10)
    betas ~ filldist(Normal(0, 4), K)
    gammas ~ filldist(Normal(0, 1/5), binomial(K, 2))
    
    params = vcat(intercept, betas, gammas)
    
    for i in eachindex(y)
        y[i] ~ Poisson(exp(dot(X[i, :], params)))
    end
end

num_chains = 4
vec(Z_test[1])
X = one_hot_encode(K)
input_y = Int64.(floor.(exp.(vec(Z_test[1]))))
m = mse_model(input_y, X)
chains = sample(m, NUTS(), MCMCThreads(), 1000, num_chains, progress = true)