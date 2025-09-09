using Pkg; Pkg.activate(".")
using Distributions, Random, NeuralEstimators, Flux, StatsPlots
using BSON: @save, @load

function generate_parameters(K::Int, intercept_dist, beta_dist, gamma_dist)
    intercept = rand(intercept_dist)
    betas = rand(beta_dist, K)
    gammas = rand(gamma_dist, binomial(K, 2))
    par_estimates = vcat(
        intercept,
        betas,
        gammas
    )
    return par_estimates
end

function generate_data(params, m)
    K = Int(-0.5 + (sqrt(8 * length(params) - 7) / 2))  # Solve for K given length of params
    intercept = params[1]
    betas = params[2:1+K]
    gammas = params[2+K:end]

    Z = zeros(K + binomial(K, 2), m)

    for i in 1:m
        main_counts = [
            rand(Poisson(exp(intercept + betas[i]))) for i in 1:K
        ]
        
        pair_counts = [
            rand(Poisson(exp(intercept + betas[i] + betas[j] + gammas[binomial(K, 2) - binomial(K - i + 1, 2) + (j - i)]))) for i in 1:K-1 for j in i+1:K
        ]
        
        Z[:, i] = vcat(main_counts, pair_counts)

    end
    return log.(Z .+ 1)  # Log-transform the counts
end

K = 6
intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)

sample(n_reps) = hcat([generate_parameters(K, intercept_dist, beta_dist, gamma_dist) for _ in 1:n_reps]...)
simulate(θ, m) = [generate_data(params, m) for params in eachcol(θ)]

n_pars = 1 + K + binomial(K, 2)
n_data = K + binomial(K, 2)

w = 128  # width of each hidden layer 

# Inner and outer networks
ψ = Chain(Dense(n_data, w, relu), Dense(w, n_pars, relu))    
ϕ = Chain(Dense(n_pars, w, relu), Dense(w, n_pars, identity))          

# Combine into a DeepSet
network = DeepSet(ψ, ϕ)

estimator = PointEstimator(network)

m = 500
estimator = train(estimator, sample, simulate, m = m)

@save joinpath("output", "nbe_model.bson") estimator





## validation checking
θ_test = sample(1000)
Z_test = simulate(θ_test, m)
assessment = assess(estimator, θ_test, Z_test, probs = [0.025, 0.975])



bias(assessment)      
rmse(assessment)     
risk(assessment)     



bernard_data = Matrix(Float32[
    54 463 907 695 316 57 15 19 3 0 0 56 19 1 3 69 10 31 8 6 1
])'  # Convert to a matrix with one column

par_estimates = NeuralEstimators.estimate(estimator, log.(bernard_data .+ 1))


exp(par_estimates[1])
