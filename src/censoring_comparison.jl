using Pkg; Pkg.activate(".")
using Distributions, Random, NeuralEstimators, Flux, StatsPlots, DataFrames, Optim, Folds, Turing, CSV
using BSON: @save, @load

include("mse_functions.jl")

K = 3
intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)

## MCMC turing code
@model function mse_model_censored(y, X, intercept_dist, beta_dist, gamma_dist, censoring_threshold)
    n_pars = size(X, 2)
    K = Int(-0.5 + (sqrt(8 * (n_pars - 1) + 1) / 2))  # Solve for K given length of params
    intercept ~ intercept_dist
    betas ~ filldist(beta_dist, K)
    gammas ~ filldist(gamma_dist, binomial(K, 2))
    
    params = vcat(intercept, betas, gammas)
    
    Turing.@addlogprob!(likelihood_censored(y, params, X, censoring_threshold))
end

## Neural network training
n_pars = 1 + K + binomial(K, 2)
n_data = K + binomial(K, 2)
w = 128  # width of each hidden layer

ψ = Chain(Dense(n_data * 2, w, relu), Dense(w, n_pars, relu))    
ϕ = Chain(Dense(n_pars, w, relu), Dense(w, n_pars, identity))     
network = DeepSet(ψ, ϕ)

# Initialise the estimator
estimator = PointEstimator(network)


# Number of independent replicates in each data set
m = 1

# Train an estimator with censoring
simulate_nbe(θ, m) = [generate_data_nbe(params, m) for params in eachcol(θ)]
sample_nbe(n_reps) = hcat([generate_parameters_nbe(K, intercept_dist, beta_dist, gamma_dist) for _ in 1:n_reps]...)
censor_threshold = 20
simulator_censored(θ, m) = simulatecensored_nbe(θ, m, c = log(censor_threshold))
estimator = train(estimator, sample_nbe, simulator_censored, m = m) 


## Now evaluate methods on test data
X = one_hot_encode(K)

test_df = DataFrame(
    intercept_true = Float64[],
    beta1_true = Float64[],
    beta2_true = Float64[],
    beta3_true = Float64[],
    intercept_nbe = Float64[],
    beta1_nbe = Float64[],
    beta2_nbe = Float64[],
    beta3_nbe = Float64[],
    intercept_mle = Float64[],
    beta1_mle = Float64[],
    beta2_mle = Float64[],
    beta3_mle = Float64[],
    intercept_mcmc = Float64[],
    beta1_mcmc = Float64[],
    beta2_mcmc = Float64[],
    beta3_mcmc = Float64[]
)

n_reps = 4
Folds.map(1:n_reps) do i
    θ = sample_nbe(1)
    Z_censored = simulatecensored_nbe(θ, 1, c = log(censor_threshold))
    Z_censored_int = Int.(round.(map(x -> x > 0 ? exp(x) : x, Z_censored[1][1:n_data])))
    NBE_estimate = vec(NeuralEstimators.estimate(estimator, Z_censored[1]))
    MLE = optimize(vars -> -1*likelihood_censored(Z_censored_int, vars, X, censor_threshold), fill(0.0, n_pars), LBFGS()).minimizer

    m = mse_model_censored(Z_censored_int, X, intercept_dist, beta_dist, gamma_dist, censor_threshold)
    num_chains = 4
    chains = sample(
        m, 
        NUTS(), 
        MCMCThreads(), 
        3000, 
        num_chains, 
        progress = false,
        parallel = false
    )

    res_mcmc = DataFrame(chains)

    DataFrame(
        intercept_true = θ[1], 
        beta1_true = θ[2], 
        beta2_true = θ[3], 
        beta3_true = θ[4],
        intercept_NBE = NBE_estimate[1], 
        beta1_NBE = NBE_estimate[2], 
        beta2_NBE = NBE_estimate[3], 
        beta3_NBE = NBE_estimate[4],
        intercept_MLE = MLE[1], 
        beta1_MLE = MLE[2], 
        beta2_MLE = MLE[3], 
        beta3 = MLE = MLE[4],
        intercept_MCMC = median(res_mcmc[!, :intercept]), 
        beta1_MCMC = median(res_mcmc[!, "betas[1]"]), 
        beta2_MCMC = median(res_mcmc[!, "betas[2]"]), 
        beta3_MCMC = median(res_mcmc[!, "betas[3]"])
    )
end

CSV.write(joinpath("output", "censoring_comparison_K$(K)_c$(censor_threshold).csv"), test_df)




y = [DataFrame((1, 2, 3)), DataFrame((4, 5, 3))]

DataFrame(
    y = 1,
    x = 4
)