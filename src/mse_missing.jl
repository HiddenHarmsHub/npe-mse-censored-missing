## Compare various methods to estimate parameters for the censored MSE model

using Random, Distributions, Optim, Turing

include("mse_functions.jl")



function censor_data(data::Vector{Int64}, censoring_threshold::Int)
    censored_data = copy(data)
    for (i, value) in enumerate(data)
        if value <= censoring_threshold
            censored_data[i] = -1
        end
    end
    return censored_data
end



function likelihood_censored(counts::Vector{Int64}, pars::Vector, X::Matrix{Int64}, censoring_threshold::Int)
    rates = exp.(X * pars)
    ll = 0.0
    for (rate, count) in zip(rates, counts)
        if count == -1
            # P(0 <= X <= censoring_threshold) = F(censoring_threshold; λ) - F(0; λ)
            ll += log(cdf(Poisson(rate), censoring_threshold)) - log(cdf(Poisson(rate), 0))
        else
            ll += count .* log.(rate) .- rate
        end
    end
    return ll
end

intercept_dist = Uniform(5, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)
K = 5

censoring_threshold = 1000
par_names, par_values = generate_parameters(K, intercept_dist, beta_dist, gamma_dist)
data_names, counts = simulate_data(par_names, par_values)
censored_data = censor_data(counts, censoring_threshold)
X = one_hot_encode(K)
objective_function = vars -> -1*likelihood_censored(censored_data, vars, X, censoring_threshold)
objective_function2 = vars -> -1*likelihood_censored2(censored_data, vars, X, censoring_threshold)
result = optimize(objective_function, par_values, LBFGS())



result.minimizer


## try MCMC

@model function mse_model_censored(y, X, intercept_dist, beta_dist, gamma_dist, censoring_threshold)
    n_pars = size(X, 2)
    K = Int(-0.5 + (sqrt(8 * (n_pars - 1) + 1) / 2))  # Solve for K given length of params
    intercept ~ intercept_dist
    betas ~ filldist(beta_dist, K)
    gammas ~ filldist(gamma_dist, binomial(K, 2))
    
    params = vcat(intercept, betas, gammas)
    
    Turing.@addlogprob!(likelihood_censored(y, params, X, censoring_threshold))
end


m = mse_model_censored(counts, X, intercept_dist, beta_dist, gamma_dist, censoring_threshold)
num_chains = 4
chains = sample(
    m, 
    NUTS(), 
    MCMCThreads(), 
    1000, 
    num_chains, 
    progress = false,
    parallel = false
)



## now try custom distributions



struct CensoredPoisson <: DiscreteUnivariateDistribution
    λ::Float64
    threshold::Int
end

