using Pkg; Pkg.activate(".")
using Distributions, Random, NeuralEstimators, Flux, StatsPlots, CSV, DataFrames, Turing, LinearAlgebra, Combinatorics, Printf
using BSON: @save, @load

include("mse_functions.jl")

K = 5
intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)

## Neural estimation
sample(n_reps) = hcat([generate_parameters_nbe(K, intercept_dist, beta_dist, gamma_dist) for _ in 1:n_reps]...)
simulate(θ, m) = [generate_data_nbe(params, m) for params in eachcol(θ)]

n_pars = 1 + K + binomial(K, 2)
n_data = K + binomial(K, 2)

q = NormalisingFlow(n_pars, n_pars) 

w = 128  # width of each hidden layer 
# Inner and outer networks
ψ = Chain(Dense(n_data, w, relu), Dense(w, n_pars, relu))    
ϕ = Chain(Dense(n_pars, w, relu), Dense(w, n_pars))          

# Combine into a DeepSet
network = DeepSet(ψ, ϕ)

load_estimator = true

if load_estimator
    @load joinpath("output", "npe_model_final.bson") estimator
else
    # Initialise the estimator
    estimator = PosteriorEstimator(q, network)

    m = 50
    # Train the estimator
    estimator = train(estimator, sample, simulate, m = m)
    @save joinpath("output", "npe_model_final.bson") estimator
end



## Define turing model for MCMC
@model function mse_model(y, X, intercept_dist, beta_dist, gamma_dist)
    n_pars = size(X, 2)
    K = Int(-0.5 + (sqrt(8 * (n_pars - 1) + 1) / 2))  # Solve for K given length of params
    intercept ~ intercept_dist
    betas ~ filldist(beta_dist, K)
    gammas ~ filldist(gamma_dist, binomial(K, 2))
    
    params = vcat(intercept, betas, gammas)
    
    for i in eachindex(y)
        y[i] ~ Poisson(exp(dot(X[i, :], params)))
    end
end

## Try on simulated data
par_names, par_values = generate_parameters(K, intercept_dist, beta_dist, gamma_dist)
count_names, count_values = simulate_data(par_names, par_values)
X = one_hot_encode(par_names, count_names)

m = mse_model(count_values, X, intercept_dist, beta_dist, gamma_dist)
num_chains = 4
out = Turing.sample(m, NUTS(), MCMCThreads(), 2000, num_chains; progress=true)
describe(out)
res = DataFrame(out)

plot(out)


NPE_samples = sampleposterior(estimator, reshape(log.(count_values .+ 1), : , 1), 2000)

main_plots, gamma_plots = compare_npe_mcmc_posteriors(res, NPE_samples, Dict(par_names .=> par_values))

plot(main_plots..., layout = (2, 3), size = (900, 600))
plot(gamma_plots..., layout = (3, 4), size = (900, 600))


## try on the real data
bernard_dict = load_bernard_data(joinpath("data", "silverman.csv"), 5)
y_bernard = collect(values(bernard_dict))
count_names_bernard = collect(keys(bernard_dict))

par_names, _ = generate_parameters(K, intercept_dist, beta_dist, gamma_dist)
X_bernard = one_hot_encode(par_names, count_names_bernard)


bernard_data = reshape(
    Float32.(get_bernard_npe(bernard_dict, K)), 
    :, 1
)  # Ensure it's a matrix with one column


post_samples = sampleposterior(estimator, log.(bernard_data .+ 1), 20000)

## Compare with turing

## Now use MCMC to infer parameters
m = mse_model(y_bernard, X_bernard, intercept_dist, beta_dist, gamma_dist)
mcmc_chains = Turing.sample(
    m, 
    NUTS(), 
    MCMCThreads(), 
    5000, 
    4, 
    progress = false,
    parallel = false
)


res = DataFrame(mcmc_chains)

dark_number_samples = exp.(res[:, Symbol("intercept")])
histogram(
    dark_number_samples,
    bins=30,
    title="Posterior predictive of Dark Number",
    xlabel="Dark Number",
    ylabel="Density",
    label="Posterior",
    alpha=0.9,
    normalize=true,
    xformatter = x -> @sprintf("%.0f", x)  # Disable scientific notation
)


MLE_pars = ["intercept", "beta_1", "beta_2", "beta_3", "beta_4", "beta_5"]
MLEs = [9.05, -5.09, -2.9, -2.1, -2.5, -3.3]

MLE_estimates = Dict(
    MLE_pars .=> MLEs
)


bernard_main_plot_list, bernard_gamma_plot_list = compare_npe_mcmc_posteriors(
    res, 
    post_samples, 
    MLE_estimates,
    linelabel = "MLE"
)
plot(bernard_main_plot_list..., layout = (2, 3), size = (900, 600))


dark_number_ppd_npe = rand.(Poisson.(exp.(post_samples[1, :])))
dark_number_ppd_mcmc = rand.(Poisson.(exp.(res[:, Symbol("intercept")])))

density(
    dark_number_ppd_npe,
    label = "NPE",
    xlabel = "Dark Number",
    ylabel = "Density",
    title = "Posterior Predictive of Dark Number",
    legend = :topright,
    normalize = true,
    alpha = 0.9,
    xformatter = x -> @sprintf("%.0f", x)  # Disable scientific notation
)
density!(
    dark_number_ppd_mcmc,
    label = "MCMC",
    alpha = 0.5
)