using Pkg; Pkg.activate(".")
using Distributions, Random, Optim, DataFrames, Plots, CSV, Combinatorics, Turing, LinearAlgebra, StatsPlots, NeuralEstimators, Flux, Folds

include("mse_functions.jl")

function run_comparison(K, m, intercept_dist, beta_dist, gamma_dist; n_reps = 100, savepath = nothing)
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
        m = m
    )

    θ_test = sample_nbe(n_reps)
    Z_test = simulate_nbe(θ_test, 1)
    assessment = assess(estimator, θ_test, Z_test, probs = [0.025, 0.975])
    output_df = select(assessment[1], [:k, :parameter, :estimate, :truth])
    rename!(output_df, :estimate => :nbe_estimate)

    ## Now use MCMC to infer parameters
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

    num_chains = 4
    output_list = []
    X = one_hot_encode(K)
    for k in 1:n_reps
        input_y = Int64.(floor.(exp.(vec(Z_test[k]))))
        m = mse_model(input_y, X, intercept_dist, beta_dist, gamma_dist)
        chains = sample(
            m, 
            NUTS(), 
            MCMCThreads(), 
            1000, 
            num_chains, 
            progress = false,
            parallel = false
        )


        res_mcmc = DataFrame(chains)
        intercept_loc = findfirst(==("intercept"), names(res_mcmc))
        push!(output_list, DataFrame(
            k = k,
            parameter = ["θ$i" for i in 1:n_pars],
            estimate = [median(res_mcmc[!, i]) for i in intercept_loc:intercept_loc + n_pars - 1],
        ))
    end

    leftjoin!(output_df, vcat(output_list...), on = [:k, :parameter])
    rename!(output_df, :estimate => :mcmc_estimate)
    CSV.write(savepath, output_df)
end

K = 3
intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)

ms = [10, 50, 100, 500]
Ks = [3, 4, 5, 6]

output_path = joinpath("output", "mcmc_nbe_comparison")
mkpath(output_path)

Folds.map(
    x -> run_comparison(x[1], x[2], intercept_dist, beta_dist, gamma_dist, n_reps = 100, savepath = joinpath(output_path, "comparison_K$(x[1])_m$(x[2]).csv")), 
    vec(collect(Base.product(Ks, ms)))
)
