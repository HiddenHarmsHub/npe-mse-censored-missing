using Pkg; Pkg.activate(".")
using Distributions, Random, Optim, DataFrames, Plots, CSV, Combinatorics

function generate_parameters(K::Int, intercept_dist, beta_dist, gamma_dist)
    intercept = rand(intercept_dist)
    betas = rand(beta_dist, K)
    gammas = rand(gamma_dist, binomial(K, 2))
    par_names = vcat(
        "intercept", 
        ["beta_$i" for i in 1:K], 
        ["gamma_$(i)$(j)" for i in 1:K-1 for j in i+1:K]
    )
    par_estimates = vcat(
        intercept,
        betas,
        gammas
    )
    return(Dict(zip(par_names, par_estimates)))
end

function simulate_data(params::Dict, K::Int)
    main_counts = [
        rand(Poisson(exp(params["intercept"] + params["beta_$i"]))) for i in 1:K
    ]
    
    pair_counts = [
        rand(Poisson(exp(params["intercept"] + params["beta_$i"] + params["beta_$j"] + params["gamma_$(i)$(j)"]))) for i in 1:K-1 for j in i+1:K
    ]
    
    count_names = vcat(
        ["N_$i" for i in 1:K], 
        ["N_$(i)$(j)" for i in 1:K-1 for j in i+1:K]
    )
    
    return(Dict(zip(count_names, vcat(main_counts, pair_counts))))
end


function loglikelihood(data::Dict{String, Int64}, params::Dict{String, Float64}, K::Int)
    ll = 0.0
    for i in 1:K
        lambda = exp(params["intercept"] + params["beta_$i"])
        ll += logpdf(Poisson(lambda), data["N_$i"])
    end
    for i in 1:K-1
        for j in i+1:K
            lambda = exp(params["intercept"] + params["beta_$i"] + params["beta_$j"] + params["gamma_$(i)$(j)"])
            ll += logpdf(Poisson(lambda), data["N_$(i)$(j)"])
        end
    end
    return(ll)
end

function log_prior_density(params::Dict{String, Float64}, K::Int, intercept_dist, beta_dist, gamma_dist)
    lp = logpdf(intercept_dist, params["intercept"])
    for i in 1:K
        lp += logpdf(beta_dist, params["beta_$i"])
    end
    for i in 1:K-1
        for j in i+1:K
            lp += logpdf(gamma_dist, params["gamma_$(i)$(j)"])
        end
    end
    return lp
end


function params_to_vector(params::Dict{String, Float64}, K::Int)
    vcat(
        params["intercept"],
        [params["beta_$i"] for i in 1:K],
        [params["gamma_$(i)$(j)"] for i in 1:K-1 for j in i+1:K]
    )
end

function vector_to_params(vec::Vector{Float64}, K::Int)
    idx = 1
    intercept = vec[idx]
    idx += 1
    betas = vec[idx:idx+K-1]
    idx += K
    gammas = vec[idx:end]
    par_names = vcat(
        "intercept", 
        ["beta_$i" for i in 1:K], 
        ["gamma_$(i)$(j)" for i in 1:K-1 for j in i+1:K]
    )
    par_estimates = vcat(
        intercept,
        betas,
        gammas
    )
    Dict(zip(par_names, par_estimates))
end

function neg_loglikelihood(vec, data, K)
    params = vector_to_params(vec, K)
    -loglikelihood(data, params, K)
end


function mcmc(
    data::Dict{String, Int64}, 
    initial_params::Dict{String, Float64}, 
    K::Int; 
    n_iter::Int = 10000, 
    step_size::Float64 = 0.1,
    burn_in::Int = 0
)
    current_params = copy(initial_params)
    current_ll = loglikelihood(data, current_params)
    current_logprior = log_prior_density(current_params, K, intercept_dist, beta_dist, gamma_dist)
    current_posterior = current_ll + current_logprior
    
    samples = Vector{Dict{String, Float64}}()
    
    for iter in 1:n_iter
        proposal_vec = params_to_vector(current_params, K) .+ randn(length(current_params)) .* step_size
        proposal_params = vector_to_params(proposal_vec, K)
        proposal_ll = loglikelihood(data, proposal_params)
        proposal_logprior = log_prior_density(proposal_params, K, intercept_dist, beta_dist, gamma_dist)
        proposal_posterior = proposal_ll + proposal_logprior
        
        if log(rand()) < (proposal_posterior - current_posterior)
            current_params = proposal_params
            current_ll = proposal_ll
        end
        
        if iter > burn_in
            push!(samples, copy(current_params))
        end
    end
    
    return(samples)
end


## Simulated data

K = 6
intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)

params = generate_parameters(K, intercept_dist, beta_dist, gamma_dist)
data = simulate_data(params, K)

loglik = loglikelihood(data, params, K)
println("Log-Likelihood: ", loglik)


initial_vec = params_to_vector(params, K)
result = optimize(v -> neg_loglikelihood(v, data, K), initial_vec, NelderMead())

opt_params = vector_to_params(Optim.minimizer(result), K)
println("Optimized parameters: ", opt_params)

df = DataFrame(
    Parameter = collect(keys(params)),
    TrueValue = collect(values(params)),
    EstimatedValue = [opt_params[k] for k in keys(params)]
);

df.error = df.EstimatedValue .- df.TrueValue;
println(df)


mcmc_samples = mcmc(data, params, K, n_iter = 20000, step_size = 0.01)


mcmc_df = DataFrame([Dict(k => s[k] for k in keys(params)) for s in mcmc_samples])
#println(mcmc_df)


plot(mcmc_df[!, "intercept"])



## Bernards data

function construct_data(df::DataFrame, K::Int)
    all_combinations = [collect(comb) for k in 1:K for comb in combinations(1:K, k)]
    count_names = ["N_" * join(comb, "") for comb in all_combinations]
    counts = Vector{Int64}()
    for combination in all_combinations
        data_entry = df.count[df.group .== join(combination, "")]
        push!(counts, isempty(data_entry) ? 0 : data_entry[1])
    end

    return Dict(zip(count_names, counts))
end

function loglikelihood(data::Dict{String, Int64}, params::Dict{String, Float64})
    ll = 0.0
    for (k, v) in data
        numbers = parse.(Int, collect(replace(k, r"[^\d]" => "")))
        rate = params["intercept"]
        for i in numbers
            rate += params["beta_$i"]
        end
        # Add all pairwise gamma terms for combinations of length >= 2
        if length(numbers) >= 2
            for comb in combinations(sort(numbers), 2)
                i, j = comb
                rate += params["gamma_$(i)$(j)"]
            end
        end
        ll += logpdf(Poisson(exp(rate)), v)
    end
    return ll
end


K = 5
bernard_data = DataFrame(CSV.File(joinpath("data", "silverman.csv")))
bernard_data.group = string.(bernard_data.group)
bernard_dict = construct_data(bernard_data, K)


intercept_dist = Uniform(5, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)

inital_params = generate_parameters(K, intercept_dist, beta_dist, gamma_dist)

mcmc_bernard = mcmc(bernard_dict, inital_params, K, n_iter = 1000000, step_size = 0.01, burn_in = 500000)
mcmc_bernard_df = DataFrame([Dict(k => s[k] for k in keys(inital_params)) for s in mcmc_bernard])

plot(mcmc_bernard_df[:, "intercept"])
plot(mcmc_bernard_df[:, "beta_1"])
scatter(mcmc_bernard_df[:, "intercept"], mcmc_bernard_df[:, "beta_1"])

plot(exp.(mcmc_bernard_df[:, "intercept"]))
median(exp.(mcmc_bernard_df[:, "intercept"]))

histogram(mcmc_bernard_df[:, "intercept"])