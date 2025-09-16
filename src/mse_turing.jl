using Pkg; Pkg.activate(".")
using Distributions, Random, Optim, DataFrames, Plots, CSV, Combinatorics, Turing, LinearAlgebra, StatsPlots

function generate_parameters(K::Int, intercept_dist, beta_dist, gamma_dist)
    intercept = rand(intercept_dist)
    betas = rand(beta_dist, K)
    gammas = rand(gamma_dist, binomial(K, 2))
    par_names = vcat(
        "intercept", 
        ["beta_$i" for i in 1:K], 
        ["gamma_$(i),$(j)" for i in 1:K-1 for j in i+1:K]
    )
    par_values = vcat(
        intercept,
        betas,
        gammas
    )
    return(
        name = par_names,
        value = par_values,
    )
end

function simulate_data(par_names::Vector{String}, par_values::Vector{Float64})
    K = Int(-0.5 + (sqrt(8 * length(par_values) - 7) / 2))  # Solve for K given length of params

    params = Dict(par_names .=> par_values)

    main_counts = [
        rand(Poisson(exp(params["intercept"] + params["beta_$i"]))) for i in 1:K
    ]
    
    pair_counts = [
        rand(Poisson(exp(params["intercept"] + params["beta_$i"] + params["beta_$j"] + params["gamma_$(i),$(j)"]))) for i in 1:K-1 for j in i+1:K
    ]
    
    count_names = vcat(
        ["N_$i" for i in 1:K], 
        ["N_$(i),$(j)" for i in 1:K-1 for j in i+1:K]
    )
    
    return (
        count_names = count_names,
        counts = vcat(main_counts, pair_counts)
    )
end

function loglikelihood(data::Dict{String, Int64}, params::Dict{String, Float64})
    ll = 0.0
    for (k, v) in data
        numbers = parse.(Int, split(replace(k, r"[^\d,]" => ""), ","))
        rate = params["intercept"]
        for i in numbers
            rate += params["beta_$i"]
        end
        # Add all pairwise gamma terms for combinations of length >= 2
        if length(numbers) >= 2
            rate += params["gamma_$(numbers[1]),$(numbers[2])"]
        end
        ll += logpdf(Poisson(exp(rate)), v)
    end
    return ll
end

function loglikelihood(data::Vector{Int64}, pars::Vector{Float64}, X::Matrix{Int64})
    return sum(logpdf.(Poisson.(exp.(X * pars)), data))
end

function one_hot_encode(par_names, count_names)
    n_pars = length(par_names)
    X = zeros(Int64, n_pars - 1, n_pars)
    X[:, findfirst(==("intercept"), par_names)] .= 1
    for (i, count_name) in enumerate(count_names)
        numbers = parse.(Int, split(replace(count_name, r"[^\d,]" => ""), ","))    
        for j in numbers
            col_idx = findfirst(==("beta_$j"), par_names)
            println("Setting X[$i, $col_idx] = 1 for count $count_name with numbers $numbers")
            X[i, col_idx] = 1
        end
        if length(numbers) >= 2
            col_idx = findfirst(==("gamma_$(numbers[1]),$(numbers[2])"), par_names)
            X[i, col_idx] = 1
        end
    end
    return X
end

function loglikelihood(data::Vector{Int64}, pars::Vector{Float64}, X::Matrix{Int64})
    return sum(logpdf.(Poisson.(exp.(X * pars)), data))
end


K = 3

intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)

par_names, par_values = generate_parameters(K, intercept_dist, beta_dist, gamma_dist)
count_names, values = simulate_data(par_names, par_values)

X = one_hot_encode(par_names, count_names)

ll = loglikelihood(values, par_values, X)

@model function mse_model(y, X, par_names)
    n_pars = length(par_names)
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
m = mse_model(values, X, par_names)
chains = sample(m, NUTS(), MCMCThreads(), 1000, num_chains; progress=true)
describe(chains)
plot(chains)




### try real data

function load_bernard_data(file_path::String)
    bernard_data = DataFrame(CSV.File(file_path))
    bernard_data.group = string.(bernard_data.group)
    bernard_dict = construct_data(bernard_data, K)
    bernard_dict = Dict(k => v for (k, v) in bernard_dict if length(parse.(Int, collect(replace(k, r"[^\d]" => "")))) ≤ 2)
    return bernard_dict
end

function construct_data(df::DataFrame, K::Int)
    all_combinations = [collect(comb) for k in 1:K for comb in combinations(1:K, k)]
    count_names = ["N_" * join(comb, ",") for comb in all_combinations]
    counts = Vector{Int64}()
    for combination in all_combinations
        data_entry = df.count[df.group .== join(combination, "")]
        push!(counts, isempty(data_entry) ? 0 : data_entry[1])
    end

    return Dict(zip(count_names, counts))
end

K = 4
num_chains = 4
bernard_dict = load_bernard_data(joinpath("data", "silverman.csv"))
y_bernard = collect(values(bernard_dict))
count_names_bernard = collect(keys(bernard_dict))

par_names, _ = generate_parameters(K, intercept_dist, beta_dist, gamma_dist)
X_bernard = one_hot_encode(par_names, count_names_bernard)

m_bernard = mse_model(y_bernard, X_bernard, par_names)
chains = sample(m_bernard, NUTS(), MCMCThreads(), 5000, num_chains; progress=true)
plot(chains)

res = DataFrame(chains)

plot(res[:, Symbol("intercept")], seriestype=:histogram, title="Posterior of Intercept", xlabel="Intercept", ylabel="Density")


function posterior_predictive(res, X, K)
    n_pars = 1 + K + binomial(K, 2)
    parameter_matrix = X * Matrix(res[:, 3:3+n_pars-1])'
    PPD = hcat([rand.(Poisson.(exp.(parameter_matrix[:, i]))) for i in axes(parameter_matrix, 2)]...)
    return PPD
end

function plot_ppd(ppd, observed_counts; plotnames = "")
    n_samples = size(ppd, 2)
    n_counts = size(ppd, 1)
    p = plot(layout = (ceil(Int, n_counts / 3), 3), size=(900, 300 * ceil(Int, n_counts / 3)))
    for i in 1:n_counts
        density!(p[i], ppd[i, :], bins=30, title=plotnames[i], xlabel="Value", ylabel="Density", label="PPD", alpha=0.9, normalize=true)
        vline!(p[i], [observed_counts[i]], color=:red, label="Observed", lw=2)
    end
    return p
end

PPD = posterior_predictive(res, X, K)
p = plot_ppd(PPD, y_bernard, plotnames = count_names_bernard)
