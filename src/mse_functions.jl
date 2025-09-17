using CSV, DataFrames, Combinatorics

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
        value = par_values
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

function one_hot_encode(K::Int64)
    par_names = vcat(
        "intercept", 
        ["beta_$i" for i in 1:K], 
        ["gamma_$(i),$(j)" for i in 1:K-1 for j in i+1:K]
    )
    count_names = vcat(
        ["N_$i" for i in 1:K], 
        ["N_$(i),$(j)" for i in 1:K-1 for j in i+1:K]
    )
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

function generate_parameters_nbe(K::Int, intercept_dist, beta_dist, gamma_dist)
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

function generate_data_nbe(params, m)
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



function load_bernard_data(file_path::String, K::Int)
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

get_bernard_npe(bernard_dict, K) = vcat([bernard_dict["N_$i"] for i in 1:K], [bernard_dict["N_$(i),$(j)"] for i in 1:K-1 for j in i+1:K])



function compare_npe_mcmc_posteriors(res_mcmc, res_npe, truth; linelabel = "truth")
    main_plot_list = []
    p = density(res_mcmc[!, "intercept"], label="MCMC", title = "intercept")
    density!(p, res_npe[1, :], label="NPE")
    vline!([truth["intercept"]], color=:red, lw=3, label=linelabel)
    push!(main_plot_list, p)
    for i in 1:K
        p = density(res_mcmc[!, "betas[$i]"], label="MCMC", title = "beta $i")
        density!(p, res_npe[i+1, :], label="NPE")
        vline!([truth["beta_$i"]], color=:red, lw=3, label=linelabel)
        push!(main_plot_list, p)
    end

    gamma_plot_list = []
    for i in 1:binomial(K, 2)
        p = density(res_mcmc[!, "gammas[$i]"], label="MCMC", title = "gamma $i")
        density!(p, res_npe[i+1+K, :], label="NPE")
        push!(gamma_plot_list, p)
    end

    return main_plot_list, gamma_plot_list
end


function plot_nbe_estimates(nbe_estimates, θ_truth=nothing; param_max = 0)
    n_params = size(nbe_estimates, 1)
    if param_max != 0
        n_params = param_max
    end
    p = plot(layout = (n_params, n_params), size = (800, 800))
    
    for i in 1:n_params
        for j in 1:n_params
            if i == j
                histogram!(p[i, j], nbe_estimates[i, :], bins = 30, label = "", legend = false)
                if θ_truth !== nothing
                    vline!(p[i, j], [θ_truth[i]], color = :red, lw = 2, label = "Truth")
                end
            elseif i < j
                scatter!(p[i, j], nbe_estimates[j, :], nbe_estimates[i, :], ms=2, alpha=0.5, label = "", legend = false)
                if θ_truth !== nothing
                    scatter!(p[i, j], [θ_truth[j]], [θ_truth[i]], marker=(:x, 10, :red), label = "Truth", legend = false)
                end
            end
        end
    end
    display(p)
end