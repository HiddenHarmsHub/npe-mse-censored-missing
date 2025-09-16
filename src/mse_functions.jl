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



