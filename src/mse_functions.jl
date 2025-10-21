using Distributions, NeuralEstimators, Flux, BSON, DataFrames, CSV, Combinatorics

function sample_parameters(
    K::Int; 
    intercept_dist = Uniform(1, 10), 
    beta_dist = Normal(0, 4), 
    gamma_dist = Normal(0, 1/5)
)
    intercept = rand(intercept_dist)
    betas = rand(beta_dist, K)
    gammas = rand(gamma_dist, binomial(K, 2))
    return Float32.(vcat(intercept, betas, gammas))
end

function enumerate_two_digit_numbers(K::Int)
    numbers = Vector{Int64}[]
    for i in 1:K-1
        for j in i+1:K
            push!(numbers, [i, j])
        end
    end
    return numbers
end

## Custom Poisson sampler to avoid overflow
## rough error point is around logλ = 43.669
function rpois(logλ; logλ_max = 43.0)
    if logλ > logλ_max  
        return rand(Poisson(exp(logλ_max)))
    else
        return rand(Poisson(exp(logλ)))
    end
end

function simulate_data(pars, m; censoring_lower = 0, censoring_threshold = 0)
    K = Int(-0.5 + (sqrt(8 * length(pars) - 7) / 2))  # Solve for K given length of pars
    intercept = pars[1]
    betas = pars[2:1+K]
    gammas = pars[2+K:end]

    Z = zeros(2^K - 1, m)
    lists = enumerate_all_combinations(K)
    γ_map = Dict(enumerate_two_digit_numbers(K) .=> collect(1:length(gammas)))
    for j in 1:m
        for (i, list) in enumerate(lists)
            logλ = intercept
            digits, digit_pairs = compute_digit_pairs(list)
            for digit in digits
                logλ += betas[digit]
            end

            for pair in digit_pairs
                logλ += gammas[γ_map[pair]]
            end
            Z[i, j] = rpois(logλ)
        end
    end

    if censoring_threshold > 0
        W = 1 * (censoring_lower .<= Z .<= censoring_threshold)
        U = ifelse.(censoring_lower .<= Z .<= censoring_threshold, -1.0, log.(Z .+ 1))
        return Float32.(vcat(U, W))
    end

    return Float32.(log.(Z .+ 1))  # Log-transform the counts
end

function enumerate_all_combinations(K::Int64)
    combos = String[]
    for n in 1:K
        for c in combinations(1:K, n)
            push!(combos, join(c, ","))
        end
    end
    return combos
end

function compute_digit_pairs(n::String)
    n_split = split(n, ",")
    combinations(1, 2)
    if length(n_split) == 1
        return [parse(Int, n)], []
    end
    digits = [parse(Int, d) for d in n_split]
    pairs = collect(combinations(digits, 2))
    return digits, pairs
end

function construct_MLP(width::Int, n_hidden::Int, n_lists::Int, censoring::Bool = false, intercept_support = nothing)
    n_data = 2^n_lists - 1
    n_pars = 1 + n_lists + binomial(n_lists, 2)  # intercept + betas + gammas
    if censoring
        n_data *= 2  # Double the input size for censored data (U and W)
    end

    if !isnothing(intercept_support)
        a, b = Float32(intercept_support[1]), Float32(intercept_support[2])
        final_layer = Parallel(
            vcat,
            Chain(Dense(width, 1, identity), Compress(a, b)),  # Compress to the support of the uniform prior
            Dense(width, n_pars - 1, identity)  # Identity for betas and gammas
        )
    else
        final_layer = Dense(width, n_pars)
    end

    return Chain(
        Dense(n_data, width, relu),
        [Dense(width, width, relu) for _ in 1:n_hidden]...,
        final_layer
    )
end

function construct_MLP_c(width::Int, n_hidden::Int, n_lists::Int, intercept_support = nothing)
    n_data = 2^n_lists - 1
    n_pars = 1 + n_lists + binomial(n_lists, 2)  # intercept + betas + gammas
    n_input = n_data * 2 + 1

    if !isnothing(intercept_support)
        a, b = Float32(intercept_support[1]), Float32(intercept_support[2])
        final_layer = Parallel(
            vcat,
            Chain(Dense(width, 1, identity), Compress(a, b)),  # Compress to the support of the uniform prior
            Dense(width, n_pars - 1, identity)  # Identity for betas and gammas
        )
    else
        final_layer = Dense(width, n_pars)
    end

    return Chain(
        Dense(n_input, width, relu),
        [Dense(width, width, relu) for _ in 1:n_hidden]...,
        final_layer
    )
end

function train_model_mlp(n_lists, width, n_hidden, train_size; m = 1, censoring_lower = 0, censoring_threshold = 0, savepath = nothing, intercept_dist = Uniform(1, 10))
    estimator_mdl_str = "model_$(n_lists)_$(width)_$(n_hidden)_$(train_size)_$(censoring_lower)_$(censoring_threshold)_$(m).bson"
    ci_mdl_str = "model_ci_$(n_lists)_$(width)_$(n_hidden)_$(train_size)_$(censoring_lower)_$(censoring_threshold)_$(m).bson"

    intercept_support = ifelse(typeof(intercept_dist) <: Uniform, params(intercept_dist), nothing)

    sample_nbe(n_reps) = hcat([sample_parameters(n_lists, intercept_dist = intercept_dist) for _ in 1:n_reps]...)
    simulate_nbe(θ, m) = hcat([simulate_data(params, m, censoring_threshold = censoring_threshold) for params in eachcol(θ)]...)
    network = construct_MLP(width, n_hidden, n_lists, censoring_threshold > 0, intercept_support)
    estimator = PointEstimator(network)

    ci_estimator = IntervalEstimator(network)

    estimator = train(
        estimator, 
        sample_nbe, 
        simulate_nbe, 
        K = train_size,
        m = m
    )

    ci_estimator = train(
        ci_estimator, 
        sample_nbe, 
        simulate_nbe, 
        K = train_size,
        m = m
    )

    if !isnothing(savepath) 
        BSON.@save joinpath(savepath, estimator_mdl_str) estimator
        BSON.@save joinpath(savepath, ci_mdl_str) ci_estimator
        return nothing
    else
        return estimator
    end
end

function load_model(n_lists, width, n_hidden, train_size, censoring_lower, censoring_threshold, m, models_path)
    mdl_str = "model_$(n_lists)_$(width)_$(n_hidden)_$(train_size)_$(censoring_lower)_$(censoring_threshold)_$m.bson"
    if isfile(joinpath(models_path, mdl_str))
        model = BSON.load(joinpath(models_path, mdl_str))
        return model[:estimator]
    else
        error("Model file $(mdl_str) not found in $(models_path).")
    end
end

function load_model(mdl_str, models_path)
    if isfile(joinpath(models_path, mdl_str))
        model = BSON.load(joinpath(models_path, mdl_str))
        return model[:estimator]
    else
        error("Model file $(mdl_str) not found in $(models_path).")
    end
end

function load_model(; 
    n_lists, 
    width, 
    n_hidden, 
    train_size, 
    censoring_lower,
    censoring_threshold,
    m, 
    models_path = joinpath("output", "models")
)
    load_model(n_lists, width, n_hidden, train_size, censoring_lower, censoring_threshold, m, models_path)
end


function load_test_data(test_path, list_size, censoring_lower = 0, censoring_threshold = 0)
    test_data = BSON.load(joinpath(test_path, "test_data_$list_size.bson"))
    if censoring_threshold > 0
        Z_test = test_data[:Z_test]
        W = 1 * (log(censoring_lower + 1) .<= Z_test .<= log(censoring_threshold + 1))
        U = ifelse.(log(censoring_lower + 1) .<= Z_test .<= log(censoring_threshold + 1), -1.0, Z_test)
        return Float32.(vcat(U, W)), test_data[:params]
    end
    return test_data[:Z_test], test_data[:params]
end

function get_param_names(n_lists)
    return vcat(
        "intercept",
        ["beta_$(i)" for i in 1:n_lists],
        ["gamma_$(i),$(j)" for i in 1:(n_lists-1) for j in (i+1):n_lists]
    )
end

function load_silverman_data()
    silverman_file_path = joinpath("data", "silverman.csv")
    silverman_data = DataFrame(CSV.File(silverman_file_path))
    silverman_data.group = string.(silverman_data.group)
    K = maximum([maximum(parse.(Int, collect(filter(isdigit, g)))) for g in silverman_data.group])
    output = Vector{Int}()
    for i in 1:K
        grp_loc = findfirst(silverman_data.group .== "$i")
        if isnothing(grp_loc)
            push!(output, 0)
        else
            push!(output, silverman_data.count[grp_loc])
        end
    end

    for i in 1:K
        for j in i+1:K
            println("i = $i, j = $j")
            loc = findfirst(silverman_data.group .== "$i$j")
            if isnothing(loc)
                push!(output, 0)
            else
                push!(output, silverman_data.count[loc])
            end
        end
    end

    return output
end

function load_king_data()
    king_file_path = joinpath("data", "king.csv")
    king_data = DataFrame(CSV.File(king_file_path))
    king_data.group = string.(king_data.group)
    K = maximum([maximum(parse.(Int, collect(filter(isdigit, g)))) for g in king_data.group])
    output = []
    for i in 1:K
        grp_loc = findfirst(king_data.group .== "$i")
        if isnothing(grp_loc)
            push!(output, 0)
        else
            push!(output, king_data.count[grp_loc])
        end
    end

    for i in 1:K
        for j in i+1:K
            println("i = $i, j = $j")
            loc = findfirst(king_data.group .== "$i$j")
            if isnothing(loc)
                push!(output, 0)
            else
                push!(output, king_data.count[loc])
            end
        end
    end


    for (i, j, k) in enumerate_three_digit_numbers(K)
        loc = findfirst(king_data.group .== "$i$j$k")
        if isnothing(loc)
            push!(output, 0)
        else
            push!(output, king_data.count[loc])
        end
    end

    W = ifelse.(output .== "missing", 1.0, 0.0)
    U = ifelse.(output .== "missing", -1.0, output)
    U = parse.(Float64, string.(U))
    return Float32.(vcat([u > 0 ? log(u + 1) : u for u in U], W))
end


function one_hot_encode_parameters(K::Int)
    n_gamma = binomial(K, 2)
    n_pars = 1 + K + n_gamma
    one_hot_matrix = zeros(Int64, 2^K - 1, n_pars)
    lists = enumerate_all_combinations(K)
    γ_map = Dict(enumerate_two_digit_numbers(K) .=> collect(1:n_gamma))
    for (i, list) in enumerate(lists)
        digits, digit_pairs = compute_digit_pairs(list)
        one_hot_matrix[i, 1] = 1.0  # Intercept
        for digit in digits
            one_hot_matrix[i, 1 + digit] = 1.0
        end
        for pair in digit_pairs
            one_hot_matrix[i, 1 + K + γ_map[pair]] = 1.0
        end
    end

    return one_hot_matrix
end


function likelihood_censored(counts::Vector{Int64}, pars::Vector, X::Matrix{Int64}, censoring_lower::Int, censoring_threshold::Int)
    rates = exp.(X * pars)
    ll = 0.0
    for (rate, count) in zip(rates, counts)
        if count == -1
            # P(0 <= X <= censoring_threshold) = F(censoring_threshold; λ) - F(0; λ)
            for k in censoring_lower:censoring_threshold
                ll += k * log.(rate) .- rate 
            end
        else
            ll += count .* log.(rate) .- rate
        end
    end
    return ll
end