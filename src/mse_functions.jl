using Distributions, NeuralEstimators, Flux, BSON, DataFrames, CSV

# Custom activation function for intercept scaling (Flux-style)
function intercept_scaling(x, a, b)
    return a .+ b .* sigmoid.(x)
end

function sample_parameters(
    K::Int; 
    intercept_dist = Uniform(1, 10), 
    beta_dist = Normal(0, 4), 
    gamma_dist = Normal(0, 1/5)
)
    while true
        intercept = rand(intercept_dist)
        betas = rand(beta_dist, K)
        gammas = rand(gamma_dist, binomial(K, 2))
        valid = true
        for i in 1:K-1
            for j in i+1:K
                if intercept + betas[i] + betas[j] >= 25
                    valid = false
                    break
                end
            end
            if !valid
                break
            end
        end
        if valid
            return Float32.(vcat(intercept, betas, gammas))
        end
    end
end

function simulate_data(params, m; censoring_threshold = 0)
    K = Int(-0.5 + (sqrt(8 * length(params) - 7) / 2))  # Solve for K given length of params
    intercept = params[1]
    betas = params[2:1+K]
    gammas = params[2+K:end]

    Z = zeros(K + binomial(K, 2), m)

    for i in 1:m
        main_counts = [rand(Poisson(exp(intercept + betas[j]))) for j in 1:K]
        pair_counts = [
            rand(Poisson(exp(intercept + betas[j] + betas[k] + gammas[binomial(K, 2) - binomial(K - j + 1, 2) + (k - j)])))
            for j in 1:K-1 for k in j+1:K
        ]
        Z[:, i] = vcat(main_counts, pair_counts)
    end

    if censoring_threshold > 0
        W = 1 * (Z .<= censoring_threshold)
        U = ifelse.(Z .<= censoring_threshold, -1.0, log.(Z .+ 1))
        return Float32.(vcat(U, W))
    end

    return Float32.(log.(Z .+ 1))  # Log-transform the counts
end

function construct_MLP(width::Int, n_hidden::Int, n_lists::Int, censoring::Bool = false, intercept_support = nothing)
    n_data = n_lists + binomial(n_lists, 2)
    n_pars = 1 + n_data  # intercept + betas + gammas
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
    n_data = n_lists + binomial(n_lists, 2)
    n_input = n_data * 2 + 1

    n_pars = 1 + n_data  # intercept + betas + gammas

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

function train_model_mlp(n_lists, width, n_hidden, train_size; m = 1, censoring_threshold = 0, savepath = nothing, intercept_dist = Uniform(1, 10))
    mdl_str = "model_$(n_lists)_$(width)_$(n_hidden)_$(censoring_threshold)_$(train_size).bson"

    intercept_support = ifelse(typeof(intercept_dist) <: Uniform, params(intercept_dist), nothing)

    sample_nbe(n_reps) = hcat([sample_parameters(n_lists, intercept_dist = intercept_dist) for _ in 1:n_reps]...)
    simulate_nbe(θ, m) = hcat([simulate_data(params, m, censoring_threshold = censoring_threshold) for params in eachcol(θ)]...)
    network = construct_MLP(width, n_hidden, n_lists, censoring_threshold > 0, intercept_support)
    estimator = PointEstimator(network)

    estimator = train(
        estimator, 
        sample_nbe, 
        simulate_nbe, 
        K = train_size,
        m = m
    )

    if !isnothing(savepath) 
        BSON.@save joinpath(savepath, mdl_str) estimator
    else
        return estimator
    end
    return
end

function load_model(n_lists, width, n_hidden, censoring_threshold, train_size, models_path)
    mdl_str = "model_$(n_lists)_$(width)_$(n_hidden)_$(censoring_threshold)_$(train_size).bson"
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
    n_lists = n_lists, 
    width = width, 
    n_hidden = n_hidden, 
    censoring_threshold = censoring_threshold, 
    train_size = train_size, 
    models_path = joinpath("output", "models")
)
    load_model(n_lists, width, n_hidden, censoring_threshold, train_size, models_path)
end


function load_test_data(test_path, list_size, censoring_threshold = 0)
    test_data = BSON.load(joinpath(test_path, "test_data_$list_size.bson"))
    if censoring_threshold > 0
        Z_test = test_data[:Z_test]
        W = 1 * (Z_test .<= log(censoring_threshold + 1))
        U = ifelse.(Z_test .<= log(censoring_threshold + 1), -1.0, Z_test)
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

    W = ifelse.(output .== "missing", 1.0, 0.0)
    U = ifelse.(output .== "missing", -1.0, output)
    U = parse.(Float64, string.(U))
    return Float32.(vcat([u > 0 ? log(u + 1) : u for u in U], W))
end