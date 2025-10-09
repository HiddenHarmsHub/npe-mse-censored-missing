using Distributions, NeuralEstimators, Flux, BSON

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
    if censoring
        n_data *= 2  # Double the input size for censored data (U and W)
    end
    n_pars = 1 + n_data  # intercept + betas + gammas

    if isnothing(intercept_support)
        final_layer = Parallel(
            vcat,
            Dense(width, 1, x -> intercept_support[1] .+ intercept_support[2] .* sigmoid.(x)),  # Compress to the support of the uniform prior
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

function train_model_mlp(n_lists, width, n_hidden, train_size; m = 1, censoring_threshold = 0, savepath = nothing, overwrite = true, intercept_dist = Uniform(1, 10))
    mdl_str = "model_$(n_lists)_$(width)_$(n_hidden)_$(censoring_threshold)_$(train_size).bson"
    if !overwrite && savepath !== nothing && isfile(joinpath(savepath, mdl_str))
        println("Model already exists at $(joinpath(savepath, mdl_str)). Loading existing model.")
        BSON.@load joinpath(savepath, mdl_str) estimator
        return estimator
    end

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
    end

    return estimator
end
