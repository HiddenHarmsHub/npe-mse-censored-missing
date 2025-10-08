using Distributions, NeuralEstimators, Flux

function sample_parameters(
    K::Int; 
    intercept_dist = Uniform(1, 10), 
    beta_dist = Normal(0, 4), 
    gamma_dist = Normal(0, 1/5)
)
    return vcat(
        rand(intercept_dist),
        rand(beta_dist, K),
        rand(gamma_dist, binomial(K, 2))
    )
end


function simulate_data(params, m; censoring_threshold = nothing)
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

    if !isnothing(censoring_threshold)
        W = 1 * (Z .<= censoring_threshold)
        U = ifelse.(Z .<= censoring_threshold, -1.0, log.(Z .+ 1))
        return Float32.(vcat(U, W))
    end

    return Float32.(log.(Z .+ 1))  # Log-transform the counts
end



function construct_MLP(width, n_hidden, K::Int, censoring::Bool = false)
    n_data = K + binomial(K, 2)
    if censoring
        n_data *= 2  # Double the input size for censored data (U and W)
    end
    n_pars = 1 + K + binomial(K, 2)  # intercept + betas + gammas

    return Chain(
        Dense(n_data, width, relu),
        [Dense(width, width, relu) for _ in 1:n_hidden]...,
        Dense(width, n_pars)
    )
end

function train_model_mlp(n_lists, width, n_hidden, train_size; m = 1, censoring_threshold = nothing)
    sample_nbe(n_reps) = hcat([sample_parameters(n_lists) for _ in 1:n_reps]...)
    simulate_nbe(θ, m) = hcat([simulate_data(params, m, censoring_threshold = censoring_threshold) for params in eachcol(θ)]...)
    network = construct_MLP(width, n_hidden, n_lists, false)
    estimator = PointEstimator(network)

    estimator = train(
        estimator, 
        sample_nbe, 
        simulate_nbe, 
        K = train_size,
        m = m
    )
    return estimator
end

#estimator = train_model_mlp(4, 128, 2, 10_000)

n_lists = 4
width = 128
n_hidden = 1
train_size = 10_000

estimator = train_model_mlp(n_lists, width, n_hidden, train_size)