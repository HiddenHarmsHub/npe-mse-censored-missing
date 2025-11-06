using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")


n_lists = 5
width = 128
n_hidden = 3
encoding_dim = 32
train_size = 10000
censoring_lower = 0
censoring_threshold = 10
m = 1
train_npe(n_lists, width, n_hidden, encoding_dim, train_size)


@everywhere function train_npe(n_lists, width, n_hidden, encoding_dim, train_size; m = 1, censoring_lower = 0, censoring_threshold = 0, savepath = nothing, intercept_dist = Uniform(1, 10))
    estimator_mdl_str = "model_$(n_lists)_$(width)_$(n_hidden)_$(encoding_dim)_$(train_size)_$(censoring_lower)_$(censoring_threshold)_$(m).bson"
    n_data = 2^n_lists - 1
    n_pars = 1 + n_lists + binomial(n_lists, 2)  # intercept + betas + gammas
    if censoring_threshold > 0
        n_data *= 2  # Double the input size for censored data (U and W)
    end

    sample_nbe(n_reps) = hcat([sample_parameters(n_lists, intercept_dist = intercept_dist) for _ in 1:n_reps]...)
    simulate_nbe(θ, m) = hcat([simulate_data(params, m, censoring_lower = censoring_lower, censoring_threshold = censoring_threshold) for params in eachcol(θ)]...)
    
    network = Chain(
        Dense(n_data, width, relu),
        [Dense(width, width, relu) for _ in 1:n_hidden]...,
        Dense(width, encoding_dim)
    )
    
    
    
    q = NormalisingFlow(n_pars, encoding_dim)
    estimator = PosteriorEstimator(q, network)
    
    estimator = train(
        estimator, 
        sample_nbe, 
        simulate_nbe, 
        K = train_size,
        m = m,
        epochs = 200
    )
    
    if !isnothing(savepath) 
        BSON.@save joinpath(savepath, estimator_mdl_str) estimator
        return nothing
    else
        return estimator
    end
end


grid_npe = collect(Base.product(
    [5],
    [8, 16, 32, 64, 128, 256],
    [1, 2, 3, 4],
    [8, 16, 32, 64, 128],
    [10000],
    [0],
    [10],
    [1]
))[:]

output_path = joinpath("output", "models_npe")
mkpath(output_path)

pmap(
    model -> begin
        #wid = myid()
        #println("Worker $wid running with parameters: list_size=$(model[1]), width=$(model[2]), n_hidden=$(model[3]), train_size=$(model[4]), censoring_lower=$(model[5]) censoring_threshold=$(model[6]), m=$(model[7])")
        train_npe(
            model[1],
            model[2],
            model[3], 
            model[4],
            model[5], 
            censoring_lower = model[6],
            censoring_threshold = model[7],
            m = model[8],
            savepath = output_path
        )
    end,
    grid_npe
)