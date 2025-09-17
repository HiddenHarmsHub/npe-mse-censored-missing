using Pkg; Pkg.activate(".")
using Distributions, Random, NeuralEstimators, Flux, StatsPlots, DataFrames, Optim
using BSON: @save, @load

include("mse_functions.jl")

K = 5
intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)

sample_nbe(n_reps) = hcat([generate_parameters_nbe(K, intercept_dist, beta_dist, gamma_dist) for _ in 1:n_reps]...)
simulate_nbe(θ, m) = [generate_data_nbe(params, m) for params in eachcol(θ)]

n_data = K + binomial(K, 2)
n_pars = 1 + n_data

load_estimator = true

if load_estimator
    @load joinpath("output", "nbe_model_final.bson") estimator
    println("Loaded existing estimator")
else
    w = 128  # width of each hidden layer 

    # Inner and outer networks
    ψ = Chain(Dense(n_data, w, relu), Dense(w, n_pars, relu))
    ϕ = Chain(Dense(n_pars, w, relu), Dense(w, n_pars))

    # Combine into a DeepSet
    network = DeepSet(ψ, ϕ)

    estimator = PointEstimator(network)

    m = 50
    estimator = train(
        estimator, 
        sample_nbe, 
        simulate_nbe, 
        m = m,
        K = 100_000
    )

    @save joinpath("output", "nbe_model_final.bson") estimator
end


## validation checking
θ_test = sample_nbe(1000)
Z_test = simulate_nbe(θ_test, 50)
assessment = assess(estimator, θ_test, Z_test, probs = [0.025, 0.975])


## check on a fixed parameter set
θ_fixed = sample_nbe(1)
n_reps = 1000
nbe_estimates = map(1:n_reps) do i
    Z_fixed = simulate_nbe(θ_fixed, 500)
    return vec(NeuralEstimators.estimate(estimator, Z_fixed))
end |> (y -> hcat(y...))

plot_nbe_estimates(nbe_estimates, θ_fixed, param_max = 6)

NeuralEstimators.estimate(estimator, Z_fixed)

## look at error estimates over a wide range
n_sims = 1000
par_names = ["intercept"; ["beta[$i]" for i in 1:K]; ["gamma[$i,$j]" for i in 1:K-1 for j in i+1:K]]

nbe_coverage_df = map(1:n_sims) do i
    θ_fixed = sample_nbe(1)
    Z_fixed = simulate_nbe(θ_fixed, 1)
    θ_hat = vec(NeuralEstimators.estimate(estimator, Z_fixed))
    return DataFrame(
        par = par_names,
        estimate = θ_hat,
        truth = vec(θ_fixed),
        error = θ_hat - vec(θ_fixed),
        relative_error = (θ_hat - vec(θ_fixed)) ./ abs.(vec(θ_fixed))
    )
end |> (x -> vcat(x...))


plot_list = map(unique(nbe_coverage_df.par)) do parname
    @df filter(x -> x.par == parname, nbe_coverage_df) scatter(:estimate, :truth, xlabel = "Estimate", ylabel = "Truth", title = "$parname", legend = false)
    plot!(xlims=xlims(), ylims=ylims())
    plot!(range(-15,15,length = 2001), range(-15,15,length = 2001), color = :red, label = "y=x")
end

plot(plot_list..., layout = (4, 4), size = (1200, 900))

@df filter(x -> x.par == "intercept", nbe_coverage_df) scatter(:estimate, :error, xlabel = "Estimate", ylabel = "Error", title = "NBE Parameter Estimates")

## try in the real data

bernard_dict = load_bernard_data(joinpath("data", "silverman.csv"), 5)
bernard_data = reshape(Float32.(get_bernard_npe(bernard_dict, K)), :, 1)
bernard_estimate = NeuralEstimators.estimate(estimator, log.(bernard_data .+ 1))

println("Bernard NBE estimate: $(exp(bernard_estimate[1]))")

DataFrame(
    par = par_names,
    estimate = vec(bernard_estimate)
) |> CSV.write(joinpath("output", "bernard_nbe_estimate.csv"))