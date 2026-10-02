## Freeze the base NPE architecture for the two model-selection systems (A: K=5 uncensored,
## B: K=4 censored on [1, 4]) by scoring the trained grid_5 / grid_4 NPE models on fresh
## validation sets. The selection rule is fixed here, before any model-selection result is seen.
using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")
@everywhere include("model_selection_functions.jl")

output_path = joinpath("output", "architecture_selection")
mkpath(joinpath(output_path, "metrics"))
npe_models_path = joinpath("output", "models_npe")

n_validation = 5000
n_posterior_draws = 2000
widths = [8, 16, 32, 64, 128, 256]
hidden = [1, 2, 3, 4]
@everywhere encoding_dim = 128
@everywhere train_size = 10000

## Pre-specified rule: smallest intercept MAE among architectures with 95% intercept coverage
## within 0.02 of nominal and mean calibration deviation (all parameters, 50/80/95%) at most 0.03.
## If none qualifies, the smallest mean calibration deviation. Ties go to fewer weights.
const selection_rule = (cov95_tolerance = 0.02, calibration_tolerance = 0.03)

## Validation sets from the fixed all-interactions prior the grid models were trained on (seed 7)
validation_file(system) = joinpath(output_path, "validation_$(system.system).bson")
for system in values(model_selection_systems)
    isfile(validation_file(system)) && continue
    Random.seed!(system.system == "A" ? 7 : 8)
    params = reduce(hcat, [sample_parameters(system.K) for _ in 1:n_validation])
    Z_counts = Int.(reduce(hcat, [simulate_data(p, 1, log_transform = false) for p in eachcol(params)]))
    N0 = [rpois(Float64(a)) for a in params[1, :]]
    BSON.@save validation_file(system) params Z_counts N0
end

@everywhere function score_architecture(system, width, n_hidden, npe_models_path, validation_file, output_path, n_posterior_draws)
    out = joinpath(output_path, "metrics", "metrics_$(system.system)_$(width)_$(n_hidden).csv")
    isfile(out) && return
    val = BSON.load(validation_file)
    params, N0 = val[:params], val[:N0]
    counts_obs = censor_counts(val[:Z_counts], system.censoring_lower, system.censoring_upper)
    y = encode_counts(counts_obs; censored = system.censoring_upper > 0)
    estimator = load_model_npe(system.K, width, n_hidden, train_size, system.censoring_lower, system.censoring_upper, 1, npe_models_path)
    keep = findall(.!is_capped.(eachcol(params)))

    Random.seed!(width + 100 * n_hidden)
    n_pars = size(params, 1)
    ranks = zeros(n_pars, length(keep))
    alpha_median, N0_median = zeros(length(keep)), zeros(length(keep))
    elapsed = @elapsed for (j, i) in enumerate(keep)
        draws = boundedsampleposterior(estimator, y[:, i:i], n_posterior_draws, 1.0, 10.0)
        ranks[:, j] = [mean(d .< t) for (d, t) in zip(eachrow(draws), params[:, i])]
        alpha_median[j] = median(draws[1, :])
        N0_median[j] = median([rpois(Float64(a)) for a in draws[1, :]])
    end
    levels = [0.5, 0.8, 0.95]
    coverage = [mean(abs.(ranks[p, :] .- 0.5) .<= L / 2) for p in 1:n_pars, L in levels]
    CSV.write(out, DataFrame(
        system = system.system, n_lists = system.K, width = width, n_hidden = n_hidden, encoding_dim = encoding_dim,
        train_size = train_size, censoring_lower = system.censoring_lower, censoring_upper = system.censoring_upper,
        n_datasets = length(keep),
        intercept_mae = mean(abs.(alpha_median .- params[1, keep])),
        log_N0_mae = mean(abs.(log.(N0_median .+ 1) .- log.(N0[keep] .+ 1))),
        cov50_alpha = coverage[1, 1], cov80_alpha = coverage[1, 2], cov95_alpha = coverage[1, 3],
        calibration_deviation = mean(abs.(coverage .- levels')),
        max_calibration_deviation = maximum(abs.(coverage .- levels')),
        seconds_per_dataset = elapsed / length(keep),
        n_weights = sum(length, Flux.params(estimator))
    ))
end

tasks = [(system, w, h, npe_models_path, validation_file(system), output_path, n_posterior_draws) for system in values(model_selection_systems) for w in widths for h in hidden
         if isfile(joinpath(npe_models_path, "model_$(system.K)_$(w)_$(h)_$(encoding_dim)_$(train_size)_$(system.censoring_lower)_$(system.censoring_upper)_1.bson"))]
println("Scoring $(length(tasks)) architectures")
pmap(t -> score_architecture(t...), tasks)

metrics = vcat([CSV.read(joinpath(output_path, "metrics", f), DataFrame) for f in readdir(joinpath(output_path, "metrics"))]...)
metrics.meets_rule = (abs.(metrics.cov95_alpha .- 0.95) .<= selection_rule.cov95_tolerance) .&
    (metrics.calibration_deviation .<= selection_rule.calibration_tolerance)
CSV.write(joinpath(output_path, "metrics.csv"), sort(metrics, [:system, :width, :n_hidden]))

selected = map(collect(groupby(metrics, :system))) do g
    candidates = any(g.meets_rule) ? sort(g[g.meets_rule, :], [:intercept_mae, :n_weights]) : sort(g, [:calibration_deviation, :n_weights])
    candidates[1, :]
end |> DataFrame
selected.m .= 1
selected.model_file = ["model_$(r.n_lists)_$(r.width)_$(r.n_hidden)_$(r.encoding_dim)_$(r.train_size)_$(r.censoring_lower)_$(r.censoring_upper)_1.bson" for r in eachrow(selected)]
CSV.write(joinpath(output_path, "selected.csv"), selected)
println(selected)
