## Pilot: does a larger training budget and flow improve NPE calibration?
## Retrains the K=5, censoring [0, 10] NPE and evaluates it and the shipped model on the reference benchmark
## datasets. Run locally from the repo root with e.g. julia --project -t 12 src/npe_pilot.jl
## Needs output/test_data/test_data_5.bson and the combined reference results (combine_results.jl).
include("mse_functions.jl")
using Random

n_lists, width, n_hidden, encoding_dim = 5, 256, 3, 128
censoring_lower, censoring_upper = 0, 10
n_samples = 20_000

pilot_path = joinpath("output", "npe_pilot")
mkpath(pilot_path)
pilot_settings = (
    epochs = 100, stopping_epochs = 15, K_val = 20_000, batchsize = 256, learning_rate = 1e-3,
    num_coupling_layers = 10, input_scale = 0.1
)
train_size = 200_000
pilot_file = joinpath(pilot_path, "model_pilot.bson")

if !isfile(pilot_file)
    Random.seed!(1)
    estimator = train_npe(
        n_lists, width, n_hidden, encoding_dim, train_size;
        censoring_lower, censoring_upper, pilot_settings..., training_path = joinpath(pilot_path, "training")
    )
    BSON.@save pilot_file estimator
end

models = Dict(
    "shipped" => load_model_npe(n_lists, width, n_hidden, 10000, censoring_lower, censoring_upper, 1, joinpath("output", "models_npe")),
    "pilot" => BSON.load(pilot_file)[:estimator]
)

## Posterior draws for every benchmark dataset, summarised by the fractional rank of the truth
test_data, test_pars = load_test_data(joinpath("output", "test_data"), n_lists, censoring_lower, censoring_upper)
benchmark = CSV.read(joinpath("output", "benchmark_datasets.csv"), DataFrame)
par_names = param_names(n_lists)

evaluation_file = joinpath(pilot_path, "npe_pilot_evaluation.csv")
if !isfile(evaluation_file)
    rows = DataFrame[]
    for (name, model) in models, dataset in benchmark.dataset
        Random.seed!(dataset)
        draws = boundedsampleposterior(model, reshape(test_data[:, dataset], :, 1), n_samples, 1.0, 10.0)
        truth = test_pars[:, dataset]
        push!(rows, DataFrame(
            model = name,
            dataset = dataset,
            parameters = par_names,
            true_values = truth,
            median = vec(median(draws, dims = 2)),
            lower_95ci = quantile.(eachrow(draws), 0.025),
            upper_95ci = quantile.(eachrow(draws), 0.975),
            rank = [mean(row .< t) for (row, t) in zip(eachrow(draws), truth)]
        ))
    end
    CSV.write(evaluation_file, vcat(rows...))
end

## Calibration on non-capped datasets, accuracy of the hidden population, and agreement with validated NUTS
evaluation = leftjoin(CSV.read(evaluation_file, DataFrame), select(benchmark, :dataset, :capped), on = :dataset)
mcmc = CSV.read(joinpath("output", "mcmc_summary.csv"), DataFrame)
status = CSV.read(joinpath("output", "mcmc_diagnostics.csv"), DataFrame)
validated = Set(status.dataset[status.status .!= "failed"])

calibrated = filter(:capped => !, evaluation)
n_datasets = length(unique(calibrated.dataset))
for name in ["shipped", "pilot"], level in [0.5, 0.8, 0.95]
    band = 1.96 * sqrt(level * (1 - level) / n_datasets)
    d = filter(:model => ==(name), calibrated)
    inside = abs.(d.rank .- 0.5) .<= level / 2
    per_parameter = [mean(inside[d.parameters .== p]) for p in par_names]
    println(rpad(name, 8), " level $level: alpha coverage ", round(per_parameter[1], digits = 3),
        ", parameters outside ±", round(band, digits = 3), ": ", sum(abs.(per_parameter .- level) .> band), "/", length(par_names))
end

for name in ["shipped", "pilot"]
    d = filter(r -> r.model == name && r.parameters == "alpha", evaluation)
    error = exp.(d.median) .- exp.(d.true_values)
    ape = abs.(error) ./ exp.(d.true_values)
    m = innerjoin(d, filter(r -> r.parameters == "alpha" && r.dataset in validated, mcmc), on = :dataset, makeunique = true)
    println(rpad(name, 8), " N0 bias ", round(mean(error), digits = 1), " (se ", round(std(error) / sqrt(length(error)), digits = 1),
        "), MAPE ", round(mean(ape), digits = 3), ", mean |alpha median - NUTS median| ", round(mean(abs.(m.median .- m.median_mcmc)), digits = 3),
        ", mean 95% width ", round(mean(d.upper_95ci .- d.lower_95ci), digits = 3))
end
