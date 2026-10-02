## Simulation study for model selection and model averaging: fixed prior-predictive test sets for
## systems A and B, evaluated with the oracle structure, all interactions, the MAP structure,
## neural model averaging (primary and reweighted sensitivity priors) and the fixed-model NPE
using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")
@everywhere include("model_selection_functions.jl")

train_size = 100_000
gamma_sd = 4
## Replicate 1 is evaluated on the full test set; the others on its first 2,000 datasets, for training variability
n_evaluate = Dict(1 => 10_000, 2 => 2000, 3 => 2000)
chunk_size = 100
n_draws = 4000

output_path = joinpath("output", "model_selection")
@everywhere results_path = joinpath("output", "model_selection", "simulation")
mkpath(results_path)
selected = CSV.read(joinpath("output", "architecture_selection", "selected.csv"), DataFrame)

## Test sets come from model_selection_test_data.jl
test_file(system) = joinpath(output_path, "test_data_$(system.system).bson")
for system in values(model_selection_systems)
    isfile(test_file(system)) || error("$(test_file(system)) not found; run src/model_selection_test_data.jl first.")
end

@everywhere chunk_file(config, rep, chunk, kind) = joinpath(results_path, "$(config.system)_rep$(rep)_chunk$(chunk)_$(kind).csv")

@everywhere function run_simulation_chunk(config, rep, chunk, chunk_size, test_file, fixed_model_file, n_draws)
    test = (; (k => v for (k, v) in BSON.load(test_file))...)
    idx = ((chunk - 1) * chunk_size + 1):min(chunk * chunk_size, size(test.y, 2))
    classifier = load_classifier(config, rep)
    cnpe = load_cnpe(config, rep)
    fixed_npe = BSON.load(fixed_model_file)[:estimator]
    Random.seed!(10^6 * rep + 10^4 * config.K + chunk)
    models, posteriors = evaluate_simulation_chunk(config, rep, idx, test, classifier, cnpe, fixed_npe; B = n_draws)
    CSV.write(chunk_file(config, rep, chunk, "posteriors"), posteriors)
    CSV.write(chunk_file(config, rep, chunk, "models"), models)  # written last: marks the chunk as done
end

## Each task carries everything the worker needs, so no master-only globals are referenced
tasks = []
for s in ["A", "B"], rep in sort(collect(keys(n_evaluate)))
    config = model_selection_config(model_selection_systems[s]; train_size, gamma_sd)
    row = only(eachrow(filter(r -> r.n_lists == config.K && r.censoring_upper == config.censoring_upper, selected)))
    fixed_model_file = joinpath("output", "models_npe", row.model_file)
    for chunk in 1:cld(n_evaluate[rep], chunk_size)
        isfile(chunk_file(config, rep, chunk, "models")) || push!(tasks, (config, rep, chunk, chunk_size, test_file(config), fixed_model_file, n_draws))
    end
end
println("Chunks to evaluate: ", length(tasks))

pmap(tasks) do task
    println("Worker $(myid()) evaluating system $(task[1].system), replicate $(task[2]), chunk $(task[3])")
    try
        run_simulation_chunk(task...)
    catch e
        msg = sprint(showerror, e, catch_backtrace())
        println("Chunk failed for system $(task[1].system), replicate $(task[2]), chunk $(task[3]):\n$msg")
        write(chunk_file(task[1], task[2], task[3], "error") * ".txt", msg)
    end
end

println("Finished the model-selection simulation study.")
