## Simulation study for model selection and model averaging: fixed prior-predictive test sets for
## systems A and B, evaluated with the oracle structure, all interactions, the MAP structure,
## neural model averaging (primary and reweighted sensitivity priors) and the fixed-model NPE.
## The run (priors and training settings) comes from MS_RUN; see model_selection_runs.
using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")
@everywhere include("model_selection_functions.jl")

run = model_selection_run()
## Replicate 1 and the ensemble of replicates 1-3 (rep 0) are evaluated on the full test set; replicates 2
## and 3 on its first 2,000 datasets, for training variability
n_evaluate = Dict(0 => 10_000, 1 => 10_000, 2 => 2000, 3 => 2000)
chunk_size = 100
n_draws = 4000

results_path = joinpath(model_selection_output_path(run), "simulation")
mkpath(results_path)
selected = CSV.read(joinpath("output", "architecture_selection", "selected.csv"), DataFrame)

## Test sets come from model_selection_test_data.jl
for system in values(model_selection_systems)
    isfile(model_selection_test_file(run, system)) || error("$(model_selection_test_file(run, system)) not found; run src/model_selection_test_data.jl first.")
end

@everywhere chunk_file(results_path, config, rep, chunk, kind) = joinpath(results_path, "$(config.system)_rep$(rep)_chunk$(chunk)_$(kind).csv")

@everywhere function run_simulation_chunk(config, rep, chunk, chunk_size, test_file, fixed_model_file, n_draws, results_path)
    test = (; (k => v for (k, v) in BSON.load(test_file))...)
    idx = ((chunk - 1) * chunk_size + 1):min(chunk * chunk_size, size(test.y, 2))
    classifier = load_classifier(config, rep)
    cnpe = load_cnpe(config, rep)
    fixed_npe = isnothing(fixed_model_file) ? nothing : BSON.load(fixed_model_file)[:estimator]
    Random.seed!(10^6 * rep + 10^4 * config.K + chunk)
    models, posteriors = evaluate_simulation_chunk(config, rep, idx, test, classifier, cnpe, fixed_npe; B = n_draws)
    CSV.write(chunk_file(results_path, config, rep, chunk, "posteriors"), posteriors)
    CSV.write(chunk_file(results_path, config, rep, chunk, "models"), models)  # written last: marks the chunk as done
end

## Each task carries everything the worker needs, so no master-only globals are referenced.
## The fixed-model NPE was trained under the original priors, so it is compared only in runs that use them.
tasks = []
for s in ["A", "B"], rep in sort(collect(keys(n_evaluate)))
    config = model_selection_config(model_selection_systems[s], run)
    row = only(eachrow(filter(r -> r.n_lists == config.K && r.censoring_upper == config.censoring_upper, selected)))
    fixed_model_file = original_priors(run) ? joinpath("output", "models_npe", row.model_file) : nothing
    ## The comparison model is optional: if it is missing (e.g. the base models were retrained under another
    ## name), the other methods still run
    if !isnothing(fixed_model_file) && !isfile(fixed_model_file)
        @warn "Fixed-structure NPE $fixed_model_file not found; system $s is evaluated without it."
        fixed_model_file = nothing
    end
    for chunk in 1:cld(n_evaluate[rep], chunk_size)
        isfile(chunk_file(results_path, config, rep, chunk, "models")) ||
            push!(tasks, (config, rep, chunk, chunk_size, model_selection_test_file(run, config), fixed_model_file, n_draws, results_path))
    end
end
println("Run $(run.run): chunks to evaluate: ", length(tasks))

## A failing chunk is logged and skipped rather than stopping every worker; rerunning retries it
pmap(tasks) do task
    config, rep, chunk = task[1], task[2], task[3]
    println("Worker $(myid()) evaluating system $(config.system), replicate $rep, chunk $chunk")
    try
        run_simulation_chunk(task...)
    catch e
        msg = sprint(showerror, e, catch_backtrace())
        println("Chunk failed for system $(config.system), replicate $rep, chunk $chunk:\n$msg")
        write(chunk_file(task[end], config, rep, chunk, "error") * ".txt", msg)
    end
end

println("Finished the model-selection simulation study.")
