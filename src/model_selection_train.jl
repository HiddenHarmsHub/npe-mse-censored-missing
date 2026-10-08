## Train the structure classifiers q(m | y) and conditional NPEs q(θ | y, m) for systems A and B,
## on the architectures frozen by architecture_selection.jl, with independent replicates.
## The run (priors and training settings) comes from MS_RUN; see model_selection_runs.
using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")
@everywhere include("model_selection_functions.jl")

run = model_selection_run()

configs = [model_selection_config(model_selection_systems[s], run) for s in ["A", "B"]]
mkpath(models_path(configs[1]))
foreach(println, configs)

tasks = vcat([(config, "classifier", rep) for config in configs for rep in classifier_replicates(config)],
             [(config, "cnpe", rep) for config in configs for rep in cnpe_replicates(config)])
tasks = filter(tasks) do (config, kind, rep)
    file = kind == "classifier" ? classifier_filename(config, rep) : cnpe_filename(config, rep)
    !isfile(joinpath(models_path(config), file))
end
println("Models to train: ", length(tasks))

pmap(tasks) do (config, kind, rep)
    println("Worker $(myid()) training $kind for system $(config.system), replicate $rep")
    kind == "classifier" ? train_classifier(config, rep) : train_cnpe(config, rep)
    return nothing
end

println("Finished training all model-selection networks.")
