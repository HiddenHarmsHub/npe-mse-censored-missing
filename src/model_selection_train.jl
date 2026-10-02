## Train the structure classifiers q(m | y) and conditional NPEs q(θ | y, m) for systems A and B,
## on the architectures frozen by architecture_selection.jl, with independent replicates
using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")
@everywhere include("model_selection_functions.jl")

train_size = 100_000
gamma_sd = 4
replicates = 1:3

mkpath(model_selection_models_path)
configs = [model_selection_config(model_selection_systems[s]; train_size, gamma_sd) for s in ["A", "B"]]
foreach(println, configs)

tasks = [(config, kind, rep) for config in configs for kind in ["classifier", "cnpe"] for rep in replicates]
tasks = filter(tasks) do (config, kind, rep)
    file = kind == "classifier" ? classifier_filename(config, rep) : cnpe_filename(config, rep)
    !isfile(joinpath(model_selection_models_path, file))
end
println("Models to train: ", length(tasks))

pmap(tasks) do (config, kind, rep)
    println("Worker $(myid()) training $kind for system $(config.system), replicate $rep")
    kind == "classifier" ? train_classifier(config, rep) : train_cnpe(config, rep)
    return nothing
end

println("Finished training all model-selection networks.")
