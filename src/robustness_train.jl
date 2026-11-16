## Robustness study: train the multi-quantile NBEs and the extra NPEs (Slurm).
using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", get(ENV, "SLURM_CPUS_PER_TASK", "1"), "--project"])

@everywhere include("mse_functions.jl")
@everywhere include("model_selection_functions.jl")
@everywhere include("robustness_functions.jl")
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(Threads.nthreads())

tasks = robustness_training_tasks()
foreach(t -> println(t.kind, " σ = ", t.sigma, ": ", t.file, isfile(t.file) ? " (exists)" : ""), tasks)
tasks = sort(filter(t -> !isfile(t.file), tasks), by = t -> t.kind == "npe", rev = true)
println("Networks to train: ", length(tasks))

pmap(tasks; on_error = e -> (println("Training task failed: ", sprint(showerror, e)); nothing)) do task
    println("Worker $(myid()) training $(task.kind) for system $(task.config.system) (censoring $(task.config.censoring_lower)-$(task.config.censoring_upper)), σ = $(task.sigma)")
    train_robustness_network(task)
end

missing_files = filter(t -> !isfile(t.file), robustness_training_tasks())
if !isempty(missing_files)
    println("Missing $(length(missing_files)) networks; resubmit this job to train them.")
    exit(1)
end
println("Finished training the robustness networks.")
