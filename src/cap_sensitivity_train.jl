## Retrain the baseline networks with capped parameter vectors redrawn (redraw) and as shipped under a new seed (reseed)
using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags = ["--threads", get(ENV, "SLURM_CPUS_PER_TASK", "1"), "--project"])

@everywhere include("mse_functions.jl")
@everywhere using LinearAlgebra, Random
@everywhere BLAS.set_num_threads(Threads.nthreads())

const baseline = (n_lists = 5, width = 256, n_hidden = 3, censoring_lower = 0, censoring_upper = 10, m = 1)
const encoding_dim = 128
const variants = Dict("redraw" => (redraw_capped = true, seed = 3), "reseed" => (redraw_capped = false, seed = 2))

tasks = [(v, type) for v in keys(variants) for type in ("npe", "nbe")]   # NPE first: it takes longest
b = baseline
model_file(type) = type == "nbe" ? "model_$(b.n_lists)_$(b.width)_$(b.n_hidden)_$(train_size_default)_$(b.censoring_lower)_$(b.censoring_upper)_$(b.m).bson" :
                                   "model_$(b.n_lists)_$(b.width)_$(b.n_hidden)_$(encoding_dim)_$(train_size_default)_$(b.censoring_lower)_$(b.censoring_upper)_$(b.m).bson"
tasks = filter(((v, type),) -> !isfile(joinpath("output", "models_cap_sensitivity", v, type, model_file(type))), tasks)
println("Models to train: ", tasks)

pmap(tasks; on_error = e -> (println("Training task failed: ", sprint(showerror, e)); nothing)) do (variant, type)
    settings = variants[variant]
    path = joinpath("output", "models_cap_sensitivity", variant, type)
    mkpath(path)
    Random.seed!(settings.seed)
    println("Worker $(myid()): training $type ($variant)")
    if type == "nbe"
        train_model_mlp(b.n_lists, b.width, b.n_hidden, train_size_default;
                        b.censoring_lower, b.censoring_upper, b.m, savepath = path, settings.redraw_capped, training_settings...)
    else
        train_npe(b.n_lists, b.width, b.n_hidden, encoding_dim, train_size_default;
                  b.censoring_lower, b.censoring_upper, b.m, savepath = path, settings.redraw_capped,
                  num_coupling_layers = npe_num_coupling_layers, training_settings...)
    end
end

missing_models = filter(((v, type),) -> !isfile(joinpath("output", "models_cap_sensitivity", v, type, model_file(type))),
                        [(v, type) for v in keys(variants) for type in ("npe", "nbe")])
isempty(missing_models) || (println("Missing models: $missing_models; resubmit to train them."); exit(1))
println("Finished training the cap-sensitivity networks.")
