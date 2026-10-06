using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
## Each worker gets the cores Slurm allocates per task: simulation (Folds) and BLAS both use threads
addprocs(SlurmManager(); exeflags=["--threads", get(ENV, "SLURM_CPUS_PER_TASK", "1"), "--project"])

@everywhere include("mse_functions.jl")
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(Threads.nthreads())

## Train models for each of the sensitivity analyses

println("Available worker count: ", nworkers())
println("Worker IDs: ", workers())

## Number of lists
grid_lists = collect(Base.product(
    [3, 4, 5, 6, 10],
    [256], 
    [3], 
    [train_size_default],
    [0],
    [10],
    [1]
))[:]

## Number of neurons
grid_neurons = collect(Base.product(
    [5],
    [8, 16, 32, 64, 128, 256],
    [3],
    [train_size_default],
    [0],
    [10],
    [1]
))[:]

grid_hidden = collect(Base.product(
    [5],
    [256],
    [1, 2, 3, 4, 5],
    [train_size_default],
    [0],
    [10],
    [1]
))[:]

## Censoring level
grid_censoring = collect(Base.product(
    [5],
    [256],
    [3],
    [train_size_default],
    [0],
    [0, 2, 4, 8, 16, 32, 64, 128],
    [1]
))[:]

## Test various architectures for 4 lists, censoring 1-4
grid_4 = collect(Base.product(
    [4],
    [8, 16, 32, 64, 128, 256],
    [1, 2, 3, 4],
    [train_size_default],
    [1],
    [4],
    [1]
))[:]

## Test various architectures for 5 lists, no censoring
grid_5 = collect(Base.product(
    [5],
    [8, 16, 32, 64, 128, 256],
    [1, 2, 3, 4],
    [train_size_default],
    [0],
    [0],
    [1]
))[:]

## Test various architectures for 6 lists, no censoring
grid_6 = collect(Base.product(
    [6],
    [8, 16, 32, 64, 128, 256],
    [1, 2, 3, 4],
    [train_size_default],
    [0],
    [0],
    [1]
))[:]

output_path_nbe = joinpath("output", "models_nbe")
mkpath(output_path_nbe)

grid_nbe = unique(vcat(grid_lists, grid_neurons, grid_censoring, grid_hidden, grid_4, grid_5, grid_6))

println("Total models to train: ", length(grid_nbe))

grid_nbe = sort(grid_nbe, by = x -> x[4]) ## ensure lower training sizes are done first
grid_nbe = filter(model -> model[5] <= model[6], grid_nbe)

overwrite_models = false
if !overwrite_models
    grid_nbe = filter(model -> !isfile(joinpath(output_path_nbe, "model_$(model[1])_$(model[2])_$(model[3])_$(model[4])_$(model[5])_$(model[6])_$(model[7]).bson")), grid_nbe)
end

println("Models to train after filtering: ", length(grid_nbe))

## Train both regular and NPE models in a single pmap
output_path_npe = joinpath("output", "models_npe")
mkpath(output_path_npe)

encoding_dim = 128

grid_npe = unique(vcat(grid_lists, grid_neurons, grid_censoring, grid_hidden, grid_4, grid_5, grid_6))

if !overwrite_models
    grid_npe = filter(model -> !isfile(joinpath(output_path_npe, "model_$(model[1])_$(model[2])_$(model[3])_$(encoding_dim)_$(model[4])_$(model[5])_$(model[6])_$(model[7]).bson")), grid_npe)
end

# Combine both model types into a single grid with type identifier
combined_grid = vcat(
    [(model..., "nbe") for model in grid_nbe],
    [(model..., "npe") for model in grid_npe]
)

## Most expensive first (more lists, then wider networks) so long jobs do not finish last
combined_grid = sort(combined_grid, by = task -> (task[1], task[2], task[3]), rev = true)

println("Total combined models to train: ", length(combined_grid))

pmap(
    task -> begin
        model = task[1:end-1]
        model_type = task[end]
        wid = myid()
        
        if model_type == "nbe"
            println("Worker $wid training NBE with parameters: list_size=$(model[1]), width=$(model[2]), n_hidden=$(model[3]), train_size=$(model[4]), censoring_lower=$(model[5]) censoring_upper=$(model[6]), m=$(model[7])")
            train_model_mlp(
                model[1],
                model[2],
                model[3], 
                model[4], 
                censoring_lower = model[5],
                censoring_upper = model[6],
                m = model[7],
                savepath = output_path_nbe;
                training_settings...
            )
        else  # "npe"
            println("Worker $wid training NPE with parameters: list_size=$(model[1]), width=$(model[2]), n_hidden=$(model[3]), train_size=$(model[4]), censoring_lower=$(model[5]) censoring_upper=$(model[6]), m=$(model[7])")
            train_npe(
                model[1],
                model[2],
                model[3], 
                encoding_dim,
                model[4], 
                censoring_lower = model[5],
                censoring_upper = model[6],
                m = model[7],
                savepath = output_path_npe,
                num_coupling_layers = npe_num_coupling_layers;
                training_settings...
            )
        end
    end,
    combined_grid
)

println("Finished training all models.")

