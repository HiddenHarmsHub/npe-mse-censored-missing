using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")

## Train models for each of the sensitivity analyses

## Number of lists
grid_lists = collect(Base.product(
    [3, 4, 5, 6, 10, 15],
    [256], 
    [3], 
    [100000],
    [0],
    [10],
    [1]
))[:]

## Number of neurons
grid_neurons = collect(Base.product(
    [5],
    [8, 16, 32, 64, 128, 256],
    [3],
    [100000],
    [0],
    [10],
    [1]
))[:]

## Censoring level
grid_censoring = collect(Base.product(
    [5],
    [256],
    [3],
    [100000],
    [0],
    [0, 2, 4, 8, 16, 32, 64, 128],
    [1]
))[:]

## multiple datasets per model
grid_m = collect(Base.product(
    [5],
    [256],
    [3],
    [100000],
    [0],
    [10],
    [1, 5, 10, 50, 100, 500]
))[:]

## Test various architectures for 5 lists
grid_5 = collect(Base.product(
    [5],
    [8, 16, 32, 64, 128, 256],
    [1, 2, 3, 4],
    [100000],
    [0],
    [10],
    [1]
))[:]

## Test various architectures for 4 lists, censoring 1-4
grid_4 = collect(Base.product(
    [4],
    [8, 16, 32, 64, 128, 256],
    [1, 2, 3, 4],
    [100000],
    [1],
    [4],
    [1]
))[:]

grid = vcat(grid_lists, grid_neurons, grid_censoring, grid_m, grid_5, grid_4)

grid = sort(grid, by = x -> x[4]) 
grid = filter(model -> model[5] <= model[6], grid)

overwrite_models = false
if !overwrite_models
    grid = filter(model -> !isfile(joinpath("output", "models", "model_$(model[1])_$(model[2])_$(model[3])_$(model[4])_$(model[5])_$(model[6]).bson")), grid)
end

output_path = joinpath("output", "models")
mkpath(output_path)

pmap(
    model -> begin
        wid = myid()
        println("Worker $wid running with parameters: list_size=$(model[1]), width=$(model[2]), n_hidden=$(model[3]), train_size=$(model[4]), censoring_threshold=$(model[5])")
        train_model_mlp(
            model[1],
            model[2],
            model[3], 
            model[4], 
            censoring_lower = model[5],
            censoring_threshold = model[6],
            m = model[7],
            savepath = output_path
        )
    end,
    grid
)

