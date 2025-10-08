using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")


widths = [8, 16, 32, 64, 128, 256]
n_hidden = [1, 2, 3, 4, 5]
censoring_thresholds = [0, 2, 4, 8, 16, 32, 64, 128, 256]
list_sizes = [3, 4, 5, 6, 10, 15]
train_sizes = [10000, 100000, 1000000]


grid = collect(Base.product(
    list_sizes,
    widths, 
    n_hidden, 
    train_sizes,
    censoring_thresholds
))[:]


output_path = joinpath("output", "models")

pmap(
    model -> train_model_mlp(
        model[1],
        model[2],
        model[3], 
        model[4], 
        censoring_threshold = model[5],
        savepath = output_path,
        overwrite = false
    ),
    grid
)


