using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")


widths = [8, 16, 32, 64, 128, 256]
n_hidden = [1, 2, 3, 4]
censoring_thresholds = [0, 2, 4, 8, 10, 16, 32, 64, 128]
list_sizes = [3, 4, 5, 6, 10, 15]
train_sizes = [10000, 100000]


grid = collect(Base.product(
    list_sizes,
    widths, 
    n_hidden, 
    train_sizes,
    censoring_thresholds
))[:]

grid = sort(grid, by = x -> x[4])

overwrite_models = false
if !overwrite_models
    grid = filter(model -> !isfile(joinpath("output", "models", "model_$(model[1])_$(model[2])_$(model[3])_$(model[5])_$(model[4]).bson")), grid)
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
            censoring_threshold = model[5],
            savepath = output_path
        )
    end,
    grid
)


