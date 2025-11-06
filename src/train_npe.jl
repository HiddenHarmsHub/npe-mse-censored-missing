using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")


grid_npe = collect(Base.product(
    [5],
    [8, 16, 32, 64, 128, 256],
    [1, 2, 3, 4],
    [8, 16, 32, 64, 128],
    [10000],
    [0],
    [10],
    [1]
))[:]

output_path = joinpath("output", "models_npe")
mkpath(output_path)

pmap(
    model -> begin
        #wid = myid()
        #println("Worker $wid running with parameters: list_size=$(model[1]), width=$(model[2]), n_hidden=$(model[3]), train_size=$(model[4]), censoring_lower=$(model[5]) censoring_threshold=$(model[6]), m=$(model[7])")
        train_npe(
            model[1],
            model[2],
            model[3], 
            model[4],
            model[5], 
            censoring_lower = model[6],
            censoring_threshold = model[7],
            m = model[8],
            savepath = output_path
        )
    end,
    grid_npe
)