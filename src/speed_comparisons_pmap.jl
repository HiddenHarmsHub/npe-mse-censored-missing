using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere using Random
@everywhere include("mse_functions.jl")
@everywhere include("mcmc_functions.jl")

@everywhere function get_nbe_estimates(nbe_model, nbe_model_ci, test_data, slice_idx)
    nbe_model(test_data[:, slice_idx])
    nbe_model_ci(test_data[:, slice_idx])
end

@everywhere function time_function(fn, args)
    start = time_ns()
    fn(args...)
    return (time_ns() - start) / 1.0e9
end

@everywhere function train_time_comparison()
    n_lists = 5
    width = 256
    n_hidden = 3
    train_size = 10000
    train_nbe_function() = train_model_mlp(
        n_lists, 
        width, 
        n_hidden, 
        train_size, 
        m = 1, 
        censoring_lower = 0, 
        censoring_upper = 10, 
        savepath = nothing, 
        intercept_dist = Uniform(1, 10)
    )

    train_npe() = train_npe(
        n_lists,
        width,
        n_hidden,
        encoding_dim,
        train_size,
        m = 1,
        censoring_lower = 0,
        censoring_upper = 10,
        savepath = nothing
    )

    train_nbe_function() ## warmup
    train_npe_function() ## warmup
    n_runs = 10
    train_df = DataFrame()
    for _ in 1:n_runs
        push!(train_df, (method = "NBE", train_time = time_function(train_nbe_function, ())))
        push!(train_df, (method = "NPE", train_time = time_function(train_npe_function, ())))
    end
    mkpath(joinpath("output", "speed_comparisons"))
    CSV.write(
        joinpath("output", "speed_comparisons", "train_time_comparison.csv"),
        DataFrame(train_time = train_times)
    )
end

@everywhere function run_speed_comparison(slice_idx)
    n_lists = 5
    n_runs = 10
    nbe_models_path = joinpath("output", "models_nbe")
    npe_models_path = joinpath("output", "models_npe")
    test_data_path = joinpath("output", "test_data")
    test_data, test_pars = load_test_data(test_data_path, n_lists, 0, 10)

    speed_comparison_path = joinpath("output", "speed_comparisons")
    mkpath(speed_comparison_path)

    npe_model = load_model_npe(
        5, 
        256, 
        3,
        10000,
        0, 
        10, 
        1, 
        npe_models_path
    )

    nbe_model = load_model_nbe(
        5,
        256,
        3,
        10000,
        0,
        10,
        1,
        nbe_models_path
    )

    nbe_model_ci = load_model_nbe(
        5,
        256,
        3,
        10000,
        0,
        10,
        1,
        nbe_models_path,
        true
    )

    out = DataFrame(
        dataset = Int64[],
        method = String[],
        iterations = Int64[],
        time = Float64[]
    )

    ## warm up first
    nbe_time = time_function(get_nbe_estimates, (nbe_model, nbe_model_ci, test_data, slice_idx))
    
    for _ in 1:n_runs
        nbe_time = time_function(get_nbe_estimates, (nbe_model, nbe_model_ci, test_data, slice_idx))
        push!(out, (
            dataset = slice_idx,
            method = "NBE",
            iterations = 0,
            time = nbe_time
        ))
    end

    iterations_list = 1000 * 2 .^ collect(0:5)

    for (i, num_iterations) in enumerate(iterations_list)
        println("Benchmarking MCMC with $num_iterations iterations...")
        run_mcmc(slice_idx) = run_mcmc_test_slice(slice_idx, n_lists, test_data, test_pars, num_chains = 4, samples_path = nothing, summary_path = nothing, n_iterations = num_iterations)
        
        i == 1 && time_function(run_mcmc, slice_idx) ## warmup

        for _ in 1:n_runs
            mcmc_time = time_function(run_mcmc, slice_idx)        
            push!(out, (
                dataset = slice_idx,
                method = "MCMC",
                iterations = num_iterations,
                time = mcmc_time
            ))
        end

        run_npe(slice_idx) = sampleposterior(npe_model, reshape(test_data[:, slice_idx], :, 1), num_iterations)
        
        i == 1 && time_function(run_npe, slice_idx) ## warmup
        
        for _ in 1:n_runs
            npe_time = time_function(run_npe, slice_idx)
            push!(out, (
                dataset = slice_idx,
                method = "NPE",
                iterations = num_iterations,
                time = npe_time
            ))
        end
    end
    savepath = joinpath("output", "speed_comparisons")
    mkpath(savepath)
    CSV.write(
        joinpath(savepath, "speed_comparison_dataset_$(slice_idx).csv"),
        out
    )
end



test_data, test_pars = load_test_data(joinpath("output", "test_data"), 5, 0, 10)


Random.seed!(42)
n_datasets_to_sample = 100
datasets_to_sample = rand(1:size(test_data, 2), n_datasets_to_sample)
savepath = joinpath("output", "speed_comparisons")
datasets_to_sample = filter(datasets_to_sample) do idx
    !isfile(joinpath(savepath, "speed_comparison_dataset_$(idx).csv"))
end

pmap(run_speed_comparison, datasets_to_sample)


## also run train time comparison

train_time_comparison()