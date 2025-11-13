using Pkg; Pkg.activate(".")

using Statistics, Random
include("mse_functions.jl")
include("mcmc_functions.jl")

function get_nbe_estimates(nbe_model, nbe_model_ci, test_data, slice_idx)
    nbe_model(test_data[:, slice_idx])
    nbe_model_ci(test_data[:, slice_idx])
end


function run_speed_comparison(nbe_model, nbe_model_ci, npe_model, test_data, test_pars, n_lists, iterations_list, slice_idx, savepath)
    # Benchmark NBE with multiple runs
    n_warmup = 1
    n_runs = 10
    
    # Warmup runs
    for _ in 1:n_warmup
        get_nbe_estimates(nbe_model, nbe_model_ci, test_data, slice_idx)
    end
    
    # Timed runs
    nbe_times = zeros(n_runs)
    for i in 1:n_runs
        nbe_times[i] = @elapsed get_nbe_estimates(nbe_model, nbe_model_ci, test_data, slice_idx)
    end
    
    # Store all runs
    out = DataFrame(
        dataset = Int[],
        method = String[],
        iterations = Int[],
        run = Int[],
        time = Float64[]
    )
    
    for (run_idx, time_val) in enumerate(nbe_times)
        push!(out, (
            dataset = slice_idx,
            method = "NBE",
            iterations = 0,
            run = run_idx,
            time = time_val * 1e9  # Convert to nanoseconds
        ))
    end

    for num_iterations in iterations_list
        println("Benchmarking MCMC with $num_iterations iterations...")
        
        # Warmup run for MCMC
        run_mcmc_test_slice(slice_idx, n_lists, test_data, test_pars, num_chains = 4, samples_path = nothing, summary_path = nothing, n_iterations = num_iterations)
        
        # Timed runs for MCMC
        mcmc_times = zeros(n_runs)
        for i in 1:n_runs
            mcmc_times[i] = @elapsed run_mcmc_test_slice(slice_idx, n_lists, test_data, test_pars, num_chains = 4, samples_path = nothing, summary_path = nothing, n_iterations = num_iterations)
        end
        
        for (run_idx, time_val) in enumerate(mcmc_times)
            push!(out, (
                dataset = slice_idx,
                method = "MCMC",
                iterations = num_iterations,
                run = run_idx,
                time = time_val * 1e9  # Convert to nanoseconds
            ))
        end

        # Warmup run for NPE
        sampleposterior(npe_model, reshape(test_data[:, slice_idx], :, 1), num_iterations)
        
        # Timed runs for NPE
        npe_times = zeros(n_runs)
        for i in 1:n_runs
            npe_times[i] = @elapsed sampleposterior(npe_model, reshape(test_data[:, slice_idx], :, 1), num_iterations)
        end

        for (run_idx, time_val) in enumerate(npe_times)
            push!(out, (
                dataset = slice_idx,
                method = "NPE",
                iterations = num_iterations,
                run = run_idx,
                time = time_val * 1e9  # Convert to nanoseconds
            ))
        end
    end

    CSV.write(
        joinpath(savepath, "speed_comparison_dataset_$(slice_idx).csv"),
        out
    )
end

n_lists = 5
test_data_path = joinpath("output", "test_data")
test_data, test_pars = load_test_data(test_data_path, n_lists, 0, 10)

nbe_models_path = joinpath("output", "models_nbe")
npe_models_path = joinpath("output", "models_npe")

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

speed_comparison_path = joinpath("output", "speed_comparisons")
mkpath(speed_comparison_path)

Random.seed!(42)
n_datasets_to_sample = 100
datasets_to_sample = rand(1:size(test_data, 2), n_datasets_to_sample)

iterations_list = 1000 * 2 .^ collect(0:5)

### TESTING
iterations_list = iterations_list[1:3]
datasets_to_sample = datasets_to_sample[1:4]

map(
    slice_idx -> begin
        run_speed_comparison(
            nbe_model,
            nbe_model_ci,
            npe_model,
            test_data,
            test_pars,
            n_lists,
            iterations_list,
            slice_idx,
            speed_comparison_path
        )
    end,
    datasets_to_sample
)


