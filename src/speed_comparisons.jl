using Pkg; Pkg.activate(".")

Pkg.add("BenchmarkTools")

using BenchmarkTools, Random
include("mse_functions.jl")
include("mcmc_functions.jl")


function get_nbe_estimates(nbe_model, nbe_model_ci, test_data, slice_idx)
    nbe_model(test_data[:, slice_idx])
    nbe_model_ci(test_data[:, slice_idx])
end

function run_speed_comparison(nbe_model, nbe_model_ci, npe_model, test_data, test_pars, n_lists, iterations_list, slice_idx, savepath)
    nbe_time = @benchmark get_nbe_estimates(nbe_model, nbe_model_ci, test_data, slice_idx)
    out = DataFrame(
        dataset = slice_idx,
        method = "NBE",
        iterations = 0,
        time = median(nbe_time).time
    )

    for num_iterations in iterations_list
        println("Benchmarking MCMC with $num_iterations iterations...")
        mcmc_time = @benchmark run_mcmc_test_slice($slice_idx, $n_lists, $test_data, $test_pars, num_chains = 4, samples_path = nothing, summary_path = nothing, n_iterations = $num_iterations)
        push!(out, (
            dataset = slice_idx,
            method = "MCMC",
            iterations = num_iterations,
            time = median(mcmc_time).time
        ))

        npe_time = @benchmark sampleposterior($npe_model, reshape($test_data[:, $slice_idx], :, 1), $num_iterations)

        push!(out, (
            dataset = slice_idx,
            method = "NPE",
            iterations = num_iterations,
            time = median(npe_time).time
        ))
    end

    CSV.write(
        joinpath("output", "speed_comparison_dataset_$(slice_idx).csv"),
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

pmap(
    slice_idx -> begin
        wid = myid()
        println("Worker $wid running speed comparison for dataset $slice_idx")
        run_speed_comparison(
            $nbe_model,
            $nbe_model_ci,
            $npe_model,
            $test_data,
            $test_pars,
            $n_lists,
            $iterations_list,
            slice_idx,
            $speed_comparison_path
        )
    end,
    datasets_to_sample
)


