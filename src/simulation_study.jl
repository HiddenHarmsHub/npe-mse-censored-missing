## In this script we asses the quality of our estimators on fixed test sets

include("mse_functions.jl")

using Random, DataFrames, CSV

models_path = joinpath("output", "models")

## First generate fixed test sets for each list size
test_size = 10_000
test_path = joinpath("output", "test_data")
mkpath(test_path)
list_sizes = [3, 4, 5, 6, 10, 15]
Random.seed!(42)  # For reproducibility
for K in list_sizes
    outfile = joinpath(test_path, "test_data_$(K).bson")
    if isfile(outfile)
        println("Test data for K=$K already exists at $outfile; skipping.")
        continue
    end

    params = [sample_parameters(K) for _ in 1:test_size]
    Z_test = [simulate_data(params[i], 1, censoring_threshold=0) for i in 1:test_size]

    # Concatenate into matrices
    params = hcat(params...)
    Z_test = hcat(Z_test...)

    BSON.@save outfile Z_test params
    println("Generated test data for K=$K")
end

## Also save K=5 to csv for comaprison with MCMC
test_data, test_pars = load_test_data(test_path, 5, 0, 10)
CSV.write(joinpath("output", "test_data_K5.csv"), DataFrame(test_data, :auto))

## Now provide a summary of the test results for each model
function test_summary(models_path)
    model_files = filter(x -> !occursin("ci", x), readdir(models_path))

    output = []
    for model_file in model_files
        n_lists, width, n_hidden, train_size, censoring_lower, censoring_threshold, m = parse.(Int, [m.match for m in eachmatch(r"\d+", model_file)])
        println("Evaluating model: $model_file")
        model = load_model(model_file, models_path)
        test_data, test_pars = load_test_data(test_path, n_lists, censoring_lower, censoring_threshold)
        estimated_pars  = model(test_data)
        push!(output, DataFrame(
            n_lists=n_lists, 
            width=width, 
            n_hidden=n_hidden, 
            train_size=train_size,
            censoring_lower=censoring_lower, 
            censoring_threshold=censoring_threshold,
            parameter = get_param_names(n_lists),
            bias = vec(mean(estimated_pars .- test_pars, dims=2)), 
            mse = vec(mean((estimated_pars .- test_pars).^2, dims=2)), 
            mae = vec(mean(abs.(estimated_pars .- test_pars), dims=2)), 
            rmse = vec(sqrt.(mean((estimated_pars .- test_pars).^2, dims=2))), 
            mape = vec(mean(abs.((estimated_pars .- test_pars) ./ test_pars), dims=2))
        ))
    end

    return vcat(output...)
end


test_summary_df = test_summary(models_path)
CSV.write(joinpath("output", "test_summary.csv"), test_summary_df)

function intercept_APE_summary(models_path, test_path)
    model_files = filter(x -> !occursin("ci", x), readdir(models_path))

    output = []
    for model_file in model_files
        n_lists, width, n_hidden, train_size, censoring_lower, censoring_threshold, m = parse.(Int, [m.match for m in eachmatch(r"\d+", model_file)])
        model = load_model(model_file, models_path)
        model_cis = load_model(replace(model_file, "model_" => "model_ci_"), models_path)
        test_data, test_pars = load_test_data(test_path, n_lists, censoring_lower, censoring_threshold)
        n_pars = size(test_pars, 1)
        estimated_pars  = model(test_data)
        estimated_cis = model_cis(test_data)

        APE = abs.((estimated_pars .- test_pars) ./ test_pars)
        push!(output, DataFrame(
            dataset = 1:size(test_pars, 2),
            n_lists=n_lists, 
            width=width, 
            n_hidden=n_hidden, 
            train_size=train_size, 
            censoring_threshold=censoring_threshold,
            parameter = "intercept",
            intercept_truth = test_pars[1, :],
            intercept_estimated = estimated_pars[1, :],
            intercept_lower_ci = estimated_cis[1, :],
            intercept_upper_ci = estimated_cis[n_pars + 1, :],
            APE = vec(APE[1, :])
        ))
    end

    return vcat(output...)
end

intercept_ape_df = intercept_APE_summary(models_path, test_path)
CSV.write(joinpath("output", "intercept_ape_summary.csv"), intercept_ape_df)



## also analyse the deepsets models

function intercept_APE_summary_ds(models_path, test_path)
    model_files = filter(x -> !occursin("ci", x), readdir(models_path))

    output = []
    for model_file in model_files
        println("Evaluating model: $model_file")
        n_lists, width, n_encoder, n_decoder, train_size, censoring_lower, censoring_threshold, m = parse.(Int, [m.match for m in eachmatch(r"\d+", model_file)])
        model = load_model(model_file, models_path)
        test_data, test_pars = load_test_data(test_path, n_lists, censoring_lower, censoring_threshold)
        estimated_pars = hcat([model(hcat(x)) for x in eachcol(test_data)]...)

        APE = abs.((estimated_pars .- test_pars) ./ test_pars)
        push!(output, DataFrame(
            dataset = 1:size(test_pars, 2),
            n_lists=n_lists, 
            width=width, 
            n_encoder=n_encoder, 
            n_dencoder=n_decoder, 
            train_size=train_size, 
            censoring_threshold=censoring_threshold,
            parameter = "intercept",
            intercept_truth = test_pars[1, :],
            intercept_estimated = estimated_pars[1, :],
            APE = vec(APE[1, :])
        ))
    end

    return vcat(output...)
end

ds_models_path = joinpath("output", "models_ds")
intercept_ape_df_ds = intercept_APE_summary_ds(ds_models_path, test_path)

CSV.write(
    joinpath("output", "intercept_ape_summary_ds.csv"), 
    intercept_ape_df_ds
)