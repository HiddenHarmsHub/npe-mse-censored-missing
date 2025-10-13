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
    params = [sample_parameters(K) for _ in 1:test_size]
    Z_test = [simulate_data(params[i], 1, censoring_threshold=0) for i in 1:test_size]

    # Concatenate into matrices
    params = hcat(params...) 
    Z_test = hcat(Z_test...)  

    BSON.@save joinpath(test_path, "test_data_$(K).bson") Z_test params
    println("Generated test data for K=$K")
end



function test_summary(models_path)
    model_files = readdir(models_path)

    output = []
    for model_file in model_files
        n_lists, width, n_hidden, censoring_threshold, train_size = parse.(Int, [m.match for m in eachmatch(r"\d+", model_file)])
        model = load_model(model_file, models_path)
        test_data, test_pars = load_test_data(test_path, n_lists, censoring_threshold)
        estimated_pars  = model(test_data)
        push!(output, DataFrame(
            n_lists=n_lists, 
            width=width, 
            n_hidden=n_hidden, 
            train_size=train_size, 
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

function intercept_APE_summary(models_path)
    model_files = readdir(models_path)

    output = []
    for model_file in model_files
        n_lists, width, n_hidden, censoring_threshold, train_size = parse.(Int, [m.match for m in eachmatch(r"\d+", model_file)])
        model = load_model(model_file, models_path)
        test_data, test_pars = load_test_data(test_path, n_lists, censoring_threshold)
        estimated_pars  = model(test_data)
        APE = abs.((estimated_pars .- test_pars) ./ test_pars)
        push!(output, DataFrame(
            n_lists=n_lists, 
            width=width, 
            n_hidden=n_hidden, 
            train_size=train_size, 
            censoring_threshold=censoring_threshold,
            parameter = "intercept",
            APE = vec(APE[1, :])
        ))
    end

    return vcat(output...)
end

intercept_ape_df = intercept_APE_summary(models_path)
CSV.write(joinpath("output", "intercept_ape_summary.csv"), intercept_ape_df)

#train_model_mlp(5, 128, 3, 10_000, censoring_threshold = 10, savepath = models_path)

nbe_model = load_model(
    n_lists = 5, 
    width = 128,
    n_hidden = 3, 
    censoring_threshold = 10,
    train_size = 10_000, 
    models_path = models_path
)
    
## now look at individual ability to recover intercept

test_data, test_pars = load_test_data(test_path, 5, 10)
estimated_pars  = nbe_model(test_data)
APE = abs.((estimated_pars .- test_pars) ./ test_pars)
scatter(test_pars[1, :], APE[1, :])
scatter(
    exp.(test_pars[1, :]), log.(APE[1, :]), 
    xlabel="True intercept", 
    ylabel="Log MAPE", 
    title="NPE-MSE Intercept Estimates (K=3)", 
    legend=false
)


## now look at sensitivity to neurons
#map(x -> train_model_mlp(5, x, 3, 10_000, censoring_threshold = 10, savepath = models_path), [8, 16, 32, 64, 128, 256])

width_outputs = []
for width in [8, 16, 32, 64, 128, 256]
    model = load_model(
        n_lists = 5, 
        width = width,
        n_hidden = 3, 
        censoring_threshold = 10,
        train_size = 10_000, 
        models_path = models_path
    )
    
    test_data, test_pars = load_test_data(test_path, 5, 10)
    estimated_pars  = model(test_data)
    APE = abs.((estimated_pars .- test_pars) ./ test_pars)
    push!(width_outputs, DataFrame(
        width = width,
        APE = APE[1, :]
    ))
end


vcat(width_outputs...) |> df -> CSV.write(joinpath("output", "width_sensitivity.csv"), df)


## now look at sensitivity to the censoring level

#map(x -> train_model_mlp(5, 128, 3, 10_000, censoring_threshold = x, savepath = models_path), [0, 2, 4, 8, 16, 32, 64, 128])

censoring_outputs = []
for censoring_threshold in [0, 2, 4, 8, 16, 32, 64, 128]
    model = load_model(
        n_lists = 5, 
        width = 128,
        n_hidden = 3, 
        censoring_threshold = censoring_threshold,
        train_size = 10_000, 
        models_path = models_path
    )
    
    test_data, test_pars = load_test_data(test_path, 5, censoring_threshold)
    estimated_pars  = model(test_data)
    APE = abs.((estimated_pars .- test_pars) ./ test_pars)
    push!(censoring_outputs, DataFrame(
        censoring_threshold = censoring_threshold,
        APE = APE[1, :]
    ))
end

vcat(censoring_outputs...) |> df -> CSV.write(joinpath("output", "censoring_sensitivity.csv"), df)

## now look at sensitivity to the number of lists

#map(x -> train_model_mlp(x, 128, 3, 10_000, censoring_threshold = 10, savepath = models_path), [3, 4, 5, 6, 10, 15])

n_lists_outputs = []
for n_lists in [3, 4, 5, 6, 10, 15]
    model = load_model(
        n_lists = n_lists, 
        width = 128,
        n_hidden = 3, 
        censoring_threshold = 10,
        train_size = 10_000, 
        models_path = models_path
    )
    
    test_data, test_pars = load_test_data(test_path, n_lists, 10)
    estimated_pars  = model(test_data)
    APE = abs.((estimated_pars .- test_pars) ./ test_pars)
    push!(n_lists_outputs, DataFrame(
        n_lists = n_lists,
        APE = APE[1, :]
    ))
end

vcat(n_lists_outputs...) |> df -> CSV.write(joinpath("output", "n_lists_sensitivity.csv"), df)

## also look at the number of hidden layers

#map(x -> train_model_mlp(5, 128, x, 10_000, censoring_threshold = 10, savepath = models_path), [1, 2, 3, 4])

n_hidden_outputs = []
for n_hidden in [1, 2, 3, 4]
    model = load_model(
        n_lists = 5, 
        width = 128,
        n_hidden = n_hidden, 
        censoring_threshold = 10,
        train_size = 10_000, 
        models_path = models_path
    )
    
    test_data, test_pars = load_test_data(test_path, 5, 10)
    estimated_pars  = model(test_data)
    APE = abs.((estimated_pars .- test_pars) ./ test_pars)
    push!(n_hidden_outputs, DataFrame(
        n_hidden = n_hidden,
        APE = APE[1, :]
    ))
end

vcat(n_hidden_outputs...) |> df -> CSV.write(joinpath("output", "n_hidden_sensitivity.csv"), df)



## Compare the varied input censoring model with a fixed input censoring model
#map(x -> train_model_mlp(4, 256, 3, 10_000, censoring_threshold = x, savepath = models_path), [0, 2, 4, 8, 16, 32, 64, 128])

variable_censoring_model = load_model(
    n_lists = 4, 
    width = 256,
    n_hidden = 3, 
    censoring_threshold = -1,
    train_size = 10_000, 
    models_path = "output"
)

n_hidden = 3
width = 256
n_lists = 4
train_size = 10_000

variable_censoring_output = []
for censoring_threshold in [0, 2, 4, 8, 16, 32, 64, 128]
    fixed_censoring_model = load_model(
        n_lists = n_lists, 
        width = width,
        n_hidden = n_hidden, 
        censoring_threshold = censoring_threshold,
        train_size = train_size, 
        models_path = models_path
    )
    
    test_data, test_pars = load_test_data(test_path, n_lists, censoring_threshold)
    if censoring_threshold == 0
        n_data, n_rep = size(test_data)
        variable_test_data = vcat(test_data, zeros(Float32, n_data + 1, n_rep))
    else
        variable_test_data = vcat(test_data, fill(censoring_threshold, 1, size(test_data, 2)))  # Add the censoring threshold input
    end
    estimated_pars_var  = variable_censoring_model(variable_test_data)
    estimated_pars_fix  = fixed_censoring_model(test_data)
    APE_var = abs.((estimated_pars_var .- test_pars) ./ test_pars)
    APE_fix = abs.((estimated_pars_fix .- test_pars) ./ test_pars)
    
    push!(variable_censoring_output, DataFrame(
        censoring_threshold = censoring_threshold,
        APE_var = APE_var[1, :],
        APE_fix = APE_fix[1, :]
    ))
end

vcat(variable_censoring_output...) |> df -> CSV.write(joinpath("output", "variable_vs_fixed_censoring.csv"), df)


