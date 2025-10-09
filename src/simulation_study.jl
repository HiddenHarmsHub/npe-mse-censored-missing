## In this script we asses the quality of our estimators on fixed test sets

include("mse_functions.jl")

using Random, DataFrames

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



function compute_assessment(models_path)

end

model_files = readdir(models_path)

output = []
for model_file in model_files[1:5]
    numbers = parse.(Int, [m.match for m in eachmatch(r"\d+", model_file)])
    n_lists, width, n_hidden, censoring_threshold, train_size = parse.(Int, [m.match for m in eachmatch(r"\d+", model_file)])
    censoring_threshold > 0 && continue
    model = load_model(model_file, models_path)
    test_data, test_pars = load_test_data(test_path, n_lists)
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



output[1]




nbe_model = load_model(
    n_lists = 5, 
    width = 128,
    n_hidden = 3, 
    censoring_threshold = 0,
    train_size = 100_000, 
    models_path = models_path
)

test_data, test_pars = load_test_data(test_path, 5)

estimated_pars  = nbe_model(test_data)

APE = abs.((estimated_pars .- test_pars) ./ test_pars)

scatter(test_pars[1, :], APE[1, :])


scatter(exp.(test_pars[1, :]), log.(APE[1, :]), xlabel="True intercept", ylabel="Estimated intercept", title="NPE-MSE Intercept Estimates (K=3)", legend=false)