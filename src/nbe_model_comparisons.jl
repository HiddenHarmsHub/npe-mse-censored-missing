using NeuralEstimators, DataFrames, Distributions, BSON, StatsPlots
using BSON: @save, @load

include("mse_functions.jl")

function load_model(path, i)
    y = BSON.load(joinpath(path, "nbe_model_$i.bson"))
    return y[:estimator]
end

@load joinpath("output", "model_list.bson") model_list


K = 5
sample_nbe(n_reps) = hcat([generate_parameters_nbe(K, intercept_dist, beta_dist, gamma_dist) for _ in 1:n_reps]...)
simulate_nbe(θ, m) = [generate_data_nbe(params, m) for params in eachcol(θ)]

intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)

model_path = joinpath("output", "models")



k5_models = filter(row -> row.K == K, model_list)

k5_model_list = map(k5_models.i) do i
    load_model(model_path, i)
end

n_reps = 2000
n_data = K + binomial(K, 2)
n_pars = 1 + n_data
n_models = length(k5_model_list)
errors = Array{Float64}(undef, n_pars, n_models, n_reps)

for rep in 1:n_reps
    θ_fixed = sample_nbe(1)
    Z_fixed = simulate_nbe(θ_fixed, 1)
    estimates = map(k5_model_list) do estimator
        vec(NeuralEstimators.estimate(estimator, Z_fixed))
    end |> (y -> hcat(y...))
    errors[:, :, rep] = estimates .- θ_fixed
end

# Optionally, convert to DataFrame for further analysis
error_dfs = DataFrame[]
for rep in 1:n_reps
    df = DataFrame(parameter = 1:n_pars)
    for (i, m) in enumerate(k5_models.m)
        df[!, Symbol("error_m_$m")] = errors[:, i, rep]
    end
    push!(error_dfs, df)
end

error_df = vcat(error_dfs...)


# Select relevant columns for parameter 1
m_values = [1, 10, 50, 100, 500]
cols = Symbol.("error_m_", string.(m_values))
df_param1 = filter(row -> row.parameter == 1, error_df)

# Prepare data for boxplot
data = [df_param1[!, col] for col in cols]

# Plot boxplots side by side
boxplot(
    m_values,
    data,
    legend = false,
    xlabel = "Model m",
    ylabel = "Error (parameter 1)",
    xticks = (1:length(m_values), string.(m_values)),
    title = "Boxplot of Errors for Parameter 1"
)

# Plot overlapping density estimates for parameter 1 errors

densityplot = plot(
    xlabel = "Error (parameter 1)",
    ylabel = "Density",
    title = "Density Estimates of Errors for Parameter 1"
)

for (i, col) in enumerate(cols)
    density!(densityplot, df_param1[!, col], label = "m = $(m_values[i])", lw = 2)
end

display(densityplot)

# Compute Mean Absolute Errors (MAEs) for each model and parameter
mae_matrix = zeros(n_pars, n_models)
for i in 1:n_models
    for p in 1:n_pars
        mae_matrix[p, i] = mean(abs.(errors[p, i, :]))
    end
end

# Optionally, create a DataFrame for MAEs
mae_df = DataFrame(parameter = 1:n_pars)
for (i, m) in enumerate(k5_models.m)
    mae_df[!, Symbol("mae_m_$m")] = mae_matrix[:, i]
end

println("Mean Absolute Errors (MAEs) for each parameter and model:")
println(mae_df)