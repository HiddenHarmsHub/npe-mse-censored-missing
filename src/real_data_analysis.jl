include("mse_functions.jl")
include("mcmc_functions.jl")

## First analyse the Silverman data
#silverman_data = Float32.(log.(load_silverman_data_6() .+ 1))
silverman_data = Float32.(log.(load_silverman_data() .+ 1))

## First infer using NBE

function MAE_df(n_lists, censoring_upper)
    intercept_df = DataFrame(CSV.File(joinpath("output", "intercept_estimate_comparison.csv")))
    filtered_df = filter(x -> x.n_lists == n_lists .&& x.censoring_upper == censoring_upper, intercept_df)
    gdf = groupby(filtered_df, [:width, :n_hidden, :train_size])
    summary = combine(gdf, 
        [:intercept_NPE, :intercept_truth] => ((npe, truth) -> mean(abs.(npe .- truth))) => :MAE_NPE,
        [:intercept_NBE, :intercept_truth] => ((nbe, truth) -> mean(abs.(nbe .- truth))) => :MAE_NBE
    )
    return summary
end

## Find the best architectures for 5 lists
MAE_5_df = MAE_df(5, 0)
best_NBE_5 = MAE_5_df[findmin(MAE_5_df.MAE_NBE)[2], :]
best_NPE_5 = MAE_5_df[findmin(MAE_5_df.MAE_NPE)[2], :]

model_silverman = load_model_nbe(
    n_lists = 5, 
    width = best_NBE_5.width,
    n_hidden = best_NBE_5.n_hidden,
    train_size = 10000,
    censoring_lower = 0, 
    censoring_upper = 0, 
    m = 1
)

ci_silverman = load_model_nbe(
    n_lists = 5, 
    width = best_NBE_5.width,
    n_hidden = best_NBE_5.n_hidden,
    train_size = 10000,
    censoring_lower = 0, 
    censoring_upper = 0, 
    m = 1,
    ci = true
)

silverman_par_estimates = model_silverman(silverman_data)
silverman_par_cis = ci_silverman(silverman_data)

silverman_dark_figure = exp(silverman_par_estimates[1])
silverman_dark_figure_lower = exp(silverman_par_cis[1])
silverman_dark_figure_upper = exp(silverman_par_cis[length(silverman_par_estimates) + 1])

## Infer using NPE
npe_models_path = joinpath("output", "models_npe")
npe_model_silverman = load_model_npe(
    5, 
    best_NPE_5.width, 
    best_NPE_5.n_hidden,
    10000,
    0, 
    0, 
    1, 
    npe_models_path
)

bounded_sample(sample, lower, upper) = sample[:, (sample[1, :] .>= lower) .& (sample[1, :] .<= upper)]

n_samples = 25000
posterior_samples_npe = bounded_sample(sampleposterior(npe_model_silverman, reshape(silverman_data, :, 1), n_samples), 1.0, 10.0)

intercept_npe_median = median(posterior_samples_npe[1, :])

##  MCMC on Silverman data

input_counts_silverman = load_silverman_data()
X = one_hot_encode_parameters(5)

intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)

censoring_lower = 0
censoring_upper = 0

m = mse_model_censored(input_counts_silverman, X, intercept_dist, beta_dist, gamma_dist, censoring_lower, censoring_upper)
num_chains = 4
n_iterations = 5000
chains = sample(
    m, 
    NUTS(), 
    MCMCSerial(), 
    n_iterations, 
    num_chains, 
    progress = false,
    parallel = false
)

chains_df = DataFrame(chains)

median(chains_df.intercept)

using StatsPlots

npe_intercept_samples = vec(posterior_samples_npe[1, :])
mcmc_intercept_samples = Vector(chains_df.intercept)

# Extract beta samples from both methods
npe_beta_samples = posterior_samples_npe[2:6, :]  # rows 2-6 are the 5 betas
mcmc_beta_samples = Matrix(chains_df[:, [Symbol("betas[$i]") for i in 1:5]])  # Extract beta columns

# Create a 2x3 subplot layout (intercept + 5 betas = 6 plots)
plots = []

# Plot intercept
p_intercept = density(npe_intercept_samples, label = "NPE", color = :steelblue)
density!(p_intercept, mcmc_intercept_samples, label = "MCMC", color = :firebrick)
vline!(p_intercept, [silverman_par_estimates[1]], label = "NBE", linestyle = :dash, color = :black)
xlabel!(p_intercept, "Intercept")
ylabel!(p_intercept, "Density")
title!(p_intercept, "Intercept")
push!(plots, p_intercept)

# Plot each beta
for i in 1:5
    p_beta = density(npe_beta_samples[i, :], label = "NPE", color = :steelblue)
    density!(p_beta, mcmc_beta_samples[:, i], label = "MCMC", color = :firebrick)
    vline!(p_beta, [silverman_par_estimates[i+1]], label = "NBE", linestyle = :dash, color = :black)
    xlabel!(p_beta, "β$i")
    ylabel!(p_beta, "Density")
    title!(p_beta, "β$i")
    push!(plots, p_beta)
end

# Combine all plots
p = plot(plots..., layout = (2, 3), size = (1200, 800), 
         plot_title = "Posterior densities: Intercept and Betas (Silverman)",
         legend = :topright)
display(p)

## also simulate from posterior predictive

ppd_silverman = exp.(hcat([simulate_data(x, 5, censoring_lower=0, censoring_upper=0) for x in eachcol(posterior_samples_npe)]...)) .- 1

# Plot histograms of posterior predictive distribution
ppd_plots = []
for i in 1:5
    p = histogram(ppd_silverman[i, :], bins = 50, alpha = 0.7, 
                  label = "Posterior Predictive", color = :steelblue,
                  normalize = :probability)
    vline!(p, [exp(silverman_data[i]) - 1], label = "Observed", 
           linestyle = :dash, color = :firebrick, linewidth = 2)
    xlabel!(p, "Count")
    ylabel!(p, "Probability")
    title!(p, "List $i")
    push!(ppd_plots, p)
end

p_ppd = plot(ppd_plots..., layout = (2, 3), size = (1200, 800),
             plot_title = "Posterior Predictive Distribution (Silverman)")
display(p_ppd)



# Summarize point estimate and 95% CI for each method (Intercept, Silverman)
nbe_est = silverman_par_estimates[1]
nbe_lower = silverman_par_cis[1]
nbe_upper = silverman_par_cis[length(silverman_par_estimates) + 1]

npe_lower, npe_upper = quantile(npe_intercept_samples, (0.025, 0.975))
npe_est = median(npe_intercept_samples)

mcmc_lower, mcmc_upper = quantile(mcmc_intercept_samples, (0.025, 0.975))
mcmc_est = median(mcmc_intercept_samples)

methods = ["NBE", "NPE", "MCMC"]
estimates = [nbe_est, npe_est, mcmc_est]
lowers = [nbe_lower, npe_lower, mcmc_lower]
uppers = [nbe_upper, npe_upper, mcmc_upper]

x = 1:length(methods)
p_ci = scatter(
    x, estimates;
    yerror = (estimates .- lowers, uppers .- estimates),
    xticks = (x, methods),
    xlabel = "Method",
    ylabel = "Intercept",
    title = "Point estimate and 95% CI: Intercept (Silverman)",
    color = :black,
    marker = :circle,
    legend = false,
)
display(p_ci)



#
#
## Now analyse King data
#
#



king_data = load_king_data()

## Find the best architectures for 5 lists, 1-4 censoring
MAE_4_df = MAE_df(5, 0)
best_NBE_4 = MAE_5_df[findmin(MAE_5_df.MAE_NBE)[2], :]
best_NPE_4 = MAE_5_df[findmin(MAE_5_df.MAE_NPE)[2], :]

model_king = load_model_nbe(
    n_lists = 4, 
    width = 128,
    n_hidden = 3,
    train_size = 10000, 
    censoring_lower = 1,
    censoring_upper = 4,
    m = 1
)

ci_king = load_model_nbe(
    n_lists = 4, 
    width = 128,
    n_hidden = 3,
    train_size = 10000, 
    censoring_lower = 1,
    censoring_upper = 4,
    m = 1,
    ci = true
)

king_par_estimates = model_king(king_data)
king_par_cis = ci_king(king_data)

king_dark_figure = exp(king_par_estimates[1])
king_dark_figure_lower = exp(king_par_cis[1])
king_dark_figure_upper = exp(king_par_cis[length(king_par_estimates) + 1])


npe_model_king = load_model_npe(
    4, 
    128, 
    2,
    10000,
    1, 
    4, 
    1, 
    npe_models_path
)


posterior_samples_npe_king = bounded_sample(
    sampleposterior(npe_model_king, reshape(king_data, :, 1), n_samples), 
    1.0, 
    10.0
)

## compare with mcmc



king_data_reduced = king_data[1:15]
input_counts = Int.(ifelse.(king_data_reduced .== -1.0, -1.0, floor.(exp.(king_data_reduced) .- 1)))
X = one_hot_encode_parameters(5)

intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)
#gamma_dist = Normal(0, 4)
censoring_lower = 1
censoring_upper = 4

m = mse_model_censored(input_counts, X, intercept_dist, beta_dist, gamma_dist, censoring_lower, censoring_upper)
num_chains = 4
n_iterations = 5000
chains = sample(
    m, 
    NUTS(), 
    MCMCSerial(), 
    n_iterations, 
    num_chains, 
    progress = false,
    parallel = false
)

res_df_king = DataFrame(chains)

intercept_median = median(res_df_king.intercept)
intercept_lower = quantile(res_df_king.intercept, 0.025)
intercept_upper = quantile(res_df_king.intercept, 0.975)

println("Posterior median estimate of dark figure (MCMC): $(exp(intercept_median))")
println("95% credible interval for dark figure (MCMC): [$(exp(intercept_lower)), $(exp(intercept_upper))]")












using Plots
plot(res_df[:, :intercept], label = "Intercept Trace")

par_medians = vec(median(Matrix(res_df[:, 3:18]), dims = 1))
sum(exp.(X * par_medians))



## try uncensored version
input_counts_min = ifelse.(input_counts .== -1, 1, input_counts)
m_uncensored = mse_model(input_counts_min, X, intercept_dist, beta_dist, gamma_dist)
chains_uncensored = sample(
    m_uncensored, 
    NUTS(), 
    MCMCSerial(), 
    n_iterations, 
    num_chains, 
    progress = false,
    parallel = false
)

res_df_uncensored = DataFrame(chains_uncensored)
intercept_median_uncens = median(res_df_uncensored.intercept)
intercept_lower_uncens = quantile(res_df_uncensored.intercept, 0.025)
intercept_upper_uncens = quantile(res_df_uncensored.intercept, 0.975)

println("Posterior median estimate of dark figure (MCMC uncensored): $(exp(intercept_median_uncens))")
println("95% credible interval for dark figure (MCMC uncensored): [$(exp(intercept_lower_uncens)), $(exp(intercept_upper_uncens))]")




## try mcmc with some terms filtered
par_names = get_param_names(4)

gamma_terms =  ["1,2", "1,3", "2,3", "1,4", "3,4"]
keep_pars = ["intercept", ["beta_$(i)" for i in 1:4]... , ["gamma_$(name)" for name in gamma_terms]...]
filtered_indexes = findall(name -> name in keep_pars, par_names)
X_reduced = X[:, filtered_indexes]

m_filtered = mse_model_filtered(input_counts_min, X_reduced, intercept_dist, beta_dist, gamma_dist, 4, filtered_indexes)
chains_filtered = sample(
    m_filtered, 
    NUTS(), 
    MCMCSerial(), 
    n_iterations, 
    num_chains, 
    progress = false,
    parallel = false
)

res_df_filtered = DataFrame(chains_filtered)
intercept_median_filt = median(res_df_filtered.intercept)
intercept_lower_filt = quantile(res_df_filtered.intercept, 0.025)
intercept_upper_filt = quantile(res_df_filtered.intercept, 0.975)

println(describe(res_df_filtered))


dark_number = exp(intercept_median_filt)
observed_sum = sum(input_counts_min)
println("Posterior median estimate of dark figure (MCMC filtered): $(exp(intercept_median_filt))")
println("95% credible interval for dark figure (MCMC filtered): [$(exp(intercept_lower_filt)), $(exp(intercept_upper_filt))]")
println("Abundance = $(dark_number + observed_sum)")