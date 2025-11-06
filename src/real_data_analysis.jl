include("mse_functions.jl")

## First analyse the Silverman data
silverman_data = Float32.(log.(load_silverman_data_6() .+ 1))
silverman_data = Float32.(log.(load_silverman_data() .+ 1))

model_silverman = load_model_nbe(
    n_lists = 5, 
    width = 256,
    n_hidden = 3,
    train_size = 10000,
    censoring_lower = 0, 
    censoring_upper = 0, 
    m = 1
)

ci_silverman = load_model_nbe(
    n_lists = 5, 
    width = 256,
    n_hidden = 3,
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




## Now analyse King data
king_data = load_king_data()

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


## compare with mcmc

include("mcmc_functions.jl")

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

res_df = DataFrame(chains)

intercept_median = median(res_df.intercept)
intercept_lower = quantile(res_df.intercept, 0.025)
intercept_upper = quantile(res_df.intercept, 0.975)

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