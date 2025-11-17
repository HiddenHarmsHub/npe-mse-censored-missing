include("mse_functions.jl")
include("mcmc_functions.jl")


output_path = joinpath("output", "real_data_analysis")
mkpath(output_path)

## First analyse the Silverman data
#silverman_data = Float32.(log.(load_silverman_data_6() .+ 1))
silverman_data = Float32.(log.(load_silverman_data() .+ 1))

## First infer using NBE

intercept_files = readdir(joinpath("output", "intercept_estimates"))

function get_intercept_df(intercept_files)
    map(intercept_files) do intercept_file
        DataFrame(CSV.File(joinpath("output", "intercept_estimates", intercept_file)))
    end |> (x -> vcat(x...))
end

function MAE_df(n_lists, censoring_lower, censoring_upper, intercept_files)
    intercept_df = get_intercept_df(intercept_files)
    filtered_df = filter(x -> x.n_lists == n_lists .&& x.censoring_lower == censoring_lower .&& x.censoring_upper == censoring_upper, intercept_df)
    gdf = groupby(filtered_df, [:width, :n_hidden, :train_size])
    summary = combine(gdf, 
        [:intercept_NPE, :intercept_truth] => ((npe, truth) -> mean(abs.(npe .- truth))) => :MAE_NPE,
        [:intercept_NBE, :intercept_truth] => ((nbe, truth) -> mean(abs.(nbe .- truth))) => :MAE_NBE
    )
    return summary
end

filter(x -> x.censoring_upper == 4, intercept_df)

## Find the best architectures for 5 lists
MAE_5_df = MAE_df(5, 0, 0, intercept_files)
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

silverman_NBE_estimates = DataFrame(
    parameter = param_names(5),
    estimate = silverman_par_estimates,
    lower_ci = silverman_par_cis[1:length(silverman_par_estimates)],
    upper_ci = silverman_par_cis[length(silverman_par_estimates) .+ (1:length(silverman_par_estimates))]
)

CSV.write(
    joinpath(output_path, "silverman_nbe_parameter_estimates.csv"),
    silverman_NBE_estimates
)

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

n_samples = 25000
posterior_samples_silverman_npe = bounded_sample(sampleposterior(npe_model_silverman, reshape(silverman_data, :, 1), n_samples), 1.0, 10.0)

silverman_NPE_estimates = DataFrame(
    parameter = param_names(5),
    estimate = median(posterior_samples_silverman_npe, dims = 2)[:],
    lower_ci = quantile.(eachrow(posterior_samples_silverman_npe), 0.025),
    upper_ci = quantile.(eachrow(posterior_samples_silverman_npe), 0.975)
)

CSV.write(
    joinpath(output_path, "silverman_npe_parameter_estimates.csv"),
    silverman_NPE_estimates
)

CSV.write(
    joinpath(output_path, "silverman_npe_posterior_samples.csv"),
    DataFrame(posterior_samples_silverman_npe', param_names(5))
)

##  MCMC on Silverman data

input_counts_silverman = load_silverman_data()
X = one_hot_encode_parameters(5)

intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)

censoring_lower = 0
censoring_upper = 0

m_silverman = mse_model_censored(input_counts_silverman, X, intercept_dist, beta_dist, gamma_dist, censoring_lower, censoring_upper)
num_chains = 4
n_iterations = 5000
chains_silverman = sample(
    m_silverman, 
    NUTS(), 
    MCMCSerial(), 
    n_iterations, 
    num_chains, 
    progress = false,
    parallel = false
)

silverman_mcmc_df = DataFrame(chains_silverman)
mcmc_param_mapping(n_lists) = ["intercept"; ["betas[$(i)]" for i in 1:n_lists]; ["gammas[$(i)]" for i in 1:binomial(n_lists,2)]...] .=> param_names(n_lists)
rename!(silverman_mcmc_df, mcmc_param_mapping(5))

CSV.write(
    joinpath(output_path, "silverman_mcmc_posterior_samples.csv"),
    silverman_mcmc_df
)

## simulate from the posterior predictive 

ppd_silverman = hcat([simulate_data(x, 1, censoring_lower=0, censoring_upper=0, log_transform = false) for x in eachcol(posterior_samples_silverman_npe)]...)
ppd_silverman_df = DataFrame(ppd_silverman', ["N_$x" for x in replace.(enumerate_all_combinations(5), "," => "")]) 
CSV.write(
    joinpath(output_path, "silverman_npe_posterior_predictive.csv"),
    ppd_silverman_df
)



########################
#
## Now analyse King data
#
#########################



king_data = load_king_data()

## Find the best architectures for 5 lists, 1-4 censoring
MAE_4_df = MAE_df(4, 1, 4, intercept_files)
best_NBE_4 = MAE_5_df[findmin(MAE_5_df.MAE_NBE)[2], :]
best_NPE_4 = MAE_5_df[findmin(MAE_5_df.MAE_NPE)[2], :]

model_king = load_model_nbe(
    n_lists = 4, 
    width = best_NBE_4.width,
    n_hidden = best_NBE_4.n_hidden,
    train_size = 10000, 
    censoring_lower = 1,
    censoring_upper = 4,
    m = 1
)

ci_king = load_model_nbe(
    n_lists = 4, 
    width = best_NBE_4.width,
    n_hidden = best_NBE_4.n_hidden,
    train_size = 10000, 
    censoring_lower = 1,
    censoring_upper = 4,
    m = 1,
    ci = true
)

king_par_estimates = model_king(king_data)
king_par_cis = ci_king(king_data)

king_NBE_estimates = DataFrame(
    parameter = param_names(4),
    estimate = king_par_estimates,
    lower_ci = king_par_cis[1:length(king_par_estimates)],
    upper_ci = king_par_cis[length(king_par_estimates)+1:end]
)

CSV.write(
    joinpath(output_path, "king_nbe_parameter_estimates.csv"),
    king_NBE_estimates
)

npe_model_king = load_model_npe(
    4, 
    best_NPE_4.width, 
    best_NPE_4.n_hidden,
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

CSV.write(
    joinpath(output_path, "king_npe_posterior_samples.csv"),
    DataFrame(posterior_samples_npe_king', param_names(4))
)

## compare with mcmc
king_data_reduced = king_data[1:15]
input_counts = Int.(ifelse.(king_data_reduced .== -1.0, -1.0, floor.(exp.(king_data_reduced) .- 1)))
X = one_hot_encode_parameters(5)

intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)

censoring_lower = 1
censoring_upper = 4

m_king = mse_model_censored(input_counts, X, intercept_dist, beta_dist, gamma_dist, censoring_lower, censoring_upper)
num_chains = 4
n_iterations = 5000
chains_king = sample(
    m_king, 
    NUTS(), 
    MCMCSerial(), 
    n_iterations, 
    num_chains, 
    progress = false,
    parallel = false
)

res_df_king = DataFrame(chains_king)
rename!(res_df_king, mcmc_param_mapping(4))

CSV.write(
    joinpath(output_path, "king_mcmc_posterior_samples.csv"),
    res_df_king
)




