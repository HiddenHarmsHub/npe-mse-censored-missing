include("mse_functions.jl")
include("mcmc_functions.jl")

import Optim
using Optim: optimize, BFGS
using ForwardDiff, LinearAlgebra

output_path = joinpath("output", "real_data_analysis")
mkpath(output_path)

## Silverman data
silverman_data = Float32.(log.(load_silverman_data() .+ 1))

## First infer using NBE

## Architectures chosen by the pre-specified rule in architecture_selection.jl
selected_architectures = CSV.read(joinpath("output", "architecture_selection", "selected.csv"), DataFrame)
selected_architecture(system) = only(filter(:system => ==(system), selected_architectures))
best_NBE_5 = best_NPE_5 = selected_architecture("A")

model_silverman = load_model_nbe(
    n_lists = 5, 
    width = best_NBE_5.width,
    n_hidden = best_NBE_5.n_hidden,
    train_size = train_size_default,
    censoring_lower = 0, 
    censoring_upper = 0, 
    m = 1
)

ci_silverman = load_model_nbe(
    n_lists = 5, 
    width = best_NBE_5.width,
    n_hidden = best_NBE_5.n_hidden,
    train_size = train_size_default,
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
    train_size_default,
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

## MCMC on Silverman data

input_counts_silverman = load_silverman_data()
X = one_hot_encode_parameters(5)

intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 4)

censoring_lower = 0
censoring_upper = 0

## Reference posterior with the validated NUTS stages used in the simulation study (run_reference_fit)
priors = (intercept_dist = intercept_dist, beta_dist = beta_dist, gamma_dist = gamma_dist)
Random.seed!(2020)
chains_silverman, _, fit_silverman = run_reference_fit(input_counts_silverman, X; censoring_lower, censoring_upper, priors...)
CSV.write(joinpath(output_path, "silverman_mcmc_diagnostics.csv"), fit_silverman)
println("Silverman MCMC: ", fit_silverman.status[1], " at stage ", fit_silverman.stage[1])

silverman_mcmc_df = select(DataFrame(chains_silverman), :iteration, :chain, r"intercept|betas|gammas")
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



#
## Now analyse King data
#



king_data = load_king_data()

## Architectures for 4 lists censored on [1, 4], chosen by the same rule
best_NBE_4 = best_NPE_4 = selected_architecture("B")

model_king = load_model_nbe(
    n_lists = 4, 
    width = best_NBE_4.width,
    n_hidden = best_NBE_4.n_hidden,
    train_size = train_size_default, 
    censoring_lower = 1,
    censoring_upper = 4,
    m = 1
)

ci_king = load_model_nbe(
    n_lists = 4, 
    width = best_NBE_4.width,
    n_hidden = best_NBE_4.n_hidden,
    train_size = train_size_default, 
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
    train_size_default,
    1, 
    4, 
    1, 
    npe_models_path
)

posterior_samples_king_npe = bounded_sample(
    sampleposterior(npe_model_king, reshape(king_data, :, 1), n_samples), 
    1.0, 
    10.0
)

CSV.write(
    joinpath(output_path, "king_npe_posterior_samples.csv"),
    DataFrame(posterior_samples_king_npe', param_names(4))
)

## Raw counts with -1 for the suppressed cells
input_counts = load_king_counts()
X = one_hot_encode_parameters(4)

intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 4)

censoring_lower = 1
censoring_upper = 4

priors = (intercept_dist = intercept_dist, beta_dist = beta_dist, gamma_dist = gamma_dist)
Random.seed!(2021)
chains_king, _, fit_king = run_reference_fit(input_counts, X; censoring_lower, censoring_upper, priors...)
CSV.write(joinpath(output_path, "king_mcmc_diagnostics.csv"), fit_king)
println("King MCMC: ", fit_king.status[1], " at stage ", fit_king.stage[1])

res_df_king = select(DataFrame(chains_king), :iteration, :chain, r"intercept|betas|gammas")
rename!(res_df_king, mcmc_param_mapping(4))

CSV.write(
    joinpath(output_path, "king_mcmc_posterior_samples.csv"),
    res_df_king
)



## also model assesmment


ppd_king = hcat([simulate_data(x, 1, censoring_lower=0, censoring_upper=0, log_transform = false) for x in eachcol(posterior_samples_king_npe)]...)
ppd_king_df = DataFrame(ppd_king', ["N_$x" for x in replace.(enumerate_all_combinations(4), "," => "")]) 
CSV.write(
    joinpath(output_path, "king_npe_posterior_predictive.csv"),
    ppd_king_df
)


## MLEs: best of several local optimisations
function mle_estimates(counts, X, censoring_lower, censoring_upper, n_lists)
    neg_loglik(pars) = -likelihood_censored(counts, pars, X, censoring_lower, censoring_upper)
    n_pars = length(param_names(n_lists))
    starts = vcat(
        [log_count_start(counts, X, censoring_lower, censoring_upper, Uniform(-100, 100), Normal(0, 100), Normal(0, 100))],
        [randn(n_pars) for _ in 1:20]
    )
    results = [optimize(neg_loglik, start, BFGS(), Optim.Options(g_tol = 1e-8, iterations = 10_000); autodiff = :forward) for start in starts]
    result = results[argmin(Optim.minimum.(results))]
    Optim.converged(result) || @warn "The MLE optimisation did not converge."
    optimal_pars = Optim.minimizer(result)
    ## 95% Wald intervals from the observed information
    standard_errors = sqrt.(diag(inv(ForwardDiff.hessian(neg_loglik, optimal_pars))))
    return DataFrame(parameter = param_names(n_lists), estimate = optimal_pars,
                     lower_ci = optimal_pars .- 1.96 .* standard_errors, upper_ci = optimal_pars .+ 1.96 .* standard_errors)
end

Random.seed!(1)
CSV.write(joinpath(output_path, "king_mle_parameter_estimates.csv"), mle_estimates(input_counts, X, censoring_lower, censoring_upper, 4))
CSV.write(joinpath(output_path, "silverman_mle_parameter_estimates.csv"), mle_estimates(input_counts_silverman, one_hot_encode_parameters(5), 0, 0, 5))
