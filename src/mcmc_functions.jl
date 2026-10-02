using Turing, LinearAlgebra, ForwardDiff
import Turing.MCMCChains.MCMCDiagnosticTools: bfmi
import Distributions: logfactorial

const AHMC = Turing.Inference.AHMC

include("irls_functions.jl")

@model function mse_model_censored(y, X, intercept_dist, beta_dist, gamma_dist, censoring_lower, censoring_upper)
    n_pars = size(X, 2)
    K = Int(-0.5 + (sqrt(8 * (n_pars - 1) + 1) / 2))  # Solve for K given length of params
    intercept ~ intercept_dist
    betas ~ filldist(beta_dist, K)
    gammas ~ filldist(gamma_dist, binomial(K, 2))

    params = vcat(intercept, betas, gammas)

    Turing.@addlogprob!(likelihood_censored(y, params, X, censoring_lower, censoring_upper))
end

## Sampled in standardised coordinates z, with θ = mode + L z and L L' the Laplace covariance.
## The intercept stays on its original scale and its prior support is enforced by rejection:
## Turing's logit transform of the bounded intercept bends the narrow ridge along which the
## intercept trades off against the main effects, and cell counts spanning many orders of
## magnitude leave posterior scales that warm-up adaptation alone cannot learn.
@model function mse_model_censored_laplace(y, X, mode, L, intercept_dist, beta_dist, gamma_dist, censoring_lower, censoring_upper)
    z ~ filldist(Flat(), length(mode))
    θ = mode .+ L * z
    Turing.@addlogprob! log_posterior(θ, y, X, censoring_lower, censoring_upper; intercept_dist, beta_dist, gamma_dist)
end

@model function mse_model(y, X, intercept_dist, beta_dist, gamma_dist)
    n_pars = size(X, 2)
    K = Int(-0.5 + (sqrt(8 * (n_pars - 1) + 1) / 2))  # Solve for K given length of params
    intercept ~ intercept_dist
    betas ~ filldist(beta_dist, K)
    gammas ~ filldist(gamma_dist, binomial(K, 2))

    params = vcat(intercept, betas, gammas)

    for i in eachindex(y)
        logλ = sum(X[i, :] .* params)
        y[i] ~ Poisson(exp(logλ))
    end
end

@model function mse_model_filtered(y, X, intercept_dist, beta_dist, gamma_dist, K, filter_indices)
    n_pars = size(X, 2)
    intercept ~ intercept_dist
    betas ~ filldist(beta_dist, K)
    gammas ~ filldist(gamma_dist, length(filter_indices) - K - 1)

    params = vcat(intercept, betas, gammas)

    for i in eachindex(y)
        logλ = sum(X[i, :] .* params)
        y[i] ~ Poisson(exp(logλ))
    end
end

## Log-likelihood up to a constant over observed cells, written relative to the saturated fit
## so it stays O(number of cells) when counts are very large.
## Cells with count -1 are interval censored and contribute log P(censoring_lower <= N <= censoring_upper).
function likelihood_censored(counts::AbstractVector{<:Integer}, pars::AbstractVector, X::AbstractMatrix, censoring_lower::Integer, censoring_upper::Integer)
    η = X * pars
    ll = zero(eltype(η))
    for (logλ, count) in zip(η, counts)
        if count == -1
            ll += log_interval_prob(logλ, censoring_lower, censoring_upper)
        else
            ll += poisson_kernel(count, logλ)
        end
    end
    return ll
end

## logpdf(Poisson(exp(η)), y) + logfactorial(y) - (y log y - y), written with δ = η - log y as
## y (δ - expm1(δ)) so it keeps full precision for counts up to the simulator cap (~e^43)
poisson_kernel(y, η) = y > 0 ? y * ((η - log(y)) - expm1(η - log(y))) : -exp(η)

function log_interval_prob(logλ, lower::Integer, upper::Integer)
    lp = [k * logλ - exp(logλ) - logfactorial(k) for k in lower:upper]
    m = maximum(lp)
    return m + log(sum(exp.(lp .- m)))
end

log_prior(pars, K, intercept_dist, beta_dist, gamma_dist) =
    logpdf(intercept_dist, pars[1]) + sum(logpdf.(beta_dist, pars[2:1+K])) + sum(logpdf.(gamma_dist, pars[2+K:end]))

## K is inferred from a full parameter vector; pass it for a submodel whose X keeps only some interaction columns
function log_posterior(pars, counts, X, censoring_lower, censoring_upper; intercept_dist = Uniform(1, 10), beta_dist = Normal(0, 4), gamma_dist = Normal(0, 4), K = nothing)
    K = isnothing(K) ? Int(-0.5 + (sqrt(8 * (length(pars) - 1) + 1) / 2)) : K
    lp = log_prior(pars, K, intercept_dist, beta_dist, gamma_dist)
    isfinite(lp) || return lp
    return lp + likelihood_censored(counts, pars, X, censoring_lower, censoring_upper)
end

## Weighted ridge least squares on log counts, used as a deterministic starting point for Newton.
## Censored cells take the midpoint of the censoring interval and the prior acts as the ridge.
function log_count_start(counts, X, censoring_lower, censoring_upper, intercept_dist, beta_dist, gamma_dist; K = nothing)
    n_pars = size(X, 2)
    K = isnothing(K) ? Int(-0.5 + (sqrt(8 * (n_pars - 1) + 1) / 2)) : K
    ỹ = ifelse.(counts .== -1, (censoring_lower + censoring_upper) / 2, counts) .+ 0.5
    prior_mean = vcat(mean(intercept_dist), fill(mean(beta_dist), K), fill(mean(gamma_dist), n_pars - K - 1))
    prior_prec = 1 ./ vcat(var(intercept_dist), fill(var(beta_dist), K), fill(var(gamma_dist), n_pars - K - 1))
    sw = sqrt.(ỹ)
    θ = vcat(sw .* X, Diagonal(sqrt.(prior_prec))) \ vcat(sw .* log.(ỹ), sqrt.(prior_prec) .* prior_mean)
    lower, upper = minimum(intercept_dist), maximum(intercept_dist)
    θ[1] = clamp(θ[1], lower + 0.01 * (upper - lower), upper - 0.01 * (upper - lower))
    return θ
end

## Posterior mode by Newton's method, and the Laplace factor L with L L' = H⁻¹.
## Eigenvalues of the negative Hessian are floored so flat directions stay bounded.
function laplace_approximation(counts, X; censoring_lower = 0, censoring_upper = 0, intercept_dist = Uniform(1, 10), beta_dist = Normal(0, 4), gamma_dist = Normal(0, 4), max_iter = 200, min_precision = 1e-2, K = nothing)
    priors = (intercept_dist = intercept_dist, beta_dist = beta_dist, gamma_dist = gamma_dist)
    f(θ) = log_posterior(θ, counts, X, censoring_lower, censoring_upper; priors..., K)
    function precision_factor(h, θ)
        E = eigen(Symmetric(-ForwardDiff.hessian(h, θ)))
        return E.vectors, max.(E.values, min_precision)
    end
    function newton(h, x, n_iter)
        for _ in 1:n_iter
            V, λ = precision_factor(h, x)
            g = ForwardDiff.gradient(h, x)
            step = V * ((V' * g) ./ λ)
            dot(g, step) < 1e-10 && break  # Newton decrement
            h0, t = h(x), 1.0
            while !(h(x .+ t .* step) >= h0) && t > 1e-12
                t /= 2
            end
            t > 1e-12 || break
            x = x .+ t .* step
        end
        return x
    end

    ## Search with the intercept on the logit scale so steps cannot leave its support,
    ## then polish on the original scale where the ridge with the main effects is straight
    lower, upper = minimum(intercept_dist), maximum(intercept_dist)
    to_θ(φ) = vcat(lower + (upper - lower) / (1 + exp(-φ[1])), φ[2:end])
    θ0 = log_count_start(counts, X, censoring_lower, censoring_upper, intercept_dist, beta_dist, gamma_dist; K)
    φ = newton(φ -> f(to_θ(φ)), vcat(log((θ0[1] - lower) / (upper - θ0[1])), θ0[2:end]), max_iter)
    θ = newton(f, to_θ(φ), 20)
    V, λ = precision_factor(f, θ)
    return θ, V * Diagonal(1 ./ sqrt.(λ))
end

## Default sampler settings, then the remediation stages used when a fit fails the diagnostics
const nuts_stages = [
    (n_adapts = 1000, n_samples = 4000, δ = 0.65, max_depth = 10, metric = :diag, parameterisation = :logit),
    (n_adapts = 2000, n_samples = 4000, δ = 0.9, max_depth = 12, metric = :dense, parameterisation = :laplace),
    (n_adapts = 5000, n_samples = 10000, δ = 0.95, max_depth = 12, metric = :dense, parameterisation = :laplace),
]

function fit_nuts(
    counts,
    X;
    censoring_lower = 0,
    censoring_upper = 0,
    intercept_dist = Uniform(1, 10),
    beta_dist = Normal(0, 4),
    gamma_dist = Normal(0, 4),
    n_adapts = 1000,
    n_samples = 4000,
    n_chains = 4,
    δ = 0.65,
    max_depth = 10,
    metric = :diag,
    parameterisation = :logit,
    laplace = nothing
)
    metricT = metric == :dense ? AHMC.DenseEuclideanMetric : AHMC.DiagEuclideanMetric
    sampler = NUTS(n_adapts, δ; max_depth = max_depth, metricT = metricT)
    if parameterisation == :logit
        m = mse_model_censored(counts, X, intercept_dist, beta_dist, gamma_dist, censoring_lower, censoring_upper)
        return sample(m, sampler, MCMCSerial(), n_samples, n_chains; progress = false)
    end

    mode, L = isnothing(laplace) ? laplace_approximation(counts, X; censoring_lower, censoring_upper, intercept_dist, beta_dist, gamma_dist) : laplace
    m = mse_model_censored_laplace(counts, X, mode, L, intercept_dist, beta_dist, gamma_dist, censoring_lower, censoring_upper)
    inits = map(1:n_chains) do _
        z = 0.5 .* randn(length(mode))
        while !isfinite(logpdf(intercept_dist, (mode .+ L * z)[1]))
            z = 0.5 .* randn(length(mode))
        end
        z
    end
    chains = sample(m, sampler, MCMCSerial(), n_samples, n_chains; progress = false, initial_params = inits)
    return laplace_to_parameters(chains, mode, L)
end

## Map draws of z back to (intercept, betas, gammas), keeping the sampler statistics
function laplace_to_parameters(chains, mode, L)
    n_pars = length(mode)
    K = Int(-0.5 + (sqrt(8 * (n_pars - 1) + 1) / 2))
    z = chains[:, [Symbol("z[$i]") for i in 1:n_pars], :].value.data
    θ = similar(z)
    for c in axes(z, 3), t in axes(z, 1)
        θ[t, :, c] = mode .+ L * z[t, :, c]
    end
    internals = names(chains, :internals)
    vals = cat(θ, chains[:, internals, :].value.data; dims = 2)
    par_names = vcat("intercept", ["betas[$i]" for i in 1:K], ["gammas[$i]" for i in 1:(n_pars - K - 1)], string.(internals))
    return Chains(vals, par_names, Dict(:internals => string.(internals)); iterations = range(chains))
end

## Thresholds fixed before any method comparison (Vehtari et al., 2021; Betancourt, 2016)
const diagnostic_thresholds = (rhat = 1.01, ess = 400, bfmi = 0.3)

function mcmc_diagnostics(chains; max_depth = 10)
    par_df = DataFrame(summarystats(chains))
    internals = names(chains, :internals)
    ## Non-finite energy errors are rejections at the edge of the intercept support, not divergences
    if :numerical_error in internals
        numerical_error = Array(chains[:numerical_error]) .> 0
        finite_error = isfinite.(Array(chains[:max_hamiltonian_energy_error]))
        n_divergent = sum(numerical_error .& finite_error)
        n_rejected = sum(numerical_error .& .!finite_error)
    else
        n_divergent, n_rejected = 0, 0
    end
    treedepth_hits = :tree_depth in internals ? Int(sum(chains[:tree_depth] .>= max_depth)) : 0
    min_bfmi = :hamiltonian_energy in internals ? minimum(bfmi(Array(chains[:hamiltonian_energy]))) : NaN
    fit_df = DataFrame(
        max_rhat = maximum(par_df.rhat),
        min_ess_bulk = minimum(par_df.ess_bulk),
        min_ess_tail = minimum(par_df.ess_tail),
        n_divergent = n_divergent,
        n_rejected = n_rejected,
        treedepth_hits = treedepth_hits,
        min_bfmi = min_bfmi
    )
    return par_df, fit_df
end

function passes_diagnostics(fit_df)
    r = fit_df[1, :]
    r.max_rhat < diagnostic_thresholds.rhat &&
        r.min_ess_bulk >= diagnostic_thresholds.ess &&
        r.min_ess_tail >= diagnostic_thresholds.ess &&
        r.n_divergent == 0 &&
        (isnan(r.min_bfmi) || r.min_bfmi >= diagnostic_thresholds.bfmi)
end

## Run the NUTS stages in order until one passes the diagnostics
function run_reference_fit(counts, X; n_chains = 4, stages = nuts_stages, kwargs...)
    chains, par_df, fit_df = nothing, nothing, nothing
    total_time = 0.0
    laplace = nothing
    for (s, stage) in enumerate(stages)
        start = time_ns()
        if stage.parameterisation == :laplace && isnothing(laplace)
            laplace = laplace_approximation(counts, X; kwargs...)
        end
        chains = fit_nuts(
            counts, X; kwargs...,
            n_adapts = stage.n_adapts, n_samples = stage.n_samples, n_chains = n_chains,
            δ = stage.δ, max_depth = stage.max_depth, metric = stage.metric,
            parameterisation = stage.parameterisation, laplace = laplace
        )
        total_time += (time_ns() - start) / 1.0e9
        par_df, fit_df = mcmc_diagnostics(chains; max_depth = stage.max_depth)
        fit_df.stage .= s - 1
        fit_df.time .= total_time
        if passes_diagnostics(fit_df)
            fit_df.status .= s == 1 ? "initial" : "remediated"
            return chains, par_df, fit_df
        end
    end
    fit_df.status .= "failed"
    return chains, par_df, fit_df
end

function chains_samples_df(chains; max_draws_per_chain = 1000)
    df = DataFrame(chains)
    thin = max(1, cld(size(chains, 1), max_draws_per_chain))
    df = filter(:iteration => i -> (i - first(df.iteration)) % thin == 0, df)
    return select(df, :chain, :iteration, r"intercept|betas|gammas")
end

function posterior_summary_df(chains, par_df, truth)
    draws = select(DataFrame(chains), r"intercept|betas|gammas")
    return DataFrame(
        parameters = par_df.parameters,
        true_values = vec(truth),
        estimated_means = mean.(eachcol(draws)),
        estimated_medians = median.(eachcol(draws)),
        estimated_std = std.(eachcol(draws)),
        lower_95ci = quantile.(eachcol(draws), 0.025),
        upper_95ci = quantile.(eachcol(draws), 0.975),
        rhat = par_df.rhat,
        ess_bulk = par_df.ess_bulk,
        ess_tail = par_df.ess_tail,
        mcse = par_df.mcse
    )
end

## Fit one benchmark dataset with the NUTS ladder, the IRLS sampler and importance sampling and write the outputs
function run_mcmc_test_slice(slice_idx, test_counts, test_pars; censoring_lower = 0, censoring_upper = 10, paths)
    println("Running reference fits for slice $slice_idx...")
    counts = test_counts[:, slice_idx]
    truth = test_pars[:, slice_idx]
    X = one_hot_encode_parameters(Int(log2(length(counts) + 1)))
    meta = (dataset = slice_idx, capped = is_capped(truth), n_censored = sum(counts .== -1))

    chains, par_df, fit_df = run_reference_fit(counts, X; censoring_lower = censoring_lower, censoring_upper = censoring_upper)
    CSV.write(joinpath(paths.mcmc_samples, "mcmc_test_results_$slice_idx.csv"), chains_samples_df(chains))
    CSV.write(joinpath(paths.mcmc_summary, "mcmc_test_summary_$slice_idx.csv"), posterior_summary_df(chains, par_df, truth))
    CSV.write(joinpath(paths.mcmc_diagnostics, "mcmc_test_diagnostics_$slice_idx.csv"), hcat(DataFrame([meta]), fit_df))

    start = time_ns()
    irls_chains = fit_irls(counts, X; censoring_lower = censoring_lower, censoring_upper = censoring_upper)
    irls_time = (time_ns() - start) / 1.0e9
    irls_par_df, irls_fit_df = mcmc_diagnostics(irls_chains)
    irls_fit_df.acceptance_rate .= mean(irls_chains[:acceptance_rate])
    irls_fit_df.time .= irls_time
    irls_fit_df.status .= passes_diagnostics(irls_fit_df) ? "passed" : "failed"
    CSV.write(joinpath(paths.irls_samples, "irls_test_results_$slice_idx.csv"), chains_samples_df(irls_chains))
    CSV.write(joinpath(paths.irls_summary, "irls_test_summary_$slice_idx.csv"), posterior_summary_df(irls_chains, irls_par_df, truth))
    CSV.write(joinpath(paths.irls_diagnostics, "irls_test_diagnostics_$slice_idx.csv"), hcat(DataFrame([meta]), irls_fit_df))

    ## IRLS-MH mixes poorly on sparse or heavily censored data, so the IS proposal is built from
    ## whichever sampler passed; the IS weights use the exact posterior either way
    pilot = irls_fit_df.status[1] == "passed" ? "IRLS" : "NUTS"
    pilot_draws = Array(select(DataFrame(pilot == "IRLS" ? irls_chains : chains), r"intercept|betas|gammas"))
    start = time_ns()
    is_ref = importance_reference(counts, X; censoring_lower = censoring_lower, censoring_upper = censoring_upper, n_draws = 2 * 10^5, pilot_draws = pilot_draws)
    CSV.write(joinpath(paths.is_summary, "is_test_summary_$slice_idx.csv"), DataFrame(
        parameters = par_df.parameters,
        true_values = vec(truth),
        estimated_means = is_ref.means,
        estimated_std = is_ref.sds,
        lower_95ci = is_ref.quantiles[:, 1],
        estimated_medians = is_ref.quantiles[:, 2],
        upper_95ci = is_ref.quantiles[:, 3],
        ess = is_ref.ess,
        pilot = pilot,
        time = (time_ns() - start) / 1.0e9
    ))
end
