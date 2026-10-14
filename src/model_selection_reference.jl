## Reference posterior model probabilities by enumerating every structure, with marginal likelihoods by importance sampling
import Optim

## Log marginal likelihood of structure `mask`, integrating over the active parameters only.
function log_evidence(counts, mask; censoring_lower = 0, censoring_upper = 0, priors = coefficient_priors(), n_draws = 50_000, n_keep = 500, ν = 4, scale = 1.5, pilot_draws = nothing, X = nothing, M = nothing, rng = Random.default_rng())
    K = Int(log2(length(counts) + 1))
    active = vcat(trues(1 + K), mask)
    Xm = (isnothing(X) ? one_hot_encode_parameters(K) : X)[:, active]
    is = importance_reference(counts, Xm; censoring_lower, censoring_upper, priors..., n_draws, ν, scale, pilot_draws, K, rng)
    keep = rand(rng, Categorical(is.weights), n_keep)
    θ = zeros(Float32, length(active), n_keep)
    θ[active, :] = is.draws[:, keep]
    isnothing(M) || (θ = Float32.(M * θ))
    return (log_evidence = is.log_evidence, log_evidence_se = is.log_evidence_se, ess = is.ess, θ = θ)
end

## Second, independent proposal: a heavier-tailed fit to the first run's weighted draws.
function log_evidence_check(counts, mask; kwargs...)
    first = log_evidence(counts, mask; kwargs...)
    K = Int(log2(length(counts) + 1))
    active = vcat(trues(1 + K), mask)
    pilot = Float64.(permutedims(first.θ[active, :]))
    second = log_evidence(counts, mask; kwargs..., pilot_draws = pilot, ν = 10, scale = 1.2)
    return (laplace = first.log_evidence, laplace_se = first.log_evidence_se, laplace_ess = first.ess,
            pilot = second.log_evidence, pilot_se = second.log_evidence_se, pilot_ess = second.ess)
end

## Marginal likelihoods of all 2^J structures; reused for every model prior
reference_evidences(counts, system; priors = coefficient_priors(), n_draws = 50_000, n_keep = 500, rng = Random.default_rng()) =
    [log_evidence(counts, model_mask(c, system.K); system.censoring_lower, system.censoring_upper, priors, n_draws, n_keep, rng) for c in 1:n_models(system.K)]

## Reference Bayesian model average under `model_prior`, from the enumerated evidences
function reference_mixture(fits, counts, system; model_prior = primary_model_prior, B = 4000, rng = Random.default_rng())
    K = system.K
    log_ev = [f.log_evidence for f in fits]
    logp = log_ev .+ model_prior_logpmf(model_prior, K)
    π = exp.(logp .- maximum(logp))
    π ./= sum(π)

    n_keep = size(fits[1].θ, 2)
    n_per_model = rand(rng, Multinomial(B, π))
    models = reduce(vcat, [fill(c, n) for (c, n) in enumerate(n_per_model) if n > 0])
    θ_eff = reduce(hcat, [fits[c].θ[:, rand(rng, 1:n_keep, n)] for (c, n) in enumerate(n_per_model) if n > 0])
    pop = population_draws(θ_eff, counts; system.censoring_lower, system.censoring_upper, rng)
    return (π = π, log_evidence = log_ev, log_evidence_se = [f.log_evidence_se for f in fits], ess = [f.ess for f in fits],
            inclusion = enumerate_models(K) * π, models = models, θ_eff = θ_eff, pop...)
end

## Reference Bayesian model averaging over all 2^J structures
function reference_model_average(counts, system; model_prior = primary_model_prior, priors = coefficient_priors(), n_draws = 50_000, n_keep = 500, B = 4000, rng = Random.default_rng())
    fits = reference_evidences(counts, system; priors, n_draws, n_keep, rng)
    return reference_mixture(fits, counts, system; model_prior, B, rng)
end

## Maximum likelihood fit of one structure, with a Wald interval for λ0 and the information criteria.
function fit_mle(counts, mask; censoring_lower = 0, censoring_upper = 0, priors = coefficient_priors())
    K = Int(log2(length(counts) + 1))
    Xm = one_hot_encode_parameters(K)[:, vcat(trues(1 + K), mask)]
    negloglik(θ) = -likelihood_censored(counts, θ, Xm, censoring_lower, censoring_upper)
    start = log_count_start(counts, Xm, censoring_lower, censoring_upper, priors...; K)
    result = Optim.optimize(negloglik, start, Optim.LBFGS(), Optim.Options(iterations = 10_000, g_tol = 1e-9); autodiff = :forward)
    θ̂ = Optim.minimizer(result)
    H = ForwardDiff.hessian(negloglik, θ̂)
    se = isposdef(Symmetric(H)) ? sqrt(inv(Symmetric(H))[1, 1]) : NaN
    censored_mean(η) = (k = censoring_lower:censoring_upper; l = k .* η .- logfactorial.(k); w = exp.(l .- maximum(l)); sum(k .* w) / sum(w))
    N̂ = exp(θ̂[1]) + sum(counts[counts .>= 0]) + sum(censored_mean(η) for η in Xm[counts .== -1, :] * θ̂; init = 0.0)
    loglik, p = -Optim.minimum(result), length(θ̂)
    return (N = N̂, converged = Optim.converged(result), alpha = θ̂[1], alpha_se = se, lambda0 = exp(θ̂[1]),
            lambda0_lower = exp(θ̂[1] - 1.96se), lambda0_upper = exp(θ̂[1] + 1.96se), min_coefficient = minimum(θ̂[2:end]),
            n_parameters = p, loglik = loglik, AIC = 2p - 2loglik, BIC_cells = log(length(counts)) * p - 2loglik,
            BIC_individuals = log(sum(counts[counts .>= 0])) * p - 2loglik)
end

## ---- g-priors (prior sensitivity) ----
## The default priors of conting (Overstall and King 2014)
function sum_to_zero_map(K)
    pairs = enumerate_two_digit_numbers(K)
    M = zeros(1 + K + length(pairs), 1 + K + length(pairs))
    M[1, 1] = 1
    for i in 1:K
        M[1+i, 1+i] = 2
    end
    for (j, (a, b)) in enumerate(pairs)
        M[1+K+j, 1+K+j] = 4
        M[1+a, 1+K+j] = M[1+b, 1+K+j] = -2
    end
    return M  # corner-point θ = M u, for u with inactive interactions set to zero
end

gprior_priors(σ²) = (intercept_dist = Uniform(1, 10), beta_dist = Normal(0, sqrt(σ²)), gamma_dist = Normal(0, sqrt(σ²)))

## Log evidences of every structure under the g-prior with fixed σ²; draws are returned as corner-point θ
function gprior_evidences(counts, system, σ²; n_draws = 50_000, n_keep = 500, rng = Random.default_rng())
    K = system.K
    M = sum_to_zero_map(K)
    X = one_hot_encode_parameters(K) * M
    return [log_evidence(counts, model_mask(c, K); system.censoring_lower, system.censoring_upper, priors = gprior_priors(σ²),
                         n_draws, n_keep, X, M, rng) for c in 1:n_models(K)]
end

## Hyper-g: integrate the evidence over τ = log σ² with σ² ~ IG(a, b)
function hyper_g_evidence(log_ev_grid, τ_grid; a = 5e-4, b = 5e-4, τ_max = 3000.0, step = 0.05)
    log_prior_τ(τ) = a * log(b) - Distributions.loggamma(a) - a * τ - b * exp(-τ)
    slope = (log_ev_grid[end] - log_ev_grid[end-1]) / (τ_grid[end] - τ_grid[end-1])
    τ_fine = collect(τ_grid[1]:step:τ_max)
    interp(τ) = τ <= τ_grid[end] ? (k = clamp(searchsortedlast(τ_grid, τ), 1, length(τ_grid) - 1);
                                    w = (τ - τ_grid[k]) / (τ_grid[k+1] - τ_grid[k]); (1 - w) * log_ev_grid[k] + w * log_ev_grid[k+1]) :
                                   log_ev_grid[end] + min(slope, 0.0) * (τ - τ_grid[end])
    l = [interp(τ) + log_prior_τ(τ) for τ in τ_fine]
    m = maximum(l)
    log_evidence = m + log(sum(exp.(l .- m)) * step)
    ## weight of each grid point: the posterior mass of τ nearest to it (the extension goes to the last point)
    nearest = [argmin(abs.(τ_grid .- min(τ, τ_grid[end]))) for τ in τ_fine]
    w = [sum(exp.(l[nearest .== k] .- m)) for k in eachindex(τ_grid)]
    return (log_evidence = log_evidence, grid_weights = w ./ sum(w), tail_slope = slope)
end
