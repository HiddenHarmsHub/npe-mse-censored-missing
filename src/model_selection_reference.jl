## Reference posterior model probabilities by enumerating every interaction structure and
## estimating each marginal likelihood p(y | m) by importance sampling around the Laplace mode.
## Include after mse_functions.jl, mcmc_functions.jl and model_selection_functions.jl.

## Log marginal likelihood of structure `mask`, integrating over the active parameters only.
## The data-only constants dropped by likelihood_censored are the same for every structure, so
## they cancel in the posterior model probabilities. Returns resampled draws on the full
## (effective) parameter vector, with zeros for inactive interactions.
function log_evidence(counts, mask; censoring_lower = 0, censoring_upper = 0, priors = coefficient_priors(), n_draws = 50_000, n_keep = 500, ν = 4, scale = 1.5, pilot_draws = nothing, rng = Random.default_rng())
    K = Int(log2(length(counts) + 1))
    active = vcat(trues(1 + K), mask)
    Xm = one_hot_encode_parameters(K)[:, active]
    is = importance_reference(counts, Xm; censoring_lower, censoring_upper, priors..., n_draws, ν, scale, pilot_draws, K, rng)
    keep = rand(rng, Categorical(is.weights), n_keep)
    θ = zeros(Float32, length(active), n_keep)
    θ[active, :] = is.draws[:, keep]
    return (log_evidence = is.log_evidence, log_evidence_se = is.log_evidence_se, ess = is.ess, θ = θ)
end

## Second, independent proposal: a heavier-tailed fit to the first run's weighted draws.
## Agreement of the two log evidences checks the Laplace-t proposal.
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
