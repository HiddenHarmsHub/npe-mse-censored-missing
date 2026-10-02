## Independent reference samplers for the log-linear Poisson model:
## IRLS Metropolis-Hastings (Gamerman, 1997) and importance sampling around the MAP
using LinearAlgebra, Random, ForwardDiff

## Draw each censored cell from Poisson(λ) truncated to [censoring_lower, censoring_upper]
function impute_censored(counts, η, censoring_lower, censoring_upper; rng = Random.default_rng())
    y = copy(counts)
    support = censoring_lower:censoring_upper
    for i in findall(counts .== -1)
        lp = [k * η[i] - exp(η[i]) - logfactorial(k) for k in support]
        p = exp.(lp .- maximum(lp))
        y[i] = support[rand(rng, Categorical(p ./ sum(p)))]
    end
    return y
end

## Gaussian proposal from one IRLS step at θ for complete counts y.
## The Uniform intercept prior is replaced by a moment-matched Normal in the proposal only.
function irls_proposal(θ, y, X, prior_mean, prior_prec)
    η = X * θ
    μ = exp.(η)
    ỹ = η .+ (y .- μ) ./ μ
    sw = sqrt.(μ)
    A = vcat(sw .* X, Diagonal(sqrt.(prior_prec)))
    b = vcat(sw .* ỹ, sqrt.(prior_prec) .* prior_mean)
    F = qr(A)
    R = UpperTriangular(F.R)
    m = R \ (F.Q' * b)[1:size(X, 2)]
    return m, R
end

proposal_logpdf(θ, m, R) = sum(log.(abs.(diag(R)))) - 0.5 * sum(abs2, R * (θ .- m))

function complete_log_posterior(θ, y, X, K, intercept_dist, beta_dist, gamma_dist)
    lp = log_prior(θ, K, intercept_dist, beta_dist, gamma_dist)
    isfinite(lp) || return lp
    η = X * θ
    return lp + sum(poisson_kernel.(y, η))
end

function irls_chain(counts, X, θ0; censoring_lower, censoring_upper, intercept_dist, beta_dist, gamma_dist, n_warmup, n_samples, rng)
    n_pars = size(X, 2)
    K = Int(-0.5 + (sqrt(8 * (n_pars - 1) + 1) / 2))
    prior_mean = vcat(mean(intercept_dist), fill(mean(beta_dist), K), fill(mean(gamma_dist), n_pars - K - 1))
    prior_prec = 1 ./ vcat(var(intercept_dist), fill(var(beta_dist), K), fill(var(gamma_dist), n_pars - K - 1))

    θ = copy(θ0)
    draws = zeros(n_samples, n_pars)
    accepted = zeros(n_samples)
    for t in 1:(n_warmup + n_samples)
        y = impute_censored(counts, X * θ, censoring_lower, censoring_upper; rng = rng)
        m, R = irls_proposal(θ, y, X, prior_mean, prior_prec)
        θ_prop = m .+ R \ randn(rng, n_pars)
        lp_prop = complete_log_posterior(θ_prop, y, X, K, intercept_dist, beta_dist, gamma_dist)
        accept = false
        if isfinite(lp_prop)
            m_rev, R_rev = irls_proposal(θ_prop, y, X, prior_mean, prior_prec)
            log_α = lp_prop - complete_log_posterior(θ, y, X, K, intercept_dist, beta_dist, gamma_dist) +
                proposal_logpdf(θ, m_rev, R_rev) - proposal_logpdf(θ_prop, m, R)
            accept = log(rand(rng)) < log_α
        end
        θ = accept ? θ_prop : θ
        if t > n_warmup
            draws[t - n_warmup, :] = θ
            accepted[t - n_warmup] = accept
        end
    end
    return draws, accepted
end

function fit_irls(
    counts,
    X;
    censoring_lower = 0,
    censoring_upper = 0,
    intercept_dist = Uniform(1, 10),
    beta_dist = Normal(0, 4),
    gamma_dist = Normal(0, 4),
    n_warmup = 5000,
    n_samples = 20000,
    n_chains = 4,
    rng = Random.default_rng()
)
    ## Each chain starts from a draw of the Laplace approximation so it begins on the posterior scale
    mode, L = laplace_approximation(counts, X; censoring_lower, censoring_upper, intercept_dist, beta_dist, gamma_dist)
    n_pars = size(X, 2)
    K = Int(-0.5 + (sqrt(8 * (n_pars - 1) + 1) / 2))
    vals = zeros(n_samples, n_pars + 1, n_chains)
    for c in 1:n_chains
        θ0 = mode .+ L * randn(rng, n_pars)
        while !isfinite(logpdf(intercept_dist, θ0[1]))
            θ0 = mode .+ L * randn(rng, n_pars)
        end
        draws, accepted = irls_chain(
            counts, X, θ0;
            censoring_lower, censoring_upper, intercept_dist, beta_dist, gamma_dist, n_warmup, n_samples, rng
        )
        vals[:, 1:n_pars, c] = draws
        vals[:, end, c] = accepted
    end
    par_names = vcat("intercept", ["betas[$i]" for i in 1:K], ["gammas[$i]" for i in 1:(n_pars - K - 1)], "acceptance_rate")
    return Chains(vals, par_names, Dict(:internals => ["acceptance_rate"]))
end

function weighted_quantile(x, w, p)
    order = sortperm(x)
    cw = cumsum(w[order]) ./ sum(w)
    return x[order][min(searchsortedfirst(cw, p), length(x))]
end

## Importance sampling from a multivariate t. By default it is the Laplace approximation; when pilot draws (e.g. from another sampler) are given their mean and covariance are used.
## The weights use the exact posterior, so the proposal only affects efficiency.
function importance_reference(
    counts,
    X;
    censoring_lower = 0,
    censoring_upper = 0,
    intercept_dist = Uniform(1, 10),
    beta_dist = Normal(0, 4),
    gamma_dist = Normal(0, 4),
    n_draws = 10^6,
    ν = 4,
    scale = 1.5,
    probs = [0.025, 0.5, 0.975],
    pilot_draws = nothing
)
    priors = (intercept_dist = intercept_dist, beta_dist = beta_dist, gamma_dist = gamma_dist)
    logpost(θ) = log_posterior(θ, counts, X, censoring_lower, censoring_upper; priors...)
    if isnothing(pilot_draws)
        μ, L = laplace_approximation(counts, X; censoring_lower, censoring_upper, priors...)
        Σ = Symmetric(L * L')
    else
        μ = vec(mean(pilot_draws, dims = 1))
        Σ = Symmetric(cov(pilot_draws))
    end
    if !isposdef(Σ)
        Σ = Symmetric(Σ + (abs(eigmin(Σ)) + 1e-6) * I)
    end
    q = MvTDist(ν, μ, Matrix(scale^2 * Σ))
    θs = rand(q, n_draws)
    logw = [logpost(θ) for θ in eachcol(θs)] .- logpdf(q, θs)
    w = exp.(logw .- maximum(logw))
    w ./= sum(w)
    ess = 1 / sum(abs2, w)

    means = θs * w
    sds = sqrt.(max.((θs .^ 2) * w .- means .^ 2, 0))
    quantiles = [weighted_quantile(collect(row), w, p) for row in eachrow(θs), p in probs]
    return (means = means, sds = sds, quantiles = quantiles, probs = probs, ess = ess)
end
