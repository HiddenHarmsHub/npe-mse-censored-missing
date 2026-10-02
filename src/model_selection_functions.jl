## Neural inference over pairwise interaction structures: a classifier q(m | y) over all
## 2^J structures and a conditional NPE q(θ | y, m), combined into a model-averaged posterior.
## Include after mse_functions.jl.
using Flux, Distributions, Random, Statistics
import Distributions: logfactorial

## Application-matched settings. A: Silverman (K=5, uncensored), B: King (K=4, cells in [1, 4] censored)
const model_selection_systems = Dict(
    "A" => (system = "A", K = 5, censoring_lower = 0, censoring_upper = 0),
    "B" => (system = "B", K = 4, censoring_lower = 1, censoring_upper = 4),
)

## ---- Structures ----
## Class c ↔ the bits of c - 1; bit j is interaction j in the order of enumerate_two_digit_numbers(K)

model_mask(c::Integer, K::Integer) = [((c - 1) >> (j - 1)) & 1 == 1 for j in 1:binomial(K, 2)]
model_index(mask) = 1 + sum((Int(mask[j]) << (j - 1) for j in eachindex(mask)); init = 0)
n_models(K) = 2^binomial(K, 2)
enumerate_models(K) = reduce(hcat, [model_mask(c, K) for c in 1:n_models(K)])  # J × 2^J
model_size(mask) = sum(mask)
model_label(mask, K) = any(mask) ? join(["$(i)$(j)" for ((i, j), on) in zip(enumerate_two_digit_numbers(K), mask) if on], ",") : "none"

## ---- Model priors ----

## ρ ~ Beta(a, b), m_kl | ρ ~ Bernoulli(ρ): beta-binomial on model size
struct BetaBinomialModelPrior
    a::Float64
    b::Float64
end
struct BernoulliModelPrior
    p::Float64
end

model_prior_logpmf(prior::BetaBinomialModelPrior, mask::AbstractVector) =
    logpdf(BetaBinomial(length(mask), prior.a, prior.b), sum(mask)) - log(binomial(length(mask), sum(mask)))
model_prior_logpmf(prior::BernoulliModelPrior, mask::AbstractVector) =
    sum(mask) * log(prior.p) + (length(mask) - sum(mask)) * log1p(-prior.p)
model_prior_logpmf(prior, K::Integer) = [model_prior_logpmf(prior, m) for m in eachcol(enumerate_models(K))]

function sample_model(prior::BetaBinomialModelPrior, K; rng = Random.default_rng())
    ρ = rand(rng, Beta(prior.a, prior.b))
    return rand(rng, binomial(K, 2)) .< ρ
end
sample_model(prior::BernoulliModelPrior, K; rng = Random.default_rng()) = rand(rng, binomial(K, 2)) .< prior.p

## p_to(m | y) ∝ p_from(m | y) p_to(m) / p_from(m); exact because p(y | m) does not involve the model prior
function reweight_model_probs(π, from, to, K)
    logπ = log.(π) .+ model_prior_logpmf(to, K) .- model_prior_logpmf(from, K)
    w = exp.(logπ .- maximum(logπ))
    return w ./ sum(w)
end

const primary_model_prior = BetaBinomialModelPrior(1, 1)
const sensitivity_model_priors = Dict(
    "sparse" => BetaBinomialModelPrior(1, 3),
    "uniform" => BernoulliModelPrior(0.5),
)

## ---- Parameters, simulation and encoding ----

## Every interaction keeps a latent coefficient; inactive ones do not enter the cell means
function effective_parameters(θ, mask)
    J = length(mask)
    θ_eff = copy(θ)
    θ_eff[end-J+1:end] .*= mask
    return θ_eff
end

coefficient_priors(gamma_sd) = (intercept_dist = Uniform(1, 10), beta_dist = Normal(0, 4), gamma_dist = Normal(0, gamma_sd))

## Raw counts with censored cells replaced by -1, as in load_test_counts
censor_counts(Z, censoring_lower, censoring_upper) =
    censoring_upper > 0 ? ifelse.(censoring_lower .<= Z .<= censoring_upper, -1, Z) : Z

## Network input from counts with -1 marking censored cells; matches simulate_data and load_king_data
function encode_counts(counts; censored::Bool)
    censored || return Float32.(log.(counts .+ 1))
    W = counts .== -1
    U = ifelse.(W, -1.0, log.(max.(counts, 0) .+ 1))
    return Float32.(vcat(U, 1 * W))
end

## King data as integer counts in cell order, with -1 for the suppressed (censored) cells
function load_king_counts()
    king = DataFrame(CSV.File(joinpath("data", "king.csv")))
    lookup = Dict(string(g) => c for (g, c) in zip(king.group, king.count))
    return [let v = get(lookup, replace(cell, "," => ""), "0"); v == "missing" ? -1 : parse(Int, string(v)) end
            for cell in enumerate_all_combinations(4)]
end

## Prior-predictive draws of (structure, latent parameters, raw counts, realised N0)
function simulate_model_selection_data(n, system; model_prior = primary_model_prior, gamma_sd = 4)
    (; K, censoring_lower, censoring_upper) = system
    priors = coefficient_priors(gamma_sd)
    masks = reduce(hcat, [sample_model(model_prior, K) for _ in 1:n])
    pars = reduce(hcat, [sample_parameters(K; priors...) for _ in 1:n])
    counts = Int.(reduce(hcat, [simulate_data(effective_parameters(p, m), 1, log_transform = false) for (p, m) in zip(eachcol(pars), eachcol(masks))]))
    N0 = [rpois(Float64(a)) for a in pars[1, :]]
    counts_obs = censor_counts(counts, censoring_lower, censoring_upper)
    y = encode_counts(counts_obs; censored = censoring_upper > 0)
    ## Totals are Float64: cells can reach the rpois cap (~e^43), so integer sums overflow
    return (masks = masks, pars = pars, counts = counts, counts_obs = counts_obs, y = y, N0 = Int.(N0), N = vec(sum(Float64.(counts), dims = 1)) .+ N0)
end

## ---- Training ----

## For the classifier the target θ is the one-hot structure label; the other fields drive the simulator
struct ClassifierParams <: ParameterConfigurations
    θ::Matrix{Float32}
    pars::Matrix{Float32}
    masks::Matrix{Bool}
end
## For the conditional NPE the target is the latent parameter vector [α; β; γ̃]
struct CNPEParams <: ParameterConfigurations
    θ::Matrix{Float32}
    masks::Matrix{Bool}
end

function sample_structures(n, K, model_prior, priors)
    masks = reduce(hcat, [sample_model(model_prior, K) for _ in 1:n])
    pars = reduce(hcat, [sample_parameters(K; priors...) for _ in 1:n])
    return Matrix{Bool}(masks), pars
end

function simulate_structures(pars, masks, system)
    (; K, censoring_lower, censoring_upper) = system
    all_combinations = enumerate_all_combinations(K)
    two_digit_numbers = enumerate_two_digit_numbers(K)
    Z = Folds.map(1:size(pars, 2)) do i
        simulate_data(effective_parameters(pars[:, i], masks[:, i]), 1; censoring_lower, censoring_upper, all_combinations, two_digit_numbers)
    end
    return reduce(hcat, Z)
end

cnpe_input(y, masks) = vcat(y, Float32.(masks))

n_inputs(system) = (2^system.K - 1) * (system.censoring_upper > 0 ? 2 : 1)

mlp_trunk(n_in, width, n_hidden) = [Dense(n_in, width, relu), [Dense(width, width, relu) for _ in 1:n_hidden]...]

## config: a system plus (width, n_hidden, encoding_dim, train_size, a, b, gamma_sd)
classifier_filename(c, rep) = "classifier_$(c.K)_$(c.width)_$(c.n_hidden)_$(c.train_size)_$(c.censoring_lower)_$(c.censoring_upper)_$(c.a)_$(c.b)_$(c.gamma_sd)_$(rep).bson"
cnpe_filename(c, rep) = "cnpe_$(c.K)_$(c.width)_$(c.n_hidden)_$(c.encoding_dim)_$(c.train_size)_$(c.censoring_lower)_$(c.censoring_upper)_$(c.gamma_sd)_$(rep).bson"

const model_selection_models_path = joinpath("output", "models_model_selection")

## Keep the loss history and training time, drop the per-epoch network snapshots
prune_training_log(dir) = foreach(f -> occursin(r"network.*\.bson", f) && rm(joinpath(dir, f)), readdir(dir))

function train_classifier(config, rep; savepath = model_selection_models_path, epochs = 200, stopping_epochs = 10, batchsize = 32, seed = 1000 * rep + config.K)
    Random.seed!(seed)
    model_prior = BetaBinomialModelPrior(config.a, config.b)
    priors = coefficient_priors(config.gamma_sd)
    M = n_models(config.K)
    function sampler(n)
        masks, pars = sample_structures(n, config.K, model_prior, priors)
        labels = zeros(Float32, M, n)
        for (i, m) in enumerate(eachcol(masks))
            labels[model_index(m), i] = 1
        end
        return ClassifierParams(labels, pars, masks)
    end
    simulator(p, _) = simulate_structures(p.pars, p.masks, config)
    network = Chain(mlp_trunk(n_inputs(config), config.width, config.n_hidden)..., Dense(config.width, M))
    estimator = train(
        PointEstimator(network), sampler, simulator;
        K = config.train_size, m = 1, epochs, stopping_epochs, batchsize, loss = Flux.logitcrossentropy,
        savepath = joinpath(savepath, "logs", replace(classifier_filename(config, rep), ".bson" => ""))
    )
    prune_training_log(joinpath(savepath, "logs", replace(classifier_filename(config, rep), ".bson" => "")))
    a, b, gamma_sd = config.a, config.b, config.gamma_sd
    BSON.@save joinpath(savepath, classifier_filename(config, rep)) estimator a b gamma_sd
    return estimator
end

function train_cnpe(config, rep; savepath = model_selection_models_path, epochs = 200, stopping_epochs = 10, batchsize = 32, seed = 1000 * rep + config.K + 500)
    Random.seed!(seed)
    model_prior = BetaBinomialModelPrior(config.a, config.b)
    priors = coefficient_priors(config.gamma_sd)
    J = binomial(config.K, 2)
    n_pars = 1 + config.K + J
    sampler(n) = (s = sample_structures(n, config.K, model_prior, priors); CNPEParams(s[2], s[1]))
    simulator(p, _) = cnpe_input(simulate_structures(p.θ, p.masks, config), p.masks)
    network = Chain(mlp_trunk(n_inputs(config) + J, config.width, config.n_hidden)..., Dense(config.width, config.encoding_dim))
    estimator = train(
        PosteriorEstimator(NormalisingFlow(n_pars, config.encoding_dim), network), sampler, simulator;
        K = config.train_size, m = 1, epochs, stopping_epochs, batchsize,
        savepath = joinpath(savepath, "logs", replace(cnpe_filename(config, rep), ".bson" => ""))
    )
    prune_training_log(joinpath(savepath, "logs", replace(cnpe_filename(config, rep), ".bson" => "")))
    gamma_sd = config.gamma_sd
    BSON.@save joinpath(savepath, cnpe_filename(config, rep)) estimator gamma_sd
    return estimator
end

function load_model_selection_estimator(filename, path = model_selection_models_path)
    file = joinpath(path, filename)
    isfile(file) || error("Model file $filename not found in $path.")
    return BSON.load(file)[:estimator]
end
load_classifier(config, rep; path = model_selection_models_path) = load_model_selection_estimator(classifier_filename(config, rep), path)
load_cnpe(config, rep; path = model_selection_models_path) = load_model_selection_estimator(cnpe_filename(config, rep), path)

## Frozen base architecture per system from architecture_selection.jl, completed into a training config
function model_selection_config(system; selected_path = joinpath("output", "architecture_selection", "selected.csv"), train_size = 100_000, a = 1, b = 1, gamma_sd = 4, encoding_dim = 128)
    selected = CSV.read(selected_path, DataFrame)
    row = only(eachrow(filter(r -> r.n_lists == system.K && r.censoring_lower == system.censoring_lower && r.censoring_upper == system.censoring_upper, selected)))
    return (; system..., width = row.width, n_hidden = row.n_hidden, encoding_dim, train_size, a, b, gamma_sd)
end

## ---- Inference ----

function model_probabilities(classifier, y::AbstractMatrix)
    π = Float64.(softmax(classifier(Float32.(y)); dims = 1))
    return π ./ sum(π, dims = 1)
end
model_probabilities(classifier, y::AbstractVector) = vec(model_probabilities(classifier, reshape(y, :, 1)))

## Smallest set of structures whose probabilities reach `level`
function credible_model_set(π, level)
    order = sortperm(π, rev = true)
    k = min(searchsortedfirst(cumsum(π[order]), level), length(π))
    return order[1:k]
end

function model_summaries(π, K; top = 20)
    masks = enumerate_models(K)
    J = binomial(K, 2)
    sizes = vec(sum(masks, dims = 1))
    order = sortperm(π, rev = true)
    return (
        inclusion = masks * π,
        size_probs = [sum(π[sizes .== s]) for s in 0:J],
        map_index = order[1],
        top = order[1:min(top, end)],
        top_probs = π[order[1:min(top, end)]],
        top_cumulative = cumsum(π[order])[1:min(top, end)],
        entropy = -sum(p * log(p) for p in π if p > 0),
    )
end

## B draws from an NPE restricted to the intercept support, sampling until enough are accepted
function bounded_draws(estimator, input, B; lower = 1.0, upper = 10.0, max_rounds = 20)
    draws = Matrix{Float32}[]
    have, rounds = 0, 0
    while have < B && rounds < max_rounds
        θ = boundedsampleposterior(estimator, input, max(2 * (B - have), 100), lower, upper)
        push!(draws, θ)
        have += size(θ, 2)
        rounds += 1
    end
    have < B && @warn "Only $have of $B posterior draws fell inside the intercept support."
    θ = reduce(hcat, draws)
    return θ[:, 1:min(B, size(θ, 2))]
end

conditional_posterior(cnpe, y, mask, B; kwargs...) = bounded_draws(cnpe, reshape(cnpe_input(y, mask), :, 1), B; kwargs...)

## Draw from Poisson(e^η) truncated to [lower, upper]; the e^η term is shared by every k and
## dropped, so very large η cannot overflow
function truncated_poisson(η, lower, upper; rng = Random.default_rng())
    support = lower:upper
    lp = [k * η - logfactorial(k) for k in support]
    p = exp.(lp .- maximum(lp))
    return support[rand(rng, Categorical(p ./ sum(p)))]
end

## Realised unseen count N0 ~ Poisson(e^α) and total N = N_obs + N0. Censored cells are imputed from
## their truncated Poisson given θ, so N_obs carries the uncertainty about suppressed counts.
function population_draws(θ_eff, counts_obs; censoring_lower = 0, censoring_upper = 0, X = nothing, rng = Random.default_rng())
    K = Int(log2(length(counts_obs) + 1))
    X = isnothing(X) ? one_hot_encode_parameters(K) : X
    censored = findall(counts_obs .== -1)
    exact = sum(Float64, counts_obs[counts_obs .>= 0]; init = 0.0)  # Float64: integer sums can overflow
    B = size(θ_eff, 2)
    N0, N_obs = zeros(Int, B), fill(exact, B)
    for b in 1:B
        α = Float64(θ_eff[1, b])
        N0[b] = rand(rng, Poisson(exp(min(α, 43.0))))
        if !isempty(censored)
            η = X[censored, :] * Float64.(θ_eff[:, b])
            N_obs[b] += sum(truncated_poisson(e, censoring_lower, censoring_upper; rng) for e in η)
        end
    end
    return (alpha = Float64.(θ_eff[1, :]), lambda0 = exp.(Float64.(θ_eff[1, :])), N0 = N0, N_obs = N_obs, N = N_obs .+ N0)
end

## Model-averaged posterior: models drawn exactly from π, then parameters from the conditional NPE
function sample_model_averaged_posterior(cnpe, y, π, B, system; counts_obs, rng = Random.default_rng())
    K = system.K
    n_per_model = rand(rng, Multinomial(B, π ./ sum(π)))
    models, θs = Int[], Matrix{Float32}[]
    for c in findall(>(0), n_per_model)
        push!(θs, conditional_posterior(cnpe, y, model_mask(c, K), n_per_model[c]))
        append!(models, fill(c, size(θs[end], 2)))
    end
    θ = reduce(hcat, θs)
    θ_eff = reduce(hcat, [effective_parameters(t, model_mask(c, K)) for (t, c) in zip(eachcol(θ), models)])
    pop = population_draws(θ_eff, counts_obs; system.censoring_lower, system.censoring_upper, rng)
    return (; models, θ, θ_eff, pop...)
end

function sample_conditional_population(cnpe, y, mask, B, system; counts_obs, rng = Random.default_rng())
    θ = conditional_posterior(cnpe, y, mask, B)
    θ_eff = reduce(hcat, [effective_parameters(t, mask) for t in eachcol(θ)])
    pop = population_draws(θ_eff, counts_obs; system.censoring_lower, system.censoring_upper, rng)
    return (; models = fill(model_index(mask), size(θ, 2)), θ, θ_eff, pop...)
end

## Law of total variance over groups: weights default to the empirical group frequencies,
## in which case within + between equals the (uncorrected) total variance exactly
function variance_decomposition(values, groups; weights = nothing)
    ids = unique(groups)
    w = isnothing(weights) ? [mean(groups .== g) for g in ids] : [weights[g] for g in ids]
    w = w ./ sum(w)
    μ = [mean(values[groups .== g]) for g in ids]
    v = [var(values[groups .== g]; corrected = false) for g in ids]
    μ̄ = sum(w .* μ)
    within, between = sum(w .* v), sum(w .* (μ .- μ̄) .^ 2)
    return (within = within, between = between, total = within + between, between_share = between / (within + between))
end

## ---- Evaluation metrics ----

const coverage_levels = [0.5, 0.8, 0.9, 0.95]

function model_metrics(π, true_mask, K; levels = coverage_levels)
    c = model_index(true_mask)
    s = model_summaries(π, K)
    rank_true = findfirst(==(c), sortperm(π, rev = true))
    row = (
        true_model = c,
        true_size = model_size(true_mask),
        prob_true = π[c],
        log_score = log(max(π[c], 1e-300)),
        rank_true = rank_true,
        top1 = rank_true <= 1, top5 = rank_true <= 5, top10 = rank_true <= 10,
        map_model = s.map_index,
        map_size = model_size(model_mask(s.map_index, K)),
        expected_size = sum((0:binomial(K, 2)) .* s.size_probs),
        entropy = s.entropy,
        brier_inclusion = mean((s.inclusion .- true_mask) .^ 2),
    )
    sets = NamedTuple(Symbol("in_set_$(round(Int, 100L))") => (c in credible_model_set(π, L)) for L in levels)
    return merge(row, sets)
end

## Fractional rank with uniform tie-breaking, so discrete quantities (N0, N) give uniform ranks under calibration
function randomised_rank(draws, truth; rng = Random.default_rng())
    below, ties = count(<(truth), draws), count(==(truth), draws)
    return (below + rand(rng, 0:ties) + rand(rng)) / (length(draws) + 1)
end

function interval_score(lo, hi, truth, level)
    a = 1 - level
    return (hi - lo) + (2 / a) * (lo - truth) * (truth < lo) + (2 / a) * (truth - hi) * (truth > hi)
end

function posterior_metrics(draws, truth; levels = coverage_levels, rng = Random.default_rng())
    draws = Float64.(draws)
    med = median(draws)
    row = (
        truth = Float64(truth), mean = mean(draws), median = med, sd = std(draws),
        error = med - truth, abs_error = abs(med - truth), ape = truth == 0 ? NaN : abs(med - truth) / abs(truth),
        rank = randomised_rank(draws, truth; rng),
    )
    ints = map(levels) do L
        lo, hi = quantile(draws, [(1 - L) / 2, (1 + L) / 2])
        tag = round(Int, 100L)
        (Symbol("lower_$tag") => lo, Symbol("upper_$tag") => hi, Symbol("inside_$tag") => lo <= truth <= hi,
         Symbol("width_$tag") => hi - lo, Symbol("score_$tag") => interval_score(lo, hi, truth, L))
    end
    return merge(row, NamedTuple(Iterators.flatten(ints)))
end

## Proportional stratified sample of dataset indices, so the subset stays prior-predictive
function stratified_sample(strata, n; seed = 1)
    rng = Random.MersenneTwister(seed)
    groups = Dict(s => findall(==(s), strata) for s in unique(strata))
    keys_sorted = sort(collect(keys(groups)))
    exact = [n * length(groups[s]) / length(strata) for s in keys_sorted]
    alloc = floor.(Int, exact)
    for i in sortperm(exact .- alloc, rev = true)[1:(n - sum(alloc))]
        alloc[i] += 1
    end
    return sort(vcat([shuffle(rng, groups[s])[1:a] for (s, a) in zip(keys_sorted, alloc)]...))
end

## ---- Simulation study ----

## Evaluate one replicate's networks on test datasets `idx`: structure metrics (one row per dataset)
## and posterior metrics for α, λ0, N0 and N under each method (one row per dataset × method × quantity)
function evaluate_simulation_chunk(config, rep, idx, test, classifier, cnpe, fixed_npe; B = 4000)
    K, J = config.K, binomial(config.K, 2)
    primary = BetaBinomialModelPrior(config.a, config.b)
    pair_names = ["$(a)$(b)" for (a, b) in enumerate_two_digit_numbers(K)]
    π_all = model_probabilities(classifier, test.y[:, idx])
    model_rows, posterior_rows = NamedTuple[], NamedTuple[]
    for (j, i) in enumerate(idx)
        y, counts_obs, mask = test.y[:, i], test.counts_obs[:, i], Bool.(test.masks[:, i])
        truth_eff = effective_parameters(Float64.(test.pars[:, i]), mask)
        π = π_all[:, j]
        s = model_summaries(π, K)
        draws = Dict(
            "oracle" => sample_conditional_population(cnpe, y, mask, B, config; counts_obs),
            "all" => sample_conditional_population(cnpe, y, trues(J), B, config; counts_obs),
            "map" => sample_conditional_population(cnpe, y, model_mask(s.map_index, K), B, config; counts_obs),
            "bma" => sample_model_averaged_posterior(cnpe, y, π, B, config; counts_obs),
            [("bma_" * name) => sample_model_averaged_posterior(cnpe, y, reweight_model_probs(π, primary, prior, K), B, config; counts_obs)
             for (name, prior) in sensitivity_model_priors]...,
        )
        if !isnothing(fixed_npe)
            θ = bounded_draws(fixed_npe, reshape(y, :, 1), B)
            draws["fixed"] = (models = fill(n_models(K), size(θ, 2)), θ = θ, θ_eff = θ, population_draws(θ, counts_obs; config.censoring_lower, config.censoring_upper)...)
        end
        decomposition = variance_decomposition(Float64.(draws["bma"].N0), draws["bma"].models)
        push!(model_rows, merge(
            (dataset = i, system = config.system, rep = rep, capped = is_capped(truth_eff), intercept = truth_eff[1],
             max_abs_gamma = maximum(abs.(truth_eff[2+K:end]); init = 0.0), n_zero = count(==(0), counts_obs),
             n_censored = count(==(-1), counts_obs), N0_true = test.N0[i], N_true = test.N[i],
             bma_between_share = decomposition.between_share),
            NamedTuple(Symbol("logw_$name") => model_prior_logpmf(prior, mask) - model_prior_logpmf(primary, mask) for (name, prior) in sensitivity_model_priors),
            model_metrics(π, mask, K),
            NamedTuple(Symbol("incl_$p") => s.inclusion[k] for (k, p) in enumerate(pair_names)),
            NamedTuple(Symbol("true_$p") => mask[k] for (k, p) in enumerate(pair_names)),
            NamedTuple(Symbol("size_$k") => s.size_probs[k+1] for k in 0:J),
        ))
        truths = (alpha = truth_eff[1], lambda0 = exp(truth_eff[1]), N0 = test.N0[i], N = test.N[i])
        for (method, d) in sort(collect(draws), by = first), quantity in keys(truths)
            push!(posterior_rows, merge((dataset = i, system = config.system, rep = rep, method = method, quantity = String(quantity)),
                posterior_metrics(getproperty(d, quantity), truths[quantity])))
        end
    end
    return DataFrame(model_rows), DataFrame(posterior_rows)
end
