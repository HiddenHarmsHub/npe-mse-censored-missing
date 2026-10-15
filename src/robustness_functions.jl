## Robustness study. Include after mse_functions.jl and model_selection_functions.jl.
using Statistics, Random, Distributions, Combinatorics

## Pairs give the central 50, 80, 90 and 95% intervals
const robustness_probs = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
const robustness_sigmas = [1, 2, 4]
const robustness_test_size = 2000
const robustness_reference_size = 20_000
const robustness_path = joinpath("output", "robustness")
const robustness_models_path = joinpath("output", "models_robustness")
robustness_test_file(system, scenario) = joinpath(robustness_path, "test_data", "$(system)_$(scenario).bson")
robustness_reference_file(system) = joinpath(robustness_path, "test_data", "$(system)_reference.bson")

stable_seed(name) = foldl((h, c) -> (31h + Int(c)) % 1_000_003, name; init = 7)

enumerate_triples(K) = collect(combinations(1:K, 3))

three_way_design(K) = [issubset(t, parse.(Int, split(c, ","))) for c in enumerate_all_combinations(K), t in enumerate_triples(K)]

function simulate_population(θ, K; δ = nothing, r = nothing, X = one_hot_encode_parameters(K), T = nothing)
    η = X * Float64.(θ)
    isnothing(δ) || (η .+= (isnothing(T) ? three_way_design(K) : T) * δ)
    draw(e) = isnothing(r) ? rpois(e) : rpois(e + log(rand(Gamma(r, 1 / r))))
    return Int.(draw.(η)), Int(draw(Float64(θ[1]))), maximum(η)
end

function apply_censoring(counts, K, lower, upper; min_lists = 1)
    upper > 0 || return copy(counts)
    n_lists = [length(split(c, ",")) for c in enumerate_all_combinations(K)]
    return ifelse.((n_lists .>= min_lists) .& (lower .<= counts .<= upper), -1, counts)
end

## Censored cells (-1) count as non-zero, since the intervals start at 1
function has_empty_overlap(counts, K)
    cells = [parse.(Int, split(c, ",")) for c in enumerate_all_combinations(K)]
    return any(all(counts[i] == 0 for (i, c) in enumerate(cells) if k in c && l in c) for (k, l) in enumerate_two_digit_numbers(K))
end

prior_draw(K; intercept_dist = Uniform(1, 10), beta_dist = Normal(0, 4), gamma_dist = Normal(0, 4)) =
    Float64.(sample_parameters(K; intercept_dist, beta_dist, gamma_dist))

function robustness_scenarios(system; observed = nothing)
    K, J = system.K, binomial(system.K, 2)
    cl, cu = system.censoring_lower, system.censoring_upper
    n_triples = binomial(K, 3)
    base = (δ = nothing, r = nothing)
    rule = (lower = cl, upper = cu, min_lists = 1)
    s(name, label, group, sample; censoring = rule, accept = nothing) = (; name, label, group, sample, censoring, accept)
    scenarios = [
        s("S0", "In distribution", "in distribution", () -> (θ = prior_draw(K), base...)),
        s("S1", "Small population, α ~ U(1, 3)", "parameter shift", () -> (θ = prior_draw(K; intercept_dist = Uniform(1, 3)), base...)),
        s("S2", "Large population, α ~ U(8.5, 10)", "parameter shift", () -> (θ = prior_draw(K; intercept_dist = Uniform(8.5, 10)), base...)),
        s("S3", "Large lists, β ~ N(2, 1)", "parameter shift", () -> (θ = prior_draw(K; intercept_dist = Uniform(5, 10), beta_dist = Normal(2, 1)), base...)),
        s("S4a", "Moderate effects, β, γ ~ N(0, 2²)", "parameter shift", () -> (θ = prior_draw(K; beta_dist = Normal(0, 2), gamma_dist = Normal(0, 2)), base...)),
        s("S4b", "Weak effects, β, γ ~ N(0, 1)", "parameter shift", () -> (θ = prior_draw(K; beta_dist = Normal(0, 1), gamma_dist = Normal(0, 1)), base...)),
        s("S5", "Strong effects, β, γ ~ N(0, 8²)", "parameter shift", () -> (θ = prior_draw(K; beta_dist = Normal(0, 8), gamma_dist = Normal(0, 8)), base...)),
        s("S6", "Exact-zero interactions", "parameter shift", () -> (θ = effective_parameters(prior_draw(K), sample_model(primary_model_prior, K)), base...)),
        s("S7", "Empty overlap between two lists", "parameter shift", () -> (θ = prior_draw(K), base...); accept = c -> has_empty_overlap(c, K)),
        s("S8a", "Low capture, β ~ N(-3, 1)", "parameter shift", () -> (θ = prior_draw(K; intercept_dist = Uniform(6, 10), beta_dist = Normal(-3, 1)), base...)),
        s("S8b", "Imbalanced lists", "parameter shift", function ()
            θ = prior_draw(K; intercept_dist = Uniform(5, 10), beta_dist = Normal(-3, 1))
            θ[2] = rand(Normal(2, 1))
            (θ = θ, base...)
        end),
        s("M1a", "Overdispersion, negative binomial r = 20", "misspecification", () -> (θ = prior_draw(K), δ = nothing, r = 20.0)),
        s("M1b", "Overdispersion, negative binomial r = 5", "misspecification", () -> (θ = prior_draw(K), δ = nothing, r = 5.0)),
        s("M2a", "One omitted three-way interaction, |δ| ~ U(1, 2)", "misspecification", function ()
            δ = zeros(n_triples)
            δ[rand(1:n_triples)] = rand([-1, 1]) * rand(Uniform(1, 2))
            (θ = prior_draw(K), δ = δ, r = nothing)
        end),
        s("M2b", "All three-way interactions, δ ~ N(0, 0.5²)", "misspecification", () -> (θ = prior_draw(K), δ = rand(Normal(0, 0.5), n_triples), r = nothing)),
    ]
    if !isnothing(observed)
        push!(scenarios, s("S9", "Observed-like", "parameter shift", () -> (θ = Float64.(observed[:, rand(1:size(observed, 2))]), base...)))
    end
    if cu > 0
        push!(scenarios,
            s("C1", "Censoring on [1, 9] at test time", "censoring", () -> (θ = prior_draw(K), base...); censoring = (lower = cl, upper = 9, min_lists = 1)),
            s("C2", "Censoring of overlap cells only", "censoring", () -> (θ = prior_draw(K), base...); censoring = (lower = cl, upper = cu, min_lists = 2)))
    end
    return scenarios
end

function simulate_scenario(scenario, system, n; max_tries = 10_000)
    K = system.K
    X, T = one_hot_encode_parameters(K), three_way_design(K)
    pars, deltas, counts, N0, max_logλ, tries = Vector{Float64}[], Vector{Float64}[], Vector{Int}[], Int[], Float64[], Int[]
    for _ in 1:n
        for t in 1:max_tries
            d = scenario.sample()
            c, n0, m = simulate_population(d.θ, K; d.δ, d.r, X, T)
            if isnothing(scenario.accept) || scenario.accept(c) || t == max_tries
                push!(pars, d.θ); push!(deltas, isnothing(d.δ) ? zeros(size(T, 2)) : d.δ)
                push!(counts, c); push!(N0, n0); push!(max_logλ, m); push!(tries, t)
                break
            end
        end
    end
    counts = reduce(hcat, counts)
    (; lower, upper, min_lists) = scenario.censoring
    counts_obs = reduce(hcat, [apply_censoring(c, K, lower, upper; min_lists) for c in eachcol(counts)])
    return (pars = reduce(hcat, pars), deltas = reduce(hcat, deltas), counts = counts, counts_obs = counts_obs,
            N0 = N0, N = vec(sum(Float64.(counts), dims = 1)) .+ N0, capped = max_logλ .> 43.0, tries = tries,
            censoring_lower = lower, censoring_upper = upper, min_lists = min_lists)
end

function robustness_experiments(scenario_names)
    experiments = NamedTuple[]
    for s in ["A", "B"], scenario in scenario_names[s]
        push!(experiments, (experiment = "main", system = s, scenario = scenario, sigma = 4, censored = s == "B", model_average = true))
    end
    for s in ["A", "B"], (scenario, sigma) in [("S4a", 2), ("S4b", 1)]
        push!(experiments, (experiment = "prior", system = s, scenario = scenario, sigma = sigma, censored = s == "B", model_average = false))
    end
    push!(experiments, (experiment = "ablation", system = "B", scenario = "S0", sigma = 4, censored = false, model_average = false))
    return experiments
end
experiment_id(e) = "$(e.experiment)_$(e.system)_$(e.scenario)_s$(e.sigma)"

function robustness_config(system_name; censored = true, selected_path = joinpath("output", "architecture_selection", "selected.csv"))
    system = model_selection_systems[system_name]
    selected = CSV.read(selected_path, DataFrame)
    row = only(eachrow(filter(r -> r.n_lists == system.K && r.censoring_upper == system.censoring_upper, selected)))
    cl, cu = censored ? (system.censoring_lower, system.censoring_upper) : (0, 0)
    return (system = system_name, K = system.K, censoring_lower = cl, censoring_upper = cu, width = row.width, n_hidden = row.n_hidden,
            encoding_dim = row.encoding_dim, train_size = train_size_default, fixed_npe = row.model_file)
end

robustness_nbe_filename(c, sigma) = "nbe_$(c.K)_$(c.width)_$(c.n_hidden)_$(c.train_size)_$(c.censoring_lower)_$(c.censoring_upper)_$(sigma)_$(sigma)_$(length(robustness_probs)).bson"
robustness_npe_filename(c, sigma) = "npe_$(c.K)_$(c.width)_$(c.n_hidden)_$(c.encoding_dim)_$(c.train_size)_$(c.censoring_lower)_$(c.censoring_upper)_$(sigma)_$(sigma).bson"

function robustness_npe_file(c, sigma)
    sigma == 4 && c.censoring_upper == model_selection_systems[c.system].censoring_upper && return joinpath("output", "models_npe", c.fixed_npe)
    return joinpath(robustness_models_path, "npe", robustness_npe_filename(c, sigma))
end
robustness_nbe_file(c, sigma) = joinpath(robustness_models_path, "nbe", robustness_nbe_filename(c, sigma))

function robustness_training_tasks()
    tasks = NamedTuple[]
    for s in ["A", "B"], sigma in robustness_sigmas
        c = robustness_config(s)
        push!(tasks, (kind = "nbe", config = c, sigma = sigma, file = robustness_nbe_file(c, sigma)))
        sigma == 4 || push!(tasks, (kind = "npe", config = c, sigma = sigma, file = robustness_npe_file(c, sigma)))
    end
    c = robustness_config("B"; censored = false)
    push!(tasks, (kind = "nbe", config = c, sigma = 4, file = robustness_nbe_file(c, 4)),
                 (kind = "npe", config = c, sigma = 4, file = robustness_npe_file(c, 4)))
    return tasks
end

function train_robustness_network(task)
    c, σ = task.config, task.sigma
    mkpath(dirname(task.file))
    priors = (intercept_dist = Uniform(1, 10), beta_dist = Normal(0, σ), gamma_dist = Normal(0, σ))
    log_path = joinpath(dirname(task.file), "logs", replace(basename(task.file), ".bson" => ""))
    common = (censoring_lower = c.censoring_lower, censoring_upper = c.censoring_upper, savepath = dirname(task.file),
              filename = basename(task.file), training_path = log_path, priors..., training_settings...)
    if task.kind == "nbe"
        train_model_mlp(c.K, c.width, c.n_hidden, c.train_size; probs = robustness_probs, common...)
    else
        train_npe(c.K, c.width, c.n_hidden, c.encoding_dim, c.train_size; num_coupling_layers = npe_num_coupling_layers, common...)
    end
    prune_training_log(log_path)
    return nothing
end

nbe_quantiles(estimator, y) = reshape(estimator(reshape(Float32.(y), :, 1)), :, length(robustness_probs))

## Same columns as posterior_metrics; mean, sd and rank are NaN
function quantile_metrics(q, truth; probs = robustness_probs, levels = coverage_levels)
    at(p) = q[findfirst(x -> isapprox(x, p), probs)]
    med = at(0.5)
    row = (truth = Float64(truth), mean = NaN, median = med, sd = NaN, error = med - truth, abs_error = abs(med - truth),
           ape = truth == 0 ? NaN : abs(med - truth) / abs(truth), rank = NaN, n_draws = length(probs))
    ints = map(levels) do L
        lo, hi = at((1 - L) / 2), at((1 + L) / 2)
        tag = round(Int, 100L)
        (Symbol("lower_$tag") => lo, Symbol("upper_$tag") => hi, Symbol("inside_$tag") => lo <= truth <= hi,
         Symbol("width_$tag") => hi - lo, Symbol("score_$tag") => interval_score(lo, hi, truth, L))
    end
    return merge(row, NamedTuple(Iterators.flatten(ints)))
end

function npe_importance_sampling(npe, y, counts_obs, fit, priors, B; X = one_hot_encode_parameters(fit.K))
    input = reshape(Float32.(y), :, 1)
    θ = sampleposterior(npe, input, B)
    logq = Float64.(vec(NeuralEstimators.logdensity(npe, θ, repeat(input, 1, B))))
    logp = map(eachcol(Float64.(θ))) do t
        lp = log_prior(t, fit.K, priors.intercept_dist, priors.beta_dist, priors.gamma_dist)
        isfinite(lp) ? lp + likelihood_censored(counts_obs, t, X, fit.censoring_lower, fit.censoring_upper) : -Inf
    end
    logw = logp .- logq
    logw[.!isfinite.(logw)] .= -Inf
    all(isinf, logw) && return (θ = zeros(Float32, size(θ, 1), 0), ess = 0.0)
    w = exp.(logw .- maximum(logw))
    w ./= sum(w)
    return (θ = θ[:, rand(Categorical(w), B)], ess = 1 / sum(abs2, w) / B)
end

function exact_posterior(counts_obs, fit, priors, B; n_draws = 20_000, min_ess = 1000)
    J = binomial(fit.K, 2)
    kw = (censoring_lower = fit.censoring_lower, censoring_upper = fit.censoring_upper, priors = priors, n_keep = B)
    ex = log_evidence(counts_obs, trues(J); kw..., n_draws)
    ex.ess >= min_ess && return (θ = ex.θ, ess = ex.ess, retried = false)
    pilot = Float64.(permutedims(ex.θ))
    ex2 = try
        log_evidence(counts_obs, trues(J); kw..., n_draws = 5n_draws, pilot_draws = pilot, ν = 10, scale = 1.2)
    catch err
        err isa PosDefException || rethrow()
        return (θ = ex.θ, ess = ex.ess, retried = true)
    end
    return ex2.ess >= ex.ess ? (θ = ex2.θ, ess = ex2.ess, retried = true) : (θ = ex.θ, ess = ex.ess, retried = true)
end

const diagnostic_summaries = [:log_total, :n_zero, :n_censored, :singleton_share]

function summary_tail(counts_obs, K, reference_summaries)
    s = data_summaries(counts_obs, K)
    p = map(diagnostic_summaries) do k
        x = reference_summaries[k]
        v = getproperty(s, k)
        isfinite(v) ? mean(x .< v) + 0.5 * mean(x .== v) : 0.5
    end
    return minimum(min.(p, 1 .- p))
end

function embedding_reference(npe, y_reference; n_holdout = 500, n_reference = min(5000, size(y_reference, 2) - n_holdout))
    embed(y) = npe.network(Float32.(y))
    E = embed(y_reference[:, 1:n_reference])
    μ, σ = mean(E, dims = 2), std(E, dims = 2) .+ 1f-6
    Es = (E .- μ) ./ σ
    nn(e) = sqrt(minimum(vec(sum(abs2, Es .- e, dims = 1))))
    holdout = (embed(y_reference[:, n_reference .+ (1:n_holdout)]) .- μ) ./ σ
    return (embed = embed, μ = μ, σ = σ, nn = nn, holdout = sort([nn(e) for e in eachcol(holdout)]))
end
embedding_percentile(ref, y) = mean(ref.holdout .<= ref.nn((ref.embed(reshape(Float32.(y), :, 1)) .- ref.μ) ./ ref.σ))

mcse_mean(x) = std(x) / sqrt(length(x))
mcse_prop(p, n) = sqrt(p * (1 - p) / n)
mcse_rmse(e) = (mse = mean(abs2, e); mse == 0 ? 0.0 : std(abs2.(e)) / sqrt(length(e)) / (2sqrt(mse)))
function mcse_median(x; n_boot = 200, rng = Random.MersenneTwister(1))
    length(x) < 2 && return NaN
    return std([median(rand(rng, x, length(x))) for _ in 1:n_boot])
end
rank_ks(r) = (r = sort(filter(isfinite, r)); n = length(r); n == 0 ? NaN : maximum(max.((1:n) ./ n .- r, r .- (0:n-1) ./ n)))

function summarise_metrics(df; levels = coverage_levels)
    n = nrow(df)
    n == 0 && return nothing
    e, rel = df.error, df.error ./ df.truth
    rel = filter(isfinite, rel)
    ape = filter(isfinite, df.ape)
    row = (n = n, bias = mean(e), bias_se = mcse_mean(e),
           relative_bias = isempty(rel) ? NaN : mean(rel), relative_bias_se = isempty(rel) ? NaN : mcse_mean(rel),
           median_relative_error = isempty(rel) ? NaN : median(rel),
           rmse = sqrt(mean(abs2, e)), rmse_se = mcse_rmse(e), mae = mean(abs.(e)), mae_se = mcse_mean(abs.(e)),
           median_ape = isempty(ape) ? NaN : median(ape), median_ape_se = isempty(ape) ? NaN : mcse_median(ape),
           rank_ks = rank_ks(df.rank))
    cov = map(levels) do L
        tag = round(Int, 100L)
        p = mean(df[!, "inside_$tag"])
        (Symbol("coverage_$tag") => p, Symbol("coverage_$(tag)_se") => mcse_prop(p, n),
         Symbol("mean_width_$tag") => mean(df[!, "width_$tag"]), Symbol("median_width_$tag") => median(df[!, "width_$tag"]),
         Symbol("interval_score_$tag") => mean(df[!, "score_$tag"]))
    end
    return merge(row, NamedTuple(Iterators.flatten(cov)))
end
