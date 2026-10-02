## Model-averaged reanalysis of the Silverman (system A: K=5, uncensored) and King (system B: K=4,
## cells in [1, 4] suppressed) data with the structure classifier and conditional NPE, compared with
## conditional fits, the fixed-model NPE, reference model averaging by enumeration, and the MLE.
## Run locally: julia --project src/model_selection_real_data.jl
include("mse_functions.jl")
include("mcmc_functions.jl")
include("model_selection_functions.jl")
include("model_selection_reference.jl")
using Optim

gamma_sd = 4
train_size = 100_000
replicates = 1:3
n_draws = 20_000
n_decomposition_draws = 1000
decomposition_mass = 0.999
max_decomposition_models = 200
n_reference_draws = 100_000
n_ppc_draws = 4000

## Interaction structures from the literature to fit conditionally, as lists of pairs such as "12".
## These are left for the authors to fill in (e.g. the structure selected by Silverman, 2020).
literature_models = Dict(
    "A" => Dict{String, Vector{String}}(),
    "B" => Dict{String, Vector{String}}(),
)

output_path = joinpath("output", "model_selection", "real_data")
mkpath(output_path)
selected = CSV.read(joinpath("output", "architecture_selection", "selected.csv"), DataFrame)

real_data = Dict("A" => load_silverman_data(), "B" => load_king_counts())

mask_from_pairs(pairs, K) = [("$(i)$(j)" in pairs) for (i, j) in enumerate_two_digit_numbers(K)]

summarise(x) = (median = median(x), mean = mean(x), sd = std(x),
                q025 = quantile(x, 0.025), q10 = quantile(x, 0.10), q90 = quantile(x, 0.90), q975 = quantile(x, 0.975))
population_rows(method, d) = [merge((method = method, quantity = String(q)), summarise(Float64.(getproperty(d, q)))) for q in (:alpha, :lambda0, :N0, :N_obs, :N)]

for s in ["A", "B"]
    system = model_selection_systems[s]
    K, J = system.K, binomial(system.K, 2)
    config = model_selection_config(system; train_size, gamma_sd)
    primary = BetaBinomialModelPrior(config.a, config.b)
    all_priors = merge(Dict("primary" => primary), sensitivity_model_priors)
    counts_obs = real_data[s]
    y = encode_counts(counts_obs; censored = system.censoring_upper > 0)
    pair_names = ["$(a)$(b)" for (a, b) in enumerate_two_digit_numbers(K)]
    labels = [model_label(model_mask(c, K), K) for c in 1:n_models(K)]
    Random.seed!(s == "A" ? 2020 : 2021)

    ## ---- Structure posteriors: neural (every replicate and prior) and reference ----
    π_neural = Dict(rep => model_probabilities(load_classifier(config, rep), y) for rep in replicates)
    π_prior = Dict(name => reweight_model_probs(π_neural[1], primary, prior, K) for (name, prior) in all_priors)
    println("System $s: enumerating reference evidences for $(n_models(K)) structures")
    fits = reference_evidences(counts_obs, system; gamma_sd, n_draws = n_reference_draws)
    reference = Dict(name => reference_mixture(fits, counts_obs, system; model_prior = prior, B = n_draws) for (name, prior) in all_priors)

    probs = DataFrame(model = 1:n_models(K), label = labels, size = [model_size(model_mask(c, K)) for c in 1:n_models(K)],
                      log_evidence = reference["primary"].log_evidence, log_evidence_se = reference["primary"].log_evidence_se,
                      ess = reference["primary"].ess)
    for (name, π) in π_prior
        probs[!, "neural_$name"] = π
        probs[!, "reference_$name"] = reference[name].π
    end
    for rep in replicates
        probs[!, "neural_primary_rep$rep"] = π_neural[rep]
    end
    CSV.write(joinpath(output_path, "$(s)_model_probabilities.csv"), sort(probs, :neural_primary, rev = true))

    masks = enumerate_models(K)
    sizes = vec(sum(masks, dims = 1))
    inclusion_rows, size_rows = NamedTuple[], NamedTuple[]
    for (label, π) in vcat([("neural_$n", p) for (n, p) in π_prior], [("reference_$n", r.π) for (n, r) in reference],
                           [("neural_primary_rep$rep", π_neural[rep]) for rep in replicates])
        append!(inclusion_rows, [(method = label, pair = p, prob = v) for (p, v) in zip(pair_names, masks * π)])
        append!(size_rows, [(method = label, size = k, prob = sum(π[sizes .== k])) for k in 0:J])
    end
    CSV.write(joinpath(output_path, "$(s)_inclusion.csv"), DataFrame(inclusion_rows))
    CSV.write(joinpath(output_path, "$(s)_model_size.csv"), DataFrame(size_rows))

    ## ---- Population size: model averaged and conditional ----
    cnpe = Dict(rep => load_cnpe(config, rep) for rep in replicates)
    rows = NamedTuple[]
    bma = sample_model_averaged_posterior(cnpe[1], y, π_neural[1], n_draws, system; counts_obs)
    append!(rows, population_rows("neural_bma", bma))
    for rep in replicates[2:end]
        append!(rows, population_rows("neural_bma_rep$rep", sample_model_averaged_posterior(cnpe[rep], y, π_neural[rep], n_draws, system; counts_obs)))
    end
    for (name, _) in sensitivity_model_priors
        append!(rows, population_rows("neural_bma_$name", sample_model_averaged_posterior(cnpe[1], y, π_prior[name], n_draws, system; counts_obs)))
    end
    map_index = argmax(π_neural[1])
    conditional = Dict("all interactions" => trues(J), "MAP ($(labels[map_index]))" => model_mask(map_index, K),
                       ["literature: $n" => mask_from_pairs(p, K) for (n, p) in literature_models[s]]...)
    for (name, mask) in conditional
        append!(rows, population_rows("neural_conditional: $name", sample_conditional_population(cnpe[1], y, mask, n_draws, system; counts_obs)))
    end
    row = only(eachrow(filter(r -> r.n_lists == K && r.censoring_upper == system.censoring_upper, selected)))
    fixed_npe = BSON.load(joinpath("output", "models_npe", row.model_file))[:estimator]
    θ_fixed = bounded_draws(fixed_npe, reshape(y, :, 1), n_draws)
    append!(rows, population_rows("fixed_model_npe", population_draws(θ_fixed, counts_obs; system.censoring_lower, system.censoring_upper)))
    for (name, r) in reference
        append!(rows, population_rows(name == "primary" ? "reference_bma" : "reference_bma_$name", r))
    end
    population = DataFrame(rows)
    population.n_observed_exact .= sum(counts_obs[counts_obs .>= 0])
    population.n_censored_cells .= count(==(-1), counts_obs)
    CSV.write(joinpath(output_path, "$(s)_population.csv"), population)
    CSV.write(joinpath(output_path, "$(s)_bma_draws.csv"), DataFrame(model = bma.models, label = labels[bma.models], alpha = bma.alpha, N0 = bma.N0, N = bma.N))

    ## ---- Within- and between-structure variance, from dedicated draws of the leading structures ----
    order = sortperm(π_neural[1], rev = true)
    top = order[1:min(searchsortedfirst(cumsum(π_neural[1][order]), decomposition_mass), max_decomposition_models)]
    per_model = [sample_conditional_population(cnpe[1], y, model_mask(c, K), n_decomposition_draws, system; counts_obs) for c in top]
    groups = reduce(vcat, [fill(c, n_decomposition_draws) for c in top])
    weights = Dict(c => π_neural[1][c] for c in top)
    CSV.write(joinpath(output_path, "$(s)_variance_decomposition.csv"), DataFrame([
        merge((quantity = q, n_models = length(top), mass = sum(π_neural[1][top])),
              variance_decomposition(reduce(vcat, [Float64.(getproperty(d, Symbol(q))) for d in per_model]), groups; weights))
        for q in ["N0", "N", "alpha"]]))

    ## ---- Model-averaged posterior predictive check; censored cells are compared as P(a ≤ ñ ≤ b) ----
    X = one_hot_encode_parameters(K)
    sub = rand(1:size(bma.θ_eff, 2), n_ppc_draws)
    replicated = reduce(hcat, [[rand(Poisson(exp(min(η, 43.0)))) for η in X * Float64.(bma.θ_eff[:, b])] for b in sub])
    CSV.write(joinpath(output_path, "$(s)_ppc.csv"), DataFrame([
        (cell = replace(cell, "," => ""), observed = counts_obs[i] == -1 ? missing : counts_obs[i], censored = counts_obs[i] == -1,
         ppd_q025 = quantile(replicated[i, :], 0.025), ppd_median = median(replicated[i, :]), ppd_q975 = quantile(replicated[i, :], 0.975),
         ppp_upper = counts_obs[i] == -1 ? missing : mean(replicated[i, :] .>= counts_obs[i]),
         prob_in_censoring_interval = system.censoring_upper > 0 ? mean(system.censoring_lower .<= replicated[i, :] .<= system.censoring_upper) : missing)
        for (i, cell) in enumerate(enumerate_all_combinations(K))]))

    ## ---- MLE under the MAP and all-interactions structures (Wald intervals on α) ----
    mle_rows = NamedTuple[]
    for (name, mask) in [("MAP ($(labels[map_index]))", model_mask(map_index, K)), ("all interactions", trues(J))]
        Xm = X[:, vcat(trues(1 + K), mask)]
        negloglik(θ) = -likelihood_censored(counts_obs, θ, Xm, system.censoring_lower, system.censoring_upper)
        start = log_count_start(counts_obs, Xm, system.censoring_lower, system.censoring_upper, coefficient_priors(gamma_sd)...; K)
        result = optimize(negloglik, start, BFGS(), autodiff = :forward)
        θ̂ = Optim.minimizer(result)
        H = ForwardDiff.hessian(negloglik, θ̂)
        se = isposdef(Symmetric(H)) ? sqrt(inv(Symmetric(H))[1, 1]) : NaN
        push!(mle_rows, (structure = name, converged = Optim.converged(result), alpha = θ̂[1], alpha_se = se,
                         lambda0 = exp(θ̂[1]), lambda0_lower = exp(θ̂[1] - 1.96se), lambda0_upper = exp(θ̂[1] + 1.96se),
                         min_coefficient = minimum(θ̂[2:end])))
    end
    CSV.write(joinpath(output_path, "$(s)_mle.csv"), DataFrame(mle_rows))
    println(population)
end
