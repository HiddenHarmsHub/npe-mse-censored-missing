## Local accuracy checks of the conditional NPE near the real datasets
include("mse_functions.jl")
include("mcmc_functions.jl")
include("model_selection_functions.jl")
include("model_selection_reference.jl")

run = model_selection_run()
n_draws = 20_000          # neural and NIS draws on the real data
n_exact_draws = 100_000   # importance-sampling draws for the exact posterior on the real data
n_sim = 100               # observed-like datasets per structure
n_sim_draws = 4000        # neural and NIS draws per observed-like dataset
n_sim_exact_draws = 20_000
output_path = joinpath(model_selection_output_path(run), "local_checks")
mkpath(output_path)

real_data = Dict("A" => load_silverman_data(), "B" => load_king_counts())
literature_file(s) = joinpath(model_selection_output_path(run), "real_data", "$(s)_conditional_structures.csv")

## Neural importance sampling for structure `mask`
function neural_importance_sampling(cnpe, y, counts, mask, system, priors, B; rng = Random.default_rng())
    K = system.K
    X = one_hot_encode_parameters(K)
    input = reshape(cnpe_input(y, mask), :, 1)
    θ = raw_draws(cnpe, input, B)
    logq = Float64.(vec(NeuralEstimators.logdensity(cnpe, θ, repeat(input, 1, B))))
    θ64 = Float64.(θ)
    logp = map(eachcol(θ64)) do t
        lp = log_prior(t, K, priors.intercept_dist, priors.beta_dist, priors.gamma_dist)
        isfinite(lp) ? lp + likelihood_censored(counts, effective_parameters(t, mask), X, system.censoring_lower, system.censoring_upper) : -Inf
    end
    logw = logp .- logq
    m = maximum(logw)
    w = exp.(logw .- m)
    log_evidence = m + log(mean(w))
    w ./= sum(w)
    keep = rand(rng, Categorical(w), B)
    return (θ = θ[:, keep], ess = 1 / sum(abs2, w) / B, log_evidence = log_evidence)
end

## Exact posterior given the structure by importance sampling around the Laplace mode
function exact_conditional(counts, mask, system, priors, B; n_draws, rng = Random.default_rng())
    fit = log_evidence(counts, mask; system.censoring_lower, system.censoring_upper, priors, n_draws, n_keep = B, rng)
    θ = Float64.(fit.θ)
    J = length(mask)
    θ[end-J+1:end, :] .+= .!mask .* rand(rng, priors.gamma_dist, J, B)
    return (θ = θ, ess = fit.ess)
end

population_N(θ, mask, counts, system; rng = Random.default_rng()) =
    population_draws(effective_draws(Float32.(θ), fill(model_index(mask), size(θ, 2)), system.K), counts; system.censoring_lower, system.censoring_upper, rng).N

real_rows, sim_rows = NamedTuple[], NamedTuple[]
for s in ["A", "B"]
    system = model_selection_systems[s]
    K = system.K
    config = model_selection_config(system, run)
    priors = coefficient_priors(run)
    cnpe = load_cnpe(config, 0)
    if cnpe isa AbstractVector
        println("System $s: skipped, the local checks need a single conditional NPE (cnpe_combination = :best)")
        continue
    end
    counts = real_data[s]
    censored = system.censoring_upper > 0
    y = encode_counts(counts; censored)
    names_K = param_names(K)
    structures = CSV.read(literature_file(s), DataFrame; types = Dict(:structure => String))
    Random.seed!(s == "A" ? 3030 : 3031)
    for row in eachrow(structures)
        mask = [("$(i)$(j)" in split(row.structure, ",")) for (i, j) in enumerate_two_digit_numbers(K)]
        exact = exact_conditional(counts, mask, system, priors, n_draws; n_draws = n_exact_draws)
        neural = Float64.(conditional_posterior(cnpe, y, mask, n_draws))
        nis = neural_importance_sampling(cnpe, y, counts, mask, system, priors, n_draws)
        N = Dict("exact" => population_N(exact.θ, mask, counts, system), "neural" => population_N(neural, mask, counts, system),
                 "nis" => population_N(nis.θ, mask, counts, system))
        active = vcat(trues(1 + K), mask)
        for (p, name) in enumerate(names_K)
            push!(real_rows, (system = s, structure = row.name, interactions = row.structure, parameter = name, active = active[p],
                exact_mean = mean(exact.θ[p, :]), exact_sd = std(exact.θ[p, :]), neural_mean = mean(neural[p, :]), neural_sd = std(neural[p, :]),
                nis_mean = mean(Float64.(nis.θ[p, :])), nis_sd = std(Float64.(nis.θ[p, :])),
                standardised_shift = (mean(neural[p, :]) - mean(exact.θ[p, :])) / std(exact.θ[p, :]),
                nis_ess = nis.ess, exact_ess = exact.ess,
                N_exact = median(N["exact"]), N_neural = median(N["neural"]), N_nis = median(N["nis"]),
                N_exact_q025 = quantile(N["exact"], 0.025), N_exact_q975 = quantile(N["exact"], 0.975),
                N_neural_q025 = quantile(N["neural"], 0.025), N_neural_q975 = quantile(N["neural"], 0.975),
                N_nis_q025 = quantile(N["nis"], 0.025), N_nis_q975 = quantile(N["nis"], 0.975)))
        end
        println("System $s, $(row.name): N exact $(round(Int, median(N["exact"]))), neural $(round(Int, median(N["neural"]))), NIS $(round(Int, median(N["nis"]))) (ESS $(round(nis.ess, digits = 3)))")

        ## Observed-like datasets from the exact posterior given the structure
        for i in 1:n_sim
            θ_true = exact.θ[:, rand(1:size(exact.θ, 2))]
            sim_counts = Int.(vec(simulate_data(effective_parameters(θ_true, mask), 1, log_transform = false)))
            sim_obs = censor_counts(sim_counts, system.censoring_lower, system.censoring_upper)
            sim_y = encode_counts(sim_obs; censored)
            ex = exact_conditional(sim_obs, mask, system, priors, n_sim_draws; n_draws = n_sim_exact_draws)
            ne = Float64.(conditional_posterior(cnpe, sim_y, mask, n_sim_draws))
            size(ne, 2) == 0 && continue
            ni = neural_importance_sampling(cnpe, sim_y, sim_obs, mask, system, priors, n_sim_draws)
            inside(d, L) = ((lo, hi) = quantile(d, [(1 - L) / 2, (1 + L) / 2]); lo <= θ_true[1] <= hi)
            α = (exact = ex.θ[1, :], neural = ne[1, :], nis = Float64.(ni.θ[1, :]))
            push!(sim_rows, (system = s, structure = row.name, interactions = row.structure, dataset = i, alpha_true = θ_true[1],
                exact_median = median(α.exact), neural_median = median(α.neural), nis_median = median(α.nis),
                exact_sd = std(α.exact), neural_sd = std(α.neural), nis_sd = std(α.nis),
                exact_inside_50 = inside(α.exact, 0.5), neural_inside_50 = inside(α.neural, 0.5), nis_inside_50 = inside(α.nis, 0.5),
                exact_inside_95 = inside(α.exact, 0.95), neural_inside_95 = inside(α.neural, 0.95), nis_inside_95 = inside(α.nis, 0.95),
                nis_ess = ni.ess, exact_ess = ex.ess))
        end
        println("  observed-like simulation done")
    end
end

## Runs that mix several conditional NPEs are skipped for both systems
if isempty(real_rows)
    println("No local checks for this run (it does not use a single conditional NPE); nothing written.")
    exit(0)
end
real = DataFrame(real_rows)
CSV.write(joinpath(output_path, "real_data_neural_vs_exact.csv"), real)
sim = DataFrame(sim_rows)
CSV.write(joinpath(output_path, "observed_like.csv"), sim)

## Neural and NIS errors in α relative to the exact posterior median, in exact posterior sds
summary = combine(groupby(sim, [:system, :structure, :interactions]),
    nrow => :n,
    [:neural_median, :exact_median, :exact_sd] => ((n, e, sd) -> mean((n .- e) ./ sd)) => :neural_shift_mean,
    [:neural_median, :exact_median, :exact_sd] => ((n, e, sd) -> mean(abs.(n .- e) ./ sd)) => :neural_shift_abs,
    [:nis_median, :exact_median, :exact_sd] => ((n, e, sd) -> mean(abs.(n .- e) ./ sd)) => :nis_shift_abs,
    [:neural_sd, :exact_sd] => ((n, e) -> mean(n ./ e)) => :neural_sd_ratio,
    [c => mean => c for c in [:exact_inside_50, :neural_inside_50, :nis_inside_50, :exact_inside_95, :neural_inside_95, :nis_inside_95]]...,
    :nis_ess => median => :nis_ess_median, :nis_ess => (e -> mean(e .< 0.01)) => :nis_ess_below_1pct)
CSV.write(joinpath(output_path, "observed_like_summary.csv"), summary)
show(summary, allrows = true, allcols = true)
println("\nLocal checks written to $output_path")
