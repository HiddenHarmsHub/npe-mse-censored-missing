## Prior specification, induced model-size priors, prior-predictive summaries, and domain checks
## locating the Silverman (system A) and King (system B) data within the training distribution.
## Run locally: MS_RUN=<run> julia --project src/model_selection_prior_predictive.jl
## The embedding check uses classifier replicate 1 when it has been trained, and is skipped otherwise.
include("mse_functions.jl")
include("model_selection_functions.jl")

run = model_selection_run()
n_prior_predictive = 20_000
n_embedding_reference = 5000
n_embedding_holdout = 500

output_path = joinpath(model_selection_output_path(run), "prior_predictive")
mkpath(output_path)

## ---- Prior specification ----
priors = coefficient_priors(run)
CSV.write(joinpath(output_path, "prior_specification.csv"), DataFrame(
    component = ["intercept α", "main effects β_k", "interactions γ_kl (latent)", "structure m (primary)", "structure m (sparse)", "structure m (uniform)"],
    prior = [string(priors.intercept_dist), string(priors.beta_dist), string(priors.gamma_dist),
             "ρ ~ Beta(1, 1), m_kl | ρ ~ Bernoulli(ρ)", "ρ ~ Beta(1, 3), m_kl | ρ ~ Bernoulli(ρ)", "m_kl ~ Bernoulli(0.5)"],
))

all_priors = merge(Dict("primary" => primary_model_prior), sensitivity_model_priors)
size_rows = NamedTuple[]
for system in values(model_selection_systems), (name, prior) in all_priors
    K, J = system.K, binomial(system.K, 2)
    p = exp.(model_prior_logpmf(prior, K))
    sizes = vec(sum(enumerate_models(K), dims = 1))
    for s in 0:J
        push!(size_rows, (system = system.system, prior = name, size = s, prob = sum(p[sizes .== s]),
                          prob_single_model = p[findfirst(==(s), sizes)], inclusion = sum(p .* (sizes ./ J))))
    end
end
CSV.write(joinpath(output_path, "model_size_prior.csv"), DataFrame(size_rows))

## ---- Data summaries computable from censored data, for simulations and real data alike ----
function data_summaries(counts_obs, K)
    cells = enumerate_all_combinations(K)
    exact = counts_obs .>= 0
    in_list(k) = [string(k) in split(c, ",") for c in cells]
    singletons = [length(split(c, ",")) == 1 for c in cells]
    total = sum(Float64, counts_obs[exact]; init = 0.0)  # cells can reach ~e^43, so integer sums overflow
    return merge(
        (total_exact = total, n_censored = count(.!exact), n_zero = count(==(0), counts_obs),
         singleton_share = total == 0 ? NaN : sum(Float64, counts_obs[exact .& singletons]; init = 0.0) / total,
         max_cell = maximum(counts_obs), log_total = log(total + 1)),
        NamedTuple(Symbol("list_$k") => sum(Float64, counts_obs[exact .& in_list(k)]; init = 0.0) for k in 1:K),
    )
end

real_data = Dict(
    "A" => (counts = load_silverman_data(), name = "Silverman"),
    "B" => (counts = load_king_counts(), name = "King"),
)

domain_flag(p) = 0.025 <= p <= 0.975 ? "within" : (0.005 <= p <= 0.995 ? "near boundary" : "outside")

quantile_table, domain_rows = NamedTuple[], NamedTuple[]
for s in ["A", "B"]
    system = model_selection_systems[s]
    Random.seed!(s == "A" ? 101 : 102)
    sim = simulate_model_selection_data(n_prior_predictive, system; priors)
    summaries = DataFrame([data_summaries(c, system.K) for c in eachcol(sim.counts_obs)])
    summaries.N0 = sim.N0
    summaries.N = sim.N
    summaries.model_size = vec(sum(sim.masks, dims = 1))
    CSV.write(joinpath(output_path, "prior_predictive_summaries_$(s).csv"), summaries)

    for col in names(summaries)
        x = filter(isfinite, Float64.(summaries[!, col]))
        q = quantile(x, [0.025, 0.25, 0.5, 0.75, 0.975])
        push!(quantile_table, (system = s, summary = col, q025 = q[1], q25 = q[2], median = q[3], q75 = q[4], q975 = q[5]))
    end

    observed = data_summaries(real_data[s].counts, system.K)
    for (k, v) in pairs(observed)
        x = filter(isfinite, Float64.(summaries[!, k]))
        p = mean(x .< v) + 0.5 * mean(x .== v)  # mid-rank, so discrete summaries equal to every draw sit at 0.5
        push!(domain_rows, (system = s, data = real_data[s].name, check = String(k), observed = Float64(v), percentile = p, flag = domain_flag(p)))
    end

    ## Nearest-neighbour distance in the classifier's learned summary space, relative to the
    ## same distance for held-out prior-predictive datasets
    config = try model_selection_config(system, run) catch; nothing end
    if !isnothing(config) && isfile(joinpath(models_path(config), classifier_filename(config, 1)))
        embed = load_classifier(config, 1).network[1:end-1]
        E = embed(sim.y[:, 1:n_embedding_reference])
        μ, σ = mean(E, dims = 2), std(E, dims = 2) .+ 1f-6
        Es = (E .- μ) ./ σ
        nn(e) = minimum(vec(sum(abs2, Es .- e, dims = 1)))
        holdout = (embed(sim.y[:, n_embedding_reference .+ (1:n_embedding_holdout)]) .- μ) ./ σ
        reference = sqrt.([nn(e) for e in eachcol(holdout)])
        y_obs = encode_counts(real_data[s].counts; censored = system.censoring_upper > 0)
        d_obs = sqrt(nn((embed(reshape(y_obs, :, 1)) .- μ) ./ σ))
        p = mean(reference .<= d_obs)
        ## Only a large distance is a concern, so the flag is one-sided
        push!(domain_rows, (system = s, data = real_data[s].name, check = "embedding_nn_distance", observed = Float64(d_obs),
                            percentile = p, flag = p <= 0.95 ? "within" : (p <= 0.99 ? "near boundary" : "outside")))
    else
        println("No trained classifier for system $s; skipping the embedding check.")
    end
end
CSV.write(joinpath(output_path, "prior_predictive_quantiles.csv"), DataFrame(quantile_table))
CSV.write(joinpath(output_path, "domain_checks.csv"), DataFrame(domain_rows))
println(DataFrame(domain_rows))
