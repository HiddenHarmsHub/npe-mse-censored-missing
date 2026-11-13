## Monte Carlo stability of the real-data reference model average: repeat the enumeration of all
## structures' marginal likelihoods by importance sampling under independent seeds.
## Run from the repo root: MS_RUN=b4_g4x julia -t auto --project src/model_selection_reference_repeat.jl
include("mse_functions.jl")
include("mcmc_functions.jl")
include("model_selection_functions.jl")
include("model_selection_reference.jl")
using CSV, DataFrames, Random, Statistics

run = model_selection_run()
n_reference_draws = 100_000   # as in model_selection_real_data.jl
n_draws = 20_000
seeds = 1:5
top_k = 5

output_path = joinpath(model_selection_output_path(run), "real_data")
real_data = Dict("A" => load_silverman_data(), "B" => load_king_counts())

rows, structure_rows = NamedTuple[], NamedTuple[]
for s in ["A", "B"], seed in seeds
    system = model_selection_systems[s]
    K = system.K
    config = model_selection_config(system, run)
    primary = BetaBinomialModelPrior(config.a, config.b)
    counts_obs = real_data[s]
    ## Structures ranked by the published reference probabilities
    published = sort(CSV.read(joinpath(output_path, "$(s)_model_probabilities.csv"), DataFrame), :reference_primary, rev = true)
    top = published.model[1:top_k]

    fits = Vector{Any}(undef, n_models(K))
    t = @elapsed Threads.@threads for c in 1:n_models(K)
        rng = Xoshiro(hash((s, seed, c)))
        fits[c] = log_evidence(counts_obs, model_mask(c, K); system.censoring_lower, system.censoring_upper,
                               priors = coefficient_priors(run), n_draws = n_reference_draws, rng)
    end
    ref = reference_mixture(fits, counts_obs, system; model_prior = primary, B = n_draws, rng = Xoshiro(seed))
    material = ref.π .> 0.01
    sizes = vec(sum(enumerate_models(K), dims = 1))
    ## Law of total variance over the structures holding 99.9% of the reference mass:
    ## Var(N0) = E_m[Var(N0 | m)] + Var_m(E[N0 | m]), from each structure's importance-sampling draws
    order = sortperm(ref.π, rev = true)
    leading = order[1:searchsortedfirst(cumsum(ref.π[order]), 0.999)]
    per_structure = [Float64.(population_draws(fits[c].θ, counts_obs; system.censoring_lower, system.censoring_upper, rng = Xoshiro(hash((s, seed, c, :vd)))).N0) for c in leading]
    vd = variance_decomposition(reduce(vcat, per_structure), reduce(vcat, [fill(c, length(d)) for (c, d) in zip(leading, per_structure)]);
                                weights = Dict(c => ref.π[c] for c in leading))
    push!(rows, (system = s, seed = seed, seconds = t,
                 N_median = median(ref.N), N_q025 = quantile(ref.N, 0.025), N_q975 = quantile(ref.N, 0.975),
                 N0_median = median(ref.N0), expected_size = sum(sizes .* ref.π),
                 top_structure = argmax(ref.π), top_prob = maximum(ref.π),
                 n_material = count(material), min_ess_material = minimum(ref.ess[material]),
                 median_ess_material = median(ref.ess[material]), max_se_material = maximum(ref.log_evidence_se[material]),
                 vd_n_structures = length(leading), vd_within = vd.within, vd_between = vd.between, between_share_N0 = vd.between_share))
    for (rank, c) in enumerate(top)
        push!(structure_rows, (system = s, seed = seed, published_rank = rank, model = c,
                               label = model_label(model_mask(c, K), K), prob = ref.π[c], published_prob = published.reference_primary[rank]))
    end
    println("System $s seed $seed: N = $(median(ref.N)) ($(quantile(ref.N, 0.025)), $(quantile(ref.N, 0.975))), top prob $(round(maximum(ref.π), digits = 3)), $(round(t, digits = 1)) s")
end

summary = DataFrame(rows)
structures = DataFrame(structure_rows)
CSV.write(joinpath(output_path, "reference_repeat.csv"), summary)
CSV.write(joinpath(output_path, "reference_repeat_structures.csv"), structures)
show(combine(groupby(structures, [:system, :published_rank, :label]), :published_prob => first => :published, :prob => mean => :mean, :prob => std => :sd), allrows = true)
println()
show(combine(groupby(summary, :system), [c => (x -> (mean(x), std(x))) => c for c in [:N_median, :N_q025, :N_q975, :expected_size, :top_prob, :min_ess_material, :median_ess_material, :between_share_N0]]...), allcols = true)
