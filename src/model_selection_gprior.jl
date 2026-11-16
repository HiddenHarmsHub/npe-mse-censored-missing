## Prior sensitivity for the supplement
include("mse_functions.jl")
include("mcmc_functions.jl")
include("model_selection_functions.jl")
include("model_selection_reference.jl")

τ_grid = collect(-4.0:1.0:8.0)   # log σ²; τ = 0 is the unit-information prior
n_draws = 50_000
n_keep = 500
B = 20_000
output_path = joinpath("output", "model_selection_gprior")
mkpath(output_path)

real_data = Dict("A" => load_silverman_data(), "B" => load_king_counts())
summarise(x) = (median = median(x), mean = mean(x), sd = std(x), q025 = quantile(x, 0.025), q975 = quantile(x, 0.975))

for s in ["A", "B"]
    system = model_selection_systems[s]
    K = system.K
    counts = real_data[s]
    M = sum_to_zero_map(K)
    X = one_hot_encode_parameters(K) * M
    labels = [model_label(model_mask(c, K), K) for c in 1:n_models(K)]
    Random.seed!(s == "A" ? 4040 : 4041)

    ## Evidence of every structure at every σ² on the grid, cached because it is the expensive step
    cache = joinpath(output_path, "$(s)_grid_fits.bson")
    grid = isfile(cache) ? BSON.load(cache)[:grid] : map(τ_grid) do τ
        fits = Vector{Any}(undef, n_models(K))
        Threads.@threads for c in 1:n_models(K)
            fits[c] = log_evidence(counts, model_mask(c, K); system.censoring_lower, system.censoring_upper,
                                   priors = gprior_priors(exp(τ)), n_draws, n_keep, X, M)
        end
        println("System $s: log σ² = $τ done")
        fits
    end
    isfile(cache) || BSON.@save cache grid
    log_ev = [grid[g][c].log_evidence for g in eachindex(τ_grid), c in 1:n_models(K)]
    CSV.write(joinpath(output_path, "$(s)_log_evidence_grid.csv"),
              DataFrame([(model = c, structure = labels[c], log_sigma2 = τ_grid[g], log_evidence = grid[g][c].log_evidence,
                          log_evidence_se = grid[g][c].log_evidence_se, ess = grid[g][c].ess)
                         for g in eachindex(τ_grid) for c in 1:n_models(K)]))

    ## Unit information: the grid point σ² = 1
    uip = grid[findfirst(==(0.0), τ_grid)]
    hyper = map(1:n_models(K)) do c
        h = hyper_g_evidence(log_ev[:, c], τ_grid)
        g = rand(Categorical(h.grid_weights), n_keep)
        θ = reduce(hcat, [grid[gi][c].θ[:, rand(1:n_keep)] for gi in g])
        (log_evidence = h.log_evidence, log_evidence_se = maximum(grid[gi][c].log_evidence_se for gi in unique(g)),
         ess = minimum(grid[gi][c].ess for gi in unique(g)), θ = θ, grid_weights = h.grid_weights)
    end
    fit_sets = Dict("unit_information" => uip, "hyper_g" => hyper)

    model_priors = merge(Dict{String, Any}("primary" => primary_model_prior), sensitivity_model_priors)
    probs = DataFrame(model = 1:n_models(K), structure = labels, size = [model_size(model_mask(c, K)) for c in 1:n_models(K)])
    population, inclusion, sigma = NamedTuple[], NamedTuple[], NamedTuple[]
    for (prior_name, fits) in fit_sets, (model_prior_name, model_prior) in model_priors
        mix = reference_mixture(fits, counts, system; model_prior, B)
        probs[!, "$(prior_name)_$(model_prior_name)"] = mix.π
        for q in (:alpha, :lambda0, :N0, :N_obs, :N)
            push!(population, merge((coefficient_prior = prior_name, model_prior = model_prior_name, quantity = String(q)), summarise(Float64.(getproperty(mix, q)))))
        end
        for (j, (a, b)) in enumerate(enumerate_two_digit_numbers(K))
            push!(inclusion, (coefficient_prior = prior_name, model_prior = model_prior_name, interaction = "$(a)$(b)", probability = mix.inclusion[j]))
        end
        if prior_name == "hyper_g"
            w = reduce(+, [mix.π[c] .* hyper[c].grid_weights for c in 1:n_models(K)])
            append!(sigma, [(model_prior = model_prior_name, log_sigma2 = τ, posterior_mass = w[g]) for (g, τ) in enumerate(τ_grid)])
        end
    end
    CSV.write(joinpath(output_path, "$(s)_model_probabilities.csv"), probs)
    CSV.write(joinpath(output_path, "$(s)_population.csv"), DataFrame(population))
    CSV.write(joinpath(output_path, "$(s)_inclusion.csv"), DataFrame(inclusion))
    CSV.write(joinpath(output_path, "$(s)_hyper_g_sigma2.csv"), DataFrame(sigma))
    show(filter(r -> r.quantity == "N", DataFrame(population)), allrows = true)
    println()
end
println("g-prior sensitivity written to $output_path")
