## Robustness study: test sets for every scenario and the prior-predictive reference sets.
include("mse_functions.jl")
include("mcmc_functions.jl")
include("model_selection_functions.jl")
include("model_selection_reference.jl")
include("robustness_functions.jl")

mkpath(joinpath(robustness_path, "test_data"))

real_data = Dict("A" => load_silverman_data(), "B" => load_king_counts())
scenario_rows = NamedTuple[]
for s in ["A", "B"]
    system = model_selection_systems[s]
    K, J = system.K, binomial(system.K, 2)

    file = robustness_reference_file(s)
    if !isfile(file)
        Random.seed!(stable_seed("$(s)_reference"))
        ref = simulate_scenario(first(robustness_scenarios(system)), system, robustness_reference_size)
        y = encode_counts(ref.counts_obs; censored = system.censoring_upper > 0)
        summaries = DataFrame([data_summaries(c, K) for c in eachcol(ref.counts_obs)])
        summaries = Dict(k => Float64.(filter(isfinite, summaries[!, k])) for k in diagnostic_summaries)
        BSON.@save file y summaries
        println("Generated the prior-predictive reference datasets for system $s")
    end

    Random.seed!(stable_seed("$(s)_observed"))
    observed = log_evidence(real_data[s], trues(J); system.censoring_lower, system.censoring_upper,
                            priors = coefficient_priors(), n_draws = 200_000, n_keep = 20_000)
    println("System $s: exact posterior of the real data, importance ESS $(round(Int, observed.ess))")

    for scenario in robustness_scenarios(system; observed = Float64.(observed.θ))
        push!(scenario_rows, (system = s, scenario = scenario.name, label = scenario.label, group = scenario.group,
                              censoring_lower = scenario.censoring.lower, censoring_upper = scenario.censoring.upper,
                              censoring_min_lists = scenario.censoring.min_lists))
        file = robustness_test_file(s, scenario.name)
        isfile(file) && continue
        seed = stable_seed("$(s)_$(scenario.name)")
        Random.seed!(seed)
        test = simulate_scenario(scenario, system, robustness_test_size)
        y = encode_counts(test.counts_obs; censored = system.censoring_upper > 0)
        (; pars, deltas, counts, counts_obs, N0, N, capped, tries, censoring_lower, censoring_upper, min_lists) = test
        BSON.@save file pars deltas counts counts_obs y N0 N capped tries censoring_lower censoring_upper min_lists seed
        println("System $s, $(scenario.name) ($(scenario.label)): $(count(capped)) capped, mean tries $(round(mean(tries), digits = 2))")
    end
end
CSV.write(joinpath(robustness_path, "scenarios.csv"), DataFrame(scenario_rows))
