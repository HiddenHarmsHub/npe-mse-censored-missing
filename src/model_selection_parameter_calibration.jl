## Calibration of the conditional NPE's parameter posteriors q(θ | y, m) given the true structure, for every
## parameter: α, each β_k, and each latent γ̃_kl split by whether the interaction is active. Simulation-based
## calibration ranks and central-interval coverage, per conditional NPE replicate on the same datasets.
## An inactive γ̃ does not enter the likelihood, so its exact posterior is its prior N(0, σ_γ²); the
## posterior sd ratio to the prior sd checks that the flow has learned this.
## Run locally after model_selection_test_data.jl: MS_RUN=<run> julia --project src/model_selection_parameter_calibration.jl
include("mse_functions.jl")
include("model_selection_functions.jl")

run = model_selection_run()
n_datasets = 2000
n_draws = 1000
output_path = model_selection_output_path(run)
levels = round.(Int, 100 .* coverage_levels)

rank_rows = NamedTuple[]
for system in sort(collect(values(model_selection_systems)), by = s -> s.system)
    config = model_selection_config(system, run)
    K, J = system.K, binomial(system.K, 2)
    test = BSON.load(model_selection_test_file(run, system))
    names_K = param_names(K)
    ## The first n_datasets that did not hit the rpois cap
    idx = filter(i -> !is_capped(effective_parameters(Float64.(test[:pars][:, i]), Bool.(test[:masks][:, i]))), 1:size(test[:y], 2))[1:n_datasets]
    selected = config.cnpe_combination == :best ? best_cnpe_replicate(config) : 0
    for rep in cnpe_replicates(config)
        cnpe = load_cnpe(config, rep)
        Random.seed!(10^4 * K + rep)
        for i in idx
            mask = Bool.(test[:masks][:, i])
            θ = Float64.(conditional_posterior(cnpe, test[:y][:, i], mask, n_draws))
            size(θ, 2) == 0 && continue
            truth = Float64.(test[:pars][:, i])
            for (p, name) in enumerate(names_K)
                role = p == 1 ? "alpha" : p <= 1 + K ? "beta" : mask[p-1-K] ? "gamma active" : "gamma inactive"
                d = θ[p, :]
                push!(rank_rows, (system = system.system, rep = rep, selected = rep == selected, dataset = i, parameter = name, role = role,
                    truth = truth[p], rank = randomised_rank(d, truth[p]), posterior_mean = mean(d), posterior_sd = std(d),
                    NamedTuple(Symbol("inside_$L") => ((lo, hi) = quantile(d, [(1 - L / 100) / 2, (1 + L / 100) / 2]); lo <= truth[p] <= hi) for L in levels)...))
            end
        end
        println("System $(system.system), conditional NPE replicate $rep done")
    end
end
ranks = DataFrame(rank_rows)
CSV.write(joinpath(output_path, "parameter_calibration_ranks.csv"), ranks)

## ECDF distance of the ranks from uniform, with the 5% Kolmogorov–Smirnov critical value 1.358 / √n
rank_ks(r) = (n = length(r); maximum(abs.(sort(r) .- (1:n) ./ n)))
prior_sd = run.gamma_sd
summarise(g) = (n = nrow(g), [Symbol("coverage_$L") => mean(g[!, "inside_$L"]) for L in levels]...,
    rank_ks = rank_ks(g.rank), ks_critical_95 = 1.358 / sqrt(nrow(g)), mean_posterior_sd = mean(g.posterior_sd),
    sd_ratio_to_prior = g.role[1] == "gamma inactive" ? mean(g.posterior_sd) / prior_sd : missing,
    mean_posterior_mean = g.role[1] == "gamma inactive" ? mean(g.posterior_mean) : missing)
by_parameter = combine(summarise, groupby(ranks, [:system, :rep, :selected, :role, :parameter]))
by_role = combine(summarise, groupby(ranks, [:system, :rep, :selected, :role]))
CSV.write(joinpath(output_path, "parameter_calibration.csv"), sort(by_parameter, [:system, :rep, :role, :parameter]))
CSV.write(joinpath(output_path, "parameter_calibration_by_role.csv"), sort(by_role, [:system, :rep, :role]))
show(select(filter(:selected => identity, by_role), :system, :rep, :role, :n, :coverage_50, :coverage_95, :rank_ks, :ks_critical_95, :sd_ratio_to_prior), allrows = true)
println("\nParameter calibration written to $output_path")
