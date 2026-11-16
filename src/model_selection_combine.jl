## Combine the per-chunk model-selection outputs and compute the summary tables.
include("mse_functions.jl")
include("model_selection_functions.jl")

run = model_selection_run()
output_path = model_selection_output_path(run)

read_all(dir, pattern) = vcat([CSV.read(joinpath(dir, f), DataFrame) for f in readdir(dir) if occursin(pattern, f)]...; cols = :union)

## ---- Simulation study ----
## Without the per-chunk folder (it is not copied from the cluster by default) the combined CSVs are reused
sim_path = joinpath(output_path, "simulation")
if isdir(sim_path) && !isempty(readdir(sim_path))
    models = read_all(sim_path, r"_models\.csv$")
    posteriors = read_all(sim_path, r"_posteriors\.csv$")
    CSV.write(joinpath(output_path, "simulation_models.csv"), models)
    CSV.write(joinpath(output_path, "simulation_posteriors.csv"), posteriors)
else
    println("No per-chunk simulation results; using the combined simulation_models.csv and simulation_posteriors.csv")
    models = CSV.read(joinpath(output_path, "simulation_models.csv"), DataFrame)
    posteriors = CSV.read(joinpath(output_path, "simulation_posteriors.csv"), DataFrame)
end

## Rows with no posterior draws inside the intercept support have NaN metrics
"n_draws" in names(posteriors) || (posteriors.n_draws .= missing)
posteriors.no_draws = coalesce.(posteriors.n_draws .== 0, false)
no_draws = combine(groupby(posteriors, [:system, :rep, :method]), :no_draws => (x -> count(x) ÷ 4) => :datasets_without_draws, nrow => :rows)
CSV.write(joinpath(output_path, "datasets_without_draws.csv"), no_draws)
println("Datasets with no draws inside the intercept support, by method:")
show(filter(:datasets_without_draws => >(0), no_draws), allrows = true); println()
posteriors = filter(:no_draws => !, posteriors)

## Datasets that hit the rpois cap are not exact draws from the model, so they are excluded
valid = filter(:capped => !, models)
valid.truth_is_map = valid.map_model .== valid.true_model
valid.intercept_band = clamp.(floor.(Int, (valid.intercept .- 1) ./ 1.8), 0, 4)
valid = transform(groupby(valid, :system), :entropy => (e -> searchsortedfirst.(Ref(quantile(e, [1/3, 2/3])), e)) => :entropy_tercile)
levels = round.(Int, 100 .* coverage_levels)

## Interaction strength
gamma_band(g) = g < 0.5 ? "a: <0.5" : g < 1 ? "b: 0.5-1" : g < 2 ? "c: 1-2" : g < 4 ? "d: 2-4" : "e: >=4"
test_sets = Dict(s => BSON.load(model_selection_test_file(run, sys)) for (s, sys) in model_selection_systems if isfile(model_selection_test_file(run, sys)))
function active_gammas(s, i)
    t, K = test_sets[s], model_selection_systems[s].K
    return abs.(Float64.(t[:pars][(2+K):end, i]))[Bool.(t[:masks][:, i])]
end
strength_strata = Symbol[]
if length(test_sets) == length(unique(valid.system))
    valid.weakest_gamma = [(g = active_gammas(r.system, r.dataset); isempty(g) ? "none" : gamma_band(minimum(g))) for r in eachrow(valid)]
    valid.strongest_gamma = [(g = active_gammas(r.system, r.dataset); isempty(g) ? "none" : gamma_band(maximum(g))) for r in eachrow(valid)]
    strength_strata = [:weakest_gamma, :strongest_gamma]
else
    @warn "Test sets not found; skipping the interaction-strength strata. Run src/model_selection_test_data.jl for this run first."
end

## Structure posterior: scores, recovery and credible-set coverage, overall and by true model size
structure_summary(df) = combine(df,
    nrow => :n, :log_score => mean => :log_score, :prob_true => mean => :prob_true,
    :top1 => mean => :top1, :top5 => mean => :top5, :top10 => mean => :top10,
    :brier_inclusion => mean => :brier_inclusion, :entropy => mean => :entropy,
    [Symbol("in_set_$L") => mean => Symbol("set_coverage_$L") for L in levels]...)
CSV.write(joinpath(output_path, "structure_summary.csv"), structure_summary(groupby(valid, [:system, :rep])))
CSV.write(joinpath(output_path, "structure_by_size.csv"), structure_summary(groupby(valid, [:system, :rep, :true_size])))
isempty(strength_strata) || CSV.write(joinpath(output_path, "structure_by_strength.csv"), structure_summary(groupby(valid, [:system, :rep, :weakest_gamma])))

## Reliability of inclusion probabilities (10 equal-width bins) and of model-size probabilities
reliability = NamedTuple[]
size_calibration = NamedTuple[]
for g in groupby(valid, [:system, :rep])
    pairs = [replace(c, "incl_" => "") for c in names(g) if startswith(c, "incl_") && !all(ismissing, g[!, c])]
    p = reduce(vcat, [g[!, "incl_$q"] for q in pairs])
    t = reduce(vcat, [g[!, "true_$q"] for q in pairs])
    bin = clamp.(floor.(Int, p .* 10), 0, 9)
    for b in 0:9
        idx = bin .== b
        n = count(idx)
        n == 0 && continue
        f = mean(t[idx])
        push!(reliability, (system = g.system[1], rep = g.rep[1], bin = b, n = n, mean_predicted = mean(p[idx]), frequency = f, se = sqrt(f * (1 - f) / n)))
    end
    for c in names(g)
        (startswith(c, "size_") && !all(ismissing, g[!, c])) || continue
        k = parse(Int, replace(c, "size_" => ""))
        push!(size_calibration, (system = g.system[1], rep = g.rep[1], size = k, mean_predicted = mean(g[!, c]), frequency = mean(g.true_size .== k)))
    end
end
CSV.write(joinpath(output_path, "inclusion_reliability.csv"), DataFrame(reliability))

## Inclusion probability of each interaction by its true |γ| (inactive interactions as their own band)
if !isempty(strength_strata)
    strength = NamedTuple[]
    for g in groupby(valid, [:system, :rep])
        s, K = g.system[1], model_selection_systems[g.system[1]].K
        t = test_sets[s]
        pairs = ["$(a)$(b)" for (a, b) in enumerate_two_digit_numbers(K)]
        p = reduce(vcat, [g[!, "incl_$q"] for q in pairs])
        active = reduce(vcat, [Bool.(t[:masks][j, g.dataset]) for j in eachindex(pairs)])
        band = reduce(vcat, [gamma_band.(abs.(Float64.(t[:pars][1+K+j, g.dataset]))) for j in eachindex(pairs)])
        band[.!active] .= "inactive"
        for b in sort(unique(band))
            idx = band .== b
            push!(strength, (system = s, rep = g.rep[1], gamma_band = b, n = count(idx), mean_inclusion = mean(p[idx]),
                             share_above_50 = mean(p[idx] .> 0.5), se = std(p[idx]) / sqrt(count(idx))))
        end
    end
    CSV.write(joinpath(output_path, "inclusion_by_strength.csv"), DataFrame(strength))
end
CSV.write(joinpath(output_path, "size_calibration.csv"), DataFrame(size_calibration))

## Population inference
joined = innerjoin(posteriors, select(valid, :dataset, :system, :rep, :true_size, :intercept_band, :entropy_tercile, :truth_is_map,
                                       :n_censored, :n_zero, strength_strata..., r"^logw_"), on = [:dataset, :system, :rep])
weighted_mean(x, w) = sum(x .* w) / sum(w)
function population_summary(df, w)
    return (n = nrow(df), mean_error = weighted_mean(df.error, w), median_error = median(df.error),
            median_abs_error = median(df.abs_error), median_ape = median(filter(!isnan, df.ape)),
            [Symbol("coverage_$L") => weighted_mean(df[!, "inside_$L"], w) for L in levels]...,
            median_width_95 = median(df.width_95), mean_score_95 = weighted_mean(df.score_95, w),
            rank_ks = maximum(abs.(sort(df.rank) .- (1:nrow(df)) ./ nrow(df))))
end
summary_rows = NamedTuple[]
for g in groupby(joined, [:system, :rep, :method, :quantity]), target in ["primary", keys(sensitivity_model_priors)...]
    w = target == "primary" ? ones(nrow(g)) : exp.(g[!, "logw_$target"])
    push!(summary_rows, merge((system = g.system[1], rep = g.rep[1], method = g.method[1], quantity = g.quantity[1], target_prior = target), population_summary(g, w)))
end
CSV.write(joinpath(output_path, "population_summary.csv"), DataFrame(summary_rows))

stratum_rows = NamedTuple[]
for stratum in [:true_size, :intercept_band, :entropy_tercile, :truth_is_map, :n_censored, :n_zero, strength_strata...]
    for g in groupby(filter(:quantity => in(["N0", "N"]), joined), [:system, :rep, :method, :quantity, stratum])
        push!(stratum_rows, merge((system = g.system[1], rep = g.rep[1], method = g.method[1], quantity = g.quantity[1],
                                   stratum = String(stratum), level = string(g[1, stratum])), population_summary(g, ones(nrow(g)))))
    end
end
CSV.write(joinpath(output_path, "population_by_stratum.csv"), DataFrame(stratum_rows))

## ---- Reference model averaging and its agreement with the neural structure posterior ----
## Correlation is undefined when either inclusion vector is constant
safe_cor(a, b) = std(a) > 1e-12 && std(b) > 1e-12 ? cor(a, b) : missing
## The neural side is the ensemble when all replicates exist, otherwise replicate 1
reference_path = joinpath(output_path, "reference")
if isdir(reference_path) && !isempty(readdir(reference_path))
    reference_models = read_all(reference_path, r"_models\.csv$")
    reference_posteriors = read_all(reference_path, r"_posteriors\.csv$")
    CSV.write(joinpath(output_path, "reference_models.csv"), reference_models)
    CSV.write(joinpath(output_path, "reference_posteriors.csv"), reference_posteriors)
    any(f -> occursin("_check.csv", f), readdir(reference_path)) &&
        CSV.write(joinpath(output_path, "reference_checks.csv"), read_all(reference_path, r"_check\.csv$"))

    comparison = NamedTuple[]
    for s in unique(reference_models.system)
        system = model_selection_systems[s]
        K = system.K
        config = model_selection_config(system, run)
        neural_rep = all(r -> isfile(joinpath(models_path(config), classifier_filename(config, r))), classifier_replicates(config)) ? 0 : 1
        classifier = load_classifier(config, neural_rep)
        test = BSON.load(model_selection_test_file(run, system))
        masks = enumerate_models(K)
        for g in groupby(filter(:system => ==(s), reference_models), :dataset)
            i = g.dataset[1]
            ## Datasets that hit the rpois cap are not draws from the model; no structure fits them
            is_capped(effective_parameters(Float64.(test[:pars][:, i]), Bool.(test[:masks][:, i]))) && continue
            π_ref = g.prob[sortperm(g.model)]
            π_nn = model_probabilities(classifier, test[:y][:, i])
            push!(comparison, (system = s, dataset = i, neural_rep = neural_rep, total_variation = 0.5 * sum(abs.(π_nn .- π_ref)),
                pip_max_abs_diff = maximum(abs.(masks * π_nn .- masks * π_ref)), pip_correlation = safe_cor(masks * π_nn, masks * π_ref),
                same_top_model = argmax(π_nn) == argmax(π_ref), neural_rank_of_reference_top = findfirst(==(argmax(π_ref)), sortperm(π_nn, rev = true)),
                reference_entropy = -sum(p * log(p) for p in π_ref if p > 0), min_ess = minimum(g.ess), max_log_evidence_se = maximum(g.log_evidence_se)))
        end
    end
    comparison = DataFrame(comparison)
    neural_N0 = select(filter(r -> r.method == "bma" && r.quantity == "N0", posteriors), :system, :dataset, :rep => :neural_rep, :median => :neural_median, :lower_95 => :neural_lower_95, :upper_95 => :neural_upper_95)
    reference_N0 = select(filter(:quantity => ==("N0"), reference_posteriors), :system, :dataset, :truth, :median => :reference_median, :lower_95 => :reference_lower_95, :upper_95 => :reference_upper_95)
    comparison = leftjoin(comparison, innerjoin(neural_N0, reference_N0, on = [:system, :dataset]), on = [:system, :dataset, :neural_rep])
    CSV.write(joinpath(output_path, "reference_comparison.csv"), comparison)
end
println("Combined model-selection results written to $output_path")
