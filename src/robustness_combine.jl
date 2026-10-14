## Robustness study: summary tables with Monte Carlo standard errors.
include("mse_functions.jl")
include("model_selection_functions.jl")
include("robustness_functions.jl")

results_path = joinpath(robustness_path, "simulation")
read_chunks(kind) = reduce((a, b) -> vcat(a, b; cols = :union),
    [CSV.read(joinpath(results_path, f), DataFrame) for f in readdir(results_path) if endswith(f, "_$(kind).csv")])
datasets = read_chunks("datasets")
posteriors = read_chunks("posteriors")
posteriors = semijoin(posteriors, datasets, on = [:experiment, :system, :scenario, :sigma, :dataset])
CSV.write(joinpath(robustness_path, "datasets.csv"), datasets)
CSV.write(joinpath(robustness_path, "posteriors.csv"), posteriors)
scenarios = CSV.read(joinpath(robustness_path, "scenarios.csv"), DataFrame)

keys_experiment = [:experiment, :system, :scenario, :sigma]
valid = filter(r -> !r.capped, datasets)
p = semijoin(posteriors, valid, on = vcat(keys_experiment, :dataset))
n_without_draws = combine(groupby(p, vcat(keys_experiment, :method, :quantity)), :median => (x -> count(isnan, x)) => :n_without_draws)
p = filter(r -> !isnan(r.median), p)

summarise(df, by) = combine(groupby(df, by)) do g
    s = summarise_metrics(g)
    isnothing(s) ? DataFrame() : DataFrame([s])
end

dataset_info = combine(groupby(datasets, keys_experiment),
    nrow => :n_datasets, :capped => mean => :capped_share, :exact_ess => (x -> mean(x .< 1000)) => :exact_low_ess_share,
    :nis_ess => median => :median_nis_ess, :empty_overlap => mean => :empty_overlap_share,
    :n_censored => mean => :mean_censored, :n_zero => mean => :mean_zero,
    :time_exact => median => :median_time_exact, :time_npe => median => :median_time_npe)
summary = summarise(p, vcat(keys_experiment, :method, :quantity))
summary = leftjoin(summary, n_without_draws, on = vcat(keys_experiment, :method, :quantity))
summary = leftjoin(summary, dataset_info, on = keys_experiment)
summary = leftjoin(summary, select(scenarios, :system, :scenario, :label, :group), on = [:system, :scenario])
sort!(summary, vcat(keys_experiment, :quantity, :method))
CSV.write(joinpath(robustness_path, "scenario_summary.csv"), summary)

band_edges = range(1, 10, length = 6)
band(α) = clamp(searchsortedlast(band_edges, α), 1, 5)
band_label(b) = "[$(round(band_edges[b], digits = 1)), $(round(band_edges[b + 1], digits = 1)))"
main = filter(r -> r.experiment == "main", p)
main.band = band_label.(band.(leftjoin(select(main, keys_experiment..., :dataset), select(valid, keys_experiment..., :dataset, :intercept),
                                         on = vcat(keys_experiment, :dataset), order = :left).intercept))
by_band = summarise(main, vcat(keys_experiment, :band, :method, :quantity))
CSV.write(joinpath(robustness_path, "by_band.csv"), sort!(by_band, vcat(keys_experiment, :band, :quantity, :method)))

reliable = filter(r -> r.exact_ess >= 1000, valid)
pr = semijoin(p, reliable, on = vcat(keys_experiment, :dataset))
exact = select(filter(r -> r.method == "exact", pr), keys_experiment..., :dataset, :quantity,
               :median => :exact_median, :lower_95 => :exact_lower_95, :upper_95 => :exact_upper_95, :sd => :exact_sd, :width_95 => :exact_width_95)
neural = innerjoin(filter(r -> r.method != "exact", pr), exact, on = vcat(keys_experiment, :dataset, :quantity))
neural = filter(r -> r.exact_sd > 0, neural)
neural.d_median = (neural.median .- neural.exact_median) ./ neural.exact_sd
neural.d_lower = (neural.lower_95 .- neural.exact_lower_95) ./ neural.exact_sd
neural.d_upper = (neural.upper_95 .- neural.exact_upper_95) ./ neural.exact_sd
neural.width_ratio = neural.width_95 ./ neural.exact_width_95
approx_summary(g) = (n = nrow(g),
    mean_d_median = mean(g.d_median), mean_d_median_se = mcse_mean(g.d_median), mean_abs_d_median = mean(abs.(g.d_median)),
    mean_d_lower = mean(g.d_lower), mean_d_lower_se = mcse_mean(g.d_lower), mean_d_upper = mean(g.d_upper), mean_d_upper_se = mcse_mean(g.d_upper),
    mean_abs_d_tails = mean(vcat(abs.(g.d_lower), abs.(g.d_upper))),
    share_material = mean(abs.(g.d_median) .> 0.5), share_material_se = mcse_prop(mean(abs.(g.d_median) .> 0.5), nrow(g)),
    median_width_ratio = median(g.width_ratio))
approximation = combine(groupby(neural, vcat(keys_experiment, :method, :quantity)), g -> DataFrame([approx_summary(g)]))
CSV.write(joinpath(robustness_path, "approximation.csv"), sort!(approximation, vcat(keys_experiment, :quantity, :method)))

flags = Dict(
    "nis_ess_below_0.1" => r -> r.nis_ess < 0.1,
    "embedding_above_0.99" => r -> r.embedding_percentile > 0.99,
    "summary_tail_below_0.005" => r -> r.summary_tail < 0.005,
)
flag_info = select(filter(r -> r.experiment == "main", reliable), keys_experiment..., :dataset, :nis_ess, :embedding_percentile, :summary_tail)
npe_alpha = innerjoin(filter(r -> r.method in ("npe", "nbe") && r.quantity == "alpha", neural), flag_info, on = vcat(keys_experiment, :dataset))
npe_alpha = leftjoin(npe_alpha, select(scenarios, :system, :scenario, :group), on = [:system, :scenario])
flag_rows = NamedTuple[]
for (flag, f) in flags, scope in ["all", "in distribution", "parameter shift", "misspecification", "censoring"]
    d = scope == "all" ? npe_alpha : filter(r -> r.group == scope, npe_alpha)
    for method in ["npe", "nbe"], status in [true, false]
        g = filter(r -> r.method == method && f(r) == status, d)
        nrow(g) == 0 && continue
        push!(flag_rows, (flag = flag, scope = scope, method = method, flagged = status, n = nrow(g), share_flagged = mean(f.(eachrow(filter(r -> r.method == method, d)))),
            mean_abs_d_median = mean(abs.(g.d_median)), share_material = mean(abs.(g.d_median) .> 0.5),
            share_material_se = mcse_prop(mean(abs.(g.d_median) .> 0.5), nrow(g)),
            coverage_95 = mean(g.inside_95), coverage_95_se = mcse_prop(mean(g.inside_95), nrow(g))))
    end
end
CSV.write(joinpath(robustness_path, "diagnostic_flags.csv"), DataFrame(flag_rows))

ablation_pairs = [("main", 4), ("ablation", 4)]
ab = filter(r -> r.system == "B" && r.scenario == "S0" && (r.experiment, r.sigma) in ablation_pairs, p)
ab.censored = ab.experiment .== "main"
ablation = summarise(ab, [:censored, :method, :quantity])
ablation_approx = filter(r -> r.system == "B" && r.scenario == "S0" && (r.experiment, r.sigma) in ablation_pairs, approximation)
ablation_approx.censored = ablation_approx.experiment .== "main"
ablation = leftjoin(ablation, select(ablation_approx, :censored, :method, :quantity, :mean_abs_d_median, :mean_abs_d_tails, :share_material, :median_width_ratio),
                    on = [:censored, :method, :quantity])
CSV.write(joinpath(robustness_path, "censoring_ablation.csv"), sort!(ablation, [:quantity, :method, :censored]))

grid_file = joinpath("output", "intercept_estimate_comparison.csv")
if isfile(grid_file)
    grid = CSV.read(grid_file, DataFrame)
    grid_rows = NamedTuple[]
    for g in groupby(grid, [:n_lists, :width, :n_hidden, :train_size, :censoring_lower, :censoring_upper]), method in ["NBE", "NPE"]
        est, lo, hi = g[!, "intercept_$method"], g[!, "intercept_$(method)_lower_ci"], g[!, "intercept_$(method)_upper_ci"]
        ok = isfinite.(est)
        α, est, lo, hi = g.intercept_truth[ok], est[ok], lo[ok], hi[ok]
        e = exp.(est) .- exp.(α)
        rel = e ./ exp.(α)
        ape = abs.(rel)
        cov = mean(lo .<= α .<= hi)
        push!(grid_rows, (n_lists = g.n_lists[1], width = g.width[1], n_hidden = g.n_hidden[1], train_size = g.train_size[1],
            censoring_lower = g.censoring_lower[1], censoring_upper = g.censoring_upper[1], method = method, n = length(α), n_failed = count(.!ok),
            relative_bias = mean(rel), relative_bias_se = mcse_mean(rel), median_relative_error = median(rel),
            median_ape = median(ape), median_ape_se = mcse_median(ape), mean_ape = mean(ape), mean_ape_se = mcse_mean(ape),
            alpha_bias = mean(est .- α), alpha_bias_se = mcse_mean(est .- α), alpha_mae = mean(abs.(est .- α)), alpha_mae_se = mcse_mean(abs.(est .- α)),
            coverage_95 = cov, coverage_95_se = mcse_prop(cov, length(α)), mean_width_95 = mean(hi .- lo)))
    end
    CSV.write(joinpath(robustness_path, "grid_summary.csv"), DataFrame(grid_rows))
else
    println("$grid_file not found (it is not committed); skipping the grid summary.")
end

overview = filter(r -> r.experiment == "main" && r.quantity in ("N0", "lambda0"), summary)
show(select(overview, :system, :scenario, :method, :quantity, :n, :relative_bias, :median_ape, :coverage_95, :capped_share, :exact_low_ess_share), allrows = true)
println()
