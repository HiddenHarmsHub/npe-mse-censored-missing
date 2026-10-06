## Compare the model-selection runs (coefficient priors N(0, 4²), N(0, 2²), N(0, 1), and the original
## N(0, 4²) networks against the same priors retrained with the new settings) side by side.
## Reads each run's combined outputs (model_selection_combine.jl, model_selection_real_data.jl and
## model_selection_prior_predictive.jl) and the training logs, for every run whose results exist, and
## writes the tables to output/model_selection_comparison/.
## Run locally: julia --project src/model_selection_compare_runs.jl
##
## Each run is calibrated against its own prior, so simulation coverage is not a ranking of priors. The
## comparison answers: how sensitive are the real-data conclusions to the prior; how well does the neural
## approximation match exact enumeration under each prior; how stable is training; and does each prior's
## training distribution cover the real data.
include("mse_functions.jl")
include("model_selection_functions.jl")

output_path = joinpath("output", "model_selection_comparison")
mkpath(output_path)

## Widest prior first; the original training before its retrained counterpart
runs = filter(r -> isdir(model_selection_output_path(r)), sort(collect(values(model_selection_runs)), by = r -> (-r.beta_sd, r.run)))
println("Runs with results: ", join([r.run for r in runs], ", "))
prior_label(r) = r.label
read_if(file; kw...) = isfile(file) ? CSV.read(file, DataFrame; kw...) : nothing
## The ensemble (rep 0) where it exists, otherwise replicate 1
preferred_rep(df) = 0 in df.rep ? 0 : 1
tag(df, r) = (df.run .= r.run; df.prior .= prior_label(r); df)
## Summaries of possibly empty groups (e.g. a reference dataset not in the evaluated part of the test set)
emedian(x) = isempty(x) ? missing : median(x)
emean(x) = isempty(x) ? missing : mean(x)
## Stack per-system tables, or nothing when a run has none yet (e.g. its real-data step has not run)
stack_systems(tables) = (t = filter(!isnothing, tables); isempty(t) ? nothing : vcat(t...; cols = :union))
stack_runs(f) = vcat(filter(!isnothing, [f(r) for r in runs])...; cols = :union)

## ---- Training: epochs run, best epoch and best validation loss per network ----
training = stack_runs() do r
    logs = joinpath(model_selection_models_path(r), "logs")
    isdir(logs) || return nothing
    rows = map(readdir(logs)) do d
        L = Matrix(CSV.read(joinpath(logs, d, "loss_per_epoch.csv"), DataFrame; header = false))
        n = size(L, 1) - 1
        (network = d, kind = startswith(d, "classifier") ? "classifier" : "cnpe", system = match(r"^[a-z]+_(\d+)_", d).captures[1] == "5" ? "A" : "B",
         rep = parse(Int, match(r"_(\d+)$", d).captures[1]), epochs = n, best_epoch = argmin(L[:, 2]) - 1,
         best_validation = minimum(L[:, 2]), final_train = L[end, 1],
         hours = CSV.read(joinpath(logs, d, "train_time.csv"), DataFrame; header = false)[1, 1] / 3600)
    end
    tag(DataFrame(rows), r)
end
CSV.write(joinpath(output_path, "training.csv"), training)

## ---- Simulation study: structure posterior and population coverage ----
structure = stack_runs() do r
    df = read_if(joinpath(model_selection_output_path(r), "structure_summary.csv"))
    isnothing(df) ? nothing : tag(filter(:rep => ==(preferred_rep(df)), df), r)
end
CSV.write(joinpath(output_path, "structure.csv"), structure)

population = stack_runs() do r
    df = read_if(joinpath(model_selection_output_path(r), "population_summary.csv"))
    isnothing(df) && return nothing
    df = filter(x -> x.rep == preferred_rep(df) && x.target_prior == "primary" && x.quantity in ("N0", "N"), df)
    tag(select(df, :system, :rep, :method, :quantity, :n, :median_abs_error, :median_ape, :mean_error,
               r"^coverage_", :median_width_95, :mean_score_95), r)
end
CSV.write(joinpath(output_path, "simulation_population.csv"), population)

## ---- Agreement with exact enumeration on the reference subset ----
agreement = stack_runs() do r
    path = model_selection_output_path(r)
    rc = read_if(joinpath(path, "reference_comparison.csv"))
    isnothing(rc) && return nothing
    "neural_rep" in names(rc) || (rc.neural_rep .= 1)  # combined before ensembles existed: replicate 1
    sm = read_if(joinpath(path, "simulation_models.csv"))
    rows = map(collect(groupby(rc, :system))) do g
        v = dropmissing(g, [:neural_median, :reference_median])
        rel = (v.neural_median .- v.reference_median) ./ max.(v.reference_median, 1)
        neural = isnothing(sm) ? nothing : filter(x -> x.system == g.system[1] && x.rep == g.neural_rep[1] && x.dataset in g.dataset, sm)
        (system = g.system[1], neural_rep = g.neural_rep[1], n = nrow(g),
         median_total_variation = median(g.total_variation), median_pip_max_abs_diff = median(g.pip_max_abs_diff),
         same_top_model = mean(g.same_top_model), reference_top_in_neural_top5 = mean(g.neural_rank_of_reference_top .<= 5),
         mean_reference_entropy = mean(g.reference_entropy),
         mean_neural_entropy = isnothing(neural) ? missing : emean(neural.entropy),
         neural_log_score = isnothing(neural) ? missing : emean(neural.log_score),
         n_with_neural_N0 = nrow(v),
         N0_median_abs_relative_diff = emedian(abs.(rel)), N0_share_relative_diff_over_25pct = emean(abs.(rel) .> 0.25),
         reference_coverage_95 = emean(v.reference_lower_95 .<= v.truth .<= v.reference_upper_95),
         neural_coverage_95 = emean(v.neural_lower_95 .<= v.truth .<= v.neural_upper_95))
    end
    tag(DataFrame(rows), r)
end
CSV.write(joinpath(output_path, "reference_agreement.csv"), agreement)

## ---- Real data: population size, inclusion probabilities, leading structures, domain checks ----
real_population = stack_runs() do r
    stack_systems(map(["A", "B"]) do s
        df = read_if(joinpath(model_selection_output_path(r), "real_data", "$(s)_population.csv"))
        isnothing(df) ? nothing : (df.system .= s; tag(filter(:quantity => in(["N0", "N"]), df), r))
    end)
end
CSV.write(joinpath(output_path, "real_data_population.csv"), real_population)

real_inclusion = stack_runs() do r
    stack_systems(map(["A", "B"]) do s
        df = read_if(joinpath(model_selection_output_path(r), "real_data", "$(s)_inclusion.csv"); types = Dict(:pair => String))
        isnothing(df) ? nothing : (df.system .= s; tag(df, r))
    end)
end
CSV.write(joinpath(output_path, "real_data_inclusion.csv"), real_inclusion)

real_top = stack_runs() do r
    stack_systems(map(["A", "B"]) do s
        df = read_if(joinpath(model_selection_output_path(r), "real_data", "$(s)_model_probabilities.csv"); types = Dict(:label => String))
        isnothing(df) && return nothing
        rows = [(system = s, source = col, rank = k, structure = row.label, prob = row[col])
                for col in ["neural_primary", "reference_primary"] for (k, row) in enumerate(eachrow(first(sort(df, col, rev = true), 5)))]
        tag(DataFrame(rows), r)
    end)
end
CSV.write(joinpath(output_path, "real_data_top_structures.csv"), real_top)

domain = stack_runs() do r
    df = read_if(joinpath(model_selection_output_path(r), "prior_predictive", "domain_checks.csv"))
    isnothing(df) ? nothing : tag(df, r)
end
CSV.write(joinpath(output_path, "domain_checks.csv"), domain)

## A compact view of the headline real-data result
if nrow(real_population) > 0
    headline = filter(x -> x.quantity == "N" && x.method in ("neural_bma", "reference_bma", "neural_conditional: all interactions"), real_population)
    show(select(headline, :prior, :system, :method, :median => ByRow(round) => :median, :q025 => ByRow(round) => :q025, :q975 => ByRow(round) => :q975), allrows = true)
    println()
end
println("Run comparison written to $output_path")
