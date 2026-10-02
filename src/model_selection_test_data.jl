## Fixed prior-predictive test sets for the model-selection systems, read by
## model_selection_simulation.jl and model_selection_reference_study.jl.
## Single process: MS_RUN=<run> julia --project src/model_selection_test_data.jl
include("mse_functions.jl")
include("model_selection_functions.jl")

run = model_selection_run()
test_size = 10_000
mkpath(model_selection_output_path(run))
println("Run $(run.run): β ~ N($(run.beta_mean), $(run.beta_sd)²), γ ~ N(0, $(run.gamma_sd)²)")

for (system, seed) in [(model_selection_systems["A"], 43), (model_selection_systems["B"], 44)]
    file = model_selection_test_file(run, system)
    isfile(file) && continue
    Random.seed!(seed)
    test = simulate_model_selection_data(test_size, system; priors = coefficient_priors(run))
    masks, pars, counts, counts_obs, y, N0, N = Matrix(test.masks), test.pars, test.counts, test.counts_obs, test.y, test.N0, test.N
    BSON.@save file masks pars counts counts_obs y N0 N
    println("Generated test data for system $(system.system)")
end
