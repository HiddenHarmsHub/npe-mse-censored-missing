## Fixed prior-predictive test sets for the model-selection systems, read by
## model_selection_simulation.jl and model_selection_reference_study.jl.
## Single process: julia --project src/model_selection_test_data.jl
include("mse_functions.jl")
include("model_selection_functions.jl")

gamma_sd = 4
test_size = 10_000

output_path = joinpath("output", "model_selection")
mkpath(output_path)

for (system, seed) in [(model_selection_systems["A"], 43), (model_selection_systems["B"], 44)]
    file = joinpath(output_path, "test_data_$(system.system).bson")
    isfile(file) && continue
    Random.seed!(seed)
    test = simulate_model_selection_data(test_size, system; gamma_sd)
    masks, pars, counts, counts_obs, y, N0, N = Matrix(test.masks), test.pars, test.counts, test.counts_obs, test.y, test.N0, test.N
    BSON.@save file masks pars counts counts_obs y N0 N
    println("Generated test data for system $(system.system)")
end
