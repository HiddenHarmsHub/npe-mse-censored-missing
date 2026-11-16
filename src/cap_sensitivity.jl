## Compare the cap-sensitivity networks (cap_sensitivity_train.jl) with the shipped networks on the benchmark datasets
include("mse_functions.jl")
using CSV, DataFrames, Random, Statistics

const b = (n_lists = 5, width = 256, n_hidden = 3, train_size = train_size_default, censoring_lower = 0, censoring_upper = 10, m = 1)
const probs = [0.025, 0.5, 0.975]
nbe_file = "model_$(b.n_lists)_$(b.width)_$(b.n_hidden)_$(b.train_size)_$(b.censoring_lower)_$(b.censoring_upper)_$(b.m).bson"
paths = Dict("shipped" => (nbe = joinpath("output", "models_nbe"), npe = joinpath("output", "models_npe")),
             [v => (nbe = joinpath("output", "models_cap_sensitivity", v, "nbe"), npe = joinpath("output", "models_cap_sensitivity", v, "npe"))
              for v in ("redraw", "reseed")]...)

diagnostics = CSV.read(joinpath("output", "mcmc_diagnostics.csv"), DataFrame)
datasets = sort(diagnostics.dataset[.!diagnostics.capped .& (diagnostics.status .!= "failed")])
println("Evaluating on $(length(datasets)) benchmark datasets")
Z, pars = load_test_data(joinpath("output", "test_data"), b.n_lists, b.censoring_lower, b.censoring_upper)
Z, α = Z[:, datasets], pars[1, datasets]

## α quantiles (3 × n) for each method and variant
function alpha_quantiles(variant)
    p = paths[variant]
    nbe_median = load_model_nbe(nbe_file, p.nbe)(Z)[1, :]
    nbe_ci = load_model_nbe(replace(nbe_file, "model_" => "model_ci_"), p.nbe)(Z)
    n_pars = size(pars, 1)
    nbe = vcat(nbe_ci[1, :]', nbe_median', nbe_ci[n_pars + 1, :]')
    npe_model = load_model_npe(b.n_lists, b.width, b.n_hidden, b.train_size, b.censoring_lower, b.censoring_upper, b.m, p.npe)
    Random.seed!(1)
    npe = reduce(hcat, [boundedposteriorquantile(npe_model, reshape(z, :, 1), probs, 5000, 1.0, 10.0)[1, :] for z in eachcol(Z)])
    return Dict("NBE" => Float64.(nbe), "NPE" => Float64.(npe))
end

quantiles = Dict(v => alpha_quantiles(v) for v in ("shipped", "redraw", "reseed"))
λ = exp.(α)
rows = NamedTuple[]
for v in ("shipped", "redraw", "reseed"), method in ("NBE", "NPE")
    q = quantiles[v][method]
    err = exp.(q[2, :]) .- λ
    ape = abs.(err) ./ λ
    shift = v == "shipped" ? fill(0.0, 3) : [median(abs.(q[k, :] .- quantiles["shipped"][method][k, :])) for k in 1:3]
    push!(rows, (variant = v, method = method, n = length(α), bias = mean(err), bias_se = std(err) / sqrt(length(err)),
                 median_ape = 100 * median(ape), mean_ape = 100 * mean(ape),
                 coverage_95 = mean(q[1, :] .<= α .<= q[3, :]),
                 median_abs_diff_q025 = shift[1], median_abs_diff_q50 = shift[2], median_abs_diff_q975 = shift[3]))
end
result = DataFrame(rows)
CSV.write(joinpath("output", "cap_sensitivity.csv"), result)
show(result, allcols = true)
println()
