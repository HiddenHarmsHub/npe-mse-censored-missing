## Software validation of the reference samplers (run locally with e.g. julia --project -t 12 src/reference_validation.jl [comparison] [coverage])
## 1. comparison: NUTS, IRLS-MH and importance sampling agree on small simulated datasets
## 2. coverage: NUTS credible intervals have nominal coverage under prior-predictive simulation (K=3)
include("mse_functions.jl")
include("mcmc_functions.jl")
BLAS.set_num_threads(1)

parts = isempty(ARGS) ? ["comparison", "coverage"] : ARGS
output_path = joinpath("output", "reference_validation")
mkpath(output_path)

function simulate_counts(K, censoring_upper)
    pars = Float64.(sample_parameters(K))
    while is_capped(pars)
        pars = Float64.(sample_parameters(K))
    end
    Z = Int.(simulate_data(pars, 1, log_transform = false))[:, 1]
    counts = censoring_upper > 0 ? ifelse.(0 .<= Z .<= censoring_upper, -1, Z) : Z
    return counts, pars
end

if "comparison" in parts
## Cross-method comparison
Random.seed!(2024)
settings = vcat([(K, cu) for K in (3, 5) for cu in (0, 10) for _ in 1:12], [(3, 10), (5, 10)])
datasets = [(K = K, censoring_upper = cu, data = simulate_counts(K, cu)) for (K, cu) in settings]

## Each dataset is saved as it finishes so an interrupted run can resume
comparison_path = joinpath(output_path, "comparison")
mkpath(comparison_path)
todo = filter(i -> !isfile(joinpath(comparison_path, "dataset_$i.csv")), eachindex(datasets))
## One task per dataset so a slow dataset does not hold up a block of others
@sync for i in todo
    Threads.@spawn begin
    K, cu, (counts, pars) = datasets[i]
    X = one_hot_encode_parameters(K)
    names_K = param_names(K)

    chains, par_df, fit_df = run_reference_fit(counts, X; censoring_lower = 0, censoring_upper = cu, stages = nuts_stages[2:3])
    irls = fit_irls(counts, X; censoring_lower = 0, censoring_upper = cu)
    irls_par_df, irls_fit_df = mcmc_diagnostics(irls)
    pilot = passes_diagnostics(irls_fit_df) ? irls : chains
    pilot_draws = Array(select(DataFrame(pilot), r"intercept|betas|gammas"))
    is_ref = importance_reference(counts, X; censoring_lower = 0, censoring_upper = cu, n_draws = 2 * 10^5, pilot_draws = pilot_draws)

    sampler_rows(method, ch, pdf, status) = begin
        draws = select(DataFrame(ch), r"intercept|betas|gammas")
        DataFrame(
            method = method, parameters = names_K, true_values = pars,
            mean = mean.(eachcol(draws)), sd = std.(eachcol(draws)),
            lower_95ci = quantile.(eachcol(draws), 0.025), upper_95ci = quantile.(eachcol(draws), 0.975),
            mcse = pdf.mcse, status = status
        )
    end
    is_rows = DataFrame(
        method = "IS", parameters = names_K, true_values = pars,
        mean = is_ref.means, sd = is_ref.sds,
        lower_95ci = is_ref.quantiles[:, 1], upper_95ci = is_ref.quantiles[:, 3],
        mcse = is_ref.sds ./ sqrt(is_ref.ess), status = is_ref.ess >= 1e4 ? "passed" : "failed"
    )
    df = vcat(
        sampler_rows("NUTS", chains, par_df, fit_df.status[1]),
        sampler_rows("IRLS", irls, irls_par_df, passes_diagnostics(irls_fit_df) ? "passed" : "failed"),
        is_rows
    )
    df.dataset .= i
    df.n_lists .= K
    df.censoring_upper .= cu
    df.n_censored .= sum(counts .== -1)
    CSV.write(joinpath(comparison_path, "dataset_$i.csv"), df)
    println("Finished cross-method dataset $i"); flush(stdout)
    end
end
comparison = vcat([CSV.read(joinpath(comparison_path, "dataset_$i.csv"), DataFrame) for i in eachindex(datasets)]...)
CSV.write(joinpath(output_path, "reference_comparison.csv"), comparison)

## Standardised differences between pairs of methods, among fits that passed their diagnostics
passed = filter(:status => !=("failed"), comparison)
wide = unstack(passed, [:dataset, :parameters], :method, :mean)
wide_mcse = unstack(passed, [:dataset, :parameters], :method, :mcse)
for (a, b) in [("NUTS", "IRLS"), ("NUTS", "IS"), ("IRLS", "IS")]
    z = (wide[!, a] .- wide[!, b]) ./ sqrt.(wide_mcse[!, a] .^ 2 .+ wide_mcse[!, b] .^ 2)
    z = collect(skipmissing(z))
    println("$a vs $b: n = $(length(z)), max |z| = $(round(maximum(abs.(z)), digits = 2)), share |z| > 3 = $(round(mean(abs.(z) .> 3), digits = 3))")
end
end

if "coverage" in parts
## Prior-predictive coverage of NUTS for K=3 with censoring on [0, 10]
Random.seed!(2025)
n_sbc = 200
sbc_datasets = [simulate_counts(3, 10) for _ in 1:n_sbc]
X3 = one_hot_encode_parameters(3)

sbc = Vector{DataFrame}(undef, n_sbc)
Threads.@threads for i in 1:n_sbc
    counts, pars = sbc_datasets[i]
    chains, par_df, fit_df = run_reference_fit(counts, X3; censoring_lower = 0, censoring_upper = 10, stages = nuts_stages[2:3])
    draws = select(DataFrame(chains), r"intercept|betas|gammas")
    sbc[i] = DataFrame(
        dataset = i,
        parameters = param_names(3),
        true_values = pars,
        rank = [mean(d .< t) for (d, t) in zip(eachcol(draws), pars)],
        status = fit_df.status[1],
        stage = fit_df.stage[1],
        n_censored = sum(counts .== -1)
    )
    println("Finished coverage dataset $i"); flush(stdout)
end
sbc = vcat(sbc...)
CSV.write(joinpath(output_path, "reference_sbc_K3.csv"), sbc)

validated = filter(:status => !=("failed"), sbc)
println("NUTS status counts: ", combine(groupby(unique(sbc, :dataset), :status), nrow => :n))
for level in [0.5, 0.8, 0.95], parameter in ["alpha", "all"]
    ranks = parameter == "all" ? validated.rank : validated.rank[validated.parameters .== parameter]
    n_datasets = length(unique(validated.dataset))
    println("Coverage of $parameter at $level: $(round(mean(abs.(ranks .- 0.5) .<= level / 2), digits = 3)) (± $(round(2 * sqrt(level * (1 - level) / n_datasets), digits = 3)) for one parameter, n = $n_datasets)")
end
end
