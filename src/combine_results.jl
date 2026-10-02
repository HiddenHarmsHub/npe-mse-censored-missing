using DataFrames, CSV, Statistics


## combine intercept files
intercept_files = readdir(joinpath("output", "intercept_estimates"))

CSV.write(
    joinpath("output", "intercept_estimate_comparison.csv"),
    vcat(map(
        file -> CSV.read(joinpath("output", "intercept_estimates", file), DataFrame),
        intercept_files
    )...)
)

dataset_number(file) = parse(Int, match(r"\d+", file).match)

## Turing names (intercept, betas[i], gammas[j]) to the names used for NPE (alpha, beta_i, gamma_ij)
function sampler_param_names(n_lists)
    pairs = [(i, j) for i in 1:(n_lists - 1) for j in (i + 1):n_lists]
    return Dict(vcat(
        "intercept" => "alpha",
        ["betas[$i]" => "beta_$i" for i in 1:n_lists],
        ["gammas[$k]" => "gamma_$(i)$(j)" for (k, (i, j)) in enumerate(pairs)]
    ))
end
name_map = sampler_param_names(5)

## combine mcmc and irls files
function combine_sampler(method)
    summary_path = joinpath("output", "$(method)_summary")
    summary_df = map(readdir(summary_path)) do file
        df = CSV.read(joinpath(summary_path, file), DataFrame)
        df.dataset .= dataset_number(file)
        df.parameters = [name_map[p] for p in df.parameters]
        return select(df,
            :dataset,
            :parameters,
            :true_values,
            :estimated_means => "mean_$method",
            :estimated_medians => "median_$method",
            :lower_95ci => "lower_ci_$method",
            :upper_95ci => "upper_ci_$method",
            :rhat,
            :ess_bulk,
            :ess_tail,
            :mcse
        )
    end
    CSV.write(joinpath("output", "$(method)_summary.csv"), vcat(summary_df...))

    diagnostics_path = joinpath("output", "$(method)_diagnostics")
    diagnostics_df = [CSV.read(joinpath(diagnostics_path, file), DataFrame) for file in readdir(diagnostics_path)]
    CSV.write(joinpath("output", "$(method)_diagnostics.csv"), vcat(diagnostics_df...))
end

combine_sampler("mcmc")
combine_sampler("irls")

## importance-sampling reference for each benchmark dataset
is_df = map(readdir(joinpath("output", "is_summary"))) do file
    df = CSV.read(joinpath("output", "is_summary", file), DataFrame)
    df.dataset .= dataset_number(file)
    df.parameters = [name_map[p] for p in df.parameters]
    return df
end
CSV.write(joinpath("output", "is_summary.csv"), vcat(is_df...))

## combine npe files
npe_summary_files = readdir(joinpath("output", "npe_summary"))
npe_df = map(npe_summary_files) do file
    df = CSV.read(joinpath("output", "npe_summary", file), DataFrame)
    df.dataset .= dataset_number(file)
    return select(df,
        :dataset,
        :parameters,
        :true_values,
        :estimated_medians => :median_npe,
        :lower_95ci => :lower_ci_npe,
        :upper_95ci => :upper_ci_npe
    )
end

CSV.write(
    joinpath("output", "npe_summary.csv"),
    vcat(npe_df...)
)

## Fractional rank of the truth among the posterior draws for each benchmark dataset.
## A central credible interval at level L contains the truth iff (1 - L) / 2 <= rank <= (1 + L) / 2.
benchmark_datasets = CSV.read(joinpath("output", "benchmark_datasets.csv"), DataFrame).dataset

function sbc_ranks(method, samples_dir, samples_prefix, summary_dir, summary_prefix)
    rows = DataFrame[]
    for dataset in benchmark_datasets
        samples_file = joinpath("output", samples_dir, "$(samples_prefix)_$dataset.csv")
        isfile(samples_file) || continue
        samples = CSV.read(samples_file, DataFrame)
        summary = CSV.read(joinpath("output", summary_dir, "$(summary_prefix)_$dataset.csv"), DataFrame)
        rename!(samples, [n => get(name_map, n, n) for n in names(samples)])
        parameters = [get(name_map, p, p) for p in summary.parameters]
        push!(rows, DataFrame(
            method = method,
            dataset = dataset,
            parameters = parameters,
            true_values = summary.true_values,
            rank = [mean(samples[!, p] .< t) for (p, t) in zip(parameters, summary.true_values)]
        ))
    end
    return vcat(rows...)
end

CSV.write(
    joinpath("output", "sbc_ranks.csv"),
    vcat(
        sbc_ranks("MCMC", "mcmc_samples", "mcmc_test_results", "mcmc_summary", "mcmc_test_summary"),
        sbc_ranks("IRLS", "irls_samples", "irls_test_results", "irls_summary", "irls_test_summary"),
        sbc_ranks("NPE", "npe_samples", "npe_test_results", "npe_summary", "npe_test_summary")
    )
)
