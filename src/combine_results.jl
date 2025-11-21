using DataFrames, CSV


## combine intercept files
intercept_files = readdir(joinpath("output", "intercept_estimates"))

CSV.write(
    joinpath("output", "intercept_estimate_comparison.csv"),
    vcat(map(
        file -> CSV.read(joinpath("output", "intercept_estimates", file), DataFrame),
        intercept_files
    )...)
)

## combine mcmc files
mcmc_summary_files = readdir(joinpath("output", "mcmc_summary"))

mcmc_df = map(mcmc_summary_files) do file
    df = CSV.read(joinpath("output", "mcmc_summary", file), DataFrame)
    matches = match(r"\d+", file)
    numbers = matches !== nothing ? parse(Int, matches.match) : nothing
    df.dataset .= numbers
    return select(df,
        :dataset,
        :parameters,
        :true_values,
        :estimated_medians => :median_mcmc,
        :lower_95ci => :lower_ci_mcmc,
        :upper_95ci => :upper_ci_mcmc,
        :rhat
    )
end

CSV.write(
    joinpath("output", "mcmc_summary.csv"),
    vcat(mcmc_df...)
)

## combine npe files
npe_summary_files = readdir(joinpath("output", "npe_summary"))
npe_df = map(npe_summary_files) do file
    df = CSV.read(joinpath("output", "npe_summary", file), DataFrame)
    matches = match(r"\d+", file)
    numbers = matches !== nothing ? parse(Int, matches.match) : nothing
    df.dataset .= numbers
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