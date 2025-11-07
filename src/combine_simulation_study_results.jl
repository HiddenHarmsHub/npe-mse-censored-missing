using DataFrames, CSV

intercept_files = readdir(joinpath("output", "intercept_estimates"))

CSV.write(
    joinpath("output", "intercept_estimate_comparison.csv"),
    vcat(map(
        file -> CSV.read(joinpath("output", "intercept_estimates", file), DataFrame),
        intercept_files
    )...)
)