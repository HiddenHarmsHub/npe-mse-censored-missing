## In this script we compare MCMC and NPE posterior samples

using Pkg; Pkg.activate(".")

include("mse_functions.jl")

npe_samples_path = joinpath("output", "npe_samples")
mcmc_samples_path = joinpath("output", "mcmc_samples")
test_data_path = joinpath("output", "test_data")
test_data, test_pars = load_test_data(test_data_path, 5, 0, 10)
levels = collect(0.99:-0.01:0.01)

coverage_tables = Folds.map(
    i -> begin
        true_params = test_pars[:, i]
        npe_samples = DataFrame(CSV.File(joinpath(npe_samples_path, "npe_test_results_$(i).csv")))
        mcmc_samples = DataFrame(CSV.File(joinpath(mcmc_samples_path, "mcmc_test_results_$(i).csv")))

        npe_alpha = npe_samples[:, 1]
        mcmc_alpha = mcmc_samples[:, 1]
        true_alpha = true_params[1]

        coverage = vcat(
            coverage_table(npe_alpha, true_alpha, levels, "NPE"),
            coverage_table(mcmc_alpha, true_alpha, levels, "MCMC"),
        )
        coverage.dataset .= i
        return coverage
    end,
    1:size(test_data, 2)
)

CSV.write(
    joinpath("output", "coverage_comparison_npe_mcmc.csv"),
    vcat(coverage_tables...),
)

rank_list = Folds.map(
    i -> begin
        true_params = test_pars[:, i]
        npe_file = joinpath(npe_samples_path, "npe_test_results_$(i).csv")
        mcmc_file = joinpath(mcmc_samples_path, "mcmc_test_results_$(i).csv")
        if !isfile(npe_file) || !isfile(mcmc_file)
            return (
                NPE_rank = missing,
                MCMC_rank = missing
            )
        end
        npe_samples = DataFrame(CSV.File(npe_file))
        mcmc_samples = DataFrame(CSV.File(mcmc_file))

        npe_alpha = npe_samples[:, 1]
        mcmc_alpha = mcmc_samples[:, 1]
        true_alpha = true_params[1]

        return (
            dataset = i,
            NPE_rank = mean(npe_alpha .< true_alpha),
            MCMC_rank = mean(mcmc_alpha .< true_alpha)
        )
    end,
    1:size(test_data, 2)
)

CSV.write(
    joinpath("output", "rank_comparison_npe_mcmc.csv"),
    DataFrame(rank_list)
)