## In this script we asses the quality of our estimators on fixed test sets
using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")

using Random

nbe_models_path = joinpath("output", "models_nbe")
npe_models_path = joinpath("output", "models_npe")

## First generate fixed test sets for each list size
test_size = 10_000
test_path = joinpath("output", "test_data")
mkpath(test_path)
list_sizes = [3, 4, 5, 6, 10, 15]
Random.seed!(42)  # For reproducibility
for K in list_sizes
    outfile = joinpath(test_path, "test_data_$(K).bson")
    if isfile(outfile)
        println("Test data for K=$K already exists at $outfile; skipping.")
        continue
    end

    params = [sample_parameters(K) for _ in 1:test_size]
    Z_test = [simulate_data(params[i], 1, censoring_upper=0) for i in 1:test_size]

    # Concatenate into matrices
    params = hcat(params...)
    Z_test = hcat(Z_test...)

    BSON.@save outfile Z_test params
    println("Generated test data for K=$K")
end

## Also save K=5 to csv for comaprison with MCMC
test_data, test_pars = load_test_data(test_path, 5, 0, 10)
CSV.write(joinpath("output", "test_data_K5.csv"), DataFrame(test_data, :auto))


@everywhere function model_intecept_summary(model_file, nbe_models_path, npe_models_path, test_path)
    n_lists, width, n_hidden, train_size, censoring_lower, censoring_upper, m = parse.(Int, [m.match for m in eachmatch(r"\d+", model_file)])

    test_data, test_pars = load_test_data(test_path, n_lists, censoring_lower, censoring_upper)
    n_pars = size(test_pars, 1)

    model_nbe = load_model_nbe(model_file, nbe_models_path)
    model_nbe_cis = load_model_nbe(replace(model_file, "model_" => "model_ci_"), nbe_models_path)

    estimated_pars_nbe  = model_nbe(test_data)
    estimated_nbe_cis = model_nbe_cis(test_data)

    model_npe = load_model_npe(n_lists, width, n_hidden, train_size, censoring_lower, censoring_upper, m, npe_models_path)
    estimated_intercept_npe = map(eachcol(test_data)) do x
        posteriormedian(model_npe, reshape(x, :, 1))[1]
    end

    npe_estimates = map(eachcol(test_data)) do x
        posteriorquantile(model_npe, reshape(x, :, 1), [0.025, 0.5, 0.975])[1, :]
    end |> x -> hcat(x...)


    APE_NBE = abs.((estimated_pars_nbe .- test_pars) ./ test_pars)
    APE_NPE = abs.((npe_estimates[2, :] .- test_pars[1, :]) ./ test_pars[1, :])
    return DataFrame(
        dataset = 1:size(test_pars, 2),
        n_lists=n_lists, 
        width=width, 
        n_hidden=n_hidden, 
        train_size=train_size, 
        censoring_upper=censoring_upper,
        parameter = "intercept",
        intercept_truth = test_pars[1, :],
        intercept_NBE = estimated_pars_nbe[1, :],
        intercept_NBE_lower_ci = estimated_nbe_cis[1, :],
        intercept_NBE_upper_ci = estimated_nbe_cis[n_pars + 1, :],
        intercept_NPE = npe_estimates[2, :],
        intercept_NPE_lower_ci = npe_estimates[1, :],
        intercept_NPE_upper_ci = npe_estimates[3, :],
        APE_NBE = vec(APE_NBE[1, :]),
        APE_NPE = vec(APE_NPE)
    )
end

model_files = filter(x -> !occursin("ci", x), readdir(nbe_models_path))

intercept_summaries = pmap(
    model_file -> begin
        wid = myid()
        println("Worker $wid evaluating model: $model_file")
        model_intecept_summary(model_file, nbe_models_path, npe_models_path, test_path)
    end,
    model_files
)

CSV.write(
    joinpath("output", "intercept_estimate_comparison.csv"), 
    vcat(intercept_summaries...)
)
