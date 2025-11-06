using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")

npe_models = readdir(joinpath("output", "models_npe"))

output_df_list = pmap(
    npe_model -> begin
        n_lists = parse(Int, split(npe_model, "_")[2])
        width = parse(Int, split(npe_model, "_")[3])
        n_hidden = parse(Int, split(npe_model, "_")[4])
        encoding_dim = parse(Int, split(npe_model, "_")[5])
        train_size = parse(Int, split(npe_model, "_")[6])
        censoring_lower = parse(Int, split(npe_model, "_")[7])
        censoring_upper = parse(Int, split(npe_model, "_")[8])
        m = parse(Int, split(split(npe_model, "_")[9], ".")[1])

        test_data, test_pars = load_test_data(
            joinpath("output", "test_data"), 
            n_lists, 
            censoring_lower, 
            censoring_upper
        )

        NPE_model = BSON.load(joinpath("output", "models_npe", npe_model))[:estimator]
        intercept_NPE = map(eachcol(test_data)) do x
            posteriormedian(NPE_model, reshape(x, :, 1))[1]
        end

        return DataFrame(
            dataset = 1:size(test_pars, 2),
            n_lists = n_lists,
            width = width,
            n_hidden = n_hidden,
            encoding_dim = encoding_dim,
            train_size = train_size,
            censoring_lower = censoring_lower,
            censoring_upper = censoring_upper,
            m = m,
            intercept_NPE = intercept_NPE,
            intercept_true = test_pars[1, :]
        )
    end,
    npe_models
)

output_df = vcat(output_df_list...)
CSV.write("output/npe_vs_nbe_intercept_estimates.csv", output_df)


