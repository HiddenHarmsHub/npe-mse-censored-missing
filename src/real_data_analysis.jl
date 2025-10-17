include("mse_functions.jl")


## First analyse the Silverman data
silverman_data = Float32.(log.(load_silverman_data() .+ 1))

model1 = load_model(
    n_lists = 5, 
    width = 128, 
    censoring_threshold = 0, 
    n_hidden = 2, 
    train_size = 10000
)

silverman_par_estimates = model1(silverman_data)


## Now analyse King data
king_data = load_king_data()

model_king = load_model(
    n_lists = 4, 
    width = 128, 
    censoring_threshold = 4, 
    n_hidden = 3, 
    train_size = 100000
)

king_par_estimates = model_king(king_data)