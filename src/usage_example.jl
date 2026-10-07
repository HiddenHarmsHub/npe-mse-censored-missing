using BSON, NeuralEstimators
include("nbe_layers.jl")  # output layer of the NBE models, needed to load them

## User supplied parameters
K = 5;
censoring_lower = 0;
censoring_upper = 10; 


function prepare_data(data::Vector{Int})
    ## first log transform positive counts
    out = [x > 0 ? log(x + 1) : x for x in data]

    if any(data .< 0)
        ## censoring present
        append!(out, 1 * (out .< 0))
    end

    return reshape(Float32.(out), (length(out), 1))
end


## First generate some random input data and set parameters
## Generate random input data the user supplies
user_inputted_data = vec(Int.(floor.(rand(Float64, K * 6 + 1, 1)*20)));
user_inputted_data[user_inputted_data .< censoring_upper] .= -1; ## simulate censoring with -1

input_data = prepare_data(user_inputted_data);


###############
# NBE Example #
###############

## The NBE gives the 2.5%, 50% and 97.5% posterior quantiles of every parameter, stacked in that order
function load_nbe(K, censoring_lower, censoring_upper, models_path)
    width = 256
    n_hidden = 3
    train_size = 200000
    m = 1
    mdl_str = "model_$(K)_$(width)_$(n_hidden)_$(train_size)_$(censoring_lower)_$(censoring_upper)_$m.bson"
    model = BSON.load(joinpath(models_path, mdl_str))
    return model[:estimator]
end

nbe_models_path = joinpath("output", "models_nbe");

## Load the NBE model using user supplied arguments
nbe_estimator = load_nbe(K, censoring_lower, censoring_upper, nbe_models_path);

## Perform inference: columns are the 2.5%, 50% and 97.5% quantiles, rows the parameters
nbe_quantiles = reshape(nbe_estimator(input_data), :, 3)
nbe_median_estimates = nbe_quantiles[:, 2]



###############
# NPE Example #
###############

function load_npe(K, censoring_lower, censoring_upper, models_path; encoding_dim = 128)
    width = 256
    n_hidden = 3
    train_size = 200000
    m = 1
    mdl_str = "model_$(K)_$(width)_$(n_hidden)_$(encoding_dim)_$(train_size)_$(censoring_lower)_$(censoring_upper)_$m.bson"
    model = BSON.load(joinpath(models_path, mdl_str))
    return model[:estimator]
end

npe_models_path = joinpath("output", "models_npe")

npe_estimator = load_npe(K, censoring_lower, censoring_upper, npe_models_path)
npe_samples = sampleposterior(npe_estimator, input_data, 1000)

## Possibly include a final rejection step based on prior constraints
bounded_sample(sample, lower, upper) = sample[:, (sample[1, :] .>= lower) .& (sample[1, :] .<= upper)]

output_npe_samples = bounded_sample(npe_samples, 1, 10)