using BSON, NeuralEstimators

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

function load_nbe(K, censoring_lower, censoring_upper, models_path, ci = false)
    width = 256
    n_hidden = 3
    train_size = 10000
    m = 1
    if ci
        mdl_str = "model_ci_$(K)_$(width)_$(n_hidden)_$(train_size)_$(censoring_lower)_$(censoring_upper)_$m.bson"
        model = BSON.load(joinpath(models_path, mdl_str))
        return model[:ci_estimator]
    else
        mdl_str = "model_$(K)_$(width)_$(n_hidden)_$(train_size)_$(censoring_lower)_$(censoring_upper)_$m.bson"
        model = BSON.load(joinpath(models_path, mdl_str))
        return model[:estimator]
    end
end

nbe_models_path = joinpath("output", "models_nbe");

## Load the NBE model and CI using user supplied arguments
nbe_estimator = load_nbe(K, censoring_lower, censoring_upper, nbe_models_path, false);
nbe_estimator_ci = load_nbe(K, censoring_lower, censoring_upper, nbe_models_path, true);

## Perform inference, for CI first column is 2.5% and second column is 97.5%
nbe_median_estimates = nbe_estimator(input_data)
nbe_estimates_ci = reshape(nbe_estimator_ci(input_data), (16, 2))



###############
# NPE Example #
###############

function load_npe(K, censoring_lower, censoring_upper, models_path; encoding_dim = 128)
    width = 256
    n_hidden = 3
    train_size = 10000
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