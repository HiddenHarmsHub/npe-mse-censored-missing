using Flux
using Distributions # For defining priors and likelihoods
using ProgressMeter # To show a nice progress bar
using MLUtils
using Printf
using CUDA

CUDA.allowscalar(false)  # recommended

device = gpu_device()

function generate_parameters_nbe(K::Int, intercept_dist, beta_dist, gamma_dist)
    intercept = rand(intercept_dist)
    betas = rand(beta_dist, K)
    gammas = rand(gamma_dist, binomial(K, 2))
    par_estimates = vcat(
        intercept,
        betas,
        gammas
    )
    return par_estimates
end

function generate_data_nbe(params, m)
    K = Int(-0.5 + (sqrt(8 * length(params) - 7) / 2))  # Solve for K given length of params
    intercept = params[1]
    betas = params[2:1+K]
    gammas = params[2+K:end]

    Z = zeros(K + binomial(K, 2), m)

    for i in 1:m
        main_counts = [
            rand(Poisson(exp(intercept + betas[i]))) for i in 1:K
        ]
        
        pair_counts = [
            rand(Poisson(exp(intercept + betas[i] + betas[j] + gammas[binomial(K, 2) - binomial(K - i + 1, 2) + (j - i)]))) for i in 1:K-1 for j in i+1:K
        ]
        
        Z[:, i] = vcat(main_counts, pair_counts)

    end
    return log.(Z .+ 1)  # Log-transform the counts
end

K = 5
intercept_dist = Uniform(1, 10)
beta_dist = Normal(0, 4)
gamma_dist = Normal(0, 1/5)

sample_nbe(n_reps) = hcat([generate_parameters_nbe(K, intercept_dist, beta_dist, gamma_dist) for _ in 1:n_reps]...)
simulate_nbe(θ, m) = [generate_data_nbe(params, m) for params in eachcol(θ)]


function generate_simulation(n_samples_per_sim::Int)
    # 1. Simulate parameters from the prior
    θ = sample_nbe(1)[:, 1]  # Get a single parameter set

    # 2. Simulate data from the model (likelihood)
    data = simulate_nbe(θ, n_samples_per_sim)[1]  # Get the simulated data

    # Return the data and the parameters that generated it
    return (Float32.(data), Float32.(θ))
end


function generate_batch(n_simulations::Int, n_replicates_per_param::Int)
    # First, generate one simulation to determine dimensions
    sample_data, sample_params = generate_simulation(n_replicates_per_param)
    
    data_dim = size(sample_data, 1)  # Number of features per replicate (N)
    param_dim = length(sample_params)  # Number of parameters (M)
    
    # Pre-allocate arrays - we'll have multiple replicates per parameter set
    total_replicates = n_simulations * n_replicates_per_param
    all_data = zeros(Float32, data_dim, total_replicates)  # (N, total_replicates)
    all_params = zeros(Float32, param_dim, total_replicates)  # (M, total_replicates)
    
    replicate_idx = 1
    for sim in 1:n_simulations
        # Generate one parameter set
        data_matrix, params = generate_simulation(n_replicates_per_param)
        
        # Each column of data_matrix is one replicate
        for rep in 1:n_replicates_per_param
            all_data[:, replicate_idx] = data_matrix[:, rep]
            all_params[:, replicate_idx] = params  # Same parameters for all replicates
            replicate_idx += 1
        end
    end
    
    return all_data, all_params
end

# Function to prepare data for neural network input
function prepare_data_for_training(data_2d, params_2d)
    # Data is already in the correct format: (N, total_replicates)
    # Each column is one replicate of size N
    
    return data_2d, params_2d
end

# --- Simulation settings ---
n_simulations_per_epoch = 10_000_000  # Number of different parameter sets per epoch
n_replicates_per_param = 1     # Number of data replicates per parameter set

# Get dimensions from sample data
sample_data, sample_params = generate_simulation(1)  # Generate single replicate
input_dim = size(sample_data, 1)   # N - size of each replicate (15)
output_dim = length(sample_params) # M - number of parameters

println("Input dimension (N): $input_dim")
println("Output dimension (M): $output_dim")
println("Total replicates per epoch: $(n_simulations_per_epoch * n_replicates_per_param)")

# Network maps from single replicate (N) to parameters (M)
estimator_nn = Chain(
    Dense(input_dim, 256, relu),
    Dense(256, 256, relu),
    Dense(256, 128, relu),
    Dense(128, output_dim) # Output all NBE parameters
)

# MAE loss function
loss(model, x, y) = Flux.mae(model(x), y)
opt = Flux.setup(Adam(), estimator_nn)

batch_size = 64  # Reduced due to higher dimensionality

# --- The Training Loop with Fresh Data Each Epoch ---
println("Starting training with fresh data each epoch...")
epochs = 2

for epoch in 1:epochs
    epoch_start_time = time()
    
    # Generate fresh training data for this epoch
    println("Generating fresh data for epoch $epoch...")
    epoch_data, epoch_params = generate_batch(n_simulations_per_epoch, n_replicates_per_param)
    
    # Prepare data for training (already in correct format)
    epoch_data, epoch_params = prepare_data_for_training(epoch_data, epoch_params)
    
    # Create data loader for this epoch
    train_loader = DataLoader((epoch_data, epoch_params), batchsize=batch_size, shuffle=true)
    
    # Train for one full pass over this epoch's data
    Flux.train!(loss, estimator_nn, train_loader, opt)

    # Calculate and report the loss on this epoch's training set
    current_loss = loss(estimator_nn, epoch_data, epoch_params)
    
    epoch_time = time() - epoch_start_time
    @printf("Epoch %d: Training MAE = %.4f, Time = %.2f seconds\n", epoch, current_loss, epoch_time)
end


# --- Test the estimator ---

# 1. Generate a new, unseen dataset with known parameters  
test_data, true_params = generate_simulation(1)  # Single replicate
test_replicate = test_data[:, 1]  # Extract the single replicate vector

# 2. Use the trained network to predict the parameters
predicted_params = estimator_nn(test_replicate)

# 3. Compare the prediction to the true values
@printf("\n--- Test Results ---\n")
for i in eachindex(true_params)
    @printf(
        "Param %d - True: %.4f, Predicted: %.4f, Error: %.4f\n", 
        i, true_params[i], predicted_params[i], abs(true_params[i] - predicted_params[i])
    )
end

# Calculate overall MAE
test_mae = mean(abs.(true_params .- predicted_params))
@printf("\nOverall Test MAE: %.4f\n", test_mae)


n_test = 1000
test_data, test_params = generate_batch(n_test, 1)  # Single replicate per parameter set
test_data, test_params = prepare_data_for_training(test_data, test_params)
predicted_test_params = estimator_nn(test_data)
test_errors = abs.(test_params .- predicted_test_params)
test_mae = mean(abs.(test_params .- predicted_test_params))



using NeuralEstimators
n_data = K + binomial(K, 2)
n_pars = 1 + n_data
w = 128  # width of each hidden layer 

# Inner and outer networks
ψ = Chain(Dense(n_data, w, relu), Dense(w, 64, relu))
ϕ = Chain(Dense(64, w, relu), Dense(w, n_pars))

# Combine into a DeepSet
network = DeepSet(ψ, ϕ)

estimator = PointEstimator(network)

m = 1
estimator = train(
    estimator, 
    sample_nbe, 
    simulate_nbe, 
    m = m,
    K = 100_000
)

NBE_est = [NeuralEstimators.estimate(estimator, reshape(test_data[:, i], :, 1)) for i in 1:size(test_data, 2)] 

NBE_mae = abs.(test_params .- hcat(NBE_est...))

NBE_mae

mean(NBE_mae, dims=2)
mean(test_errors, dims=2)