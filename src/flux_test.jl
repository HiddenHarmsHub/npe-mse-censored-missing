using Flux
using Distributions # For defining priors and likelihoods
using ProgressMeter # To show a nice progress bar
using MLUtils
using Printf

# Define the generative model
μ_prior = Normal(0, 5)
σ_prior = Gamma(2, 2)

function generate_simulation(n_samples_per_sim::Int)
    # 1. Simulate parameters from the prior
    μ = rand(μ_prior)
    σ = rand(σ_prior)

    # 2. Simulate data from the model (likelihood)
    data_dist = Normal(μ, σ)
    data = rand(data_dist, n_samples_per_sim)

    # Return the data and the parameters that generated it
    return (Float32.(data), Float32.([μ, σ]))
end

# Function to generate a batch of simulations
function generate_batch(n_simulations::Int, n_samples::Int)
    all_data = zeros(Float32, n_samples, n_simulations)
    all_params = zeros(Float32, 2, n_simulations)
    
    for i in 1:n_simulations
        data, params = generate_simulation(n_samples)
        all_data[:, i] = data
        all_params[:, i] = params
    end
    
    return all_data, all_params
end

# --- Simulation settings ---
n_simulations_per_epoch = 10000 # Smaller batches per epoch
n_samples = 50       # How many data points in each simulated dataset 'x'

# The network takes a dataset of size `n_samples` and outputs 2 parameter estimates
estimator_nn = Chain(
    Dense(n_samples, 128, relu),
    Dense(128, 128, relu),
    Dense(128, 2) # Output layer with 2 neurons for μ and σ
)

# MAE loss function
loss(model, x, y) = Flux.mae(model(x), y)
opt = Flux.setup(Adam(), estimator_nn)

batch_size = 128

# --- The Training Loop with Fresh Data Each Epoch ---
println("Starting training with fresh data each epoch...")
epochs = 1000

for epoch in 1:epochs
    # Generate fresh training data for this epoch
    println("Generating fresh data for epoch $epoch...")
    epoch_data, epoch_params = generate_batch(n_simulations_per_epoch, n_samples)
    
    # Create data loader for this epoch
    train_loader = DataLoader((epoch_data, epoch_params), batchsize=batch_size, shuffle=true)
    
    # Train for one full pass over this epoch's data
    Flux.train!(loss, estimator_nn, train_loader, opt)

    # Calculate and report the loss on this epoch's training set
    current_loss = loss(estimator_nn, epoch_data, epoch_params)
    @printf("Epoch %d: Training MAE = %.4f\n", epoch, current_loss)
end

println("Training complete! 🎉")



# --- Test the estimator ---

# 1. Generate a new, unseen dataset with known parameters
μ_true = 2.5
σ_true = 1.2
observed_data = rand(Normal(μ_true, σ_true), n_samples)
observed_data = Float32.(observed_data) # Ensure correct type

# 2. Use the trained network to predict the parameters
predicted_params = estimator_nn(observed_data)

# 3. Compare the prediction to the true values
@printf("\n--- Test Results ---\n")
@printf("True μ: %.2f,  Predicted μ: %.2f\n", μ_true, predicted_params[1])
@printf("True σ: %.2f,  Predicted σ: %.2f\n", σ_true, predicted_params[2])