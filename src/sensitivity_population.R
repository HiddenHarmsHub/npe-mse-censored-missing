library("NeuralEstimators")
library("JuliaConnectoR")
library(magrittr)
library(dplyr)
juliaEval('using NeuralEstimators, Flux') 



# Sampler Code ------------------------------------------------------------


#Generate a matrix of K samples from the prior distribution with N lists
generate_theta <- function(N, K) {
  matrix_data <- matrix(nrow = K, ncol = 1 + N + choose(N, 2))
  
  # Intercept Parameter (mu)
  matrix_data[, 1] <- runif(K, 5, 10)
  
  # Main effects
  matrix_data[, 2:(N+1)] <- replicate(N, rnorm(K, 0, 5))
  
  # Pairwise effects
  matrix_data[, (N+2):(choose(N, 2) + N + 1)] <- replicate(choose(N, 2), rnorm(K, 0, 1/5))
  
  return(t(matrix_data))
}




one_hot_combinations <- function(n_lists) { 
  # Generate all binary combinations
  combos <- expand.grid(rep(list(c(0, 1)), n_lists))
  
  # Filter to keep only rows with exactly 0, 1 or 2 ones
  combos <- combos[rowSums(combos) %in% c(1, 2), ]
  
  # Clean up row and column names
  rownames(combos) <- NULL
  colnames(combos) <- paste0("List_", 1:n_lists)
  
  return(combos)
}


# Marginal simulation from the statistical model
# N: number of lists
# M: Population Size
# params: a vector of parameters drawn from the prior
# m: number of conditionally independent replicates for each parameter vector
simulate_mse_poisson_compact <- function(N, params, m, threshold) {
  combos <- expand.grid(rep(list(c(0, 1)), N))
  combos <- combos[rowSums(combos) %in% c(1, 2), ]
  combos <- rbind(rep(0, N), combos)  # add row of all zeros
  colnames(combos) <- paste0("List_", 1:N)
  
  mu <- params[1]
  alphas <- params[2:(1 + N)]
  beta_index <- 1 + N
  betas <- params[(beta_index + 1):length(params)]
  
  interaction_matrix <- combn(N, 2)
  
  lambda_log <- numeric(nrow(combos))
  
  for (i in 1:nrow(combos)) {
    included_lists <- which(combos[i, ] == 1)
    log_lambda <- mu + sum(alphas[included_lists])
    
    if (length(included_lists) == 2) {
      for (j in 1:ncol(interaction_matrix)) {
        if (all(sort(included_lists) == interaction_matrix[, j])) {
          log_lambda <- log_lambda + betas[j]
          break
        }
      }
    }
    
    lambda_log[i] <- log_lambda
  }
  
  lambda <- exp(lambda_log)
  
  # Replicate each Poisson draw m times
  simulated <- replicate(m, log1p(rpois(length(lambda), lambda)[-1]))
  mask <- ifelse(simulated < log1p(threshold), 1, 0)
  simulated <- simulated*(1-mask)
  
  return(rbind(simulated, mask))
}


# Train Neural Network ----------------------------------------------------

# Initialise the estimator
estimator <- juliaEval(' 
  d = 15     # dimension of each replicate 
  w = 64   # number of neurons in each hidden layer
  
  #Layer to ensure valid estimates
  final_layer = Parallel(
    vcat,
    Dense(w, 1, softplus),
    Dense(w, d, identity)
  )

  psi = Chain(Dense(d * 2, w, relu), Dense(w, w, relu), Dense(w, w, relu))
  phi = Chain(Dense(w, w, relu), Dense(w, w, relu), final_layer)
  deepset = DeepSet(psi, phi)
  estimator = PointEstimator(deepset)
')



# Construct training and validation data sets
K <- 5000
m <- 250
N <- 5
threshold <- 10
theta_train <- generate_theta(N, K)
theta_val   <- generate_theta(N, K/10)

#Simulate K lists using the training parameters
Z_train <- apply(theta_train, 2, function(p) {
  simulate_mse_poisson_compact(N, p, m, threshold)
}, simplify = FALSE)

#Simulate K lists using the validation parameters
Z_val <- apply(theta_val, 2, function(p) {
  simulate_mse_poisson_compact(N, p, m, threshold)
}, simplify = FALSE)


#Set as doubles
Z_val <- lapply(Z_val, function(mat) {
  storage.mode(mat) <- "double"
  as.matrix(mat)
})

Z_train <- lapply(Z_train, function(mat) {
  storage.mode(mat) <- "double"
  as.matrix(mat)
})


# Train the estimator
estimator <- train(
  estimator,
  theta_train = theta_train,
  theta_val   = theta_val,
  Z_train = Z_train,
  Z_val   = Z_val, 
  epochs = 100
)




# Test Method ------------------------------------------------------------
theta_test <- generate_theta(N, 1000)

Z_test <- apply(theta_test, 2, function(p) {
  simulate_mse_poisson_compact(N, p, 1000, threshold)
}, simplify = FALSE)

Z_test <- lapply(Z_test, function(mat) {
  storage.mode(mat) <- "double"
  as.matrix(mat)
})


assessment <- assess(estimator, theta_test, Z_test)
plotestimates(assessment)


log_alpha_estimates <- assessment[[1]] %>% filter(parameter=="θ1") %>% 
  select(estimate,truth) 
alpha_estimates <- exp(log_alpha_estimates)
error <- 100* abs(alpha_estimates$estimate - alpha_estimates$truth) / alpha_estimates$truth


p <-  ggplot(alpha_estimates, aes(x = truth, y = log(error))) +
  geom_point(alpha = 0.6, color = "#2c7bb6", size = 2) +       # points
  labs(
    x = expression("True exp" * alpha),
    y = "Log MAPE",
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )
 
 
 ggsave("log_ape_vs_alpha.png", plot = p, width = 6, height = 4, dpi = 300)

