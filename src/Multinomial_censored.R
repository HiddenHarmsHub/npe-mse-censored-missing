library("NeuralEstimators")
library("JuliaConnectoR")
juliaEval('using NeuralEstimators, Flux') 



# Sampler Code ------------------------------------------------------------


#Generate a matrix of K samples from the prior distirbution with N lists
generate_theta <- function(N, K) {
  matrix_data <- matrix(nrow = K, ncol = 1 + N + choose(N, 2))
  
  # Intercept Parameter (mu)
  matrix_data[, 1] <- runif(K, 0, 10)
  
  # Main effects
  matrix_data[, 2:(N+1)] <- replicate(N, runif(K, -2, 2))
  
  # Pairwise effects
  matrix_data[, (N+2):(choose(N, 2) + N + 1)] <- replicate(choose(N, 2), rnorm(K, 0, 1))
  
  return(t(matrix_data))
}



#Function to find the row of a given pair
find_pair_row <- function(x, matrix) {
  match_row <- which(matrix[,1] == x[1] & matrix[,2] == x[2])
  if (length(match_row) == 0) {
    return(NA)
  } else {
    return(match_row)
  }
}


#One hot design matrix
one_hot_combinations <- function(n_lists) {
  # Generate all binary combinations
  combos <- expand.grid(rep(list(c(0, 1)), n_lists))
  rownames(combos) <- NULL
  colnames(combos) <- paste0("List_", 1:n_lists)
  return(combos)
}


one_or_two_hot_combinations <- function(n_lists) {
  # Generate all binary combinations
  combos <- expand.grid(rep(list(c(0, 1)), n_lists))
  
  # Filter to keep only rows with exactly 1 or 2 ones
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
simulate_lists <- function(N, M, params, m){
  
  one_hot_matrix <- one_or_two_hot_combinations(N)
  num_rows <- nrow(one_hot_matrix)
  interaction_matrix <- combn(N, 2)
  number_combinations <- (N^2 + N + 2)/2
  
  
  lambda <- numeric(number_combinations)
  lambda[1] <- exp(params[1])
  for (i in 1:num_rows) {
    lists <- which(one_hot_matrix[i, ] == 1)
    
    # Start with intercept
    log_lambda <- params[1]
    
    # Add main effects
    log_lambda <- log_lambda + sum(params[1 + lists])
    
    # Add interaction if there are 2 lists
    if (length(lists) == 2) {
      for (j in 1:ncol(interaction_matrix)) {
        if (all(sort(lists) == interaction_matrix[, j])) {
          log_lambda <- log_lambda + params[1 + N + j]
          break
        }
      }
    }
    lambda[i+1] <- exp(log_lambda)
  }
  
  # Simulate count
  observations <- rmultinom(m, M, lambda)
  observations <- observations[-1, ]
  
  dimnames(observations) <- NULL
  return(observations)
}


# Train Neural Network ----------------------------------------------------

# Initialise the estimator
estimator <- juliaEval(' 
  d = 15     # dimension of each replicate 
  w = 32     # number of neurons in each hidden layer
  
  #Layer to ensure valid estimates
  final_layer = Parallel(
    vcat,
    Dense(w, 16, identity)
  )

  psi = Chain(Dense(d, w, relu), Dense(w, w, relu), Dense(w, w, relu))
  phi = Chain(Dense(w, w, relu), Dense(w, w, relu), final_layer)
  deepset = DeepSet(psi, phi)
  estimator = PointEstimator(deepset)
')



# Construct training and validation data sets
K <- 1000
m <- 250
N <- 5
theta_train <- generate_theta(N, K)
theta_val   <- generate_theta(N, K/10)

#Simulate K lists using the training parameters
Z_train <- apply(theta_train, 2, function(p) {
  simulate_lists(N = N, M = 1000, params = p, m = m)
}, simplify = FALSE)

#Simulate K lists using the validation parameters
Z_val <- apply(theta_val, 2, function(p) {
  simulate_lists(N = N, M = 1000, params = p, m = m)
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

#Censor small counts with NA
# Z_train_censored <- lapply(Z_train, function(mat) {
#   mat[mat < 20] <- NA
#   return(mat)
# })
# 
# Z_val_censored <- lapply(Z_val, function(mat) {
#   mat[mat < 20] <- NA
#   return(mat)
# })

#Censor small counts with -1
Z_train_censored <- lapply(Z_train, function(mat) {
  mat[mat < 20] <- 999
  return(mat)
})

Z_val_censored <- lapply(Z_val, function(mat) {
  mat[mat < 20] <- 999
  return(mat)
})




# Train the estimator
estimator <- train(
  estimator,
  theta_train = theta_train,
  theta_val   = theta_val,
  Z_train = Z_train_censored,
  Z_val   = Z_val_censored, 
  epochs = 50
)




# Test Method ------------------------------------------------------------
theta_test <- generate_theta(N, 1000)

Z_test <- apply(theta_test, 2, function(p) {
  simulate_lists(N = N, M = 1000, params = p, m = m)
}, simplify = FALSE)

Z_train <- lapply(Z_test, function(mat) {
  storage.mode(mat) <- "double"
  as.matrix(mat)
})

Z_test_censored <- lapply(Z_test, function(mat) {
  mat[mat < 20] <- 999
  return(mat)
})


assessment <- assess(estimator, theta_test, Z_test_censored, 
                     estimator_names = "NBE") 


# Some Diagnostics --------------------------------------------------------
plotestimates(assessment)

hist((assessment$estimates$estimate[assessment$estimates$parameter == "θ1"]),, breaks = 20, main = "Log Prevalence", xlab = expression(log(mu)))
abline(v = log(1000), col = 2, lwd =2)
plot(theta_test[1, ], (assessment$estimates$estimate[assessment$estimates$parameter == "θ1"]))
abline(0, 1)
