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



get_lambdas <- function(N, params) {
  combos <- expand.grid(rep(list(c(0, 1)), N))
  combos <- combos[rowSums(combos) %in% c(1, 2), ]
  combos <- rbind(rep(0, N), combos)  # add row of all zeros
  
  mu <- params[1]
  alphas <- params[2:(1 + N)]
  betas <- params[(1 + (1 + N)):length(params)]
  
  interaction_matrix <- combn(N, 2)
  
  lambda_log <- numeric(nrow(combos))
  
  for (i in 1:nrow(combos)) {
    included_lists <- which(combos[i, ] == 1)
    log_lambda <- mu + sum(alphas[included_lists])
    
    if (length(included_lists) == 2) {
      for (j in 1:ncol(interaction_matrix)) {
        if (all(sort(included_lists) == interaction_matrix[, j])) {
          log_lambda <- log_lambda + betas[j]
        }
      }
    }
    
    lambda_log[i] <- log_lambda
  }
  
  lambda <- exp(lambda_log)
  
  return(lambda)
}

# Marginal simulation from the statistical model
# N: number of lists
# M: Population Size
# params: a vector of parameters drawn from the prior
# m: number of conditionally independent replicates for each parameter vector
simulate_mse_poisson_compact <- function(lambda) {
  
  # Replicate each Poisson draw m times
  simulated <- rpois(length(lambda), lambda)[-1]

  
  return(simulated)
}

params <- numeric(1 + N + choose(N, 2))
params[1] <- runif(1, 5, 15)

# Main effects
params[2:(N+1)] <- replicate(N, runif(1, -params[1], 0))

# Pairwise effects
params[(N+2):(choose(N, 2) + N + 1)] <- replicate(choose(N, 2), runif(1, -2, 2))


lambdas <- get_lambdas(6, params)
test <- simulate_mse_poisson_compact(lambdas)
test

log_likelihood <- function(pars, data, N) {
  lambdas <- get_lambdas(N, pars)
  
  # Calculate the log-likelihood
  value <- -sum(dpois(data, lambdas[-1], log = TRUE))
  
  return(value)
}

lower_bounds <- c(8, rep(-10, 6), rep(-4, choose(N, 2)))
upper_bounds <- c(Inf, rep(1, 6), rep(4, choose(N, 2)))


result <- optim(
  par = lower_bounds,
  fn = log_likelihood,
  data = c(54, 463, 907, 695, 316, 57, 15, 19, 3, 0 , 0, 56, 19, 1, 3, 69, 10, 31, 8, 6, 1),
  N = 6,
  control = list(maxit = 50000),
  method = "L-BFGS-B",
  lower = lower_bounds, upper = upper_bounds,
)
result

plot(params, result$par)
abline(0, 1)
# plot(get_lambdas(6, params), (get_lambdas(6, result$par)))
# abline(0, 1)

# 
# bernard <- c(NA, 54, 463, 907, 695, 316, 57, 15, 19, 3, 0 , 0, 56, 19, 1, 3, 69, 10, 31, 8, 6, 1)
# 
