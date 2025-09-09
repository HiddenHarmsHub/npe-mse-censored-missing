#install.packages("neuralnet")
#install.packages("dplyr")
library(neuralnet)
library(dplyr)

# Simulate abilities and outcomes
set.seed(123)
n_items <- 10
n_matches <- 2000
theta <- runif(n_items, 0, 5)

simulate_bt <- function(n_matches, theta) {
  outcomes <- data.frame()
  for (i in 1:n_matches) {
    item1 <- sample(1:length(theta), 1)
    item2 <- sample(setdiff(1:length(theta), item1), 1)
    prob <- exp(theta[item1]) / (exp(theta[item1]) + exp(theta[item2]))
    result <- ifelse(runif(1) < prob, 1, 0)
    outcomes <- rbind(outcomes, data.frame(item1, item2, result))
  }
  return(outcomes)
}

outcomes <- simulate_bt(n_matches, theta)


# One-hot encode item1 and item2
one_hot <- function(index, n) {
  vec <- rep(0, n)
  vec[index] <- 1
  return(vec)
}

# Create one-hot encoded matrix
X <- do.call(rbind, lapply(1:nrow(outcomes), function(i) {
  c(one_hot(outcomes$item1[i], n_items), one_hot(outcomes$item2[i], n_items))
}))

# Assign column names
colnames(X) <- c(paste0("i1_", 1:n_items), paste0("i2_", 1:n_items))

# Combine with result column
data_nn <- as.data.frame(cbind(X, result = outcomes$result))

# Now build the formula
input_vars <- paste(colnames(X), collapse = " + ")
formula <- as.formula(paste("result ~", input_vars))

# Train the neural network
nn_model <- neuralnet(formula, data = data_nn, hidden = c(10, 5), linear.output = FALSE)


# Predict item 1 vs item 3
new_match <- as.data.frame(t(c(one_hot(1, n_items), one_hot(3, n_items))))
colnames(new_match) <- colnames(X)

# Use predict instead of compute
prediction <- predict(nn_model, newdata = new_match)
cat("Predicted probability that item 1 beats item 3:", prediction, "\n")
exp(theta[1])/(exp(theta[1]) + exp(theta[3]))








