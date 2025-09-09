freq <- c(NA, 54, 463, 907, 695, 316, 57, 15, 19, 3, 0 , 0, 56, 19, 1, 3, 69, 10, 31, 8, 6, 1)
pair_index <- combn(N, 2)

# Create binary table: each row corresponds to one combination
binary_table <- t(apply(pair_index, 2, function(pair) {
  row <- rep(0, N)
  row[pair] <- 1
  return(row)
}))
binary_table <- rbind(diag(N), binary_table)
binary_table <- rbind(rep(0, N), binary_table)
data <- as.data.frame(cbind(binary_table, freq))
result <- closedpMS.t(data, dfreq=TRUE, h='Poisson', stopiflong = FALSE, maxorder = 2)
print(result)


