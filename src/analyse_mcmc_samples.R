pacman::p_load(tidyverse)


mcmc_files <- list.files(
    path = file.path("output", "mcmc_summary"), 
    pattern = "*.csv", 
    full.names = TRUE
)

mcmc_df <- lapply(mcmc_files, function(mcmc_file) {
    mcmc_df <- read_csv(mcmc_file, show_col_types = FALSE)
    mcmc_df$dataset <- parse_number(mcmc_file)
    return(mcmc_df[1,])
}) %>% 
    bind_rows() %>% 
    select(
        dataset, 
        true_intercept = true_values, 
        median_mcmc = estimated_medians,
        rhat
    )


worst_rhats <- mcmc_df %>%
    arrange(desc(rhat))


samples_path <- file.path("output", "mcmc_samples")
dataset_id <- worst_rhats$dataset[1]

plot_simulation_intercept <- function(dataset_id, samples_path, mcmc_df) {
    mcmc_samples <- read_csv(
        file.path(
            samples_path, 
            paste0("mcmc_test_results_", dataset_id, ".csv")
        )
    )

    plot(
        mcmc_samples$intercept, 
        type = "l",
        main = paste("MCMC Intercept Samples for Dataset", dataset_id),
        xlab = "Iteration",
        ylab = "Intercept Value"
    )
    abline(
        h = mcmc_df$true_intercept[mcmc_df$dataset == dataset_id], 
        col = "red",
        lwd = 3
    )
}

plot_simulation_intercept(2, samples_path, mcmc_df)

plot_simulation_intercept(worst_rhats$dataset[200], samples_path, mcmc_df)
plot_simulation_intercept(worst_rhats$dataset[203], samples_path, mcmc_df)
plot_simulation_intercept(worst_rhats$dataset[205], samples_path, mcmc_df)
plot_simulation_intercept(worst_rhats$dataset[1000], samples_path, mcmc_df)

# open a new plotting device with less height (in inches)
dev.new(width = 7, height = 3.5)




