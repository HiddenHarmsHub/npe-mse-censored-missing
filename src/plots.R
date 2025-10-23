pacman::p_load(tidyverse)

ape_df <- read_csv(file.path("output", "intercept_ape_summary.csv"))

train_size_fixed <- 10000

## Width/number of neurons sensitivity plot
width_plot <- ape_df %>% 
    filter(
        n_lists == 5, 
        train_size == train_size_fixed,
        censoring_threshold == 10,
        n_hidden == 2
    ) %>% 
    ggplot(aes(x = as.factor(width), y = log(APE))) +
    geom_boxplot(fill = "#2c7bb6", alpha = 0.6) +
    theme_minimal(base_size = 14) +
    labs(x = "Number of Neurons", y = "log APE")

ggsave(
    filename = file.path("output", "figures", "log_ape_vs_neurons.png"), 
    plot = width_plot, width = 7, height = 5, dpi = 300
)

## Censoring threshold sensitivity plot
censoring_plot <- ape_df %>% 
    filter(
        n_lists == 5, 
        train_size == train_size_fixed,
        width == 256
    ) %>% 
    ggplot(aes(x = as.factor(censoring_threshold), y = log(APE))) +
    geom_boxplot(fill = "#2c7bb6", alpha = 0.6) +
    theme_minimal(base_size = 14) +
    labs(x = "Censoring Threshold", y = "log APE")

ggsave(
    filename = file.path("output", "figures", "log_ape_vs_threshold.png"), 
    plot = censoring_plot, width = 7, height = 5, dpi = 300
)

## number of lists sensitivity
n_lists_plot <- ape_df %>% 
    filter(
        censoring_threshold == 10, 
        train_size == train_size_fixed,
        width == 256,
        n_hidden == 3
    ) %>% 
    ggplot(aes(x = as.factor(n_lists), y = log(APE))) +
    geom_boxplot(fill = "#2c7bb6", alpha = 0.6) +
    theme_minimal(base_size = 14) +
    labs(x = "Number of Lists", y = "log APE")

ggsave(
    filename = file.path("output", "figures", "log_ape_vs_lists.png"), 
    plot = n_lists_plot, width = 7, height = 5, dpi = 300
)

## hidden layers sensitivity
hidden_layers_plot <- ape_df %>% 
    filter(
        n_lists == 5, 
        train_size == train_size_fixed,
        censoring_threshold == 10,
        width == 256
    ) %>% 
    ggplot(aes(x = as.factor(n_hidden), y = log(APE))) +
    geom_boxplot(fill = "#2c7bb6", alpha = 0.6) +
    theme_minimal(base_size = 14) +
    labs(x = "Number of Hidden Layers", y = "log APE")

ggsave(
    filename = file.path("output", "figures", "log_ape_vs_hidden_layers.png"), 
    plot = hidden_layers_plot, width = 7, height = 5, dpi = 300
)

## alpha sensitivity
intercept_sensitivity <- ape_df %>% 
    filter(
        n_lists == 5, 
        train_size == train_size_fixed,
        censoring_threshold == 10,
        width == 256,
        n_hidden == 3
    ) %>%
    mutate(
        error = abs(intercept_estimated - intercept_truth)
    ) %>%
    ggplot(aes(x = exp(intercept_truth), y = log(APE))) +
    geom_point(alpha = 0.6, color = "#2c7bb6", size = 2) +
    labs(
        x = expression("True exp" * alpha),
        y = "Log APE",
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank()
    )

ggsave(
    filename = file.path("output", "figures", "log_ape_vs_alpha.png"), 
    plot = intercept_sensitivity, width = 7, height = 5, dpi = 300
)

## Print the table of cnesoring APEs
censoring_summary_table <- ape_df %>% 
    filter(
        n_lists == 5, 
        train_size == train_size_fixed,
        width == 256,
        n_hidden == 3
    ) %>% 
    group_by(censoring_threshold) %>% 
    summarise(
        `1st Qu.` = quantile(APE, 0.75),
        Median = median(APE),
        Mean = mean(APE),
        `3rd Qu.` = quantile(APE, 0.25)
    ) %>%
    pivot_longer(-censoring_threshold, names_to = "Statistic", values_to = "Value") %>%
    pivot_wider(names_from = censoring_threshold, values_from = Value)

## Print for latex
censoring_summary_table %>%
    mutate(across(where(is.numeric), ~round(., 2))) %>%
    {
        cat(names(.), sep = " & ")
        cat(" \\\\\n")
        pwalk(., ~{cat(..., sep = " & "); cat(" \\\\\n")})
    }




### Compare MCMC estimates
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
        lower_ci = lower_95ci,
        upper_ci = upper_95ci,
        rhat
    )

write.csv(
    mcmc_df, 
    file.path("output", "mcmc_intercept_summary.csv"), 
    row.names = FALSE
)

reduced_ape_df <- ape_df %>% 
    filter( 
        n_lists == 5, 
        train_size == train_size_fixed,
        censoring_threshold == 10,
        width == 256,
        n_hidden == 3
    ) %>% 
    select(
        dataset,
        median_nbe = intercept_estimated
    )

mcmc_nbe_comparison_df <- mcmc_df %>% 
    left_join(reduced_ape_df) %>%
    mutate(
        ape_mcmc = abs((exp(true_intercept) - exp(median_mcmc)) / exp(true_intercept)),
        ape_nbe = abs((exp(true_intercept) - exp(median_nbe)) / exp(true_intercept))
    )

mcmc_nbe_comparison_df %>% 
    ggplot(aes(x = median_mcmc, y = median_nbe)) +
    geom_point(alpha = 0.6, color = "#2c7bb6", size = 2) +       # points
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(
        x = expression("NPE Estimated " * exp(alpha)),
        y = expression("MCMC Estimated " * exp(alpha)),
        title = "Comparison of NPE and MCMC Estimates of exp(alpha)"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank()
    )


mcmc_nbe_comparison_df %>% 
    ggplot(aes(x = exp(median_mcmc), y = exp(median_mcmc))) +
    geom_point(alpha = 0.6, color = "#2c7bb6", size = 2) +       # points
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(
        x = expression("NPE Estimated " * exp(alpha)),
        y = expression("MCMC Estimated " * exp(alpha)),
        title = "Comparison of NPE and MCMC Estimates of exp(alpha)"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank()
    )

mcmc_nbe_comparison_df %>% 
    ggplot(aes(x = ape_mcmc, y = ape_nbe, col = rhat)) +
    geom_point(alpha = 0.6, color = "#2c7bb6", size = 2) +       # points
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(
        x = "MCMC APE",
        y = "NPE APE",
        title = "Comparison of NPE and MCMC APEs"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank()
    )

mcmc_nbe_comparison_df %>% 
    ggplot(aes(x = log(ape_mcmc), y = log(ape_nbe))) +
    geom_point(alpha = 0.6, color = "#2c7bb6", size = 2) +       # points
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(
        x = "MCMC Log APE",
        y = "NPE Log APE",
        title = "Comparison of NPE and MCMC Log APEs"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank()
    )


mcmc_nbe_comparison_df %>% 
    ggplot(aes(x = log(ape_mcmc), y = log(ape_nbe), color = rhat > 1.2)) +
    geom_point(alpha = 0.6, size = 2) +       # points
    scale_color_manual(
        values = c("FALSE" = "#2c7bb6", "TRUE" = "red"),
        labels = c("FALSE" = "rhat <= 1.2", "TRUE" = "rhat > 1.2"),
        name = "rhat"
    ) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(
        x = "MCMC Log APE",
        y = "NPE Log APE",
        title = "Comparison of NPE and MCMC Log APEs"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank()
    )

mcmc_nbe_comparison_df %>%
    pivot_longer(
        cols = c(ape_mcmc, ape_nbe),
        names_to = "Method",
        values_to = "APE"
    ) %>%
    ggplot(aes(x = true_intercept, y = log(APE), col = Method)) +
    geom_point() +
    geom_smooth(method = "lm", se = FALSE)


mcmc_df %>% 
    left_join(reduced_ape_df) %>% 
    arrange(desc(rhat))

