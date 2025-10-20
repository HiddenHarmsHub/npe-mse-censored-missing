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
    geom_point(alpha = 0.6, color = "#2c7bb6", size = 2) +       # points
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
