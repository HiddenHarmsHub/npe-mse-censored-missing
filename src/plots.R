pacman::p_load(tidyverse)

## Width sensitivity plot
width_df <- read_csv(file.path("output", "width_sensitivity.csv"))

width_plot <- width_df %>% 
    ggplot(aes(group = width, y = log(APE))) +
    geom_boxplot(fill = "#2c7bb6", alpha = 0.6) +
    theme_minimal(base_size = 14) +
    scale_x_discrete(labels = c(8, 16, 32, 64, 128, 256)) +
    labs(title = "", x = "Number of Neurons", y = "log APE")

ggsave(
    filename = file.path("output", "figures", "log_ape_vs_neurons_new.png"), 
    plot = width_plot, width = 7, height = 5, dpi = 300
)


## Censoring threshold sensitivity plot
censoring_df <- read_csv(file.path("output", "censoring_sensitivity.csv"))

censoring_plot <- censoring_df %>% 
    ggplot(aes(group = censoring_threshold, y = log(APE))) +
    geom_boxplot(fill = "#2c7bb6", alpha = 0.6) +
    theme_minimal(base_size = 14) +
    scale_x_discrete(labels = c(0, 2, 4, 8, 16, 32, 64, 128)) +
    labs(title = "", x = "Censoring Threshold", y = "log APE")

ggsave(
    filename = file.path("output", "figures", "log_ape_vs_threshold_new.png"), 
    plot = censoring_plot, width = 7, height = 5, dpi = 300
)


## number of lists sensitivity
n_lists_df <- read_csv(file.path("output", "n_lists_sensitivity.csv"))

n_lists_plot <- n_lists_df %>% 
    ggplot(aes(group = n_lists, y = log(APE))) +
    geom_boxplot(fill = "#2c7bb6", alpha = 0.6) +
    theme_minimal(base_size = 14) +
    scale_x_discrete(labels = c(1, 2, 4, 8, 16, 32, 64, 128)) +
    labs(title = "", x = "Number of Lists", y = "log APE")

ggsave(
    filename = file.path("output", "figures", "log_ape_vs_n_lists_new.png"), 
    plot = n_lists_plot, width = 7, height = 5, dpi = 300
)


## number of hidden layers sensitivity
n_layers_df <- read_csv(file.path("output", "n_hidden_sensitivity.csv"))
n_layers_plot <- n_layers_df %>% 
    ggplot(aes(group = n_hidden, y = log(APE))) +
    geom_boxplot(fill = "#2c7bb6", alpha = 0.6) +
    theme_minimal(base_size = 14) +
    scale_x_discrete(labels = c(1, 2, 3, 4)) +
    labs(title = "", x = "Number of Hidden Layers", y = "log APE")

ggsave(
    filename = file.path("output", "figures", "log_ape_vs_n_layers_new.png"), 
    plot = n_layers_plot, width = 7, height = 5, dpi = 300
)
