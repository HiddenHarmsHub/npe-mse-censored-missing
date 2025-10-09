pacman::p_load(tidyverse)


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
