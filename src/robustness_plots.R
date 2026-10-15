## Figures for the robustness study.
pacman::p_load(tidyverse)

input_dir <- file.path("output", "robustness")
figures_dir <- file.path("output", "figures")
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

## Fixed categorical order, validated for colour-vision deficiency on a light surface; shapes as secondary encoding
method_labels <- c(npe = "NPE", nbe = "NBE", exact = "Exact posterior", npe_nis = "NPE with importance sampling", bma = "Model averaged")
method_colours <- c("NPE" = "#eb6834", "NBE" = "#8b5cc7", "Exact posterior" = "#1baf7a", "NPE with importance sampling" = "#eda100", "Model averaged" = "#2a78d6")
method_shapes <- c("NPE" = 16, "NBE" = 17, "Exact posterior" = 15, "NPE with importance sampling" = 18, "Model averaged" = 4)
system_labels <- c(A = "A: five lists, uncensored", B = "B: four lists, censored [1, 4]")

theme_set(theme_bw(base_size = 11) + theme(panel.grid.minor = element_blank(), legend.position = "bottom"))
save_figure <- function(plot, name, width = 9, height = 6) ggsave(file.path(figures_dir, name), plot, width = width, height = height, dpi = 300)
method_scales <- list(scale_colour_manual(values = method_colours, name = NULL), scale_shape_manual(values = method_shapes, name = NULL))

summary <- read_csv(file.path(input_dir, "scenario_summary.csv"), show_col_types = FALSE)
scenarios <- read_csv(file.path(input_dir, "scenarios.csv"), show_col_types = FALSE)
scenario_order <- rev(unique(scenarios$label))

## NPE with importance sampling is a diagnostic only: its weights degenerate on most simulated datasets
plotted_methods <- c("npe", "nbe", "exact", "bma")

## ---- Accuracy and coverage by scenario: N0 for the sampling methods, λ0 for the NBE ----
main <- summary |>
    filter(experiment == "main", method %in% plotted_methods,
           (quantity == "N0" & method != "nbe") | (quantity == "lambda0" & method == "nbe")) |>
    mutate(method = factor(method_labels[method], levels = names(method_colours)),
           label = factor(label, levels = scenario_order), system = system_labels[system])
long <- bind_rows(
    main |> transmute(system, label, method, metric = "Median relative error", value = median_relative_error, se = median_relative_error_se),
    main |> transmute(system, label, method, metric = "95% coverage", value = coverage_95, se = coverage_95_se)
)
reference_lines <- tibble(metric = c("Median relative error", "95% coverage"), x = c(0, 0.95))
p <- ggplot(long, aes(value, label, colour = method, shape = method)) +
    geom_vline(data = reference_lines, aes(xintercept = x), linetype = "dashed", colour = "grey50") +
    geom_linerange(aes(xmin = value - 2 * se, xmax = value + 2 * se), position = position_dodge(width = 0.7), linewidth = 0.5) +
    geom_point(position = position_dodge(width = 0.7), size = 1.8) +
    facet_grid(system ~ factor(metric, levels = unique(metric)), scales = "free", space = "free_y") +
    method_scales +
    labs(x = NULL, y = NULL)
save_figure(p, "robustness_scenarios.png", height = 9)

## ---- Approximation to the exact posterior ----
approximation_file <- file.path(input_dir, "approximation.csv")
if (file.exists(approximation_file)) {
    approximation <- read_csv(approximation_file, show_col_types = FALSE) |>
        filter(experiment == "main", quantity == "alpha", method %in% c("npe", "nbe")) |>
        left_join(select(scenarios, system, scenario, label), by = c("system", "scenario")) |>
        mutate(method = factor(method_labels[method], levels = names(method_colours)),
               label = factor(label, levels = scenario_order), system = system_labels[system])
    long <- bind_rows(
        approximation |> transmute(system, label, method, metric = "Neural minus exact median (exact sds)", value = median_d_median, se = median_d_median_se),
        approximation |> transmute(system, label, method, metric = "Share with |difference| > 0.5 sd", value = share_material, se = share_material_se)
    )
    p <- ggplot(long, aes(value, label, colour = method, shape = method)) +
        geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
        geom_linerange(aes(xmin = value - 2 * se, xmax = value + 2 * se), position = position_dodge(width = 0.6), linewidth = 0.5) +
        geom_point(position = position_dodge(width = 0.6), size = 1.8) +
        facet_grid(system ~ factor(metric, levels = unique(metric)), scales = "free", space = "free_y") +
        method_scales +
        labs(x = NULL, y = NULL)
    save_figure(p, "robustness_approximation.png", height = 9)
}

## ---- By band of the true intercept, in distribution ----
by_band <- read_csv(file.path(input_dir, "by_band.csv"), show_col_types = FALSE) |>
    filter(scenario == "S0", method %in% plotted_methods, (quantity == "N0" & method != "nbe") | (quantity == "lambda0" & method == "nbe")) |>
    mutate(method = factor(method_labels[method], levels = names(method_colours)), system = system_labels[system])
long <- bind_rows(
    by_band |> transmute(system, band, method, metric = "Median APE", value = median_ape, se = median_ape_se),
    by_band |> transmute(system, band, method, metric = "95% coverage", value = coverage_95, se = coverage_95_se)
)
p <- ggplot(long, aes(band, value, colour = method, shape = method)) +
    geom_hline(data = tibble(metric = "95% coverage", y = 0.95), aes(yintercept = y), linetype = "dashed", colour = "grey50") +
    geom_linerange(aes(ymin = value - 2 * se, ymax = value + 2 * se), position = position_dodge(width = 0.6), linewidth = 0.5) +
    geom_point(position = position_dodge(width = 0.6), size = 1.8) +
    facet_grid(metric ~ system, scales = "free_y") +
    method_scales +
    labs(x = "True intercept α", y = NULL)
save_figure(p, "robustness_by_band.png", height = 6)

## ---- Coefficient prior: each network on data from its own prior ----
prior <- summary |>
    filter((experiment == "main" & scenario == "S0") | experiment == "prior", quantity == "alpha", method %in% c("npe", "nbe", "exact")) |>
    pivot_longer(c(coverage_50, coverage_80, coverage_90, coverage_95), names_to = "level", values_to = "coverage") |>
    mutate(level = as.numeric(sub("coverage_", "", level)) / 100, se = sqrt(coverage * (1 - coverage) / n),
           prior = factor(paste0("N(0, ", sigma, "²)"), levels = c("N(0, 1²)", "N(0, 2²)", "N(0, 4²)")),
           method = factor(method_labels[method], levels = names(method_colours)), system = system_labels[system])
p <- ggplot(prior, aes(factor(level), coverage - level, colour = method, shape = method)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
    geom_linerange(aes(ymin = coverage - level - 2 * se, ymax = coverage - level + 2 * se), position = position_dodge(width = 0.5), linewidth = 0.5) +
    geom_point(position = position_dodge(width = 0.5), size = 1.8) +
    facet_grid(system ~ prior) +
    method_scales +
    labs(x = "Nominal level", y = "Coverage minus nominal (α)")
save_figure(p, "robustness_prior_scale.png", height = 5.5)

## ---- Original architecture, list and censoring grids ----
grid_file <- file.path(input_dir, "grid_summary.csv")
if (file.exists(grid_file)) {
    grid <- read_csv(grid_file, show_col_types = FALSE) |> mutate(method = factor(method, levels = c("NPE", "NBE")))
    panels <- bind_rows(
        grid |> filter(censoring_upper == 10, width == 256, n_hidden == 3) |> mutate(panel = "Number of lists", setting = n_lists),
        grid |> filter(n_lists == 5, censoring_upper == 10, n_hidden == 3) |> mutate(panel = "Neurons per layer", setting = width),
        grid |> filter(n_lists == 5, censoring_upper == 10, width == 256) |> mutate(panel = "Hidden layers", setting = n_hidden),
        grid |> filter(n_lists == 5, censoring_lower == 0, width == 256, n_hidden == 3) |> mutate(panel = "Upper censoring bound", setting = censoring_upper)
    ) |> mutate(setting = factor(setting, levels = sort(unique(setting))))
    long <- bind_rows(
        panels |> transmute(panel, setting, method, metric = "Median APE", value = median_ape, se = median_ape_se),
        panels |> transmute(panel, setting, method, metric = "95% coverage", value = coverage_95, se = coverage_95_se)
    )
    p <- ggplot(long, aes(setting, value, colour = method, shape = method)) +
        geom_hline(data = tibble(metric = "95% coverage", y = 0.95), aes(yintercept = y), linetype = "dashed", colour = "grey50") +
        geom_linerange(aes(ymin = value - 2 * se, ymax = value + 2 * se), position = position_dodge(width = 0.5), linewidth = 0.5) +
        geom_point(position = position_dodge(width = 0.5), size = 1.8) +
        facet_grid(metric ~ panel, scales = "free") +
        method_scales +
        labs(x = NULL, y = NULL)
    save_figure(p, "grid_summary.png", width = 11, height = 5.5)
}
