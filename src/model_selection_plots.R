## Figures for the model-selection study. Reads output/model_selection/*.csv (from
## model_selection_combine.jl and model_selection_real_data.jl), writes PNGs to output/figures/
pacman::p_load(tidyverse)

input_dir <- file.path("output", "model_selection")
figures_dir <- file.path("output", "figures")
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

## Fixed categorical order, validated for colour-vision deficiency on a light surface; two slots sit
## below 3:1 contrast, so every series also gets its own point shape
method_colours <- c(
    "Model averaged" = "#2a78d6",
    "All interactions" = "#eb6834",
    "MAP structure" = "#1baf7a",
    "Oracle structure" = "#eda100"
)
method_shapes <- c("Model averaged" = 16, "All interactions" = 17, "MAP structure" = 15, "Oracle structure" = 18)
method_labels <- c(bma = "Model averaged", all = "All interactions", map = "MAP structure", oracle = "Oracle structure")
real_labels <- c(A = "UK modern slavery (A)", B = "King drug use (B)")
system_labels <- c(A = "A: five lists, uncensored", B = "B: four lists, censored [1, 4]")

theme_set(theme_bw(base_size = 11) + theme(panel.grid.minor = element_blank(), legend.position = "bottom"))

save_figure <- function(plot, name, width = 8, height = 4.5) {
    ggsave(file.path(figures_dir, name), plot, width = width, height = height, dpi = 300)
}

## ---- Inclusion-probability reliability ----
reliability_file <- file.path(input_dir, "inclusion_reliability.csv")
if (file.exists(reliability_file)) {
    reliability <- read_csv(reliability_file, show_col_types = FALSE) |>
        filter(rep == 1) |>
        mutate(system = system_labels[system])
    p <- ggplot(reliability, aes(mean_predicted, frequency)) +
        geom_abline(linetype = "dashed", colour = "grey50") +
        geom_linerange(aes(ymin = pmax(frequency - 2 * se, 0), ymax = pmin(frequency + 2 * se, 1)), colour = "#2a78d6", linewidth = 0.6) +
        geom_point(colour = "#2a78d6", size = 2) +
        facet_wrap(~system) +
        coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
        labs(x = "Predicted inclusion probability", y = "Observed inclusion frequency")
    save_figure(p, "model_selection_inclusion_reliability.png", height = 4.2)
}

## ---- Structure credible-set coverage ----
structure_file <- file.path(input_dir, "structure_summary.csv")
if (file.exists(structure_file)) {
    sets <- read_csv(structure_file, show_col_types = FALSE) |>
        filter(rep == 1) |>
        pivot_longer(starts_with("set_coverage_"), names_to = "level", values_to = "coverage") |>
        mutate(level = as.numeric(str_remove(level, "set_coverage_")) / 100, system = system_labels[system])
    p <- ggplot(sets, aes(level, coverage)) +
        geom_abline(linetype = "dashed", colour = "grey50") +
        geom_line(colour = "#2a78d6", linewidth = 0.7) +
        geom_point(colour = "#2a78d6", size = 2) +
        facet_wrap(~system) +
        coord_equal(xlim = c(0.4, 1), ylim = c(0.4, 1)) +
        labs(x = "Nominal credible-set level", y = "Frequency the true structure is in the set")
    save_figure(p, "model_selection_credible_sets.png", height = 4.2)
}

## ---- Coverage of N0 and N intervals by method ----
population_file <- file.path(input_dir, "population_summary.csv")
if (file.exists(population_file)) {
    coverage <- read_csv(population_file, show_col_types = FALSE) |>
        filter(rep == 1, target_prior == "primary", method %in% names(method_labels), quantity %in% c("N0", "N")) |>
        pivot_longer(starts_with("coverage_"), names_to = "level", values_to = "coverage") |>
        mutate(
            level = as.numeric(str_remove(level, "coverage_")) / 100,
            method = factor(method_labels[method], levels = names(method_colours)),
            system = system_labels[system],
            quantity = factor(quantity, levels = c("N0", "N"), labels = c("Unseen~N[0]", "Total~N"))
        )
    p <- ggplot(coverage, aes(level, coverage, colour = method, shape = method)) +
        geom_abline(linetype = "dashed", colour = "grey50") +
        geom_line(linewidth = 0.7) +
        geom_point(size = 2.2) +
        facet_grid(quantity ~ system, labeller = labeller(quantity = label_parsed)) +
        scale_colour_manual(values = method_colours, name = NULL) +
        scale_shape_manual(values = method_shapes, name = NULL) +
        labs(x = "Nominal interval level", y = "Empirical coverage")
    save_figure(p, "model_selection_population_coverage.png", height = 6)
}

## ---- Real data: inclusion probabilities, neural versus reference ----
inclusion_sources <- c(neural_primary = "Neural", reference_primary = "Reference (enumeration)")
real_inclusion <- map_dfr(c("A", "B"), function(s) {
    f <- file.path(input_dir, "real_data", paste0(s, "_inclusion.csv"))
    if (file.exists(f)) read_csv(f, show_col_types = FALSE, col_types = cols(pair = col_character())) |> mutate(system = s) else NULL
})
if (nrow(real_inclusion) > 0) {
    real_inclusion <- real_inclusion |>
        filter(method %in% names(inclusion_sources)) |>
        mutate(method = factor(inclusion_sources[method], levels = inclusion_sources),
               system = factor(real_labels[system], levels = real_labels),
               pair = paste0("γ[", pair, "]"))
    p <- ggplot(real_inclusion, aes(pair, prob, colour = method, shape = method)) +
        geom_point(size = 2.4, position = position_dodge(width = 0.5)) +
        facet_wrap(~system, scales = "free_x") +
        scale_colour_manual(values = unname(method_colours[1:2]), name = NULL) +
        scale_shape_manual(values = c(16, 17), name = NULL) +
        scale_y_continuous(limits = c(0, 1)) +
        labs(x = NULL, y = "Posterior inclusion probability")
    save_figure(p, "model_selection_real_inclusion.png", height = 4.2)
}

## ---- Real data: total population by method ----
real_population <- map_dfr(c("A", "B"), function(s) {
    f <- file.path(input_dir, "real_data", paste0(s, "_population.csv"))
    if (file.exists(f)) read_csv(f, show_col_types = FALSE) |> mutate(system = s) else NULL
})
if (nrow(real_population) > 0) {
    totals <- real_population |>
        filter(quantity == "N", !str_detect(method, "_rep")) |>
        mutate(system = factor(real_labels[system], levels = real_labels),
               method = fct_reorder(method, median))
    p <- ggplot(totals, aes(median, method)) +
        geom_linerange(aes(xmin = q025, xmax = q975), colour = "#2a78d6", linewidth = 0.7) +
        geom_point(colour = "#2a78d6", size = 2.2) +
        facet_wrap(~system, scales = "free") +
        labs(x = "Total population N (posterior median and 95% interval)", y = NULL)
    save_figure(p, "model_selection_real_population.png", width = 9, height = 5)
}
