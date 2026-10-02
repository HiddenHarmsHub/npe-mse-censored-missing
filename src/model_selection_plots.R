## Figures for the model-selection study. Reads the CSVs written by model_selection_combine.jl and
## model_selection_real_data.jl for the run named by MS_RUN (default b4_g4), writes PNGs to output/figures/
pacman::p_load(tidyverse)

run <- Sys.getenv("MS_RUN", "b4_g4")
input_dir <- if (run == "b4_g4") file.path("output", "model_selection") else file.path("output", paste0("model_selection_", run))
figures_dir <- file.path("output", "figures")
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)
## Neural results for the ensemble of replicates (rep 0) when present, otherwise replicate 1
plot_rep <- function(df) if (0 %in% df$rep) 0 else 1

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
    if (run != "b4_g4") name <- sub("\\.png$", paste0("_", run, ".png"), name)
    ggsave(file.path(figures_dir, name), plot, width = width, height = height, dpi = 300)
}

## ---- Inclusion-probability reliability ----
reliability_file <- file.path(input_dir, "inclusion_reliability.csv")
if (file.exists(reliability_file)) {
    reliability <- read_csv(reliability_file, show_col_types = FALSE)
    reliability <- reliability |>
        filter(rep == plot_rep(reliability)) |>
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
    sets <- read_csv(structure_file, show_col_types = FALSE)
    sets <- sets |>
        filter(rep == plot_rep(sets)) |>
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
    coverage <- read_csv(population_file, show_col_types = FALSE)
    coverage <- coverage |>
        filter(rep == plot_rep(coverage), target_prior == "primary", method %in% names(method_labels), quantity %in% c("N0", "N")) |>
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

## ---- Comparison across runs (coefficient priors), from model_selection_compare_runs.jl ----
comparison_dir <- file.path("output", "model_selection_comparison")
## Up to four runs: fixed hue order, plus a shape per run because two of the hues are low-contrast
prior_colours <- c("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
prior_shapes <- c(16, 17, 15, 18)
if (file.exists(file.path(comparison_dir, "real_data_population.csv"))) {
    totals <- read_csv(file.path(comparison_dir, "real_data_population.csv"), show_col_types = FALSE) |>
        filter(quantity == "N", method %in% c("neural_bma", "reference_bma")) |>
        mutate(system = factor(real_labels[system], levels = real_labels),
               method = factor(c(neural_bma = "Neural model average", reference_bma = "Reference (enumeration)")[method],
                               levels = c("Neural model average", "Reference (enumeration)")),
               prior = fct_inorder(prior))
    p <- ggplot(totals, aes(median, prior, colour = method, shape = method)) +
        geom_linerange(aes(xmin = q025, xmax = q975), linewidth = 0.7, position = position_dodge(width = 0.5)) +
        geom_point(size = 2.2, position = position_dodge(width = 0.5)) +
        facet_wrap(~system, scales = "free_x") +
        scale_colour_manual(values = unname(method_colours[1:2]), name = NULL) +
        scale_shape_manual(values = c(16, 17), name = NULL) +
        labs(x = "Total population N (posterior median and 95% interval)", y = NULL)
    ggsave(file.path(figures_dir, "model_selection_prior_comparison_population.png"), p, width = 9, height = 4, dpi = 300)

    inclusion <- read_csv(file.path(comparison_dir, "real_data_inclusion.csv"), show_col_types = FALSE, col_types = cols(pair = col_character())) |>
        filter(method %in% c("neural_primary", "reference_primary")) |>
        mutate(system = factor(real_labels[system], levels = real_labels),
               source = factor(c(neural_primary = "Neural", reference_primary = "Reference")[method], levels = c("Neural", "Reference")),
               prior = fct_inorder(prior), pair = paste0("γ[", pair, "]"))
    n_priors <- nlevels(inclusion$prior)
    p <- ggplot(inclusion, aes(pair, prob, colour = prior, shape = prior)) +
        geom_point(size = 2.2, position = position_dodge(width = 0.6)) +
        facet_grid(source ~ system, scales = "free_x", space = "free_x") +
        scale_colour_manual(values = prior_colours[seq_len(n_priors)], name = NULL) +
        scale_shape_manual(values = prior_shapes[seq_len(n_priors)], name = NULL) +
        scale_y_continuous(limits = c(0, 1)) +
        labs(x = NULL, y = "Posterior inclusion probability")
    ggsave(file.path(figures_dir, "model_selection_prior_comparison_inclusion.png"), p, width = 10, height = 5.5, dpi = 300)
}
