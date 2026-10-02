pacman::p_load(tidyverse, Rcapture, ggh4x, gridExtra)

figures_dir <- file.path("output", "figures")
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

colour_map <- c(
    "NBE" = "#a6cee3",
    "NPE" = "#1f78b4",
    "MCMC" = "#b2df8a",
    "IRLS" = "#33a02c"
)

## Helper functions translated from mse_functions.jl

enumerate_two_digit_numbers <- function(K) {
    numbers <- list()
    for (i in 1:(K - 1)) {
        for (j in (i + 1):K) {
            numbers <- c(numbers, list(c(i, j)))
        }
    }
    return(numbers)
}

enumerate_all_combinations <- function(K) {
    combos <- character()
    for (n in 1:K) {
        for (c in combn(1:K, n, simplify = FALSE)) {
            combos <- c(combos, paste(c, collapse = ","))
        }
    }
    return(combos)
}

compute_digit_pairs <- function(n, filter_terms = NULL) {
    n_split <- strsplit(n, ",")[[1]]
    if (length(n_split) == 1) {
        return(list(digits = as.integer(n), pairs = list()))
    }
    digits <- as.integer(n_split)
    pairs <- combn(digits, 2, simplify = FALSE)
    if (!is.null(filter_terms)) {
        pairs <- Filter(function(x) x %in% filter_terms, pairs)
    }
    return(list(digits = digits, pairs = pairs))
}

one_hot_encode_parameters <- function(K) {
    n_gamma <- choose(K, 2)
    n_pars <- 1 + K + n_gamma
    one_hot_matrix <- matrix(0, nrow = 2^K - 1, ncol = n_pars)
    
    lists <- enumerate_all_combinations(K)
    two_digit_numbers <- enumerate_two_digit_numbers(K)
    
    # Create mapping from two-digit pairs to gamma indices
    gamma_map <- setNames(1:n_gamma, sapply(two_digit_numbers, paste, collapse = ","))
    
    for (i in seq_along(lists)) {
        list <- lists[i]
        result <- compute_digit_pairs(list)
        digits <- result$digits
        digit_pairs <- result$pairs
        
        one_hot_matrix[i, 1] <- 1  # Intercept
        
        for (digit in digits) {
            one_hot_matrix[i, 1 + digit] <- 1
        }
        
        for (pair in digit_pairs) {
            pair_key <- paste(pair, collapse = ",")
            one_hot_matrix[i, 1 + K + gamma_map[pair_key]] <- 1
        }
    }
    
    return(one_hot_matrix)
}

ape_df <- read_csv(
    file.path("output", "intercept_estimate_comparison.csv"),
    show_col_types = FALSE
) %>%
    pivot_longer(
        cols = c(APE_NBE, APE_NPE),
        names_to = "Method",
        values_to = "APE"
    ) %>%
    mutate(
        Method = recode(Method, "APE_NBE" = "NBE", "APE_NPE" = "NPE"),
        width = as.factor(width),
        n_hidden = as.factor(n_hidden),
        n_lists = as.factor(n_lists)
    )

train_size_fixed <- 10000
panel_spacing_fixed <- 4

## Width/number of neurons sensitivity plot
width_plot <- ape_df %>% 
    filter(
        n_lists == 5, 
        train_size == train_size_fixed,
        censoring_upper == 10,
        n_hidden == 3
    ) %>%
    ggplot(aes(x = width, y = log(APE), fill = Method)) +
    geom_boxplot(alpha = 0.6) +
    scale_fill_manual(values = colour_map) +
    theme_minimal(base_size = 14) +
    labs(x = "Number of Neurons", y = "Log APE") +
    facet_wrap(~Method) +
    theme(
        strip.text = element_blank(),
        legend.position = "top",
        panel.spacing = grid::unit(panel_spacing_fixed, "lines")
    )

ggsave(
    filename = file.path(figures_dir, "log_ape_vs_neurons.png"), 
    plot = width_plot, width = 7, height = 5, dpi = 300
)

## Censoring threshold sensitivity plot
censoring_plot <- ape_df %>%
    filter(
        n_lists == 5, 
        train_size == train_size_fixed,
        width == 256
    ) %>%
    ggplot(aes(x = as.factor(censoring_upper), y = log(APE), fill = Method)) +
    geom_boxplot(alpha = 0.6) +
    scale_fill_manual(values = colour_map) +
    theme_minimal(base_size = 14) +
    labs(x = "Censoring Threshold", y = "Log APE") +
    scale_x_discrete(expand = c(0.01, 0)) +
    facet_wrap(~Method) +
    theme(
        strip.text = element_blank(),
        legend.position = "top",
        panel.spacing = grid::unit(panel_spacing_fixed, "lines")
    )

ggsave(
    filename = file.path(figures_dir, "log_ape_vs_threshold.png"), 
    plot = censoring_plot, width = 7, height = 5, dpi = 300
)

## number of lists sensitivity
n_lists_plot <- ape_df %>% 
    filter(
        censoring_upper == 10, 
        train_size == train_size_fixed,
        width == 256,
        n_hidden == 3
    ) %>% 
    ggplot(aes(x = n_lists, y = log(APE), fill = Method)) +
    geom_boxplot(alpha = 0.6) +
    scale_fill_manual(values = colour_map) +
    theme_minimal(base_size = 14) +
    labs(x = "Number of Lists", y = "Log APE") +
    facet_wrap(~Method) +
    theme(
        strip.text = element_blank(),
        legend.position = "top",
        panel.spacing = grid::unit(panel_spacing_fixed, "lines")
    )

ggsave(
    filename = file.path(figures_dir, "log_ape_vs_lists.png"), 
    plot = n_lists_plot, width = 7, height = 5, dpi = 300
)

## hidden layers sensitivity
hidden_layers_plot <- ape_df %>% 
    filter(
        n_lists == 5, 
        train_size == train_size_fixed,
        censoring_upper == 10,
        width == 256
    ) %>%
    ggplot(aes(x = as.factor(n_hidden), y = log(APE), fill = Method)) +
    geom_boxplot(alpha = 0.6) +
    scale_fill_manual(values = colour_map) +
    theme_minimal(base_size = 14) +
    labs(x = "Number of Hidden Layers", y = "Log APE") +
    facet_wrap(~Method) +
    theme(
        strip.text = element_blank(),
        legend.position = "top",
        panel.spacing = grid::unit(panel_spacing_fixed, "lines")
    )

ggsave(
    filename = file.path(figures_dir, "log_ape_vs_hidden_layers.png"), 
    plot = hidden_layers_plot, width = 7, height = 5, dpi = 300
)

## alpha sensitivity
intercept_sensitivity <- ape_df %>% 
    filter(
        n_lists == 5, 
        train_size == train_size_fixed,
        censoring_upper == 10,
        width == 256,
        n_hidden == 3
    ) %>%
    ggplot(aes(x = exp(intercept_truth), y = log(APE), col = Method)) +
    geom_point(alpha = 0.3, size = 2) +
    geom_smooth(method = "loess", se = TRUE) +
    labs(
        x = "True hidden population size",
        y = "Log APE",
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position = "top"
    )

ggsave(
    filename = file.path(figures_dir, "log_ape_vs_alpha.png"), 
    plot = intercept_sensitivity, width = 7, height = 5, dpi = 300
)


### Compare MCMC estimates on the stratified benchmark datasets
benchmark_sets <- read_csv(file.path("output", "benchmark_datasets.csv"), show_col_types = FALSE)

mcmc_diagnostics_df <- read_csv(file.path("output", "mcmc_diagnostics.csv"), show_col_types = FALSE)
irls_diagnostics_df <- read_csv(file.path("output", "irls_diagnostics.csv"), show_col_types = FALSE)

mcmc_status <- mcmc_diagnostics_df %>%
    select(dataset, status, stage, capped) %>%
    mutate(status = factor(status, levels = c("initial", "remediated", "failed")))

mcmc_summary_file <- file.path("output", "mcmc_summary.csv")
mcmc_df <- read_csv(mcmc_summary_file, show_col_types = FALSE)

npe_summary_file <- file.path("output", "npe_summary.csv")
npe_df <- read_csv(npe_summary_file, show_col_types = FALSE) %>%
    filter(dataset %in% benchmark_sets$dataset)


nbe_df <- ape_df %>% 
    filter( 
        n_lists == 5, 
        train_size == train_size_fixed,
        censoring_lower == 0,
        censoring_upper == 10,
        width == 256,
        n_hidden == 3,
        Method == "NBE",
        dataset %in% benchmark_sets$dataset
    ) %>% 
    select(
        dataset,
        median_nbe = intercept_NBE,
        lower_ci_nbe = intercept_NBE_lower_ci,
        upper_ci_nbe = intercept_NBE_upper_ci,
    )

comparison_df <- mcmc_df %>%
    filter(parameters == "alpha") %>%
    select(-c(parameters, mean_mcmc, ess_bulk, ess_tail, mcse)) %>% 
    left_join(
        npe_df %>%
            filter(parameters == "alpha") %>%
            select(-parameters),
        by = c("dataset", "true_values")
    ) %>%
    left_join(nbe_df, by = "dataset") %>%
    left_join(mcmc_status, by = "dataset") %>%
    pivot_longer(
        cols = c(
            median_mcmc, median_npe, median_nbe,
            lower_ci_mcmc, lower_ci_npe, lower_ci_nbe,
            upper_ci_mcmc, upper_ci_npe, upper_ci_nbe
        ),
        names_to = c(".value", "Method"),
        names_pattern = "(.+)_(mcmc|npe|nbe)"
    ) %>%
    mutate(
        Method = recode(Method, mcmc = "MCMC", npe = "NPE", nbe = "NBE"),
        error = exp(median) - exp(true_values),
        ape = abs((exp(true_values) - exp(median)) / exp(true_values)),
        absolute_error = abs(error)
    ) %>%
    pivot_longer(
        cols = c(error, ape, absolute_error),
        names_to = "Metric",
        values_to = "Value"
    )

## 1. Accuracy against the simulated truth (all benchmark datasets, MCMC labelled by fit status)
point_estimates <- comparison_df %>%
    mutate(
        Metric = recode(
            Metric, 
            "absolute_error" = "absolute error",
            "ape" = "absolute percentage error"
        ),
        Metric = factor(Metric, levels = c("error", "absolute error", "absolute percentage error"))
    ) %>%
    ggplot(aes(y = Value, x = Method, fill = Method)) +
    geom_violin(alpha = 0.5) +
    scale_fill_manual(values = colour_map) +
    ggh4x::facet_wrap2(
        ~Metric, 
        scales = "free_y",
        labeller = labeller(Metric = function(x) toupper(x))
    ) +
    labs(
        x = "Method",
        y = "Value",
        fill = "Method"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position = "top"
    )

ggsave(
    filename = file.path(figures_dir, "point_estimates_comparison.png"), 
    plot = point_estimates, width = 10, height = 6, dpi = 300
)

point_estimates_zoomed <- comparison_df %>%
    mutate(
        Metric = recode(
            Metric, 
            "absolute_error" = "absolute error",
            "ape" = "absolute percentage error"
        ),
        Metric = factor(Metric, levels = c("error", "absolute error", "absolute percentage error"))
    ) %>%
    ggplot(aes(y = Value, x = Method, fill = Method)) +
    geom_violin(alpha = 0.5) +
    scale_fill_manual(values = colour_map) +
    ggh4x::facet_wrap2(
        ~Metric, 
        scales = "free_y",
        labeller = labeller(Metric = function(x) toupper(x))
    ) +
    ggh4x::facetted_pos_scales(
        y = list(
            Metric == "absolute error"  ~ scale_y_continuous(trans = "log1p"),
            Metric == "absolute percentage error" ~ scale_y_continuous(trans = "log1p"),
            Metric == "error" ~ scale_y_continuous(limits = c(-5000, 5000))
        )
    ) +
    labs(
        x = "Method",
        y = "Value",
        fill = "Method"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position = "top"
    )

ggsave(
    filename = file.path(figures_dir, "point_estimates_comparison_zoomed.png"), 
    plot = point_estimates_zoomed, width = 10, height = 6, dpi = 300
)

intercept_mcmc_status_plot <- comparison_df %>%
    filter(Method == "MCMC", Metric == "ape") %>%
    ggplot(aes(x = exp(true_values), y = log(Value))) +
    geom_point(alpha = 0.5, aes(col = status)) +
    scale_x_log10() +
    labs(
        x = "True hidden population size",
        y = "Log APE",
        col = "MCMC fit"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position = "top"
    )

ggsave(
    filename = file.path(figures_dir, "mcmc_status_ape.png"), 
    plot = intercept_mcmc_status_plot, width = 7, height = 5, dpi = 300
)

## Signed bias, MAPE and RMSE of the hidden population with Monte Carlo standard errors
point_estimate_summaries <- bind_rows(
    comparison_df %>% mutate(Subset = "all"),
    comparison_df %>% filter(status != "failed") %>% mutate(Subset = "validated MCMC fits")
) %>%
    filter(Metric %in% c("error", "ape")) %>%
    pivot_wider(names_from = Metric, values_from = Value) %>%
    group_by(Subset, Method) %>%
    summarise(
        n = n(),
        Bias = mean(error),
        Bias_se = sd(error) / sqrt(n),
        MAPE = mean(ape),
        MAPE_se = sd(ape) / sqrt(n),
        RMSE = sqrt(mean(error^2)),
        .groups = "drop"
    )

write_csv(point_estimate_summaries, file.path("output", "point_estimate_summary.csv"))
print(point_estimate_summaries)

## 2. Approximation to the reference posterior: neural minus MCMC, validated fits only
approximation_df <- comparison_df %>%
    filter(Metric == "error", status != "failed") %>%
    select(dataset, Method, median, lower_ci, upper_ci) %>%
    pivot_longer(c(median, lower_ci, upper_ci), names_to = "Quantity", values_to = "value") %>%
    pivot_wider(names_from = Method, values_from = value) %>%
    pivot_longer(c(NBE, NPE), names_to = "Method", values_to = "neural") %>%
    mutate(
        difference = neural - MCMC,
        Quantity = factor(
            recode(Quantity, lower_ci = "2.5% quantile", median = "median", upper_ci = "97.5% quantile"),
            levels = c("2.5% quantile", "median", "97.5% quantile")
        )
    )

approximation_plot <- approximation_df %>%
    ggplot(aes(x = Quantity, y = difference, fill = Method)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_boxplot(outlier.alpha = 0.3) +
    scale_fill_manual(values = colour_map) +
    labs(
        x = "Posterior quantity of the intercept",
        y = "Neural minus MCMC",
        fill = "Method"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position = "top"
    )

ggsave(
    filename = file.path(figures_dir, "approximation_to_mcmc.png"), 
    plot = approximation_plot, width = 8, height = 5, dpi = 300
)

approximation_df %>%
    group_by(Method, Quantity) %>%
    summarise(
        n = n(),
        mean_difference = mean(difference),
        mean_absolute_difference = mean(abs(difference)),
        .groups = "drop"
    ) %>%
    write_csv(file.path("output", "approximation_to_mcmc_summary.csv"))

## 3. Algorithmic reliability of the reference samplers
reliability_summary <- bind_rows(
    mcmc_diagnostics_df %>% mutate(Sampler = "NUTS"),
    irls_diagnostics_df %>% mutate(Sampler = "IRLS-MH")
) %>%
    group_by(Sampler, status) %>%
    summarise(
        n = n(),
        median_time = median(time),
        max_time = max(time),
        median_min_ess_bulk = median(min_ess_bulk),
        .groups = "drop"
    )

write_csv(reliability_summary, file.path("output", "reference_reliability_summary.csv"))
print(reliability_summary)

## Agreement of NUTS with the importance-sampling reference in Monte Carlo standard errors
is_df <- read_csv(file.path("output", "is_summary.csv"), show_col_types = FALSE)
reference_agreement <- mcmc_df %>%
    select(dataset, parameters, mean_mcmc, mcse) %>%
    inner_join(is_df %>% select(dataset, parameters, mean_is = estimated_means, sd_is = estimated_std, ess, pilot), by = c("dataset", "parameters")) %>%
    left_join(mcmc_status %>% select(dataset, status), by = "dataset") %>%
    mutate(z = (mean_mcmc - mean_is) / sqrt(mcse^2 + sd_is^2 / ess)) %>%
    group_by(status, pilot) %>%
    summarise(
        n_datasets = n_distinct(dataset),
        min_is_ess = min(ess),
        share_abs_z_above_3 = mean(abs(z) > 3),
        max_abs_z = max(abs(z)),
        .groups = "drop"
    )

write_csv(reference_agreement, file.path("output", "reference_agreement_summary.csv"))
print(reference_agreement)

## Calibration under prior-predictive simulation, excluding datasets that hit the simulator cap
sbc_df <- read_csv(file.path("output", "sbc_ranks.csv"), show_col_types = FALSE) %>%
    left_join(benchmark_sets %>% select(dataset, capped), by = "dataset") %>%
    left_join(mcmc_status %>% select(dataset, mcmc_status = status), by = "dataset") %>%
    left_join(irls_diagnostics_df %>% select(dataset, irls_status = status), by = "dataset") %>%
    filter(
        !capped,
        method != "MCMC" | mcmc_status != "failed",
        method != "IRLS" | irls_status == "passed"
    )

levels_grid <- c(seq(0.05, 0.95, by = 0.05), 0.99)
coverage_df <- sbc_df %>%
    crossing(level = levels_grid) %>%
    mutate(inside = abs(rank - 0.5) <= level / 2)

coverage_band <- function(df) {
    df %>%
        summarise(coverage = mean(inside), n = n_distinct(dataset), .groups = "drop") %>%
        mutate(
            band_lower = level - 1.96 * sqrt(level * (1 - level) / n),
            band_upper = level + 1.96 * sqrt(level * (1 - level) / n)
        )
}

coverage_alpha <- coverage_df %>%
    filter(parameters == "alpha") %>%
    group_by(method, level) %>%
    coverage_band()

write_csv(
    coverage_df %>% group_by(method, parameters, level) %>% coverage_band(),
    file.path("output", "coverage_comparison.csv")
)

coverage_plot <- coverage_alpha %>%
    ggplot(aes(x = level, y = coverage)) +
    geom_ribbon(aes(ymin = band_lower, ymax = band_upper), fill = "grey85") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    geom_line(aes(col = method), lwd = 1.2) +
    labs(
        x = "Credible Interval Level",
        y = "Empirical Coverage",
        col = "Method"
    ) +
    scale_color_manual(values = colour_map) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position = "top"
    )

ggsave(
    filename = file.path(figures_dir, "coverage_comparison.png"), 
    plot = coverage_plot, width = 7, height = 5, dpi = 300
)

sbc_histogram <- function(df) {
    n_bins <- 20
    n_datasets <- n_distinct(df$dataset)
    expected <- n_datasets / n_bins
    df %>%
        ggplot(aes(x = rank)) +
        annotate(
            "rect", xmin = -Inf, xmax = Inf,
            ymin = qbinom(0.005, n_datasets, 1 / n_bins), ymax = qbinom(0.995, n_datasets, 1 / n_bins),
            fill = "grey85"
        ) +
        geom_hline(yintercept = expected, linetype = "dashed") +
        geom_histogram(aes(fill = method), breaks = seq(0, 1, length.out = n_bins + 1), alpha = 0.8) +
        scale_fill_manual(values = colour_map) +
        labs(x = "Fractional rank of the true value", y = "Count", fill = "Method") +
        theme_minimal(base_size = 14) +
        theme(panel.grid.minor = element_blank(), legend.position = "top")
}

ggsave(
    filename = file.path(figures_dir, "sbc_alpha.png"), 
    plot = sbc_histogram(filter(sbc_df, parameters == "alpha")) + facet_wrap(~method),
    width = 10, height = 4, dpi = 300
)

ggsave(
    filename = file.path(figures_dir, "sbc_all_parameters.png"), 
    plot = sbc_histogram(sbc_df) + facet_grid(parameters ~ method),
    width = 10, height = 24, dpi = 300
)

## Trace plots for the worst and a random sample of validated NUTS fits
set.seed(1)
trace_datasets <- c(
    mcmc_diagnostics_df %>% slice_max(max_rhat, n = 3) %>% pull(dataset),
    mcmc_diagnostics_df %>% filter(status != "failed") %>% slice_sample(n = 3) %>% pull(dataset)
)

trace_plot <- lapply(trace_datasets, function(idx) {
    read_csv(
        file.path("output", "mcmc_samples", paste0("mcmc_test_results_", idx, ".csv")),
        show_col_types = FALSE
    ) %>%
        select(chain, iteration, intercept) %>%
        mutate(dataset = idx)
}) %>%
    bind_rows() %>%
    left_join(mcmc_status, by = "dataset") %>%
    mutate(label = paste0("dataset ", dataset, " (", status, ")")) %>%
    ggplot(aes(x = iteration, y = intercept, col = factor(chain))) +
    geom_line(alpha = 0.6) +
    facet_wrap(~label, scales = "free_y") +
    labs(x = "Iteration", y = expression(alpha), col = "Chain") +
    theme_minimal(base_size = 12) +
    theme(panel.grid.minor = element_blank(), legend.position = "top")

ggsave(
    filename = file.path(figures_dir, "mcmc_trace_plots.png"), 
    plot = trace_plot, width = 12, height = 7, dpi = 300
)

## speed benchmarking
benchmark_files <- list.files(
    path = file.path("output", "speed_comparisons"), 
    pattern = "*.csv", 
    full.names = TRUE
) %>%
    .[!grepl("train_time_comparison", .)]

benchmark_df <- lapply(benchmark_files, function(benchmark_file) {
    benchmark_df <- read_csv(benchmark_file, show_col_types = FALSE)
    return(benchmark_df)
}) %>%
    bind_rows()

speed_comparison_plot <- benchmark_df %>%
    filter(method %in% c("NPE", "MCMC")) %>%
    group_by(method, iterations) %>%
    summarise(
        mean_time = mean(time),
        sd_time = sd(time),
        .groups = "drop"
    ) %>%
    ggplot(aes(x = iterations, y = mean_time, col = method)) +
    geom_line(lwd = 1.5) +
    geom_hline(
        data = benchmark_df %>% 
            filter(method == "NBE") %>% 
            summarise(mean_time = mean(time)) %>%
            mutate(method = "NBE"),
        aes(yintercept = mean_time, col = method),
        lwd = 1.5
    ) +
    scale_color_manual(values = colour_map) +
    scale_y_log10() +
    scale_x_log10() +
    labs(
        x = "Number of Iterations",
        y = "Mean Time (seconds)",
        col = "Method"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position = "top"
    )

ggsave(
    filename = file.path(figures_dir, "speed_comparison.png"), 
    plot = last_plot(), width = 7, height = 5, dpi = 300
)


lm_comparison_speed <- benchmark_df %>%
    group_by(method, iterations) %>%
    summarise(
        mean_time = mean(time),
        sd_time = sd(time),
        .groups = "drop"
    ) %>%
    group_by(method) %>%
    group_modify(~ {
        lm_fit <- lm(mean_time ~ iterations - 1, data = .x)
        tibble(
            slope = coef(lm_fit)[1],
            r_squared = summary(lm_fit)$r.squared
        )
    }) %>%
    mutate(slope_ms = slope * 1000)

NBE_mean_inference_time <- mean(benchmark_df$time[benchmark_df$method == "NBE"])
print(paste0("Mean inference time for NBE (ms): ", NBE_mean_inference_time * 1000))


train_time_df <- read_csv(
    file.path("output", "speed_comparisons", "train_time_comparison.csv"),
    show_col_types = FALSE
)

train_time_df %>%
    group_by(method) %>%
    summarise(
        mean_time = mean(train_time),
        sd_time = sd(train_time)
    )


ape_hiddenpop_points <- comparison_df %>%
    filter(Metric == "ape") %>%
    select(-Metric) %>%
    mutate(APE = abs((exp(median) - exp(true_values))) / exp(true_values)) %>%
    ggplot(aes(x = exp(true_values), y = log(APE), col = Method)) +
    geom_point() +
    labs(
        x = "True Hidden Population Size",
        y = "Log APE",
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position = "top"
    ) +
    scale_color_manual(values = colour_map)

ape_hiddenpop_lm <- comparison_df %>%
    filter(Metric == "ape") %>%
    select(-Metric) %>%
    mutate(APE = abs((exp(median) - exp(true_values))) / exp(true_values)) %>%
    ggplot(aes(x = exp(true_values), y = log(APE), col = Method)) +
    geom_smooth(method = "lm", se = TRUE) +
    labs(
        x = "True Hidden Population Size",
        y = "Log APE",
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position = "top"
    ) +
    scale_color_manual(values = colour_map)

ggsave(
    filename = file.path(figures_dir, "ape_hiddenpop_combined.png"), 
    plot = gridExtra::grid.arrange(ape_hiddenpop_points, ape_hiddenpop_lm, ncol = 2), 
    width = 14, height = 6, dpi = 300
)




extract_main_lists <- function(idx, test_data) {
    log_counts <- test_data[[idx]][1:5]
    out <- data.frame(t(ifelse(log_counts == -1, NA, round(exp(log_counts) - 1))))
    names(out) <- paste0("L_", 1:5)
    out$dataset <- idx
    rownames(out) <- NULL
    return(out)
}

test_data <- read_csv(file.path("output", "test_data_K5.csv"), show_col_types = FALSE)
test_df <- lapply(benchmark_sets$dataset, function(i) extract_main_lists(i, test_data)) %>% 
    bind_rows()

convergence_comparison <- mcmc_status %>% 
    left_join(test_df, by = "dataset") %>%
    rowwise() %>%
    mutate(main_list_mean = mean(c(L_1, L_2, L_3, L_4, L_5), na.rm = TRUE)) %>%
    ungroup() %>%
    ggplot(aes(x = log1p(main_list_mean), fill = status)) +
    geom_density(alpha = 0.5) +
    labs(
        x = "Log Mean of Observed Main Lists",
        y = "Density",
        fill = "MCMC fit"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position = "top"
    )

ggsave(
    filename = file.path(figures_dir, "convergence_comparison.png"), 
    plot = convergence_comparison, width = 7, height = 5, dpi = 300
)



## real data analysis plots

## silverman analysis
silverman_nbe <- read_csv(
   file.path("output", "real_data_analysis", "silverman_nbe_parameter_estimates.csv"),
   show_col_types = FALSE
)

silverman_npe_samples <- read_csv(
    file.path("output", "real_data_analysis", "silverman_npe_posterior_samples.csv"),
    show_col_types = FALSE
)

silverman_mcmc_samples <- read_csv(
    file.path("output", "real_data_analysis", "silverman_mcmc_posterior_samples.csv"),
    show_col_types = FALSE
)

silverman_mcmc_long <- silverman_mcmc_samples %>%
    select(matches("alpha|beta|gamma")) %>%
    pivot_longer(
        cols = everything(),
        names_to = "Parameter",
        values_to = "Value"
    ) %>%
    mutate(
        Method = "MCMC",
        Parameter = str_replace_all(Parameter, "\\[", "_"),
        Parameter = str_replace_all(Parameter, "\\]", "")
    )

silverman_npe_long <- silverman_npe_samples %>%
    select(matches("alpha|beta|gamma")) %>%
    pivot_longer(
        cols = everything(),
        names_to = "Parameter",
        values_to = "Value"
    ) %>%
    mutate(Method = "NPE")

silverman_nbe_long <- silverman_nbe %>%
    pivot_longer(
        cols = -parameter,
        names_to = "Statistic",
        values_to = "Value"
    ) %>%
    filter(Statistic == "estimate") %>%
    mutate(
        Parameter = parameter,
        Method = "NBE"
    ) %>%
    select(Parameter, Value, Method)

silverman_combined_samples <- bind_rows(
    silverman_mcmc_long,
    silverman_npe_long
)

## also compute frequentist estimates for vertical lines
raw_data_silverman <- read_csv(
    file.path("data", "silverman_5.csv"),
    show_col_types = FALSE
)

expanded_data_silver <- raw_data_silverman %>% 
    mutate(
        count = as.integer(count),
        S1 = as.integer(str_detect(group, "1")),
        S2 = as.integer(str_detect(group, "2")),
        S3 = as.integer(str_detect(group, "3")),
        S4 = as.integer(str_detect(group, "4")),
        S5 = as.integer(str_detect(group, "5"))
    ) %>%
    select(S1, S2, S3, S4, S5, count)

silverman_data <- expand.grid(replicate(5, 0:1, simplify = FALSE)) %>%
    setNames(paste0("S", 1:5)) %>%
    filter(rowSums(.) > 0) %>%
    left_join(expanded_data_silver, by = c("S1", "S2", "S3", "S4", "S5")) %>%
    mutate(count = ifelse(is.na(count), 0, count))

interactions <- function(x) {
    terms <- character()
    for(i in 1:(length(x)-1)) {
      for (j in (i + 1):length(x)) {
            terms  <- c(terms, paste0("S", x[i], "*S", x[j]))
        }
    }
    return(paste(terms, collapse = " + "))
}

full_formula <- function(x) {
    main_terms <- paste(paste0("S", x), collapse = " + ")
    interaction_terms <-  interactions(x)
    formula_str <- paste0("count ~ ", paste(main_terms,interaction_terms, sep = " + "))
    return(formula_str)
}

fit_glm <- glm(
    formula = as.formula(full_formula(1:5)),
    data = silverman_data,
    family = poisson(link = "log")
)

create_parameter_mapping <- function(n_lists) {
    # Create intercept mapping
    mapping <- c("(Intercept)" = "alpha")
    
    # Create main effect mappings (beta)
    for (i in 1:n_lists) {
        mapping[paste0("S", i)] <- paste0("beta_", i)
    }
    
    # Create interaction mappings (gamma)
    for (i in 1:(n_lists - 1)) {
        for (j in (i + 1):n_lists) {
            mapping[paste0("S", i, ":S", j)] <- paste0("gamma_", i, j)
        }
    }
    
    return(mapping)
}

silverman_frequentist_df <- data.frame(
    Parameter = names(fit_glm$coefficients),
    Value = as.numeric(fit_glm$coefficients)
) %>%
    mutate(
        Method = "Frequentist",
        Parameter = recode(Parameter, !!!create_parameter_mapping(5))
    )

silverman_estimate_comparison <- silverman_combined_samples %>%
    ggplot(aes(x = Value, fill = Method)) +
    geom_density(alpha = 0.6) +
    geom_vline(
        data = silverman_nbe_long,
        aes(xintercept = Value, linetype = "NBE"),
        linewidth = 1
    ) +
    geom_vline(
        data = silverman_frequentist_df ,
        aes(xintercept = Value, linetype = "MLE"),
        col = "orange",
        linewidth = 1
    ) +
    facet_wrap(~Parameter, scales = "free") +
    scale_fill_manual(values = colour_map) +
    scale_linetype_manual(
        name = "Point Estimates",
        values = c("NBE" = "solid", "MLE" = "solid"),
        guide = guide_legend(override.aes = list(
            color = c("NBE" = "orange", "MLE" = "black")
        ))
    ) +
    theme_minimal(base_size = 14) +
    labs(
        x = "Parameter Value",
        y = "Density",
        fill = "Posterior Distributions"
    ) +
    theme(
        legend.position = "top",
        panel.spacing = grid::unit(2, "lines")
    )


ggsave(
    filename = file.path(figures_dir, "silverman_parameter_estimate_comparison.png"), 
    plot = silverman_estimate_comparison, width = 10, height = 8, dpi = 300
)

## also plot model assessment

ppd_silverman_npe_df <- read_csv(
    file.path("output", "real_data_analysis", "silverman_npe_posterior_predictive.csv"),
    show_col_types = FALSE
)

silverman_counts <- raw_data_silverman %>%
    mutate(group = paste0("N_", group)) %>%
    right_join(
        ppd_silverman_npe_df %>% 
            pivot_longer(everything(), names_to = "group") %>% 
            distinct(group),
        by = "group"
    ) %>%
    mutate(count = replace_na(count, 0))

ppd_silverman <- ppd_silverman_npe_df %>%
    pivot_longer(
        cols = everything(),
        names_to = "group",
        values_to = "predicted_count"
    ) %>%
    ggplot(aes(x = predicted_count)) +
    geom_histogram(fill = "lightblue", color = "black", alpha = 0.7) +
    geom_vline(
        data = silverman_counts,
        aes(xintercept = count),
        color = "red",
        linewidth = 1
    ) +
    facet_wrap(~group, ncol = 5, scales = "free") +
    labs(
        x = "Count",
        y = "Frequency"
    )


ggsave(
    filename = file.path(figures_dir, "silverman_ppd_npe.png"), 
    plot = ppd_silverman, width = 10, height = 8, dpi = 300
)

## Create parameter estimate table for alpha (intercept/hidden population)
alpha_estimates_table <- bind_rows(
    silverman_mcmc_long %>% 
        filter(Parameter == "alpha") %>%
        summarise(
            Method = "MCMC",
            Parameter = "alpha",
            Median = median(Value),
            Lower_CI = quantile(Value, 0.025),
            Upper_CI = quantile(Value, 0.975)
        ),
    silverman_npe_long %>% 
        filter(Parameter == "alpha") %>%
        summarise(
            Method = "NPE",
            Parameter = "alpha",
            Median = median(Value),
            Lower_CI = quantile(Value, 0.025),
            Upper_CI = quantile(Value, 0.975)
        ),
    silverman_nbe %>% 
        filter(parameter == "alpha") %>%
        summarise(
            Method = "NBE",
            Parameter = "alpha",
            Median = estimate,
            Lower_CI = lower_ci,
            Upper_CI = upper_ci
        ),
    {
        ci_alpha <- suppressMessages(confint.default(fit_glm, parm = "(Intercept)", level = 0.95))
        tibble(
            Method = "MLE",
            Parameter = "alpha",
            Median = coef(fit_glm)["(Intercept)"],
            Lower_CI = ci_alpha[1],
            Upper_CI = ci_alpha[2]
        )
    }
) %>%
    mutate(
        `Hidden Population` = round(exp(Median), 0),
        `95% CI` = ifelse(
            is.na(Lower_CI),
            "-",
            paste0("(", round(exp(Lower_CI), 0), ", ", round(exp(Upper_CI), 0), ")")
        )
    ) %>%
    select(Method, `Hidden Population`, `95% CI`)

## Print LaTeX table
table_lines <- c(
    "\\begin{tabular}{lcc}",
    "\\hline",
    "Method & Hidden Population & 95\\% CI \\\\",
    "\\hline",
    apply(alpha_estimates_table, 1, function(r) paste(r, collapse = " & ") %>% paste0(" \\\\")),
    "\\hline",
    "\\end{tabular}"
)

cat(paste(table_lines, collapse = "\n"), "\n")



### analyse king data

raw_data_king <- read_csv(
    file.path("data", "king.csv"),
    show_col_types = FALSE
)

ppd_king_npe_df <- read_csv(
    file.path("output", "real_data_analysis", "king_npe_posterior_predictive.csv"),
    show_col_types = FALSE
)

king_nbe <- read_csv(
   file.path("output", "real_data_analysis", "king_nbe_parameter_estimates.csv"),
   show_col_types = FALSE
)

king_npe_samples <- read_csv(
    file.path("output", "real_data_analysis", "king_npe_posterior_samples.csv"),
    show_col_types = FALSE
)

king_mcmc_samples <- read_csv(
    file.path("output", "real_data_analysis", "king_mcmc_posterior_samples.csv"),
    show_col_types = FALSE
)

king_mle <- read_csv(
    file.path("output", "real_data_analysis", "king_mle_parameter_estimates.csv"),
    show_col_types = FALSE
)

king_mcmc_long <- king_mcmc_samples %>%
    select(matches("alpha|beta|gamma")) %>%
    pivot_longer(
        cols = everything(),
        names_to = "Parameter",
        values_to = "Value"
    ) %>%
    mutate(
        Method = "MCMC",
        Parameter = str_replace_all(Parameter, "\\[", "_"),
        Parameter = str_replace_all(Parameter, "\\]", "")
    )

king_npe_long <- king_npe_samples %>%
    select(matches("alpha|beta|gamma")) %>%
    pivot_longer(
        cols = everything(),
        names_to = "Parameter",
        values_to = "Value"
    ) %>%
    mutate(Method = "NPE")

king_nbe_long <- king_nbe %>%
    pivot_longer(
        cols = -parameter,
        names_to = "Statistic",
        values_to = "Value"
    ) %>%
    filter(Statistic == "estimate") %>%
    mutate(
        Parameter = parameter,
        Method = "NBE"
    ) %>%
    select(Parameter, Value, Method)

king_combined_samples <- bind_rows(
    king_mcmc_long,
    king_npe_long
)

two_digit_numbers <- enumerate_two_digit_numbers(4)

gamma_names <- sapply(two_digit_numbers, function(x) {
    paste0("gamma_", x[1], x[2])
})

par_names <- c("alpha", paste0("beta_", 1:4), gamma_names)

king_frequentist_df <- king_mle %>%
    rename(Parameter = parameter, Value = estimate) %>%
    mutate(
        Method = "Frequentist",
    )

king_estimate_comparison <- king_combined_samples %>%
    ggplot(aes(x = Value, fill = Method)) +
    geom_density(alpha = 0.6) +
    geom_vline(
        data = king_nbe_long ,
        aes(xintercept = Value, linetype = "NBE"),
        linewidth = 1
    ) +
    geom_vline(
        data = king_frequentist_df,
        aes(xintercept = Value, linetype = "MLE"),
        col = "orange",
        linewidth = 1
    ) +
    facet_wrap(~Parameter, scales = "free") +
    scale_fill_manual(values = colour_map) +
    scale_linetype_manual(
        name = "Point Estimates",
        values = c("NBE" = "solid", "MLE" = "solid"),
        guide = guide_legend(override.aes = list(
            color = c("NBE" = "orange", "MLE" = "black")
        ))
    ) +
    theme_minimal(base_size = 14) +
    labs(
        x = "Parameter Value",
        y = "Density",
        fill = "Posterior Distributions"
    ) +
    theme(
        legend.position = "top",
        panel.spacing = grid::unit(2, "lines")
    )

ggsave(
    filename = file.path(figures_dir, "king_parameter_estimate_comparison.png"), 
    plot = king_estimate_comparison, width = 10, height = 8, dpi = 300
)

## also plot model assessment

king_counts <- raw_data_king %>%
    mutate(group = paste0("N_", group)) %>%
    right_join(
        ppd_king_npe_df %>% 
            pivot_longer(everything(), names_to = "group") %>% 
            distinct(group),
        by = "group"
    ) %>%
    mutate(
        count = ifelse(count == "missing", NA, count),
        count = as.integer(count),
        lower = ifelse(is.na(count), 1, NA),
        upper = ifelse(is.na(count), 4, NA)
    )

ppd_king <- ppd_king_npe_df %>%
    pivot_longer(
        cols = everything(),
        names_to = "group",
        values_to = "predicted_count"
    ) %>%
    ggplot(aes(x = predicted_count)) +
    geom_histogram(fill = "lightblue", color = "black", alpha = 0.7) +
    geom_rect(
        data = king_counts %>% filter(!is.na(lower)),
        aes(xmin = lower, xmax = upper),
        ymin = -Inf,
        ymax = Inf,
        fill = "orange",
        alpha = 0.3,
        inherit.aes = FALSE
    ) +
    geom_vline(
        data = king_counts,
        aes(xintercept = count),
        color = "red",
        linewidth = 1
    ) +
    facet_wrap(~group, ncol = 5, scales = "free") +
    labs(
        x = "Count",
        y = "Frequency"
    )


ggsave(
    filename = file.path(figures_dir, "king_ppd_npe.png"), 
    plot = ppd_king, width = 10, height = 8, dpi = 300
)



## Create parameter estimate table for alpha (intercept/hidden population) for King data
king_alpha_estimates_table <- bind_rows(
    king_mcmc_long %>% 
        filter(Parameter == "alpha") %>%
        summarise(
            Method = "MCMC",
            Parameter = "alpha",
            Median = median(Value),
            Lower_CI = quantile(Value, 0.025),
            Upper_CI = quantile(Value, 0.975)
        ),
    king_npe_long %>% 
        filter(Parameter == "alpha") %>%
        summarise(
            Method = "NPE",
            Parameter = "alpha",
            Median = median(Value),
            Lower_CI = quantile(Value, 0.025),
            Upper_CI = quantile(Value, 0.975)
        ),
    king_nbe %>% 
        filter(parameter == "alpha") %>%
        summarise(
            Method = "NBE",
            Parameter = "alpha",
            Median = estimate,
            Lower_CI = lower_ci,
            Upper_CI = upper_ci
        ),
    king_frequentist_df %>%
        filter(Parameter == "alpha") %>%
        summarise(
            Method = "MLE",
            Parameter = "alpha",
            Median = Value,
            Lower_CI = lower_ci,
            Upper_CI = upper_ci
        )
) %>%
    mutate(
        `Hidden Population` = round(exp(Median), 0),
        `95% CI` = ifelse(
            is.na(Lower_CI),
            "-",
            paste0("(", round(exp(Lower_CI), 0), ", ", round(exp(Upper_CI), 0), ")")
        )
    ) %>%
    select(Method, `Hidden Population`, `95% CI`)

## Print LaTeX table for King data
king_table_lines <- c(
    "\\begin{tabular}{lcc}",
    "\\hline",
    "Method & Hidden Population & 95\\% CI \\\\",
    "\\hline",
    apply(king_alpha_estimates_table, 1, function(r) paste(r, collapse = " & ") %>% paste0(" \\\\")),
    "\\hline",
    "\\end{tabular}"
)

cat(paste(king_table_lines, collapse = "\n"), "\n")
