pacman::p_load(tidyverse, Rcapture, ggh4x)

colour_map <- c(
    "NBE" = "#a6cee3",
    "NPE" = "#1f78b4",
    "MCMC" = "#b2df8a"
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

ape_df <- read_csv(file.path("output", "intercept_estimate_comparison.csv")) %>%
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
    labs(x = "Number of Neurons", y = "log APE") +
    facet_wrap(~Method) +
    theme(
        strip.text = element_blank(),
        legend.position = "top",
        panel.spacing = grid::unit(panel_spacing_fixed, "lines")
    )

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
    ggplot(aes(x = as.factor(censoring_upper), y = log(APE), fill = Method)) +
    geom_boxplot(alpha = 0.6) +
    scale_fill_manual(values = colour_map) +
    theme_minimal(base_size = 14) +
    labs(x = "Censoring Threshold", y = "log APE") +
    scale_x_discrete(expand = c(0.01, 0)) +
    facet_wrap(~Method) +
    theme(
        strip.text = element_blank(),
        legend.position = "top",
        panel.spacing = grid::unit(panel_spacing_fixed, "lines")
    )

ggsave(
    filename = file.path("output", "figures", "log_ape_vs_threshold.png"), 
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
    labs(x = "Number of Lists", y = "log APE") +
    facet_wrap(~Method) +
    theme(
        strip.text = element_blank(),
        legend.position = "top",
        panel.spacing = grid::unit(panel_spacing_fixed, "lines")
    )

ggsave(
    filename = file.path("output", "figures", "log_ape_vs_lists.png"), 
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
    labs(x = "Number of Hidden Layers", y = "log APE") +
    facet_wrap(~Method) +
    theme(
        strip.text = element_blank(),
        legend.position = "top",
        panel.spacing = grid::unit(panel_spacing_fixed, "lines")
    )

ggsave(
    filename = file.path("output", "figures", "log_ape_vs_hidden_layers.png"), 
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

intercept_sensitivity_smoothed <- ape_df %>% 
    filter(
        n_lists == 5, 
        train_size == train_size_fixed,
        censoring_upper == 10,
        width == 256,
        n_hidden == 3
    ) %>%
    ggplot(aes(x = exp(intercept_truth), y = log(APE), col = Method)) +
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
    group_by(censoring_upper) %>% 
    summarise(
        `1st Qu.` = quantile(APE, 0.75),
        Median = median(APE),
        Mean = mean(APE),
        `3rd Qu.` = quantile(APE, 0.25)
    ) %>%
    pivot_longer(-censoring_upper, names_to = "Statistic", values_to = "Value") %>%
    pivot_wider(names_from = censoring_upper, values_from = Value)

## Print for latex
censoring_summary_table %>%
    mutate(across(where(is.numeric), ~round(., 2))) %>%
    {
        cat(names(.), sep = " & ")
        cat(" \\\\\n")
        pwalk(., ~{cat(..., sep = " & "); cat(" \\\\\n")})
    }




### Compare MCMC estimates

mcmc_summary_file <- file.path("output", "mcmc_summary.csv")

if(file.exists(mcmc_summary_file)) {
    mcmc_df <- read_csv(mcmc_summary_file, show_col_types = FALSE)
} else {
    mcmc_files <- list.files(
        path = file.path("output", "mcmc_summary"), 
        pattern = "*.csv", 
        full.names = TRUE
    )

    mcmc_df <- lapply(mcmc_files, function(mcmc_file) {
        mcmc_df <- read_csv(mcmc_file, show_col_types = FALSE)
        mcmc_df$dataset <- parse_number(mcmc_file)
        return(mcmc_df)
    }) %>% 
        bind_rows() %>% 
        select(
            dataset, 
            parameters,
            true_values,
            median_mcmc = estimated_medians,
            lower_ci_mcmc = lower_95ci,
            upper_ci_mcmc = upper_95ci,
            rhat
        )

    write.csv(
        mcmc_df, 
        file.path("output", "mcmc_summary.csv"), 
        row.names = FALSE
    )
}

npe_summary_file <- file.path("output", "npe_summary.csv")
if(file.exists(npe_summary_file)) {
    npe_df <- read_csv(npe_summary_file, show_col_types = FALSE)
} else {
    npe_files <- list.files(
        path = file.path("output", "npe_summary"), 
        pattern = "*.csv", 
        full.names = TRUE
    )

    npe_df <- lapply(npe_files, function(npe_file) {
        npe_df <- read_csv(npe_file, show_col_types = FALSE)
        npe_df$dataset <- parse_number(npe_file)
        return(npe_df)
    }) %>% 
        bind_rows() %>% 
        select(
            dataset, 
            parameters,
            true_values,
            median_npe = estimated_medians,
            lower_ci_npe = lower_95ci,
            upper_ci_npe = upper_95ci
        )

    write.csv(
        npe_df, 
        file.path("output", "npe_summary.csv"), 
        row.names = FALSE
    )
}

nbe_df <- ape_df %>% 
    filter( 
        n_lists == 5, 
        train_size == train_size_fixed,
        censoring_lower == 0,
        censoring_upper == 10,
        width == 256,
        n_hidden == 3,
        Method == "NBE"
    ) %>% 
    select(
        dataset,
        median_nbe = intercept_NBE,
        lower_ci_nbe = intercept_NBE_lower_ci,
        upper_ci_nbe = intercept_NBE_upper_ci,
    )

comparison_df <- mcmc_df %>%
    filter(parameters == "intercept") %>%
    select(-parameters) %>% 
    left_join(
        npe_df %>%
            filter(parameters == "alpha") %>%
            select(-parameters),
        by = c("dataset", "true_values")
    ) %>%
    left_join(nbe_df, by = "dataset") %>%
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
    filename = file.path("output", "figures", "point_estimates_comparison.png"), 
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
    filename = file.path("output", "figures", "point_estimates_comparison_zoomed.png"), 
    plot = point_estimates_zoomed, width = 10, height = 6, dpi = 300
)


point_estimate_summaries <- comparison_df %>%
    group_by(Method, Metric) %>%
    summarise(
        Mean = mean(Value),
        Median = median(Value),
        SD = sd(Value),
        Q1 = quantile(Value, 0.25),
        Q3 = quantile(Value, 0.75)
    ) %>%
    arrange(Metric, Method)

comparison_df %>%
    group_by(Method, Metric) %>%
    summarise(
        Mean = mean(Value),
        RMSE = sqrt(mean(Value^2)),
        .groups = "drop"
    ) %>%
    filter(Metric %in% c("error", "ape")) %>%
    pivot_wider(
        names_from = Metric,
        values_from = c(Mean, RMSE)
    ) %>%
    select(Method, Mean_error, Mean_ape, RMSE_error) %>%
    rename(
        Bias = Mean_error,
        `Mean APE` = Mean_ape,
        `RMSE` = RMSE_error
    )


## uncertainty quantification plots



comparison_df %>%
    select(-c(Metric, Value)) %>%
    distinct() %>%
    mutate(
        param_estimate_in_CI = ifelse(
            (true_values >= lower_ci) & (true_values <= upper_ci),
            TRUE,
            FALSE
        )
    ) %>%
    group_by(Method) %>%
    summarise(
        coverage_par = mean(param_estimate_in_CI)
    )

comparison_df %>%
    select(-c(Metric, Value)) %>%
    distinct() %>%
    mutate(width = exp(upper_ci) - exp(lower_ci)) %>%
    arrange(desc(width))

width_plot <- comparison_df %>%
    select(-c(Metric, Value)) %>%
    distinct() %>%
    mutate(width = exp(upper_ci) - exp(lower_ci)) %>%
    ggplot(aes(x = Method, y = width, fill = Method)) +
    geom_boxplot(alpha = 0.5) +
    scale_fill_manual(values = colour_map) +
    theme_minimal(base_size = 14) +
    labs(x = "Method", y = "95% Credible Interval Width (Hidden Population Size)") +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position = "top"
    )

ggsave(
    filename = file.path("output", "figures", "ci_width_comparison.png"), 
    plot = width_plot, width = 7, height = 5, dpi = 300
)


coverage_df <- read_csv(file.path("output", "coverage_comparison_npe_mcmc.csv"))

converged_datasets <- mcmc_df %>%
    filter(parameters == "intercept", rhat <= 1.01) %>%
    pull(dataset) %>%
    unique()


coverage_plot <- coverage_df %>%
    group_by(method, level) %>%
    summarise(mean_coverage = mean(inside)) %>%
    bind_rows(data.frame(
        method = "y=x",
        level = seq(0, 1, by = 0.01),
        mean_coverage = seq(0, 1, by = 0.01)
    )) %>%
    ggplot(aes(x = level, y = mean_coverage)) +
    geom_line(aes(col = method), lwd = 1.5) +
    labs(
        x = "Credible Interval Level",
        y = "Empirical Coverage",
        col = "Method"
    ) +
    scale_color_manual(
        values = c(colour_map, "y=x" = "red"),
        breaks = c(names(colour_map), "y=x")
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position = "top"
    )

ggsave(
    filename = file.path("output", "figures", "coverage_comparison.png"), 
    plot = coverage_plot, width = 7, height = 5, dpi = 300
)





## speed benchmarking

benchmark_files <- list.files(
    path = file.path("output", "speed_comparisons"), 
    pattern = "*.csv", 
    full.names = TRUE
)



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
    filename = file.path("output", "figures", "speed_comparison.png"), 
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
    filename = file.path("output", "figures", "ape_hiddenpop_combined.png"), 
    plot = gridExtra::grid.arrange(ape_hiddenpop_points, ape_hiddenpop_lm, ncol = 2), 
    width = 14, height = 6, dpi = 300
)




extract_main_lists <- function(idx, test_data) {
    out <- data.frame(t(floor(exp(test_data[1:15, idx] - 1))))
    names(out) <- c(paste0("L_", 1:5), paste0("L_", c(12, 13, 14, 15, 23, 24, 25, 34, 35, 45)))
    out$dataset <- idx
    rownames(out) <- NULL
    return(out)
}

test_data <- read_csv(file.path("output", "test_data_K5.csv"))
test_df <- lapply(1:ncol(test_data), function(i) extract_main_lists(i, test_data)) %>% 
    bind_rows()

convergence_comparison <- mcmc_nbe_comparison_df %>% 
    left_join(test_df, by = "dataset") %>%
    rowwise() %>%
    mutate(
        convergence = ifelse(rhat > 1.01, FALSE, TRUE),
        main_list_mean = mean(c(L_1, L_2, L_3, L_4, L_5))
    ) %>%
    ungroup() %>%
    ggplot(aes(x = log(main_list_mean), fill = convergence)) +
    geom_density(alpha = 0.5) +
    scale_fill_manual(
        values = c("FALSE" = "red", "TRUE" = "#2c7bb6"),
        labels = c("FALSE" = "R > 1.01", "TRUE" = "R \u2264 1.01")
    ) +
    labs(
        x = "Log Mean of Main Lists",
        y = "Density",
        fill = "GR Statistic"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position = "top"
    )

ggsave(
    filename = file.path("output", "figures", "convergence_comparison.png"), 
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
    select(matches("intercept|beta|gamma")) %>%
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
    select(matches("intercept|beta|gamma")) %>%
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

silverman_combined_samples <- bind_rows(silverman_mcmc_long, silverman_npe_long)

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
        data = silverman_nbe_long %>% filter(Parameter != "alpha"),
        aes(xintercept = Value, linetype = "NBE"),
        linewidth = 1
    ) +
    geom_vline(
        data = silverman_frequentist_df %>% filter(Parameter != "alpha"),
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
            color = c("NBE" = "black", "MLE" = "orange")
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
    filename = file.path("output", "figures", "silverman_parameter_estimate_comparison.png"), 
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
        size = 1
    ) +
    facet_wrap(~group, ncol = 5, scales = "free") +
    labs(
        x = "Count",
        y = "Frequency"
    )


ggsave(
    filename = file.path("output", "figures", "silverman_ppd_npe.png"), 
    plot = ppd_silverman, width = 10, height = 8, dpi = 300
)





### analyse king data

raw_data_king <- read_csv(
    file.path("data", "king.csv"),
    show_col_types = FALSE
)

ppd_king_npe_df <- read_csv(
    file.path("output", "real_data_analysis", "king_npe_posterior_predictive.csv"),
    show_col_types = FALSE
)

king_data <- raw_data_king %>%
    mutate(count = ifelse(count == "missing", "-1", count)) %>%
    mutate(count = as.integer(count))

king_loglikelihood <- function(king_data, X, pars) {
    logliks <- numeric(nrow(king_data))
    rates <- exp(X %*% pars)
    logliks <- ifelse(
        king_data$count == -1,
        dpois(1:4, lambda = rates, log = TRUE),
        dpois(king_data$count, lambda = rates, log = TRUE)
    )
    return(sum(logliks))
}


X_king <- one_hot_encode_parameters(4)
pars_init <- c(3.0, -2.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01)


optimised <- optim(
    par = pars_init,
    fn = function(pars) -king_loglikelihood(king_data, X_king, pars),
    method = "BFGS"
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

king_combined_samples <- bind_rows(king_mcmc_long, king_npe_long)

two_digit_numbers <- enumerate_two_digit_numbers(4)

gamma_names <- sapply(two_digit_numbers, function(x) {
    paste0("gamma_", x[1], x[2])
})

par_names <- c("alpha", paste0("beta_", 1:4), gamma_names)

king_frequentist_df <- data.frame(
    Parameter = par_names,
    Value = as.numeric(optimised$par)
) %>%
    mutate(
        Method = "Frequentist",
        Parameter = recode(Parameter, !!!create_parameter_mapping(5))
    )

king_estimate_comparison <- king_combined_samples %>%
    ggplot(aes(x = Value, fill = Method)) +
    geom_density(alpha = 0.6) +
    geom_vline(
        data = king_nbe_long %>% filter(Parameter != "alpha"),
        aes(xintercept = Value, linetype = "NBE"),
        linewidth = 1
    ) +
    geom_vline(
        data = king_frequentist_df %>% filter(Parameter != "alpha"),
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
            color = c("NBE" = "black", "MLE" = "orange")
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
    filename = file.path("output", "figures", "king_parameter_estimate_comparison.png"), 
    plot = king_estimate_comparison, width = 10, height = 8, dpi = 300
)
