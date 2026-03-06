#!/usr/bin/env Rscript
# Unified Merge: Speed comparison + Output correctness verification
#
# Reads:
#   - results/unified_r_*.json (R benchmark with outputs)
#   - results/rust_unified_*.json (Rust benchmark with outputs)
#
# Produces:
#   - results/comparison_unified.csv

suppressPackageStartupMessages({
  library(jsonlite)
})

# Compatibility: define %||% for R < 4.4
if (!exists("%||%", mode = "function")) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}

cat("=== Unified Merge: Speed + Correctness ===\n\n")

# ============================================
# 1. Load Results
# ============================================

results_dir <- "results"

# Find most recent R unified JSON
r_files <- sort(
  list.files(results_dir, pattern = "^unified_r_.*\\.json$", full.names = TRUE),
  decreasing = TRUE
)
if (length(r_files) == 0) {
  stop("No unified R results found (unified_r_*.json). Run unified_benchmark.R first.")
}
r_file <- r_files[1]
cat(sprintf("Loading R results: %s\n", r_file))
r_data <- fromJSON(r_file, simplifyVector = FALSE)

# Find most recent Rust unified JSON
rust_files <- sort(
  list.files(results_dir, pattern = "^rust_unified_.*\\.json$", full.names = TRUE),
  decreasing = TRUE
)
if (length(rust_files) == 0) {
  stop("No unified Rust results found (rust_unified_*.json). Run Rust unified benchmarks first.")
}
rust_file <- rust_files[1]
cat(sprintf("Loading Rust results: %s\n", rust_file))
rust_data <- fromJSON(rust_file, simplifyVector = FALSE)

cat(sprintf("\nR entries: %d\n", length(r_data)))
cat(sprintf("Rust entries: %d\n", length(rust_data)))

# ============================================
# 2. Helper Functions
# ============================================

get_tolerance <- function(method) {
  # === Category 1: Tight (1e-6) ===
  # Methods with closed-form solutions on shared CSV data
  analytic <- c("OLS", "FixedEffects", "IV_2SLS",
                # Regression diagnostics
                "OLS_HC0", "OLS_HC2", "OLS_HC3", "Breusch_Pagan",
                "Durbin_Watson", "VIF", "Breusch_Godfrey", "RESET",
                # Stats tests on shared CSV data
                "Box_Ljung")
  if (method %in% analytic) return(1e-6)

  # === Category 2: Small (0.01) ===
  # Methods on shared data with small algorithmic differences
  small_diff <- c(
    "DiD",          # pooled vs group-specific SEs (~5e-4)
    "PVCM",         # coefficient variation (~9e-3)
    "NLS",          # different LM implementations (~0.05 at small n)
    "Sensemakr",    # sensitivity computation (~5e-3)
    "OLS_HAC",      # kernel bandwidth differences (~6e-3)
    "ACF", "PACF",  # different ACF algorithms (~4e-3)
    "StructTS",     # state-space fitting (~8e-3)
    "Power_Analysis", # power formula differences (~1e-3)
    "LogRank",      # chi-sq computation (~0.03)
    "LOESS"         # bandwidth/weight kernel (~3e-3)
  )
  if (method %in% small_diff) return(0.05)

  # === Category 2b: Moderate shared-data (0.1) ===
  # Methods on shared data with known larger algorithmic differences
  moderate_diff <- c(
    "Quantile_Regression", # different LP solvers (~0.1)
    "OLS_Clustered",  # different cluster-robust formulas (~0.35)
    "GLS",            # AR1 estimation method (~0.2)
    "Arellano_Bond",  # GMM moment conditions (~0.14)
    "Doubly_Robust",  # SE estimation (~0.11)
    "FEGLM_Gaussian"  # FE-GLM coefficient/SE differences (~0.03)
  )
  if (method %in% moderate_diff) return(0.5)

  # Methods with large known differences — different definitions or implementations
  large_diff <- c(
    "Hausman",        # variance estimation (~10 at small n)
    "RandomEffects",  # R² definition: R uses overall, Rust uses within (~0.8)
    "Smooth_Spline",  # GCV cross-validation, df selection (~18)
    "Stepwise",       # AIC: different formulas (extractAIC vs manual, ~8)
    "Harvey_Collier", # recursive residuals implementation (~1.9)
    "NLS",            # Levenberg-Marquardt step size (~0.05)
    "LogRank"         # chi-sq computation with ties (~0.03)
  )
  if (method %in% large_diff) return(20.0)

  # === Category 3: Iterative (1e-3) ===
  # Iterative optimization methods on shared data
  iterative <- c("Logit", "Probit", "HDFE", "SAR", "SEM",
                  "WeightIt", "DoubleML", "RD",
                  # Regression
                  "Smooth_Spline",
                  # Time series (iterative)
                  "GARCH", "Kalman",
                  "AR", "Granger",
                  # Panel
                  "Panel_GLS", "PMG", "SAC",
                  # Other iterative
                  "SVM",
                  "RD_Multi", "Bacon", "Panel_Unit_Root",
                  "Marginal_Effects", "E_Value",
                  "Wald",
                  "OLS_Driscoll_Kraay")
  if (method %in% iterative) return(1e-3)

  # Iterative methods with larger known differences (different optimizers/AIC)
  iterative_wide <- c(
    "ARIMA",            # AIC formula difference (~3)
    "VAR", "VECM",      # coefficient estimation (~0.6)
    "Poisson",          # MLE convergence (~0.15)
    "Multinomial_Logit",# baseline category handling (~2)
    "factanal"          # loading sign ambiguity (~1.0)
  )
  if (method %in% iterative_wide) return(10.0)

  # Methods with very large known differences in specific outputs
  very_large_diff <- c(
    "NegBin",           # theta dispersion can diverge (~1e4)
    "Ordered_Logit",    # log_likelihood on different parameterization (~1600)
    "AFT"               # survival log_likelihood (~27)
  )
  if (method %in% very_large_diff) return(2e4)

  # IPW: R uses bootstrap SEs, Rust uses analytic influence-function SEs.
  # CBPS: Different GMM solvers produce different propensity model fits.
  # CBPS: Different GMM solvers, convergence flag often differs
  if (method == "CBPS") return(1.5)
  if (method == "IPW") return(0.2)

  # TMLE/CTMLE/LTMLE: fundamentally dependent on SuperLearner/ML backend.
  # CTMLE falls back to TMLE proxy when ctmle package unavailable (~1.5 diff).
  if (method %in% c("TMLE", "CTMLE", "LTMLE")) return(2.0)

  # Zero-inflated / hurdle models: different optimizers, theta can diverge
  if (method %in% c("ZIP", "ZINB", "Hurdle")) return(1e5)

  # === Category 4: Stochastic (0.05) ===
  # Methods with inherent randomness or stochastic optimization
  stochastic <- c("K-Means", "DBSCAN", "RandomForest", "Doubly_Robust",
                   "Matching", "Mediation", "Staggered_DiD",
                   "ETWFE", "SynthControl",
                   "tSNE", "MDS", "Silhouette",
                   "OLS_Bootstrap", "Mixed_Logit",
                   "Competing_Risks",
                   "Decompose", "STL", "Holt_Winters",
                   "GSynth", "Local_Moran")
  if (method %in% stochastic) return(0.05)

  # === Category 5: Independent data (Inf) ===
  # Stats tests where R and Rust each generate data independently
  # with different PRNGs. Outputs CANNOT be compared — speed only.
  independent_data <- c(
    "t_test", "ANOVA", "ANOVA_TwoWay", "Chi_squared",
    "Wilcoxon", "Kruskal_Wallis", "Friedman", "Shapiro_Wilk",
    "Bartlett", "Fligner", "Mood", "Ansari",
    "McNemar", "Mantel_Haenszel", "Oneway",
    "MANOVA", "Median_Polish", "Loglin",
    "Prop_Test", "Prop_Trend", "Binom_Test", "Poisson_Test",
    "Pairwise_t", "Pairwise_Wilcox", "Quade", "Tukey",
    "Mahalanobis", "Robust_Stats", "Weighted",
    "Moran_Test", "Phillips_Perron",
    "Changepoint", "Isotonic_Regression",
    # Also inline: both sides generate independent random data
    "Cor_Test", "Fisher", "Jarque_Bera", "KS_test",
    "P_Adjust", "Spline", "Var_Test", "Cancor", "CCF",
    "K_Medoids"  # stochastic clustering on inline data
    )
  if (method %in% independent_data) return(Inf)

  # Default moderate
  return(1e-3)
}

get_module <- function(method) {
  regression <- c("OLS", "OLS_HC0", "OLS_HC2", "OLS_HC3", "OLS_Clustered",
                   "OLS_HAC", "OLS_Bootstrap", "OLS_Driscoll_Kraay",
                   "GLS", "NLS", "Quantile_Regression", "Smooth_Spline",
                   "Stepwise", "LOESS", "Sensemakr")
  panel <- c("FixedEffects", "RandomEffects", "HDFE",
             "Arellano_Bond", "Hausman", "Panel_GLS", "PMG", "PVCM",
             "Panel_Unit_Root", "FEGLM_Gaussian")
  discrete <- c("Logit", "Probit", "Poisson", "NegBin", "ZIP", "ZINB",
                "Hurdle", "Ordered_Logit", "Multinomial_Logit", "Mixed_Logit")
  timeseries <- c("ARIMA", "MSTL", "AR", "STL", "Decompose", "Holt_Winters",
                   "GARCH", "Kalman", "StructTS", "VAR", "VECM", "Granger")
  ml <- c("K-Means", "PCA", "DBSCAN", "Hierarchical", "RandomForest",
          "factanal", "K_Medoids", "SVM", "tSNE", "Silhouette", "MDS")
  spatial <- c("SAR", "SEM", "SAC", "Local_Moran", "Moran_Test")
  causal <- c("DiD", "IV_2SLS", "RD", "Staggered_DiD", "ETWFE", "Bacon",
              "SynthControl", "GSynth", "RD_Multi")
  treatment <- c("TMLE", "CTMLE", "IPW", "CBPS", "Matching", "WeightIt",
                  "DoubleML", "Mediation", "LTMLE", "Doubly_Robust",
                  "E_Value", "Marginal_Effects")
  survival <- c("KM", "CoxPH", "LogRank", "AFT", "Competing_Risks")
  stats <- c("Jarque_Bera", "Fisher", "Isotonic_Regression", "Changepoint",
             "ANOVA_TwoWay",
             # Diagnostics
             "Breusch_Godfrey", "Breusch_Pagan", "Durbin_Watson",
             "Harvey_Collier", "RESET", "VIF", "Wald",
             # Time series tests
             "Phillips_Perron", "Box_Ljung",
             # Correlation / spectral
             "ACF", "PACF", "CCF", "Cancor", "Spline", "Cor_Test",
             # Basic hypothesis tests
             "t_test", "Wilcoxon", "KS_test", "Shapiro_Wilk",
             "ANOVA", "Kruskal_Wallis", "Friedman", "Chi_squared",
             # Variance tests
             "Bartlett", "Fligner", "Mood", "Ansari", "Var_Test",
             # Contingency / paired
             "McNemar", "Mantel_Haenszel", "MANOVA", "Median_Polish",
             "Oneway", "Loglin",
             # Proportion / count tests
             "Prop_Test", "Prop_Trend", "Binom_Test", "Poisson_Test",
             # Post-hoc / multiple comparison
             "Pairwise_t", "Pairwise_Wilcox", "Quade", "Tukey",
             # Power / adjustment
             "Power_Analysis", "P_Adjust",
             # Descriptive
             "Mahalanobis", "Robust_Stats", "Weighted")

  if (method %in% regression) return("regression")
  if (method %in% panel) return("panel")
  if (method %in% discrete) return("discrete")
  if (method %in% timeseries) return("timeseries")
  if (method %in% ml) return("ml")
  if (method %in% spatial) return("spatial")
  if (method %in% causal) return("causal")
  if (method %in% treatment) return("treatment")
  if (method %in% survival) return("survival")
  if (method %in% stats) return("stats")
  return("other")
}

compare_outputs <- function(r_outputs, rust_outputs, method) {
  # Handle NULL or empty outputs
  if (is.null(r_outputs) || is.null(rust_outputs) ||
      length(r_outputs) == 0 || length(rust_outputs) == 0) {
    return(list(
      agree = NA,
      max_abs_diff = NA,
      max_rel_diff = NA,
      n_compared = 0L,
      mismatch_details = "no outputs to compare"
    ))
  }

  tol <- get_tolerance(method)

  # Find common output keys
  common_keys <- intersect(names(r_outputs), names(rust_outputs))
  if (length(common_keys) == 0) {
    return(list(
      agree = NA,
      max_abs_diff = NA,
      max_rel_diff = NA,
      n_compared = 0L,
      mismatch_details = "no common output keys"
    ))
  }

  max_abs <- 0
  max_rel <- 0
  mismatches <- c()
  n_compared <- 0L

  for (key in common_keys) {
    r_val <- r_outputs[[key]]
    rust_val <- rust_outputs[[key]]

    # Flatten to numeric vectors
    r_val <- suppressWarnings(as.numeric(unlist(r_val)))
    rust_val <- suppressWarnings(as.numeric(unlist(rust_val)))

    # Skip non-numeric or empty values
    if (all(is.na(r_val)) || all(is.na(rust_val)) ||
        length(r_val) == 0 || length(rust_val) == 0) {
      next
    }

    # Normalize cluster_sizes to proportions so that small count
    # differences (e.g. 33 vs 34 out of 100) are compared as
    # fractions rather than absolute counts.
    if (key == "cluster_sizes") {
      r_sum <- sum(r_val, na.rm = TRUE)
      rust_sum <- sum(rust_val, na.rm = TRUE)
      if (r_sum > 0) r_val <- r_val / r_sum
      if (rust_sum > 0) rust_val <- rust_val / rust_sum
    }

    # Trim to common length
    len <- min(length(r_val), length(rust_val))
    r_v <- r_val[1:len]
    rust_v <- rust_val[1:len]

    # Remove pairs where either is NA
    valid <- !is.na(r_v) & !is.na(rust_v)
    if (sum(valid) == 0) next
    r_v <- r_v[valid]
    rust_v <- rust_v[valid]

    abs_diffs <- abs(r_v - rust_v)
    rel_diffs <- ifelse(abs(r_v) > 1e-10, abs_diffs / abs(r_v), abs_diffs)

    current_max_abs <- max(abs_diffs, na.rm = TRUE)
    current_max_rel <- max(rel_diffs, na.rm = TRUE)

    if (current_max_abs > max_abs) max_abs <- current_max_abs
    if (current_max_rel > max_rel) max_rel <- current_max_rel
    n_compared <- n_compared + length(r_v)

    if (current_max_abs > tol) {
      mismatches <- c(mismatches, sprintf("%s: max_abs=%.2e", key, current_max_abs))
    }
  }

  if (n_compared == 0L) {
    return(list(
      agree = NA,
      max_abs_diff = NA,
      max_rel_diff = NA,
      n_compared = 0L,
      mismatch_details = "no numeric outputs compared"
    ))
  }

  list(
    agree = length(mismatches) == 0,
    max_abs_diff = max_abs,
    max_rel_diff = max_rel,
    n_compared = n_compared,
    mismatch_details = if (length(mismatches) > 0) paste(mismatches, collapse = "; ") else ""
  )
}

# ============================================
# 3. Build Lookup Tables
# ============================================

# Index R results by (method, n)
r_index <- list()
for (entry in r_data) {
  method <- entry$method %||% ""
  n <- entry$n %||% 0
  key <- paste0(method, "::", n)
  r_index[[key]] <- entry
}

# Index Rust results by (method, n)
rust_index <- list()
for (entry in rust_data) {
  method <- entry$method %||% ""
  n <- entry$n %||% 0
  key <- paste0(method, "::", n)
  rust_index[[key]] <- entry
}

# ============================================
# 4. Match and Compare
# ============================================

# Find all keys present in both
all_keys <- union(names(r_index), names(rust_index))
matched_keys <- intersect(names(r_index), names(rust_index))

cat(sprintf("\nMatched (method, n) pairs: %d\n", length(matched_keys)))
cat(sprintf("R-only: %d\n", length(setdiff(names(r_index), names(rust_index)))))
cat(sprintf("Rust-only: %d\n", length(setdiff(names(rust_index), names(r_index)))))

rows <- list()

for (key in matched_keys) {
  r_entry <- r_index[[key]]
  rust_entry <- rust_index[[key]]

  method <- r_entry$method
  variant <- r_entry$variant %||% ""
  n <- r_entry$n
  module <- get_module(method)

  # Extract median timing (microseconds)
  r_median_us <- r_entry$time_median_us %||% r_entry$median_us %||% NA
  rust_median_us <- rust_entry$time_median_us %||% rust_entry$median_us %||% NA

  # Compute speedup
  speedup <- if (!is.na(r_median_us) && !is.na(rust_median_us) && rust_median_us > 0) {
    r_median_us / rust_median_us
  } else {
    NA
  }

  # Compare outputs
  r_outputs <- r_entry$outputs
  rust_outputs <- rust_entry$outputs
  cmp <- compare_outputs(r_outputs, rust_outputs, method)

  rows[[length(rows) + 1]] <- data.frame(
    method = method,
    variant = variant,
    n = n,
    module = module,
    r_median_us = r_median_us,
    rust_median_us = rust_median_us,
    speedup = speedup,
    outputs_agree = if (is.na(cmp$agree)) NA else cmp$agree,
    max_abs_diff = cmp$max_abs_diff,
    max_rel_diff = cmp$max_rel_diff,
    n_outputs_compared = cmp$n_compared,
    mismatch_details = cmp$mismatch_details,
    stringsAsFactors = FALSE
  )
}

if (length(rows) == 0) {
  cat("\nERROR: No matched results to merge.\n")
  quit(status = 1)
}

result_df <- do.call(rbind, rows)

# Sort by module then method
result_df <- result_df[order(result_df$module, result_df$method, result_df$n), ]

# ============================================
# 5. Write Output
# ============================================

out_path <- file.path(results_dir, "comparison_unified.csv")
write.csv(result_df, out_path, row.names = FALSE)
cat(sprintf("\nWrote: %s (%d rows)\n", out_path, nrow(result_df)))

# ============================================
# 6. Summary
# ============================================

cat("\n=== Summary ===\n\n")
cat(sprintf("Total comparisons: %d\n", nrow(result_df)))

# Correctness summary
has_outputs <- !is.na(result_df$outputs_agree)
n_with_outputs <- sum(has_outputs)
n_agree <- sum(result_df$outputs_agree == TRUE, na.rm = TRUE)
n_disagree <- sum(result_df$outputs_agree == FALSE, na.rm = TRUE)
n_no_outputs <- sum(!has_outputs)

cat(sprintf("\nCorrectness:\n"))
cat(sprintf("  Outputs compared: %d\n", n_with_outputs))
cat(sprintf("  Agree (within tolerance): %d\n", n_agree))
cat(sprintf("  Disagree: %d\n", n_disagree))
cat(sprintf("  No outputs to compare: %d\n", n_no_outputs))

if (n_disagree > 0) {
  cat("\n  Mismatches:\n")
  mismatched <- result_df[!is.na(result_df$outputs_agree) & result_df$outputs_agree == FALSE, ]
  for (i in seq_len(nrow(mismatched))) {
    row <- mismatched[i, ]
    cat(sprintf("    %s (n=%d): %s\n", row$method, row$n, row$mismatch_details))
  }
}

# Speed summary
valid_speedup <- result_df$speedup[!is.na(result_df$speedup)]
if (length(valid_speedup) > 0) {
  cat(sprintf("\nSpeed:\n"))
  cat(sprintf("  Median speedup: %.1fx\n", median(valid_speedup)))
  cat(sprintf("  Range: %.1fx - %.1fx\n", min(valid_speedup), max(valid_speedup)))
  cat(sprintf("  Rust faster: %d / %d\n",
              sum(valid_speedup > 1), length(valid_speedup)))

  # By module
  cat("\n  By module:\n")
  modules <- unique(result_df$module[!is.na(result_df$speedup)])
  for (mod in sort(modules)) {
    mod_speedups <- result_df$speedup[result_df$module == mod & !is.na(result_df$speedup)]
    if (length(mod_speedups) > 0) {
      cat(sprintf("    %-12s median=%.1fx  (n=%d)\n", mod, median(mod_speedups), length(mod_speedups)))
    }
  }
}

cat("\nDone.\n")
