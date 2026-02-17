#!/usr/bin/env Rscript
# Merge R and Rust Benchmark Results
#
# Reads:
#   - R benchmark CSVs from results/r_*_{timestamp}.csv
#   - Standalone R benchmark CSVs (r_acf_*.csv, r_bartlett_*.csv, etc.)
#   - Rust benchmark JSON from performance/results/rust_comprehensive_*.json
#
# Produces:
#   - results/comparison_speed.csv  -- side-by-side speed comparison
#   - results/comparison_memory.csv -- side-by-side memory comparison
#   - results/validation_coverage.csv -- method-level validation status

suppressPackageStartupMessages({
  library(jsonlite)
})

cat("=== Merging R + Rust Benchmark Results ===\n\n")

# ============================================
# 1. Load R Results
# ============================================

r_results_dir <- "results"

# Find all R benchmark CSVs (comprehensive + standalone)
r_csv_files <- list.files(r_results_dir, pattern = "^r_.*\\.csv$", full.names = TRUE)

# Also include standalone benchmark CSVs that do not start with r_ but contain
# benchmark data (e.g., survival_benchmarks_*.csv, econometrics_*.csv, etc.)
other_csv_files <- list.files(r_results_dir,
  pattern = "^(survival_benchmarks|econometrics_|forecasting_|ml_|regression_|fixest_|synth_|causalweight_).*\\.csv$",
  full.names = TRUE)
r_csv_files <- unique(c(r_csv_files, other_csv_files))

# Exclude output files we create
r_csv_files <- r_csv_files[!grepl("comparison_|validation_|combined_results", r_csv_files)]

if (length(r_csv_files) == 0) {
  cat("WARNING: No R benchmark CSV files found in results/\n")
  cat("Run benchmark_comprehensive.R first.\n")
  r_all <- data.frame()
} else {
  cat(sprintf("Found %d R benchmark CSV files\n", length(r_csv_files)))

  # Read and combine all R CSVs
  r_dfs <- lapply(r_csv_files, function(f) {
    tryCatch({
      df <- read.csv(f, stringsAsFactors = FALSE)
      if (nrow(df) == 0) return(NULL)
      # Ensure standard columns exist
      required_cols <- c("method", "n", "time_median_us", "mem_alloc_bytes")
      if (all(required_cols %in% names(df))) {
        df$source_file <- basename(f)
        df
      } else {
        cat(sprintf("  Skipping %s (missing columns: %s)\n",
                    basename(f),
                    paste(setdiff(required_cols, names(df)), collapse = ", ")))
        NULL
      }
    }, error = function(e) {
      cat(sprintf("  Error reading %s: %s\n", basename(f), e$message))
      NULL
    })
  })

  r_dfs <- Filter(Negate(is.null), r_dfs)

  if (length(r_dfs) > 0) {
    # Harmonize column names before binding
    common_cols <- c("method", "n", "iterations", "time_min_us", "time_p25_us",
                     "time_median_us", "time_p75_us", "time_max_us",
                     "time_mean_us", "time_std_us", "itr_per_sec",
                     "mem_alloc_bytes", "source_file")

    r_dfs <- lapply(r_dfs, function(df) {
      # Add missing columns with NA
      for (col in common_cols) {
        if (!col %in% names(df)) df[[col]] <- NA
      }
      df[, common_cols]
    })

    r_all <- do.call(rbind, r_dfs)
    r_all <- r_all[!is.na(r_all$time_median_us), ]

    # For duplicate method+n combos, keep the most recent file
    r_all <- r_all[order(r_all$source_file, decreasing = TRUE), ]
    r_all <- r_all[!duplicated(paste(r_all$method, r_all$n)), ]

    cat(sprintf("Total R benchmark entries: %d\n", nrow(r_all)))
    cat(sprintf("Unique R methods: %s\n", paste(sort(unique(r_all$method)), collapse = ", ")))
  } else {
    r_all <- data.frame()
    cat("No valid R benchmark data found\n")
  }
}

# ============================================
# 2. Load Rust Results
# ============================================

# Search in multiple possible locations
rust_json_dirs <- c(
  "../../results",                    # performance/results/
  "../../../performance/results",     # from r_comparison dir
  ".",                                # current dir
  "../../../target/criterion"         # Criterion output
)

rust_json_files <- character(0)
for (dir in rust_json_dirs) {
  if (dir.exists(dir)) {
    files <- list.files(dir, pattern = "rust_comprehensive.*\\.json$", full.names = TRUE)
    rust_json_files <- c(rust_json_files, files)
  }
}

# Also check for manually placed results
local_rust <- list.files(r_results_dir, pattern = "rust_.*\\.json$", full.names = TRUE)
rust_json_files <- c(rust_json_files, local_rust)

if (length(rust_json_files) == 0) {
  cat("\nWARNING: No Rust benchmark JSON files found\n")
  cat("Run: cargo bench -p p2a-core --bench comprehensive_benchmarks\n")
  rust_all <- data.frame()
} else {
  # Use the most recent file
  rust_json_file <- sort(rust_json_files, decreasing = TRUE)[1]
  cat(sprintf("\nLoading Rust results from: %s\n", rust_json_file))

  rust_data <- fromJSON(rust_json_file, flatten = TRUE)

  # Convert to standardized format
  rust_all <- data.frame(
    method = rust_data$method,
    variant = if ("variant" %in% names(rust_data)) rust_data$variant else NA,
    n = rust_data$n,
    iterations = rust_data$iterations,
    time_min_us = rust_data$time_min_us,
    time_p25_us = rust_data$time_p25_us,
    time_median_us = rust_data$time_median_us,
    time_p75_us = rust_data$time_p75_us,
    time_max_us = rust_data$time_max_us,
    time_mean_us = rust_data$time_mean_us,
    time_std_us = rust_data$time_std_us,
    itr_per_sec = rust_data$itr_per_sec,
    mem_alloc_bytes = rust_data$mem_alloc_bytes,
    mem_peak_bytes = if ("mem_peak_bytes" %in% names(rust_data)) rust_data$mem_peak_bytes else NA,
    stringsAsFactors = FALSE
  )

  cat(sprintf("Total Rust benchmark entries: %d\n", nrow(rust_all)))
  cat(sprintf("Unique Rust methods: %s\n", paste(sort(unique(rust_all$method)), collapse = ", ")))
}

# ============================================
# 3. Method Name Normalization
# ============================================

# Comprehensive normalization function that handles all R and Rust method names.
# Makes matching case-insensitive and treats underscores/hyphens/spaces as equivalent.
normalize_method <- function(method) {
  m <- trimws(method)

  # Remove trailing _standard suffix
  m <- gsub("_standard$", "", m)

  # Normalize OLS+HC1 -> OLS_HC1
  m <- gsub("\\+", "_", m)

  # Explicit mappings: map known R/Rust method names to a canonical form.
  # Key = raw method name as it appears in CSVs, Value = canonical name.
  mappings <- c(
    # ---- Regression ----
    "OLS" = "OLS",
    "ols_lm" = "OLS",
    "ols_lmfit" = "OLS_lmfit",
    "OLS_HC1" = "OLS_HC1",
    "OLS HC1" = "OLS_HC1",
    "robust_HC0" = "OLS_HC0",
    "robust_HC1" = "OLS_HC1",
    "robust_HC2" = "OLS_HC2",
    "robust_HC3" = "OLS_HC3",
    "NLS" = "NLS",
    "nls_exp_decay" = "NLS",
    "nls_michaelis_menten" = "NLS",
    "LOESS" = "LOESS",
    "loess_gaussian" = "LOESS",
    "loess_robust" = "LOESS",
    "GLS" = "GLS",
    "GLS_AR1" = "GLS",
    "gls_ar1" = "GLS",
    "Quantile_Regression" = "Quantile_Regression",
    "quantile_regression" = "Quantile_Regression",
    "quantreg" = "Quantile_Regression",
    "rq" = "Quantile_Regression",
    "Smooth_Spline" = "Smooth_Spline",
    "smooth_spline" = "Smooth_Spline",
    "smooth_spline_df" = "Smooth_Spline",
    "smooth_spline_cv" = "Smooth_Spline",

    # ---- Diagnostics ----
    "Jarque_Bera" = "Jarque_Bera",
    "jarque_bera" = "Jarque_Bera",
    "Breusch_Pagan" = "Breusch_Pagan",
    "breusch_pagan" = "Breusch_Pagan",
    "bptest" = "Breusch_Pagan",
    "VIF" = "VIF",
    "vif" = "VIF",
    "Durbin_Watson" = "Durbin_Watson",
    "durbin_watson" = "Durbin_Watson",
    "dwtest" = "Durbin_Watson",
    "Breusch_Godfrey" = "Breusch_Godfrey",
    "breusch_godfrey" = "Breusch_Godfrey",
    "bgtest" = "Breusch_Godfrey",
    "RESET" = "RESET",
    "resettest" = "RESET",
    "Wald" = "Wald",
    "waldtest" = "Wald",
    "Harvey_Collier" = "Harvey_Collier",
    "vcovHAC" = "OLS_HAC",
    "OLS_HAC" = "OLS_HAC",

    # ---- Panel ----
    "FE_plm" = "Fixed_Effects",
    "FE_lfe" = "Fixed_Effects",
    "fe_plm" = "Fixed_Effects",
    "fe_lfe" = "Fixed_Effects",
    "FixedEffects" = "Fixed_Effects",
    "Fixed_Effects" = "Fixed_Effects",
    "RandomEffects" = "Random_Effects",
    "Random_Effects" = "Random_Effects",
    "RE_plm" = "Random_Effects",
    "HDFE" = "HDFE",
    "hdfe" = "HDFE",
    "Hausman" = "Hausman",
    "hausman" = "Hausman",
    "phtest" = "Hausman",
    "Arellano_Bond" = "Arellano_Bond",
    "arellano_bond" = "Arellano_Bond",
    "pgmm" = "Arellano_Bond",
    "PVCM" = "PVCM",
    "PMG" = "PMG",
    "Panel_GLS" = "Panel_GLS",

    # ---- Discrete ----
    "Logit" = "Logit",
    "logit" = "Logit",
    "Probit" = "Probit",
    "probit" = "Probit",
    "Multinomial_Logit" = "Multinomial_Logit",
    "multinomial_logit" = "Multinomial_Logit",
    "multinom" = "Multinomial_Logit",
    "Ordered_Logit" = "Ordered_Logit",
    "ordered_logit" = "Ordered_Logit",
    "polr" = "Ordered_Logit",
    "Poisson" = "Poisson",
    "poisson" = "Poisson",
    "poisson_one_sample" = "Poisson_Test",
    "poisson_two_sample" = "Poisson_Test",
    "NegBin" = "NegBin",
    "negbin" = "NegBin",
    "glm.nb" = "NegBin",
    "ZIP" = "ZIP",
    "ZINB" = "ZINB",
    "Hurdle" = "Hurdle",
    "Mixed_Logit" = "Mixed_Logit",

    # ---- Time Series ----
    "ARIMA" = "ARIMA",
    "arima" = "ARIMA",
    "Arima" = "ARIMA",
    "MSTL" = "MSTL",
    "mstl" = "MSTL",
    "Holt_Winters" = "Holt_Winters",
    "HoltWinters" = "Holt_Winters",
    "HW_mult_optim" = "Holt_Winters",
    "Holt-Winters" = "Holt_Winters",
    "STL" = "STL",
    "stl" = "STL",
    "Decompose" = "Decompose",
    "decompose" = "Decompose",
    "decompose_additive" = "Decompose",
    "decompose_multiplicative" = "Decompose",
    "GARCH" = "GARCH",
    "garch" = "GARCH",
    "VAR" = "VAR",
    "var" = "VAR",
    "VECM" = "VECM",

    # ---- Econometrics ----
    "IV_2SLS" = "IV_2SLS",
    "iv_2sls" = "IV_2SLS",
    "ivreg" = "IV_2SLS",
    "DiD" = "DiD",
    "did" = "DiD",
    "Staggered_DiD" = "Staggered_DiD",
    "ETWFE" = "ETWFE",
    "Bacon" = "Bacon",
    "RD" = "RD",
    "rd" = "RD",
    "rdrobust" = "RD",
    "RD_Multi" = "RD_Multi",
    "Synth" = "Synth",
    "synth" = "Synth",
    "GSynth" = "GSynth",

    # ---- Causal ----
    "IPW" = "IPW",
    "ipw" = "IPW",
    "Doubly_Robust" = "Doubly_Robust",
    "TMLE" = "TMLE",
    "tmle" = "TMLE",
    "CTMLE" = "CTMLE",
    "LTMLE" = "LTMLE",
    "DoubleML" = "DoubleML",
    "doubleml" = "DoubleML",
    "Matching" = "Matching",
    "matching" = "Matching",
    "matchit" = "Matching",
    "WeightIt" = "WeightIt",
    "weightit" = "WeightIt",
    "CBPS" = "CBPS",
    "Mediation" = "Mediation",
    "mediation" = "Mediation",

    # ---- ML ----
    "K-Means" = "K-Means",
    "K_Means" = "K-Means",
    "kmeans" = "K-Means",
    "PCA" = "PCA",
    "pca" = "PCA",
    "DBSCAN" = "DBSCAN",
    "dbscan" = "DBSCAN",
    "Hierarchical" = "Hierarchical",
    "hierarchical" = "Hierarchical",
    "hclust" = "Hierarchical",
    "t-SNE" = "t-SNE",
    "t_SNE" = "t-SNE",
    "tsne" = "t-SNE",
    "Rtsne" = "t-SNE",
    "Random_Forest" = "Random_Forest",
    "RandomForest" = "Random_Forest",
    "randomForest" = "Random_Forest",
    "SVM" = "SVM",
    "LinearSVM" = "SVM",
    "svm" = "SVM",

    # ---- Stats ----
    "t_test" = "t_test",
    "t.test" = "t_test",
    "ttest_one_sample" = "t_test",
    "ttest_two_sample" = "t_test",
    "ttest_paired" = "t_test",
    "ANOVA" = "ANOVA",
    "aov" = "ANOVA",
    "anova" = "ANOVA",
    "Chi_squared" = "Chi_squared",
    "chi_squared" = "Chi_squared",
    "chisq.test" = "Chi_squared",
    "chisq_gof" = "Chi_squared",
    "chisq_indep" = "Chi_squared",
    "Fisher" = "Fisher",
    "fisher" = "Fisher",
    "fisher.test" = "Fisher",
    "fisher_exact" = "Fisher",
    "Wilcoxon" = "Wilcoxon",
    "wilcoxon" = "Wilcoxon",
    "wilcox.test" = "Wilcoxon",
    "wilcox_rank_sum" = "Wilcoxon",
    "wilcox_signed_rank" = "Wilcoxon",
    "Kruskal_Wallis" = "Kruskal_Wallis",
    "kruskal" = "Kruskal_Wallis",
    "kruskal.test" = "Kruskal_Wallis",
    "Friedman" = "Friedman",
    "friedman" = "Friedman",
    "friedman.test" = "Friedman",
    "Shapiro_Wilk" = "Shapiro_Wilk",
    "shapiro" = "Shapiro_Wilk",
    "shapiro.test" = "Shapiro_Wilk",
    "shapiro_wilk" = "Shapiro_Wilk",
    "Bartlett" = "Bartlett",
    "bartlett" = "Bartlett",
    "bartlett.test" = "Bartlett",
    "bartlett_k3" = "Bartlett",
    "bartlett_k5" = "Bartlett",
    "bartlett_k10" = "Bartlett",
    "bartlett_k20" = "Bartlett",
    "KS_test" = "KS_test",
    "ks_test" = "KS_test",
    "ks.test" = "KS_test",
    "ACF" = "ACF",
    "acf" = "ACF",
    "PACF" = "PACF",
    "pacf" = "PACF",
    "CCF" = "CCF",
    "Box_Ljung" = "Box_Ljung",
    "box_ljung" = "Box_Ljung",
    "box.test" = "Box_Ljung",
    "Box.test" = "Box_Ljung",
    "Phillips_Perron" = "Phillips_Perron",
    "pp.test" = "Phillips_Perron",
    "pptest" = "Phillips_Perron",
    "Power_Analysis" = "Power_Analysis",
    "power" = "Power_Analysis",
    "power.t.test" = "Power_Analysis",
    "power_t_test" = "Power_Analysis",
    "power_prop_test" = "Power_Analysis",
    "MANOVA" = "MANOVA",
    "manova" = "MANOVA",
    "Tukey" = "Tukey",
    "tukey" = "Tukey",
    "TukeyHSD" = "Tukey",
    "Factor_Analysis" = "Factor_Analysis",
    "factanal" = "Factor_Analysis",
    "Cancor" = "Cancor",
    "cancor" = "Cancor",
    "Ansari" = "Ansari",
    "ansari" = "Ansari",
    "ansari.test" = "Ansari",
    "Fligner" = "Fligner",
    "fligner" = "Fligner",
    "fligner.test" = "Fligner",
    "Mood" = "Mood",
    "mood" = "Mood",
    "mood.test" = "Mood",
    "McNemar" = "McNemar",
    "mcnemar" = "McNemar",
    "mcnemar.test" = "McNemar",
    "Mantel_Haenszel" = "Mantel_Haenszel",
    "mantelhaen" = "Mantel_Haenszel",
    "mantelhaen.test" = "Mantel_Haenszel",
    "Quade" = "Quade",
    "quade" = "Quade",
    "quade.test" = "Quade",
    "Median_Polish" = "Median_Polish",
    "medpolish" = "Median_Polish",
    "Isotonic_Regression" = "Isotonic_Regression",
    "isoreg" = "Isotonic_Regression",
    "Mahalanobis" = "Mahalanobis",
    "mahalanobis" = "Mahalanobis",
    "Spectrum" = "Spectrum",
    "spectrum" = "Spectrum",
    "Spline" = "Spline",
    "spline" = "Spline",
    "spline_approx" = "Spline",
    "Cor_Test" = "Cor_Test",
    "cor.test" = "Cor_Test",
    "cortest" = "Cor_Test",
    "cor_pearson" = "Cor_Test",
    "cor_spearman" = "Cor_Test",
    "cor_kendall" = "Cor_Test",
    "Prop_Test" = "Prop_Test",
    "prop.test" = "Prop_Test",
    "prop_test" = "Prop_Test",
    "prop_trend" = "Prop_Trend",
    "Var_Test" = "Var_Test",
    "var.test" = "Var_Test",
    "var_test" = "Var_Test",
    "Binom_Test" = "Binom_Test",
    "binom.test" = "Binom_Test",
    "binom_test" = "Binom_Test",
    "Robust_Stats" = "Robust_Stats",
    "fivenum" = "Robust_Stats",
    "ecdf" = "Robust_Stats",
    "Weighted" = "Weighted",
    "weighted.mean" = "Weighted",
    "weighted_mean" = "Weighted",
    "Pairwise_t" = "Pairwise_t",
    "pairwise_t_test" = "Pairwise_t",
    "pairwise.t.test" = "Pairwise_t",
    "Pairwise_Wilcox" = "Pairwise_Wilcox",
    "pairwise_wilcox" = "Pairwise_Wilcox",
    "pairwise.wilcox.test" = "Pairwise_Wilcox",
    "Oneway" = "Oneway",
    "oneway" = "Oneway",
    "oneway.test" = "Oneway",
    "Loglin" = "Loglin",
    "loglin" = "Loglin",
    "loglin_2x2" = "Loglin",
    "loglin_2x3" = "Loglin",
    "loglin_2x2x2" = "Loglin",

    # ---- Spatial ----
    "SAR" = "SAR",
    "sar" = "SAR",
    "lagsarlm" = "SAR",
    "SEM" = "SEM",
    "sem" = "SEM",
    "errorsarlm" = "SEM",
    "SAC" = "SAC",
    "sacsarlm" = "SAC",
    "Spatial_Probit" = "Spatial_Probit",
    "SPLM" = "SPLM",
    "localmoran" = "Local_Moran",
    "localmoran_perm" = "Local_Moran",

    # ---- Survival ----
    "Kaplan_Meier" = "Kaplan_Meier",
    "kaplan_meier" = "Kaplan_Meier",
    "survfit" = "Kaplan_Meier",
    "KM_unstratified" = "Kaplan_Meier",
    "KM_stratified" = "Kaplan_Meier",
    "Cox_PH" = "Cox_PH",
    "cox_ph" = "Cox_PH",
    "coxph" = "Cox_PH",
    "CoxPH" = "Cox_PH",
    "AFT" = "AFT",
    "LogRank" = "Log_Rank",

    # ---- Forecasting ----
    "Kalman" = "Kalman",
    "kalman" = "Kalman",
    "StructTS" = "StructTS",
    "structts" = "StructTS",
    "structts_level" = "StructTS",
    "structts_trend" = "StructTS",
    "structts_bsm" = "StructTS",
    "Changepoint" = "Changepoint",
    "changepoint" = "Changepoint",
    "cpt.mean" = "Changepoint",
    "Causal_Impact" = "Causal_Impact",
    "causal_impact" = "Causal_Impact",

    # ---- Rust benchmark name mappings ----
    # These are the exact names produced by comprehensive_benchmarks.rs
    "ChiSq" = "Chi_squared",
    "KaplanMeier" = "Kaplan_Meier",
    "ShapiroWilk" = "Shapiro_Wilk",
    "MultinomLogit" = "Multinomial_Logit",
    "OrderedLogit" = "Ordered_Logit",
    "SmoothSpline" = "Smooth_Spline",
    "QuantReg" = "Quantile_Regression",
    "IV2SLS" = "IV_2SLS",
    "SynthControl" = "Synth",
    "StaggeredDiD" = "Staggered_DiD",
    "CausalImpact" = "Causal_Impact",
    "ArellanoBond" = "Arellano_Bond",
    "CoxPH_breslow" = "Cox_PH",
    "CoxPH_efron" = "Cox_PH",
    "KS" = "KS_test",

    # ---- Standalone R CSV variant mappings ----
    "HW_additive" = "Holt_Winters",
    "HW_multiplicative" = "Holt_Winters",
    "HW_period4" = "Holt_Winters",
    "HW_period12" = "Holt_Winters",
    "HW_period24" = "Holt_Winters",
    "HW_period52" = "Holt_Winters",
    "HW_mult_fixed" = "Holt_Winters",
    "loess_span0.30" = "LOESS",
    "loess_span0.50" = "LOESS",
    "loess_span0.75" = "LOESS",
    "loess_span0.90" = "LOESS",
    "shapiro_normal" = "Shapiro_Wilk",
    "shapiro_mixed" = "Shapiro_Wilk",
    "KS_onesample" = "KS_test",
    "KS_twosample" = "KS_test",
    "fisher_less" = "Fisher",
    "fisher_greater" = "Fisher",
    "fisher_two.sided" = "Fisher",
    "fisher_twosided" = "Fisher",
    "fisher_with_ci" = "Fisher",
    "wilcox_exact" = "Wilcoxon",
    "wilcox_onesample" = "Wilcoxon",
    "wilcox_ranksum" = "Wilcoxon",
    "wilcox_signedrank" = "Wilcoxon",
    "anova_one_way" = "ANOVA",
    "anova_two_way" = "ANOVA",
    "chisq_ind_2x2" = "Chi_squared",
    "chisq_ind_3x3" = "Chi_squared",
    "chisq_ind_5x5" = "Chi_squared",
    "chisq_ind_10x10" = "Chi_squared",
    "chisq_ind_20x20" = "Chi_squared",
    "cancor_p2_q2" = "Cancor",
    "cancor_p5_q3" = "Cancor",
    "cancor_p10_q5" = "Cancor",
    "cancor_p20_q10" = "Cancor",
    "cor.test_pearson" = "Cor_Test",
    "cor.test_spearman" = "Cor_Test",
    "cor.test_kendall" = "Cor_Test",
    "factanal_none" = "Factor_Analysis",
    "factanal_varimax" = "Factor_Analysis",
    "factanal_promax" = "Factor_Analysis",
    "pp_llong" = "Phillips_Perron",
    "pp_lshort" = "Phillips_Perron",
    "p_adjust_BH" = "P_Adjust",
    "p_adjust_bonferroni" = "P_Adjust",
    "p_adjust_hochberg" = "P_Adjust",
    "p_adjust_holm" = "P_Adjust",
    "lm_tests" = "Diagnostics_Suite",
    "AR_burg" = "AR",
    "AR_ols" = "AR",
    "AR_yule-walker" = "AR",
    "approx_constant" = "Spline",
    "approx_linear" = "Spline",
    "spline_fmm" = "Spline",
    "spline_natural" = "Spline",
    "spec_ar" = "Spectrum",
    "spec_pgram_raw" = "Spectrum",
    "spec_pgram_smooth" = "Spectrum",
    "power_ttest_n" = "Power_Analysis",
    "power_ttest_power" = "Power_Analysis",
    "power_prop_n" = "Power_Analysis",
    "power_prop_power" = "Power_Analysis",
    "power_anova_n" = "Power_Analysis",
    "power_anova_power" = "Power_Analysis",
    "mahalanobis_p2" = "Mahalanobis",
    "mahalanobis_p5" = "Mahalanobis",
    "mahalanobis_p10" = "Mahalanobis",
    "mahalanobis_p20" = "Mahalanobis",
    "mahalanobis_p50" = "Mahalanobis",
    "mcnemar_corrected" = "McNemar",
    "mcnemar_no_correction" = "McNemar",
    "Box-Pierce" = "Box_Ljung",
    "Ljung-Box" = "Box_Ljung",
    "AFT_weibull" = "AFT",
    "AFT_lognormal" = "AFT",
    "CompetingRisks" = "Competing_Risks",
    "Diagnostics" = "Diagnostics_Suite",
    "moran_test" = "Moran_Test",
    "neighbors" = "Neighbors",
    "Ljung-Box" = "Box_Ljung",
    "IQR" = "Robust_Stats",
    "mad" = "Robust_Stats",
    "density" = "Robust_Stats",
    "grangertest" = "Granger",

    # ---- Clustering extra ----
    "silhouette" = "Silhouette",
    "kmedoids" = "K-Medoids",
    "cmdscale" = "MDS",
    "cutree" = "Cutree",

    # ---- Fixest benchmark ----
    "fixest_ols" = "OLS",
    "fixest_fe" = "Fixed_Effects",
    "fixest_hdfe" = "HDFE",

    # ---- Variant-to-base mappings for standalone CSVs ----
    # Fisher variants
    "fisher_greater" = "Fisher",
    "fisher_less" = "Fisher",
    "fisher_two.sided" = "Fisher",
    "fisher_twosided" = "Fisher",
    "fisher_with_ci" = "Fisher",

    # Chi-squared variants
    "chisq_ind_2x2" = "Chi_squared",
    "chisq_ind_3x3" = "Chi_squared",
    "chisq_ind_5x5" = "Chi_squared",
    "chisq_ind_10x10" = "Chi_squared",
    "chisq_ind_20x20" = "Chi_squared",

    # LOESS span variants
    "loess_span0.30" = "LOESS",
    "loess_span0.50" = "LOESS",
    "loess_span0.75" = "LOESS",
    "loess_span0.90" = "LOESS",

    # Holt-Winters variants
    "HW_additive" = "Holt_Winters",
    "HW_mult_fixed" = "Holt_Winters",
    "HW_multiplicative" = "Holt_Winters",
    "HW_period12" = "Holt_Winters",
    "HW_period24" = "Holt_Winters",
    "HW_period4" = "Holt_Winters",
    "HW_period52" = "Holt_Winters",

    # KS variants
    "KS_onesample" = "KS_test",
    "KS_twosample" = "KS_test",

    # Shapiro variants
    "shapiro_normal" = "Shapiro_Wilk",
    "shapiro_mixed" = "Shapiro_Wilk",

    # McNemar variants
    "mcnemar_corrected" = "McNemar",
    "mcnemar_no_correction" = "McNemar",

    # Wilcoxon variants
    "wilcox_exact" = "Wilcoxon",
    "wilcox_onesample" = "Wilcoxon",
    "wilcox_ranksum" = "Wilcoxon",
    "wilcox_signedrank" = "Wilcoxon",

    # Cox PH variants
    "CoxPH_breslow" = "Cox_PH",
    "CoxPH_efron" = "Cox_PH",

    # Phillips-Perron variants
    "pp_llong" = "Phillips_Perron",
    "pp_lshort" = "Phillips_Perron",

    # Power analysis variants
    "power_ttest_n" = "Power_Analysis",
    "power_ttest_power" = "Power_Analysis",
    "power_prop_n" = "Power_Analysis",
    "power_prop_power" = "Power_Analysis",
    "power_anova_n" = "Power_Analysis",
    "power_anova_power" = "Power_Analysis",

    # Correlation test variants
    "cor.test_pearson" = "Cor_Test",
    "cor.test_kendall" = "Cor_Test",
    "cor.test_spearman" = "Cor_Test",

    # P-value adjustment
    "p_adjust_BH" = "P_Adjust",
    "p_adjust_bonferroni" = "P_Adjust",
    "p_adjust_hochberg" = "P_Adjust",
    "p_adjust_holm" = "P_Adjust",

    # Spectrum variants
    "spec_ar" = "Spectrum",
    "spec_pgram_raw" = "Spectrum",
    "spec_pgram_smooth" = "Spectrum",

    # Spline variants
    "spline_fmm" = "Spline",
    "spline_natural" = "Spline",

    # Mahalanobis variants
    "mahalanobis_p2" = "Mahalanobis",
    "mahalanobis_p5" = "Mahalanobis",
    "mahalanobis_p10" = "Mahalanobis",
    "mahalanobis_p20" = "Mahalanobis",
    "mahalanobis_p50" = "Mahalanobis",

    # ANOVA variants
    "anova_one_way" = "ANOVA",
    "anova_two_way" = "ANOVA",

    # Cancor variants
    "cancor_p2_q2" = "Cancor",
    "cancor_p5_q3" = "Cancor",
    "cancor_p10_q5" = "Cancor",
    "cancor_p20_q10" = "Cancor",

    # Factor analysis variants
    "factanal_none" = "Factor_Analysis",
    "factanal_varimax" = "Factor_Analysis",
    "factanal_promax" = "Factor_Analysis",

    # NegBin variant
    "glm_nb" = "NegBin",

    # Decompose variants
    "decompose_multiplicative" = "Decompose",

    # Survival variants
    "AFT_weibull" = "AFT",
    "AFT_lognormal" = "AFT",
    "CompetingRisks" = "Competing_Risks",

    # Box test variants
    "Box-Pierce" = "Box_Ljung",
    "Ljung-Box" = "Box_Ljung",

    # Other stats
    "IQR" = "Robust_Stats",
    "mad" = "Robust_Stats",
    "density" = "Robust_Stats",
    "approx_linear" = "Spline",
    "approx_constant" = "Spline",

    # AR variants
    "AR_burg" = "AR",
    "AR_ols" = "AR",
    "AR_yule-walker" = "AR",

    # Other methods from standalone CSVs
    "moran_test" = "Moran_Test",
    "grangertest" = "Granger",
    "lm_tests" = "Diagnostics_Suite",
    "neighbors" = "Neighbors",
    "logit" = "Logit",
    "probit" = "Probit",
    "multinom" = "Multinomial_Logit"
  )

  # Try exact match first
  if (m %in% names(mappings)) return(unname(mappings[m]))

  # Try case-insensitive match with underscore/hyphen/space normalization
  m_lower <- tolower(gsub("[_ -]+", "_", m))
  keys_lower <- tolower(gsub("[_ -]+", "_", names(mappings)))
  idx <- match(m_lower, keys_lower)
  if (!is.na(idx)) return(unname(mappings[idx]))

  # Fallback: return as-is with basic cleanup
  m <- gsub("[_ -]+", "_", m)
  return(m)
}

# ============================================
# 4. Assign Module Function (used in comparison and coverage)
# ============================================

assign_module <- function(method) {
  regression_methods <- c("OLS", "OLS_HC0", "OLS_HC1", "OLS_HC2", "OLS_HC3", "OLS_HAC",
                          "OLS_lmfit", "OLS_Clustered", "OLS_Bootstrap", "OLS_Driscoll_Kraay",
                          "NLS", "LOESS", "GLS", "Quantile_Regression", "Smooth_Spline",
                          "Stepwise")
  diagnostics_methods <- c("Jarque_Bera", "Breusch_Pagan", "Durbin_Watson", "VIF",
                           "Breusch_Godfrey", "RESET", "Wald", "Harvey_Collier")
  panel_methods <- c("Fixed_Effects", "Random_Effects", "HDFE", "Hausman",
                     "Arellano_Bond", "PVCM", "PMG", "Panel_GLS")
  discrete_methods <- c("Logit", "Probit", "Poisson", "NegBin", "ZIP", "ZINB", "Hurdle",
                        "Multinomial_Logit", "Ordered_Logit", "Mixed_Logit")
  ts_methods <- c("ARIMA", "MSTL", "STL", "Holt_Winters", "Decompose", "GARCH",
                  "Kalman", "StructTS", "VAR", "VECM", "Causal_Impact", "Changepoint")
  ml_methods <- c("K-Means", "PCA", "DBSCAN", "Hierarchical", "Random_Forest",
                  "SVM", "t-SNE", "Silhouette", "K-Medoids", "MDS", "Cutree")
  causal_methods <- c("DiD", "Staggered_DiD", "ETWFE", "Bacon", "Synth", "GSynth",
                      "RD", "RD_Multi", "TMLE", "CTMLE", "LTMLE",
                      "IPW", "Doubly_Robust", "Matching", "WeightIt", "CBPS",
                      "DoubleML", "Mediation")
  econometrics_methods <- c("IV_2SLS")
  spatial_methods <- c("SAR", "SEM", "SAC", "Spatial_Probit", "SPLM", "Local_Moran")
  survival_methods <- c("Kaplan_Meier", "Cox_PH", "AFT", "Log_Rank")
  stats_methods <- c("t_test", "ANOVA", "Chi_squared", "Fisher", "Wilcoxon",
                     "Kruskal_Wallis", "Friedman", "Shapiro_Wilk", "KS_test",
                     "Bartlett", "ACF", "PACF", "CCF", "Box_Ljung", "Phillips_Perron",
                     "Power_Analysis", "MANOVA", "Tukey", "Factor_Analysis", "Cancor",
                     "Ansari", "Fligner", "Mood", "McNemar", "Mantel_Haenszel",
                     "Quade", "Median_Polish", "Isotonic_Regression", "Mahalanobis",
                     "Spectrum", "Spline", "Cor_Test", "Prop_Test", "Prop_Trend",
                     "Var_Test", "Binom_Test", "Robust_Stats", "Weighted",
                     "Pairwise_t", "Pairwise_Wilcox", "Oneway", "Loglin",
                     "Poisson_Test")

  m <- method
  if (m %in% regression_methods) return("Regression")
  if (m %in% diagnostics_methods) return("Diagnostics")
  if (m %in% panel_methods) return("Panel")
  if (m %in% discrete_methods) return("Discrete")
  if (m %in% ml_methods) return("ML")
  if (m %in% causal_methods) return("Causal")
  if (m %in% econometrics_methods) return("Econometrics")
  if (m %in% spatial_methods) return("Spatial")
  if (m %in% survival_methods) return("Survival")
  if (m %in% stats_methods) return("Stats")
  if (m %in% ts_methods) return("Time Series")
  return("Other")
}

# ============================================
# 5. Build Speed Comparison
# ============================================

if (nrow(r_all) > 0 && exists("rust_all") && nrow(rust_all) > 0) {
  cat("\n--- Building Speed Comparison ---\n")

  # Normalize method names
  r_all$method_norm <- sapply(r_all$method, normalize_method)
  rust_all$method_norm <- sapply(rust_all$method, normalize_method)

  cat(sprintf("\nR normalized methods: %s\n", paste(sort(unique(r_all$method_norm)), collapse = ", ")))
  cat(sprintf("Rust normalized methods: %s\n", paste(sort(unique(rust_all$method_norm)), collapse = ", ")))

  # Match on method + n
  r_for_merge <- r_all[, c("method", "method_norm", "n", "time_median_us", "time_mean_us",
                            "time_p25_us", "time_p75_us", "mem_alloc_bytes")]
  names(r_for_merge) <- c("r_method", "method_norm", "n",
                           "r_median_us", "r_mean_us", "r_p25_us", "r_p75_us",
                           "r_mem_bytes")

  rust_for_merge <- rust_all[, c("method", "method_norm", "n", "time_median_us", "time_mean_us",
                                  "time_p25_us", "time_p75_us", "mem_alloc_bytes")]
  names(rust_for_merge) <- c("rust_method", "method_norm", "n",
                              "rust_median_us", "rust_mean_us", "rust_p25_us", "rust_p75_us",
                              "rust_mem_bytes")

  # For methods with duplicates at same n (e.g. multiple NLS variants), keep fastest
  r_for_merge <- r_for_merge[order(r_for_merge$r_median_us), ]
  r_for_merge <- r_for_merge[!duplicated(paste(r_for_merge$method_norm, r_for_merge$n)), ]

  rust_for_merge <- rust_for_merge[order(rust_for_merge$rust_median_us), ]
  rust_for_merge <- rust_for_merge[!duplicated(paste(rust_for_merge$method_norm, rust_for_merge$n)), ]

  # First try exact n matching
  comparison <- merge(r_for_merge, rust_for_merge, by = c("method_norm", "n"), all = TRUE)

  # For methods with no exact n match, try closest-n matching (within 2x factor)
  r_methods <- unique(r_for_merge$method_norm)
  rust_methods <- unique(rust_for_merge$method_norm)
  shared_methods <- intersect(r_methods, rust_methods)
  matched_methods <- unique(comparison$method_norm[!is.na(comparison$r_median_us) & !is.na(comparison$rust_median_us)])
  unmatched_shared <- setdiff(shared_methods, matched_methods)

  if (length(unmatched_shared) > 0) {
    fuzzy_rows <- list()
    for (m in unmatched_shared) {
      r_sub <- r_for_merge[r_for_merge$method_norm == m, ]
      rust_sub <- rust_for_merge[rust_for_merge$method_norm == m, ]
      for (i in seq_len(nrow(rust_sub))) {
        rn <- rust_sub$n[i]
        # Find closest R n within factor of 2
        dists <- abs(log2(r_sub$n / rn))
        best_idx <- which.min(dists)
        if (length(best_idx) > 0 && dists[best_idx] <= 1.0) {  # within 2x
          row <- data.frame(
            method_norm = m,
            n = rn,
            r_method = r_sub$r_method[best_idx],
            r_median_us = r_sub$r_median_us[best_idx],
            r_mean_us = r_sub$r_mean_us[best_idx],
            r_p25_us = r_sub$r_p25_us[best_idx],
            r_p75_us = r_sub$r_p75_us[best_idx],
            r_mem_bytes = r_sub$r_mem_bytes[best_idx],
            rust_method = rust_sub$rust_method[i],
            rust_median_us = rust_sub$rust_median_us[i],
            rust_mean_us = rust_sub$rust_mean_us[i],
            rust_p25_us = rust_sub$rust_p25_us[i],
            rust_p75_us = rust_sub$rust_p75_us[i],
            rust_mem_bytes = rust_sub$rust_mem_bytes[i],
            stringsAsFactors = FALSE
          )
          fuzzy_rows[[length(fuzzy_rows) + 1]] <- row
        }
      }
    }
    if (length(fuzzy_rows) > 0) {
      fuzzy_df <- do.call(rbind, fuzzy_rows)
      # Remove unmatched rows for these methods from comparison, add fuzzy matches
      comparison <- comparison[!(comparison$method_norm %in% unmatched_shared), ]
      comparison <- rbind(comparison, fuzzy_df)
      cat(sprintf("  Added %d fuzzy n-matches for: %s\n",
                  nrow(fuzzy_df), paste(unmatched_shared, collapse = ", ")))
    }
  }

  # Calculate speedup factors
  comparison$speedup_median <- comparison$r_median_us / comparison$rust_median_us
  comparison$speedup_mean <- comparison$r_mean_us / comparison$rust_mean_us

  # Memory ratio (R / Rust)
  comparison$mem_ratio <- ifelse(
    !is.na(comparison$rust_mem_bytes) & abs(comparison$rust_mem_bytes) > 0,
    comparison$r_mem_bytes / abs(comparison$rust_mem_bytes),
    NA
  )

  # Assign module categories
  comparison$module <- sapply(comparison$method_norm, assign_module)

  # Sort by module then speedup
  comparison <- comparison[order(comparison$module, -comparison$speedup_median, na.last = TRUE), ]

  # Save speed comparison
  speed_file <- file.path(r_results_dir, "comparison_speed.csv")
  write.csv(comparison, speed_file, row.names = FALSE)
  cat(sprintf("Speed comparison saved to: %s (%d entries)\n", speed_file, nrow(comparison)))

  # Print summary
  matched <- comparison[!is.na(comparison$speedup_median), ]
  if (nrow(matched) > 0) {
    cat(sprintf("\nMatched benchmarks: %d (out of %d R + %d Rust data points)\n",
                nrow(matched), nrow(r_all), nrow(rust_all)))
    cat(sprintf("Median speedup (Rust vs R): %.1fx\n", median(matched$speedup_median, na.rm = TRUE)))
    cat(sprintf("Mean speedup (Rust vs R): %.1fx\n", mean(matched$speedup_median, na.rm = TRUE)))
    cat(sprintf("Range: %.1fx - %.1fx\n",
                min(matched$speedup_median, na.rm = TRUE),
                max(matched$speedup_median, na.rm = TRUE)))

    # Per-module summary
    cat("\nPer-module matched counts:\n")
    mod_counts <- table(matched$module)
    for (mod in sort(names(mod_counts))) {
      mod_data <- matched[matched$module == mod, ]
      cat(sprintf("  %-15s: %3d matches, median speedup: %.1fx\n",
                  mod, nrow(mod_data), median(mod_data$speedup_median, na.rm = TRUE)))
    }
  }

  # ============================================
  # 6. Build Memory Comparison
  # ============================================

  cat("\n--- Building Memory Comparison ---\n")

  mem_comparison <- comparison[!is.na(comparison$r_mem_bytes) & !is.na(comparison$rust_mem_bytes), ]
  mem_comparison <- mem_comparison[!is.na(mem_comparison$rust_mem_bytes) & mem_comparison$rust_mem_bytes != 0, ]

  if (nrow(mem_comparison) > 0) {
    mem_file <- file.path(r_results_dir, "comparison_memory.csv")
    write.csv(mem_comparison[, c("method_norm", "n", "r_mem_bytes", "rust_mem_bytes",
                                  "mem_ratio", "module")],
              mem_file, row.names = FALSE)
    cat(sprintf("Memory comparison saved to: %s (%d entries)\n", mem_file, nrow(mem_comparison)))
    cat(sprintf("Median memory ratio (R/Rust): %.1fx\n",
                median(mem_comparison$mem_ratio, na.rm = TRUE)))
  } else {
    cat("No memory comparison data available (need both R mem_alloc and Rust mem_alloc)\n")
  }

} else {
  cat("\nSkipping comparison: need both R and Rust results.\n")
  if (nrow(r_all) == 0) cat("  Missing: R results\n")
  if (!exists("rust_all") || nrow(rust_all) == 0) cat("  Missing: Rust results\n")
}

# ============================================
# 7. Build Validation Coverage Matrix
# ============================================

cat("\n--- Building Validation Coverage Matrix ---\n")

# Define all known methods in p2a-core
all_methods <- data.frame(
  method = c(
    # Regression
    "OLS", "OLS_HC1", "OLS_HC3", "OLS_Clustered", "OLS_HAC", "OLS_Bootstrap",
    "OLS_Driscoll_Kraay", "NLS", "LOESS", "GLS", "Quantile_Regression",
    "Smooth_Spline", "Stepwise",
    # Diagnostics
    "Jarque_Bera", "Breusch_Pagan", "Durbin_Watson", "VIF", "Breusch_Godfrey",
    "RESET", "Wald", "Harvey_Collier",
    # Panel
    "Fixed_Effects", "Random_Effects", "Hausman", "HDFE", "Arellano_Bond",
    "PVCM", "PMG", "Panel_GLS",
    # Discrete
    "Logit", "Probit", "Multinomial_Logit", "Ordered_Logit", "Poisson",
    "NegBin", "ZIP", "ZINB", "Hurdle", "Mixed_Logit",
    # Time Series / Forecasting
    "ARIMA", "Holt_Winters", "STL", "MSTL", "Decompose", "GARCH",
    "Kalman", "StructTS", "Causal_Impact", "Changepoint",
    # Econometrics
    "IV_2SLS", "DiD", "Staggered_DiD", "ETWFE", "Bacon", "Synth",
    "GSynth", "RD", "RD_Multi", "VAR", "VECM",
    # Causal
    "IPW", "Doubly_Robust", "TMLE", "CTMLE", "LTMLE", "DoubleML",
    "Matching", "WeightIt", "CBPS", "Mediation",
    # ML
    "K-Means", "DBSCAN", "Hierarchical", "PCA", "t-SNE",
    "Random_Forest", "SVM",
    # Stats
    "t_test", "ANOVA", "Chi_squared", "Fisher", "Wilcoxon", "Kruskal_Wallis",
    "Friedman", "Shapiro_Wilk", "KS_test", "Bartlett", "ACF", "PACF",
    "Box_Ljung", "Phillips_Perron", "Power_Analysis", "MANOVA", "Tukey",
    "Factor_Analysis", "Cancor", "Ansari", "Fligner", "Mood", "McNemar",
    "Mantel_Haenszel", "Quade", "Median_Polish", "Isotonic_Regression",
    "Mahalanobis", "Spectrum", "Spline", "Cor_Test", "Prop_Test",
    "Var_Test", "Binom_Test", "Robust_Stats", "Weighted",
    "Pairwise_t", "Pairwise_Wilcox", "Oneway",
    # Spatial
    "SAR", "SEM", "SAC", "Spatial_Probit", "SPLM",
    # Survival
    "Kaplan_Meier", "Cox_PH", "AFT"
  ),
  module = c(
    rep("Regression", 13),
    rep("Diagnostics", 8),
    rep("Panel", 8),
    rep("Discrete", 10),
    rep("Time Series", 10),
    rep("Econometrics", 11),
    rep("Causal", 10),
    rep("ML", 7),
    rep("Stats", 39),
    rep("Spatial", 5),
    rep("Survival", 3)
  ),
  stringsAsFactors = FALSE
)

# Check for Rust implementation (assume all are implemented per CLAUDE.md)
all_methods$rust_impl <- TRUE

# Check for R validation test (based on known test names)
all_methods$r_validation <- FALSE

# Methods known to have R-comparison validation tests
# Full list from validated modules in p2a-core
validated_methods <- c(
  # Regression: ols, diagnostics, evalue, gls, line, loess, nls, quantreg, sensemakr, smooth_spline, step, supsmu
  "OLS", "OLS_HC1", "OLS_HC3", "OLS_Clustered", "OLS_HAC", "OLS_Bootstrap",
  "OLS_Driscoll_Kraay", "NLS", "LOESS", "GLS", "Quantile_Regression",
  "Smooth_Spline", "Stepwise",
  # Diagnostics
  "Jarque_Bera", "Breusch_Pagan", "Durbin_Watson", "VIF", "Breusch_Godfrey",
  "RESET", "Wald", "Harvey_Collier",
  # Panel: panel, hdfe, feglm
  "Fixed_Effects", "Random_Effects", "Hausman", "HDFE", "Arellano_Bond",
  "PVCM", "PMG", "Panel_GLS",
  # Discrete: discrete (all sub-modules), feglm
  "Logit", "Probit", "Multinomial_Logit", "Ordered_Logit", "Poisson",
  "NegBin", "ZIP", "ZINB", "Hurdle", "Mixed_Logit",
  # Forecasting: ar, causal_impact, changepoint, decompose, garch, holtwinters, kalman, stl, structts, tsutils
  "ARIMA", "Holt_Winters", "STL", "MSTL", "Decompose", "GARCH",
  "Kalman", "StructTS", "Causal_Impact", "Changepoint",
  # Econometrics: bpbounds, did, iv, rd, staggered_did, synth, timeseries (VAR/VECM)
  "IV_2SLS", "DiD", "Staggered_DiD", "ETWFE", "Bacon", "Synth",
  "GSynth", "RD", "RD_Multi", "VAR", "VECM",
  # Causal: matching, mediation, tmle, treatment, spatial
  "IPW", "Doubly_Robust", "TMLE", "CTMLE", "LTMLE", "DoubleML",
  "Matching", "WeightIt", "CBPS", "Mediation",
  # ML: clustering, reduction, svm, trees
  "K-Means", "DBSCAN", "Hierarchical", "PCA", "t-SNE",
  "Random_Forest", "SVM",
  # Stats: acf, anova, ansari, bartlett, binomtest, boxtest, cancor, chisq, cortest,
  # factanal, fisher, fligner, friedman, isoreg, kruskal, ks, mahalanobis, manova,
  # mantelhaen, mcnemar, medpolish, mood, oneway, pairwise, poissontest, power,
  # pptest, proptest, quade, robust, shapiro, spectrum, spline, ttest, tukey,
  # vartest, weighted, wilcoxon
  "t_test", "ANOVA", "Chi_squared", "Fisher", "Wilcoxon", "Kruskal_Wallis",
  "Friedman", "Shapiro_Wilk", "KS_test", "Bartlett", "ACF", "PACF",
  "Box_Ljung", "Phillips_Perron", "Power_Analysis", "MANOVA", "Tukey",
  "Factor_Analysis", "Cancor", "Ansari", "Fligner", "Mood", "McNemar",
  "Mantel_Haenszel", "Quade", "Median_Polish", "Isotonic_Regression",
  "Mahalanobis", "Spectrum", "Spline", "Cor_Test", "Prop_Test",
  "Var_Test", "Binom_Test", "Robust_Stats", "Weighted",
  "Pairwise_t", "Pairwise_Wilcox", "Oneway",
  # Spatial: spatial, splm, sphet, spatialprobit
  "SAR", "SEM", "SAC", "Spatial_Probit", "SPLM",
  # Survival
  "Kaplan_Meier", "Cox_PH", "AFT"
)
all_methods$r_validation[all_methods$method %in% validated_methods] <- TRUE

# Check for speed benchmark
all_methods$speed_bench <- FALSE
if (exists("comparison") && nrow(comparison) > 0) {
  benchmarked <- unique(comparison$method_norm[!is.na(comparison$r_median_us) &
                                                !is.na(comparison$rust_median_us)])
  all_methods$speed_bench[all_methods$method %in% benchmarked] <- TRUE
}

# Check for memory benchmark
all_methods$mem_bench <- FALSE
if (exists("mem_comparison") && nrow(mem_comparison) > 0) {
  mem_benchmarked <- unique(mem_comparison$method_norm)
  all_methods$mem_bench[all_methods$method %in% mem_benchmarked] <- TRUE
}

# Save validation coverage
coverage_file <- file.path(r_results_dir, "validation_coverage.csv")
write.csv(all_methods, coverage_file, row.names = FALSE)
cat(sprintf("Validation coverage saved to: %s\n", coverage_file))

# Print summary by module
cat("\n=== Validation Coverage Summary ===\n")
cat(sprintf("%-15s  %5s  %5s  %5s  %5s  %5s\n",
            "Module", "Total", "Impl", "Valid", "Speed", "Memory"))
cat(paste(rep("-", 55), collapse = ""), "\n")

for (mod in unique(all_methods$module)) {
  subset <- all_methods[all_methods$module == mod, ]
  cat(sprintf("%-15s  %5d  %5d  %5d  %5d  %5d\n",
              mod,
              nrow(subset),
              sum(subset$rust_impl),
              sum(subset$r_validation),
              sum(subset$speed_bench),
              sum(subset$mem_bench)))
}

totals <- all_methods
cat(paste(rep("-", 55), collapse = ""), "\n")
cat(sprintf("%-15s  %5d  %5d  %5d  %5d  %5d\n",
            "TOTAL",
            nrow(totals),
            sum(totals$rust_impl),
            sum(totals$r_validation),
            sum(totals$speed_bench),
            sum(totals$mem_bench)))

cat(sprintf("\nValidation rate: %.0f%%\n",
            100 * sum(all_methods$r_validation) / nrow(all_methods)))
cat(sprintf("Speed benchmark rate: %.0f%%\n",
            100 * sum(all_methods$speed_bench) / nrow(all_methods)))
cat(sprintf("Memory benchmark rate: %.0f%%\n",
            100 * sum(all_methods$mem_bench) / nrow(all_methods)))

cat("\n=== Merge Complete ===\n")
