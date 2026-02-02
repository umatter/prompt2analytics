//! Tool Registry - Metadata for all MCP tools
//!
//! This module provides structured metadata about all analytics tools,
//! enabling programmatic discovery and better LLM understanding.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Category of an analytics tool
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolCategory {
    /// Data loading, management, and export
    Data,
    /// Data quality profiling and cleaning
    Cleaning,
    /// Data transformation and munging
    Munging,
    /// Descriptive statistics and correlation
    Descriptive,
    /// Hypothesis tests and power analysis
    Statistics,
    /// OLS, GLS, NLS, quantile regression
    Regression,
    /// Panel data models (FE, RE, GMM)
    Panel,
    /// Instrumental variables and 2SLS
    IV,
    /// Difference-in-differences, synthetic control
    DiD,
    /// Regression discontinuity
    RD,
    /// Propensity score matching and weighting
    Matching,
    /// IPW, doubly robust, TMLE
    Treatment,
    /// Mediation analysis
    Mediation,
    /// Logit, probit, multinomial, count models
    Discrete,
    /// ARIMA, VAR, GARCH, time series utilities
    TimeSeries,
    /// Spatial regression and diagnostics
    Spatial,
    /// Survival analysis (Kaplan-Meier, Cox, AFT)
    Survival,
    /// Clustering, PCA, t-SNE, Random Forest
    MachineLearning,
    /// Plots and charts (static and interactive)
    Visualization,
    /// SQLite and DuckDB queries
    Database,
    /// Seed, reports, session management
    Utility,
}

impl ToolCategory {
    /// Get a human-readable description of the category
    pub fn description(&self) -> &'static str {
        match self {
            Self::Data => "Data loading, listing, describing, and export",
            Self::Cleaning => "Data quality profiling, cleaning sessions, rollback",
            Self::Munging => "Data transformation: filter, join, pivot, mutate",
            Self::Descriptive => "Descriptive statistics, correlation, ANOVA",
            Self::Statistics => "Hypothesis tests, power analysis, multivariate",
            Self::Regression => "OLS, GLS, NLS, quantile, robust standard errors",
            Self::Panel => "Fixed effects, random effects, Hausman, GMM, HDFE",
            Self::IV => "Instrumental variables, 2SLS, Sargan, bounds",
            Self::DiD => "Difference-in-differences, staggered DiD, synthetic control",
            Self::RD => "Regression discontinuity (sharp, fuzzy, multi-cutoff)",
            Self::Matching => "Propensity score matching, CEM, nearest neighbor",
            Self::Treatment => "IPW, doubly robust, TMLE, CBPS, entropy balancing",
            Self::Mediation => "Causal mediation, natural effects",
            Self::Discrete => "Logit, probit, multinomial, ordered, count models",
            Self::TimeSeries => "ARIMA, VAR, GARCH, Kalman, changepoint detection",
            Self::Spatial => "SAR, SEM, Moran's I, spatial weights, panel spatial",
            Self::Survival => "Kaplan-Meier, Cox PH, AFT, competing risks",
            Self::MachineLearning => "K-means, DBSCAN, PCA, t-SNE, Random Forest, SVM",
            Self::Visualization => "Histograms, scatter plots, heatmaps, interactive charts",
            Self::Database => "SQLite and DuckDB queries, schema inspection",
            Self::Utility => "Random seed, reports, session export/import",
        }
    }

    /// Get all categories
    pub fn all() -> &'static [ToolCategory] {
        use ToolCategory::*;
        &[
            Data, Cleaning, Munging, Descriptive, Statistics, Regression, Panel,
            IV, DiD, RD, Matching, Treatment, Mediation, Discrete, TimeSeries,
            Spatial, Survival, MachineLearning, Visualization, Database, Utility,
        ]
    }
}

/// Metadata for a single tool
#[derive(Debug, Clone, Serialize)]
pub struct ToolInfo {
    /// Tool name (MCP tool identifier)
    pub name: &'static str,
    /// Primary category
    pub category: ToolCategory,
    /// Brief description
    pub description: &'static str,
    /// Related tools (for discovery)
    #[serde(serialize_with = "serialize_static_strs")]
    pub related: &'static [&'static str],
    /// R equivalent function(s) if applicable
    pub r_equivalent: Option<&'static str>,
}

fn serialize_static_strs<S>(value: &&'static [&'static str], serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(value.len()))?;
    for s in *value {
        seq.serialize_element(s)?;
    }
    seq.end()
}

/// Get the complete tool registry
pub fn get_registry() -> Vec<ToolInfo> {
    vec![
        // ══════════════════════════════════════════════════════════════════════════
        // Data Management
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "list_datasets",
            category: ToolCategory::Data,
            description: "List all loaded datasets with dimensions and column types",
            related: &["load_dataset", "describe_dataset"],
            r_equivalent: Some("ls()"),
        },
        ToolInfo {
            name: "load_dataset",
            category: ToolCategory::Data,
            description: "Load data from CSV, Parquet, Excel, Stata, or SAS files",
            related: &["create_dataset", "list_datasets"],
            r_equivalent: Some("read.csv, read_parquet, read_dta"),
        },
        ToolInfo {
            name: "export_dataset",
            category: ToolCategory::Data,
            description: "Export dataset to CSV, Parquet, or JSON format",
            related: &["load_dataset"],
            r_equivalent: Some("write.csv, write_parquet"),
        },
        ToolInfo {
            name: "create_dataset",
            category: ToolCategory::Data,
            description: "Create dataset from inline CSV content",
            related: &["load_dataset", "generate_random_data"],
            r_equivalent: None,
        },
        ToolInfo {
            name: "describe_dataset",
            category: ToolCategory::Data,
            description: "Get descriptive statistics for all columns",
            related: &["list_datasets", "head_dataset"],
            r_equivalent: Some("summary()"),
        },
        ToolInfo {
            name: "head_dataset",
            category: ToolCategory::Data,
            description: "Preview first N rows of a dataset",
            related: &["describe_dataset"],
            r_equivalent: Some("head()"),
        },
        ToolInfo {
            name: "compare_datasets",
            category: ToolCategory::Data,
            description: "Compare two datasets for differences",
            related: &["describe_dataset"],
            r_equivalent: Some("all.equal()"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Data Cleaning
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "data_quality_profile",
            category: ToolCategory::Cleaning,
            description: "Generate quality profile: missing values, outliers, duplicates",
            related: &["suggest_cleaning", "preview_cleaning"],
            r_equivalent: None,
        },
        ToolInfo {
            name: "suggest_cleaning",
            category: ToolCategory::Cleaning,
            description: "Get AI-powered cleaning suggestions with priorities",
            related: &["data_quality_profile", "preview_cleaning"],
            r_equivalent: None,
        },
        ToolInfo {
            name: "preview_cleaning",
            category: ToolCategory::Cleaning,
            description: "Preview cleaning operation without applying",
            related: &["verify_cleaning", "cleaning_session_apply"],
            r_equivalent: None,
        },
        ToolInfo {
            name: "verify_cleaning",
            category: ToolCategory::Cleaning,
            description: "Verify cleaning results match expectations",
            related: &["preview_cleaning"],
            r_equivalent: None,
        },
        ToolInfo {
            name: "cleaning_session_start",
            category: ToolCategory::Cleaning,
            description: "Start a cleaning session with checkpoints",
            related: &["cleaning_session_apply", "cleaning_rollback"],
            r_equivalent: None,
        },
        ToolInfo {
            name: "cleaning_session_apply",
            category: ToolCategory::Cleaning,
            description: "Apply a cleaning operation within session",
            related: &["cleaning_session_start", "cleaning_rollback"],
            r_equivalent: None,
        },
        ToolInfo {
            name: "cleaning_rollback",
            category: ToolCategory::Cleaning,
            description: "Rollback to previous checkpoint",
            related: &["cleaning_session_start"],
            r_equivalent: None,
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Descriptive Statistics
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "compute_correlation",
            category: ToolCategory::Descriptive,
            description: "Compute correlation matrix for numeric columns",
            related: &["describe_dataset", "viz_heatmap"],
            r_equivalent: Some("cor()"),
        },
        ToolInfo {
            name: "stats_fivenum",
            category: ToolCategory::Descriptive,
            description: "Tukey's five-number summary (min, Q1, median, Q3, max)",
            related: &["stats_iqr", "viz_boxplot"],
            r_equivalent: Some("fivenum()"),
        },
        ToolInfo {
            name: "stats_iqr",
            category: ToolCategory::Descriptive,
            description: "Interquartile range (Q3 - Q1)",
            related: &["stats_fivenum", "stats_mad"],
            r_equivalent: Some("IQR()"),
        },
        ToolInfo {
            name: "stats_mad",
            category: ToolCategory::Descriptive,
            description: "Median absolute deviation (robust scale measure)",
            related: &["stats_iqr"],
            r_equivalent: Some("mad()"),
        },
        ToolInfo {
            name: "stats_ecdf",
            category: ToolCategory::Descriptive,
            description: "Empirical cumulative distribution function",
            related: &["stats_density"],
            r_equivalent: Some("ecdf()"),
        },
        ToolInfo {
            name: "stats_density",
            category: ToolCategory::Descriptive,
            description: "Kernel density estimation",
            related: &["stats_ecdf", "viz_histogram"],
            r_equivalent: Some("density()"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Data Munging
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "munge_filter",
            category: ToolCategory::Munging,
            description: "Filter rows based on conditions",
            related: &["munge_select", "munge_sort"],
            r_equivalent: Some("dplyr::filter()"),
        },
        ToolInfo {
            name: "munge_select",
            category: ToolCategory::Munging,
            description: "Select specific columns",
            related: &["munge_drop_columns", "munge_rename"],
            r_equivalent: Some("dplyr::select()"),
        },
        ToolInfo {
            name: "munge_mutate",
            category: ToolCategory::Munging,
            description: "Create or modify columns with expressions",
            related: &["munge_standardize", "munge_bin"],
            r_equivalent: Some("dplyr::mutate()"),
        },
        ToolInfo {
            name: "munge_join",
            category: ToolCategory::Munging,
            description: "Join two datasets (left, inner, full)",
            related: &["munge_concat"],
            r_equivalent: Some("dplyr::left_join()"),
        },
        ToolInfo {
            name: "munge_pivot",
            category: ToolCategory::Munging,
            description: "Reshape from long to wide format",
            related: &["munge_melt"],
            r_equivalent: Some("tidyr::pivot_wider()"),
        },
        ToolInfo {
            name: "munge_melt",
            category: ToolCategory::Munging,
            description: "Reshape from wide to long format",
            related: &["munge_pivot"],
            r_equivalent: Some("tidyr::pivot_longer()"),
        },
        ToolInfo {
            name: "munge_group_by",
            category: ToolCategory::Munging,
            description: "Group data and compute aggregations",
            related: &["munge_value_counts"],
            r_equivalent: Some("dplyr::group_by() %>% summarize()"),
        },
        ToolInfo {
            name: "munge_drop_na",
            category: ToolCategory::Munging,
            description: "Remove rows with missing values",
            related: &["munge_fill_na"],
            r_equivalent: Some("tidyr::drop_na()"),
        },
        ToolInfo {
            name: "munge_fill_na",
            category: ToolCategory::Munging,
            description: "Fill missing values with specified strategy",
            related: &["munge_drop_na"],
            r_equivalent: Some("tidyr::fill()"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Regression
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "regression_ols",
            category: ToolCategory::Regression,
            description: "OLS regression with optional robust standard errors (HC0-HC3)",
            related: &["regression_clustered", "regression_diagnostics"],
            r_equivalent: Some("lm(), sandwich::vcovHC()"),
        },
        ToolInfo {
            name: "regression_clustered",
            category: ToolCategory::Regression,
            description: "OLS with clustered standard errors",
            related: &["regression_ols", "regression_hac"],
            r_equivalent: Some("sandwich::vcovCL()"),
        },
        ToolInfo {
            name: "regression_hac",
            category: ToolCategory::Regression,
            description: "HAC (Newey-West) standard errors for time series",
            related: &["regression_ols", "regression_driscoll_kraay"],
            r_equivalent: Some("sandwich::vcovHAC()"),
        },
        ToolInfo {
            name: "regression_gls",
            category: ToolCategory::Regression,
            description: "Generalized least squares with AR1 or custom correlation",
            related: &["regression_ols"],
            r_equivalent: Some("nlme::gls()"),
        },
        ToolInfo {
            name: "regression_nls",
            category: ToolCategory::Regression,
            description: "Nonlinear least squares (Levenberg-Marquardt)",
            related: &["regression_ols"],
            r_equivalent: Some("nls()"),
        },
        ToolInfo {
            name: "regression_quantreg",
            category: ToolCategory::Regression,
            description: "Quantile regression (interior point/simplex)",
            related: &["regression_ols"],
            r_equivalent: Some("quantreg::rq()"),
        },
        ToolInfo {
            name: "regression_loess",
            category: ToolCategory::Regression,
            description: "Local polynomial regression (LOESS/LOWESS)",
            related: &["regression_smooth_spline"],
            r_equivalent: Some("loess()"),
        },
        ToolInfo {
            name: "regression_diagnostics",
            category: ToolCategory::Regression,
            description: "Regression diagnostics: VIF, Breusch-Pagan, Durbin-Watson",
            related: &["regression_ols", "regression_bgtest"],
            r_equivalent: Some("car::vif(), lmtest::bptest()"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Panel Data
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "panel_fixed_effects",
            category: ToolCategory::Panel,
            description: "Fixed effects (within) estimator",
            related: &["panel_random_effects", "hausman_test"],
            r_equivalent: Some("plm::plm(model='within')"),
        },
        ToolInfo {
            name: "panel_random_effects",
            category: ToolCategory::Panel,
            description: "Random effects (GLS) estimator",
            related: &["panel_fixed_effects", "hausman_test"],
            r_equivalent: Some("plm::plm(model='random')"),
        },
        ToolInfo {
            name: "hausman_test",
            category: ToolCategory::Panel,
            description: "Hausman specification test (FE vs RE)",
            related: &["panel_fixed_effects", "panel_random_effects"],
            r_equivalent: Some("plm::phtest()"),
        },
        ToolInfo {
            name: "panel_hdfe",
            category: ToolCategory::Panel,
            description: "High-dimensional fixed effects (multiple FE)",
            related: &["panel_fixed_effects", "feglm"],
            r_equivalent: Some("fixest::feols()"),
        },
        ToolInfo {
            name: "panel_gmm",
            category: ToolCategory::Panel,
            description: "Arellano-Bond dynamic panel GMM",
            related: &["panel_fixed_effects", "gmm_iv"],
            r_equivalent: Some("plm::pgmm()"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Instrumental Variables
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "iv_2sls",
            category: ToolCategory::IV,
            description: "Two-stage least squares IV regression",
            related: &["iv_first_stage", "iv_sargan_test"],
            r_equivalent: Some("AER::ivreg()"),
        },
        ToolInfo {
            name: "iv_first_stage",
            category: ToolCategory::IV,
            description: "First-stage diagnostics: F-stat, partial R²",
            related: &["iv_2sls"],
            r_equivalent: Some("lmtest::waldtest()"),
        },
        ToolInfo {
            name: "iv_sargan_test",
            category: ToolCategory::IV,
            description: "Sargan overidentification test",
            related: &["iv_2sls"],
            r_equivalent: Some("AER::summary.ivreg()"),
        },
        ToolInfo {
            name: "bp_bounds",
            category: ToolCategory::IV,
            description: "Balke-Pearl bounds for IV with binary outcomes",
            related: &["iv_2sls", "iv_mte"],
            r_equivalent: Some("bpbounds package"),
        },
        ToolInfo {
            name: "iv_mte",
            category: ToolCategory::IV,
            description: "Marginal treatment effects for IV",
            related: &["iv_2sls", "bp_bounds"],
            r_equivalent: Some("ivmte package"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Difference-in-Differences
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "diff_in_diff",
            category: ToolCategory::DiD,
            description: "Canonical 2x2 difference-in-differences",
            related: &["staggered_did", "etwfe"],
            r_equivalent: Some("Basic DiD in lm()"),
        },
        ToolInfo {
            name: "staggered_did",
            category: ToolCategory::DiD,
            description: "Callaway-Sant'Anna staggered treatment DiD",
            related: &["diff_in_diff", "bacon_decomp"],
            r_equivalent: Some("did::att_gt()"),
        },
        ToolInfo {
            name: "etwfe",
            category: ToolCategory::DiD,
            description: "Extended TWFE (Wooldridge approach)",
            related: &["diff_in_diff", "staggered_did"],
            r_equivalent: Some("etwfe package"),
        },
        ToolInfo {
            name: "bacon_decomp",
            category: ToolCategory::DiD,
            description: "Goodman-Bacon TWFE decomposition",
            related: &["staggered_did"],
            r_equivalent: Some("bacondecomp::bacon()"),
        },
        ToolInfo {
            name: "synthetic_control",
            category: ToolCategory::DiD,
            description: "Synthetic control method",
            related: &["gsynth", "scpi"],
            r_equivalent: Some("Synth package"),
        },
        ToolInfo {
            name: "gsynth",
            category: ToolCategory::DiD,
            description: "Generalized synthetic control (matrix completion)",
            related: &["synthetic_control"],
            r_equivalent: Some("gsynth package"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Regression Discontinuity
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "rd_estimate",
            category: ToolCategory::RD,
            description: "Sharp RD with CCT robust inference",
            related: &["rd_fuzzy", "rd_bw"],
            r_equivalent: Some("rdrobust::rdrobust()"),
        },
        ToolInfo {
            name: "rd_fuzzy",
            category: ToolCategory::RD,
            description: "Fuzzy RD design (two-stage)",
            related: &["rd_estimate"],
            r_equivalent: Some("rdrobust::rdrobust(fuzzy=)"),
        },
        ToolInfo {
            name: "rd_bw",
            category: ToolCategory::RD,
            description: "Optimal bandwidth selection (MSE, CCT)",
            related: &["rd_estimate"],
            r_equivalent: Some("rdrobust::rdbwselect()"),
        },
        ToolInfo {
            name: "rd_multi",
            category: ToolCategory::RD,
            description: "Multi-cutoff or multi-score RD",
            related: &["rd_estimate"],
            r_equivalent: Some("rdmulti package"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Matching
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "propensity_matching",
            category: ToolCategory::Matching,
            description: "Propensity score matching (1:1, 1:k, caliper)",
            related: &["treatment_ipw", "treatment_cbps"],
            r_equivalent: Some("MatchIt::matchit()"),
        },
        ToolInfo {
            name: "treatment_cbps",
            category: ToolCategory::Treatment,
            description: "Covariate balancing propensity scores",
            related: &["propensity_matching", "treatment_weightit"],
            r_equivalent: Some("CBPS::CBPS()"),
        },
        ToolInfo {
            name: "treatment_weightit",
            category: ToolCategory::Treatment,
            description: "Flexible propensity score weighting",
            related: &["treatment_cbps", "treatment_entropy_balance"],
            r_equivalent: Some("WeightIt::weightit()"),
        },
        ToolInfo {
            name: "treatment_entropy_balance",
            category: ToolCategory::Treatment,
            description: "Entropy balancing weights",
            related: &["treatment_cbps"],
            r_equivalent: Some("ebal::ebalance()"),
        },
        ToolInfo {
            name: "treatment_twang",
            category: ToolCategory::Treatment,
            description: "GBM-based propensity score estimation",
            related: &["treatment_cbps"],
            r_equivalent: Some("twang::ps()"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Treatment Effects
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "treatment_ipw",
            category: ToolCategory::Treatment,
            description: "Inverse probability weighting",
            related: &["treatment_doubly_robust", "treatment_tmle"],
            r_equivalent: Some("ipw package"),
        },
        ToolInfo {
            name: "treatment_doubly_robust",
            category: ToolCategory::Treatment,
            description: "Doubly robust (AIPW) estimation",
            related: &["treatment_ipw", "treatment_tmle"],
            r_equivalent: Some("drtmle package"),
        },
        ToolInfo {
            name: "treatment_tmle",
            category: ToolCategory::Treatment,
            description: "Targeted maximum likelihood estimation",
            related: &["treatment_doubly_robust", "collaborative_tmle"],
            r_equivalent: Some("tmle::tmle()"),
        },
        ToolInfo {
            name: "collaborative_tmle",
            category: ToolCategory::Treatment,
            description: "Collaborative TMLE for high-dimensional data",
            related: &["treatment_tmle", "ltmle"],
            r_equivalent: Some("ctmle package"),
        },
        ToolInfo {
            name: "ltmle",
            category: ToolCategory::Treatment,
            description: "Longitudinal TMLE for time-varying treatments",
            related: &["treatment_tmle"],
            r_equivalent: Some("ltmle::ltmle()"),
        },
        ToolInfo {
            name: "treatment_double_ml",
            category: ToolCategory::Treatment,
            description: "Double/debiased machine learning",
            related: &["treatment_doubly_robust"],
            r_equivalent: Some("DoubleML package"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Mediation Analysis
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "mediation_analysis",
            category: ToolCategory::Mediation,
            description: "Causal mediation analysis (ACME, ADE, total effect)",
            related: &["mediation_natural_effects"],
            r_equivalent: Some("mediation::mediate()"),
        },
        ToolInfo {
            name: "mediation_natural_effects",
            category: ToolCategory::Mediation,
            description: "Natural effect models (NDE, NIE)",
            related: &["mediation_analysis"],
            r_equivalent: Some("medflex::neModel()"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Discrete Choice
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "logit",
            category: ToolCategory::Discrete,
            description: "Logistic regression (binary outcome)",
            related: &["probit", "multinom"],
            r_equivalent: Some("glm(family=binomial)"),
        },
        ToolInfo {
            name: "probit",
            category: ToolCategory::Discrete,
            description: "Probit regression (binary outcome)",
            related: &["logit"],
            r_equivalent: Some("glm(family=binomial(probit))"),
        },
        ToolInfo {
            name: "multinom",
            category: ToolCategory::Discrete,
            description: "Multinomial logit for unordered categories",
            related: &["mlogit", "ordered_model"],
            r_equivalent: Some("nnet::multinom()"),
        },
        ToolInfo {
            name: "ordered_model",
            category: ToolCategory::Discrete,
            description: "Ordered logit/probit for ordinal outcomes",
            related: &["multinom"],
            r_equivalent: Some("MASS::polr()"),
        },
        ToolInfo {
            name: "negbin",
            category: ToolCategory::Discrete,
            description: "Negative binomial for overdispersed counts",
            related: &["zeroinfl", "hurdle_model"],
            r_equivalent: Some("MASS::glm.nb()"),
        },
        ToolInfo {
            name: "zeroinfl",
            category: ToolCategory::Discrete,
            description: "Zero-inflated Poisson/negative binomial",
            related: &["negbin", "hurdle_model"],
            r_equivalent: Some("pscl::zeroinfl()"),
        },
        ToolInfo {
            name: "feglm",
            category: ToolCategory::Discrete,
            description: "GLM with high-dimensional fixed effects",
            related: &["logit", "panel_hdfe"],
            r_equivalent: Some("fixest::feglm()"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Time Series
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "ts_arima_fit",
            category: ToolCategory::TimeSeries,
            description: "Fit ARIMA(p,d,q) model",
            related: &["ts_arima_forecast", "ts_garch_fit"],
            r_equivalent: Some("arima()"),
        },
        ToolInfo {
            name: "ts_arima_forecast",
            category: ToolCategory::TimeSeries,
            description: "Generate ARIMA forecasts with confidence intervals",
            related: &["ts_arima_fit"],
            r_equivalent: Some("predict.Arima()"),
        },
        ToolInfo {
            name: "ts_var",
            category: ToolCategory::TimeSeries,
            description: "Vector autoregression",
            related: &["ts_vecm", "ts_var_irf"],
            r_equivalent: Some("vars::VAR()"),
        },
        ToolInfo {
            name: "ts_var_irf",
            category: ToolCategory::TimeSeries,
            description: "Impulse response functions",
            related: &["ts_var"],
            r_equivalent: Some("vars::irf()"),
        },
        ToolInfo {
            name: "ts_garch_fit",
            category: ToolCategory::TimeSeries,
            description: "GARCH(p,q) volatility model",
            related: &["ts_arima_fit"],
            r_equivalent: Some("rugarch::ugarchfit()"),
        },
        ToolInfo {
            name: "ts_mstl",
            category: ToolCategory::TimeSeries,
            description: "Multiple seasonal decomposition (MSTL)",
            related: &["timeseries_decompose"],
            r_equivalent: Some("forecast::mstl()"),
        },
        ToolInfo {
            name: "ts_changepoint",
            category: ToolCategory::TimeSeries,
            description: "Changepoint detection (PELT, binary segmentation)",
            related: &["ts_mstl"],
            r_equivalent: Some("changepoint::cpt.*()"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Spatial
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "spatial_neighbors",
            category: ToolCategory::Spatial,
            description: "Create spatial neighbors (k-NN, distance)",
            related: &["moran_test", "sar_model"],
            r_equivalent: Some("spdep::knearneigh()"),
        },
        ToolInfo {
            name: "moran_test",
            category: ToolCategory::Spatial,
            description: "Moran's I test for spatial autocorrelation",
            related: &["spatial_lm_tests_tool"],
            r_equivalent: Some("spdep::moran.test()"),
        },
        ToolInfo {
            name: "sar_model",
            category: ToolCategory::Spatial,
            description: "Spatial autoregressive lag model (SAR)",
            related: &["sem_model", "moran_test"],
            r_equivalent: Some("spatialreg::lagsarlm()"),
        },
        ToolInfo {
            name: "sem_model",
            category: ToolCategory::Spatial,
            description: "Spatial error model (SEM)",
            related: &["sar_model"],
            r_equivalent: Some("spatialreg::errorsarlm()"),
        },
        ToolInfo {
            name: "spatial_panel_ml",
            category: ToolCategory::Spatial,
            description: "Spatial panel ML estimation",
            related: &["sar_model", "panel_fixed_effects"],
            r_equivalent: Some("splm::spml()"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Survival Analysis
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "survival_kaplan_meier",
            category: ToolCategory::Survival,
            description: "Kaplan-Meier survival curves estimation",
            related: &["survival_cox", "survival_log_rank"],
            r_equivalent: Some("survfit()"),
        },
        ToolInfo {
            name: "survival_log_rank",
            category: ToolCategory::Survival,
            description: "Log-rank test for survival curve comparison",
            related: &["survival_kaplan_meier"],
            r_equivalent: Some("survdiff()"),
        },
        ToolInfo {
            name: "survival_cox",
            category: ToolCategory::Survival,
            description: "Cox proportional hazards regression",
            related: &["survival_kaplan_meier", "survival_aft"],
            r_equivalent: Some("coxph()"),
        },
        ToolInfo {
            name: "survival_aft",
            category: ToolCategory::Survival,
            description: "Accelerated failure time model",
            related: &["survival_cox"],
            r_equivalent: Some("survreg()"),
        },
        ToolInfo {
            name: "survival_competing_risks",
            category: ToolCategory::Survival,
            description: "Competing risks analysis (Fine-Gray)",
            related: &["survival_cox"],
            r_equivalent: Some("cmprsk::crr()"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Machine Learning
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "ml_kmeans",
            category: ToolCategory::MachineLearning,
            description: "K-means clustering (k-means++ initialization)",
            related: &["ml_dbscan", "ml_hierarchical"],
            r_equivalent: Some("kmeans()"),
        },
        ToolInfo {
            name: "ml_dbscan",
            category: ToolCategory::MachineLearning,
            description: "Density-based clustering (DBSCAN)",
            related: &["ml_kmeans"],
            r_equivalent: Some("dbscan::dbscan()"),
        },
        ToolInfo {
            name: "ml_hierarchical",
            category: ToolCategory::MachineLearning,
            description: "Hierarchical clustering (Ward, complete, single)",
            related: &["ml_kmeans", "ml_cutree"],
            r_equivalent: Some("hclust()"),
        },
        ToolInfo {
            name: "ml_pca",
            category: ToolCategory::MachineLearning,
            description: "Principal component analysis",
            related: &["ml_tsne"],
            r_equivalent: Some("prcomp()"),
        },
        ToolInfo {
            name: "ml_tsne",
            category: ToolCategory::MachineLearning,
            description: "t-SNE dimensionality reduction",
            related: &["ml_pca"],
            r_equivalent: Some("Rtsne::Rtsne()"),
        },
        ToolInfo {
            name: "ml_random_forest",
            category: ToolCategory::MachineLearning,
            description: "Random forest classification/regression",
            related: &["ml_svm", "ml_causal_forest"],
            r_equivalent: Some("randomForest::randomForest()"),
        },
        ToolInfo {
            name: "ml_causal_forest",
            category: ToolCategory::MachineLearning,
            description: "Causal forest for heterogeneous treatment effects",
            related: &["ml_random_forest"],
            r_equivalent: Some("grf::causal_forest()"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Visualization
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "viz_histogram",
            category: ToolCategory::Visualization,
            description: "Create histogram plot (PNG)",
            related: &["viz_histogram_interactive"],
            r_equivalent: Some("hist()"),
        },
        ToolInfo {
            name: "viz_scatter",
            category: ToolCategory::Visualization,
            description: "Create scatter plot (PNG)",
            related: &["viz_scatter_interactive"],
            r_equivalent: Some("plot()"),
        },
        ToolInfo {
            name: "viz_line",
            category: ToolCategory::Visualization,
            description: "Create line chart (PNG)",
            related: &["viz_line_interactive"],
            r_equivalent: Some("plot(type='l')"),
        },
        ToolInfo {
            name: "viz_heatmap",
            category: ToolCategory::Visualization,
            description: "Create correlation heatmap",
            related: &["compute_correlation"],
            r_equivalent: Some("heatmap()"),
        },
        ToolInfo {
            name: "viz_scatter_interactive",
            category: ToolCategory::Visualization,
            description: "Interactive scatter plot (Plotly HTML)",
            related: &["viz_scatter"],
            r_equivalent: Some("plotly::plot_ly()"),
        },
        ToolInfo {
            name: "viz_coefficient",
            category: ToolCategory::Visualization,
            description: "Coefficient plot with confidence intervals",
            related: &["regression_ols"],
            r_equivalent: Some("coefplot::coefplot()"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Statistics
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "hypothesis_t_test",
            category: ToolCategory::Statistics,
            description: "One-sample, two-sample, or paired t-test",
            related: &["hypothesis_wilcoxon"],
            r_equivalent: Some("t.test()"),
        },
        ToolInfo {
            name: "hypothesis_wilcoxon",
            category: ToolCategory::Statistics,
            description: "Wilcoxon rank-sum or signed-rank test",
            related: &["hypothesis_t_test"],
            r_equivalent: Some("wilcox.test()"),
        },
        ToolInfo {
            name: "hypothesis_chisq_gof",
            category: ToolCategory::Statistics,
            description: "Chi-squared goodness of fit test",
            related: &["hypothesis_chisq_independence"],
            r_equivalent: Some("chisq.test()"),
        },
        ToolInfo {
            name: "anova_one_way",
            category: ToolCategory::Statistics,
            description: "One-way ANOVA",
            related: &["anova_two_way", "anova_tukey_hsd"],
            r_equivalent: Some("aov()"),
        },
        ToolInfo {
            name: "power_t_test",
            category: ToolCategory::Statistics,
            description: "Power analysis for t-tests",
            related: &["power_prop_test", "power_anova_test"],
            r_equivalent: Some("power.t.test()"),
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Database
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "db_sqlite_query",
            category: ToolCategory::Database,
            description: "Execute SQL query on SQLite database",
            related: &["db_sqlite_tables", "db_duckdb_query"],
            r_equivalent: Some("DBI::dbGetQuery()"),
        },
        ToolInfo {
            name: "db_duckdb_query",
            category: ToolCategory::Database,
            description: "Execute SQL query on DuckDB (can query Parquet/CSV directly)",
            related: &["db_duckdb_tables"],
            r_equivalent: Some("duckdb::dbGetQuery()"),
        },
        ToolInfo {
            name: "db_query_file",
            category: ToolCategory::Database,
            description: "Query Parquet/CSV files directly with SQL",
            related: &["db_duckdb_query"],
            r_equivalent: None,
        },

        // ══════════════════════════════════════════════════════════════════════════
        // Utility
        // ══════════════════════════════════════════════════════════════════════════
        ToolInfo {
            name: "set_seed",
            category: ToolCategory::Utility,
            description: "Set random seed for reproducibility",
            related: &["generate_random_data"],
            r_equivalent: Some("set.seed()"),
        },
        ToolInfo {
            name: "generate_random_data",
            category: ToolCategory::Utility,
            description: "Generate synthetic dataset with various distributions",
            related: &["set_seed", "create_dataset"],
            r_equivalent: Some("Various rnorm, runif, etc."),
        },
        ToolInfo {
            name: "generate_report",
            category: ToolCategory::Utility,
            description: "Generate HTML report from analysis results",
            related: &["export_session"],
            r_equivalent: Some("rmarkdown::render()"),
        },
        ToolInfo {
            name: "export_session",
            category: ToolCategory::Utility,
            description: "Export session state for later import",
            related: &["import_session"],
            r_equivalent: Some("save(), saveRDS()"),
        },
    ]
}

/// Get tools by category
pub fn tools_by_category(category: ToolCategory) -> Vec<ToolInfo> {
    get_registry()
        .into_iter()
        .filter(|t| t.category == category)
        .collect()
}

/// Get category counts for summary
pub fn category_counts() -> HashMap<ToolCategory, usize> {
    let registry = get_registry();
    let mut counts = HashMap::new();
    for tool in &registry {
        *counts.entry(tool.category).or_insert(0) += 1;
    }
    counts
}

/// Get total tool count
pub fn tool_count() -> usize {
    get_registry().len()
}

/// Search tools by keyword in name or description
pub fn search_tools(query: &str) -> Vec<ToolInfo> {
    let query_lower = query.to_lowercase();
    get_registry()
        .into_iter()
        .filter(|t| {
            t.name.to_lowercase().contains(&query_lower)
                || t.description.to_lowercase().contains(&query_lower)
        })
        .collect()
}

/// Generate markdown documentation for all tools
pub fn generate_markdown_docs() -> String {
    let mut docs = String::from("# MCP Tools Reference\n\n");
    docs.push_str(&format!("Total tools: {}\n\n", tool_count()));

    for category in ToolCategory::all() {
        docs.push_str(&format!("## {}\n\n", format!("{:?}", category)));
        docs.push_str(&format!("*{}*\n\n", category.description()));
        docs.push_str("| Tool | Description | R Equivalent |\n");
        docs.push_str("|------|-------------|-------------|\n");

        for tool in get_registry().iter().filter(|t| t.category == *category) {
            let r_equiv = tool.r_equivalent.unwrap_or("-");
            docs.push_str(&format!(
                "| `{}` | {} | {} |\n",
                tool.name, tool.description, r_equiv
            ));
        }
        docs.push('\n');
    }

    docs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_not_empty() {
        assert!(!get_registry().is_empty());
    }

    #[test]
    fn test_all_categories_have_tools() {
        let counts = category_counts();
        for category in ToolCategory::all() {
            assert!(
                counts.get(category).copied().unwrap_or(0) > 0,
                "Category {:?} has no tools",
                category
            );
        }
    }

    #[test]
    fn test_search() {
        let results = search_tools("regression");
        assert!(!results.is_empty());
    }
}
