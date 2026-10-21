//! The rule tables of the request-limits catalogue: [`PARAM_RULES`] classifies every cost-bearing
//! request field of the MCP tools, [`ROW_RULES`] names the tools whose cost grows faster than
//! linearly in the dataset's row count.

use super::{Family, ParamRule, RowClass, RowRule, Rule};

const fn p(tool: &'static str, path: &'static str, rule: Rule) -> ParamRule {
    ParamRule { tool, path, rule }
}

const fn row(tool: &'static str, class: RowClass) -> RowRule {
    RowRule {
        tool,
        dataset_field: "dataset",
        class,
    }
}

// Classification rules for PARAM_RULES:
//
// 1. A generic ("*") rule is used only when the field name means the same thing in every tool
//    that has it (`n_bootstrap` is always a replication count, `max_iter` always an iteration
//    limit). A tool-specific rule for the same path replaces the generic one for that tool.
// 2. Ambiguous names are always tool-specific: `p`, `q`, `d`, `n`, `k`, `order`, `lag`, `period`,
//    `start` and `values` mean a model order in one tool, a row count, a cluster count or an
//    index in another.
// 3. Every `Exempt` states why the field costs nothing (it selects, indexes or shrinks work, or a
//    closed-form formula uses it). The schema test in this file requires every integer field,
//    every array field and every `tol`/`eps` float field of every tool to be matched here.
//
// The list covers every integer property, every array property and every `tol`/`eps` float of
// the tool schemas at d6a3a3b.

/// Every classified request field (see the classification rules above).
pub static PARAM_RULES: &[ParamRule] = {
    use Family::*;
    use Rule::*;
    &[
        // Generic rules: apply to every tool that has the field and no tool-specific rule for it.
        p(
            "*",
            "seed",
            Exempt("random seed; selects a stream, costs nothing"),
        ),
        p("*", "bootstrap", Max(Replications)),
        p("*", "n_bootstrap", Max(Replications)),
        p("*", "n_boot", Max(Replications)),
        p("*", "n_permutations", Max(Replications)),
        p("*", "n_simulations", Max(SimulationSize)),
        p("*", "n_samples", Max(SimulationSize)),
        p("*", "n_trees", Max(Trees)),
        p("*", "committees", Max(Trees)),
        p("*", "mstop", Max(BoostingIterations)),
        p("*", "trials", Max(BoostingIterations)),
        p("*", "nterms", Max(BoostingIterations)),
        p("*", "max_terms", Max(BoostingIterations)),
        p("*", "n_draws", Max(McmcDraws)),
        p("*", "burn_in", Max(McmcDraws)),
        p("*", "cv_folds", Max(Folds)),
        p("*", "n_folds", Max(Folds)),
        p("*", "nfolds", Max(Folds)),
        p("*", "max_iter", Max(Iterations)),
        p("*", "max_iterations", Max(Iterations)),
        p("*", "n_init", Max(Restarts)),
        p("*", "n_grid", Max(GridPoints)),
        p("*", "nlambda", Max(GridPoints)),
        p("*", "bins", Max(Bins)),
        p("*", "lag_max", Max(Lags)),
        p("*", "lags", Max(Lags)),
        p("*", "max_lag", Max(Lags)),
        p("*", "min_lag", Max(Lags)),
        p("*", "order_max", Max(Lags)),
        p("*", "ar_order", Max(Lags)),
        p("*", "differences", Max(Lags)),
        p("*", "horizon", Max(Horizon)),
        p("*", "steps", Max(Horizon)),
        p("*", "mte_degree", Max(ModelOrder)),
        p("*", "instrument_order", Max(ModelOrder)),
        p("*", "bandwidth", Max(Bandwidth)),
        p("*", "n_components", Max(Components)),
        p("*", "n_factors", Max(Components)),
        p("*", "max_factors", Max(Components)),
        p("*", "rank", Max(Components)),
        p("*", "n_clusters", Max(Components)),
        p("*", "n_neighbors", Max(Neighbors)),
        p("*", "n_rows", Max(GeneratedRows)),
        p("*", "width", Max(ImageDimension)),
        p("*", "height", Max(ImageDimension)),
        p("*", "max_depth", Max(TreeDepth)),
        p("*", "maxdepth", Max(TreeDepth)),
        p("*", "tree_depth", Max(TreeDepth)),
        p(
            "*",
            "min_samples_split",
            Exempt("minimum node or segment size; larger values reduce work"),
        ),
        p(
            "*",
            "min_node_size",
            Exempt("minimum node or segment size; larger values reduce work"),
        ),
        p(
            "*",
            "min_cases",
            Exempt("minimum node or segment size; larger values reduce work"),
        ),
        p(
            "*",
            "min_split",
            Exempt("minimum node or segment size; larger values reduce work"),
        ),
        p(
            "*",
            "min_bucket",
            Exempt("minimum node or segment size; larger values reduce work"),
        ),
        p(
            "*",
            "minsplit",
            Exempt("minimum node or segment size; larger values reduce work"),
        ),
        p(
            "*",
            "minbucket",
            Exempt("minimum node or segment size; larger values reduce work"),
        ),
        p(
            "*",
            "min_samples",
            Exempt("minimum node or segment size; larger values reduce work"),
        ),
        p(
            "*",
            "min_segment_length",
            Exempt("minimum node or segment size; larger values reduce work"),
        ),
        p("*", "tolerance", Tolerance),
        p("*", "x", UniqueList(ListLength)),
        p("*", "z", UniqueList(ListLength)),
        p("*", "columns", UniqueList(ListLength)),
        p("*", "covariates", UniqueList(ListLength)),
        p("*", "features", UniqueList(ListLength)),
        p("*", "x_cols", UniqueList(ListLength)),
        p("*", "predictors", UniqueList(ListLength)),
        p("*", "y_columns", UniqueList(ListLength)),
        p("*", "fe", UniqueList(ListLength)),
        p("*", "controls", UniqueList(ListLength)),
        p("*", "instruments", UniqueList(ListLength)),
        p("*", "x_exog", UniqueList(ListLength)),
        p("*", "x_endog", UniqueList(ListLength)),
        p("*", "group_by", UniqueList(ListLength)),
        p("*", "by", UniqueList(ListLength)),
        p("*", "id_vars", UniqueList(ListLength)),
        p("*", "value_vars", UniqueList(ListLength)),
        p("*", "index", UniqueList(ListLength)),
        p("*", "confounders", UniqueList(ListLength)),
        p("*", "benchmark_covariates", UniqueList(ListLength)),
        p("*", "baseline_covariates", UniqueList(ListLength)),
        p("*", "time_varying_covariates", UniqueList(ListLength)),
        p("*", "treatment_cols", UniqueList(ListLength)),
        p("*", "alt_specific", UniqueList(ListLength)),
        p("*", "ind_specific", UniqueList(ListLength)),
        p("*", "random_vars", UniqueList(ListLength)),
        p("*", "variables", UniqueList(ListLength)),
        p("*", "response_vars", UniqueList(ListLength)),
        p("*", "factor_columns", UniqueList(ListLength)),
        p("*", "factors", UniqueList(ListLength)),
        p("*", "x_columns", UniqueList(ListLength)),
        p("*", "x_restricted", UniqueList(ListLength)),
        p("*", "x_unrestricted", UniqueList(ListLength)),
        p("*", "left_on", UniqueList(ListLength)),
        p("*", "right_on", UniqueList(ListLength)),
        p("*", "control_cols", UniqueList(ListLength)),
        p("*", "outcomes", UniqueList(ListLength)),
        p("*", "treatments", UniqueList(ListLength)),
        p("*", "datasets", ArrayLen(ListLength)),
        p("*", "xout", ArrayLen(ArrayLength)),
        p("*", "at", ArrayLen(ArrayLength)),
        p("*", "probs", ArrayLen(ArrayLength)),
        p("*", "scores", ArrayLen(ArrayLength)),
        p("*", "lambda", ArrayLen(GridPoints)),
        p("*", "taus", ArrayLen(ListLength)),
        p("*", "cutoffs", ArrayLen(ListLength)),
        p("*", "bandwidths", ArrayLen(ListLength)),
        // Tool-specific rules, sorted by tool.
        p("causal_impact_analysis", "seasonal_period", Max(Lags)),
        p(
            "causal_impact_analysis",
            "pre_period_start",
            Exempt("time value that selects rows"),
        ),
        p(
            "causal_impact_analysis",
            "pre_period_end",
            Exempt("time value that selects rows"),
        ),
        p(
            "causal_impact_analysis",
            "post_period_start",
            Exempt("time value that selects rows"),
        ),
        p(
            "causal_impact_analysis",
            "post_period_end",
            Exempt("time value that selects rows"),
        ),
        p(
            "cleaning_rollback",
            "checkpoint_index",
            Exempt("index into existing checkpoints"),
        ),
        p("cleaning_session_apply", "value", Max(StringBytes)),
        p("cleaning_session_apply", "old_value", Max(StringBytes)),
        p("cleaning_session_apply", "filter_value", Max(StringBytes)),
        p("collaborative_tmle", "max_covariates", Max(ListLength)),
        p("descriptive_medpolish", "eps", Tolerance),
        p("generate_random_data", "columns", ArrayLen(ListLength)),
        p("generate_random_data", "columns[].name", Max(StringBytes)),
        p(
            "generate_random_data",
            "columns[].distribution.categories",
            ArrayLen(ArrayLength),
        ),
        p(
            "generate_random_data",
            "columns[].distribution.categories[]",
            Max(StringBytes),
        ),
        p(
            "generate_random_data",
            "columns[].distribution.weights",
            ArrayLen(ArrayLength),
        ),
        p(
            "generate_random_data",
            "columns[].distribution.value",
            Max(StringBytes),
        ),
        p("generate_report", "sections", ArrayLen(ListLength)),
        p(
            "generate_report",
            "sections[].content",
            ArrayLen(ListLength),
        ),
        p(
            "generate_report",
            "sections[].content[].headers",
            ArrayLen(ListLength),
        ),
        p(
            "generate_report",
            "sections[].content[].rows",
            ArrayLen(ArrayLength),
        ),
        p(
            "generate_report",
            "sections[].content[].rows[]",
            ArrayLen(ListLength),
        ),
        p(
            "generate_report",
            "sections[].content[].stats",
            ArrayLen(ListLength),
        ),
        p("gformula", "time_points", Max(ListLength)),
        p(
            "gformula",
            "threshold_variable",
            Exempt("index into the time-varying covariate list"),
        ),
        p(
            "head_dataset",
            "n",
            Exempt(
                "rows returned are bounded by the dataset height; text output by response_bytes",
            ),
        ),
        p(
            "hypothesis_mcnemar",
            "b",
            Exempt("cell count in a closed-form statistic"),
        ),
        p(
            "hypothesis_mcnemar",
            "c",
            Exempt("cell count in a closed-form statistic"),
        ),
        p("hypothesis_poisson", "x", ArrayLen(ListLength)),
        p("hypothesis_poisson", "t", ArrayLen(ListLength)),
        p(
            "hypothesis_prop_trend_test",
            "trials",
            Exempt("column name, not a boosting count"),
        ),
        p("linalg_toeplitz", "column", SquareLen),
        p("linalg_toeplitz", "row", SquareLen),
        p("ltmle", "covariates", ArrayLen(ListLength)),
        p("ml_cmdscale", "k", Max(Components)),
        p("ml_cubist", "neighbors", Max(Neighbors)),
        p("ml_cutree", "k", Max(Components)),
        p(
            "ml_dbscan",
            "eps",
            Exempt(
                "neighbourhood radius, not a convergence tolerance; dense cost is bounded by dense_rows",
            ),
        ),
        p("ml_kmeans", "k", Max(Components)),
        p(
            "ml_shap_values",
            "max_obs",
            Exempt("bounded by the dataset height"),
        ),
        p("multivariate_mahalanobis", "center", ArrayLen(ListLength)),
        p("munge_bin", "bins", ArrayLen(Bins)),
        p("munge_bin", "labels", ArrayLen(Bins)),
        p("munge_bin", "labels[]", Max(StringBytes)),
        p("munge_cast", "casts", ArrayLen(ListLength)),
        p(
            "munge_diff",
            "periods",
            Exempt("polars shift; cost is bounded by the column length"),
        ),
        p("munge_filter", "value", Max(StringBytes)),
        p("munge_group_by", "aggs", ArrayLen(ListLength)),
        p("munge_join", "suffix", Max(StringBytes)),
        p(
            "munge_lag_lead",
            "periods",
            Exempt("polars shift; cost is bounded by the column length"),
        ),
        p("munge_mutate", "left", Max(StringBytes)),
        p("munge_mutate", "right", Max(StringBytes)),
        p("munge_rename", "renames", ArrayLen(ListLength)),
        p("munge_sample", "n", Max(GeneratedRows)),
        p(
            "power_anova_test",
            "groups",
            Exempt("number of groups in a closed-form power formula"),
        ),
        p(
            "preview_cleaning",
            "sample_size",
            Exempt("bounded by the number of changed rows"),
        ),
        p("propensity_matching", "ratio", Max(Neighbors)),
        p("propensity_matching", "n_subclasses", Max(Bins)),
        p("rd_bw", "p", Max(ModelOrder)),
        p("rd_estimate", "p", Max(ModelOrder)),
        p("rd_fuzzy", "p", Max(ModelOrder)),
        p("rd_multi", "p", Max(ModelOrder)),
        p(
            "rd_multi",
            "bandwidth",
            Exempt("kernel bandwidth in units of the running variable, not a lag count"),
        ),
        p("regression_bgtest", "order", Max(Lags)),
        p("regression_line", "iter", Max(Iterations)),
        p("regression_loess", "degree", Max(ModelOrder)),
        p("regression_nls", "start", ArrayLen(ListLength)),
        p("regression_resettest", "powers", ArrayLen(ListLength)),
        p("regression_resettest", "powers", ArrayElemMax(ModelOrder)),
        p(
            "scpi",
            "treatment_time",
            Exempt("time value that selects rows"),
        ),
        p(
            "search_tools",
            "limit",
            Exempt("the handler clamps it to 25"),
        ),
        p("spatial_neighbors", "k", Max(Neighbors)),
        p(
            "staggered_did",
            "base_period",
            Exempt("relative period offset"),
        ),
        p("stats_density", "n", Max(GridPoints)),
        p("stats_iqr", "qtype", Exempt("quantile type selector 1-9")),
        p("stats_loglin", "eps", Tolerance),
        p("stats_loglin", "margins", ArrayLen(ListLength)),
        p("stats_loglin", "margins[]", ArrayLen(ListLength)),
        p("stats_se_contrast", "contrasts", ArrayLen(ListLength)),
        p("stats_se_contrast", "contrasts[]", ArrayLen(ListLength)),
        p("str_concat", "columns", ArrayLen(ListLength)),
        p("str_concat", "separator", Max(StringBytes)),
        p("str_regex_count", "pattern", Max(StringBytes)),
        p("str_regex_extract", "group", Exempt("capture-group index")),
        p("str_regex_extract", "pattern", Max(StringBytes)),
        p("str_regex_replace", "pattern", Max(StringBytes)),
        p("str_regex_replace", "replacement", Max(StringBytes)),
        p("str_replace_value", "old_value", Max(StringBytes)),
        p("str_replace_value", "new_value", Max(StringBytes)),
        p("str_split", "max_splits", Max(ListLength)),
        p("str_split", "pattern", Max(StringBytes)),
        p("str_split", "prefix", Max(StringBytes)),
        p(
            "str_substring",
            "start",
            Exempt("position inside existing strings"),
        ),
        p(
            "str_substring",
            "length",
            Exempt("bounded by the length of existing strings"),
        ),
        p(
            "suggest_cleaning",
            "limit",
            Exempt("truncates an already computed suggestion list"),
        ),
        p(
            "synthetic_control",
            "treatment_time",
            Exempt("time value that selects rows"),
        ),
        p("synthetic_control", "predictors", ArrayLen(ListLength)),
        p(
            "synthetic_control",
            "predictors[].time_window",
            Exempt("[start, end] pair of time values"),
        ),
        p(
            "synthetic_control",
            "optimization_window",
            Exempt("[start, end] pair of time values"),
        ),
        p(
            "synthetic_control",
            "custom_v_weights",
            ArrayLen(ListLength),
        ),
        p("timeseries_acf_to_ar", "acf", ArrayLen(Lags)),
        p("timeseries_ar", "order", Max(Lags)),
        p("timeseries_arima_sim", "n", Max(GeneratedRows)),
        p("timeseries_arima_sim", "d", Max(ModelOrder)),
        p("timeseries_arima_sim", "ar", ArrayLen(ModelOrder)),
        p("timeseries_arima_sim", "ma", ArrayLen(ModelOrder)),
        p("timeseries_arma_acf", "ar", ArrayLen(ModelOrder)),
        p("timeseries_arma_acf", "ma", ArrayLen(ModelOrder)),
        p("timeseries_arma_to_ma", "ar", ArrayLen(ModelOrder)),
        p("timeseries_arma_to_ma", "ma", ArrayLen(ModelOrder)),
        p("timeseries_box_test", "lag", Max(Lags)),
        p(
            "timeseries_box_test",
            "fitdf",
            Exempt("degrees-of-freedom adjustment; no allocation or loop"),
        ),
        p("timeseries_decompose", "period", Max(Lags)),
        p("timeseries_diffinv", "lag", Max(Lags)),
        p("timeseries_diffinv", "xi", ArrayLen(ArrayLength)),
        p("timeseries_embed", "dimension", Max(Lags)),
        p("timeseries_filter", "sides", Exempt("selector 1 or 2")),
        p("timeseries_filter", "filter", ArrayLen(SpanWidth)),
        p(
            "timeseries_lag",
            "k",
            Exempt("core rejects |k| >= n; the output is a slice of the input"),
        ),
        p("timeseries_runmed", "k", Max(SpanWidth)),
        p("timeseries_spectrum", "spans", ArrayLen(ListLength)),
        p("timeseries_spectrum", "spans", ArrayElemMax(SpanWidth)),
        p("timeseries_structts", "period", Max(Lags)),
        p(
            "timeseries_window",
            "start",
            Exempt("index into the series; the output is a slice"),
        ),
        p(
            "timeseries_window",
            "end",
            Exempt("index into the series; the output is a slice"),
        ),
        p(
            "treatment_entropy_balance",
            "target_means",
            ArrayLen(ListLength),
        ),
        p(
            "treatment_sbw",
            "balance_tol",
            Exempt("balance tolerance; 0 (exact balance) is the default and valid"),
        ),
        p("ts_arima_fit", "p", Max(ModelOrder)),
        p("ts_arima_fit", "d", Max(ModelOrder)),
        p("ts_arima_fit", "q", Max(ModelOrder)),
        p("ts_arima_forecast", "p", Max(ModelOrder)),
        p("ts_arima_forecast", "d", Max(ModelOrder)),
        p("ts_arima_forecast", "q", Max(ModelOrder)),
        p("ts_garch_fit", "p", Max(ModelOrder)),
        p("ts_garch_fit", "q", Max(ModelOrder)),
        p("ts_holt_winters", "period", Max(Lags)),
        p("ts_mstl", "periods", ArrayLen(ListLength)),
        p("ts_mstl", "periods", ArrayElemMax(Lags)),
        p("ts_varma", "p", Max(ModelOrder)),
        p("ts_varma", "q", Max(ModelOrder)),
        p("viz_dendrogram", "linkage_matrix", SquareLen),
        p("viz_dendrogram", "labels", ArrayLen(ArrayLength)),
    ]
};

/// Tools whose dataset height is limited, by [`RowClass`]. All take the dataset in `dataset`.
///
/// Deliberately not row-limited, because their wave-2 packages remove the superlinear cost:
/// `hypothesis_cor_test` (Kendall), `stats_ecdf`, `timeseries_cpgram`, `ml_ctree`,
/// `treatment_twang`, `treatment_sbw`, `bacon_decomp`, `regression_gls` (AR(1) and CS closed
/// forms; a user-supplied correlation relies on the per-allocation ceiling), `panel_gls`,
/// `moran_test`, `spatial_lm_tests_tool` and `spatial_neighbors`.
pub static ROW_RULES: &[RowRule] = &[
    // Dense: clustering.rs:829-830 keeps the condensed n(n-1)/2 f64 distance matrix.
    row("ml_hierarchical", RowClass::Dense),
    row("ml_cutree", RowClass::Dense),
    // Dense: neighbour lists of up to n^2 indices (ML-23).
    row("ml_dbscan", RowClass::Dense),
    // Dense: smooth_spline.rs:377 allocates the n x n penalty `vec![vec![0.0; n]; n]`.
    row("regression_smooth_spline", RowClass::Dense),
    // Cubic: reduction.rs:474/582/641/690 builds several n x n matrices per iteration.
    row("ml_tsne", RowClass::Cubic),
    // Cubic: reduction.rs:912/934/1004 builds three n x n matrices plus an O(n^3) eigensolve.
    row("ml_cmdscale", RowClass::Cubic),
    // Cubic: weights.rs:320 computes dense eigenvalues of W (R parity, `method = "eigen"`).
    row("sar_model", RowClass::Cubic),
    row("sem_model", RowClass::Cubic),
    // Cubic: spatialprobit.rs:401 uses a dense W; the impacts do dense n^3 products per draw.
    row("sar_probit_model", RowClass::Cubic),
    row("sem_probit_model", RowClass::Cubic),
    // Cubic: splm.rs:1793 builds the dense NT x NT projection `P_Z`.
    row("spatial_panel_ml", RowClass::Cubic),
    row("spatial_panel_gmm", RowClass::Cubic),
    // Quadratic: nearest-neighbour search over all controls.
    row("propensity_matching", RowClass::Quadratic),
    // Quadratic: O(B n^2) matching permutations (ECON-10).
    row("heterogeneity_test", RowClass::Quadratic),
    // Quadratic: SMO over all pairs (ML-39; still O(n^2) after T0-WP15's error cache).
    row("ml_svm", RowClass::Quadratic),
    // Quadratic: loess.rs:240-290 fits each point on its span * n nearest neighbours.
    row("regression_loess", RowClass::Quadratic),
];

#[cfg(test)]
mod tests {
    use super::super::{Family, LimitViolation, Limits, Rule, rules_for};
    use super::{PARAM_RULES, ROW_RULES};
    use serde_json::{Value, json};

    fn hosted(tool: &str, args: Value) -> Result<(), LimitViolation> {
        Limits::hosted().check_tool_arguments(tool, &args)
    }

    /// One representative per family; value = cap passes, cap + 1 fails.
    const REPRESENTATIVE: &[(Family, &str, &str)] = &[
        (Family::Replications, "regression_bootstrap_cov", "n_boot"),
        (Family::SimulationSize, "gformula", "n_simulations"),
        (Family::Trees, "treatment_twang", "n_trees"),
        (Family::BoostingIterations, "ml_mboost", "mstop"),
        (Family::McmcDraws, "sar_probit_model", "n_draws"),
        (Family::Folds, "regression_cv_glmnet", "nfolds"),
        (Family::Iterations, "panel_hdfe", "max_iterations"),
        (Family::Restarts, "ml_kmeans", "n_init"),
        (Family::GridPoints, "stats_density", "n"),
        (Family::Bins, "viz_histogram", "bins"),
        (Family::Lags, "timeseries_arma_to_ma", "lag_max"),
        (Family::Horizon, "ts_var_irf", "steps"),
        (Family::ModelOrder, "rd_estimate", "p"),
        (Family::Bandwidth, "regression_hac", "bandwidth"),
        (Family::SpanWidth, "timeseries_runmed", "k"),
        (Family::Components, "ml_pca", "n_components"),
        (Family::Neighbors, "propensity_matching", "ratio"),
        (Family::GeneratedRows, "timeseries_arima_sim", "n"),
        (Family::ImageDimension, "viz_dendrogram", "width"),
        (Family::ListLength, "stats_cov_wt", "columns"),
        (Family::ArrayLength, "stats_spline", "xout"),
        (Family::StringBytes, "str_regex_count", "pattern"),
        (Family::TreeDepth, "ml_random_forest", "max_depth"),
    ];

    fn value_of_size(f: Family, n: u64) -> Value {
        match f {
            Family::ListLength => json!((0..n).map(|i| format!("c{i}")).collect::<Vec<_>>()),
            Family::ArrayLength => json!(vec![0.5; n as usize]),
            Family::StringBytes => json!("a".repeat(n as usize)),
            _ => json!(n),
        }
    }

    #[test]
    fn every_family_cap_is_enforced_at_the_boundary() {
        let h = Limits::hosted();
        for f in Family::ALL {
            assert!(
                REPRESENTATIVE.iter().any(|r| r.0 == *f),
                "no representative for {}",
                f.key()
            );
        }
        for &(f, tool, field) in REPRESENTATIVE {
            let cap = h.cap(f).unwrap();
            assert!(
                hosted(tool, json!({ field: value_of_size(f, cap) })).is_ok(),
                "{tool}.{field} at cap"
            );
            let v = hosted(tool, json!({ field: value_of_size(f, cap + 1) })).unwrap_err();
            assert_eq!(
                (v.parameter.as_str(), v.override_key),
                (field, f.key()),
                "{tool}.{field}"
            );
        }
    }

    /// Request values reproduced as process aborts or endless loops in the review (X-1, TOOL-1,
    /// CAUS-3/4/6/8, ECON-9/10/18, ML-3/4/39, REG-1/2, SPAT-21, STAT-7/9/43, TS-3, TOOL-2,
    /// DATA-3/5).
    #[test]
    fn reviewed_abort_and_hang_values_are_refused_in_hosted_profile() {
        let cases: &[(&str, Value, &str, &str)] = &[
            (
                "timeseries_arma_to_ma",
                json!({"lag_max": 10_000_000_000u64, "ar": [0.5], "ma": [0.3]}),
                "lag_max",
                "lags",
            ),
            (
                "timeseries_arma_acf",
                json!({"lag_max": 10_000_000_000u64}),
                "lag_max",
                "lags",
            ),
            (
                "stats_density",
                json!({"dataset": "d", "column": "x", "n": 10_000_000_000u64}),
                "n",
                "grid_points",
            ),
            (
                "viz_histogram",
                json!({"bins": 100_000_000_000u64}),
                "bins",
                "bins",
            ),
            (
                "regression_glmnet",
                json!({"nlambda": 10_000_000_000u64}),
                "nlambda",
                "grid_points",
            ),
            (
                "regression_cv_glmnet",
                json!({"nfolds": 1_000_000}),
                "nfolds",
                "folds",
            ),
            (
                "rd_estimate",
                json!({"p": 1_099_511_627_776u64}),
                "p",
                "model_order",
            ),
            (
                "rd_multi",
                json!({"p": 10_000_000_000u64}),
                "p",
                "model_order",
            ),
            (
                "iv_mte",
                json!({"n_grid": 100_000_000_000u64}),
                "n_grid",
                "grid_points",
            ),
            (
                "iv_mte",
                json!({"mte_degree": 1_000}),
                "mte_degree",
                "model_order",
            ),
            (
                "mixed_logit",
                json!({"n_draws": 1_000_000_000_000u64}),
                "n_draws",
                "mcmc_draws",
            ),
            (
                "treatment_ipw",
                json!({"bootstrap": 10_000_000_000u64}),
                "bootstrap",
                "replications",
            ),
            (
                "treatment_doubly_robust",
                json!({"bootstrap": 1_000_000}),
                "bootstrap",
                "replications",
            ),
            (
                "staggered_did",
                json!({"bootstrap": 100_000_000}),
                "bootstrap",
                "replications",
            ),
            (
                "treatment_twang",
                json!({"n_trees": 1_000_000_000_000u64}),
                "n_trees",
                "trees",
            ),
            (
                "mediation_analysis",
                json!({"bootstrap": 1_000_000_000_000u64}),
                "bootstrap",
                "replications",
            ),
            (
                "natural_effects_mediation",
                json!({"n_bootstrap": 1_000_000_000_000u64}),
                "n_bootstrap",
                "replications",
            ),
            (
                "regression_standardization",
                json!({"n_bootstrap": 10_000_000_000u64}),
                "n_bootstrap",
                "replications",
            ),
            (
                "gformula",
                json!({"n_simulations": 1_000_000_000u64}),
                "n_simulations",
                "simulation_size",
            ),
            (
                "collaborative_tmle",
                json!({"n_folds": 10_000_000_000u64}),
                "n_folds",
                "folds",
            ),
            (
                "bp_bounds",
                json!({"n_bootstrap": 1_000_000_000_000u64}),
                "n_bootstrap",
                "replications",
            ),
            (
                "scpi",
                json!({"variance_method": "kfold_cv", "cv_folds": 1_000_000_000_000_000_000u64}),
                "cv_folds",
                "folds",
            ),
            (
                "gsynth",
                json!({"n_factors": 1_000_000_000u64}),
                "n_factors",
                "components",
            ),
            (
                "synthetic_control",
                json!({"max_iter": 1_000_000_000_000u64}),
                "max_iter",
                "iterations",
            ),
            (
                "synthetic_control",
                json!({"tolerance": 0.0}),
                "tolerance",
                "min_tolerance",
            ),
            (
                "panel_hdfe",
                json!({"max_iterations": 1_000_000_000_000_000u64}),
                "max_iterations",
                "iterations",
            ),
            (
                "panel_hdfe",
                json!({"tolerance": 0.0}),
                "tolerance",
                "min_tolerance",
            ),
            (
                "feglm",
                json!({"tolerance": -1.0}),
                "tolerance",
                "min_tolerance",
            ),
            (
                "gmm_iv",
                json!({"bandwidth": 1_000_000_000_000_000u64}),
                "bandwidth",
                "bandwidth",
            ),
            (
                "heterogeneity_test",
                json!({"n_permutations": 1_000_000_000_000u64}),
                "n_permutations",
                "replications",
            ),
            (
                "ts_var_irf",
                json!({"steps": 100_000_000_000u64}),
                "steps",
                "horizon",
            ),
            (
                "ts_holt_winters",
                json!({"horizon": 10_000_000_000u64}),
                "horizon",
                "horizon",
            ),
            (
                "ts_arima_forecast",
                json!({"horizon": 10_000_000_000u64}),
                "horizon",
                "horizon",
            ),
            (
                "timeseries_arima_sim",
                json!({"n": 10_000_000_000u64}),
                "n",
                "generated_rows",
            ),
            (
                "timeseries_diffinv",
                json!({"lag": 10_000_000_000u64}),
                "lag",
                "lags",
            ),
            (
                "timeseries_diffinv",
                json!({"differences": 10_000_000_000u64}),
                "differences",
                "lags",
            ),
            (
                "timeseries_spectrum",
                json!({"spans": [3, 1_000_000_001u64]}),
                "spans",
                "span_width",
            ),
            (
                "ml_random_forest",
                json!({"n_trees": 10_000_000_000u64}),
                "n_trees",
                "trees",
            ),
            (
                "ml_shap_values",
                json!({"n_trees": 10_000_000_000u64}),
                "n_trees",
                "trees",
            ),
            (
                "ml_causal_forest",
                json!({"n_trees": 1_000_000_000_000u64}),
                "n_trees",
                "trees",
            ),
            (
                "ml_bart_causal",
                json!({"n_bootstrap": 1_000_000_000_000u64}),
                "n_bootstrap",
                "replications",
            ),
            (
                "ml_mboost",
                json!({"mstop": 1_000_000_000_000u64}),
                "mstop",
                "boosting_iterations",
            ),
            (
                "ml_cubist",
                json!({"committees": 10_000_000_000u64}),
                "committees",
                "trees",
            ),
            (
                "ml_c50",
                json!({"trials": 10_000_000_000u64}),
                "trials",
                "boosting_iterations",
            ),
            (
                "ml_ppr",
                json!({"nterms": 10_000_000_000u64}),
                "nterms",
                "boosting_iterations",
            ),
            (
                "ml_svm",
                json!({"max_iterations": 1_000_000_000u64}),
                "max_iterations",
                "iterations",
            ),
            (
                "ml_kmeans",
                json!({"n_init": 1_000_000}),
                "n_init",
                "restarts",
            ),
            (
                "ml_tsne",
                json!({"max_iterations": 1_000_000_000u64}),
                "max_iterations",
                "iterations",
            ),
            (
                "propensity_matching",
                json!({"ratio": 10_000_000_000u64}),
                "ratio",
                "neighbors",
            ),
            (
                "regression_bootstrap_cov",
                json!({"n_boot": 100_000_000_000u64}),
                "n_boot",
                "replications",
            ),
            (
                "regression_hac",
                json!({"bandwidth": 1_000_000_000_000u64}),
                "bandwidth",
                "bandwidth",
            ),
            (
                "regression_driscoll_kraay",
                json!({"bandwidth": 10_000_000}),
                "bandwidth",
                "bandwidth",
            ),
            (
                "sphet_model",
                json!({"bandwidth": 1_000_000_000_000_000u64}),
                "bandwidth",
                "bandwidth",
            ),
            (
                "sphet_model",
                json!({"instrument_order": 1_000_000_000u64}),
                "instrument_order",
                "model_order",
            ),
            (
                "sar_probit_model",
                json!({"burn_in": 1_000_000_000u64}),
                "burn_in",
                "mcmc_draws",
            ),
            (
                "sem_probit_model",
                json!({"n_draws": 1_000_000_000u64}),
                "n_draws",
                "mcmc_draws",
            ),
            (
                "viz_dendrogram",
                json!({"width": 65_536, "height": 65_536}),
                "width",
                "image_dimension",
            ),
            (
                "generate_random_data",
                json!({"n_rows": 1_000_000_000u64, "columns": []}),
                "n_rows",
                "generated_rows",
            ),
            (
                "munge_bin",
                json!({"bins": vec![1.0; 1_001]}),
                "bins",
                "bins",
            ),
        ];
        for (tool, args, param, key) in cases {
            let v = hosted(tool, args.clone()).expect_err(tool);
            assert_eq!(
                (v.parameter.as_str(), v.override_key),
                (*param, *key),
                "{tool}"
            );
            assert!(
                !v.message.contains('/'),
                "message must not contain paths: {}",
                v.message
            );
            assert!(Limits::off().check_tool_arguments(tool, args).is_ok());
        }
        // TOOL-2 / STAT-43 / STAT-70 / DATA-5: shape-driven allocations
        assert_eq!(
            hosted("linalg_toeplitz", json!({"column": vec![1.0; 1_001]}))
                .unwrap_err()
                .override_key,
            "table_cells"
        );
        assert!(hosted("linalg_toeplitz", json!({"column": vec![1.0; 1_000]})).is_ok());
        assert_eq!(
            hosted(
                "viz_dendrogram",
                json!({"linkage_matrix": vec![[0, 1, 0, 2]; 1_001]})
            )
            .unwrap_err()
            .override_key,
            "table_cells"
        );
        assert!(
            hosted(
                "multivariate_cancor",
                json!({"x_columns": ["a", "a"], "y_columns": ["b"]})
            )
            .unwrap_err()
            .message
            .contains("more than once")
        );
        assert!(hosted("stats_loglin", json!({"factor_columns": ["g", "g", "g"]})).is_err());
        assert_eq!(
            hosted("stats_spline", json!({"xout": vec![0.0; 100_001]}))
                .unwrap_err()
                .override_key,
            "array_length"
        );
        assert_eq!(
            hosted(
                "generate_random_data",
                json!({"n_rows": 10, "columns": [
                    {"name": "s", "distribution": {"type": "constant_string", "value": "x".repeat(5_000)}}
                ]})
            )
            .unwrap_err()
            .parameter,
            "columns[].distribution.value"
        );
    }

    /// Every hosted cap is at least 1.5x the tool's default (handler or core default).
    #[test]
    fn hosted_caps_accept_one_and_a_half_times_every_default() {
        const DEFAULTS: &[(&str, &str, u64)] = &[
            ("regression_bootstrap_cov", "n_boot", 999),
            ("treatment_ipw", "bootstrap", 999),
            ("treatment_doubly_robust", "bootstrap", 999),
            ("staggered_did", "bootstrap", 999),
            ("mediation_analysis", "bootstrap", 999),
            ("regression_standardization", "n_bootstrap", 999),
            ("natural_effects_mediation", "n_bootstrap", 1000),
            ("bp_bounds", "n_bootstrap", 1000),
            ("gsynth", "n_bootstrap", 500),
            ("gsynth", "n_factors", 2),
            ("gsynth", "max_factors", 5),
            ("gformula", "n_bootstrap", 200),
            ("gformula", "n_simulations", 1000),
            ("heterogeneity_test", "n_permutations", 1000),
            ("heterogeneity_test", "n_neighbors", 3),
            ("treatment_twang", "n_trees", 3000),
            ("ml_random_forest", "n_trees", 100),
            ("ml_random_forest", "max_depth", 10),
            ("ml_shap_values", "n_trees", 50),
            ("ml_shap_values", "max_depth", 6),
            ("ml_causal_forest", "n_trees", 2000),
            ("ml_causal_forest", "max_depth", 10),
            ("ml_bart_causal", "n_trees", 200),
            ("ml_bart_causal", "max_depth", 4),
            ("ml_bart_causal", "n_bootstrap", 100),
            ("ml_cubist", "max_depth", 10),
            ("ml_cubist", "committees", 1),
            ("ml_c50", "trials", 1),
            ("ml_ppr", "nterms", 1),
            ("ml_mboost", "mstop", 100),
            ("ml_mboost", "tree_depth", 1),
            ("ml_kmeans", "max_iterations", 300),
            ("ml_kmeans", "n_init", 10),
            ("ml_tsne", "max_iterations", 1000),
            ("ml_tsne", "n_components", 2),
            ("ml_svm", "max_iterations", 1000),
            ("ml_cmdscale", "k", 2),
            ("panel_hdfe", "max_iterations", 10000),
            ("feglm", "max_iter", 25),
            ("cox_ph", "max_iter", 100),
            ("aft", "max_iter", 100),
            ("synthetic_control", "max_iter", 1000),
            ("stats_loglin", "max_iter", 20),
            ("descriptive_medpolish", "max_iter", 10),
            ("regression_line", "iter", 1),
            ("mixed_logit", "n_draws", 500),
            ("sar_probit_model", "n_draws", 1000),
            ("sar_probit_model", "burn_in", 200),
            ("sem_probit_model", "n_draws", 1000),
            ("sem_probit_model", "burn_in", 200),
            ("regression_cv_glmnet", "nfolds", 10),
            ("regression_cv_glmnet", "nlambda", 100),
            ("regression_glmnet", "nlambda", 100),
            ("collaborative_tmle", "n_folds", 5),
            ("collaborative_tmle", "max_covariates", 10),
            ("treatment_double_ml", "n_folds", 5),
            ("scpi", "cv_folds", 5),
            ("iv_mte", "n_grid", 100),
            ("iv_mte", "mte_degree", 2),
            ("stats_density", "n", 512),
            ("timeseries_arma_acf", "lag_max", 10),
            ("timeseries_arma_to_ma", "lag_max", 10),
            ("timeseries_diffinv", "lag", 1),
            ("timeseries_diffinv", "differences", 1),
            ("timeseries_box_test", "lag", 1),
            ("viz_dendrogram", "width", 800),
            ("viz_dendrogram", "height", 600),
            ("sphet_model", "instrument_order", 2),
            ("spatial_neighbors", "k", 5),
            ("regression_loess", "degree", 2),
            ("regression_bgtest", "order", 1),
            ("propensity_matching", "ratio", 1),
            ("propensity_matching", "n_subclasses", 5),
            ("panel_gmm", "lags", 1),
            ("panel_gmm", "min_lag", 2),
            ("rd_estimate", "p", 1),
            ("ts_garch_fit", "p", 1),
            ("ts_garch_fit", "q", 1),
        ];
        for &(tool, field, d) in DEFAULTS {
            let v = (d * 3).div_ceil(2);
            assert!(
                hosted(tool, json!({ field: v })).is_ok(),
                "{tool}.{field} = {v}"
            );
        }
        for (tool, field, v) in [
            ("panel_hdfe", "tolerance", 1e-8),
            ("feglm", "tolerance", 1e-8),
            ("cox_ph", "tolerance", 1e-9),
            ("aft", "tolerance", 1e-9),
            ("synthetic_control", "tolerance", 1e-6),
            ("ml_svm", "tolerance", 1e-3),
            ("stats_loglin", "eps", 0.1),
            ("descriptive_medpolish", "eps", 0.01),
            ("treatment_sbw", "balance_tol", 0.0),
            ("ml_dbscan", "eps", 1e-30),
        ] {
            assert!(hosted(tool, json!({ field: v })).is_ok(), "{tool}.{field}");
        }
    }

    #[test]
    fn every_capped_scalar_rule_refuses_1e15_and_off_accepts_it() {
        for r in PARAM_RULES {
            let Rule::Max(f) = r.rule else { continue };
            if f == Family::StringBytes || r.path.contains(['.', '[']) {
                continue;
            }
            let tool = if r.tool == "*" { "__generic__" } else { r.tool };
            let args = json!({ r.path: 1_000_000_000_000_000u64 });
            assert_eq!(
                hosted(tool, args.clone()).unwrap_err().parameter,
                r.path,
                "{tool}.{}",
                r.path
            );
            assert!(Limits::off().check_tool_arguments(tool, &args).is_ok());
        }
    }

    #[test]
    fn table_is_well_formed() {
        let mut seen = std::collections::HashSet::new();
        for r in PARAM_RULES {
            assert!(
                seen.insert((r.tool, r.path, format!("{:?}", r.rule))),
                "duplicate rule {}.{}",
                r.tool,
                r.path
            );
            assert!(!r.path.is_empty() && !r.path.starts_with('.') && !r.path.ends_with('.'));
            if let Rule::Exempt(reason) = r.rule {
                assert!(reason.len() >= 10, "{}.{} needs a reason", r.tool, r.path);
            }
        }
        assert!(
            rules_for("hypothesis_poisson")
                .iter()
                .all(|r| !(r.path == "x" && r.tool == "*"))
        );
        assert_eq!(ROW_RULES.len(), 16);
    }
}

/// Fail-closed guarantee: every cost-bearing field of every registered tool has a rule.
#[cfg(all(test, feature = "http"))]
mod schema_tests {
    use super::super::{Limits, Rule, rules_for};
    use super::{PARAM_RULES, ROW_RULES};
    use crate::server::AnalyticsServer;
    use serde_json::{Value, json};

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Kind {
        Integer,
        Tolerance,
        Array,
    }

    /// Most `$ref` hops followed from one schema.
    const MAX_REF_HOPS: usize = 16;
    /// Deepest nesting of object and array-of-object properties walked.
    const MAX_DEPTH: usize = 8;

    /// Collects `schema` and every schema it stands for: `$ref` targets (`#/definitions/...` and
    /// `#/$defs/...`, resolved against `root`) and `anyOf` / `oneOf` / `allOf` variants.
    fn variants<'a>(root: &'a Value, schema: &'a Value, hops: usize, out: &mut Vec<&'a Value>) {
        if hops > MAX_REF_HOPS {
            return;
        }
        out.push(schema);
        if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
            let target = if let Some(name) = reference.strip_prefix("#/definitions/") {
                &root["definitions"][name]
            } else if let Some(name) = reference.strip_prefix("#/$defs/") {
                &root["$defs"][name]
            } else {
                &Value::Null
            };
            if !target.is_null() {
                variants(root, target, hops + 1, out);
            }
        }
        for key in ["anyOf", "oneOf", "allOf"] {
            for variant in schema
                .get(key)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                variants(root, variant, hops + 1, out);
            }
        }
    }

    /// The JSON types of a set of variants, `"null"` ignored.
    fn types<'a>(variants: &[&'a Value]) -> Vec<&'a str> {
        let mut out = Vec::new();
        for v in variants {
            match v.get("type") {
                Some(Value::String(t)) => out.push(t.as_str()),
                Some(Value::Array(ts)) => out.extend(ts.iter().filter_map(Value::as_str)),
                _ => {}
            }
        }
        out.retain(|t| *t != "null");
        out
    }

    fn walk(
        root: &Value,
        schema: &Value,
        prefix: &str,
        depth: usize,
        out: &mut Vec<(String, Kind)>,
    ) {
        if depth > MAX_DEPTH {
            return;
        }
        let mut own = Vec::new();
        variants(root, schema, 0, &mut own);
        for props in own
            .iter()
            .filter_map(|v| v.get("properties").and_then(Value::as_object))
        {
            for (name, prop) in props {
                let path = format!("{prefix}{name}");
                let mut pvs = Vec::new();
                variants(root, prop, 0, &mut pvs);
                let ts = types(&pvs);
                let mut push = |kind: Kind| {
                    if !out.iter().any(|(p, k)| *p == path && *k == kind) {
                        out.push((path.clone(), kind));
                    }
                };
                if ts.contains(&"integer") {
                    push(Kind::Integer);
                }
                if ts.contains(&"number") && (name.contains("tol") || name.contains("eps")) {
                    push(Kind::Tolerance);
                }
                if ts.contains(&"array") {
                    push(Kind::Array);
                    for items in pvs.iter().filter_map(|v| v.get("items")) {
                        walk(root, items, &format!("{path}[]."), depth + 1, out);
                    }
                }
                if ts.contains(&"object") {
                    walk(root, prop, &format!("{path}."), depth + 1, out);
                }
            }
        }
    }

    /// Walks one tool's input schema and returns every (path, kind) that must be classified.
    fn cost_properties(schema: &Value) -> Vec<(String, Kind)> {
        let mut out = Vec::new();
        walk(schema, schema, "", 0, &mut out);
        out
    }

    fn tools() -> Vec<(String, Value)> {
        AnalyticsServer::new()
            .list_tools()
            .into_iter()
            .map(|t| (t.name, t.input_schema))
            .collect()
    }

    #[test]
    fn every_cost_property_of_every_tool_is_classified() {
        let tools = tools();
        assert!(tools.len() >= 250, "walker saw {} tools", tools.len());
        let (mut ints, mut arrays, mut tols, mut missing) = (0, 0, 0, Vec::new());
        for (tool, schema) in &tools {
            let rules = rules_for(tool);
            for (path, kind) in cost_properties(schema) {
                match kind {
                    Kind::Integer => ints += 1,
                    Kind::Array => arrays += 1,
                    Kind::Tolerance => tols += 1,
                }
                let ok = rules
                    .iter()
                    .filter(|r| r.path == path)
                    .any(|r| match (kind, r.rule) {
                        (_, Rule::Exempt(_)) => true,
                        (Kind::Integer, Rule::Max(_)) => true,
                        (Kind::Tolerance, Rule::Tolerance | Rule::Max(_)) => true,
                        (
                            Kind::Array,
                            Rule::ArrayLen(_)
                            | Rule::ArrayElemMax(_)
                            | Rule::UniqueList(_)
                            | Rule::SquareLen,
                        ) => true,
                        _ => false,
                    });
                if !ok {
                    missing.push(format!("{tool}.{path} ({kind:?})"));
                }
            }
        }
        assert!(
            missing.is_empty(),
            "request fields without a limits rule (add a ParamRule or a reasoned Exempt in \
             tools/limits/rules.rs):\n{}",
            missing.join("\n")
        );
        eprintln!(
            "schema walker: {} tools, {ints} integer, {arrays} array, {tols} tolerance fields",
            tools.len()
        );
        // Sanity: a schema format change must not silently hide properties
        // (counts at d6a3a3b: 191 / 199 / 10).
        assert!(
            ints >= 150 && arrays >= 150 && tols >= 8,
            "walker saw {ints} integer, {arrays} array, {tols} tolerance fields"
        );
    }

    #[test]
    fn rules_point_at_real_tools_and_fields() {
        let tools: std::collections::HashMap<String, Value> = tools().into_iter().collect();
        for r in PARAM_RULES {
            let first = r.path.split(['.', '[']).next().unwrap();
            if r.tool == "*" {
                assert!(
                    tools.values().any(|s| s["properties"].get(first).is_some()),
                    "generic rule `{}` matches no tool",
                    r.path
                );
            } else {
                let s = tools
                    .get(r.tool)
                    .unwrap_or_else(|| panic!("rule for unknown tool `{}`", r.tool));
                assert!(
                    s["properties"].get(first).is_some(),
                    "{}.{} is not a field",
                    r.tool,
                    r.path
                );
            }
        }
        for rr in ROW_RULES {
            let s = tools
                .get(rr.tool)
                .unwrap_or_else(|| panic!("row rule for unknown tool `{}`", rr.tool));
            assert_eq!(
                s["properties"][rr.dataset_field]["type"], "string",
                "{}",
                rr.tool
            );
            assert!(
                s["required"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|v| v == rr.dataset_field),
                "{}",
                rr.tool
            );
        }
    }

    #[test]
    fn small_typical_arguments_pass_for_every_tool_in_hosted_profile() {
        let h = Limits::hosted();
        for (tool, schema) in tools() {
            let mut args = serde_json::Map::new();
            for (path, kind) in cost_properties(&schema) {
                if path.contains(['.', '[']) {
                    continue;
                }
                args.insert(
                    path,
                    match kind {
                        Kind::Integer => json!(1),
                        Kind::Tolerance => json!(1e-6),
                        Kind::Array => json!([]),
                    },
                );
            }
            let args = Value::Object(args);
            assert!(
                h.check_tool_arguments(&tool, &args).is_ok(),
                "{tool}: {:?}",
                h.check_tool_arguments(&tool, &args)
            );
            assert!(Limits::off().check_tool_arguments(&tool, &args).is_ok());
        }
    }
}
