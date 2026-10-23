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
//    every array field (arrays nested in arrays and tuples included), the integer elements of
//    every array (at the element path `a[]`), every untyped (any-JSON) field and every
//    `tol`/`eps` float field of every tool to be matched here; the module documentation
//    (`Classifying a new request field`) says which rules match which.

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
            "columns[].distribution",
            Exempt(
                "free-form distribution object; its cost fields categories, weights and value are \
                 ruled separately",
            ),
        ),
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
        p(
            "generate_report",
            "sections[].content[].stats[]",
            Exempt(
                "[key, value] pair; the handler reads the first two strings and ignores the rest",
            ),
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
        p(
            "hypothesis_poisson",
            "x[]",
            Exempt("event counts used only in closed-form distribution functions"),
        ),
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
            "munge_cast",
            "casts[]",
            Exempt("[column, dtype] pair; the handler refuses any other length"),
        ),
        p(
            "munge_diff",
            "periods",
            Exempt("polars shift; cost is bounded by the column length"),
        ),
        p("munge_filter", "value", Max(StringBytes)),
        p("munge_group_by", "aggs", ArrayLen(ListLength)),
        p(
            "munge_group_by",
            "aggs[]",
            Exempt(
                "[column, function] pair; the handler reads the first two strings and ignores the rest",
            ),
        ),
        p("munge_join", "suffix", Max(StringBytes)),
        p(
            "munge_lag_lead",
            "periods",
            Exempt("polars shift; cost is bounded by the column length"),
        ),
        p("munge_mutate", "left", Max(StringBytes)),
        p("munge_mutate", "right", Max(StringBytes)),
        p("munge_rename", "renames", ArrayLen(ListLength)),
        p(
            "munge_rename",
            "renames[]",
            Exempt("[old, new] pair; the handler reads the first two strings and ignores the rest"),
        ),
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
        p(
            "stats_loglin",
            "margins[][]",
            Exempt("factor indices, each checked against the number of factors"),
        ),
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
            Exempt("[start, end] pair; serde refuses any other length"),
        ),
        p(
            "synthetic_control",
            "predictors[].time_window[]",
            Exempt("time value that selects rows"),
        ),
        p(
            "synthetic_control",
            "optimization_window",
            Exempt("[start, end] pair; serde refuses any other length"),
        ),
        p(
            "synthetic_control",
            "optimization_window[]",
            Exempt("time value that selects rows"),
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
        p(
            "viz_dendrogram",
            "linkage_matrix[]",
            Exempt("linkage row; the handler reads the first four values and ignores the rest"),
        ),
        p("viz_dendrogram", "labels", ArrayLen(ArrayLength)),
    ]
};

/// Tools whose dataset height is limited, by [`RowClass`]. All take the dataset in `dataset`.
///
/// Deliberately not row-limited, because their cost is close to linear in the row count:
/// `hypothesis_cor_test` (Kendall's tau by sort and merge), `stats_ecdf` (evaluation by binary
/// search), `timeseries_cpgram` (an FFT periodogram), `ml_ctree` (splits from one sort and
/// running sums), `treatment_twang` (features sorted once, stumps swept with prefix sums),
/// `treatment_sbw` (solved in the dimension of the balance constraints), `bacon_decomp` (cell
/// sums, no dense design), `regression_gls` (O(n) whitening for each of its correlation
/// structures), `panel_gls` (a T x T covariance with T at most N, linear work per entity),
/// `moran_test` and `spatial_lm_tests_tool` (traces from the sparse W) and `spatial_neighbors`
/// (a k-d tree, bounded by `spatial_links`).
pub static ROW_RULES: &[RowRule] = &[
    // Dense: `hierarchical` keeps the condensed n(n-1)/2 f64 distance matrix; `ml_cutree` builds
    // the same tree before cutting it.
    row("ml_hierarchical", RowClass::Dense),
    row("ml_cutree", RowClass::Dense),
    // Dense class: each neighbourhood query can return all n rows, so clustering takes O(n^2)
    // time when eps covers the data (memory is O(n)).
    row("ml_dbscan", RowClass::Dense),
    // The penalty is stored as bands and each solve is O(n) since the banded rewrite, so neither
    // memory nor time grows with n^2; kept in the Dense class as in the design table, re-audit
    // pending.
    row("regression_smooth_spline", RowClass::Dense),
    // Cubic: exact t-SNE (no Barnes-Hut) keeps the n x n affinity and similarity matrices and
    // does O(n^2) work per iteration.
    row("ml_tsne", RowClass::Cubic),
    // Cubic: `cmdscale` double-centres an n x n matrix and runs a full O(n^3) eigensolve on it.
    row("ml_cmdscale", RowClass::Cubic),
    // Cubic: the log-determinant uses the eigenvalues of the dense n x n W
    // (`SpatialWeights::try_eigenvalues`; R parity, `method = "eigen"`).
    row("sar_model", RowClass::Cubic),
    row("sem_model", RowClass::Cubic),
    // Cubic: the range of the spatial parameter comes from the eigenvalues of the dense W
    // (`try_rho_range`); the sampler and the impacts run on the sparse W.
    row("sar_probit_model", RowClass::Cubic),
    row("sem_probit_model", RowClass::Cubic),
    // Cubic: the spatial-lag models and the within spatial-error model take the eigenvalues of
    // the dense N x N W of the entities.
    row("spatial_panel_ml", RowClass::Cubic),
    // The GMM estimator forms P_Z X in O(NT k) memory instead of the NT x NT projection and needs
    // no eigenvalues; kept in the Cubic class as in the design table, re-audit pending.
    row("spatial_panel_gmm", RowClass::Cubic),
    // Quadratic: nearest-neighbour search over all controls.
    row("propensity_matching", RowClass::Quadratic),
    // Quadratic: with matching, each of the B permutations compares every unit with every unit
    // of the other group, O(B n^2).
    row("heterogeneity_test", RowClass::Quadratic),
    // Quadratic: an SMO sweep examines each of the n rows, and every step it takes updates the
    // cached errors of all n rows.
    row("ml_svm", RowClass::Quadratic),
    // Quadratic: each of the n points is fitted on its ceil(span * n) nearest neighbours.
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

    /// Values far above each family cap are refused in the hosted profile, naming the parameter
    /// and the override key, and accepted when limits are off.
    #[test]
    fn values_far_above_each_cap_are_refused_in_hosted_profile() {
        let cases: &[(&str, Value, &str, &str)] = &[
            (
                "timeseries_arma_to_ma",
                json!({"lag_max": 10_000_000_000u64}),
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
                json!({"cv_folds": 1_000_000_000_000_000_000u64}),
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
                json!({"width": 65_536}),
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
        // shape-driven sizes: square results, duplicate names, array lengths, string bytes
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
        // The element Exempts of these arrays (`x[]`, `margins[][]`) do not bound their length;
        // these rules do.
        for (tool, path) in [("hypothesis_poisson", "x"), ("stats_loglin", "margins[]")] {
            assert!(
                rules_for(tool)
                    .iter()
                    .any(|r| r.path == path && matches!(r.rule, Rule::ArrayLen(_))),
                "{tool}.{path} needs a length rule"
            );
        }
        assert_eq!(ROW_RULES.len(), 16);
    }
}

/// Fail-closed guarantee: every cost-bearing field of every registered tool has a rule.
#[cfg(all(test, feature = "http"))]
mod schema_tests {
    use super::super::{Family, Limits, ParamRule, Rule, rules_for};
    use super::{PARAM_RULES, ROW_RULES};
    use crate::server::AnalyticsServer;
    use serde_json::{Value, json};

    /// What a schema walk found at a path, and so which rules can classify it.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Kind {
        /// An integer field.
        Integer,
        /// A float field whose name contains `tol` or `eps`.
        Tolerance,
        /// An array: a field, an element of an array (`a[]`) or a tuple.
        Array,
        /// Integer elements of an array, recorded at the element path (`a[]` for the elements of
        /// the array `a`), so that a rule for them is never mistaken for the array's length rule.
        IntegerElems,
        /// A map whose values are integers, arrays, objects or any JSON. No rule path can
        /// address map values, so only `Exempt` classifies it.
        MapValues,
        /// A field or element whose schema has no `type`, `enum` or `const` (a
        /// `serde_json::Value`): any JSON. Only `Exempt` classifies it; its cost sub-fields need
        /// rules of their own by path, because the schema does not list them.
        Untyped,
        /// A path nested deeper than [`MAX_DEPTH`], where the walk stopped. No rule classifies
        /// it, so the schema test fails instead of skipping what lies below.
        TooDeep,
    }

    /// True when `rule`, on the path where the walk found `kind`, checks that value (or exempts
    /// it). `Max(StringBytes)` checks only strings. `ArrayElemMax` sits on the array's path, not
    /// on the element path of [`Kind::IntegerElems`]; [`classified`] handles it.
    fn classifies(kind: Kind, rule: Rule) -> bool {
        match (kind, rule) {
            (Kind::TooDeep, _) => false,
            (_, Rule::Exempt(_)) => true,
            (Kind::Integer | Kind::IntegerElems | Kind::Tolerance, Rule::Max(f)) => {
                f != Family::StringBytes
            }
            (Kind::Tolerance, Rule::Tolerance) => true,
            (Kind::Array, Rule::ArrayLen(_) | Rule::UniqueList(_) | Rule::SquareLen) => true,
            _ => false,
        }
    }

    /// True when one of a tool's `rules` classifies `kind` at `path`: a rule on `path` itself
    /// ([`classifies`]) or, for the integer elements at `a[]`, `ArrayElemMax` on the array `a`.
    fn classified(rules: &[&ParamRule], path: &str, kind: Kind) -> bool {
        let on_path = rules
            .iter()
            .any(|r| r.path == path && classifies(kind, r.rule));
        let by_array = kind == Kind::IntegerElems
            && path.strip_suffix("[]").is_some_and(|array| {
                rules
                    .iter()
                    .any(|r| r.path == array && matches!(r.rule, Rule::ArrayElemMax(_)))
            });
        on_path || by_array
    }

    #[test]
    fn classification_matches_what_check_value_checks() {
        use Family::{Lags, ListLength, StringBytes};
        assert!(classifies(Kind::Integer, Rule::Max(Lags)));
        assert!(!classifies(Kind::Integer, Rule::Max(StringBytes)));
        assert!(classifies(Kind::Tolerance, Rule::Tolerance));
        assert!(classifies(Kind::Tolerance, Rule::Max(Lags)));
        assert!(!classifies(Kind::Tolerance, Rule::Max(StringBytes)));
        assert!(classifies(Kind::Array, Rule::ArrayLen(ListLength)));
        assert!(classifies(Kind::Array, Rule::UniqueList(ListLength)));
        assert!(classifies(Kind::Array, Rule::SquareLen));
        assert!(!classifies(Kind::Array, Rule::ArrayElemMax(Lags)));
        assert!(classifies(Kind::IntegerElems, Rule::Max(Lags)));
        assert!(!classifies(Kind::IntegerElems, Rule::Max(StringBytes)));
        assert!(!classifies(Kind::IntegerElems, Rule::ArrayLen(ListLength)));
        assert!(!classifies(Kind::MapValues, Rule::ArrayLen(ListLength)));
        assert!(!classifies(Kind::Untyped, Rule::Max(Lags)));
        for kind in [
            Kind::Integer,
            Kind::Tolerance,
            Kind::Array,
            Kind::IntegerElems,
            Kind::MapValues,
            Kind::Untyped,
        ] {
            assert!(classifies(kind, Rule::Exempt("reviewed")));
        }
        assert!(!classifies(Kind::TooDeep, Rule::Exempt("reviewed")));

        // An element rule classifies the elements only, never the array's length.
        let at = |path, rule| ParamRule {
            tool: "t",
            path,
            rule,
        };
        let elem_max = [at("a", Rule::ArrayElemMax(Lags))];
        let elem_exempt = [at("a[]", Rule::Exempt("reviewed"))];
        let elem_cap = [at("a[]", Rule::Max(Lags))];
        for table in [&elem_max, &elem_exempt, &elem_cap] {
            let rules: Vec<&ParamRule> = table.iter().collect();
            assert!(classified(&rules, "a[]", Kind::IntegerElems), "{table:?}");
            assert!(!classified(&rules, "a", Kind::Array), "{table:?}");
        }
        let length = [at("a", Rule::ArrayLen(ListLength))];
        let rules: Vec<&ParamRule> = length.iter().collect();
        assert!(classified(&rules, "a", Kind::Array));
        assert!(!classified(&rules, "a[]", Kind::IntegerElems));
        // `ArrayElemMax` on `a[]` covers the integers of the inner arrays, at `a[][]`
        let inner = [at("a[]", Rule::ArrayElemMax(Lags))];
        let rules: Vec<&ParamRule> = inner.iter().collect();
        assert!(classified(&rules, "a[][]", Kind::IntegerElems));
        assert!(!classified(&rules, "a[]", Kind::IntegerElems));

        // The runtime agrees: StringBytes ignores numbers, ArrayElemMax ignores an array's
        // length, and a Max on `a[]` checks each element.
        let h = Limits::hosted();
        let t = &[ParamRule {
            tool: "*",
            path: "n",
            rule: Rule::Max(StringBytes),
        }];
        assert!(h.check_with("t", &json!({"n": 1e30}), t).is_ok());
        let t = &[ParamRule {
            tool: "*",
            path: "a",
            rule: Rule::ArrayElemMax(Lags),
        }];
        assert!(
            h.check_with("t", &json!({"a": vec![1; 100_000]}), t)
                .is_ok()
        );
        let t = &[ParamRule {
            tool: "*",
            path: "a[]",
            rule: Rule::Max(Lags),
        }];
        assert!(h.check_with("t", &json!({"a": [1, 1_000]}), t).is_ok());
        assert!(h.check_with("t", &json!({"a": [1, 1_001]}), t).is_err());
    }

    #[test]
    fn walker_reports_nested_arrays_tuple_elements_and_maps() {
        let schema = json!({
            "type": "object",
            "properties": {
                "counts": {"type": "array", "items": {"type": "integer"}},
                "pairs": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                "grid": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                "window": {"type": "array",
                           "prefixItems": [{"type": "integer"}, {"type": "integer"}]},
                "legacy": {"type": "array",
                           "items": [{"type": "string"}, {"$ref": "#/$defs/Lens"}]},
                "reps": {"type": "object", "additionalProperties": {"type": "integer"}},
                "names": {"type": "object", "additionalProperties": {"type": "string"}},
                "tables": {"type": "array",
                           "items": {"type": "object", "additionalProperties": {"type": "array"}}},
                "closed": {"type": "object", "additionalProperties": false,
                           "properties": {"k": {"type": ["integer", "null"]}}},
                "spec": true,
                "options": {"description": "any JSON"},
                "anys": {"type": "array", "items": true},
                "anymap": {"type": "object", "additionalProperties": true},
                "pair": {"type": "array", "prefixItems": [{"type": "number"}, {"type": "number"}],
                         "items": false},
                "level": {"enum": ["low", "high"]}
            },
            "$defs": {"Lens": {"type": "array", "items": {"type": "integer"}}}
        });
        let mut found = cost_properties(&schema);
        found.sort_by(|a, b| a.0.cmp(&b.0));
        let found: Vec<(&str, Kind)> = found.iter().map(|(p, k)| (p.as_str(), *k)).collect();
        assert_eq!(
            found,
            [
                ("anymap", Kind::MapValues),
                ("anys", Kind::Array),
                ("anys[]", Kind::Untyped),
                ("closed.k", Kind::Integer),
                ("counts", Kind::Array),
                ("counts[]", Kind::IntegerElems),
                ("grid", Kind::Array),
                ("grid[]", Kind::Array),
                ("grid[][]", Kind::IntegerElems),
                ("legacy", Kind::Array),
                ("legacy[]", Kind::Array),
                ("legacy[][]", Kind::IntegerElems),
                ("options", Kind::Untyped),
                ("pair", Kind::Array),
                ("pairs", Kind::Array),
                ("pairs[]", Kind::Array),
                ("reps", Kind::MapValues),
                ("spec", Kind::Untyped),
                ("tables", Kind::Array),
                ("tables[]", Kind::MapValues),
                ("window", Kind::Array),
                ("window[]", Kind::IntegerElems),
            ]
        );
    }

    #[test]
    fn walker_records_a_path_nested_too_deep_and_no_rule_classifies_it() {
        // A recursive type nests without end; the walk stops below MAX_DEPTH and says so.
        let schema = json!({
            "type": "object",
            "properties": {"node": {"$ref": "#/$defs/Node"}},
            "$defs": {"Node": {"type": "object", "properties": {
                "reps": {"type": "integer"},
                "child": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]}
            }}}
        });
        let found = cost_properties(&schema);
        let mut deep: Vec<&str> = found
            .iter()
            .filter(|(_, k)| *k == Kind::TooDeep)
            .map(|(p, _)| p.as_str())
            .collect();
        deep.sort_unstable();
        let last = format!("node{}.", ".child".repeat(MAX_DEPTH));
        assert_eq!(deep, [format!("{last}child"), format!("{last}reps")]);
        // the levels above are walked as usual
        assert!(found.contains(&(
            format!("node{}.reps", ".child".repeat(MAX_DEPTH - 1)),
            Kind::Integer
        )));
        // not even an Exempt classifies a path the walk did not finish
        let path: &'static str = deep[0].to_string().leak();
        let exempt = ParamRule {
            tool: "t",
            path,
            rule: Rule::Exempt("reviewed"),
        };
        assert!(!classified(&[&exempt], path, Kind::TooDeep));
    }

    /// Most `$ref` hops followed from one schema.
    const MAX_REF_HOPS: usize = 16;
    /// Deepest nesting of objects and arrays walked. Each object property and each array level
    /// takes one level; a value nested deeper is recorded as [`Kind::TooDeep`].
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

    /// True when no variant says what the value is (no `type`, `enum` or `const`): the schema
    /// of a `serde_json::Value` (`true` or `{}`), which accepts any JSON.
    fn untyped(variants: &[&Value]) -> bool {
        !variants
            .iter()
            .any(|v| ["type", "enum", "const"].iter().any(|k| v.get(k).is_some()))
    }

    fn push(out: &mut Vec<(String, Kind)>, path: &str, kind: Kind) {
        if !out.iter().any(|(p, k)| p == path && *k == kind) {
            out.push((path.to_string(), kind));
        }
    }

    /// The element schemas of an array: `items` (one schema, `true` for any JSON, or a list of
    /// tuple positions) and `prefixItems` (tuple positions). `items: false` (no elements beyond
    /// the tuple positions) describes none.
    fn element_schemas<'a>(variants: &[&'a Value]) -> Vec<&'a Value> {
        let mut out = Vec::new();
        for v in variants {
            match v.get("items") {
                Some(Value::Array(positions)) => out.extend(positions),
                Some(items @ (Value::Object(_) | Value::Bool(true))) => out.push(items),
                _ => {}
            }
            out.extend(
                v.get("prefixItems")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten(),
            );
        }
        out
    }

    /// Records the properties of the object schemas among `variants` under `prefix`.
    fn walk_properties(
        root: &Value,
        variants_of: &[&Value],
        prefix: &str,
        depth: usize,
        out: &mut Vec<(String, Kind)>,
    ) {
        for props in variants_of
            .iter()
            .filter_map(|v| v.get("properties").and_then(Value::as_object))
        {
            for (name, prop) in props {
                record(
                    root,
                    prop,
                    &format!("{prefix}{name}"),
                    Some(name),
                    depth,
                    out,
                );
            }
        }
    }

    /// Records what the value at `path`, described by `schema`, needs: its integers, an array
    /// (and what its elements need, at `{path}[]`), the properties of an object (under
    /// `{path}.`) and map values, or [`Kind::Untyped`]. `name` is the field name of a property
    /// and `None` for an array element, whose integers are [`Kind::IntegerElems`] and which is
    /// never a tolerance (that check goes by field name).
    fn record(
        root: &Value,
        schema: &Value,
        path: &str,
        name: Option<&str>,
        depth: usize,
        out: &mut Vec<(String, Kind)>,
    ) {
        if depth > MAX_DEPTH {
            push(out, path, Kind::TooDeep);
            return;
        }
        let mut vs = Vec::new();
        variants(root, schema, 0, &mut vs);
        if untyped(&vs) {
            push(out, path, Kind::Untyped);
        }
        let ts = types(&vs);
        if ts.contains(&"integer") {
            let kind = if name.is_some() {
                Kind::Integer
            } else {
                Kind::IntegerElems
            };
            push(out, path, kind);
        }
        if ts.contains(&"number") && name.is_some_and(|n| n.contains("tol") || n.contains("eps")) {
            push(out, path, Kind::Tolerance);
        }
        if ts.contains(&"array") {
            push(out, path, Kind::Array);
            let elements = format!("{path}[]");
            for item in element_schemas(&vs) {
                record(root, item, &elements, None, depth + 1, out);
            }
        }
        if ts.contains(&"object") {
            walk_properties(root, &vs, &format!("{path}."), depth + 1, out);
            map_values(root, &vs, path, out);
        }
    }

    /// Records [`Kind::MapValues`] at `path` when an object schema among `variants_of` is a map
    /// (`additionalProperties` given as a schema) whose values are integers, arrays, objects or
    /// any JSON.
    fn map_values(root: &Value, variants_of: &[&Value], path: &str, out: &mut Vec<(String, Kind)>) {
        for values in variants_of
            .iter()
            .filter_map(|v| v.get("additionalProperties"))
            .filter(|values| !matches!(values, Value::Bool(false)))
        {
            let mut vvs = Vec::new();
            variants(root, values, 0, &mut vvs);
            let vts = types(&vvs);
            if untyped(&vvs)
                || ["integer", "array", "object"]
                    .iter()
                    .any(|t| vts.contains(t))
            {
                push(out, path, Kind::MapValues);
            }
        }
    }

    /// Walks one tool's input schema and returns every (path, kind) that must be classified.
    fn cost_properties(schema: &Value) -> Vec<(String, Kind)> {
        let mut root = Vec::new();
        variants(schema, schema, 0, &mut root);
        let mut out = Vec::new();
        walk_properties(schema, &root, "", 0, &mut out);
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
        let (mut ints, mut arrays, mut tols, mut elems, mut maps, mut untyped) = (0, 0, 0, 0, 0, 0);
        let mut missing = Vec::new();
        for (tool, schema) in &tools {
            let rules = rules_for(tool);
            for (path, kind) in cost_properties(schema) {
                match kind {
                    Kind::Integer => ints += 1,
                    Kind::Array => arrays += 1,
                    Kind::Tolerance => tols += 1,
                    Kind::IntegerElems => elems += 1,
                    Kind::MapValues => maps += 1,
                    Kind::Untyped => untyped += 1,
                    Kind::TooDeep => {}
                }
                if !classified(&rules, &path, kind) {
                    missing.push(format!("{tool}.{path} ({kind:?})"));
                }
            }
        }
        assert!(
            missing.is_empty(),
            "request fields without a limits rule (add a ParamRule or a reasoned Exempt in \
             tools/limits/rules.rs; the limits module documentation says which rule fits which \
             kind; TooDeep means the walk stopped at MAX_DEPTH):\n{}",
            missing.join("\n")
        );
        eprintln!(
            "schema walker: {} tools, {ints} integer, {arrays} array, {tols} tolerance, {elems} \
             integer-element, {maps} map and {untyped} untyped fields",
            tools.len()
        );
        // Sanity: a schema format change must not silently hide properties (the walker saw
        // 191 integer, 207 array, 10 tolerance, 7 integer-element and 1 untyped field when it
        // learned element paths and untyped fields).
        assert!(
            ints >= 150 && arrays >= 150 && tols >= 8 && elems >= 5 && untyped >= 1,
            "walker saw {ints} integer, {arrays} array, {tols} tolerance, {elems} \
             integer-element and {untyped} untyped fields"
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
                // top-level fields, and the integer elements of a top-level array (`a[]`), which
                // the walk records after the array and whose value replaces the empty one
                let (field, value) = match kind {
                    Kind::Integer => (path.as_str(), json!(1)),
                    Kind::Tolerance => (path.as_str(), json!(1e-6)),
                    Kind::Array => (path.as_str(), json!([])),
                    Kind::IntegerElems => match path.strip_suffix("[]") {
                        Some(array) => (array, json!([1])),
                        None => continue,
                    },
                    Kind::MapValues => (path.as_str(), json!({})),
                    Kind::Untyped | Kind::TooDeep => continue,
                };
                if field.contains(['.', '[']) {
                    continue;
                }
                args.insert(field.to_string(), value);
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
