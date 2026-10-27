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
//    every float field, every array field (arrays nested in arrays and tuples included), the
//    integer and float elements of every array (at the element path `a[]`) and every untyped
//    (any-JSON) field of every tool to be matched here; the module documentation (`Classifying
//    a new request field`) says which rules match which. A float is a cost parameter when the
//    work grows with what it counts or sizes (the `n` of `power_t_test`), so it gets a `Max` like
//    an integer.

// Reasons shared by the `Exempt`s of several float fields and float elements (rule 3).
const LEVEL: &str = "confidence or significance level; it sets a quantile or the target of a \
                     bounded search, never a loop count or an allocation";
const TARGET_POWER: &str = "target power; the solvers bisect at most 100 steps whatever its value";
const EFFECT: &str = "effect size, proportion or variance; it enters the noncentrality \
                      parameter only";
const CLOSED_FORM: &str = "input of a closed-form formula; no loop or allocation depends on it";
const TRIM: &str = "trimming bound; it clips scores or weights, no loop or allocation";
const SMOOTHING: &str = "smoothing weight of each recursion step; a given value skips its \
                         optimisation";
const DISTRIBUTION: &str = "parameter of the reference distribution; it enters its CDF only";
const NULL_VALUE: &str = "null-hypothesis value; it shifts the data or the statistic";
const CUTOFF: &str = "threshold value; it splits or selects rows";
const RD_BANDWIDTH: &str = "kernel bandwidth in units of the running variable; it selects rows";
const DISPLAY: &str = "display threshold; it filters the reported weights";
const ARRAY_VALUE: &str = "value of a caller array; the array's length rule bounds the work";
const COEFFICIENT: &str = "coefficient value; the array's length rule bounds the work";
const CUT_HEIGHT: &str = "cut height; it selects where to cut the built tree";
const MIXING: &str = "elastic-net mixing weight; it changes the penalty, not the number of fits";
const PENALTY_VALUE: &str = "penalty of one fit; the lambda length rule bounds their number";
const DISTANCE_BAND: &str = "distance band; spatial_links bounds the links it produces";
const TAPER: &str = "taper proportion; core keeps it in [0, 0.5], and it reweights the series \
                     only";

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
        p("*", "conf_level", Exempt(LEVEL)),
        p("*", "confidence_level", Exempt(LEVEL)),
        p("*", "sig_level", Exempt(LEVEL)),
        p("*", "xout[]", Exempt(ARRAY_VALUE)),
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
        p("causal_impact_analysis", "alpha", Exempt(LEVEL)),
        p(
            "cleaning_rollback",
            "checkpoint_index",
            Exempt("index into existing checkpoints"),
        ),
        p("cleaning_session_apply", "value", Max(StringBytes)),
        p("cleaning_session_apply", "old_value", Max(StringBytes)),
        p("cleaning_session_apply", "filter_value", Max(StringBytes)),
        p("collaborative_tmle", "max_covariates", Max(ListLength)),
        p("collaborative_tmle", "ps_lower", Exempt(TRIM)),
        p("collaborative_tmle", "ps_upper", Exempt(TRIM)),
        p("descriptive_medpolish", "eps", Tolerance),
        p("evalue", "baseline_risk", Exempt(CLOSED_FORM)),
        p("evalue", "ci_lower", Exempt(CLOSED_FORM)),
        p("evalue", "ci_upper", Exempt(CLOSED_FORM)),
        p("evalue", "point", Exempt(CLOSED_FORM)),
        p("evalue", "se", Exempt(CLOSED_FORM)),
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
        p("gformula", "threshold_cutoff", Exempt(CUTOFF)),
        p(
            "head_dataset",
            "n",
            Exempt(
                "rows returned are bounded by the dataset height; text output by response_bytes",
            ),
        ),
        p("hypothesis_chisq_gof", "probs[]", Exempt(ARRAY_VALUE)),
        p("hypothesis_ks_test", "a", Exempt(DISTRIBUTION)),
        p("hypothesis_ks_test", "b", Exempt(DISTRIBUTION)),
        p("hypothesis_ks_test", "mean", Exempt(DISTRIBUTION)),
        p("hypothesis_ks_test", "rate", Exempt(DISTRIBUTION)),
        p("hypothesis_ks_test", "sd", Exempt(DISTRIBUTION)),
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
            "hypothesis_poisson",
            "r",
            Exempt(
                "hypothesised rate or rate ratio; it sets the parameter of the Poisson or \
                 binomial distribution whose CDF gives the p-value",
            ),
        ),
        p("hypothesis_poisson", "t[]", Exempt(ARRAY_VALUE)),
        p(
            "hypothesis_prop_trend_test",
            "trials",
            Exempt("column name, not a boosting count"),
        ),
        p(
            "hypothesis_prop_trend_test",
            "scores[]",
            Exempt(ARRAY_VALUE),
        ),
        p("hypothesis_t_test", "mu", Exempt(NULL_VALUE)),
        p("hypothesis_wilcoxon", "mu", Exempt(NULL_VALUE)),
        p("linalg_toeplitz", "column", SquareLen),
        p("linalg_toeplitz", "row", SquareLen),
        p("linalg_toeplitz", "column[]", Exempt(ARRAY_VALUE)),
        p("linalg_toeplitz", "row[]", Exempt(ARRAY_VALUE)),
        p("ltmle", "covariates", ArrayLen(ListLength)),
        p("ltmle", "ps_lower", Exempt(TRIM)),
        p("ltmle", "ps_upper", Exempt(TRIM)),
        p("mediation_analysis", "trim", Exempt(TRIM)),
        p(
            "ml_c50",
            "cf",
            Exempt("pruning confidence factor; it enters the pessimistic error estimate only"),
        ),
        p("ml_cmdscale", "k", Max(Components)),
        p(
            "ml_ctree",
            "mincriterion",
            Exempt(
                "split criterion (1 - p-value); a low value allows more splits, which the \
                 node-size minimums and maxdepth bound as at the default",
            ),
        ),
        p("ml_cubist", "neighbors", Max(Neighbors)),
        p("ml_cutree", "k", Max(Components)),
        p("ml_cutree", "cut_height", Exempt(CUT_HEIGHT)),
        p(
            "ml_dbscan",
            "eps",
            Exempt(
                "neighbourhood radius, not a convergence tolerance; dense cost is bounded by dense_rows",
            ),
        ),
        p("ml_hierarchical", "distance_threshold", Exempt(CUT_HEIGHT)),
        p("ml_kmeans", "k", Max(Components)),
        p(
            "ml_mboost",
            "nu",
            Exempt("step size; core refuses one outside (0, 1], and mstop bounds the rounds"),
        ),
        p(
            "ml_ppr",
            "bass",
            Exempt(
                "bass tone; kept in the fit's configuration, no loop or allocation depends on it",
            ),
        ),
        p(
            "ml_shap_values",
            "max_obs",
            Exempt("bounded by the dataset height"),
        ),
        p(
            "ml_svm",
            "c",
            Exempt("regularisation constant; max_iterations bounds the SMO sweeps"),
        ),
        p(
            "ml_tsne",
            "learning_rate",
            Exempt("gradient step size; max_iterations bounds the steps"),
        ),
        p(
            "ml_tsne",
            "perplexity",
            Exempt("target perplexity; the bandwidth search of each point runs 50 steps"),
        ),
        p("multivariate_mahalanobis", "center", ArrayLen(ListLength)),
        p("multivariate_mahalanobis", "center[]", Exempt(ARRAY_VALUE)),
        p("munge_bin", "bins", ArrayLen(Bins)),
        p("munge_bin", "labels", ArrayLen(Bins)),
        p("munge_bin", "labels[]", Max(StringBytes)),
        p("munge_bin", "bins[]", Exempt(ARRAY_VALUE)),
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
            "negbin",
            "init_theta",
            Exempt("ignored by core: run_negbin does not use a starting dispersion"),
        ),
        p("power_anova_test", "groups", Max(DesignGroups)),
        p("power_anova_test", "n", Max(PowerSampleSize)),
        p("power_anova_test", "between_var", Exempt(EFFECT)),
        p("power_anova_test", "within_var", Exempt(EFFECT)),
        p("power_anova_test", "power", Exempt(TARGET_POWER)),
        p(
            "power_prop_test",
            "n",
            Exempt(
                "per-group sample size of a closed-form normal power; the n solver bisects at \
                 most 100 steps",
            ),
        ),
        p("power_prop_test", "p1", Exempt(EFFECT)),
        p("power_prop_test", "p2", Exempt(EFFECT)),
        p("power_prop_test", "power", Exempt(TARGET_POWER)),
        p("power_t_test", "n", Max(PowerSampleSize)),
        p("power_t_test", "delta", Exempt(EFFECT)),
        p(
            "power_t_test",
            "sd",
            Exempt(
                "standard deviation; it scales the standard error and the delta search bracket, \
                 which the solver bisects in at most 100 steps",
            ),
        ),
        p("power_t_test", "power", Exempt(TARGET_POWER)),
        p(
            "preview_cleaning",
            "sample_size",
            Exempt("bounded by the number of changed rows"),
        ),
        p("propensity_matching", "ratio", Max(Neighbors)),
        p("propensity_matching", "n_subclasses", Max(Bins)),
        p(
            "propensity_matching",
            "caliper",
            Exempt("maximum match distance; it filters candidate matches"),
        ),
        p("rd_bw", "p", Max(ModelOrder)),
        p("rd_bw", "cutoff", Exempt(CUTOFF)),
        p("rd_estimate", "p", Max(ModelOrder)),
        p("rd_estimate", "b", Exempt(RD_BANDWIDTH)),
        p("rd_estimate", "cutoff", Exempt(CUTOFF)),
        p("rd_estimate", "h", Exempt(RD_BANDWIDTH)),
        p("rd_estimate", "level", Exempt(LEVEL)),
        p("rd_fuzzy", "p", Max(ModelOrder)),
        p("rd_fuzzy", "cutoff", Exempt(CUTOFF)),
        p("rd_fuzzy", "h", Exempt(RD_BANDWIDTH)),
        p("rd_fuzzy", "level", Exempt(LEVEL)),
        p("rd_multi", "p", Max(ModelOrder)),
        p(
            "rd_multi",
            "bandwidth",
            Exempt("kernel bandwidth in units of the running variable, not a lag count"),
        ),
        p("rd_multi", "bandwidths[]", Exempt(RD_BANDWIDTH)),
        p("rd_multi", "cutoffs[]", Exempt(CUTOFF)),
        p("rd_multi", "level", Exempt(LEVEL)),
        p("regression_bgtest", "order", Max(Lags)),
        p(
            "regression_bgtest",
            "fill",
            Exempt("fill value of the initial lagged residuals"),
        ),
        p("regression_cv_glmnet", "alpha", Exempt(MIXING)),
        p("regression_glmnet", "alpha", Exempt(MIXING)),
        p("regression_glmnet", "lambda[]", Exempt(PENALTY_VALUE)),
        p(
            "regression_gls",
            "rho",
            Exempt("correlation parameter of the O(n) whitening"),
        ),
        p("regression_lasso", "lambda[]", Exempt(PENALTY_VALUE)),
        p("regression_line", "iter", Max(Iterations)),
        p("regression_loess", "degree", Max(ModelOrder)),
        p(
            "regression_loess",
            "span",
            Exempt("neighbourhood fraction; core clamps the neighbour count to the row count"),
        ),
        p("regression_nls", "start", ArrayLen(ListLength)),
        p(
            "regression_nls",
            "start[]",
            Exempt(
                "starting value of a parameter; the fit's iteration limit does not depend on it",
            ),
        ),
        p(
            "regression_quantreg",
            "tau",
            Exempt("quantile in (0, 1); one fit whatever its value"),
        ),
        p(
            "regression_quantreg",
            "taus[]",
            Exempt("quantile in (0, 1); one fit per value, and taus bounds their number"),
        ),
        p("regression_resettest", "powers", ArrayLen(ListLength)),
        p("regression_resettest", "powers", ArrayElemMax(ModelOrder)),
        p("regression_ridge", "lambda[]", Exempt(PENALTY_VALUE)),
        p(
            "regression_smooth_spline",
            "df",
            Exempt("target equivalent degrees of freedom; a 30-step search finds its penalty"),
        ),
        p(
            "regression_smooth_spline",
            "spar",
            Exempt("smoothing parameter; it sets the penalty of one fit"),
        ),
        p(
            "regression_supsmu",
            "bass",
            Exempt(
                "bass tone of the super smoother; core clamps the span it gives to the row count",
            ),
        ),
        p(
            "regression_supsmu",
            "span",
            Exempt("span fraction; core clamps the span to (0, 1] of the row count"),
        ),
        p(
            "scpi",
            "treatment_time",
            Exempt("time value that selects rows"),
        ),
        p("scpi", "alpha", Exempt(LEVEL)),
        p(
            "scpi",
            "lambda",
            Exempt(
                "penalty weight of the weight constraint; the solver's iteration limit does not \
                 depend on it",
            ),
        ),
        p("scpi", "weight_threshold", Exempt(DISPLAY)),
        p(
            "search_tools",
            "limit",
            Exempt("the handler clamps it to 25"),
        ),
        p("sensemakr", "alpha", Exempt(LEVEL)),
        p("sensemakr", "kd", Exempt(CLOSED_FORM)),
        p("sensemakr", "ky", Exempt(CLOSED_FORM)),
        p("sensemakr", "q", Exempt(CLOSED_FORM)),
        p("spatial_neighbors", "k", Max(Neighbors)),
        p("spatial_neighbors", "d_max", Exempt(DISTANCE_BAND)),
        p("spatial_neighbors", "d_min", Exempt(DISTANCE_BAND)),
        p(
            "staggered_did",
            "base_period",
            Exempt("relative period offset"),
        ),
        p("stats_density", "n", Max(GridPoints)),
        p(
            "stats_density",
            "bw",
            Exempt("kernel bandwidth; the grid has n points whatever its value"),
        ),
        p("stats_ecdf", "at[]", Exempt(ARRAY_VALUE)),
        p("stats_iqr", "qtype", Exempt("quantile type selector 1-9")),
        p("stats_loglin", "eps", Tolerance),
        p("stats_loglin", "margins", ArrayLen(ListLength)),
        p("stats_loglin", "margins[]", ArrayLen(ListLength)),
        p(
            "stats_loglin",
            "margins[][]",
            Exempt("factor indices, each checked against the number of factors"),
        ),
        p("stats_mad", "constant", Exempt(CLOSED_FORM)),
        p("stats_se_contrast", "contrasts", ArrayLen(ListLength)),
        p("stats_se_contrast", "contrasts[]", ArrayLen(ListLength)),
        p("stats_se_contrast", "contrasts[][]", Exempt(ARRAY_VALUE)),
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
        p(
            "synthetic_control",
            "custom_v_weights[]",
            Exempt(ARRAY_VALUE),
        ),
        p("synthetic_control", "weight_threshold", Exempt(DISPLAY)),
        p("timeseries_acf_to_ar", "acf", ArrayLen(Lags)),
        p("timeseries_acf_to_ar", "acf[]", Exempt(ARRAY_VALUE)),
        p("timeseries_ar", "order", Max(Lags)),
        p("timeseries_arima_sim", "n", Max(GeneratedRows)),
        p("timeseries_arima_sim", "d", Max(ModelOrder)),
        p("timeseries_arima_sim", "ar", ArrayLen(ModelOrder)),
        p("timeseries_arima_sim", "ma", ArrayLen(ModelOrder)),
        p("timeseries_arima_sim", "ar[]", Exempt(COEFFICIENT)),
        p("timeseries_arima_sim", "ma[]", Exempt(COEFFICIENT)),
        p(
            "timeseries_arima_sim",
            "sd",
            Exempt("innovation standard deviation; it scales the values"),
        ),
        p("timeseries_arma_acf", "ar", ArrayLen(ModelOrder)),
        p("timeseries_arma_acf", "ma", ArrayLen(ModelOrder)),
        p("timeseries_arma_acf", "ar[]", Exempt(COEFFICIENT)),
        p("timeseries_arma_acf", "ma[]", Exempt(COEFFICIENT)),
        p("timeseries_arma_to_ma", "ar", ArrayLen(ModelOrder)),
        p("timeseries_arma_to_ma", "ma", ArrayLen(ModelOrder)),
        p("timeseries_arma_to_ma", "ar[]", Exempt(COEFFICIENT)),
        p("timeseries_arma_to_ma", "ma[]", Exempt(COEFFICIENT)),
        p("timeseries_box_test", "lag", Max(Lags)),
        p(
            "timeseries_box_test",
            "fitdf",
            Exempt("degrees-of-freedom adjustment; no allocation or loop"),
        ),
        p("timeseries_cpgram", "taper", Exempt(TAPER)),
        p("timeseries_decompose", "period", Max(Lags)),
        p("timeseries_diffinv", "lag", Max(Lags)),
        p("timeseries_diffinv", "xi", ArrayLen(ArrayLength)),
        p("timeseries_diffinv", "xi[]", Exempt(ARRAY_VALUE)),
        p("timeseries_embed", "dimension", Max(Lags)),
        p("timeseries_filter", "sides", Exempt("selector 1 or 2")),
        p("timeseries_filter", "filter", ArrayLen(SpanWidth)),
        p("timeseries_filter", "filter[]", Exempt(COEFFICIENT)),
        p(
            "timeseries_lag",
            "k",
            Exempt("core rejects |k| >= n; the output is a slice of the input"),
        ),
        p("timeseries_runmed", "k", Max(SpanWidth)),
        p("timeseries_spectrum", "spans", ArrayLen(ListLength)),
        p("timeseries_spectrum", "spans", ArrayElemMax(SpanWidth)),
        p("timeseries_spectrum", "taper", Exempt(TAPER)),
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
            "treatment_cbps",
            "balance_threshold",
            Exempt("balance threshold; it marks covariates in the balance table only"),
        ),
        p("treatment_double_ml", "trim", Exempt(TRIM)),
        p("treatment_doubly_robust", "trim", Exempt(TRIM)),
        p(
            "treatment_entropy_balance",
            "target_means",
            ArrayLen(ListLength),
        ),
        p(
            "treatment_entropy_balance",
            "target_means[]",
            Exempt(ARRAY_VALUE),
        ),
        p("treatment_ipw", "trim", Exempt(TRIM)),
        p(
            "treatment_sbw",
            "balance_tol",
            Exempt("balance tolerance; 0 (exact balance) is the default and valid"),
        ),
        p(
            "treatment_sbw",
            "balance_penalty",
            Exempt("penalty weight of approximate balance; the problem keeps its size"),
        ),
        p(
            "treatment_sbw",
            "min_weight",
            Exempt("lower bound on the weights; a constraint value"),
        ),
        p("treatment_tmle", "ps_lower", Exempt(TRIM)),
        p("treatment_tmle", "ps_upper", Exempt(TRIM)),
        p(
            "treatment_twang",
            "balance_threshold",
            Exempt("early-stopping threshold; n_trees bounds the boosting"),
        ),
        p(
            "treatment_twang",
            "shrinkage",
            Exempt("learning rate; n_trees bounds the boosting"),
        ),
        p("treatment_weightit", "trim_quantile", Exempt(TRIM)),
        p("ts_arima_fit", "p", Max(ModelOrder)),
        p("ts_arima_fit", "d", Max(ModelOrder)),
        p("ts_arima_fit", "q", Max(ModelOrder)),
        p("ts_arima_forecast", "p", Max(ModelOrder)),
        p("ts_arima_forecast", "d", Max(ModelOrder)),
        p("ts_arima_forecast", "q", Max(ModelOrder)),
        p(
            "ts_changepoint",
            "penalty",
            Exempt(
                "changepoint penalty; PELT's work is at most quadratic in the rows whatever its \
                 value, and quadratic_rows bounds the rows",
            ),
        ),
        p("ts_garch_fit", "p", Max(ModelOrder)),
        p("ts_garch_fit", "q", Max(ModelOrder)),
        p("ts_holt_winters", "period", Max(Lags)),
        p("ts_holt_winters", "alpha", Exempt(SMOOTHING)),
        p("ts_holt_winters", "beta", Exempt(SMOOTHING)),
        p("ts_holt_winters", "gamma", Exempt(SMOOTHING)),
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
        p(
            "viz_dendrogram",
            "linkage_matrix[][]",
            Exempt(
                "linkage entry: merged cluster ids, which core checks against the merges so far, a \
                 height, and a size it ignores",
            ),
        ),
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
    // The penalty is stored as bands and each solve is O(n), so neither memory nor time grows
    // with n^2; the Dense row cap is kept as a conservative bound.
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
    // no eigenvalues; the Cubic row cap is kept as a conservative bound.
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
    // Quadratic: at each t PELT examines every candidate changepoint that survived pruning, and
    // on a series without changes, or under a large penalty, it prunes none, so its work is
    // O(n^2). PELT checks the call's budget at every step, so an overrun ends at the deadline;
    // the row cap keeps the default arguments well inside it. Binary segmentation (at most 10
    // changepoints, one scan each) is linear and shares the rule.
    row("ts_changepoint", RowClass::Quadratic),
];

/// Tools that reshape a dataset from long to wide, whose output shape
/// ([`super::Limits::check_pivot`]) is bounded by `pivot_columns` and `pivot_cells` before they
/// run. The field names are the request's; the schema test checks that they name a required
/// string (`dataset`, `on`, `values`) and a required list of strings (`index`) of the tool.
///
/// `munge_pivot` (polars' pivot) puts every distinct value of `on` in a column of its own,
/// whatever the height of the input, groups every input row by its index and `on` values, and
/// allocates its rows x columns output in one buffer before the dataset quotas see the result.
pub static PIVOT_RULES: &[super::PivotRule] = &[super::PivotRule {
    tool: "munge_pivot",
    dataset_field: "dataset",
    index_field: "index",
    on_field: "on",
    values_field: "values",
}];

/// Work of an ARIMA fit as the `arima_fit_work` budget counts it:
/// `n_obs * ((p + q + 3)^2 + d)`, saturating at `u64::MAX`.
///
/// `ts_arima_fit` and `ts_arima_forecast` fit with `arima::estimate::fit` (arima 0.3.0), whose
/// optimiser never looks at the call's budget: once it starts, the governor's deadline cannot
/// stop it, so an expensive fit has to be refused before it starts. The fit minimises the
/// conditional sum of squares with L-BFGS (liblbfgs 0.1.0, More-Thuente line search): at most
/// 200 iterations, the first without a line search and each of the others a line search of at
/// most 40 evaluations, so at most 1 + 199 x 40 = 7,961 evaluations. The bound is reachable: a
/// failed line search is reverted and the next iteration starts instead of ending the fit, and
/// some series reach it. One evaluation is the objective and its forward-difference gradient,
/// `p + q + 3` passes over the differenced series. A pass costs at most about one unit per
/// observation for each of `p + q + 3` terms (the residual, its square and its storage count as
/// about three), measured on a release build over AR-only, MA-only and mixed orders; an MA term
/// costs about twice an AR term, so the unit is an MA term's. So the slowest fit the data can
/// force takes a time proportional to `n_obs * (p + q + 3)^2`. The `d` term counts the
/// differencing (`d` passes, done by the fit and again by `run_arima` for the residuals), so
/// that the count bounds every uninterruptible pass even where `model_order` is raised;
/// differencing only shortens the series, and the starting values, the residuals and the
/// forecast after the fit are a few passes more.
///
/// The time per unit holds only in ordinary floating-point arithmetic: see
/// [`ARIMA_MIN_MAGNITUDE`] for the series the budget therefore refuses as well.
pub fn arima_fit_work(n_obs: usize, p: usize, d: usize, q: usize) -> u64 {
    let terms = (p as u128).saturating_add(q as u128).saturating_add(3);
    let per_obs = terms.saturating_mul(terms).saturating_add(d as u128);
    let work = (n_obs as u128).saturating_mul(per_obs);
    u64::try_from(work).unwrap_or(u64::MAX)
}

/// The smallest magnitude a nonzero value of an ARIMA fit's series may have while the
/// `arima_fit_work` budget is set ([`super::Limits::check_arima_series`]).
///
/// The budget's time per unit ([`arima_fit_work`]) is that of ordinary floating-point
/// arithmetic. An operation whose operand or result is subnormal (nonzero and below about
/// 2.2e-308) takes a much slower path on common CPUs: on a release build a pass over a series
/// of subnormal values took up to 40 times as long as one over ordinary values. A series whose
/// values, or the differences, products and squares the fit forms from them, fall into that
/// range would make the fit run that much longer than the budget allows for.
///
/// The floor is on the column itself, because it bounds the differenced series the fit reads:
/// every value of the column is a multiple of the spacing of doubles at its smallest nonzero
/// value `m` (about `m * 2^-52`), a difference of two such multiples rounds to another one, so
/// every nonzero value of every difference is at least about `m * 2^-53`. From 1e-100 that is
/// about 1e-116. Where a value of the differenced series is nonzero, the residual the fit forms
/// at it is, when it is not zero, at least about 2^-52 of that value, about 2e-132, and its
/// square about 6e-264: ordinary numbers. So the floor covers the column, its differences, and
/// their products and squares with ordinary-sized coefficients, with a wide margin. NaN,
/// infinities and zero cost no more than other numbers and pass.
///
/// Where a value of the differenced series is zero, the floor bounds nothing. There
/// `arima::estimate::residuals` gives the residual as
/// `-(intercept + sum phi_j x_{t-j-1} + sum theta_j e_{t-j-1})`, which the coefficients the
/// optimiser tries set, and the release-build measurements behind [`arima_fit_work`] cover it,
/// not the floor. One case is known to leave the ordinary range: at an intercept of exactly 0.0
/// over a stretch of exact zeros, the MA recursion `e_t = -sum theta_j e_{t-j-1}` decays
/// geometrically into the subnormal range and, for `|theta| > 1/2`, can stay on a subnormal
/// rounding fixed point (for example 4 x 2^-1074 at `theta = 0.9`). Exact zeros pass the floor,
/// and on a release build one pass of the objective in that state was measured at about 28
/// times the time of a pass over ordinary data, a cost the budget's time per unit does not
/// cover. A whole fit is not believed to stay in that state over its many evaluations: the
/// intercept starts at the differenced series' mean, the forward-difference gradient moves it
/// off zero, and where the data are large enough to absorb that step the first line search
/// reached its minimum step, and in the runs measured the fit ended at the next one: the
/// reverted step leaves the next L-BFGS update with zero correction pairs, whose 0/0 scaling
/// made that line search fail its step check (a failed line search does not by itself end the
/// fit; see [`arima_fit_work`]). This is a residual risk: a change of the optimiser (the TS-1
/// replacement) must examine it again.
pub const ARIMA_MIN_MAGNITUDE: f64 = 1e-100;

#[cfg(test)]
mod tests {
    use super::super::{Family, LimitViolation, Limits, Rule, rules_for, struct_paths};
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
        (Family::PowerSampleSize, "power_t_test", "n"),
        (Family::DesignGroups, "power_anova_test", "groups"),
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

    /// Arguments that hold a JSON array at rule path `path`, with one element on each array
    /// level above it.
    fn array_at(path: &str) -> Value {
        let mut value = json!([1]);
        for segment in path.rsplit('.') {
            let name = segment.trim_end_matches("[]");
            for _ in 0..(segment.len() - name.len()) / 2 {
                value = json!([value]);
            }
            value = json!({ name: value });
        }
        value
    }

    /// [`struct_paths`] are the structs the check refuses as JSON arrays, so the audit log
    /// redacts by size exactly the structs that the hosted profile refuses by position.
    #[test]
    fn struct_paths_are_the_structs_the_check_refuses_as_arrays() {
        let mut tools: Vec<&str> = PARAM_RULES.iter().map(|r| r.tool).collect();
        tools.sort_unstable();
        tools.dedup();
        let mut checked = Vec::new();
        for tool in tools.into_iter().filter(|t| *t != "*") {
            for path in struct_paths(tool) {
                let args = array_at(path);
                let v = hosted(tool, args.clone()).unwrap_err();
                assert_eq!(
                    (v.parameter.as_str(), v.limit.as_str()),
                    (path, "a JSON object"),
                    "{tool}: {args}"
                );
                checked.push(format!("{tool}.{path}"));
            }
        }
        assert_eq!(
            checked,
            [
                "generate_random_data.columns[]",
                "generate_random_data.columns[].distribution",
                "generate_report.sections[]",
                "generate_report.sections[].content[]",
            ]
        );
        // Only `Exempt` rules go into a synthetic control predictor, so it is no struct here.
        assert!(struct_paths("synthetic_control").is_empty());
        assert!(struct_paths("no_such_tool").is_empty());
    }

    /// A nested request struct given by position (serde fills it from a JSON array) is refused
    /// in the hosted profile instead of passing the rules that address its fields by name; the
    /// same values by name are checked as before.
    #[test]
    fn nested_structs_given_by_position_are_refused_in_hosted_profile() {
        let items: Vec<Value> = (0..501).map(|i| json!(i.to_string())).collect();
        let text = |t: &str| json!({"content_type": "text", "text": t});
        // By name: the content cap applies; a valid section is accepted.
        assert_eq!(
            hosted(
                "generate_report",
                json!({"title": "t", "sections": [{"title": "s", "content": items}]})
            )
            .unwrap_err()
            .parameter,
            "sections[].content"
        );
        let valid = json!({"title": "t", "sections": [{"title": "s", "content": [text("hello")]}]});
        assert!(hosted("generate_report", valid).is_ok());
        let cases: &[(&str, Value, &str)] = &[
            // The section by position, with 501 content items.
            (
                "generate_report",
                json!({"title": "t", "sections": [["s", items]]}),
                "sections[]",
            ),
            // A content item by position inside a section given by name.
            (
                "generate_report",
                json!({"title": "t", "sections": [
                    {"title": "s", "content": [text("ok"), ["table", null, null, ["h"], [["r"]]]]}
                ]}),
                "sections[].content[]",
            ),
            // A column by position, and a distribution by position (its tag first).
            (
                "generate_random_data",
                json!({"n_rows": 10, "columns": [["s".repeat(5_000), {"type": "normal"}]]}),
                "columns[]",
            ),
            (
                "generate_random_data",
                json!({"n_rows": 10, "columns": [
                    {"name": "s", "distribution": ["constant_string", "x".repeat(5_000)]}
                ]}),
                "columns[].distribution",
            ),
        ];
        for (tool, args, parameter) in cases {
            let v = hosted(tool, args.clone()).unwrap_err();
            assert_eq!(
                (v.parameter.as_str(), v.limit.as_str(), v.override_key),
                (*parameter, "a JSON object", ""),
                "{tool}: {}",
                v.message
            );
            assert!(
                v.message
                    .starts_with(&format!("Parameter `{parameter}` of `{tool}` is an array")),
                "{}",
                v.message
            );
            assert!(Limits::off().check_tool_arguments(tool, args).is_ok());
        }
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

    /// The float sample size `n` of `power_t_test` and `power_anova_test` and the group count
    /// of `power_anova_test` are capped in the hosted profile: the refusal names the parameter,
    /// the cap and its override key; every value up to the cap (fractional, negative or written
    /// as an integer) passes; nothing is refused when limits are off. `power_prop_test`'s `n`
    /// is not capped.
    #[test]
    fn power_sample_sizes_and_groups_are_capped_in_hosted_profile() {
        let cases: &[(&str, &str, Value, &str, &str)] = &[
            (
                "power_t_test",
                "n",
                json!({"n": 1e7, "delta": 1.0, "sd": 1.0, "sig_level": 0.05}),
                "power_sample_size",
                "1000000",
            ),
            (
                "power_t_test",
                "n",
                json!({"n": 1e300, "delta": 1.0}),
                "power_sample_size",
                "1000000",
            ),
            (
                "power_t_test",
                "n",
                json!({"n": -1e300, "delta": 1.0}),
                "power_sample_size",
                "1000000",
            ),
            (
                "power_t_test",
                "n",
                json!({"n": 1_000_001, "delta": 1.0}),
                "power_sample_size",
                "1000000",
            ),
            (
                "power_anova_test",
                "n",
                json!({"groups": 3, "n": 1e7, "between_var": 0.25, "within_var": 1.0}),
                "power_sample_size",
                "1000000",
            ),
            (
                "power_anova_test",
                "groups",
                json!({"groups": 95, "between_var": 1.0, "within_var": 3.0, "power": 0.8}),
                "design_groups",
                "10",
            ),
            (
                "power_anova_test",
                "groups",
                json!({"groups": u64::MAX, "n": 10, "between_var": 1.0}),
                "design_groups",
                "10",
            ),
        ];
        for (tool, param, args, key, cap) in cases {
            let v = hosted(tool, args.clone()).expect_err(tool);
            assert_eq!(
                (v.parameter.as_str(), v.override_key, v.limit.as_str()),
                (*param, *key, *cap),
                "{tool}: {args}"
            );
            // The cap bounds the magnitude, and a negative value is told so.
            let limit = if v.requested.starts_with('-') {
                format!("±{cap}")
            } else {
                cap.to_string()
            };
            assert!(
                v.message.starts_with(&format!("Parameter `{param}` = "))
                    && v.message
                        .contains(&format!("limit of {limit} ({key}) for `{tool}`")),
                "{}",
                v.message
            );
            assert!(Limits::off().check_tool_arguments(tool, args).is_ok());
        }
        assert_eq!(
            hosted("power_t_test", json!({"n": -1e300, "delta": 1.0}))
                .unwrap_err()
                .message,
            format!(
                "Parameter `n` = {} is outside this server's limit of ±1000000 \
                 (power_sample_size) for `power_t_test`. Use a value of smaller magnitude.",
                json!(-1e300)
            )
        );
        // Up to the caps everything passes, including every n the solvers can return (their
        // bisection searches n in [2, 1e6]) and a solve for n with the most groups allowed.
        for args in [
            json!({"n": 1e6, "delta": 1.0}),
            json!({"n": 999_999.5, "delta": 1.0}),
            json!({"n": -1e6, "delta": 1.0}),
            json!({"n": 2, "delta": 0.5, "sd": 1.0, "sig_level": 0.05}),
            json!({"delta": 0.5, "power": 0.8}),
        ] {
            assert!(hosted("power_t_test", args.clone()).is_ok(), "{args}");
        }
        // power_prop_test computes with the normal distribution only, so its `n` is not capped:
        // a power question about ten million users per arm is answered.
        for n in [1e6, 1e7, 1e9, 1e300, u64::MAX as f64] {
            let args = json!({"n": n, "p1": 0.3, "p2": 0.5});
            assert!(hosted("power_prop_test", args.clone()).is_ok(), "{args}");
        }
        for args in [
            json!({"groups": 10, "n": 1e6, "between_var": 1.0, "within_var": 3.0}),
            json!({"groups": 10, "between_var": 0.0, "within_var": 1.0, "power": 0.8}),
            json!({"groups": 2, "n": 5.5, "between_var": 1.0}),
        ] {
            assert!(hosted("power_anova_test", args.clone()).is_ok(), "{args}");
        }
        // The override key is `power_sample_size`, so it cannot be read as the `sample_size`
        // request field of other tools; `sample_size` is not a limit key.
        let raised = Limits::hosted()
            .with_overrides(&["power_sample_size=2000000".to_string()])
            .unwrap();
        assert!(
            raised
                .check_tool_arguments("power_t_test", &json!({"n": 2e6, "delta": 1.0}))
                .is_ok()
        );
        assert!(
            Limits::hosted()
                .with_overrides(&["sample_size=2000000".to_string()])
                .is_err()
        );
    }

    /// PELT's work is quadratic in the rows when it prunes nothing, so `ts_changepoint` is
    /// row-limited by the quadratic class in the hosted profile and not at all when limits are
    /// off.
    #[test]
    fn ts_changepoint_rows_are_capped_by_the_quadratic_class() {
        let h = Limits::hosted();
        let v = h.check_rows("ts_changepoint", 50_001, 1).unwrap_err();
        assert_eq!(
            (
                v.parameter.as_str(),
                v.requested.as_str(),
                v.limit.as_str(),
                v.override_key
            ),
            ("dataset", "50001", "50000", "quadratic_rows")
        );
        assert!(
            v.message
                .contains("`ts_changepoint` has 50001 rows and 1 columns")
                && v.message.contains("at most 50000 rows"),
            "{}",
            v.message
        );
        assert!(h.check_rows("ts_changepoint", 50_000, 1).is_ok());
        for rows in [50_000, 50_001] {
            assert!(Limits::off().check_rows("ts_changepoint", rows, 1).is_ok());
        }
    }

    #[test]
    fn arima_fit_work_is_observations_times_the_squared_order_sum_plus_three_and_d() {
        use super::arima_fit_work;
        assert_eq!(arima_fit_work(1_000, 1, 0, 1), 25_000);
        assert_eq!(arima_fit_work(1_000, 1, 1, 1), 26_000);
        assert_eq!(arima_fit_work(1_000, 0, 0, 0), 9_000);
        assert_eq!(arima_fit_work(1_000, 0, 20, 0), 29_000);
        assert_eq!(arima_fit_work(10, 20, 0, 20), 18_490);
        assert_eq!(arima_fit_work(0, 20, 20, 20), 0);
        // saturating, never wrapping
        assert_eq!(arima_fit_work(2, usize::MAX, 0, 0), u64::MAX);
        assert_eq!(arima_fit_work(2, 0, usize::MAX, 0), u64::MAX);
        assert_eq!(
            arima_fit_work(usize::MAX, usize::MAX, usize::MAX, usize::MAX),
            u64::MAX
        );
    }

    /// The `arima` crate's fit cannot be stopped once it starts, so the hosted profile bounds
    /// its work before it starts, for both ARIMA tools, at every order the `model_order` cap
    /// allows (each of p, d and q from 0 to the cap); nothing is bounded when limits are off.
    #[test]
    fn arima_fit_work_is_capped_in_hosted_profile() {
        use super::arima_fit_work;
        let h = Limits::hosted();
        let cap = h.arima_fit_work.expect("hosted ARIMA work budget");
        let max_order = usize::try_from(h.cap(Family::ModelOrder).expect("hosted model_order"))
            .expect("model_order cap fits usize");
        assert_eq!(max_order, 20);
        let mut checked = 0;
        for tool in ["ts_arima_fit", "ts_arima_forecast"] {
            for p in 0..=max_order {
                for d in 0..=max_order {
                    for q in 0..=max_order {
                        let per_obs = arima_fit_work(1, p, d, q);
                        let n_max = usize::try_from(cap / per_obs).unwrap();
                        // Even at the largest orders the cap leaves enough observations to
                        // fit (p + d + q + 10), so the refusal can point to fewer of them.
                        assert!(n_max >= p + d + q + 10, "{p} {d} {q}: {n_max}");
                        assert!(h.check_arima_fit(tool, n_max, p, d, q).is_ok());
                        let v = h.check_arima_fit(tool, n_max + 1, p, d, q).unwrap_err();
                        assert!(
                            v.message
                                .contains(&format!("at most {n_max} observations at these orders")),
                            "{p} {d} {q}: {}",
                            v.message
                        );
                        assert_eq!(
                            (
                                v.tool.as_str(),
                                v.parameter.as_str(),
                                v.requested.clone(),
                                v.limit.clone(),
                                v.override_key
                            ),
                            (
                                tool,
                                "arima_fit_work",
                                ((n_max as u64 + 1) * per_obs).to_string(),
                                cap.to_string(),
                                "arima_fit_work"
                            ),
                            "{p} {d} {q}"
                        );
                        for rows in [n_max + 1, usize::MAX] {
                            assert!(Limits::off().check_arima_fit(tool, rows, p, d, q).is_ok());
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 2 * 21 * 21 * 21);
        // The fits the tool description suggests run on long series: ARIMA(1,0,1) and
        // ARIMA(1,1,1) on 10,000 observations.
        assert!(h.check_arima_fit("ts_arima_fit", 10_000, 1, 0, 1).is_ok());
        assert!(h.check_arima_fit("ts_arima_fit", 10_000, 1, 1, 1).is_ok());
        assert!(
            h.check_arima_fit(
                "ts_arima_fit",
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX
            )
            .is_err()
        );
    }

    /// The refusal names the orders, the observations, the work and the limit, and says how
    /// far the observations or the orders must shrink, never to fewer observations than the fit
    /// needs.
    #[test]
    fn arima_fit_refusal_says_what_to_change() {
        let l = Limits::off()
            .with_overrides(&["arima_fit_work=10000".to_string()])
            .unwrap();
        // 1,000 x 5^2 = 25,000; 10,000 / 25 = 400 observations; 10,000 / 1,000 = 10 >= 3^2, so
        // p = q = 0 still fits 1,000 observations.
        let v = l
            .check_arima_fit("ts_arima_fit", 1_000, 1, 0, 1)
            .unwrap_err();
        assert_eq!(
            v.message,
            "`ts_arima_fit` with p = 1, d = 0 and q = 1 on 1000 observations needs 25000 \
             (arima_fit_work: observations x ((p + q + 3)^2 + d)), which exceeds this server's \
             limit of 10000: the ARIMA fit cannot be stopped once it starts. Use at most 400 \
             observations at these orders, for example the most recent ones, or set p and q to \
             0."
        );
        // 1,000 x (5^2 + 1) = 26,000; 10,000 / 26 = 384; 10 - 1 = 9 = 3^2, so p = q = 0 fits.
        let v = l
            .check_arima_fit("ts_arima_fit", 1_000, 1, 1, 1)
            .unwrap_err();
        assert!(
            v.message.starts_with(
                "`ts_arima_fit` with p = 1, d = 1 and q = 1 on 1000 observations needs 26000 "
            ) && v.message.ends_with(
                "Use at most 384 observations at these orders, for example the most recent ones, \
                 or set p and q to 0."
            ),
            "{}",
            v.message
        );
        // 100 x 43^2; 10,000 / 1,849 = 5 observations, fewer than the 50 the fit needs;
        // 10,000 / 100 = 100 = 10^2, so p + q <= 7.
        let v = l
            .check_arima_fit("ts_arima_fit", 100, 20, 0, 20)
            .unwrap_err();
        assert!(
            v.message.ends_with(
                "the ARIMA fit cannot be stopped once it starts. Lower the orders so that \
                 p + q is at most 7."
            ),
            "{}",
            v.message
        );
        // 10,000 / 10,000 = 1 < 3^2: no order fits 10,000 observations.
        let v = l
            .check_arima_fit("ts_arima_fit", 10_000, 1, 0, 1)
            .unwrap_err();
        assert!(
            v.message.ends_with(
                "Use at most 400 observations at these orders, for example the most recent ones."
            ),
            "{}",
            v.message
        );
        // 10,000 / 1,000 = 10 < 9 + 11: d alone leaves no room for any p and q.
        let v = l
            .check_arima_fit("ts_arima_fit", 1_000, 0, 11, 0)
            .unwrap_err();
        assert!(
            v.message.ends_with(
                "Use at most 500 observations at these orders, for example the most recent ones."
            ),
            "{}",
            v.message
        );
        // (100 + 100 + 3)^2 = 41,209 > 10,000: no series fits these orders.
        let v = l
            .check_arima_fit("ts_arima_fit", 10, 100, 0, 100)
            .unwrap_err();
        assert!(
            v.message
                .ends_with("Lower the orders so that p + q is at most 28."),
            "{}",
            v.message
        );
        // Neither alone: 5 observations are too few for these orders, and 1 per observation is
        // below 3^2.
        let v = l
            .check_arima_fit("ts_arima_fit", 10_000, 20, 0, 20)
            .unwrap_err();
        assert!(
            v.message
                .ends_with("Use fewer observations and lower orders."),
            "{}",
            v.message
        );
        assert!(l.check_arima_fit("ts_arima_fit", 400, 1, 0, 1).is_ok());
        assert!(
            l.with_overrides(&["arima_fit_work=none".to_string()])
                .unwrap()
                .check_arima_fit("ts_arima_fit", 1_000_000, 20, 20, 20)
                .is_ok()
        );
    }

    /// While the ARIMA work budget is set, a column holding a nonzero value below
    /// `ARIMA_MIN_MAGNITUDE` is refused, because the differences, products and squares the fit
    /// forms from such values can be subnormal, and arithmetic on those is far slower than the
    /// budget assumes.
    /// Zero, NaN and infinities cost no more than other numbers and pass, as does everything when
    /// the budget is off.
    #[test]
    fn arima_series_below_the_magnitude_floor_is_refused_while_the_budget_is_set() {
        use super::ARIMA_MIN_MAGNITUDE;
        let h = Limits::hosted();
        let noisy = |scale: f64| -> Vec<f64> {
            (0..200)
                .map(|t| scale * (1.0 + ((t * 37 + 11) % 23) as f64 / 23.0))
                .collect()
        };
        for tool in ["ts_arima_fit", "ts_arima_forecast"] {
            // Subnormal values, and ordinary values whose squares are subnormal.
            for (series, found) in [
                (vec![1.0, 2.5, f64::from_bits(1), 3.0], "5e-324".to_string()),
                (noisy(1e-160), format!("{:e}", 1e-160 * (1.0 + 11.0 / 23.0))),
                ([noisy(1.0), vec![-1e-101]].concat(), "1e-101".to_string()),
            ] {
                let v = h.check_arima_series(tool, "y", series).unwrap_err();
                assert_eq!(
                    (
                        v.tool.as_str(),
                        v.parameter.as_str(),
                        v.requested.as_str(),
                        v.limit.as_str(),
                        v.override_key
                    ),
                    (tool, "column", found.as_str(), "1e-100", "arima_fit_work")
                );
                assert_eq!(
                    v.message,
                    format!(
                        "`{tool}` cannot fit column `y`: it holds a nonzero value of magnitude \
                         {found}, below this server's floor of 1e-100 for an ARIMA fit \
                         (arima_fit_work). Values that small can make the fit's arithmetic many \
                         times slower, and the fit cannot be stopped once it starts. Set values \
                         that small to 0, or rescale the column, for example multiply it by a \
                         power of ten."
                    )
                );
            }
            for series in [
                noisy(1.0),
                noisy(ARIMA_MIN_MAGNITUDE),
                noisy(1e300),
                vec![0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0],
                Vec::new(),
            ] {
                assert!(h.check_arima_series(tool, "y", series).is_ok());
            }
            assert!(
                Limits::off()
                    .check_arima_series(tool, "y", noisy(1e-310))
                    .is_ok()
            );
            assert!(
                h.clone()
                    .with_overrides(&["arima_fit_work=none".to_string()])
                    .unwrap()
                    .check_arima_series(tool, "y", noisy(1e-310))
                    .is_ok()
            );
        }
        // A long column name is cut as other echoed caller strings are.
        let long = "c".repeat(100);
        let v = h
            .check_arima_series("ts_arima_fit", &long, [1e-200])
            .unwrap_err();
        assert!(v.message.contains(&format!("column `{}`:", "c".repeat(64))));
    }

    /// The floor on the column bounds the differenced series the fit reads: every nonzero
    /// value of every difference of the column is at least about 2^-53 of the column's smallest
    /// nonzero value, so a column that passes the floor never differences into subnormal values.
    #[test]
    fn arima_differences_of_a_column_above_the_floor_stay_above_it_times_2_pow_minus_53() {
        use super::ARIMA_MIN_MAGNITUDE;
        // `arima::util::diff` (arima 0.3.0) for f64, with the same subtractions in the same
        // order: the differencing the fit applies.
        fn crate_diff(x: &[f64], d: usize) -> Vec<f64> {
            let mut y: Vec<f64> = x.to_vec();
            let len = y.len();
            for s in 0..d {
                for i in 1..len - s {
                    y[len - i] -= y[len - i - 1];
                }
            }
            y.drain(0..d);
            y
        }
        let bound = ARIMA_MIN_MAGNITUDE * 2f64.powi(-53);
        // Values one spacing apart near the floor and at other scales, the cancellations most
        // likely to leave a small difference.
        let m = ARIMA_MIN_MAGNITUDE;
        let up = |x: f64, k: u64| f64::from_bits(x.to_bits() + k);
        let series: Vec<Vec<f64>> = vec![
            (0..60).map(|t| up(m, (t * t) % 7)).collect(),
            (0..60)
                .map(|t| up(m * (1.0 + (t % 3) as f64), t % 5))
                .collect(),
            (0..60)
                .map(|t| if t % 2 == 0 { up(m, t) } else { -up(1.0, t) })
                .collect(),
            (0..60)
                .map(|t| up(1e10, (t * 7) % 11) * if t % 4 == 0 { -1.0 } else { 1.0 })
                .collect(),
        ];
        for x in &series {
            assert!(
                Limits::hosted()
                    .check_arima_series("ts_arima_fit", "y", x.clone())
                    .is_ok()
            );
            for d in 0..=20 {
                for v in crate_diff(x, d) {
                    assert!(v == 0.0 || v.abs() >= bound, "d = {d}: {v:e}");
                }
            }
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
        assert_eq!(ROW_RULES.len(), 17);
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
        /// A float field. A count or size given as a float (the `n` of `power_t_test`) takes a
        /// numeric `Max`, a convergence tolerance `Rule::Tolerance`.
        Float,
        /// An array: a field, an element of an array (`a[]`) or a tuple.
        Array,
        /// Integer elements of an array, recorded at the element path (`a[]` for the elements of
        /// the array `a`), so that a rule for them is never mistaken for the array's length rule.
        IntegerElems,
        /// Float elements of an array, recorded at the element path like
        /// [`Kind::IntegerElems`] and classified the same way.
        FloatElems,
        /// A map whose values are integers, floats, arrays, objects or any JSON. No rule path
        /// can address map values, so only `Exempt` classifies it.
        MapValues,
        /// A field or element that can hold any JSON: a variant of its schema says nothing about
        /// its value (a `serde_json::Value`, also as one arm of an untagged enum), or its schema
        /// refers to a definition the walk cannot find. Only `Exempt` classifies it; its cost
        /// sub-fields need rules of their own by path, because the schema does not list them.
        Untyped,
        /// A path nested deeper than [`MAX_DEPTH`], or whose schema leads through more than
        /// [`MAX_REF_HOPS`] references and variants, where the walk stopped. No rule classifies
        /// it, so the schema test fails instead of skipping what lies below.
        TooDeep,
    }

    /// True when `rule`, on the path where the walk found `kind`, checks that value (or exempts
    /// it). `Max(StringBytes)` checks only strings. `ArrayElemMax` sits on the array's path, not
    /// on the element path of [`Kind::IntegerElems`] or [`Kind::FloatElems`]; [`classified`]
    /// handles it.
    fn classifies(kind: Kind, rule: Rule) -> bool {
        match (kind, rule) {
            (Kind::TooDeep, _) => false,
            (_, Rule::Exempt(_)) => true,
            (Kind::Integer | Kind::IntegerElems | Kind::Float | Kind::FloatElems, Rule::Max(f)) => {
                f != Family::StringBytes
            }
            (Kind::Float, Rule::Tolerance) => true,
            (Kind::Array, Rule::ArrayLen(_) | Rule::UniqueList(_) | Rule::SquareLen) => true,
            _ => false,
        }
    }

    /// True when one of a tool's `rules` classifies `kind` at `path`: a rule on `path` itself
    /// ([`classifies`]) or, for the integer or float elements at `a[]`, `ArrayElemMax` on the
    /// array `a`. Nothing classifies the empty path, where [`cost_properties`] records what the
    /// root of a schema needs.
    fn classified(rules: &[&ParamRule], path: &str, kind: Kind) -> bool {
        if path.is_empty() {
            return false;
        }
        let on_path = rules
            .iter()
            .any(|r| r.path == path && classifies(kind, r.rule));
        let by_array = matches!(kind, Kind::IntegerElems | Kind::FloatElems)
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
        assert!(!classifies(Kind::Integer, Rule::Tolerance));
        assert!(classifies(Kind::Float, Rule::Tolerance));
        assert!(classifies(Kind::Float, Rule::Max(Lags)));
        assert!(!classifies(Kind::Float, Rule::Max(StringBytes)));
        assert!(!classifies(Kind::Float, Rule::ArrayElemMax(Lags)));
        assert!(classifies(Kind::Array, Rule::ArrayLen(ListLength)));
        assert!(classifies(Kind::Array, Rule::UniqueList(ListLength)));
        assert!(classifies(Kind::Array, Rule::SquareLen));
        assert!(!classifies(Kind::Array, Rule::ArrayElemMax(Lags)));
        for elems in [Kind::IntegerElems, Kind::FloatElems] {
            assert!(classifies(elems, Rule::Max(Lags)));
            assert!(!classifies(elems, Rule::Max(StringBytes)));
            assert!(!classifies(elems, Rule::ArrayLen(ListLength)));
            assert!(!classifies(elems, Rule::Tolerance));
        }
        assert!(!classifies(Kind::MapValues, Rule::ArrayLen(ListLength)));
        assert!(!classifies(Kind::Untyped, Rule::Max(Lags)));
        for kind in [
            Kind::Integer,
            Kind::Float,
            Kind::Array,
            Kind::IntegerElems,
            Kind::FloatElems,
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
            for elems in [Kind::IntegerElems, Kind::FloatElems] {
                assert!(classified(&rules, "a[]", elems), "{table:?}");
            }
            assert!(!classified(&rules, "a", Kind::Array), "{table:?}");
        }
        let length = [at("a", Rule::ArrayLen(ListLength))];
        let rules: Vec<&ParamRule> = length.iter().collect();
        assert!(classified(&rules, "a", Kind::Array));
        assert!(!classified(&rules, "a[]", Kind::IntegerElems));
        assert!(!classified(&rules, "a[]", Kind::FloatElems));
        // `ArrayElemMax` on `a[]` covers the numbers of the inner arrays, at `a[][]`
        let inner = [at("a[]", Rule::ArrayElemMax(Lags))];
        let rules: Vec<&ParamRule> = inner.iter().collect();
        assert!(classified(&rules, "a[][]", Kind::IntegerElems));
        assert!(classified(&rules, "a[][]", Kind::FloatElems));
        assert!(!classified(&rules, "a[]", Kind::IntegerElems));
        // A float field needs a rule of its own: a length rule on its name does not classify it.
        assert!(!classified(&rules, "a", Kind::Float));
        let scalar = [at("x", Rule::Max(Lags)), at("tol", Rule::Tolerance)];
        let rules: Vec<&ParamRule> = scalar.iter().collect();
        assert!(classified(&rules, "x", Kind::Float));
        assert!(classified(&rules, "tol", Kind::Float));
        assert!(!classified(&rules, "y", Kind::Float));

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
        // A numeric Max checks a float by absolute value, fractions and negatives included.
        assert!(h.check_with("t", &json!({"a": [0.5, -1_000.0]}), t).is_ok());
        assert!(h.check_with("t", &json!({"a": [1_000.5]}), t).is_err());
        assert!(h.check_with("t", &json!({"a": [-1e300]}), t).is_err());
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
                // a map of floats (`HashMap<String, f64>`) needs a decision like one of integers
                "fmap": {"type": "object", "additionalProperties": {"type": "number"}},
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
                "level": {"enum": ["low", "high"]},
                // floats: a field whatever its name, an `Option<f64>`, elements and a nested one
                "n": {"type": "number", "format": "double"},
                "abs_tol": {"type": ["number", "null"]},
                "weights": {"type": "array", "items": {"type": "number"}},
                "shape": {"type": "object", "properties": {"scale": {"type": "number"}}},
                // an untagged enum with a `Value` arm: the typed arm does not hide the other
                "mixed": {"anyOf": [{"type": "integer"}, true]},
                // an `Option<Struct>` whose definition is missing: the null arm does not hide it
                "dangling": {"anyOf": [{"$ref": "#/$defs/Missing"}, {"type": "null"}]},
                // the same `Option` with a definition, as rmcp's AddNullable writes the null arm
                "maybe": {"anyOf": [{"$ref": "#/$defs/Lens"}, {"const": null, "nullable": true}]},
                // a reference cycle: the walk stops after MAX_REF_HOPS and says so
                "cycle": {"$ref": "#/$defs/Cycle"},
                // properties, elements and map values are walked whatever the `type`
                "untyped_object": {"properties": {"k": {"type": "integer"}}},
                "untyped_list": {"items": {"type": "integer"}},
                "untyped_map": {"additionalProperties": {"items": {"type": "string"}}},
                // a flattened map next to a subschema
                "flat": {"type": "object", "allOf": [{"$ref": "#/$defs/Closed"}],
                         "unevaluatedProperties": {"type": "integer"}},
                // a variant below a typed schema is constrained by it
                "narrowed": {"type": "object", "anyOf": [{"required": ["k"]}, true],
                             "properties": {"k": {"type": "integer"}}},
                "never": false
            },
            "$defs": {
                "Lens": {"type": "array", "items": {"type": "integer"}},
                "Cycle": {"$ref": "#/$defs/Cycle"},
                "Closed": {"type": "object", "properties": {}}
            }
        });
        let mut found = cost_properties(&schema);
        found.sort_by(|a, b| a.0.cmp(&b.0));
        let found: Vec<(&str, Kind)> = found.iter().map(|(p, k)| (p.as_str(), *k)).collect();
        assert_eq!(
            found,
            [
                ("abs_tol", Kind::Float),
                ("anymap", Kind::MapValues),
                ("anys", Kind::Array),
                ("anys[]", Kind::Untyped),
                ("closed.k", Kind::Integer),
                ("counts", Kind::Array),
                ("counts[]", Kind::IntegerElems),
                ("cycle", Kind::TooDeep),
                ("dangling", Kind::Untyped),
                ("flat", Kind::MapValues),
                ("fmap", Kind::MapValues),
                ("grid", Kind::Array),
                ("grid[]", Kind::Array),
                ("grid[][]", Kind::IntegerElems),
                ("legacy", Kind::Array),
                ("legacy[]", Kind::Array),
                ("legacy[][]", Kind::IntegerElems),
                ("maybe", Kind::Array),
                ("maybe[]", Kind::IntegerElems),
                ("mixed", Kind::Untyped),
                ("mixed", Kind::Integer),
                ("n", Kind::Float),
                ("narrowed.k", Kind::Integer),
                ("options", Kind::Untyped),
                ("pair", Kind::Array),
                ("pair[]", Kind::FloatElems),
                ("pairs", Kind::Array),
                ("pairs[]", Kind::Array),
                ("reps", Kind::MapValues),
                ("shape.scale", Kind::Float),
                ("spec", Kind::Untyped),
                ("tables", Kind::Array),
                ("tables[]", Kind::MapValues),
                ("untyped_list", Kind::Array),
                ("untyped_list[]", Kind::IntegerElems),
                ("untyped_map", Kind::MapValues),
                ("untyped_object.k", Kind::Integer),
                ("weights", Kind::Array),
                ("weights[]", Kind::FloatElems),
                ("window", Kind::Array),
                ("window[]", Kind::IntegerElems),
            ]
        );
    }

    #[test]
    fn walker_reports_a_map_or_any_json_at_the_root_and_no_rule_classifies_it() {
        let fields = json!({"n": {"type": "integer"}});
        let at_root = |extra: Value| {
            let mut schema = json!({"type": "object", "properties": fields});
            if let (Some(o), Value::Object(extra)) = (schema.as_object_mut(), extra) {
                o.extend(extra);
            }
            let mut found = cost_properties(&schema);
            found.retain(|(p, _)| p.is_empty());
            found.into_iter().map(|(_, k)| k).collect::<Vec<_>>()
        };
        // a flattened map at the root, whatever its values
        assert_eq!(
            at_root(json!({"additionalProperties": {"type": "integer"}})),
            [Kind::MapValues]
        );
        assert_eq!(
            at_root(json!({"additionalProperties": {"type": "string"}})),
            [Kind::MapValues]
        );
        assert_eq!(
            at_root(json!({"additionalProperties": true})),
            [Kind::MapValues]
        );
        assert_eq!(
            at_root(json!({"unevaluatedProperties": true})),
            [Kind::MapValues]
        );
        // `deny_unknown_fields`, and the usual struct, add nothing
        assert!(at_root(json!({"additionalProperties": false})).is_empty());
        assert!(at_root(json!({})).is_empty());
        // a root that is any JSON, or that the walk cannot see
        assert_eq!(
            cost_properties(&json!(true)),
            [(String::new(), Kind::Untyped)]
        );
        assert_eq!(
            cost_properties(&json!({"$ref": "#/components/Args"})),
            [(String::new(), Kind::Untyped)]
        );
        let cycle = json!({"$ref": "#/$defs/A", "$defs": {"A": {"$ref": "#/$defs/A"}}});
        assert_eq!(cost_properties(&cycle), [(String::new(), Kind::TooDeep)]);
        // the fields beside a root map are still walked
        let schema = json!({"type": "object", "properties": fields, "additionalProperties": true});
        assert!(cost_properties(&schema).contains(&("n".to_string(), Kind::Integer)));
        // not even an Exempt classifies the root
        let exempt = ParamRule {
            tool: "t",
            path: "",
            rule: Rule::Exempt("reviewed"),
        };
        for kind in [Kind::MapValues, Kind::Untyped] {
            assert!(!classified(&[&exempt], "", kind));
        }
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

    /// A schema and every schema it stands for, as [`variants`] collects them.
    #[derive(Default)]
    struct Variants<'a> {
        /// The schema, its `$ref` targets and its `anyOf` / `oneOf` / `allOf` variants.
        all: Vec<&'a Value>,
        /// True when a leaf (a variant with no `anyOf`, `oneOf` or `allOf` and no `$ref` that
        /// resolves) says nothing about its value, and nothing above it does ([`says_what`]), or
        /// when a `$ref` does not resolve: any JSON can reach the value that way, whatever the
        /// other variants say.
        untyped: bool,
        /// True when the walk stopped after [`MAX_REF_HOPS`], so variants may be missing.
        too_deep: bool,
    }

    /// True when a schema by itself says what its value is: it has a `type`, `enum` or `const`,
    /// lists properties, elements or map values (which [`record`] then walks), or it is `false`
    /// (no value at all). `true`, `{}` and a schema with only a description accept any JSON.
    fn says_what(schema: &Value) -> bool {
        const KEYS: [&str; 8] = [
            "type",
            "enum",
            "const",
            "properties",
            "items",
            "prefixItems",
            "additionalProperties",
            "unevaluatedProperties",
        ];
        match schema {
            Value::Bool(any) => !any,
            Value::Object(o) => KEYS.iter().any(|k| o.contains_key(*k)),
            _ => false,
        }
    }

    /// The target of a `$ref` (`#/definitions/...` or `#/$defs/...`, resolved against `root`).
    fn resolve<'a>(root: &'a Value, reference: &Value) -> Option<&'a Value> {
        let reference = reference.as_str()?;
        let (defs, name) = if let Some(name) = reference.strip_prefix("#/definitions/") {
            ("definitions", name)
        } else {
            ("$defs", reference.strip_prefix("#/$defs/")?)
        };
        root.get(defs)?.get(name)
    }

    /// The variants `schema` stands for, resolved against `root`.
    fn expand<'a>(root: &'a Value, schema: &'a Value) -> Variants<'a> {
        let mut out = Variants::default();
        variants(root, schema, 0, false, &mut out);
        out
    }

    /// Adds `schema` and the schemas it stands for to `out`. `typed_above` is true when a schema
    /// that leads here says what the value is, which then constrains this one as well.
    fn variants<'a>(
        root: &'a Value,
        schema: &'a Value,
        hops: usize,
        typed_above: bool,
        out: &mut Variants<'a>,
    ) {
        if hops > MAX_REF_HOPS {
            out.too_deep = true;
            return;
        }
        out.all.push(schema);
        let typed = typed_above || says_what(schema);
        let mut leaf = true;
        if let Some(reference) = schema.get("$ref") {
            match resolve(root, reference) {
                Some(target) => {
                    leaf = false;
                    variants(root, target, hops + 1, typed, out);
                }
                // what lies behind it is unknown, so it may be anything
                None => out.untyped = true,
            }
        }
        for key in ["anyOf", "oneOf", "allOf"] {
            if let Some(list) = schema.get(key).and_then(Value::as_array) {
                leaf = false;
                for variant in list {
                    variants(root, variant, hops + 1, typed, out);
                }
            }
        }
        if leaf && !typed {
            out.untyped = true;
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

    /// True when a variant describes an array: by its `type`, or by `items` or `prefixItems`
    /// given without one.
    fn is_array(vs: &Variants) -> bool {
        types(&vs.all).contains(&"array")
            || vs
                .all
                .iter()
                .any(|v| v.get("items").is_some() || v.get("prefixItems").is_some())
    }

    /// The schemas a map's values must match: `additionalProperties`, or the
    /// `unevaluatedProperties` that schemars writes for a flattened map next to subschemas,
    /// unless it is `false` (no keys beyond the listed properties).
    fn map_value_schemas<'a>(variants_of: &[&'a Value]) -> Vec<&'a Value> {
        variants_of
            .iter()
            .flat_map(|v| ["additionalProperties", "unevaluatedProperties"].map(|k| v.get(k)))
            .flatten()
            .filter(|values| !matches!(values, Value::Bool(false)))
            .collect()
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

    /// Records what the value at `path`, described by `schema`, needs: its integers and floats,
    /// an array (and what its elements need, at `{path}[]`), the properties of an object (under
    /// `{path}.`) and map values, [`Kind::Untyped`], or [`Kind::TooDeep`] when the walk cannot
    /// see all of it. Properties, elements and map values are walked whenever a variant lists
    /// them, whatever its `type`. `name` is the field name of a property and `None` for an
    /// array element, whose numbers are [`Kind::IntegerElems`] and [`Kind::FloatElems`].
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
        let vs = expand(root, schema);
        if vs.too_deep {
            push(out, path, Kind::TooDeep);
        }
        if vs.untyped {
            push(out, path, Kind::Untyped);
        }
        let ts = types(&vs.all);
        let numbers = [
            ("integer", Kind::Integer, Kind::IntegerElems),
            ("number", Kind::Float, Kind::FloatElems),
        ];
        for (json_type, field, elements) in numbers {
            if ts.contains(&json_type) {
                push(out, path, if name.is_some() { field } else { elements });
            }
        }
        if is_array(&vs) {
            push(out, path, Kind::Array);
            let elements = format!("{path}[]");
            for item in element_schemas(&vs.all) {
                record(root, item, &elements, None, depth + 1, out);
            }
        }
        walk_properties(root, &vs.all, &format!("{path}."), depth + 1, out);
        map_values(root, &vs.all, path, out);
    }

    /// Records [`Kind::MapValues`] at `path` when a schema among `variants_of` is a map (its
    /// [`map_value_schemas`]) whose values are integers, floats, arrays, objects or any JSON, and
    /// [`Kind::TooDeep`] when the walk cannot see what they are.
    fn map_values(root: &Value, variants_of: &[&Value], path: &str, out: &mut Vec<(String, Kind)>) {
        for values in map_value_schemas(variants_of) {
            let vvs = expand(root, values);
            if vvs.too_deep {
                push(out, path, Kind::TooDeep);
            }
            let vts = types(&vvs.all);
            let object = vts.contains(&"object")
                || vvs
                    .all
                    .iter()
                    .any(|v| v.get("properties").is_some() || !map_value_schemas(&[*v]).is_empty());
            if vvs.untyped
                || vts.contains(&"integer")
                || vts.contains(&"number")
                || is_array(&vvs)
                || object
            {
                push(out, path, Kind::MapValues);
            }
        }
    }

    /// Walks one tool's input schema and returns every (path, kind) that must be classified.
    ///
    /// What the root itself needs is recorded at the empty path, which no rule can classify (a
    /// rule path is never empty, [`classified`]): a root that may be any JSON or that the walk
    /// cannot see, and a map at the root (a `#[serde(flatten)]` map) whatever its values, since
    /// its keys are request fields the schema does not name.
    fn cost_properties(schema: &Value) -> Vec<(String, Kind)> {
        let root = expand(schema, schema);
        let mut out = Vec::new();
        if root.too_deep {
            push(&mut out, "", Kind::TooDeep);
        }
        if root.untyped {
            push(&mut out, "", Kind::Untyped);
        }
        if !map_value_schemas(&root.all).is_empty() {
            push(&mut out, "", Kind::MapValues);
        }
        walk_properties(schema, &root.all, "", 0, &mut out);
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
        let (mut ints, mut arrays, mut floats, mut elems, mut float_elems) = (0, 0, 0, 0, 0);
        let (mut maps, mut untyped) = (0, 0);
        let mut missing = Vec::new();
        for (tool, schema) in &tools {
            let rules = rules_for(tool);
            for (path, kind) in cost_properties(schema) {
                match kind {
                    Kind::Integer => ints += 1,
                    Kind::Array => arrays += 1,
                    Kind::Float => floats += 1,
                    Kind::IntegerElems => elems += 1,
                    Kind::FloatElems => float_elems += 1,
                    Kind::MapValues => maps += 1,
                    Kind::Untyped => untyped += 1,
                    Kind::TooDeep => {}
                }
                if !classified(&rules, &path, kind) {
                    let at = if path.is_empty() { "<root>" } else { &path };
                    missing.push(format!("{tool}.{at} ({kind:?})"));
                }
            }
        }
        assert!(
            missing.is_empty(),
            "request fields without a limits rule (add a ParamRule or a reasoned Exempt in \
             tools/limits/rules.rs; the limits module documentation says which rule fits which \
             kind; TooDeep means the walk stopped at MAX_DEPTH or MAX_REF_HOPS, and <root> \
             that the request itself is a map or any JSON, which no rule can classify):\n{}",
            missing.join("\n")
        );
        eprintln!(
            "schema walker: {} tools, {ints} integer, {arrays} array, {floats} float, {elems} \
             integer-element, {float_elems} float-element, {maps} map and {untyped} untyped \
             fields",
            tools.len()
        );
        // Sanity: a schema format change must not silently hide properties (the walker saw
        // 191 integer, 207 array, 7 integer-element and 1 untyped field when it learned element
        // paths and untyped fields, and 118 float and 31 float-element fields when it learned
        // floats). Untyped fields have no floor: giving the last `serde_json::Value` request
        // field a type would be an improvement, and
        // `walker_reports_nested_arrays_tuple_elements_and_maps` shows they are found.
        assert!(
            ints >= 150 && arrays >= 150 && floats >= 100 && elems >= 5 && float_elems >= 25,
            "walker saw {ints} integer, {arrays} array, {floats} float, {elems} integer-element \
             and {float_elems} float-element fields"
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

    /// The schemas at rule path `path` of `root` (a tool's input schema): per segment, the
    /// field among the properties of the variants so far, then the element schemas once for
    /// each `[]`.
    fn schemas_at<'a>(root: &'a Value, path: &str) -> Vec<&'a Value> {
        let mut at = vec![root];
        for segment in path.split('.') {
            let name = segment.trim_end_matches("[]");
            at = at
                .iter()
                .flat_map(|s| expand(root, s).all)
                .filter_map(|v| v.get("properties")?.get(name))
                .collect();
            for _ in 0..(segment.len() - name.len()) / 2 {
                at = at
                    .iter()
                    .flat_map(|s| element_schemas(&expand(root, s).all))
                    .collect();
            }
        }
        at
    }

    /// The struct prefixes of `rules` (the part of a non-`Exempt` rule path before a `.`) that
    /// `schema` does not resolve, or whose schema admits an array, as `path (rule)` lines, and
    /// how many prefixes were looked at. The hosted check refuses a JSON array at such a
    /// prefix, and the audit log redacts one, so a valid array there would be refused.
    fn array_struct_prefixes(schema: &Value, rules: &[&ParamRule]) -> (usize, Vec<String>) {
        let (mut checked, mut wrong) = (0, Vec::new());
        for rule in rules {
            if matches!(rule.rule, Rule::Exempt(_)) {
                continue;
            }
            for (dot, _) in rule.path.match_indices('.') {
                let prefix = &rule.path[..dot];
                checked += 1;
                let at = schemas_at(schema, prefix);
                if at.is_empty() {
                    wrong.push(format!("{prefix} (rule {}): no such field", rule.path));
                } else if at.iter().any(|s| is_array(&expand(schema, s))) {
                    wrong.push(format!("{prefix} (rule {}): admits an array", rule.path));
                }
            }
        }
        (checked, wrong)
    }

    #[test]
    fn no_struct_a_rule_goes_into_admits_an_array() {
        let (mut checked, mut wrong) = (0, Vec::new());
        for (tool, schema) in tools() {
            let (n, w) = array_struct_prefixes(&schema, &rules_for(&tool));
            checked += n;
            wrong.extend(w.into_iter().map(|w| format!("{tool}.{w}")));
        }
        assert!(
            wrong.is_empty(),
            "a rule path goes on into a field that is not a struct: the hosted check refuses \
             (and the audit log redacts) a JSON array there, so a valid array would be refused; \
             rule the field itself, or exempt the path below it:\n{}",
            wrong.join("\n")
        );
        // generate_random_data and generate_report: 18 prefixes of 10 rules.
        assert!(checked >= 18, "{checked} prefixes");
        // The check finds a struct that may also be an array (an untagged Vec-or-struct enum)
        // and a prefix that is not a field; untyped any-JSON passes.
        let schema = json!({"type": "object", "properties": {
            "items": {"type": "array", "items": {"anyOf": [
                {"type": "object", "properties": {"name": {"type": "string"}}},
                {"type": "array", "items": {"type": "string"}}
            ]}},
            "spec": {"description": "any JSON"}
        }});
        let rule = |path| ParamRule {
            tool: "t",
            path,
            rule: Rule::Max(Family::StringBytes),
        };
        let (items, typo, spec) = (rule("items[].name"), rule("itmes[].name"), rule("spec.v"));
        let (n, wrong) = array_struct_prefixes(&schema, &[&items, &typo, &spec]);
        assert_eq!(n, 3);
        assert_eq!(
            wrong,
            [
                "items[] (rule items[].name): admits an array",
                "itmes[] (rule itmes[].name): no such field"
            ]
        );
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
                    // at least the smallest tolerance, and below every cap
                    Kind::Float => (path.as_str(), json!(1e-6)),
                    Kind::Array => (path.as_str(), json!([])),
                    Kind::IntegerElems => match path.strip_suffix("[]") {
                        Some(array) => (array, json!([1])),
                        None => continue,
                    },
                    Kind::FloatElems => match path.strip_suffix("[]") {
                        Some(array) => (array, json!([0.5])),
                        None => continue,
                    },
                    Kind::MapValues => (path.as_str(), json!({})),
                    Kind::Untyped | Kind::TooDeep => continue,
                };
                // nested paths, and the root's own entries at the empty path
                if field.is_empty() || field.contains(['.', '[']) {
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

    /// `Limits::check_pivot` passes a request whose fields it cannot read, so a renamed or
    /// retyped field would silently switch the pivot budgets off: each rule's fields must be
    /// required fields of its tool, a list of names for the index and a name for the others.
    #[test]
    fn pivot_rules_name_required_fields_of_their_tool() {
        let tools: std::collections::HashMap<String, Value> = tools().into_iter().collect();
        assert!(!super::PIVOT_RULES.is_empty());
        for rule in super::PIVOT_RULES {
            let schema = tools
                .get(rule.tool)
                .unwrap_or_else(|| panic!("pivot rule for unknown tool `{}`", rule.tool));
            let required = schema["required"].as_array().unwrap();
            let fields = [
                (rule.dataset_field, json!("string")),
                (rule.index_field, json!("array")),
                (rule.on_field, json!("string")),
                (rule.values_field, json!("string")),
            ];
            for (field, kind) in fields {
                assert!(
                    required.iter().any(|v| v == field),
                    "{}.{field} is not required",
                    rule.tool
                );
                assert_eq!(
                    schema["properties"][field]["type"], kind,
                    "{}.{field}",
                    rule.tool
                );
            }
            assert_eq!(
                schema["properties"][rule.index_field]["items"]["type"], "string",
                "{}.{}",
                rule.tool, rule.index_field
            );
        }
    }
}
