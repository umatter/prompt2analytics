//! Request types for hypothesis testing tools.
//!
//! This module contains request structs for:
//! - T-tests (one-sample, two-sample, paired)
//! - Wilcoxon rank tests
//! - Chi-squared tests (goodness-of-fit, independence)
//! - Fisher's exact test
//! - Kolmogorov-Smirnov tests
//! - Kruskal-Wallis test
//! - Friedman test
//! - Shapiro-Wilk normality test
//! - Various other hypothesis tests

use schemars::JsonSchema;
use serde::Deserialize;

/// Request for Kolmogorov-Smirnov test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct KsTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// First variable column name
    #[schemars(description = "Name of the first variable column (must be numeric).")]
    pub x: String,

    /// Second variable column name (optional)
    #[schemars(
        description = "Name of the second variable column for two-sample test. Omit for one-sample test against a theoretical distribution."
    )]
    pub y: Option<String>,

    /// Theoretical distribution for one-sample test
    #[schemars(
        description = "For one-sample test: distribution to test against. Options: 'normal' (default), 'uniform', 'exponential'. Ignored for two-sample test."
    )]
    pub distribution: Option<String>,

    /// Mean parameter for normal distribution
    #[schemars(
        description = "Mean parameter for normal distribution (default: 0). Only used when distribution='normal'."
    )]
    pub mean: Option<f64>,

    /// Standard deviation parameter for normal distribution
    #[schemars(
        description = "Standard deviation parameter for normal distribution (default: 1). Only used when distribution='normal'."
    )]
    pub sd: Option<f64>,

    /// Lower bound for uniform distribution
    #[schemars(
        description = "Lower bound for uniform distribution (default: 0). Only used when distribution='uniform'."
    )]
    pub a: Option<f64>,

    /// Upper bound for uniform distribution
    #[schemars(
        description = "Upper bound for uniform distribution (default: 1). Only used when distribution='uniform'."
    )]
    pub b: Option<f64>,

    /// Rate parameter for exponential distribution
    #[schemars(
        description = "Rate parameter for exponential distribution (default: 1). Only used when distribution='exponential'."
    )]
    pub rate: Option<f64>,

    /// Alternative hypothesis direction
    #[schemars(
        description = "Alternative hypothesis: 'two.sided' (default), 'greater', or 'less'. For two-sample: 'greater' means CDF of x is not below CDF of y (x is stochastically greater)."
    )]
    pub alternative: Option<String>,
}

/// Request for Fisher's exact test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FisherExactRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Row variable column name
    #[schemars(
        description = "Name of the column for rows of the 2×2 table. Must have exactly 2 unique values."
    )]
    pub row_var: String,

    /// Column variable column name
    #[schemars(
        description = "Name of the column for columns of the 2×2 table. Must have exactly 2 unique values."
    )]
    pub col_var: String,

    /// Alternative hypothesis
    #[schemars(
        description = "Alternative hypothesis: 'two.sided' (default), 'greater', or 'less'. One-sided tests compare odds ratio to 1."
    )]
    pub alternative: Option<String>,

    /// Confidence level
    #[schemars(
        description = "Confidence level for odds ratio confidence interval (e.g., 0.95). If omitted, CI is not computed."
    )]
    pub conf_level: Option<f64>,
}

/// Request for trend test in proportions.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PropTrendTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column with number of successes per group
    #[schemars(description = "Name of column containing number of successes in each group.")]
    pub successes: String,

    /// Column with number of trials per group
    #[schemars(description = "Name of column containing number of trials in each group.")]
    pub trials: String,

    /// Optional scores for the trend
    #[schemars(
        description = "Optional scores for each group. If omitted, uses 1, 2, 3, ... (equally spaced)."
    )]
    pub scores: Option<Vec<f64>>,
}

/// Request for Wilcoxon rank sum / signed rank test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct WilcoxonTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// First variable column name
    #[schemars(description = "Name of the first variable column (must be numeric).")]
    pub x: String,

    /// Second variable column name (optional)
    #[schemars(
        description = "Name of the second variable column for two-sample or paired tests. Omit for one-sample signed rank test."
    )]
    pub y: Option<String>,

    /// Hypothesized location shift
    #[schemars(
        description = "Null hypothesis value (default: 0). For one-sample: hypothesized median. For two-sample: hypothesized location shift."
    )]
    pub mu: Option<f64>,

    /// Alternative hypothesis direction
    #[schemars(
        description = "Alternative hypothesis: 'two.sided' (default), 'greater', or 'less'."
    )]
    pub alternative: Option<String>,

    /// Paired test flag
    #[schemars(
        description = "If true with two samples, perform paired signed rank test. If false (default), perform rank sum test."
    )]
    pub paired: Option<bool>,

    /// Use exact p-value calculation
    #[schemars(
        description = "If true, compute exact p-value (only for small samples without ties). If omitted, automatically decides based on sample size."
    )]
    pub exact: Option<bool>,

    /// Apply continuity correction
    #[schemars(
        description = "If true (default), apply continuity correction to normal approximation."
    )]
    pub correct: Option<bool>,

    /// Compute confidence interval
    #[schemars(
        description = "If true, compute confidence interval and location estimate. Default: false."
    )]
    pub conf_int: Option<bool>,

    /// Confidence level
    #[schemars(description = "Confidence level for the interval (default: 0.95).")]
    pub conf_level: Option<f64>,
}

/// Request for t-test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// First variable column name
    #[schemars(description = "Name of the first variable column (must be numeric).")]
    pub x: String,

    /// Second variable column name (optional)
    #[schemars(
        description = "Name of the second variable column for two-sample or paired tests. Omit for one-sample test."
    )]
    pub y: Option<String>,

    /// Hypothesized mean or difference
    #[schemars(
        description = "Null hypothesis value (default: 0). For one-sample: hypothesized mean. For two-sample: hypothesized difference."
    )]
    pub mu: Option<f64>,

    /// Alternative hypothesis direction
    #[schemars(
        description = "Alternative hypothesis: 'two.sided' (default), 'greater', or 'less'."
    )]
    pub alternative: Option<String>,

    /// Paired test flag
    #[schemars(
        description = "If true, perform a paired t-test. Requires both x and y columns. Default: false."
    )]
    pub paired: Option<bool>,

    /// Equal variances assumption
    #[schemars(
        description = "For two-sample tests: if true, assume equal variances (Student's t); if false (default), use Welch's t-test."
    )]
    pub var_equal: Option<bool>,

    /// Confidence level
    #[schemars(description = "Confidence level for the interval (default: 0.95).")]
    pub conf_level: Option<f64>,
}

/// Request for chi-squared goodness-of-fit test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ChiSquaredGofRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with categorical values to count
    #[schemars(
        description = "Name of the categorical column. Counts of each unique value will be tested."
    )]
    pub column: String,

    /// Expected probabilities (optional)
    #[schemars(
        description = "Expected probabilities for each category. Must sum to 1.0. If omitted, uniform distribution is assumed."
    )]
    pub probs: Option<Vec<f64>>,
}

/// Request for chi-squared test of independence.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ChiSquaredIndependenceRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Row variable column name
    #[schemars(description = "Name of the column for rows of the contingency table.")]
    pub row_var: String,

    /// Column variable column name
    #[schemars(description = "Name of the column for columns of the contingency table.")]
    pub col_var: String,

    /// Apply Yates' continuity correction
    #[schemars(
        description = "If true and table is 2×2, apply Yates' continuity correction. Default: true."
    )]
    pub correct: Option<bool>,
}

/// Request for exact Poisson test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PoissonTestRequest {
    /// Number of events (one or two values)
    #[schemars(
        description = "Number of events. For one-sample test: single value. For two-sample rate comparison: two values [x1, x2]."
    )]
    pub x: Vec<u64>,

    /// Time base (one or two values)
    #[schemars(
        description = "Time base(s) or exposure(s). Must match length of x. For one-sample: single value. For two-sample: two values [t1, t2]."
    )]
    pub t: Vec<f64>,

    /// Hypothesized rate or rate ratio
    #[schemars(
        description = "Hypothesized rate (one-sample) or rate ratio (two-sample). Default: 1.0."
    )]
    pub r: Option<f64>,

    /// Alternative hypothesis
    #[schemars(
        description = "Direction of alternative hypothesis: 'two.sided' (default), 'greater', or 'less'."
    )]
    pub alternative: Option<String>,

    /// Confidence level
    #[schemars(description = "Confidence level for the interval. Default: 0.95.")]
    pub conf_level: Option<f64>,
}

/// Request for McNemar's chi-squared test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct McnemarTestRequest {
    /// Upper-right cell (b) of the 2x2 table
    #[schemars(
        description = "The count in the upper-right cell (row 1, column 2) of the 2x2 table. Represents discordant pairs where first classifier is positive and second is negative."
    )]
    pub b: u64,

    /// Lower-left cell (c) of the 2x2 table
    #[schemars(
        description = "The count in the lower-left cell (row 2, column 1) of the 2x2 table. Represents discordant pairs where first classifier is negative and second is positive."
    )]
    pub c: u64,

    /// Apply continuity correction
    #[schemars(
        description = "If true (default), apply Yates' continuity correction. Set to false for the uncorrected version."
    )]
    pub correct: Option<bool>,
}

/// Request for Bartlett's test of homogeneity of variances.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BartlettTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response variable column name
    #[schemars(description = "Name of the response (numeric) variable column.")]
    pub response: String,

    /// Factor (grouping) variable column name
    #[schemars(description = "Name of the factor (grouping) variable column.")]
    pub factor: String,
}

/// Request for Cochran-Mantel-Haenszel test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MantelhaenTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Row variable column name (binary factor)
    #[schemars(
        description = "Name of the column containing the row variable (must have exactly 2 levels, e.g., 'exposed'/'unexposed')."
    )]
    pub row_var: String,

    /// Column variable column name (binary factor)
    #[schemars(
        description = "Name of the column containing the column variable (must have exactly 2 levels, e.g., 'disease'/'healthy')."
    )]
    pub col_var: String,

    /// Stratum variable column name
    #[schemars(
        description = "Name of the column containing stratum identifiers (e.g., 'hospital', 'study_site')."
    )]
    pub stratum_var: String,

    /// Continuity correction
    #[schemars(description = "Whether to apply Yates' continuity correction. Default: true.")]
    pub correct: Option<bool>,

    /// Alternative hypothesis
    #[schemars(
        description = "Direction of alternative hypothesis: 'two.sided' (default), 'greater', or 'less'."
    )]
    pub alternative: Option<String>,
}

/// Request for Kruskal-Wallis rank sum test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct KruskalWallisRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Value column name
    #[schemars(
        description = "Name of the numeric column containing the values to compare across groups."
    )]
    pub value: String,

    /// Group column name
    #[schemars(description = "Name of the column containing group labels/factors.")]
    pub group: String,
}

/// Request for pairwise t-tests with multiple comparison correction.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PairwiseTTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response variable column name
    #[schemars(description = "Name of the numeric response variable column.")]
    pub response: String,

    /// Grouping factor column name
    #[schemars(description = "Name of the grouping factor column (can be string or numeric).")]
    pub factor: String,

    /// P-value adjustment method
    #[schemars(
        description = "Method for adjusting p-values: 'holm' (default), 'bonferroni', 'hochberg', 'hommel', 'BH' (Benjamini-Hochberg FDR), 'BY', or 'none'."
    )]
    pub p_adjust_method: Option<String>,

    /// Use pooled standard deviation
    #[schemars(
        description = "If true, use pooled SD from all groups (Student's); if false (default), use Welch's t-test for each pair."
    )]
    pub pool_sd: Option<bool>,

    /// Alternative hypothesis direction
    #[schemars(
        description = "Alternative hypothesis: 'two.sided' (default), 'greater', or 'less'."
    )]
    pub alternative: Option<String>,
}

/// Request for Welch's one-way ANOVA test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct OnewayTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Value column name
    #[schemars(
        description = "Name of the numeric column containing the values to compare across groups."
    )]
    pub value: String,

    /// Group column name
    #[schemars(description = "Name of the column containing group labels/factors.")]
    pub group: String,

    /// Assume equal variances
    #[schemars(
        description = "If true, use standard ANOVA assuming equal variances. If false (default), use Welch's ANOVA."
    )]
    pub var_equal: Option<bool>,
}

/// Request for pairwise Wilcoxon rank sum tests with multiple comparison correction.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PairwiseWilcoxRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response variable column name
    #[schemars(description = "Name of the numeric response variable column.")]
    pub response: String,

    /// Grouping factor column name
    #[schemars(description = "Name of the grouping factor column (can be string or numeric).")]
    pub factor: String,

    /// P-value adjustment method
    #[schemars(
        description = "Method for adjusting p-values: 'holm' (default), 'bonferroni', 'hochberg', 'hommel', 'BH' (Benjamini-Hochberg FDR), 'BY', or 'none'."
    )]
    pub p_adjust_method: Option<String>,

    /// Alternative hypothesis direction
    #[schemars(
        description = "Alternative hypothesis: 'two.sided' (default), 'greater', or 'less'."
    )]
    pub alternative: Option<String>,

    /// Use exact p-value computation
    #[schemars(
        description = "If true, compute exact p-values (slow for large samples); if false, use normal approximation; if not specified, auto-decide based on sample size."
    )]
    pub exact: Option<bool>,
}

/// Request for Mood's two-sample test of scale.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MoodTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// First sample column name
    #[schemars(description = "Name of the first sample column (numeric).")]
    pub x: String,

    /// Second sample column name
    #[schemars(description = "Name of the second sample column (numeric).")]
    pub y: String,

    /// Alternative hypothesis direction
    #[schemars(
        description = "Alternative hypothesis: 'two.sided' (default), 'greater', or 'less'. For 'greater': scale of x > scale of y."
    )]
    pub alternative: Option<String>,
}

/// Request for correlation test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CorTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// First variable column name
    #[schemars(description = "Name of the first numeric variable column.")]
    pub x: String,

    /// Second variable column name
    #[schemars(description = "Name of the second numeric variable column.")]
    pub y: String,

    /// Correlation method
    #[schemars(
        description = "Correlation method: 'pearson' (default), 'spearman' (rank), or 'kendall' (tau)."
    )]
    pub method: Option<String>,

    /// Alternative hypothesis
    #[schemars(
        description = "Alternative hypothesis: 'two.sided' (default), 'greater', or 'less'."
    )]
    pub alternative: Option<String>,

    /// Confidence level
    #[schemars(description = "Confidence level for the interval (0-1). Default: 0.95.")]
    pub conf_level: Option<f64>,
}

/// Request for Quade test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct QuadeTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Value column name
    #[schemars(description = "Name of the numeric column containing the measured values.")]
    pub value: String,

    /// Treatment/group column name
    #[schemars(description = "Name of the column containing treatment/group labels.")]
    pub treatment: String,

    /// Block/subject column name
    #[schemars(description = "Name of the column containing block/subject identifiers.")]
    pub block: String,
}

/// Request for Friedman rank sum test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FriedmanTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Value column name
    #[schemars(description = "Name of the numeric column containing the measured values.")]
    pub value: String,

    /// Treatment/group column name
    #[schemars(description = "Name of the column containing treatment/group labels.")]
    pub treatment: String,

    /// Block/subject column name
    #[schemars(description = "Name of the column containing block/subject identifiers.")]
    pub block: String,
}

/// Request for Shapiro-Wilk normality test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ShapiroWilkRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to test for normality
    #[schemars(description = "Name of the numeric column to test for normality.")]
    pub column: String,
}
