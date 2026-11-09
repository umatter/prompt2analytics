//! Request types for statistical tests and analysis.
//!
//! This module contains request structs for:
//! - Log-linear models
//! - ANOVA model tables and contrasts
//! - Weighted statistics
//! - Sphericity tests
//! - Robust statistics (fivenum, IQR, MAD, ECDF, density)
//! - Spline/interpolation tools

use schemars::JsonSchema;
use serde::Deserialize;

/// Request for log-linear model fitting.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoglinRequest {
    /// Name/ID of the dataset
    #[schemars(
        description = "Name or ID of a previously loaded dataset containing contingency table data."
    )]
    pub dataset: String,

    /// Count column
    #[schemars(
        description = "Name of the column containing the cell counts. Rows with a missing count or a missing factor label are dropped (as R's xtabs) and reported."
    )]
    pub count_column: String,

    /// Factor columns that define the contingency table dimensions
    #[schemars(
        description = "Names of the factor columns that define the contingency table dimensions. Levels are ordered as R's factor() (numeric codes numerically)."
    )]
    pub factor_columns: Vec<String>,

    /// Model margins to fit
    #[schemars(
        description = "Margins to fit in the model. Each margin is a list of factor indices (0-based). For example, [[0,1], [1,2]] fits the (0,1) and (1,2) two-way interactions. If not specified, fits an independence model (all main effects only)."
    )]
    pub margins: Option<Vec<Vec<usize>>>,

    /// Convergence tolerance
    #[schemars(
        description = "Convergence tolerance of the IPF algorithm: the largest absolute deviation of the fitted margins from the observed ones (R's eps; default 0.1)."
    )]
    pub eps: Option<f64>,

    /// Maximum iterations
    #[schemars(description = "Maximum iterations for IPF algorithm (default: 20).")]
    pub max_iter: Option<usize>,
}

/// Request for model tables from ANOVA.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelTablesRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response column
    #[schemars(description = "Name of the response column (Y variable).")]
    #[serde(alias = "y", alias = "value")]
    pub response: String,

    /// Factor column(s) for ANOVA
    #[schemars(
        description = "Name of the factor column for one-way ANOVA, or list of two factor columns for two-way ANOVA."
    )]
    pub factors: Vec<String>,

    /// Type of table
    #[schemars(
        description = "Type of table: 'means' (default; also 'mean') or 'effects' (deviations from the grand mean; also 'effect'). Other values are refused."
    )]
    pub table_type: Option<String>,

    /// Whether to compute standard errors
    #[schemars(description = "Whether to compute standard errors. Default is true.")]
    pub se: Option<bool>,
}

/// Request for standard errors of contrasts (se.contrast).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SeContrastRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response column
    #[schemars(description = "Name of the response column (Y variable).")]
    #[serde(alias = "y", alias = "value")]
    pub response: String,

    /// Factor column for ANOVA
    #[schemars(description = "Name of the factor column for one-way ANOVA.")]
    #[serde(alias = "group", alias = "group_col")]
    pub factor: String,

    /// Contrast coefficients
    #[schemars(
        description = "Contrast coefficients. Each inner array is one contrast (must sum to 0)."
    )]
    pub contrasts: Option<Vec<Vec<f64>>>,

    /// Contrast type (if contrasts not provided)
    #[schemars(
        description = "Type of contrasts to generate: 'treatment' (default), 'helmert', 'sum' or 'poly' (also 'polynomial'); R's names 'contr.treatment', 'contr.helmert', 'contr.sum' and 'contr.poly' are accepted. Only used if contrasts not provided."
    )]
    pub contrast_type: Option<String>,
}

/// Request for weighted mean.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WeightedMeanRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to compute weighted mean of
    #[schemars(description = "Name of the numeric column to compute weighted mean.")]
    pub column: String,

    /// Weight column
    #[schemars(description = "Name of the weight column.")]
    pub weights: String,

    /// Whether to remove NA values
    #[schemars(
        description = "true (default): rows where the value or the weight is missing are dropped together and reported; false: a missing value or weight is an error (R's weighted.mean(na.rm = FALSE) would return NA)."
    )]
    pub na_rm: Option<bool>,
}

/// Request for weighted covariance matrix.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CovWtRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to compute covariance for
    #[schemars(
        description = "Names of the numeric columns to include in the covariance matrix. Missing values are refused, as R's cov.wt does."
    )]
    pub columns: Vec<String>,

    /// Weight column
    #[schemars(
        description = "Name of the weight column. Missing values are refused, as R's cov.wt does."
    )]
    pub weights: String,

    /// Whether to center the data
    #[schemars(
        description = "Whether to center the data (subtract weighted mean). Default is true."
    )]
    pub center: Option<bool>,

    /// Covariance method
    #[schemars(
        description = "Method for computing covariance, R's cov.wt method: 'unbiased' (default) or 'ML' (also 'mle', 'maximum_likelihood'). Other values are refused."
    )]
    pub method: Option<String>,
}

/// Request for Mauchly's sphericity test.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MauchlyTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns for repeated measures
    #[schemars(
        description = "Names of columns representing repeated measures (at least 3). Subjects (rows) with a missing value in any of them are dropped, as R's lm-based mauchly.test does, and reported."
    )]
    pub columns: Vec<String>,

    /// Covariance structure under the null (R's Sigma)
    #[schemars(
        description = "Optional covariance structure Sigma under the null (R's mauchly.test Sigma), p x p in row-major order for the p columns; default the identity. The test then checks sphericity of the transformed measures relative to Sigma."
    )]
    pub sigma: Option<Vec<f64>>,
}

/// Request for Tukey's five-number summary.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FivenumRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to summarize
    #[schemars(
        description = "Name of the numeric column to compute five-number summary. Missing values are dropped and reported (as na.rm = TRUE)."
    )]
    pub column: String,
}

/// Request for interquartile range.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IqrRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to compute IQR for
    #[schemars(
        description = "Name of the numeric column. Missing values are dropped and reported (as na.rm = TRUE)."
    )]
    pub column: String,

    /// Quantile type (1-9, default 7)
    #[schemars(description = "Quantile type (1-9). Type 7 (default) matches R's default.")]
    pub qtype: Option<usize>,
}

/// Request for median absolute deviation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MadRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to compute MAD for
    #[schemars(
        description = "Name of the numeric column. Missing values are dropped and reported (as na.rm = TRUE)."
    )]
    pub column: String,

    /// Center (default: median)
    #[schemars(
        description = "Center to use: 'median' (default) or 'mean'. Other values are refused."
    )]
    pub center: Option<String>,

    /// Scaling constant (default: 1.4826 for normal consistency)
    #[schemars(
        description = "Scaling constant. Default 1.4826 makes MAD consistent for normal data."
    )]
    pub constant: Option<f64>,
}

/// Request for empirical CDF.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EcdfRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to compute ECDF for
    #[schemars(
        description = "Name of the numeric column. Missing values are dropped and reported (as na.rm = TRUE)."
    )]
    pub column: String,

    /// Optional values at which to evaluate the ECDF
    #[schemars(
        description = "Optional specific values at which to evaluate the ECDF. If omitted, returns ECDF at all sorted unique data values."
    )]
    pub at: Option<Vec<f64>>,
}

/// Request for kernel density estimation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DensityRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to estimate density for
    #[schemars(
        description = "Name of the numeric column. Missing values are dropped and reported (as na.rm = TRUE)."
    )]
    pub column: String,

    /// Kernel type
    #[schemars(
        description = "Kernel function: 'gaussian' (default; also 'normal'), 'epanechnikov', 'rectangular' (also 'uniform'), 'triangular', 'biweight' (also 'quartic') or 'cosine'. Other values are refused."
    )]
    pub kernel: Option<String>,

    /// Bandwidth
    #[schemars(
        description = "Bandwidth for smoothing. If omitted, uses Silverman's rule of thumb."
    )]
    pub bw: Option<f64>,

    /// Number of evaluation points
    #[schemars(description = "Number of points to evaluate density at. Default is 512.")]
    pub n: Option<usize>,
}

/// Request for spline interpolation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SplineRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X column (knot locations)
    #[schemars(
        description = "Name of the x (independent) column. Rows with a missing x or y are dropped (as R) and reported."
    )]
    pub x: String,

    /// Y column (values at knots)
    #[schemars(description = "Name of the y (dependent) column.")]
    pub y: String,

    /// Points at which to interpolate
    #[schemars(
        description = "Points at which to evaluate the spline. If omitted, the spline is evaluated at n equally spaced points over the range of x (default 3 times the number of complete (x, y) pairs, as R's spline(), capped at 200)."
    )]
    pub xout: Option<Vec<f64>>,

    /// Number of evaluation points (R's n)
    #[schemars(
        description = "Number of equally spaced evaluation points over the range of x when xout is omitted (R's n). Default: 3 times the number of complete (x, y) pairs, as R's spline(), capped at 200; give n to get more. Ignored when xout is given."
    )]
    pub n: Option<usize>,

    /// Spline method
    #[schemars(
        description = "Spline method: 'fmm' (default, Forsythe-Malcolm-Moler), 'natural', 'periodic' or 'monoH.FC' (monotone Fritsch-Carlson; also 'hyman', 'monotone'). Other values are refused."
    )]
    pub method: Option<String>,
}

/// Request for linear approximation/interpolation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ApproxRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X column
    #[schemars(
        description = "Name of the x (independent) column. Rows with a missing x or y are dropped (as R) and reported."
    )]
    pub x: String,

    /// Y column
    #[schemars(description = "Name of the y (dependent) column.")]
    pub y: String,

    /// Points at which to interpolate
    #[schemars(description = "Points at which to interpolate.")]
    pub xout: Vec<f64>,

    /// Interpolation method
    #[schemars(
        description = "Interpolation method: 'linear' (default) or 'constant' (also 'step'). Other values are refused."
    )]
    pub method: Option<String>,

    /// Rule for extrapolation
    #[schemars(
        description = "Rule for points outside the range of x, R's rule: 'na' (default, return NA; also '1') or 'nearest' (use the boundary value; also 'boundary', 'const', '2'). Other values are refused."
    )]
    pub rule: Option<String>,
}

/// Request for MANOVA (Multivariate Analysis of Variance).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ManovaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response variable column names (must have at least 2)
    #[schemars(
        description = "Names of the response (dependent) variable columns. Must be numeric. MANOVA requires at least 2 response variables."
    )]
    pub response_vars: Vec<String>,

    /// Factor (grouping) variable column name
    #[schemars(
        description = "Name of the factor (grouping) variable column. Groups observations for comparison."
    )]
    pub factor: String,

    /// Which test statistic to emphasize (default: Pillai)
    #[schemars(
        description = "The test summarised in the interpretation (all four are always reported): 'pillai' (default), 'wilks', 'hotelling-lawley' (also 'hotelling'), or 'roy' (Roy's largest root)."
    )]
    pub test: Option<String>,
}

/// Request for factor analysis.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FactorAnalysisRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Variable columns to include in factor analysis
    #[schemars(
        description = "Names of the numeric columns to include in factor analysis. Should have at least n_factors + 1 variables."
    )]
    pub columns: Vec<String>,

    /// Number of factors to extract
    #[schemars(
        description = "Number of factors to extract. Must be between 1 and (number of variables - 1). Rule of thumb: fewer factors that explain most variance."
    )]
    pub n_factors: usize,

    /// Rotation method
    #[schemars(
        description = "Factor rotation method: 'varimax' (default, orthogonal), 'promax' (oblique, allows correlated factors) or 'none'. Other values are refused."
    )]
    pub rotation: Option<String>,

    /// Factor scores method
    #[schemars(
        description = "Method for computing factor scores: 'none' (default), 'regression' (Thomson; also 'thomson') or 'bartlett' (weighted least squares). The scores are returned (the first 50 rows, fewer with many factors) or stored for every row with output_name."
    )]
    pub scores: Option<String>,

    /// Dataset that receives the scores
    #[schemars(
        description = "With scores = 'regression' or 'bartlett': name of a new dataset that receives the input dataset plus one column per factor (Factor1, Factor2, ...). Without it, the first 50 rows of scores (fewer with many factors) are returned in the output."
    )]
    pub output_name: Option<String>,
}

/// Request for median polish (robust two-way decomposition).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MedpolishRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column names to use as the matrix
    #[schemars(
        description = "Names of the numeric columns to use as the matrix for median polish. The columns become the columns of the matrix, and each row of the dataset becomes a row of the matrix. Integer and float columns are accepted."
    )]
    pub columns: Vec<String>,

    /// Convergence tolerance
    #[schemars(
        description = "Convergence tolerance (default: 0.01). Iteration stops when the proportional reduction in sum of absolute residuals is less than this value."
    )]
    pub eps: Option<f64>,

    /// Maximum iterations
    #[schemars(description = "Maximum number of iterations (default: 10).")]
    pub max_iter: Option<usize>,

    /// Handle missing values
    #[schemars(
        description = "Whether to ignore missing cells (null or NaN) when computing medians (default: false, as R's medpolish; then a missing cell is an error)."
    )]
    pub na_rm: Option<bool>,
}

/// Request for power analysis of proportion test.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PowerPropTestRequest {
    /// Sample size per group
    #[schemars(
        description = "Sample size per group. Leave None to solve for required sample size."
    )]
    pub n: Option<f64>,

    /// Proportion in group 1
    #[schemars(description = "Proportion in first group (0-1). Leave None to solve for it.")]
    pub p1: Option<f64>,

    /// Proportion in group 2
    #[schemars(
        description = "Proportion in second group (0-1). Leave None to solve for detectable difference."
    )]
    pub p2: Option<f64>,

    /// Significance level
    #[schemars(description = "Significance level (Type I error rate). Default: 0.05.")]
    pub sig_level: Option<f64>,

    /// Power
    #[schemars(
        description = "Power (1 - Type II error rate). Leave None to solve for power. Common target: 0.80."
    )]
    pub power: Option<f64>,

    /// Alternative hypothesis
    #[schemars(
        description = "Alternative hypothesis: 'two.sided' (default) or 'one.sided'; spellings such as 'two-sided' and 'one-sided' are accepted."
    )]
    pub alternative: Option<String>,

    /// Parameter to solve for
    #[schemars(
        description = "Parameter to solve for when it is not the one omitted: 'n', 'p1', 'p2', 'sig.level' (or 'alpha') or 'power'. Needed only to solve for sig_level, which otherwise defaults to 0.05 (R's power.prop.test)."
    )]
    pub solve_for: Option<String>,
}

/// Request for isotonic regression.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IsoregRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column for x values (predictor)
    #[schemars(
        description = "Name of the column to use as x (predictor). If not provided, uses row index. Missing values are refused, as R's isoreg does."
    )]
    pub x_column: Option<String>,

    /// Column for y values (response)
    #[schemars(
        description = "Name of the column to use as y (response variable). Missing values are refused, as R's isoreg does."
    )]
    pub y_column: String,
}

/// Request for power analysis of ANOVA test.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PowerAnovaTestRequest {
    /// Number of groups
    #[schemars(
        description = "Number of groups (at least 2). Leave None to solve for it (reported fractional, as R). In the hosted profile at most the server's design_groups limit (100 by default)."
    )]
    pub groups: Option<usize>,

    /// Sample size per group
    #[schemars(
        description = "Sample size per group. Leave None to solve for required sample size."
    )]
    pub n: Option<f64>,

    /// Between-group variance
    #[schemars(
        description = "Between-group variance (variance of the group means, divisor groups - 1). Leave None to solve for it."
    )]
    pub between_var: Option<f64>,

    /// Within-group variance
    #[schemars(
        description = "Within-group variance (variance within each group). Leave None to solve for it; there is no default (R's power.anova.test)."
    )]
    pub within_var: Option<f64>,

    /// Significance level
    #[schemars(description = "Significance level (Type I error rate). Default: 0.05.")]
    pub sig_level: Option<f64>,

    /// Power
    #[schemars(
        description = "Power (1 - Type II error rate). Leave None to solve for power. Common target: 0.80."
    )]
    pub power: Option<f64>,

    /// Parameter to solve for
    #[schemars(
        description = "Parameter to solve for when it is not the one omitted: 'groups', 'n', 'between.var', 'within.var', 'sig.level' (or 'alpha') or 'power'. Needed only to solve for sig_level, which otherwise defaults to 0.05 (R's power.anova.test)."
    )]
    pub solve_for: Option<String>,
}

/// Request for Tukey's HSD (Honest Significant Differences) test.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TukeyHsdRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response variable column name
    #[schemars(description = "Name of the response (dependent) variable column. Must be numeric.")]
    #[serde(alias = "y", alias = "value")]
    pub response: String,

    /// Factor (grouping) variable column name
    #[schemars(
        description = "Name of the factor (grouping) variable column. Defines groups to compare."
    )]
    #[serde(alias = "group", alias = "group_col")]
    pub factor: String,

    /// Confidence level (default: 0.95)
    #[schemars(
        description = "Confidence level for intervals (default: 0.95). Common values: 0.90, 0.95, 0.99."
    )]
    pub conf_level: Option<f64>,
}

/// Request for Mahalanobis distance computation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MahalanobisRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column names to use as variables
    #[schemars(
        description = "Names of the columns to use as variables for computing Mahalanobis distance."
    )]
    pub columns: Vec<String>,

    /// Optional center vector
    #[schemars(description = "Optional center vector. If not provided, uses column means.")]
    pub center: Option<Vec<f64>>,
}

/// Request for canonical correlation analysis.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CancorRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// First set of variables (X)
    #[schemars(
        description = "Names of the columns for the first set of variables (X). These are the 'predictor' or 'input' variables."
    )]
    pub x_columns: Vec<String>,

    /// Second set of variables (Y)
    #[schemars(
        description = "Names of the columns for the second set of variables (Y). These are the 'response' or 'output' variables."
    )]
    pub y_columns: Vec<String>,

    /// Whether to center X variables
    #[schemars(
        description = "Whether to center X variables by subtracting column means. Default: true."
    )]
    pub xcenter: Option<bool>,

    /// Whether to center Y variables
    #[schemars(
        description = "Whether to center Y variables by subtracting column means. Default: true."
    )]
    pub ycenter: Option<bool>,
}

/// Request for correlation matrix.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CorrelationRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,
}

/// Request for one-way ANOVA.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OneWayAnovaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response variable column name
    #[schemars(description = "Name of the response (dependent) variable column. Must be numeric.")]
    #[serde(alias = "y", alias = "value")]
    pub response: String,

    /// Factor (grouping) variable column name
    #[schemars(
        description = "Name of the factor (grouping) variable column. Groups observations for comparison."
    )]
    #[serde(alias = "group", alias = "group_col")]
    pub factor: String,
}

/// Request for two-way ANOVA.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TwoWayAnovaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response variable column name
    #[schemars(description = "Name of the response (dependent) variable column. Must be numeric.")]
    #[serde(alias = "y", alias = "value")]
    pub response: String,

    /// First factor variable column name
    #[schemars(description = "Name of the first factor variable column (e.g., 'treatment').")]
    pub factor_a: String,

    /// Second factor variable column name
    #[schemars(description = "Name of the second factor variable column (e.g., 'block').")]
    pub factor_b: String,

    /// Whether to include interaction term
    #[schemars(
        description = "Whether to include the interaction term (factor_a × factor_b). Default is true."
    )]
    pub interaction: Option<bool>,

    /// Type of sums of squares.
    #[schemars(
        description = "Sums of squares: 'I' (default; sequential, as R aov), 'II' or 'III' (as car::Anova; Type III uses sum-to-zero contrasts and is refused when empty cells alias coefficients). Types I and II agree on balanced designs; on unbalanced data they differ."
    )]
    pub ss_type: Option<String>,
}

/// Request for power analysis of t-test.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PowerTTestRequest {
    /// Sample size (per group for two-sample)
    #[schemars(
        description = "Sample size per group. Leave None to solve for required sample size."
    )]
    pub n: Option<f64>,

    /// True difference in means
    #[schemars(
        description = "True difference in means (effect size in original units). Leave None to solve for detectable effect."
    )]
    pub delta: Option<f64>,

    /// Standard deviation
    #[schemars(
        description = "Standard deviation. Default: 1 unless solve_for = 'sd' (use d = delta/sd as standardized effect size)."
    )]
    pub sd: Option<f64>,

    /// Significance level
    #[schemars(description = "Significance level (Type I error rate). Default: 0.05.")]
    pub sig_level: Option<f64>,

    /// Power
    #[schemars(
        description = "Power (1 - Type II error rate). Leave None to solve for power. Common target: 0.80."
    )]
    pub power: Option<f64>,

    /// Type of t-test
    #[schemars(
        description = "Type of t-test: 'two.sample' (default), 'one.sample' or 'paired'; spellings such as 'one-sample' and 'two_sample' are accepted."
    )]
    pub test_type: Option<String>,

    /// Alternative hypothesis
    #[schemars(
        description = "Alternative hypothesis: 'two.sided' (default) or 'one.sided'; spellings such as 'two-sided' and 'one-sided' are accepted."
    )]
    pub alternative: Option<String>,

    /// Parameter to solve for
    #[schemars(
        description = "Parameter to solve for when it is not the one omitted: 'n', 'delta', 'sd', 'sig.level' (or 'alpha') or 'power'. Needed only to solve for sd or sig_level, which otherwise default to 1 and 0.05 (R's power.t.test)."
    )]
    pub solve_for: Option<String>,
}
