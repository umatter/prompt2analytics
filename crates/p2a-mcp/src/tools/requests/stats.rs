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
pub struct LoglinRequest {
    /// Name/ID of the dataset
    #[schemars(
        description = "Name or ID of a previously loaded dataset containing contingency table data."
    )]
    pub dataset: String,

    /// Count column
    #[schemars(description = "Name of the column containing the cell counts.")]
    pub count_column: String,

    /// Factor columns that define the contingency table dimensions
    #[schemars(
        description = "Names of the factor columns that define the contingency table dimensions."
    )]
    pub factor_columns: Vec<String>,

    /// Model margins to fit
    #[schemars(
        description = "Margins to fit in the model. Each margin is a list of factor indices (0-based). For example, [[0,1], [1,2]] fits the (0,1) and (1,2) two-way interactions. If not specified, fits an independence model (all main effects only)."
    )]
    pub margins: Option<Vec<Vec<usize>>>,

    /// Convergence tolerance
    #[schemars(description = "Convergence tolerance for IPF algorithm (default: 0.1).")]
    pub eps: Option<f64>,

    /// Maximum iterations
    #[schemars(description = "Maximum iterations for IPF algorithm (default: 20).")]
    pub max_iter: Option<usize>,
}

/// Request for model tables from ANOVA.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ModelTablesRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response column
    #[schemars(description = "Name of the response column (Y variable).")]
    pub response: String,

    /// Factor column(s) for ANOVA
    #[schemars(
        description = "Name of the factor column for one-way ANOVA, or list of two factor columns for two-way ANOVA."
    )]
    pub factors: Vec<String>,

    /// Type of table
    #[schemars(
        description = "Type of table: 'means' (default) or 'effects' (deviations from grand mean)."
    )]
    pub table_type: Option<String>,

    /// Whether to compute standard errors
    #[schemars(description = "Whether to compute standard errors. Default is true.")]
    pub se: Option<bool>,
}

/// Request for standard errors of contrasts (se.contrast).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SeContrastRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response column
    #[schemars(description = "Name of the response column (Y variable).")]
    pub response: String,

    /// Factor column for ANOVA
    #[schemars(description = "Name of the factor column for one-way ANOVA.")]
    pub factor: String,

    /// Contrast coefficients
    #[schemars(
        description = "Contrast coefficients. Each inner array is one contrast (must sum to 0)."
    )]
    pub contrasts: Option<Vec<Vec<f64>>>,

    /// Contrast type (if contrasts not provided)
    #[schemars(
        description = "Type of contrasts to generate: 'treatment', 'helmert', 'sum', or 'poly'. Only used if contrasts not provided."
    )]
    pub contrast_type: Option<String>,
}

/// Request for weighted mean.
#[derive(Debug, Deserialize, JsonSchema)]
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
    #[schemars(description = "Whether to remove NA values before computation. Default is true.")]
    pub na_rm: Option<bool>,
}

/// Request for weighted covariance matrix.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CovWtRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to compute covariance for
    #[schemars(description = "Names of the numeric columns to include in the covariance matrix.")]
    pub columns: Vec<String>,

    /// Weight column
    #[schemars(description = "Name of the weight column.")]
    pub weights: String,

    /// Whether to center the data
    #[schemars(
        description = "Whether to center the data (subtract weighted mean). Default is true."
    )]
    pub center: Option<bool>,

    /// Covariance method
    #[schemars(description = "Method for computing covariance: 'unbiased' (default) or 'ml'.")]
    pub method: Option<String>,
}

/// Request for Mauchly's sphericity test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MauchlyTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns for repeated measures
    #[schemars(
        description = "Names of columns representing repeated measures (at least 3 required)."
    )]
    pub columns: Vec<String>,
}

/// Request for Tukey's five-number summary.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FivenumRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to summarize
    #[schemars(description = "Name of the numeric column to compute five-number summary.")]
    pub column: String,
}

/// Request for interquartile range.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IqrRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to compute IQR for
    #[schemars(description = "Name of the numeric column.")]
    pub column: String,

    /// Quantile type (1-9, default 7)
    #[schemars(description = "Quantile type (1-9). Type 7 (default) matches R's default.")]
    pub qtype: Option<usize>,
}

/// Request for median absolute deviation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MadRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to compute MAD for
    #[schemars(description = "Name of the numeric column.")]
    pub column: String,

    /// Center (default: median)
    #[schemars(description = "Center to use. Default is 'median'. Can also use 'mean'.")]
    pub center: Option<String>,

    /// Scaling constant (default: 1.4826 for normal consistency)
    #[schemars(
        description = "Scaling constant. Default 1.4826 makes MAD consistent for normal data."
    )]
    pub constant: Option<f64>,
}

/// Request for empirical CDF.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EcdfRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to compute ECDF for
    #[schemars(description = "Name of the numeric column.")]
    pub column: String,

    /// Optional values at which to evaluate the ECDF
    #[schemars(
        description = "Optional specific values at which to evaluate the ECDF. If omitted, returns ECDF at all sorted unique data values."
    )]
    pub at: Option<Vec<f64>>,
}

/// Request for kernel density estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DensityRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to estimate density for
    #[schemars(description = "Name of the numeric column.")]
    pub column: String,

    /// Kernel type
    #[schemars(
        description = "Kernel function: 'gaussian' (default), 'epanechnikov', 'rectangular', 'triangular', 'biweight', or 'cosine'."
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
pub struct SplineRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X column (knot locations)
    #[schemars(description = "Name of the x (independent) column.")]
    pub x: String,

    /// Y column (values at knots)
    #[schemars(description = "Name of the y (dependent) column.")]
    pub y: String,

    /// Points at which to interpolate
    #[schemars(
        description = "Points at which to interpolate. If omitted, returns the spline coefficients."
    )]
    pub xout: Option<Vec<f64>>,

    /// Spline method
    #[schemars(
        description = "Spline method: 'fmm' (default, Forsythe-Malcolm-Moler), 'natural', 'periodic', or 'hyman' (monotone)."
    )]
    pub method: Option<String>,
}

/// Request for linear approximation/interpolation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ApproxRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X column
    #[schemars(description = "Name of the x (independent) column.")]
    pub x: String,

    /// Y column
    #[schemars(description = "Name of the y (dependent) column.")]
    pub y: String,

    /// Points at which to interpolate
    #[schemars(description = "Points at which to interpolate.")]
    pub xout: Vec<f64>,

    /// Interpolation method
    #[schemars(description = "Interpolation method: 'linear' (default) or 'constant'.")]
    pub method: Option<String>,

    /// Rule for extrapolation
    #[schemars(
        description = "Rule for handling points outside range: 'na' (default, return NA) or 'nearest' (use boundary value)."
    )]
    pub rule: Option<String>,
}

/// Request for MANOVA (Multivariate Analysis of Variance).
#[derive(Debug, Deserialize, JsonSchema)]
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
        description = "Test statistic to use: 'pillai' (default, most robust), 'wilks' (most popular), 'hotelling' (Hotelling-Lawley), or 'roy' (Roy's largest root). All four are always computed."
    )]
    pub test: Option<String>,
}

/// Request for factor analysis.
#[derive(Debug, Deserialize, JsonSchema)]
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
        description = "Factor rotation method: 'varimax' (default, orthogonal), 'promax' (oblique, allows correlated factors), or 'none'."
    )]
    pub rotation: Option<String>,

    /// Factor scores method
    #[schemars(
        description = "Method for computing factor scores: 'none' (default), 'regression' (Thomson's method), or 'bartlett' (weighted least squares)."
    )]
    pub scores: Option<String>,
}

/// Request for median polish (robust two-way decomposition).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MedpolishRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column names to use as the matrix
    #[schemars(
        description = "Names of the numeric columns to use as the matrix for median polish. The columns become the columns of the matrix, and each row of the dataset becomes a row of the matrix."
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
        description = "Whether to remove NaN values when computing medians (default: false). If false, the function fails if NaN values are present."
    )]
    pub na_rm: Option<bool>,
}

/// Request for power analysis of proportion test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PowerPropTestRequest {
    /// Sample size per group
    #[schemars(
        description = "Sample size per group. Leave None to solve for required sample size."
    )]
    pub n: Option<f64>,

    /// Proportion in group 1
    #[schemars(description = "Proportion in first group (0-1).")]
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
    #[schemars(description = "Alternative hypothesis: 'two.sided' (default) or 'one.sided'.")]
    pub alternative: Option<String>,
}

/// Request for isotonic regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IsoregRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column for x values (predictor)
    #[schemars(
        description = "Name of the column to use as x (predictor). If not provided, uses row index."
    )]
    pub x_column: Option<String>,

    /// Column for y values (response)
    #[schemars(description = "Name of the column to use as y (response variable).")]
    pub y_column: String,
}

/// Request for power analysis of ANOVA test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PowerAnovaTestRequest {
    /// Number of groups
    #[schemars(description = "Number of groups. Leave None to solve for minimum groups.")]
    pub groups: Option<usize>,

    /// Sample size per group
    #[schemars(
        description = "Sample size per group. Leave None to solve for required sample size."
    )]
    pub n: Option<f64>,

    /// Between-group variance
    #[schemars(description = "Between-group variance (variance of group means).")]
    pub between_var: Option<f64>,

    /// Within-group variance
    #[schemars(description = "Within-group variance (variance within each group). Default: 1.")]
    pub within_var: Option<f64>,

    /// Significance level
    #[schemars(description = "Significance level (Type I error rate). Default: 0.05.")]
    pub sig_level: Option<f64>,

    /// Power
    #[schemars(
        description = "Power (1 - Type II error rate). Leave None to solve for power. Common target: 0.80."
    )]
    pub power: Option<f64>,
}

/// Request for Tukey's HSD (Honest Significant Differences) test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TukeyHsdRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response variable column name
    #[schemars(description = "Name of the response (dependent) variable column. Must be numeric.")]
    pub response: String,

    /// Factor (grouping) variable column name
    #[schemars(
        description = "Name of the factor (grouping) variable column. Defines groups to compare."
    )]
    pub factor: String,

    /// Confidence level (default: 0.95)
    #[schemars(
        description = "Confidence level for intervals (default: 0.95). Common values: 0.90, 0.95, 0.99."
    )]
    pub conf_level: Option<f64>,
}

/// Request for Mahalanobis distance computation.
#[derive(Debug, Deserialize, JsonSchema)]
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
pub struct CorrelationRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,
}

/// Request for one-way ANOVA.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct OneWayAnovaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response variable column name
    #[schemars(description = "Name of the response (dependent) variable column. Must be numeric.")]
    pub response: String,

    /// Factor (grouping) variable column name
    #[schemars(
        description = "Name of the factor (grouping) variable column. Groups observations for comparison."
    )]
    pub factor: String,
}

/// Request for two-way ANOVA.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TwoWayAnovaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response variable column name
    #[schemars(description = "Name of the response (dependent) variable column. Must be numeric.")]
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
}

/// Request for power analysis of t-test.
#[derive(Debug, Deserialize, JsonSchema)]
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
        description = "Standard deviation. Default: 1 (use d = delta/sd as standardized effect size)."
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
    #[schemars(description = "Type of t-test: 'two.sample' (default), 'one.sample', or 'paired'.")]
    pub test_type: Option<String>,

    /// Alternative hypothesis
    #[schemars(description = "Alternative hypothesis: 'two.sided' (default) or 'one.sided'.")]
    pub alternative: Option<String>,
}
