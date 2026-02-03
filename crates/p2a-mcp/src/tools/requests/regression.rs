//! Request types for regression analysis tools.
//!
//! This module contains request structs for:
//! - OLS regression
//! - Regression diagnostics (Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF)
//! - Breusch-Godfrey test for serial correlation
//! - RESET test for functional form
//! - Wald test for nested models
//! - Harvey-Collier test for linearity
//! - HAC (Newey-West) standard errors
//! - Bootstrap covariance estimation
//! - Driscoll-Kraay panel-robust SEs
//! - Quantile regression
//! - Clustered standard errors
//! - Nonlinear least squares (NLS)
//! - LOESS local regression
//! - Super smoother (supsmu)
//! - Tukey's resistant line
//! - Stepwise regression
//! - GLS (Generalized Least Squares)
//! - Smoothing splines

use schemars::JsonSchema;
use serde::Deserialize;

/// Request for OLS regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct OlsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,
}

/// Request for regression diagnostics.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiagnosticsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,
}

/// Request for Breusch-Godfrey test for serial correlation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BgTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Order of serial correlation to test (default: 1)
    #[schemars(
        description = "Maximum lag order for serial correlation test. Default is 1 (first-order)."
    )]
    pub order: Option<usize>,

    /// Test statistic type: 'chisq' (default) or 'f'
    #[schemars(
        description = "Type of test statistic: 'chisq' for chi-squared (asymptotic) or 'f' for F-test (finite sample correction). Default: 'chisq'."
    )]
    pub test_type: Option<String>,

    /// Fill value for initial lagged residuals (default: 0.0)
    #[schemars(
        description = "Value to fill for initial lagged residuals before enough observations exist. Default: 0.0."
    )]
    pub fill: Option<f64>,
}

/// Request for Ramsey's RESET test for functional form.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ResetTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Powers to test (default: [2, 3])
    #[schemars(
        description = "Powers to use for augmentation. Default: [2, 3] tests for quadratic and cubic terms."
    )]
    pub powers: Option<Vec<usize>>,

    /// Type of augmentation: 'fitted' (default), 'regressor', or 'princomp'
    #[schemars(
        description = "Type of augmentation: 'fitted' (powers of fitted values, default), 'regressor' (powers of regressors), or 'princomp' (powers of first principal component)."
    )]
    pub reset_type: Option<String>,
}

/// Request for Wald test comparing nested models.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct WaldTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables for the unrestricted (full) model
    #[schemars(
        description = "Column names for the unrestricted (full) model. Must be a superset of restricted model variables."
    )]
    pub x_unrestricted: Vec<String>,

    /// Independent variables for the restricted (null) model
    #[schemars(
        description = "Column names for the restricted (null) model. Must be a subset of unrestricted model variables."
    )]
    pub x_restricted: Vec<String>,

    /// Use F-test (default: true) or Chi-squared test
    #[schemars(
        description = "If true (default), use F-test. If false, use Chi-squared test. F-test is more common for finite samples."
    )]
    pub use_f_test: Option<bool>,
}

/// Request for Harvey-Collier test for linearity.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HarveyCollierRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,
}

/// Request for HAC (Newey-West) standard errors.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HacRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Bandwidth (number of lags). Default: automatic Newey-West selection
    #[schemars(
        description = "Number of lags for HAC estimation. If not provided, uses automatic Newey-West bandwidth: floor(4 * (n/100)^(2/9))."
    )]
    pub bandwidth: Option<usize>,

    /// Kernel type: 'bartlett' (default), 'parzen', 'qs', 'truncated', 'tukey-hanning'
    #[schemars(
        description = "Kernel function for weighting lags: 'bartlett' (Newey-West, default), 'parzen', 'qs' (quadratic spectral), 'truncated', or 'tukey-hanning'."
    )]
    pub kernel: Option<String>,

    /// Whether to use VAR(1) prewhitening (default: false)
    #[schemars(description = "Apply VAR(1) prewhitening before HAC estimation. Default: false.")]
    pub prewhiten: Option<bool>,
}

/// Request for bootstrap covariance estimation (vcovBS).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BootstrapCovRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Number of bootstrap replications (default: 999)
    #[schemars(
        description = "Number of bootstrap replications. Default: 999. More replications give more precise SE estimates."
    )]
    pub n_boot: Option<usize>,

    /// Bootstrap type: 'pairs' (default), 'residual', or 'wild'
    #[schemars(
        description = "Bootstrap method: 'pairs' (xy, most robust), 'residual' (assumes homoskedasticity), 'wild' (Rademacher weights, robust to heteroskedasticity)."
    )]
    pub bootstrap_type: Option<String>,

    /// Random seed for reproducibility
    #[schemars(description = "Random seed for reproducibility. If not provided, uses entropy.")]
    pub seed: Option<u64>,
}

/// Request for OLS with clustered standard errors.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct OlsClusteredRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// First cluster dimension column (e.g., "firm_id")
    #[schemars(description = "Column name for first clustering dimension (e.g., 'firm_id').")]
    pub cluster1: String,

    /// Second cluster dimension column (optional, for two-way clustering)
    #[schemars(
        description = "Optional column for second clustering dimension (e.g., 'year'). If provided, two-way clustering is used."
    )]
    pub cluster2: Option<String>,
}

/// Request for Driscoll-Kraay panel-robust standard errors.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DriscollKraayRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Time period identifier column
    #[schemars(
        description = "Column containing time period identifiers (e.g., 'year', 'quarter'). Used to aggregate scores within periods."
    )]
    pub time_col: String,

    /// Bandwidth for HAC kernel (optional)
    #[schemars(
        description = "Bandwidth for HAC kernel. Default uses Newey-West rule: floor(T^0.25). Higher values allow for more serial correlation."
    )]
    pub bandwidth: Option<usize>,

    /// Kernel type for HAC
    #[schemars(
        description = "HAC kernel: 'bartlett' (default), 'parzen', 'qs', 'truncated', 'tukey-hanning'."
    )]
    pub kernel: Option<String>,
}

/// Request for quantile regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct QuantRegRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Quantile (tau) to estimate (0-1)
    #[schemars(
        description = "Quantile to estimate. Must be between 0 and 1. Common values: 0.25 (first quartile), 0.5 (median), 0.75 (third quartile). Default: 0.5 (median regression)."
    )]
    pub tau: Option<f64>,

    /// Multiple quantiles to estimate
    #[schemars(
        description = "Optional: estimate multiple quantiles at once (e.g., [0.25, 0.5, 0.75] for quartiles). If provided, 'tau' is ignored."
    )]
    pub taus: Option<Vec<f64>>,
}

/// Request for nonlinear least squares (NLS).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct NlsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variable (X) column name
    #[schemars(description = "Name of the independent variable (X) column.")]
    pub x: String,

    /// Model type to fit
    #[schemars(
        description = "Nonlinear model to fit: 'exponential_decay' (y = a*exp(-b*x) + c), 'exponential_growth' (y = a*exp(b*x)), 'michaelis_menten' (y = Vmax*x/(Km+x)), 'logistic' (y = K/(1+exp(-r*(x-x0)))), 'power' (y = a*x^b), 'asymptotic' (y = a - b*exp(-c*x))."
    )]
    pub model: String,

    /// Starting values for parameters
    #[schemars(
        description = "Initial parameter values. For exponential_decay: [a, b, c]. For exponential_growth: [a, b]. For michaelis_menten: [Vmax, Km]. For logistic: [K, r, x0]. For power: [a, b]. For asymptotic: [a, b, c]."
    )]
    pub start: Vec<f64>,

    /// Algorithm to use
    #[schemars(
        description = "Optimization algorithm: 'levenberg_marquardt' (default, more robust) or 'gauss_newton' (faster but may not converge)."
    )]
    pub algorithm: Option<String>,
}

/// Request for LOESS local regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LoessRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column to smooth.")]
    pub y: String,

    /// Independent variable (X) column name
    #[schemars(description = "Name of the independent variable (X) column.")]
    pub x: String,

    /// Smoothing parameter (span)
    #[schemars(
        description = "Smoothing parameter controlling neighborhood size. Range (0,1] uses proportion of data; >1 uses all points. Default: 0.75. Smaller = more wiggly, larger = smoother."
    )]
    pub span: Option<f64>,

    /// Polynomial degree (1=linear, 2=quadratic)
    #[schemars(
        description = "Degree of local polynomial: 1 (linear) or 2 (quadratic, default). Quadratic captures curvature better at boundaries."
    )]
    pub degree: Option<usize>,

    /// Use robust fitting
    #[schemars(
        description = "Use robust fitting with iterative reweighting to downweight outliers. Default: false (gaussian family). Set true for 'symmetric' family."
    )]
    pub robust: Option<bool>,
}

/// Request for super smoother (supsmu).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SupsmuRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X column name
    #[schemars(description = "Name of the predictor column (X variable).")]
    pub x: String,

    /// Y column name
    #[schemars(description = "Name of the response column (Y variable).")]
    pub y: String,

    /// Optional weight column
    #[schemars(description = "Optional name of the weight column.")]
    pub weights: Option<String>,

    /// Span parameter (0-1)
    #[schemars(
        description = "Fixed span fraction (0-1). If not specified, cross-validation selects optimal span."
    )]
    pub span: Option<f64>,

    /// Bass parameter for smoothness (0-10)
    #[schemars(
        description = "Bass parameter controlling smoothness (0-10). Higher = smoother. Default is 0."
    )]
    pub bass: Option<f64>,

    /// Periodic boundary conditions
    #[schemars(description = "Whether to treat x as periodic on [0, 1]. Default is false.")]
    pub periodic: Option<bool>,
}

/// Request for Tukey's resistant line.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LineRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X column name
    #[schemars(description = "Name of the predictor column (X variable).")]
    pub x: String,

    /// Y column name
    #[schemars(description = "Name of the response column (Y variable).")]
    pub y: String,

    /// Number of polishing iterations
    #[schemars(description = "Number of polishing iterations. Default is 1.")]
    pub iter: Option<usize>,
}

/// Request for stepwise regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StepRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response variable
    #[schemars(description = "Name of the response column (Y variable).")]
    pub response: String,

    /// All candidate predictors
    #[schemars(description = "Names of all candidate predictor columns.")]
    pub predictors: Vec<String>,

    /// Direction of stepwise selection
    #[schemars(description = "Direction: 'both' (default), 'forward', or 'backward'.")]
    pub direction: Option<String>,

    /// Selection criterion
    #[schemars(description = "Selection criterion: 'aic' (default) or 'bic'.")]
    pub criterion: Option<String>,
}

/// Request for GLS (Generalized Least Squares).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GlsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Response variable column
    #[schemars(description = "Name of the dependent (y) variable column.")]
    pub y: String,

    /// Predictor variable columns
    #[schemars(description = "Names of the independent (x) variable columns.")]
    pub x: Vec<String>,

    /// Include intercept
    #[schemars(description = "Whether to include an intercept. Default is true.")]
    pub intercept: Option<bool>,

    /// Correlation structure type
    #[schemars(
        description = "Correlation structure: 'ar1' (default), 'compound_symmetry', or 'identity' (OLS)."
    )]
    pub correlation: Option<String>,

    /// Correlation parameter (rho)
    #[schemars(
        description = "Correlation parameter rho for AR(1) or compound symmetry. If omitted with 'ar1', auto-estimated from OLS residuals."
    )]
    pub rho: Option<f64>,
}

/// Request for smooth spline fitting.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SmoothSplineRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X column
    #[schemars(description = "Name of the x (independent) column.")]
    pub x: String,

    /// Y column
    #[schemars(description = "Name of the y (dependent) column.")]
    pub y: String,

    /// Smoothing parameter (spar)
    #[schemars(
        description = "Smoothing parameter (0 to 1). If omitted, uses generalized cross-validation (GCV) to select automatically."
    )]
    pub spar: Option<f64>,

    /// Degrees of freedom
    #[schemars(
        description = "Equivalent degrees of freedom. Alternative to spar. If both omitted, uses GCV."
    )]
    pub df: Option<f64>,

    /// Points at which to predict
    #[schemars(
        description = "Optional points at which to evaluate the fitted spline. If omitted, returns fit at data points."
    )]
    pub xout: Option<Vec<f64>>,
}

/// Request for glmnet (elastic net, lasso, ridge) regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GlmnetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Elastic net mixing parameter (0=ridge, 1=lasso)
    #[schemars(
        description = "Elastic net mixing parameter: 0 for pure ridge, 1 for lasso (default), between 0-1 for elastic net."
    )]
    pub alpha: Option<f64>,

    /// Sequence of lambda values
    #[schemars(
        description = "Optional sequence of lambda (regularization) values. If omitted, automatically generates a path."
    )]
    pub lambda: Option<Vec<f64>>,

    /// Number of lambda values in path
    #[schemars(description = "Number of lambda values in the path. Default is 100.")]
    pub nlambda: Option<usize>,

    /// Whether to standardize predictors
    #[schemars(description = "Whether to standardize predictors before fitting. Default is true.")]
    pub standardize: Option<bool>,

    /// Model family: gaussian (linear) or binomial (logistic)
    #[schemars(
        description = "Model family: 'gaussian' (default) for linear regression, 'binomial' for logistic regression."
    )]
    pub family: Option<String>,
}

/// Request for cross-validated glmnet.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CvGlmnetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Elastic net mixing parameter (0=ridge, 1=lasso)
    #[schemars(
        description = "Elastic net mixing parameter: 0 for pure ridge, 1 for lasso (default), between 0-1 for elastic net."
    )]
    pub alpha: Option<f64>,

    /// Number of cross-validation folds
    #[schemars(description = "Number of cross-validation folds. Default is 10.")]
    pub nfolds: Option<usize>,

    /// Random seed for reproducibility
    #[schemars(description = "Random seed for fold assignment reproducibility.")]
    pub seed: Option<u64>,

    /// Number of lambda values in path
    #[schemars(description = "Number of lambda values in the path. Default is 100.")]
    pub nlambda: Option<usize>,

    /// Whether to standardize predictors
    #[schemars(description = "Whether to standardize predictors before fitting. Default is true.")]
    pub standardize: Option<bool>,

    /// Model family: gaussian (linear) or binomial (logistic)
    #[schemars(
        description = "Model family: 'gaussian' (default) for linear regression, 'binomial' for logistic regression."
    )]
    pub family: Option<String>,
}

/// Request for ridge regression (shortcut for glmnet with alpha=0).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RidgeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Sequence of lambda values
    #[schemars(
        description = "Optional sequence of lambda (regularization) values. If omitted, automatically generates a path."
    )]
    pub lambda: Option<Vec<f64>>,
}

/// Request for lasso regression (shortcut for glmnet with alpha=1).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LassoRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Sequence of lambda values
    #[schemars(
        description = "Optional sequence of lambda (regularization) values. If omitted, automatically generates a path."
    )]
    pub lambda: Option<Vec<f64>>,
}
