//! Request types for timeseries tools.

use schemars::JsonSchema;
use serde::Deserialize;

/// Request for autocorrelation function (ACF).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AcfRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time series column name
    #[schemars(description = "Name of the numeric time series column.")]
    pub column: String,

    /// Maximum lag
    #[schemars(description = "Maximum lag to compute. Default: min(10*log10(n), n-1).")]
    pub lag_max: Option<usize>,

    /// Type of ACF
    #[schemars(
        description = "Type of autocorrelation: 'correlation' (default), 'covariance', or 'partial' (PACF)."
    )]
    pub acf_type: Option<String>,
}

/// Request for cross-correlation function (CCF).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CcfRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// First time series column name
    #[schemars(description = "Name of the first (X) numeric time series column.")]
    pub x: String,

    /// Second time series column name
    #[schemars(description = "Name of the second (Y) numeric time series column.")]
    pub y: String,

    /// Maximum lag
    #[schemars(
        description = "Maximum lag to compute (in both directions). Default: min(10*log10(n), n-1)."
    )]
    pub lag_max: Option<usize>,
}

/// Request for spectral density estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SpectrumRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time series column name
    #[schemars(description = "Name of the numeric time series column.")]
    pub column: String,

    /// Smoothing spans (Daniell kernel widths)
    #[schemars(
        description = "Vector of odd integers for modified Daniell smoothers, e.g., [3,3] or [5,5,5]. If omitted, returns raw periodogram. Multiple values are convolved for more smoothing."
    )]
    pub spans: Option<Vec<usize>>,

    /// Taper proportion
    #[schemars(
        description = "Proportion of data to taper at ends using split cosine bell (0 to 0.5). Default: 0.1. Reduces spectral leakage."
    )]
    pub taper: Option<f64>,

    /// Whether to detrend
    #[schemars(
        description = "Whether to remove linear trend before computing spectrum. Default: true."
    )]
    pub detrend: Option<bool>,

    /// Method for spectrum estimation
    #[schemars(
        description = "Estimation method: 'pgram' (periodogram, default) or 'ar' (autoregressive). AR method fits an AR model and computes its theoretical spectrum."
    )]
    pub method: Option<String>,

    /// AR order (for method='ar')
    #[schemars(
        description = "AR model order for method='ar'. If omitted, order is selected by AIC."
    )]
    pub ar_order: Option<usize>,
}

/// Request for Box-Pierce or Ljung-Box test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BoxTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time series column name
    #[schemars(
        description = "Name of the numeric time series column to test for autocorrelation."
    )]
    pub column: String,

    /// Number of lags
    #[schemars(
        description = "Number of autocorrelation lags to include in the test statistic. Default: 1. Common choices: 10 for short series, 20 for longer series."
    )]
    pub lag: Option<usize>,

    /// Test type
    #[schemars(
        description = "Test type: 'ljung-box' (default, better finite-sample properties) or 'box-pierce' (simpler, classic version)."
    )]
    pub test_type: Option<String>,

    /// Degrees of freedom adjustment
    #[schemars(
        description = "Number of parameters already estimated (subtract from df). When testing ARMA(p,q) residuals, set fitdf = p + q for proper df adjustment. Default: 0."
    )]
    pub fitdf: Option<usize>,
}

/// Request for Phillips-Perron unit root test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PPTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time series column name
    #[schemars(description = "Name of the numeric time series column to test for unit root.")]
    pub column: String,

    /// Use short truncation lag
    #[schemars(
        description = "Whether to use short truncation lag formula: trunc(4*(n/100)^0.25). If false, uses long formula: trunc(12*(n/100)^0.25). Default: true."
    )]
    pub lshort: Option<bool>,
}

/// Request for VAR (Vector Autoregression) model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VarRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to include in the VAR model
    #[schemars(
        description = "Names of the columns to include in the VAR model (e.g., ['gdp', 'inflation', 'interest_rate'])."
    )]
    pub columns: Vec<String>,

    /// Number of lags
    #[schemars(description = "Number of lags to include in the VAR model.")]
    pub lags: usize,
}

/// Request for Granger causality test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GrangerRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (the "caused" variable)
    #[schemars(
        description = "Name of the dependent variable column - the variable being predicted (the 'caused' variable)."
    )]
    pub dependent: String,

    /// Potential causing variable
    #[schemars(
        description = "Name of the potential causing variable column - tests whether this variable helps predict the dependent variable."
    )]
    pub cause: String,

    /// Number of lags (optional, default: automatic selection)
    #[schemars(
        description = "Number of lags to include in the test. Default: automatic selection using Schwert's rule."
    )]
    pub lags: Option<usize>,
}

/// Request for VARMA (Vector ARMA) model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VarmaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to include in the VARMA model
    #[schemars(description = "Names of the columns to include in the VARMA model.")]
    pub columns: Vec<String>,

    /// AR lags (p)
    #[schemars(description = "Number of autoregressive (AR) lags.")]
    pub p: usize,

    /// MA lags (q)
    #[schemars(description = "Number of moving average (MA) lags.")]
    pub q: usize,
}

/// Request for VECM (Vector Error Correction Model).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VecmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to include in the VECM model
    #[schemars(
        description = "Names of the columns to include in the VECM model. Should be I(1) cointegrated series."
    )]
    pub columns: Vec<String>,

    /// Number of lags
    #[schemars(description = "Number of lags for the VECM (must be at least 2).")]
    pub lags: usize,

    /// Cointegration rank
    #[schemars(
        description = "Cointegration rank (number of cointegrating relationships). Must be between 1 and k-1 where k is the number of variables."
    )]
    pub rank: usize,
}

/// Request for VAR Impulse Response Functions.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VarIrfRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to include in the VAR model
    #[schemars(description = "Names of the columns to include in the VAR model.")]
    pub columns: Vec<String>,

    /// Number of lags
    #[schemars(description = "Number of lags for the VAR model.")]
    pub lags: usize,

    /// Number of IRF steps/periods
    #[schemars(description = "Number of periods to compute impulse responses for.")]
    pub steps: usize,
}

/// Request for ARIMA model fitting.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ArimaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// AR order (p)
    #[schemars(description = "Number of autoregressive (AR) terms.")]
    pub p: usize,

    /// Differencing order (d)
    #[schemars(description = "Number of differences to make the series stationary.")]
    pub d: usize,

    /// MA order (q)
    #[schemars(description = "Number of moving average (MA) terms.")]
    pub q: usize,
}

/// Request for ARIMA forecasting.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ArimaForecastRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// AR order (p)
    #[schemars(description = "Number of autoregressive (AR) terms.")]
    pub p: usize,

    /// Differencing order (d)
    #[schemars(description = "Number of differences to make the series stationary.")]
    pub d: usize,

    /// MA order (q)
    #[schemars(description = "Number of moving average (MA) terms.")]
    pub q: usize,

    /// Forecast horizon
    #[schemars(description = "Number of periods to forecast ahead.")]
    pub horizon: usize,
}

/// Request for GARCH model fitting (volatility modeling).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GarchRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with return series
    #[schemars(
        description = "Name of the column containing returns (e.g., stock returns, percentage changes)."
    )]
    pub column: String,

    /// ARCH order (p) - default 1
    #[schemars(description = "Number of ARCH (lagged squared residual) terms. Default: 1.")]
    pub p: Option<usize>,

    /// GARCH order (q) - default 1
    #[schemars(description = "Number of GARCH (lagged variance) terms. Default: 1.")]
    pub q: Option<usize>,

    /// Include mean in model - default true
    #[schemars(description = "Whether to include a mean term in the model. Default: true.")]
    pub include_mean: Option<bool>,
}

/// Request for MSTL decomposition.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MstlRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// Seasonal periods
    #[schemars(
        description = "Seasonal periods to extract (e.g., [7, 365] for daily data with weekly and yearly seasonality)."
    )]
    pub periods: Vec<usize>,
}

/// Request for changepoint detection.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ChangepointRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// Penalty for adding a changepoint (optional, uses BIC if not specified)
    #[schemars(
        description = "Penalty for adding a changepoint. Higher values = fewer changepoints. Default uses BIC (log(n))."
    )]
    pub penalty: Option<f64>,

    /// Minimum segment length between changepoints
    #[schemars(description = "Minimum number of observations between changepoints. Default is 2.")]
    pub min_segment_length: Option<usize>,

    /// Detection method: 'pelt' or 'binary'
    #[schemars(
        description = "Algorithm to use: 'pelt' (Pruned Exact Linear Time, default) or 'binary' (Binary Segmentation)."
    )]
    pub method: Option<String>,

    /// Type of change to detect: 'mean', 'variance', or 'both'
    #[schemars(description = "Type of change to detect: 'mean' (default), 'variance', or 'both'.")]
    pub change_type: Option<String>,
}

/// Request for Holt-Winters exponential smoothing.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HoltWintersRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// Seasonal period
    #[schemars(
        description = "Number of observations per seasonal cycle (e.g., 12 for monthly data with yearly seasonality, 4 for quarterly)."
    )]
    pub period: usize,

    /// Seasonal type
    #[schemars(
        description = "Type of seasonality: 'additive' (default) for constant seasonal variation, 'multiplicative' for proportional variation."
    )]
    pub seasonal: Option<String>,

    /// Level smoothing parameter alpha (0-1)
    #[schemars(
        description = "Smoothing parameter for the level component (0-1). If not specified, will be optimized."
    )]
    pub alpha: Option<f64>,

    /// Trend smoothing parameter beta (0-1)
    #[schemars(
        description = "Smoothing parameter for the trend component (0-1). If not specified, will be optimized."
    )]
    pub beta: Option<f64>,

    /// Seasonal smoothing parameter gamma (0-1)
    #[schemars(
        description = "Smoothing parameter for the seasonal component (0-1). If not specified, will be optimized."
    )]
    pub gamma: Option<f64>,

    /// Forecast horizon (optional)
    #[schemars(
        description = "Number of periods to forecast ahead. If specified, returns forecasts in addition to fitted values."
    )]
    pub horizon: Option<usize>,
}

/// Request for AR (autoregressive) model fitting.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ArModelRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// Use AIC for order selection
    #[schemars(description = "Whether to use AIC for automatic order selection. Default: true.")]
    pub aic: Option<bool>,

    /// Maximum order to consider
    #[schemars(description = "Maximum AR order to consider. Default: min(n-1, 10*log10(n)).")]
    pub order_max: Option<usize>,

    /// Specific order to use
    #[schemars(
        description = "Specific AR order to fit. If provided with aic=false, uses this exact order."
    )]
    pub order: Option<usize>,

    /// Fitting method
    #[schemars(description = "Method for fitting: 'yule-walker' (default), 'burg', or 'ols'.")]
    pub method: Option<String>,

    /// Demean the series
    #[schemars(description = "Whether to remove the mean before fitting. Default: true.")]
    pub demean: Option<bool>,
}

/// Request for classical seasonal decomposition.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DecomposeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// Seasonal period
    #[schemars(
        description = "Number of observations per seasonal cycle (e.g., 12 for monthly data with yearly seasonality, 4 for quarterly)."
    )]
    pub period: usize,

    /// Decomposition type
    #[schemars(
        description = "Type of decomposition: 'additive' (default, Y = Trend + Seasonal + Random) or 'multiplicative' (Y = Trend × Seasonal × Random)."
    )]
    pub decompose_type: Option<String>,
}

/// Request for structural time series model fitting.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StructTsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// Model type
    #[schemars(
        description = "Type of structural model: 'level' (local level/random walk + noise), 'trend' (local linear trend), or 'bsm' (basic structural model with seasonality)."
    )]
    pub model_type: Option<String>,

    /// Seasonal period (for BSM only)
    #[schemars(
        description = "Seasonal period for BSM model (e.g., 12 for monthly data with yearly seasonality). Required for 'bsm' type."
    )]
    pub period: Option<usize>,
}

/// Request for CausalImpact analysis (Bayesian Structural Time Series for causal inference).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CausalImpactRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with response variable (outcome)
    #[schemars(description = "Name of the column containing the response variable to analyze.")]
    pub response_col: String,

    /// Column name with time index
    #[schemars(
        description = "Name of the column containing the time index (must be integer-like: integers, dates as days since epoch, etc.)."
    )]
    pub time_col: String,

    /// Start of pre-intervention period (inclusive)
    #[schemars(
        description = "Start time value of the pre-intervention period (inclusive). This is the period used to train the model."
    )]
    pub pre_period_start: i64,

    /// End of pre-intervention period (inclusive)
    #[schemars(description = "End time value of the pre-intervention period (inclusive).")]
    pub pre_period_end: i64,

    /// Start of post-intervention period (inclusive)
    #[schemars(
        description = "Start time value of the post-intervention period (inclusive). This is when the intervention occurs."
    )]
    pub post_period_start: i64,

    /// End of post-intervention period (inclusive)
    #[schemars(description = "End time value of the post-intervention period (inclusive).")]
    pub post_period_end: i64,

    /// Optional control series columns
    #[schemars(
        description = "Optional names of columns to use as control time series. These should be correlated with the response but unaffected by the intervention."
    )]
    pub control_cols: Option<Vec<String>>,

    /// Significance level (default 0.05)
    #[schemars(
        description = "Significance level for credible intervals. Default: 0.05 (95% credible intervals)."
    )]
    pub alpha: Option<f64>,

    /// Include trend component
    #[schemars(description = "Whether to include a trend component in the model. Default: false.")]
    pub include_trend: Option<bool>,

    /// Seasonal period
    #[schemars(
        description = "If the data has seasonality, specify the period (e.g., 12 for monthly data with yearly seasonality)."
    )]
    pub seasonal_period: Option<usize>,
}

/// Request for cumulative periodogram (cpgram).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CpgramRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time series column name
    #[schemars(description = "Name of the time series column.")]
    pub column: String,

    /// Taper proportion
    #[schemars(description = "Proportion of data to taper at each end (0.0-0.5). Default is 0.1.")]
    pub taper: Option<f64>,
}

/// Request for Toeplitz matrix construction.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ToeplitzRequest {
    /// First column (and row for symmetric matrix)
    #[schemars(
        description = "Values for the first column. For a symmetric matrix, this is also the first row."
    )]
    pub column: Vec<f64>,

    /// First row (optional, for asymmetric matrix)
    #[schemars(
        description = "Values for the first row. If not specified, creates a symmetric matrix using column values."
    )]
    pub row: Option<Vec<f64>>,
}

/// Request for time series lag operation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LagRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to lag
    #[schemars(description = "Name of the time series column to lag.")]
    pub column: String,

    /// Number of lags
    #[schemars(
        description = "Number of positions to lag. Positive = shift back, negative = shift forward. Default is 1."
    )]
    pub k: Option<i32>,
}

/// Request for time series embedding.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EmbedRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to embed
    #[schemars(description = "Name of the time series column to embed.")]
    pub column: String,

    /// Embedding dimension
    #[schemars(description = "Number of columns in the embedding matrix (lag dimension).")]
    pub dimension: usize,
}

/// Request for inverse differencing.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiffinvRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column containing differences
    #[schemars(description = "Name of the column containing differenced values.")]
    pub column: String,

    /// Initial values for integration
    #[schemars(description = "Initial values to start the cumulative sum.")]
    pub xi: Option<Vec<f64>>,

    /// Difference lag
    #[schemars(description = "Lag for differencing. Default is 1.")]
    pub lag: Option<usize>,

    /// Number of differences to invert
    #[schemars(description = "Number of times differencing was applied. Default is 1.")]
    pub differences: Option<usize>,
}

/// Request for linear filtering.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FilterRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to filter
    #[schemars(description = "Name of the time series column to filter.")]
    pub column: String,

    /// Filter coefficients
    #[schemars(description = "Filter coefficients for convolution or recursive filtering.")]
    pub filter: Vec<f64>,

    /// Filter method
    #[schemars(description = "Method: 'convolution' (default) or 'recursive'.")]
    pub method: Option<String>,

    /// Sides for convolution
    #[schemars(description = "For convolution: 1 (past only) or 2 (centered, default).")]
    pub sides: Option<usize>,
}

/// Request for time series window extraction.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct WindowRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to extract window from
    #[schemars(description = "Name of the time series column.")]
    pub column: String,

    /// Start index (0-based)
    #[schemars(
        description = "Start index (0-based, inclusive). If not specified, starts from beginning."
    )]
    pub start: Option<usize>,

    /// End index (0-based)
    #[schemars(description = "End index (0-based, exclusive). If not specified, goes to end.")]
    pub end: Option<usize>,
}

/// Request for theoretical ARMA ACF.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ArmaAcfRequest {
    /// AR coefficients
    #[schemars(description = "Autoregressive coefficients (phi). Empty for MA-only model.")]
    pub ar: Option<Vec<f64>>,

    /// MA coefficients
    #[schemars(description = "Moving average coefficients (theta). Empty for AR-only model.")]
    pub ma: Option<Vec<f64>>,

    /// Maximum lag
    #[schemars(description = "Maximum lag for ACF computation. Default is 10.")]
    pub lag_max: Option<usize>,

    /// Whether to compute PACF
    #[schemars(description = "If true, compute partial ACF instead of ACF. Default is false.")]
    pub pacf: Option<bool>,
}

/// Request for ARMA to MA conversion.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ArmaToMaRequest {
    /// AR coefficients
    #[schemars(description = "Autoregressive coefficients (phi).")]
    pub ar: Option<Vec<f64>>,

    /// MA coefficients
    #[schemars(description = "Moving average coefficients (theta).")]
    pub ma: Option<Vec<f64>>,

    /// Number of MA coefficients to compute
    #[schemars(description = "Number of MA (psi) weights to compute. Default is 10.")]
    pub lag_max: Option<usize>,
}

/// Request for ACF to AR conversion.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct Acf2ArRequest {
    /// ACF values
    #[schemars(description = "Autocorrelation function values starting at lag 1.")]
    pub acf: Vec<f64>,
}

/// Request for ARIMA simulation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ArimaSimRequest {
    /// Number of observations to simulate
    #[schemars(description = "Number of observations to simulate.")]
    pub n: usize,

    /// AR coefficients
    #[schemars(description = "Autoregressive coefficients.")]
    pub ar: Option<Vec<f64>>,

    /// MA coefficients
    #[schemars(description = "Moving average coefficients.")]
    pub ma: Option<Vec<f64>>,

    /// Differencing order
    #[schemars(description = "Order of differencing. Default is 0.")]
    pub d: Option<usize>,

    /// Innovation standard deviation
    #[schemars(description = "Standard deviation of innovations. Default is 1.0.")]
    pub sd: Option<f64>,

    /// Random seed
    #[schemars(description = "Random seed for reproducibility.")]
    pub seed: Option<u64>,
}

/// Request for running median smoothing.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunmedRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column to smooth
    #[schemars(description = "Name of the column to apply running median.")]
    pub column: String,

    /// Window width
    #[schemars(description = "Width of the median window (must be odd).")]
    pub k: usize,

    /// End rule
    #[schemars(description = "How to handle ends: 'keep' (default), 'constant', or 'median'.")]
    pub endrule: Option<String>,
}

