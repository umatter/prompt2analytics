//! Request types for spatial tools.

use schemars::JsonSchema;
use serde::Deserialize;

/// Request for creating spatial neighbors.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SpatialNeighborsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column containing longitude or x coordinate
    #[schemars(description = "Column name for longitude or x coordinate.")]
    pub x_coord: String,

    /// Column containing latitude or y coordinate
    #[schemars(description = "Column name for latitude or y coordinate.")]
    pub y_coord: String,

    /// Neighbor method: 'knn' (default), 'distance', or 'distance_longlat'
    #[schemars(
        description = "Method for defining neighbors: 'knn' (k nearest neighbours, default; spdep::knearneigh), 'distance' (d_min < d <= d_max in coordinate units; spdep::dnearneigh) or 'distance_longlat' (the same band in great-circle km on lon/lat). Case-insensitive; anything else is an error."
    )]
    pub method: Option<String>,

    /// Number of neighbors for knn method
    #[schemars(description = "Number of nearest neighbors (for 'knn' method). Default is 5.")]
    pub k: Option<usize>,

    /// Maximum distance (for distance-based methods)
    #[schemars(
        description = "Maximum distance threshold (for 'distance' or 'distance_longlat' methods). Units are in coordinate units or kilometers for longlat."
    )]
    pub d_max: Option<f64>,

    /// Minimum distance (for distance-based methods)
    #[schemars(description = "Minimum distance threshold (for 'distance' methods). Default is 0.")]
    pub d_min: Option<f64>,

    /// Name to store the spatial weights under
    #[schemars(
        description = "Name to store the spatial weights for later use. If not provided, uses dataset name + '_weights'."
    )]
    pub weights_name: Option<String>,

    /// Weight style
    #[schemars(
        description = "Weight style as spdep::nb2listw(style =): 'W' (row-standardised, default; also 'row'), 'B' (binary), 'C' (globally standardised: sums to n), 'U' (C / n: sums to 1), 'minmax' (divided by min(max row sum, max column sum)), 'S' (variance-stabilising, Tiefelsdorf et al. 1999). Case-insensitive; anything else is an error. A style other than 'W' changes the scale of W, and so of rho and lambda in the spatial models."
    )]
    pub style: Option<String>,
}

/// Request for Moran's I test for spatial autocorrelation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MoranTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Variable to test for spatial autocorrelation
    #[schemars(description = "Name of the variable column to test for spatial autocorrelation.")]
    pub variable: String,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool)."
    )]
    pub weights: String,

    /// Alternative hypothesis
    #[schemars(
        description = "Alternative hypothesis, as spdep::moran.test: 'greater' (positive autocorrelation, default), 'less' (negative) or 'two.sided' (also 'twosided'). Case-insensitive; anything else is an error."
    )]
    pub alternative: Option<String>,
}

/// Request for spatial LM tests.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SpatialLmTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool)."
    )]
    pub weights: String,
}

/// Request for Spatial Autoregressive (SAR) model.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SarModelRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool)."
    )]
    pub weights: String,

    /// Spatial Durbin model (include WX)
    #[schemars(
        description = "If true, estimates Spatial Durbin Model (SDM) which includes spatially lagged covariates (WX). Default is false."
    )]
    pub durbin: Option<bool>,

    /// Compute spatial impacts
    #[schemars(
        description = "If true, computes direct, indirect, and total spatial impacts. Default is true."
    )]
    pub compute_impacts: Option<bool>,
}

/// Request for Spatial Error Model (SEM).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SemModelRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool)."
    )]
    pub weights: String,
}

/// Request for Spatial GMM with Heteroscedasticity Robustness (sphet).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SphetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool)."
    )]
    pub weights: String,

    /// Model type: 'lag' (SAR), 'error' (SEM), or 'sarar' (both)
    #[schemars(
        description = "Model type: 'lag' (default; SAR, y = lambda*Wy + Xb + e), 'error' (SEM, y = Xb + u, u = rho*Wu + e) or 'sarar' (both). Case-insensitive; anything else is an error."
    )]
    pub model: Option<String>,

    /// Standard error type: 'robust', 'hac', or 'standard'
    #[schemars(
        description = "Standard error type: 'robust' (default; White, heteroskedasticity-robust), 'hac' (spatial HAC on distances, Kelejian-Prucha 2007; needs coord_x and coord_y) or 'standard' (classical). Case-insensitive; anything else is an error."
    )]
    pub se_type: Option<String>,

    /// Spatial HAC kernel (for se_type='hac')
    #[schemars(
        description = "Kernel of the spatial HAC (se_type='hac'), a function of distance over bandwidth: 'epanechnikov' (default, as sphet), 'triangular' (also 'bartlett'), 'rectangular' (also 'truncated'), 'bisquare', 'parzen', 'tukey_hanning' or 'quadratic_spectral'."
    )]
    pub kernel: Option<String>,

    /// No longer used with se_type='hac'
    #[schemars(
        description = "Not used: the spatial HAC takes each unit's bandwidth as its largest distance to its neighbours in the weights (sphet's variable bandwidth). Leave unset; a value with se_type='hac' is refused."
    )]
    pub bandwidth: Option<usize>,

    /// x coordinate column for the spatial HAC
    #[schemars(
        description = "Column of the x coordinate (or longitude) of each unit, required with se_type='hac': the column the weights were built from, one row per unit of the weights."
    )]
    #[serde(default)]
    pub coord_x: Option<String>,

    /// y coordinate column for the spatial HAC
    #[schemars(
        description = "Column of the y coordinate (or latitude) of each unit, required with se_type='hac'."
    )]
    #[serde(default)]
    pub coord_y: Option<String>,

    /// Instrument order (default 2)
    #[schemars(description = "Order of spatial lag instruments [X, WX, W^2X, ...]. Default is 2.")]
    pub instrument_order: Option<usize>,
}

/// Request for SAR Probit model (spatial lag probit).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SarProbitRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Binary dependent variable (Y) column name
    #[schemars(description = "Name of the binary dependent variable (Y) column (0/1 values).")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool)."
    )]
    pub weights: String,

    /// Number of MCMC draws
    #[schemars(description = "Number of MCMC draws after burn-in. Default is 1000.")]
    pub n_draws: Option<usize>,

    /// Burn-in draws
    #[schemars(description = "Number of burn-in draws to discard. Default is 200.")]
    pub burn_in: Option<usize>,

    /// Compute spatial impacts
    #[schemars(
        description = "If true, computes the direct, indirect and total marginal effects on the probability scale, each with its posterior SD. Default true; needs at least one x column (an intercept-only model has no marginal effects: set false)."
    )]
    pub compute_impacts: Option<bool>,

    /// Random seed
    #[schemars(
        description = "Random seed of the MCMC sampler; when omitted, the value of set_seed, else 42. The output reports the seed used."
    )]
    pub seed: Option<u64>,
}

/// Request for SEM Probit model (spatial error probit).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SemProbitRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Binary dependent variable (Y) column name
    #[schemars(description = "Name of the binary dependent variable (Y) column (0/1 values).")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool)."
    )]
    pub weights: String,

    /// Number of MCMC draws
    #[schemars(description = "Number of MCMC draws after burn-in. Default is 1000.")]
    pub n_draws: Option<usize>,

    /// Burn-in draws
    #[schemars(description = "Number of burn-in draws to discard. Default is 200.")]
    pub burn_in: Option<usize>,

    /// Random seed
    #[schemars(
        description = "Random seed of the MCMC sampler; when omitted, the value of set_seed, else 42. The output reports the seed used."
    )]
    pub seed: Option<u64>,
}

/// Request for Spatial Panel ML estimation (spml).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SpmlRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset containing panel data.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity identifier column
    #[schemars(description = "Name of the entity/cross-sectional identifier column.")]
    pub entity_col: String,

    /// Time identifier column
    #[schemars(description = "Name of the time period identifier column.")]
    pub time_col: String,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool). Must match number of cross-sectional entities."
    )]
    pub weights: String,

    /// Panel model type
    #[schemars(
        description = "Panel model type: 'within' (fixed effects, default), 'random' (random effects) or 'pooling' (no effects). Case-insensitive; anything else is an error."
    )]
    pub model: Option<String>,

    /// Include spatial lag
    #[schemars(
        description = "If true, includes the spatial lag of the dependent variable (rho*W*y). Default false; lag = true with the default spatial_error fits the SARAR model, as spml(lag = TRUE); set spatial_error = 'none' for the lag model alone."
    )]
    pub lag: Option<bool>,

    /// Spatial error type
    #[schemars(
        description = "Spatial error term as splm::spml(spatial.error =): 'b' (Baltagi, default; also 'baltagi'), 'kkp' (Kapoor-Kelejian-Prucha; differs from 'b' only with model = 'random') or 'none'. With lag = false, 'none' is refused (no spatial parameter). Case-insensitive; anything else is an error."
    )]
    pub spatial_error: Option<String>,

    /// Effect type
    #[schemars(
        description = "Effect type: 'individual' (default), 'time' or 'twoways'. Case-insensitive; anything else is an error."
    )]
    pub effect: Option<String>,
}

/// Request for Spatial Panel GMM estimation (spgm).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SpgmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset containing panel data.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity identifier column
    #[schemars(description = "Name of the entity/cross-sectional identifier column.")]
    pub entity_col: String,

    /// Time identifier column
    #[schemars(description = "Name of the time period identifier column.")]
    pub time_col: String,

    /// Name of stored spatial weights
    #[schemars(
        description = "Name of previously created spatial weights (from spatial_neighbors tool)."
    )]
    pub weights: String,

    /// Estimation method
    #[schemars(
        description = "GMM estimation method: 'w2sls' (within/fixed effects, default), 'b2sls' (between), 'g2sls' (GLS random effects) or 'ec2sls' (Baltagi's EC2SLS). Case-insensitive; anything else is an error."
    )]
    pub method: Option<String>,

    /// Include spatial lag
    #[schemars(
        description = "If true, includes spatial lag of dependent variable (uses IV/GMM for identification). Default: false for 'w2sls', true for the other methods (they estimate the spatial lag model only)."
    )]
    pub lag: Option<bool>,

    /// Include spatial error
    #[schemars(
        description = "If true, includes spatially correlated error term (estimated with method 'w2sls' only). Default: true for 'w2sls', false for the other methods."
    )]
    pub spatial_error: Option<bool>,
}
