//! Request types for spatial tools.

use schemars::JsonSchema;
use serde::Deserialize;

/// Request for creating spatial neighbors.
#[derive(Debug, Deserialize, JsonSchema)]
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
        description = "Method for defining neighbors: 'knn' (k-nearest neighbors, default), 'distance' (within distance), 'distance_longlat' (great-circle distance for lon/lat)."
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
        description = "Weight style: 'W' or 'row' (row-standardized, default), 'B' (binary), 'C' (global standardized), 'U' (unstandardized)."
    )]
    pub style: Option<String>,
}

/// Request for Moran's I test for spatial autocorrelation.
#[derive(Debug, Deserialize, JsonSchema)]
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
        description = "Alternative hypothesis: 'greater' (positive autocorrelation, default), 'less' (negative), 'two.sided'."
    )]
    pub alternative: Option<String>,
}

/// Request for spatial LM tests.
#[derive(Debug, Deserialize, JsonSchema)]
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
        description = "Model type: 'lag' for SAR (y=lambda*Wy+Xb+e), 'error' for SEM (y=Xb+u, u=rho*Wu+e), or 'sarar' for combined. Default is 'lag'."
    )]
    pub model: Option<String>,

    /// Standard error type: 'robust', 'hac', or 'standard'
    #[schemars(
        description = "Standard error type: 'robust' for heteroscedasticity-robust (Kelejian-Prucha 2010), 'hac' for HAC (Kelejian-Prucha 2007), or 'standard' for homoscedastic. Default is 'robust'."
    )]
    pub se_type: Option<String>,

    /// HAC kernel type (for se_type='hac')
    #[schemars(
        description = "HAC kernel: 'bartlett', 'parzen', 'quadratic_spectral', 'tukey_hanning', or 'truncated'. Default is 'bartlett'."
    )]
    pub kernel: Option<String>,

    /// HAC bandwidth (for se_type='hac')
    #[schemars(
        description = "Bandwidth for HAC estimation. If not specified, uses automatic bandwidth selection."
    )]
    pub bandwidth: Option<usize>,

    /// Instrument order (default 2)
    #[schemars(description = "Order of spatial lag instruments [X, WX, W^2X, ...]. Default is 2.")]
    pub instrument_order: Option<usize>,
}

/// Request for SAR Probit model (spatial lag probit).
#[derive(Debug, Deserialize, JsonSchema)]
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
        description = "If true, computes direct, indirect, and total spatial impacts. Default is true."
    )]
    pub compute_impacts: Option<bool>,

    /// Random seed
    #[schemars(description = "Random seed for reproducibility. Optional.")]
    pub seed: Option<u64>,
}

/// Request for SEM Probit model (spatial error probit).
#[derive(Debug, Deserialize, JsonSchema)]
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
    #[schemars(description = "Random seed for reproducibility. Optional.")]
    pub seed: Option<u64>,
}

/// Request for Spatial Panel ML estimation (spml).
#[derive(Debug, Deserialize, JsonSchema)]
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
        description = "Panel model type: 'within' (fixed effects, default), 'random' (random effects), or 'pooling' (no effects)."
    )]
    pub model: Option<String>,

    /// Include spatial lag
    #[schemars(
        description = "If true, includes spatial lag of dependent variable (rho*W*y). Default is false."
    )]
    pub lag: Option<bool>,

    /// Spatial error type
    #[schemars(
        description = "Spatial error specification: 'none' (default), 'baltagi' (Baltagi-type), or 'kkp' (Kapoor-Kelejian-Prucha type)."
    )]
    pub spatial_error: Option<String>,

    /// Effect type
    #[schemars(description = "Effect type: 'individual' (default), 'time', or 'twoways'.")]
    pub effect: Option<String>,
}

/// Request for Spatial Panel GMM estimation (spgm).
#[derive(Debug, Deserialize, JsonSchema)]
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
        description = "GMM estimation method: 'w2sls' (within/fixed effects, default), 'g2sls' (GLS random effects), 'b2sls' (between), or 'ec2sls' (Baltagi EC2SLS)."
    )]
    pub method: Option<String>,

    /// Include spatial lag
    #[schemars(
        description = "If true, includes spatial lag of dependent variable (uses IV/GMM for identification). Default is false."
    )]
    pub lag: Option<bool>,

    /// Include spatial error
    #[schemars(
        description = "If true, includes spatially correlated error term. Default is true."
    )]
    pub spatial_error: Option<bool>,
}
