//! Request types for panel data analysis tools.
//!
//! This module contains request structs for:
//! - Fixed Effects (FE) regression
//! - Random Effects (RE) regression
//! - Hausman specification test
//! - Variable Coefficients Model (PVCM)
//! - Mean Group (PMG) estimator
//! - Arellano-Bond / System GMM
//! - Panel GLS
//! - Panel unit root tests (LLC, IPS, Hadri)
//! - High-dimensional fixed effects (HDFE)

use schemars::JsonSchema;
use serde::Deserialize;

/// Request for Panel Fixed Effects regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PanelFERequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity/individual identifier column
    #[schemars(
        description = "Column name for entity/individual identifier (e.g., 'firm_id', 'person_id')."
    )]
    pub entity_var: String,
}

/// Request for Panel Random Effects regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PanelRERequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity/individual identifier column
    #[schemars(
        description = "Column name for entity/individual identifier (e.g., 'firm_id', 'person_id')."
    )]
    pub entity_var: String,
}

/// Request for Hausman specification test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HausmanRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity/individual identifier column
    #[schemars(
        description = "Column name for entity/individual identifier (e.g., 'firm_id', 'person_id')."
    )]
    pub entity_var: String,
}

/// Request for Variable Coefficients Model (PVCM).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PvcmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity/individual identifier column
    #[schemars(
        description = "Column name for entity/individual identifier (e.g., 'firm_id', 'person_id')."
    )]
    pub entity_var: String,

    /// Model type: 'within' or 'random'
    #[schemars(
        description = "Model type: 'within' for separate OLS per entity (default), 'random' for Swamy (1970) GLS estimator."
    )]
    pub model: Option<String>,
}

/// Request for Arellano-Bond / System GMM dynamic panel estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GmmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(
        description = "Names of the independent variable (X) columns. Include lagged dependent variable if desired."
    )]
    pub x: Vec<String>,

    /// Entity/individual identifier column
    #[schemars(
        description = "Column name for entity/individual identifier (e.g., 'firm_id', 'person_id')."
    )]
    pub entity_var: String,

    /// Time period identifier column
    #[schemars(description = "Column name for time period identifier (e.g., 'year', 'quarter').")]
    pub time_var: String,

    /// Number of lags of the dependent variable to include
    #[schemars(description = "Number of lags of Y to use as regressors. Default: 1.")]
    pub lags: Option<usize>,

    /// Transformation type: 'difference' (Arellano-Bond 1991) or 'system' (Blundell-Bond 1998)
    #[schemars(
        description = "Transformation: 'difference' for Arellano-Bond, 'system' for Blundell-Bond. Default: 'difference'."
    )]
    pub transform: Option<String>,

    /// Estimation step: 'onestep' or 'twostep'
    #[schemars(
        description = "Estimation step: 'onestep' or 'twostep'. Two-step is more efficient. Default: 'twostep'."
    )]
    pub step: Option<String>,

    /// Maximum lag for instruments
    #[schemars(
        description = "Maximum lag for instruments. If not specified, uses all available lags."
    )]
    pub max_lag: Option<usize>,

    /// Minimum lag for instruments (default: 2)
    #[schemars(description = "Minimum lag for instruments. Default: 2.")]
    pub min_lag: Option<usize>,

    /// Whether to collapse instruments
    #[schemars(
        description = "Whether to collapse instruments to reduce instrument count. Default: false."
    )]
    pub collapse: Option<bool>,

    /// Whether to use robust standard errors
    #[schemars(
        description = "Whether to use robust (Windmeijer-corrected) standard errors. Default: true."
    )]
    pub robust: Option<bool>,
}

/// Request for Panel GLS.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PanelGlsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity/individual identifier column
    #[schemars(
        description = "Column name for entity/individual identifier (e.g., 'firm_id', 'person_id')."
    )]
    pub entity_var: String,

    /// Time period identifier column
    #[schemars(description = "Column name for time period identifier (e.g., 'year', 'quarter').")]
    pub time_var: String,

    /// Model type: 'fe' (fixed effects), 'pooling', or 'fd' (first difference)
    #[schemars(
        description = "Model type: 'fe' for fixed effects GLS (default), 'pooling' for pooled GLS, 'fd' for first-difference GLS."
    )]
    pub model: Option<String>,
}

/// Request for Panel Unit Root tests.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PanelUnitRootRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Variable to test for unit root
    #[schemars(description = "Name of the variable column to test for unit root.")]
    pub variable: String,

    /// Unit/entity identifier column
    #[schemars(
        description = "Column name for panel unit identifier (e.g., 'country', 'firm_id')."
    )]
    pub unit_col: String,

    /// Time period column
    #[schemars(description = "Column name for time period identifier (e.g., 'year', 'quarter').")]
    pub time_col: String,

    /// Test type
    #[schemars(
        description = "Test type: 'llc' (Levin-Lin-Chu, default), 'ips' (Im-Pesaran-Shin), 'fisher' (Maddala-Wu), 'hadri' (stationarity null)."
    )]
    pub test: Option<String>,

    /// Model specification
    #[schemars(
        description = "Model specification: 'none' (no deterministic terms), 'constant' (individual intercepts, default), 'trend' (individual intercepts and trends)."
    )]
    pub model: Option<String>,

    /// Number of lags for ADF regression
    #[schemars(
        description = "Number of lags for ADF regression. If not specified, uses automatic selection."
    )]
    pub lags: Option<usize>,
}

/// Request for High-Dimensional Fixed Effects (HDFE).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PanelHdfeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Fixed effect columns to absorb
    #[schemars(
        description = "Column names for fixed effects to absorb (e.g., ['firm_id', 'year']). Supports multiple dimensions."
    )]
    pub fe: Vec<String>,

    /// Convergence tolerance for MAP algorithm
    #[schemars(
        description = "Convergence tolerance for the Method of Alternating Projections. Default is 1e-8."
    )]
    pub tolerance: Option<f64>,

    /// Maximum iterations for MAP algorithm
    #[schemars(description = "Maximum iterations for MAP algorithm. Default is 10000.")]
    pub max_iterations: Option<usize>,

    /// Standard error type
    #[schemars(
        description = "Standard error type: 'standard', 'hc0', 'hc1' (default), 'hc2', or 'hc3'."
    )]
    pub se_type: Option<String>,
}
