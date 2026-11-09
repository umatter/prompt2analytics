//! Request types for panel data analysis tools.
//!
//! This module contains request structs for:
//! - Fixed Effects (FE) regression
//! - Random Effects (RE) regression
//! - Hausman specification test
//! - Variable Coefficients Model (PVCM)
//! - Mean Group (PMG) estimator
//! - Arellano-Bond difference GMM
//! - Panel GLS
//! - Panel unit root tests (LLC, IPS, Hadri)
//! - High-dimensional fixed effects (HDFE)

use schemars::JsonSchema;
use serde::Deserialize;

/// Request for Panel Fixed Effects regression.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
    #[serde(alias = "unit_col", alias = "entity_col", alias = "entity")]
    pub entity_var: String,

    /// Standard-error type
    #[schemars(
        description = "Standard errors: 'classical' (default, as plm), 'cluster' (entity-clustered Arellano, Stata small-sample), 'cluster-hc0', 'cluster-hc1'."
    )]
    #[serde(alias = "se_type")]
    pub vcov: Option<String>,
}

/// Request for Panel Random Effects regression.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
    #[serde(alias = "unit_col", alias = "entity_col", alias = "entity")]
    pub entity_var: String,

    /// Standard-error type
    #[schemars(
        description = "Standard errors: 'classical' (default, as plm), 'cluster' (entity-clustered Arellano, Stata small-sample), 'cluster-hc0', 'cluster-hc1'."
    )]
    #[serde(alias = "se_type")]
    pub vcov: Option<String>,
}

/// Request for Hausman specification test.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
    #[serde(alias = "unit_col", alias = "entity_col", alias = "entity")]
    pub entity_var: String,
}

/// Request for Variable Coefficients Model (PVCM).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
    #[serde(alias = "unit_col", alias = "entity_col", alias = "entity")]
    pub entity_var: String,

    /// Model type: 'within' or 'random'
    #[schemars(
        description = "Model type: 'within' for separate OLS per entity (default; also 'individual', 'separate'), 'random' (also 'swamy') for the Swamy (1970) GLS estimator. panel_pmg accepts only 'within'. Other values are an error."
    )]
    pub model: Option<String>,
}

/// Request for Arellano-Bond difference GMM dynamic panel estimation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GmmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Strictly exogenous regressors (X) column names
    #[schemars(
        description = "Names of the strictly exogenous regressors. They enter in differences and serve as their own instruments. Do not include the lagged dependent variable: `lags` adds it."
    )]
    pub x: Vec<String>,

    /// Entity/individual identifier column
    #[schemars(
        description = "Column name for entity/individual identifier (e.g., 'firm_id', 'person_id')."
    )]
    #[serde(alias = "unit_col", alias = "entity_col", alias = "entity")]
    pub entity_var: String,

    /// Time period identifier column
    #[schemars(
        description = "Column name for the time period (e.g., 'year'): whole numbers; lags are steps of 1 (recode biennial years or dates as consecutive period numbers)."
    )]
    #[serde(alias = "time_col", alias = "time", alias = "period")]
    pub time_var: String,

    /// Number of lags of the dependent variable to include
    #[schemars(description = "Number of lags of Y to use as regressors. Default: 1.")]
    pub lags: Option<usize>,

    /// Transformation type: 'difference' (Arellano-Bond 1991); 'system' is not implemented yet
    #[schemars(
        description = "'difference' (Arellano-Bond; default) or 'system' (Blundell-Bond, as plm::pgmm(transformation = 'ld')); also 'd', 'ld', 'diff', 'sys'. Unknown values are an error."
    )]
    pub transform: Option<String>,

    /// Estimation step: 'onestep' or 'twostep'
    #[schemars(
        description = "'twostep' (default) or 'onestep'; one-step system fits need robust = true. Unknown values are an error."
    )]
    pub step: Option<String>,

    /// Maximum lag for instruments
    #[schemars(
        description = "Maximum lag for instruments. If not specified, uses all available lags."
    )]
    pub max_lag: Option<usize>,

    /// Minimum lag for instruments (default: 2)
    #[schemars(
        description = "Minimum lag of y used as instrument. Default 2; lag 1 is not a valid instrument under serially uncorrelated errors."
    )]
    pub min_lag: Option<usize>,

    /// Whether to collapse instruments
    #[schemars(
        description = "Whether to collapse instruments to reduce instrument count. Default: false."
    )]
    pub collapse: Option<bool>,

    /// Whether to use robust standard errors
    #[schemars(
        description = "Default true: one-step robust or two-step Windmeijer (2005)-corrected SEs; false: one-step Arellano-Bond (1991) classical or uncorrected two-step SEs."
    )]
    pub robust: Option<bool>,
}

/// Request for Panel GLS.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
    #[serde(alias = "unit_col", alias = "entity_col", alias = "entity")]
    pub entity_var: String,

    /// Time period identifier column
    #[schemars(description = "Column name for time period identifier (e.g., 'year', 'quarter').")]
    #[serde(alias = "time_col", alias = "time", alias = "period")]
    pub time_var: String,

    /// Model type: 'fe' (fixed effects), 'pooling', or 'fd' (first difference)
    #[schemars(
        description = "Model type: 'fe' for fixed effects GLS (default; also 'within', 'fixed_effects'), 'pooling' for pooled GLS (also 'pool', 'pooled'), 'fd' for first-difference GLS (also 'first_difference'). Other values are an error."
    )]
    pub model: Option<String>,
}

/// Request for Panel Unit Root tests.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
    #[serde(alias = "entity_var", alias = "entity_col", alias = "entity")]
    pub unit_col: String,

    /// Time period column
    #[schemars(description = "Column name for time period identifier (e.g., 'year', 'quarter').")]
    #[serde(alias = "time_var", alias = "time", alias = "period")]
    pub time_col: String,

    /// Test type
    #[schemars(
        description = "Test type: 'llc' (Levin-Lin-Chu, default; also 'levinlin'), 'ips' (Im-Pesaran-Shin), 'madwu' (Maddala-Wu; also 'fisher'), 'hadri' (stationarity null). Other values are an error."
    )]
    pub test: Option<String>,

    /// Model specification
    #[schemars(
        description = "Model specification: 'none' (no deterministic terms), 'intercept' (individual intercepts, default; also 'drift'), 'trend' (individual intercepts and trends; also 'ct'). Other values are an error."
    )]
    pub model: Option<String>,

    /// Number of lags for ADF regression
    #[schemars(
        description = "Number of lags of every unit's ADF regression. If neither lags nor lag_selection is given, floor(T_max^(1/3)) lags for every unit. Not with lag_selection = aic | sic | hall (use pmax)."
    )]
    pub lags: Option<usize>,

    /// Per-unit lag selection
    #[schemars(
        description = "Per-unit lag selection as plm::purtest(lags =): 'aic', 'sic' (also 'bic') or 'hall' choose each unit's order up to pmax on a common sample; 'fixed' uses lags. Without it, lags (if given) or floor(T_max^(1/3)) lags for every unit. Not for 'hadri'."
    )]
    pub lag_selection: Option<String>,

    /// Largest lag order considered by lag selection
    #[schemars(
        description = "Largest lag order considered by lag_selection = aic | sic | hall (plm's pmax). Default 10. At least 1."
    )]
    pub pmax: Option<usize>,
}

/// Request for High-Dimensional Fixed Effects (HDFE).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
        description = "Convergence tolerance of the Method of Alternating Projections, relative: ||delta|| / ||x|| (fixest). Default is 1e-8."
    )]
    pub tolerance: Option<f64>,

    /// Maximum iterations for MAP algorithm
    #[schemars(description = "Maximum iterations for MAP algorithm. Default is 10000.")]
    pub max_iterations: Option<usize>,

    /// Standard error type
    #[schemars(
        description = "Standard error type: 'standard' (also 'classical'), 'hc0', 'hc1' (default), 'hc2' or 'hc3'. Other values are an error."
    )]
    pub se_type: Option<String>,
}
