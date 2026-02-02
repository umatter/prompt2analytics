//! Request types for discrete choice model tools.
//!
//! This module contains request structs for:
//! - Binary choice (Logit, Probit)
//! - Multinomial logit (unordered categorical)
//! - McFadden's conditional logit (mlogit)
//! - Mixed logit (random parameters)
//! - Ordered logit/probit
//! - Count models (Negative binomial)
//! - Zero-inflated models (ZIP, ZINB)
//! - Hurdle models
//! - FEGLM (GLM with HDFE)

use schemars::JsonSchema;
use serde::Deserialize;

/// Request for Logit regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LogitRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name (must be binary 0/1)
    #[schemars(description = "Name of the dependent variable (Y) column. Must be binary (0/1).")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,
}

/// Request for Probit regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ProbitRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name (must be binary 0/1)
    #[schemars(description = "Name of the dependent variable (Y) column. Must be binary (0/1).")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,
}

/// Request for multinomial logit regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MultinomRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name (categorical outcome)
    #[schemars(description = "Name of the categorical dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Reference category (optional)
    #[schemars(
        description = "Reference category for the model. If not specified, the first category (alphabetically) is used."
    )]
    pub reference: Option<String>,
}

/// Request for McFadden's conditional logit (mlogit) model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MlogitRequest {
    /// Name/ID of the dataset (in long format)
    #[schemars(
        description = "Name or ID of a dataset in long format (one row per individual-alternative combination)."
    )]
    pub dataset: String,

    /// Column identifying choice situations (individuals)
    #[schemars(
        description = "Column name identifying each choice situation (individual chooser)."
    )]
    pub choice_id: String,

    /// Column identifying alternatives
    #[schemars(
        description = "Column name identifying alternatives (e.g., 'car', 'bus', 'train')."
    )]
    pub alt_id: String,

    /// Column with binary choice indicator (1 = chosen)
    #[schemars(
        description = "Column with binary choice indicator (1 if alternative is chosen, 0 otherwise)."
    )]
    pub choice: String,

    /// Alternative-specific variables (generic coefficients)
    #[schemars(
        description = "Alternative-specific variables that vary across alternatives (e.g., 'price', 'time'). These get generic coefficients (same B across all alternatives)."
    )]
    pub alt_specific: Vec<String>,

    /// Individual-specific variables (alternative-specific coefficients)
    #[schemars(
        description = "Individual-specific variables that are constant across alternatives (e.g., 'income', 'age'). These get alternative-specific coefficients (different gamma_j for each alternative vs reference)."
    )]
    #[serde(default)]
    pub ind_specific: Vec<String>,

    /// Reference alternative (optional)
    #[schemars(
        description = "Reference alternative for identification. Default: first alternative (alphabetically)."
    )]
    pub reference: Option<String>,
}

/// Request for mixed logit (random parameters logit) estimation.
/// Covers both gmnl and mixl R packages.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MixedLogitRequest {
    /// Name/ID of the dataset (in long format)
    #[schemars(
        description = "Name or ID of a dataset in long format (one row per individual-alternative combination)."
    )]
    pub dataset: String,

    /// Column identifying choice situations (individuals)
    #[schemars(
        description = "Column name identifying each choice situation (individual chooser)."
    )]
    pub choice_id: String,

    /// Column identifying alternatives
    #[schemars(
        description = "Column name identifying alternatives (e.g., 'car', 'bus', 'train')."
    )]
    pub alt_id: String,

    /// Column with binary choice indicator (1 = chosen)
    #[schemars(
        description = "Column with binary choice indicator (1 if alternative is chosen, 0 otherwise)."
    )]
    pub choice: String,

    /// Variables to include in the model
    #[schemars(description = "Variable columns to include in the choice model.")]
    pub variables: Vec<String>,

    /// Variables with random coefficients
    #[schemars(
        description = "Variable names that should have random (mixed) coefficients. If not specified, all variables are random."
    )]
    pub random_vars: Option<Vec<String>>,

    /// Distribution for random parameters
    #[schemars(
        description = "Distribution for random parameters: 'normal' (default), 'lognormal', 'triangular', 'uniform'."
    )]
    pub distribution: Option<String>,

    /// Number of simulation draws
    #[schemars(
        description = "Number of simulation draws for MSL estimation. Default: 500. Higher values improve accuracy but increase computation time."
    )]
    pub n_draws: Option<usize>,

    /// Use Halton sequences (quasi-random)
    #[schemars(
        description = "Use Halton quasi-random sequences instead of pseudo-random draws. Default: true. Improves accuracy."
    )]
    pub halton: Option<bool>,
}

/// Request for ordered logit/probit regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct OrderedRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name (ordered categorical outcome)
    #[schemars(description = "Name of the ordered categorical dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Model type: "logit" (default) or "probit"
    #[schemars(description = "Model type: 'logit' (default) or 'probit'.")]
    pub model_type: Option<String>,
}

/// Request for negative binomial regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct NegBinRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name (count data)
    #[schemars(description = "Name of the count dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Initial theta (dispersion) parameter
    #[schemars(
        description = "Optional initial theta (dispersion) parameter. If not specified, estimated from data."
    )]
    pub init_theta: Option<f64>,
}

/// Request for zero-inflated models.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ZeroInflRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name (count data with excess zeros)
    #[schemars(description = "Name of the count dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names for count model
    #[schemars(description = "Names of the independent variable (X) columns for the count model.")]
    pub x: Vec<String>,

    /// Independent variables (Z) column names for zero-inflation model
    #[schemars(
        description = "Names of the variables for the zero-inflation model. If not specified, uses intercept only."
    )]
    pub z: Option<Vec<String>>,

    /// Model type: "poisson" (default) or "negbin"
    #[schemars(description = "Distribution for count model: 'poisson' (default) or 'negbin'.")]
    pub dist: Option<String>,
}

/// Request for hurdle model (two-part model for count data with zero inflation).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HurdleModelRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name (count data with zeros)
    #[schemars(description = "Name of the count dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names for count model
    #[schemars(description = "Names of the independent variable (X) columns for the count model.")]
    pub x: Vec<String>,

    /// Independent variables (Z) column names for binary (hurdle) model
    #[schemars(
        description = "Names of the variables for the binary hurdle model. If not specified, uses same as x."
    )]
    pub z: Option<Vec<String>>,

    /// Model type: "poisson" (default) or "negbin"
    #[schemars(description = "Distribution for count model: 'poisson' (default) or 'negbin'.")]
    pub dist: Option<String>,
}

/// Request for Generalized Linear Model with High-Dimensional Fixed Effects (FEGLM).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FeglmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(
        description = "Name of the dependent variable (Y) column. For logit/probit must be binary (0/1). For Poisson must be non-negative counts."
    )]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Fixed effect columns to absorb
    #[schemars(
        description = "Column names for fixed effects to absorb (e.g., ['firm_id', 'year']). Supports multiple dimensions."
    )]
    pub fe: Vec<String>,

    /// GLM family
    #[schemars(
        description = "GLM family: 'logit' (binomial logit, default), 'probit' (binomial probit), 'poisson' (count data), or 'gaussian' (continuous, equivalent to linear HDFE)."
    )]
    pub family: Option<String>,

    /// Maximum IRLS iterations
    #[schemars(description = "Maximum IRLS iterations for estimation. Default is 25.")]
    pub max_iter: Option<usize>,

    /// Convergence tolerance
    #[schemars(description = "Convergence tolerance for coefficient changes. Default is 1e-8.")]
    pub tolerance: Option<f64>,
}
