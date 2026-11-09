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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
        description = "Reference category for the model. If not specified, the first category is used, in R factor() order (numbers numerically, text by byte: uppercase before lowercase)."
    )]
    pub reference: Option<String>,
}

/// Request for McFadden's conditional logit (mlogit) model.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
        description = "Reference alternative for identification; same as reflevel. Default: the first alternative in R factor() order (numbers numerically, text by byte: uppercase before lowercase)."
    )]
    pub reference: Option<String>,

    /// Alternative-specific constants
    #[schemars(
        description = "Alternative-specific constants (J - 1, as mlogit's default). false fits mlogit(choice ~ alt_specific | 0 + ind_specific), the survival::clogit model. Default true."
    )]
    pub asc: Option<bool>,

    /// Reference alternative (mlogit's reflevel)
    #[schemars(
        description = "Reference alternative (mlogit's reflevel); same as reference. Default: the first alternative in sorted order."
    )]
    pub reflevel: Option<String>,
}

/// Request for mixed logit (random parameters logit) estimation.
/// Covers both gmnl and mixl R packages.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MixedLogitRequest {
    /// Name/ID of the dataset (in long format)
    #[schemars(
        description = "Name or ID of a dataset in long format (one row per individual-alternative combination)."
    )]
    pub dataset: String,

    /// Column identifying choice situations
    #[schemars(description = "Column identifying each choice situation.")]
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
        description = "Distribution for random parameters: 'normal' (default), 'lognormal', 'triangular', 'uniform' or 'fixed'. Other values are an error."
    )]
    pub distribution: Option<String>,

    /// Number of simulation draws
    #[schemars(
        description = "Number of simulation draws for MSL estimation. Default: 500. Higher values improve accuracy but increase computation time."
    )]
    pub n_draws: Option<usize>,

    /// Use Halton sequences (quasi-random)
    #[schemars(
        description = "Use Halton quasi-random sequences instead of pseudo-random draws. Default: true. Pseudo-random draws (false) are seeded by the session seed (set_seed), else 42."
    )]
    pub halton: Option<bool>,

    /// Decision-maker column for repeated choices (mlogit panel = TRUE)
    #[schemars(
        description = "Column identifying the decision maker when each person makes several choices (panel data). Draws are then taken per person and the probabilities of that person's choices are multiplied (mlogit panel = TRUE). Default: every choice situation is a separate decision maker."
    )]
    #[serde(default)]
    pub individual_id: Option<String>,
}

/// Request for ordered logit/probit regression.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
    #[schemars(
        description = "Model type: 'logit' (default; also 'logistic', 'polr') or 'probit'. Other values are an error."
    )]
    pub model_type: Option<String>,

    /// Category order of a text outcome, lowest first
    #[schemars(
        description = "Ordinal order of a text outcome, lowest category first, e.g. [\"low\", \"mid\", \"high\"]. Required when the outcome column is text; every observed category must appear exactly once. Numeric outcomes are ordered numerically and need no levels."
    )]
    #[serde(default)]
    pub levels: Option<Vec<String>>,
}

/// Request for negative binomial regression.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
        description = "Optional starting value of theta (dispersion), as glm.nb's init.theta; between 1e-8 and 1e6. The estimate is the same maximum likelihood fit; omit it to start from the moment estimate."
    )]
    pub init_theta: Option<f64>,
}

/// Request for zero-inflated models.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
    #[schemars(
        description = "Distribution for count model: 'poisson' (default) or 'negbin' (also 'negbinom', 'negative_binomial', 'nb'). Other values are an error."
    )]
    pub dist: Option<String>,
}

/// Request for hurdle model (two-part model for count data with zero inflation).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
    #[schemars(
        description = "Distribution for count model: 'poisson' (default) or 'negbin' (also 'negbinom', 'negative_binomial', 'nb'). Other values are an error."
    )]
    pub dist: Option<String>,
}

/// Request for Generalized Linear Model with High-Dimensional Fixed Effects (FEGLM).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
        description = "GLM family: 'logit' (binomial logit, default; also 'binomial', 'logistic'), 'probit' (binomial probit), 'poisson' (count data) or 'gaussian' (also 'normal', 'linear'; continuous, equivalent to linear HDFE). Case is ignored; other values are an error."
    )]
    pub family: Option<String>,

    /// Maximum IRLS iterations
    #[schemars(description = "Maximum IRLS iterations for estimation. Default is 25.")]
    pub max_iter: Option<usize>,

    /// Convergence tolerance
    #[schemars(
        description = "Convergence tolerance: the relative deviance change |dev - dev_old| / (|dev| + 0.1), as fixest. Default is 1e-8."
    )]
    pub tolerance: Option<f64>,
}
