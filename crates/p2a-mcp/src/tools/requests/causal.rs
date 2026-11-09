//! Request types for causal inference tools.
//!
//! This module contains request structs for:
//! - IV methods (GMM-IV, 2SLS, first-stage, Sargan, MTE, Balke-Pearl bounds)
//! - Difference-in-Differences (DiD, staggered DiD, Bacon decomposition, ETWFE)
//! - Treatment effects (IPW, AIPW, DoubleML, CBPS, WeightIt, entropy balance, SBW, TWANG)
//! - Propensity score matching (MatchIt)
//! - TMLE family (TMLE, C-TMLE, LTMLE)
//! - Standardization and G-formula
//! - Mediation analysis
//! - Synthetic control (classic, gsynth, SCPI)
//! - Regression discontinuity (sharp, fuzzy, multi-cutoff)
//! - Causal ML (causal forests, BART)

use schemars::JsonSchema;
use serde::Deserialize;

/// Predictor specification for synthetic control.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SynthPredictorSpec {
    /// Column name of the predictor variable
    #[schemars(description = "Column name of the predictor variable.")]
    pub column: String,

    /// How to aggregate the predictor over time
    #[schemars(
        description = "How the predictor is aggregated over its window: 'mean' (default; 'average'), 'first', 'last' or 'sum' ('total'). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub aggregation: Option<String>,

    /// Optional time window (start, end) for aggregation
    #[schemars(
        description = "Time window for predictor aggregation as [start, end]. If omitted, uses all pre-treatment periods."
    )]
    pub time_window: Option<(i64, i64)>,
}

/// Request for general GMM IV estimation (Hansen 1982).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GeneralGmmIvRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent (outcome) variable column.")]
    pub y: String,

    /// Endogenous/exogenous regressors (X) column names
    #[schemars(description = "Names of the regressor columns (both endogenous and exogenous).")]
    pub x: Vec<String>,

    /// Instruments (Z) column names
    #[schemars(
        description = "Names of the instrument columns. Should include exogenous regressors plus additional instruments. Must have at least as many instruments as regressors."
    )]
    pub z: Vec<String>,

    /// GMM estimation method
    #[schemars(
        description = "Estimation method, as gmm::gmm type: 'twostep' (default), 'iterative', 'cue' (continuously updated) or 'onestep' (with a fixed weighting matrix). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub method: Option<String>,

    /// Variance-covariance type
    #[schemars(
        description = "Moment covariance: 'hac' (default: Bartlett weights with floor(4 (n/100)^(2/9)) lags and no prewhitening, as gmm(kernel = 'Bartlett', bw = m + 1, prewhite = 0); gmm's own default (Quadratic Spectral kernel, Andrews bandwidth, prewhite = 1) is not implemented), 'iid' (two-step is 2SLS), 'fixed' (one step with the identity weighting matrix, or weight_matrix) or 'mds' (heteroskedasticity-robust; 'robust'). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub vcov: Option<String>,

    /// HAC kernel type
    #[schemars(
        description = "HAC kernel: 'bartlett' (default; 'newey_west'), 'parzen', 'quadratic_spectral' ('qs'), 'truncated' or 'tukey_hanning'. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub kernel: Option<String>,

    /// HAC bandwidth
    #[schemars(description = "Bandwidth for HAC estimation. None for automatic selection.")]
    pub bandwidth: Option<usize>,

    /// Fixed weighting matrix
    #[schemars(
        description = "Fixed weighting matrix W (one-step GMM with the sandwich variance, as gmm::gmm(weightsMatrix = W)), row-major q x q with q = number of z columns + 1 (the constant first, then z in the listed order); symmetric positive definite. No J test is reported for a fixed W."
    )]
    pub weight_matrix: Option<Vec<f64>>,
}

/// Request for IV/2SLS regression.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IV2SLSRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Exogenous independent variables
    #[schemars(
        description = "Names of exogenous independent variable columns (not instrumented)."
    )]
    pub x_exog: Vec<String>,

    /// Endogenous variable to be instrumented
    #[schemars(description = "Names of endogenous variables that need instruments.")]
    pub x_endog: Vec<String>,

    /// Instrumental variables
    #[schemars(description = "Names of instrument columns (excluded from structural equation).")]
    pub instruments: Vec<String>,

    /// Use robust standard errors
    #[schemars(
        description = "Whether to use heteroskedasticity-robust standard errors. Default is true."
    )]
    pub robust: Option<bool>,
}

/// Request for first-stage diagnostics.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FirstStageRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Endogenous variable name
    #[schemars(description = "Name of the endogenous variable to test instrument strength for.")]
    pub endogenous_var: String,

    /// Instrument variable names
    #[schemars(
        description = "Names of the instrumental variables (e.g., ['parents_edu', 'distance_to_college'])."
    )]
    pub instruments: Vec<String>,

    /// Control variable names (optional)
    #[schemars(description = "Optional control variables to include in first-stage regression.")]
    pub controls: Option<Vec<String>>,
}

/// Request for Sargan test of overidentifying restrictions.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SarganTestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Exogenous independent variables
    #[schemars(
        description = "Names of exogenous independent variable columns (not instrumented). May be empty."
    )]
    pub x_exog: Vec<String>,

    /// Endogenous variable to be instrumented
    #[schemars(description = "Names of endogenous variables that need instruments.")]
    pub x_endog: Vec<String>,

    /// Instrumental variables
    #[schemars(
        description = "Names of instrument columns. Must exceed number of endogenous variables for test to be valid."
    )]
    pub instruments: Vec<String>,
}

/// Request for Balke-Pearl bounds on the Average Causal Effect (ACE).
///
/// Balke-Pearl bounds provide sharp nonparametric bounds on the causal effect
/// using instrumental variables without assuming parametric models.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BPBoundsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Instrument column (binary 0/1)
    #[schemars(
        description = "Name of the binary instrument column (Z). Example: randomized treatment assignment."
    )]
    pub instrument: String,

    /// Treatment column (binary 0/1)
    #[schemars(
        description = "Name of the binary treatment received column (D). Example: actual treatment uptake."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Outcome column (binary 0/1)
    #[schemars(description = "Name of the binary outcome column (Y). Example: recovery status.")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Assume monotonicity (no defiers)
    #[schemars(
        description = "Whether to assume monotonicity (no defiers). If true, bounds tighten but assumption may be violated. Default is false."
    )]
    pub monotonicity: Option<bool>,

    /// Compute bootstrap confidence intervals
    #[schemars(
        description = "Whether to compute bootstrap confidence intervals for the bounds. Default is true."
    )]
    pub bootstrap_ci: Option<bool>,

    /// Number of bootstrap replications
    #[schemars(
        description = "Number of bootstrap replications for confidence intervals. Default is 1000."
    )]
    pub n_bootstrap: Option<usize>,

    /// Confidence level (1 - alpha)
    #[schemars(description = "Confidence level for intervals. Default is 0.95 (95% CI).")]
    pub confidence_level: Option<f64>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for bootstrap reproducibility.")]
    pub seed: Option<u64>,
}

/// Request for Marginal Treatment Effects (MTE) estimation.
///
/// The MTE framework (Heckman & Vytlacil 2005) connects IV estimation to a
/// choice-theoretic model of treatment selection, revealing heterogeneity
/// in treatment effects.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IVMTERequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable (Y) column name
    #[schemars(description = "Name of the outcome variable (Y) column.")]
    pub y: String,

    /// Treatment indicator column (binary 0/1)
    #[schemars(description = "Name of the binary treatment indicator column (D = 0 or 1).")]
    pub d: String,

    /// Instrument column name
    #[schemars(
        description = "Name of the instrumental variable (Z) column that affects treatment but not outcome directly."
    )]
    pub z: String,

    /// Covariate columns (optional)
    #[schemars(
        description = "Covariate columns. They enter the propensity model (with z) and the second stage (Y on 1, P, ..., P^degree and X), as local IV with controls (Brinch, Mogstad & Wiswall 2017). Must not repeat y, d or z."
    )]
    pub x: Option<Vec<String>>,

    /// Polynomial degree for MTE curve
    #[schemars(
        description = "Polynomial degree for MTE approximation. Higher degrees allow more flexible MTE shapes but may overfit. Default is 2."
    )]
    pub mte_degree: Option<usize>,

    /// Propensity score model type
    #[schemars(
        description = "Propensity score model: 'probit' (default, standard in the MTE literature), 'logit' or 'linear' (linear probability model). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub propensity_model: Option<String>,

    /// Number of grid points for MTE curve
    #[schemars(
        description = "Number of grid points of the MTE curve and the estimand weights (default 100, at least 2)."
    )]
    pub n_grid: Option<usize>,

    /// Target estimand
    #[schemars(
        description = "Target estimand reported as `target`: 'mte_curve' (default: the curve and every estimand), 'late', 'att', 'atu', 'ate' or 'prte' (needs prte_weights). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub estimand: Option<String>,

    /// PRTE weights
    #[schemars(
        description = "PRTE weights, one per grid point (length n_grid), normalised by the core."
    )]
    pub prte_weights: Option<Vec<f64>>,

    /// Bootstrap replicates
    #[schemars(
        description = "Bootstrap replicates for the SEs; units are resampled and both stages refitted, seeded by the session seed. Default 0: delta method (HC1 covariance of the MTE polynomial; the first-stage estimation error is not included; bootstrap_reps > 0 includes it)."
    )]
    pub bootstrap_reps: Option<usize>,
}

/// Request for Difference-in-Differences estimation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DiDRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable column name
    #[schemars(description = "Name of the outcome/dependent variable column.")]
    #[serde(alias = "outcome", alias = "y", alias = "response")]
    pub dep_var: String,

    /// Treatment group indicator column (0/1)
    #[schemars(description = "Column indicating treatment group (1 = treated, 0 = control).")]
    #[serde(alias = "treatment", alias = "treat")]
    pub treatment_var: String,

    /// Post-treatment period indicator column (0/1)
    #[schemars(description = "Column indicating post-treatment period (1 = post, 0 = pre).")]
    #[serde(alias = "post", alias = "post_treatment")]
    pub post_var: String,

    /// Optional: cluster variable for clustered standard errors (e.g., unit or entity ID)
    #[schemars(
        description = "Column to cluster standard errors on (e.g., entity/unit ID). If omitted, HC1 robust SEs are used."
    )]
    #[serde(default, alias = "cluster", alias = "cluster_id")]
    pub cluster_var: Option<String>,
}

/// Request for Callaway-Sant'Anna staggered DiD estimation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StaggeredDiDRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Treatment timing column
    #[schemars(
        description = "Column indicating when each unit was first treated (period number). Use 0 or negative for never-treated units."
    )]
    pub treatment_time: String,

    /// Time period column
    #[schemars(
        description = "Column containing the time period identifier (e.g., year, quarter)."
    )]
    #[serde(alias = "time_var", alias = "time", alias = "period")]
    pub time_col: String,

    /// Unit identifier column
    #[schemars(
        description = "Column containing the unit/individual identifier (e.g., state_id, firm_id)."
    )]
    #[serde(alias = "entity_var", alias = "entity_col", alias = "entity")]
    pub unit_col: String,

    /// Covariate columns (optional)
    #[schemars(description = "Optional covariates for conditional parallel trends assumption.")]
    pub covariates: Option<Vec<String>>,

    /// Comparison group strategy
    #[schemars(
        description = "Comparison group, as did::att_gt `control_group`: 'never_treated' (default; did 'nevertreated') or 'not_yet_treated' (did 'notyettreated'; 'notyet' and 'nyt' are accepted). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub comparison_group: Option<String>,

    /// Estimation method
    #[schemars(
        description = "Estimation method, as did::att_gt `est_method`: 'outcome_regression' (default; did 'reg'), 'ipw' or 'doubly_robust' (did 'dr'; 'aipw' and 'drdid' are accepted). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub estimation_method: Option<String>,

    /// Base period relative to treatment
    #[schemars(
        description = "Base period for pre-treatment comparison, relative to g. Default is -1 (one period before treatment)."
    )]
    pub base_period: Option<i32>,

    /// Number of bootstrap replications
    #[schemars(
        description = "0 = analytic influence-function standard errors (did's bstrap = FALSE), otherwise at least 50 bootstrap replicates (unit bootstrap; with covariates a multiplier bootstrap of the influence functions; 1-49 are refused). Default 999. With covariates each replicate perturbs the cells' influence functions with the resampled unit counts (did's multiplier bootstrap); without covariates units are resampled. Seeded by the session seed."
    )]
    pub bootstrap: Option<usize>,

    /// Periods of anticipation
    #[schemars(
        description = "Number of periods before treatment in which units may anticipate it, as did::att_gt `anticipation` (default 0): the base period becomes g - 1 - anticipation, not-yet-treated comparison units must be first treated after max(t, base) + anticipation, and the anticipation cells are reported (and pre-tested) as pre-treatment cells. Must be smaller than the number of periods."
    )]
    pub anticipation: Option<usize>,
}

/// Request for Goodman-Bacon decomposition of staggered DiD.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BaconDecompRequest {
    /// Name/ID of the dataset
    #[schemars(
        description = "Name or ID of a previously loaded panel dataset with staggered treatment."
    )]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Unit identifier column
    #[schemars(
        description = "Column containing the unit/individual identifier (e.g., state_id, firm_id)."
    )]
    #[serde(alias = "entity_var", alias = "entity_col", alias = "entity")]
    pub unit_col: String,

    /// Time period column
    #[schemars(
        description = "Column containing the time period identifier (e.g., year, quarter)."
    )]
    #[serde(alias = "time_var", alias = "time", alias = "period")]
    pub time_col: String,

    /// Treatment indicator column
    #[schemars(
        description = "Binary treatment indicator column (0 = untreated, 1 = treated). Should be 0 before treatment and 1 after for each unit."
    )]
    pub treatment_col: String,
}

/// Request for Extended Two-Way Fixed Effects (ETWFE) estimation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EtwfeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Unit identifier column
    #[schemars(
        description = "Column containing the unit/individual identifier (e.g., state_id, firm_id)."
    )]
    #[serde(alias = "entity_var", alias = "entity_col", alias = "entity")]
    pub unit_col: String,

    /// Time period column
    #[schemars(
        description = "Column containing the time period identifier (e.g., year, quarter)."
    )]
    #[serde(alias = "time_var", alias = "time", alias = "period")]
    pub time_col: String,

    /// Treatment indicator column
    #[schemars(
        description = "Column indicating treatment status (1 = currently treated, 0 = not treated). Binary indicator."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// First treatment period column
    #[schemars(
        description = "Column indicating when each unit was first treated (period number). Use 0 for never-treated units."
    )]
    pub first_treat: String,

    /// Control variables (optional)
    #[schemars(description = "Optional control variable columns.")]
    pub controls: Option<Vec<String>>,

    /// Control group strategy
    #[schemars(
        description = "Control group, as etwfe `cgroup`: 'notyet' (default; 'not_yet_treated') or 'never' ('never_treated'). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub cgroup: Option<String>,

    /// Standard errors
    #[schemars(
        description = "Standard errors: 'cluster' (default; unit-clustered CR1 with fixest's small-sample factor; 'cr1') or 'iid' ('classical'). etwfe and fixest default to iid SEs, which vcov = 'iid' gives. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub vcov: Option<String>,

    /// One emfx table instead of the whole result
    #[schemars(
        description = "Report one aggregation, as etwfe::emfx(type): 'simple' (the overall ATT), 'event' (by periods since treatment; 'dynamic'), 'group' (by cohort) or 'calendar' (by period). Default: the whole result with every table. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub aggregation: Option<String>,

    /// Reference period
    #[schemars(
        description = "Reference period (etwfe `tref`): a period of time_col; default the first period."
    )]
    pub tref: Option<i64>,

    /// Reference cohort
    #[schemars(
        description = "Reference cohort (etwfe `gref`): a value of first_treat; default the never-treated units, or the last cohort with cgroup 'notyet' when none exist."
    )]
    pub gref: Option<i64>,
}

/// Request for Inverse Probability Weighting (IPW) treatment effect estimation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IpwRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Covariate columns for propensity score model
    #[schemars(description = "Names of covariate columns to include in propensity score model.")]
    pub covariates: Vec<String>,

    /// Estimand: 'ate' (Average Treatment Effect) or 'att' (Average Treatment Effect on Treated)
    #[schemars(
        description = "Treatment effect estimand: 'ate' (default) or 'att'. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub estimand: Option<String>,

    /// Trimming threshold for extreme propensity scores
    #[schemars(
        description = "Trim observations with propensity scores below trim or above 1-trim. Default is 0.05."
    )]
    pub trim: Option<f64>,

    /// Number of bootstrap replications for standard errors
    #[schemars(
        description = "Standard errors: 0 (default) gives the M-estimation SE with the propensity score estimated; the efficient-influence-function value is also reported (se_eif). bootstrap > 0 gives the SD of that many replicates that hold the fitted propensity scores fixed (the propensity is treated as known), seeded by the session seed."
    )]
    pub bootstrap: Option<usize>,
}

/// Request for Doubly Robust (AIPW) treatment effect estimation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DoublyRobustRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Covariate columns for propensity score and outcome models
    #[schemars(
        description = "Names of covariate columns to include in both propensity score and outcome models."
    )]
    pub covariates: Vec<String>,

    /// Estimation method: 'aipw' (default), 'ipw', or 'regression'
    #[schemars(
        description = "Estimation method: 'aipw' (default, augmented IPW, doubly robust; 'dr'), 'ipw' (weighting only) or 'regression' (outcome regression only). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub method: Option<String>,

    /// Estimand: 'ate' (Average Treatment Effect) or 'att' (Average Treatment Effect on Treated)
    #[schemars(
        description = "Treatment effect estimand: 'ate' (default) or 'att'. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub estimand: Option<String>,

    /// Trimming threshold for extreme propensity scores
    #[schemars(
        description = "Trim observations with propensity scores below trim or above 1-trim. Default is 0.05."
    )]
    pub trim: Option<f64>,

    /// Number of bootstrap replications for standard errors
    #[schemars(
        description = "Bootstrap replicates for 'ipw' and 'regression' (default 999; seeded by the session seed): the replicates reuse the fitted propensity scores (ipw) or outcome predictions (regression), so the SE ignores their estimation. AIPW reports the analytic influence-function SE and runs no replicates."
    )]
    pub bootstrap: Option<usize>,
}

/// Request for Double/Debiased Machine Learning (DoubleML) estimation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DoubleMLRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column (Y).")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Treatment variable column name
    #[schemars(
        description = "Name of the treatment variable column (D). Can be continuous or binary."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Covariate columns for nuisance model estimation
    #[schemars(description = "Names of covariate columns (X) for nuisance model estimation.")]
    pub covariates: Vec<String>,

    /// Model type: 'plr' (Partially Linear Regression, default) or 'irm' (Interactive Regression Model)
    #[schemars(
        description = "DML model: 'plr' (default, partially linear: Y = theta*D + g(X) + eps) or 'irm' (interactive, binary treatment, heterogeneous effects). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub model_type: Option<String>,

    /// Number of cross-fitting folds (default: 5)
    #[schemars(
        description = "Number of folds for cross-fitting. Default is 5. Must be at least 2."
    )]
    pub n_folds: Option<usize>,

    /// Random seed for reproducible fold splits
    #[schemars(
        description = "Random seed for reproducible cross-fitting splits. If omitted, uses random seed."
    )]
    pub seed: Option<u64>,

    /// Trimming threshold for propensity scores (IRM only)
    #[schemars(
        description = "Trim propensity scores to [trim, 1-trim] for IRM model. Default is 0.01."
    )]
    pub trim: Option<f64>,
}

/// Request for Covariate Balancing Propensity Score (CBPS) estimation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CbpsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Covariate columns for propensity score model
    #[schemars(description = "Names of covariate columns to include in propensity score model.")]
    pub covariates: Vec<String>,

    /// CBPS method: 'over' (default), 'exact' or 'logit'
    #[schemars(
        description = "CBPS method: 'over' (default, R's default: the score and balance conditions, over-identified two-step GMM), 'exact' (the balance conditions only, just identified) or 'logit' (logistic regression, no balance conditions). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub method: Option<String>,

    /// Standardized difference threshold for balance
    #[schemars(
        description = "Threshold for standardized difference to consider a covariate balanced. Default is 0.1."
    )]
    pub balance_threshold: Option<f64>,

    /// Estimand
    #[schemars(
        description = "Estimand of the weights: 'ATE' (default) or 'ATT'. CBPS::CBPS defaults to ATT = 1, WeightIt's CBPS to ATE. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub estimand: Option<String>,
}

/// Request for flexible inverse probability weighting (WeightIt).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WeightItRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Covariate columns for balance
    #[schemars(description = "Names of covariate columns to balance between treatment groups.")]
    pub covariates: Vec<String>,

    /// Weighting method: 'logistic' (default), 'entropy', 'energy', or 'stable'
    #[schemars(
        description = "Weighting method: 'logistic' (glm propensity, default; 'glm', 'ps'), 'entropy' (entropy balancing, exact dual; 'ebal'), 'energy' (Huling & Mak energy-distance QP, exact; WeightIt energy at lambda = 0; at most 2000 reweighted units) or 'stable' (stable balancing weights QP with exact balance, as WeightIt optweight; 'sbw', 'optweight'). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub method: Option<String>,

    /// Target estimand: 'ate' (default), 'att', or 'atc'
    #[schemars(
        description = "Estimand: 'ate' (default), 'att' or 'atc'. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub estimand: Option<String>,

    /// Whether to stabilize weights
    #[schemars(
        description = "Multiply the weights by the marginal treatment probability (default false): logistic only; other methods warn and leave the weights unchanged."
    )]
    pub stabilize: Option<bool>,

    /// Trimming quantile for extreme weights
    #[schemars(
        description = "Quantile at which extreme weights are trimmed (e.g. 0.99 trims at the 1st and 99th percentiles). Default 1.0 (no trimming). Must be in (0.5, 1]."
    )]
    pub trim_quantile: Option<f64>,
}

/// Request for entropy balancing.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EntropyBalanceRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Covariate columns for balance
    #[schemars(description = "Names of covariate columns to balance exactly on means.")]
    pub covariates: Vec<String>,

    /// Optional target means (defaults to treated group means for ATT)
    #[schemars(
        description = "Target means of the covariates, one per covariate. Default: from the estimand (the treated means for ATT, the pooled means for ATE, the control means for ATC)."
    )]
    pub target_means: Option<Vec<f64>>,

    /// Estimand
    #[schemars(
        description = "Estimand: 'att' (default: controls to the treated means), 'ate' (both groups to the pooled means) or 'atc' (treated to the control means). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub estimand: Option<String>,
}

/// Request for stable balancing weights (SBW).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SBWRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Covariate columns for balance
    #[schemars(description = "Names of covariate columns to balance between treatment groups.")]
    pub covariates: Vec<String>,

    /// Target estimand: 'att' (default), 'ate', or 'atc'
    #[schemars(
        description = "Estimand: 'att' (default), 'ate' or 'atc'. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub estimand: Option<String>,

    /// Balance tolerance for approximate balance (0 = exact balance)
    #[schemars(
        description = "Tolerance for approximate balance of each covariate mean, in the covariates' own units. 0 means exact balance (default)."
    )]
    pub balance_tol: Option<f64>,

    /// Minimum weight allowed (default 0 for non-negativity)
    #[schemars(description = "Minimum weight allowed. Default is 0 (non-negativity constraint).")]
    pub min_weight: Option<f64>,

    /// Penalty parameter for approximate balance (higher = stricter balance)
    #[schemars(
        description = "Not used: approximate balance is solved exactly as a quadratic program. Only the default 1000 is accepted; omit it."
    )]
    pub balance_penalty: Option<f64>,
}

/// Request for twang GBM propensity score estimation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TwangRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Covariate columns for propensity model
    #[schemars(
        description = "Names of covariate columns to include in the GBM propensity score model."
    )]
    pub covariates: Vec<String>,

    /// Stopping rule: 'es.mean' (default), 'es.max', 'ks.mean', 'ks.max'
    #[schemars(
        description = "Stopping rule for the number of boosting iterations: 'es.mean' (default, mean standardized effect size), 'es.max', 'ks.mean' or 'ks.max' (Kolmogorov-Smirnov statistics). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub stop_method: Option<String>,

    /// Target estimand: 'att' (default), 'ate', or 'atc'
    #[schemars(
        description = "Estimand: 'att' (default), 'ate' or 'atc'. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub estimand: Option<String>,

    /// Maximum number of boosting iterations (default: 3000)
    #[schemars(description = "Maximum number of gradient boosting iterations. Default is 3000.")]
    pub n_trees: Option<usize>,

    /// Learning rate / shrinkage (default: 0.01)
    #[schemars(
        description = "Learning rate for gradient boosting. Smaller values need more iterations but often give better results. Default is 0.01."
    )]
    pub shrinkage: Option<f64>,

    /// Balance threshold for early stopping (default: 0.1)
    #[schemars(description = "Balance threshold below which to stop early. Default is 0.1.")]
    pub balance_threshold: Option<f64>,
}

/// Request for propensity score matching (MatchIt).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MatchItRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Covariate columns for matching
    #[schemars(description = "Names of covariate columns to use for matching.")]
    pub covariates: Vec<String>,

    /// Matching method: 'nearest' (default), 'cem', 'full', or 'subclass'
    #[schemars(
        description = "Matching method: 'nearest' (default), 'full' (optimal full matching as optmatch::fullmatch, no bounds as MatchIt; optimal full matching is refused above 1e7 treated-control pairs, optmatch's default maximum problem size), 'cem' (coarsened exact matching with Sturges bins; one-sided strata are left unmatched with a warning) or 'subclass' (n_subclasses quantile subclasses of the propensity score). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub method: Option<String>,

    /// Distance metric: 'logit' (default), 'probit', 'mahalanobis', or 'euclidean'
    #[schemars(
        description = "Distance: 'logit' (default, propensity score by logistic regression; 'glm'), 'probit', 'mahalanobis' or 'euclidean'; not used by 'cem'. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub distance: Option<String>,

    /// Matching ratio (1:k matching, default k=1)
    #[schemars(
        description = "Number of controls per treated unit (1:k matching; default 1): nearest only; ignored with a warning for other methods."
    )]
    pub ratio: Option<usize>,

    /// Caliper width (in SD of propensity score)
    #[schemars(
        description = "Maximum distance for a valid match, in SDs of the propensity score (default none): nearest only; ignored with a warning for other methods."
    )]
    pub caliper: Option<f64>,

    /// Whether to sample with replacement
    #[schemars(
        description = "Match controls with replacement (default false): nearest only; ignored with a warning for other methods."
    )]
    pub replace: Option<bool>,

    /// Number of subclasses for subclassification
    #[schemars(
        description = "Number of propensity-score subclasses (default 5): subclass only; ignored with a warning for other methods."
    )]
    pub n_subclasses: Option<usize>,

    /// Matching order
    #[schemars(
        description = "Order in which treated units pick controls, as MatchIt `m.order`: 'largest' (default for propensity-score distances), 'smallest', 'data' (default for Mahalanobis/Euclidean) or 'random' (seeded by the session seed); nearest only; ignored with a warning for other methods. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub m_order: Option<String>,

    /// Name of the matched sample
    #[schemars(
        description = "Name for the matched sample, registered as a dataset (default '<dataset>_matched'; must differ from the input)."
    )]
    pub output_name: Option<String>,
}

/// Request for Targeted Maximum Likelihood Estimation (TMLE).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TmleRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column. Can be binary or continuous.")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Covariate columns for propensity score and outcome models
    #[schemars(
        description = "Names of covariate columns to include in both propensity score and outcome models."
    )]
    pub covariates: Vec<String>,

    /// Outcome model type: 'logistic' (default) or 'linear'
    #[schemars(
        description = "Outcome model: 'logistic' (default; a continuous outcome is rescaled to [0, 1] by its range, as R's tmle; estimates are on the original scale) or 'linear'. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub q_model: Option<String>,

    /// Lower bound for propensity score truncation
    #[schemars(description = "Lower bound for propensity score truncation. Default is 0.01.")]
    pub ps_lower: Option<f64>,

    /// Upper bound for propensity score truncation
    #[schemars(description = "Upper bound for propensity score truncation. Default is 0.99.")]
    pub ps_upper: Option<f64>,
}

/// Request for Collaborative Targeted Maximum Likelihood Estimation (C-TMLE).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CTmleRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column. Can be binary or continuous.")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Covariate columns (candidates for propensity score selection)
    #[schemars(
        description = "Names of candidate covariate columns. C-TMLE will select which ones to include in the propensity score model via cross-validation."
    )]
    pub covariates: Vec<String>,

    /// Outcome model type: 'logistic' (default) or 'linear'
    #[schemars(
        description = "Outcome model: 'logistic' (default; a logistic Q rescales a continuous outcome to [0, 1] by its range, as R's tmle) or 'linear'. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub q_model: Option<String>,

    /// Number of cross-validation folds (default: 5)
    #[schemars(
        description = "Number of cross-validation folds for covariate selection. Default is 5."
    )]
    pub n_folds: Option<usize>,

    /// Maximum number of covariates to select (optional)
    #[schemars(
        description = "Maximum number of covariates to include in propensity score model. Default is no limit."
    )]
    pub max_covariates: Option<usize>,

    /// Stopping rule: 'cv_minimum' (default), 'one_se', or 'max_covariates'
    #[schemars(
        description = "Stopping rule of the covariate selection: 'cv_minimum' (default, minimum cross-validated risk), 'one_se' (one-standard-error rule) or 'max_covariates' (max_covariates is required there). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub stopping_rule: Option<String>,

    #[schemars(
        description = "Cross-validated risk of the selection path: 'rss' (default), 'varic' (cross-validated variance of the influence curve, Gruber & van der Laan 2010) or 'penrss' (RSS + penalty * ln(n) * k). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub cv_criterion: Option<String>,

    #[schemars(
        description = "Selection order: 'forward' (default), 'backward' or 'prespecified' (the covariates in the listed order). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub order: Option<String>,

    /// Lower bound for propensity score truncation (default: 0.025)
    #[schemars(description = "Lower bound for propensity score truncation. Default is 0.025.")]
    pub ps_lower: Option<f64>,

    /// Upper bound for propensity score truncation (default: 0.975)
    #[schemars(description = "Upper bound for propensity score truncation. Default is 0.975.")]
    pub ps_upper: Option<f64>,
}

/// Request for Longitudinal Targeted Maximum Likelihood Estimation (LTMLE).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LtmleRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome column names at each time point (chronological order)
    #[schemars(
        description = "Outcome column at each time point (one per time point); only the last is the outcome node (p2a's LTMLE has no intermediate outcome or censoring nodes)."
    )]
    pub outcomes: Vec<String>,

    /// Treatment column names at each time point (chronological order)
    #[schemars(
        description = "Names of treatment indicator columns at each time point. Must be binary (0 or 1)."
    )]
    pub treatments: Vec<String>,

    /// Covariate column names at each time point (chronological order, comma-separated within each time point)
    #[schemars(
        description = "Covariate columns at each time point, one entry per time point, each a comma-separated list. The outcome is never a covariate, and a treatment may only be a covariate of later time points."
    )]
    pub covariates: Vec<String>,

    /// Outcome model type: 'linear' (default) or 'logistic'
    #[schemars(
        description = "Outcome model: 'logistic' (default, ltmle's; a continuous outcome is rescaled to [0, 1] by its range, as ltmle) or 'linear'. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub q_model: Option<String>,

    /// Lower bound for propensity score truncation
    #[schemars(
        description = "Lower bound on the cumulative probability of following the regime, ltmle `gbounds` (default 0.01)."
    )]
    pub ps_lower: Option<f64>,

    /// Upper bound for propensity score truncation
    #[schemars(
        description = "Upper bound on the cumulative probability of following the regime, ltmle `gbounds` (default 1)."
    )]
    pub ps_upper: Option<f64>,

    /// Regime
    #[schemars(
        description = "Regime whose counterfactual mean is estimated: 'always' (default), 'never' or 'dynamic' (A_t = 1{dynamic_columns[t] > dynamic_threshold}). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub regime: Option<String>,

    /// Reference regime
    #[schemars(
        description = "Reference regime of the ATE: 'never' (default), 'always', 'dynamic' or 'none' (E[Y^regime] only). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub contrast: Option<String>,

    /// Covariate of each time point's threshold rule
    #[schemars(
        description = "For a 'dynamic' regime or contrast: the covariate of each time point (one per time point) whose value above dynamic_threshold sets A_t = 1."
    )]
    pub dynamic_columns: Option<Vec<String>>,

    /// Threshold of the rule
    #[schemars(
        description = "For a 'dynamic' regime or contrast: the threshold c of A_t = 1{L_t > c}."
    )]
    pub dynamic_threshold: Option<f64>,
}

/// Request for Regression Standardization (G-computation).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StdRegRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column. Can be continuous or binary.")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Covariate columns for the outcome model
    #[schemars(description = "Names of covariate columns to include in the outcome model.")]
    pub covariates: Vec<String>,

    /// Outcome model type: 'linear' (default), 'logistic', or 'poisson'
    #[schemars(
        description = "Outcome model: 'linear' (default), 'logistic' or 'poisson'. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub model_type: Option<String>,

    /// Estimand: 'ate' (default), 'att', 'atc', or 'levels'
    #[schemars(
        description = "Estimand: 'ate' (default), 'att', 'atc' or 'levels' (the standardized means only). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub estimand: Option<String>,

    /// SE method: 'bootstrap' (default), 'delta', or 'sandwich'
    #[schemars(
        description = "Standard errors: 'bootstrap' (default), 'delta' or 'sandwich' ('robust'). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub se_method: Option<String>,

    /// Number of bootstrap replications (if using bootstrap SE)
    #[schemars(
        description = "Number of bootstrap replications for SE estimation. Default is 999."
    )]
    pub n_bootstrap: Option<usize>,

    /// Whether to include treatment-covariate interactions
    #[schemars(
        description = "Include treatment-covariate interactions in outcome model. Default is false."
    )]
    pub interactions: Option<bool>,

    /// Confidence level (e.g., 0.95 for 95% CI)
    #[schemars(description = "Confidence level for intervals. Default is 0.95.")]
    pub confidence_level: Option<f64>,
}

/// Request for Parametric G-Formula with time-varying treatments.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GFormulaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name (observed at final time point)
    #[schemars(description = "Name of the outcome variable column.")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Baseline (time-invariant) covariate column names
    #[schemars(description = "Names of baseline covariates that do not change over time.")]
    pub baseline_covariates: Vec<String>,

    /// Time-varying covariate column names (must have suffix _t0, _t1, etc.)
    #[schemars(
        description = "Base names of time-varying covariates. Columns must be named as 'varname_t0', 'varname_t1', etc. for each time point."
    )]
    pub time_varying_covariates: Vec<String>,

    /// Treatment column names for each time point (e.g., ['treat_t0', 'treat_t1'])
    #[schemars(
        description = "Column names for treatment at each time point. Order matters: first element is treatment at t=0, second at t=1, etc."
    )]
    pub treatment_cols: Vec<String>,

    /// Number of time points
    #[schemars(
        description = "Number of time points in the analysis (must match number of treatment columns)."
    )]
    pub time_points: usize,

    /// Intervention type: 'always_treat', 'never_treat', 'natural', or 'threshold'
    #[schemars(
        description = "Intervention: 'always_treat' (default; 'always'), 'never_treat' ('never'), 'natural_course' (the observed treatment; 'natural') or 'threshold' (needs threshold_variable and threshold_cutoff). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub intervention: Option<String>,

    /// For threshold intervention: variable index (0-indexed into time-varying covariates)
    #[schemars(
        description = "Intervention 'threshold' only (required there): index of the time-varying covariate to check (0-indexed)."
    )]
    pub threshold_variable: Option<usize>,

    /// For threshold intervention: cutoff value
    #[schemars(
        description = "Intervention 'threshold' only (required there): threshold value for the treatment decision."
    )]
    pub threshold_cutoff: Option<f64>,

    /// For threshold intervention: treat if above (true) or below (false)
    #[schemars(
        description = "Intervention 'threshold' only: if true, treat when variable > cutoff; if false, treat when variable <= cutoff."
    )]
    pub threshold_above: Option<bool>,

    /// Outcome type: 'continuous' (default), 'binary', or 'survival'
    #[schemars(
        description = "Outcome model: 'continuous' (default, linear model), 'binary' (logistic model) or 'survival' (needs time_col; pooled logistic discrete-time hazard on person-periods, logit h = b0 + b1 t + b2 A_t + baseline + L_t (a linear period term); risks 1 - S(t) under the intervention, standardised exactly over the covariates for static regimes without time-varying covariates, by Monte Carlo otherwise). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub outcome_type: Option<String>,

    #[schemars(
        description = "Survival outcome: the last period each subject is observed, 1..time_points; required with outcome_type 'survival'."
    )]
    #[serde(default)]
    pub time_col: Option<String>,

    #[schemars(
        description = "Survival outcome: the 0/1 event indicator at that period; default: outcome."
    )]
    #[serde(default)]
    pub event_col: Option<String>,

    /// Number of Monte Carlo simulations (default: 1000)
    #[schemars(description = "Number of Monte Carlo simulations. Default is 1000.")]
    pub n_simulations: Option<usize>,

    /// Number of bootstrap samples for standard errors (default: 200)
    #[schemars(
        description = "Number of bootstrap samples for standard error estimation. Default is 200."
    )]
    pub n_bootstrap: Option<usize>,

    /// Confidence level (default: 0.95)
    #[schemars(description = "Confidence level for intervals. Default is 0.95.")]
    pub confidence_level: Option<f64>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for Causal Mediation Analysis.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MediationRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Mediator variable column name
    #[schemars(
        description = "Name of the mediator variable column - the intermediate variable through which treatment may affect the outcome."
    )]
    pub mediator: String,

    /// Covariate columns for propensity score models
    #[schemars(
        description = "Names of covariate columns for adjustment in propensity score models."
    )]
    pub covariates: Vec<String>,

    /// Trimming threshold for extreme propensity scores
    #[schemars(
        description = "Trim observations with propensity scores below trim or above 1-trim. Default is 0.05."
    )]
    pub trim: Option<f64>,

    /// Number of bootstrap replications for standard errors
    #[schemars(
        description = "Bootstrap replications (default 999), seeded by the session seed; the estimate fails unless at least min(50, n_bootstrap) replicates are usable."
    )]
    #[serde(rename = "n_bootstrap", alias = "bootstrap")]
    pub bootstrap: Option<usize>,
}

/// Request for Natural Effect Models (medflex) mediation analysis.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NaturalEffectsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column (continuous).")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(description = "Treatment column; must be coded 0/1.")]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Mediator variable column name
    #[schemars(
        description = "Name of the mediator variable column - the intermediate variable through which treatment may affect the outcome."
    )]
    pub mediator: String,

    /// Confounder columns for adjustment
    #[schemars(
        description = "Names of confounder columns for adjustment in mediator and outcome models. Can be empty."
    )]
    pub confounders: Option<Vec<String>>,

    /// Whether to include treatment-mediator interaction
    #[schemars(
        description = "Include treatment-mediator interaction term in outcome model. Default is true. Set to false for simple product-of-coefficients decomposition."
    )]
    pub allow_interaction: Option<bool>,

    /// Number of bootstrap replications for standard errors
    #[schemars(
        description = "0 = robust sandwich SEs (medflex se = 'robust'); otherwise that many bootstrap replicates (default 1000, seeded by the session seed): bootstrap SE and bias-corrected normal intervals, as medflex confint(type = 'norm'). Effects are on the link scale for the ratio scales."
    )]
    pub n_bootstrap: Option<usize>,

    /// Confidence level for intervals
    #[schemars(
        description = "Confidence level for intervals (e.g., 0.95 for 95% CI). Default is 0.95."
    )]
    pub confidence_level: Option<f64>,

    /// Effect scale (difference, ratio, odds_ratio)
    #[schemars(
        description = "Scale of the natural effect model: 'difference' (default), 'ratio' (log link) or 'odds_ratio' (logit link). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub scale: Option<String>,

    /// Outcome working-model family
    #[schemars(
        description = "Outcome working-model family: 'gaussian' (least squares), 'poisson' (log) or 'binomial' (logit); default from scale: difference -> gaussian, ratio -> poisson, odds_ratio -> binomial. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub family: Option<String>,
}

/// Request for Synthetic Control Method.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SyntheticControlRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Unit identifier column name
    #[schemars(description = "Name of the column identifying units (e.g., 'state', 'country').")]
    #[serde(alias = "entity_var", alias = "entity_col", alias = "entity")]
    pub unit_col: String,

    /// Time period column name
    #[schemars(
        description = "Name of the column identifying time periods (must be integer, e.g., 'year')."
    )]
    #[serde(alias = "time_var", alias = "time", alias = "period")]
    pub time_col: String,

    /// Name/ID of the treated unit
    #[schemars(description = "Name or ID of the treated unit (must match values in unit_col).")]
    pub treated_unit: String,

    /// Treatment time (first post-treatment period)
    #[schemars(
        description = "First post-treatment period (treatment starts at or after this time)."
    )]
    pub treatment_time: i64,

    /// Predictor specifications
    #[schemars(
        description = "List of predictor specifications. Can be column names (strings) or detailed specs with aggregation and time windows."
    )]
    pub predictors: Vec<SynthPredictorSpec>,

    /// V matrix optimization method
    #[schemars(
        description = "Predictor importance weights V: 'datadriven' (default), 'equal' or 'custom' (needs custom_v_weights). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub v_method: Option<String>,

    /// Custom V weights (if v_method is 'custom')
    #[schemars(
        description = "Custom predictor weights (only used if v_method is 'custom'). Must sum to 1."
    )]
    pub custom_v_weights: Option<Vec<f64>>,

    /// Whether to run placebo tests for inference
    #[schemars(
        description = "Whether to run placebo tests for inference. Default is false (can be slow with many units)."
    )]
    pub run_placebos: Option<bool>,

    /// Optimization window (start, end)
    #[schemars(
        description = "Time window for optimization [start, end]. If omitted, uses all pre-treatment periods."
    )]
    pub optimization_window: Option<(i64, i64)>,

    /// Convergence tolerance
    #[schemars(description = "Tolerance for optimization convergence. Default is 1e-6.")]
    pub tolerance: Option<f64>,

    /// Maximum iterations for V optimization
    #[schemars(description = "Maximum iterations for V optimization. Default is 1000.")]
    pub max_iter: Option<usize>,

    /// Minimum weight threshold for output
    #[schemars(
        description = "Minimum weight to display in output (for readability). Default is 0.001."
    )]
    pub weight_threshold: Option<f64>,
}

/// Request for Generalized Synthetic Control (gsynth) estimation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GsynthRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column (Y).")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Treatment indicator column name
    #[schemars(
        description = "Name of the treatment indicator column (D, binary 0/1). Treatment can start at different times for different units."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Unit identifier column name
    #[schemars(description = "Name of the column identifying units (e.g., 'state', 'country').")]
    #[serde(alias = "entity_var", alias = "entity_col", alias = "entity")]
    pub unit_col: String,

    /// Time period column name
    #[schemars(description = "Name of the column identifying time periods (must be numeric).")]
    #[serde(alias = "time_var", alias = "time", alias = "period")]
    pub time_col: String,

    /// Covariate columns
    #[schemars(description = "Optional list of covariate column names to include in the model.")]
    pub covariates: Option<Vec<String>>,

    /// Number of factors (0 for auto-selection via CV)
    #[schemars(
        description = "Number of latent factors. Use 0 with cross_validate=true for automatic selection. Default is 2."
    )]
    pub n_factors: Option<usize>,

    /// Whether to cross-validate factor selection
    #[schemars(
        description = "Whether to select number of factors via cross-validation. Default is false."
    )]
    pub cross_validate: Option<bool>,

    /// Maximum factors to consider in CV
    #[schemars(
        description = "Maximum number of factors to consider during cross-validation. Default is 5."
    )]
    pub max_factors: Option<usize>,

    /// Estimator type
    #[schemars(
        description = "Estimator: 'ife' (default, interactive fixed effects) or 'mc' (matrix completion). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub estimator: Option<String>,

    /// Fixed effects specification
    #[schemars(
        description = "Additive fixed effects: 'none', 'unit' (default), 'time' or 'two_way' ('twoway'). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub force: Option<String>,

    /// Whether to compute bootstrap standard errors
    #[schemars(description = "Whether to compute bootstrap standard errors. Default is false.")]
    pub bootstrap_se: Option<bool>,

    /// Number of bootstrap iterations
    #[schemars(
        description = "Number of bootstrap iterations for standard errors. Default is 500."
    )]
    pub n_bootstrap: Option<usize>,
}

/// Request for Synthetic Control with Prediction Intervals (SCPI).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScpiRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column (Y).")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Unit identifier column name
    #[schemars(description = "Name of the column identifying units (e.g., 'state', 'country').")]
    #[serde(alias = "entity_var", alias = "entity_col", alias = "entity")]
    pub unit_col: String,

    /// Time period column name
    #[schemars(description = "Name of the column identifying time periods (must be numeric).")]
    #[serde(alias = "time_var", alias = "time", alias = "period")]
    pub time_col: String,

    /// Treated unit identifier
    #[schemars(description = "Identifier of the treated unit (must match a value in unit_col).")]
    pub treated_unit: String,

    /// Treatment time period
    #[schemars(description = "First post-treatment time period (treatment starts at this time).")]
    pub treatment_time: i64,

    /// Constraint type
    #[schemars(
        description = "Weight constraint: 'simplex' (default, non-negative weights summing to 1), 'lasso', 'ridge' or 'lasso_simplex'. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub constraint: Option<String>,

    /// Lambda for Lasso/Ridge constraints
    #[schemars(
        description = "Regularization parameter for Lasso or Ridge constraints. Default is 0.1."
    )]
    pub lambda: Option<f64>,

    /// Significance level
    #[schemars(
        description = "Significance level for prediction intervals. Default is 0.05 (95% PI)."
    )]
    pub alpha: Option<f64>,

    /// Variance estimation method
    #[schemars(
        description = "Out-of-sample variance: 'subgaussian' (default, more conservative), 'gaussian', 'loo_cv' or 'kfold_cv'. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub variance_method: Option<String>,

    /// Number of CV folds
    #[schemars(
        description = "Number of folds for K-fold cross-validation (if variance_method='kfold_cv'). Default is 5."
    )]
    pub cv_folds: Option<usize>,

    /// Minimum weight threshold
    #[schemars(
        description = "Minimum weight to report in output (for sparsity). Default is 0.001."
    )]
    pub weight_threshold: Option<f64>,
}

/// Request for Sharp Regression Discontinuity estimation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RdEstimateRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable (Y) column.")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Running variable column name
    #[schemars(description = "Name of the running (forcing) variable (X) column.")]
    #[serde(alias = "x", alias = "score")]
    pub running_var: String,

    /// Cutoff value
    #[schemars(description = "Cutoff value for the running variable. Default is 0.")]
    pub cutoff: Option<f64>,

    /// Polynomial order for estimation
    #[schemars(
        description = "Polynomial order for local polynomial estimation. Default is 1 (local linear)."
    )]
    pub p: Option<usize>,

    /// Kernel type
    #[schemars(
        description = "Kernel function: 'triangular' (default), 'epanechnikov', or 'uniform'."
    )]
    pub kernel: Option<String>,

    /// Bandwidth selection method
    #[schemars(
        description = "Bandwidth selector, as rdrobust::rdbwselect `bwselect`: 'mserd' (default; MSE-optimal, one common bandwidth), 'msetwo' (MSE-optimal, separate left and right), 'msesum' (MSE-optimal for the sum of the two regression estimates, one common bandwidth), 'msecomb1' (smaller of mserd and msesum), 'msecomb2' (median of mserd, msesum and msetwo, per side), 'cerrd' (CER-optimal, one common bandwidth), 'certwo' (CER-optimal, separate left and right), 'cersum' (CER-optimal for the sum of the two regression estimates, one common bandwidth), 'cercomb1' (smaller of cerrd and cersum) or 'cercomb2' (median of cerrd, cersum and certwo, per side). Case is ignored ('mse' and 'cer' are accepted for mserd and cerrd); other values are an error."
    )]
    pub bwselect: Option<String>,

    /// Main bandwidth (overrides automatic selection)
    #[schemars(
        description = "Main bandwidth h. If not specified, h and b are selected as rdrobust::rdbwselect (bwselect)."
    )]
    pub h: Option<f64>,

    /// Bias bandwidth (overrides automatic selection)
    #[schemars(
        description = "Bias bandwidth b. Default: selected with h, or h itself (rho = 1) when only h is given, as rdrobust. b is also used when h is selected (rdrobust ignores it there)."
    )]
    pub b: Option<f64>,

    /// Confidence level
    #[schemars(description = "Confidence level for intervals. Default is 0.95.")]
    pub level: Option<f64>,

    /// Variance estimator
    #[schemars(
        description = "Variance estimator: 'nn' (default; nearest-neighbour residuals with 3 matches, as rdrobust), 'hc0', 'hc1', 'hc2' or 'hc3'."
    )]
    #[serde(default)]
    pub vce: Option<String>,

    /// rdbwselect regularisation weight
    #[schemars(
        description = "Weight of the regularisation term of rdbwselect (`scaleregul`, default 1; 0 switches it off); must be finite and >= 0. Used only when bandwidths are selected."
    )]
    pub scaleregul: Option<f64>,
}

/// Request for RD bandwidth selection only.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RdBandwidthRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable (Y) column.")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Running variable column name
    #[schemars(description = "Name of the running (forcing) variable (X) column.")]
    #[serde(alias = "x", alias = "score")]
    pub running_var: String,

    /// Cutoff value
    #[schemars(description = "Cutoff value for the running variable. Default is 0.")]
    pub cutoff: Option<f64>,

    /// Polynomial order
    #[schemars(description = "Polynomial order for estimation. Default is 1.")]
    pub p: Option<usize>,

    /// Kernel type
    #[schemars(
        description = "Kernel function: 'triangular' (default), 'epanechnikov', or 'uniform'."
    )]
    pub kernel: Option<String>,

    /// Bandwidth selection method
    #[schemars(
        description = "Bandwidth selector, as rdrobust::rdbwselect `bwselect`: 'mserd' (default; MSE-optimal, one common bandwidth), 'msetwo' (MSE-optimal, separate left and right), 'msesum' (MSE-optimal for the sum of the two regression estimates, one common bandwidth), 'msecomb1' (smaller of mserd and msesum), 'msecomb2' (median of mserd, msesum and msetwo, per side), 'cerrd' (CER-optimal, one common bandwidth), 'certwo' (CER-optimal, separate left and right), 'cersum' (CER-optimal for the sum of the two regression estimates, one common bandwidth), 'cercomb1' (smaller of cerrd and cersum) or 'cercomb2' (median of cerrd, cersum and certwo, per side). Case is ignored ('mse' and 'cer' are accepted for mserd and cerrd); other values are an error."
    )]
    pub bwselect: Option<String>,

    /// rdbwselect regularisation weight
    #[schemars(
        description = "Weight of the regularisation term of rdbwselect (`scaleregul`, default 1; 0 switches it off); must be finite and >= 0. Used only when bandwidths are selected."
    )]
    pub scaleregul: Option<f64>,
}

/// Request for Fuzzy Regression Discontinuity estimation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FuzzyRdRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable (Y) column.")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Running variable column name
    #[schemars(description = "Name of the running (forcing) variable (X) column.")]
    #[serde(alias = "x", alias = "score")]
    pub running_var: String,

    /// Treatment indicator column name
    #[schemars(
        description = "Name of the treatment indicator column (actual treatment received, 0/1)."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Cutoff value
    #[schemars(description = "Cutoff value for the running variable. Default is 0.")]
    pub cutoff: Option<f64>,

    /// Polynomial order
    #[schemars(description = "Polynomial order for estimation. Default is 1.")]
    pub p: Option<usize>,

    /// Kernel type
    #[schemars(
        description = "Kernel function: 'triangular' (default), 'epanechnikov', or 'uniform'."
    )]
    pub kernel: Option<String>,

    /// Bandwidth selection method
    #[schemars(
        description = "Bandwidth selector, as rdrobust::rdbwselect `bwselect`: 'mserd' (default; MSE-optimal, one common bandwidth), 'msetwo' (MSE-optimal, separate left and right), 'msesum' (MSE-optimal for the sum of the two regression estimates, one common bandwidth), 'msecomb1' (smaller of mserd and msesum), 'msecomb2' (median of mserd, msesum and msetwo, per side), 'cerrd' (CER-optimal, one common bandwidth), 'certwo' (CER-optimal, separate left and right), 'cersum' (CER-optimal for the sum of the two regression estimates, one common bandwidth), 'cercomb1' (smaller of cerrd and cersum) or 'cercomb2' (median of cerrd, cersum and certwo, per side). Case is ignored ('mse' and 'cer' are accepted for mserd and cerrd); other values are an error."
    )]
    pub bwselect: Option<String>,

    /// Main bandwidth
    #[schemars(
        description = "Main bandwidth h. If not specified, h and b are selected as rdrobust::rdbwselect (bwselect) on the outcome."
    )]
    pub h: Option<f64>,

    /// Bias bandwidth
    #[schemars(
        description = "Bias bandwidth b, used for both stages. Default: selected with h, or h itself (rho = 1) when only h is given, as rdrobust. b is also used when h is selected (rdrobust ignores it there)."
    )]
    #[serde(default)]
    pub b: Option<f64>,

    /// Confidence level
    #[schemars(description = "Confidence level for intervals. Default is 0.95.")]
    pub level: Option<f64>,

    /// Variance estimator
    #[schemars(
        description = "Variance estimator: 'nn' (default; nearest-neighbour residuals with 3 matches, as rdrobust), 'hc0', 'hc1', 'hc2' or 'hc3'."
    )]
    #[serde(default)]
    pub vce: Option<String>,

    /// rdbwselect regularisation weight
    #[schemars(
        description = "Weight of the regularisation term of rdbwselect (`scaleregul`, default 1; 0 switches it off); must be finite and >= 0. Used only when bandwidths are selected."
    )]
    pub scaleregul: Option<f64>,
}

/// Request for Multi-Cutoff Regression Discontinuity estimation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RdMultiRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable (Y) column.")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Running variable column name
    #[schemars(description = "Name of the running (forcing) variable (X) column.")]
    #[serde(alias = "x", alias = "score")]
    pub running_var: String,

    /// Cutoff values
    #[schemars(description = "List of cutoff values c1, c2, ..., cJ for the running variable.")]
    pub cutoffs: Vec<f64>,

    /// Cutoff assignment column (optional)
    #[schemars(
        description = "Column indicating which cutoff each observation belongs to (0, 1, ...). If not specified, observations are assigned to the nearest cutoff."
    )]
    pub cutoff_col: Option<String>,

    /// Whether to compute pooled estimate
    #[schemars(
        description = "Whether to compute a pooled treatment effect across all cutoffs. Default is true."
    )]
    pub pooled: Option<bool>,

    /// Pooling weight scheme
    #[schemars(
        description = "Weights that pool the cutoff effects: 'sample_size' (default), 'inverse_variance' ('iv') or 'equal'. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub pooling_weights: Option<String>,

    /// Bandwidth specification
    #[schemars(
        description = "Bandwidth specification: single value for global bandwidth, or omit for per-cutoff optimal."
    )]
    pub bandwidth: Option<f64>,

    /// Per-cutoff bandwidths
    #[schemars(
        description = "List of bandwidths for each cutoff. Must match length of cutoffs if specified."
    )]
    pub bandwidths: Option<Vec<f64>>,

    /// Polynomial order for estimation
    #[schemars(
        description = "Polynomial order for local polynomial estimation. Default is 1 (local linear)."
    )]
    pub p: Option<usize>,

    /// Kernel type
    #[schemars(
        description = "Kernel function: 'triangular' (default), 'epanechnikov', or 'uniform'."
    )]
    pub kernel: Option<String>,

    /// Whether to test for heterogeneity
    #[schemars(
        description = "Whether to perform a chi-squared test for heterogeneous effects across cutoffs. Default is true."
    )]
    pub test_heterogeneity: Option<bool>,

    /// Confidence level
    #[schemars(description = "Confidence level for intervals. Default is 0.95.")]
    pub level: Option<f64>,

    /// Variance estimator
    #[schemars(
        description = "Variance estimator: 'nn' (default; nearest-neighbour residuals with 3 matches, as rdrobust), 'hc0', 'hc1', 'hc2' or 'hc3'."
    )]
    #[serde(default)]
    pub vce: Option<String>,

    /// rdbwselect regularisation weight
    #[schemars(
        description = "Weight of the regularisation term of rdbwselect (`scaleregul`, default 1; 0 switches it off); must be finite and >= 0. Used only when bandwidths are selected."
    )]
    pub scaleregul: Option<f64>,

    /// rdbwselect selector
    #[schemars(
        description = "Bandwidth selector, as rdrobust::rdbwselect `bwselect`: 'mserd' (default; MSE-optimal, one common bandwidth), 'msetwo' (MSE-optimal, separate left and right), 'msesum' (MSE-optimal for the sum of the two regression estimates, one common bandwidth), 'msecomb1' (smaller of mserd and msesum), 'msecomb2' (median of mserd, msesum and msetwo, per side), 'cerrd' (CER-optimal, one common bandwidth), 'certwo' (CER-optimal, separate left and right), 'cersum' (CER-optimal for the sum of the two regression estimates, one common bandwidth), 'cercomb1' (smaller of cerrd and cersum) or 'cercomb2' (median of cerrd, cersum and certwo, per side). Case is ignored ('mse' and 'cer' are accepted for mserd and cerrd); other values are an error."
    )]
    pub bwselect: Option<String>,
}

/// Request for Causal Forest estimation (Wager & Athey 2018).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CausalForestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome column name
    #[schemars(description = "Name of the outcome variable (Y).")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Treatment column name
    #[schemars(description = "Name of the binary treatment variable (W). Must be 0/1.")]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Covariate column names
    #[schemars(description = "Names of the covariate columns (X variables).")]
    pub covariates: Vec<String>,

    /// Number of trees (default: 2000)
    #[schemars(description = "Number of trees in the forest. Default is 2000.")]
    pub n_trees: Option<usize>,

    /// Minimum node size (default: 5)
    #[schemars(description = "Minimum number of observations in each leaf. Default is 5.")]
    pub min_node_size: Option<usize>,

    /// Maximum tree depth (default: 10)
    #[schemars(description = "Maximum depth of each tree. Default is 10.")]
    pub max_depth: Option<usize>,

    /// Use honest splitting (default: true)
    #[schemars(
        description = "Whether to use honest splitting (separate data for tree structure and estimation). Default is true."
    )]
    pub honesty: Option<bool>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for BART-based Causal Inference (bartCause style).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BartCausalRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome column name
    #[schemars(description = "Name of the outcome variable (Y).")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Treatment column name
    #[schemars(description = "Name of the binary treatment variable (W). Must be 0/1.")]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Covariate column names
    #[schemars(description = "Names of the covariate columns (X variables).")]
    pub covariates: Vec<String>,

    /// Number of trees per response surface (default: 200)
    #[schemars(description = "Number of trees in each response surface ensemble. Default is 200.")]
    pub n_trees: Option<usize>,

    /// Maximum tree depth (default: 4)
    #[schemars(
        description = "Maximum depth of each tree. Default is 4 (BART uses shallow trees)."
    )]
    pub max_depth: Option<usize>,

    /// Number of bootstrap samples for uncertainty (default: 100)
    #[schemars(
        description = "Number of bootstrap samples for confidence intervals. Default is 100."
    )]
    pub n_bootstrap: Option<usize>,

    /// Include propensity score as covariate (default: false)
    #[schemars(
        description = "Whether to include estimated propensity score as a covariate. Default is false."
    )]
    pub include_propensity: Option<bool>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for Treatment Effect Heterogeneity Test (hettx).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HetTxRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome column name
    #[schemars(description = "Name of the outcome variable (Y).")]
    #[serde(alias = "dep_var", alias = "y", alias = "response")]
    pub outcome: String,

    /// Treatment column name
    #[schemars(description = "Name of the binary treatment indicator (0/1).")]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Covariate column names
    #[schemars(description = "Names of covariate columns for matching and decomposition.")]
    pub covariates: Vec<String>,

    /// Number of permutations (default: 1000)
    #[schemars(description = "Number of permutations for Fisherian inference. Default is 1000.")]
    pub n_permutations: Option<usize>,

    /// Test statistic type (default: 'variance')
    #[schemars(description = "Test statistic: 'variance' (default), 'range', 'iqr', or 'mad'.")]
    pub test_statistic: Option<String>,

    /// Whether to decompose heterogeneity (default: true)
    #[schemars(
        description = "Whether to decompose heterogeneity into systematic and idiosyncratic components. Default is true."
    )]
    pub decompose: Option<bool>,

    /// Effect estimation method (default: 'matching')
    #[schemars(
        description = "Method for estimating individual effects: 'matching' (default), 'regression', or 'stratified'."
    )]
    pub effect_method: Option<String>,

    /// Number of nearest neighbors for matching (default: 3)
    #[schemars(
        description = "Number of nearest neighbors for matching-based imputation. Default is 3."
    )]
    pub n_neighbors: Option<usize>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for sensitivity analysis for unmeasured confounding (sensemakr).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SensemakrRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable (Y) column name
    #[schemars(description = "Name of the outcome variable (Y) column.")]
    pub y: String,

    /// Treatment variable column name
    #[schemars(
        description = "Name of the treatment variable whose coefficient will be analyzed for sensitivity."
    )]
    #[serde(alias = "treatment_var", alias = "treat")]
    pub treatment: String,

    /// Control covariate column names
    #[schemars(
        description = "Names of the control covariate columns. These are conditioned on when assessing confounding."
    )]
    pub covariates: Vec<String>,

    /// Benchmark covariates for bounding (optional)
    #[schemars(
        description = "Covariate names to use as benchmarks. Their partial R² provides intuition about confounding magnitude."
    )]
    pub benchmark_covariates: Option<Vec<String>>,

    /// Multiplier for treatment benchmark partial R² (kd)
    #[schemars(
        description = "Multiplier applied to benchmark partial R² with treatment. E.g., kd=2 assumes confounder is twice as strong. Default: 1.0."
    )]
    pub kd: Option<f64>,

    /// Multiplier for outcome benchmark partial R² (ky)
    #[schemars(
        description = "Multiplier applied to benchmark partial R² with outcome. Default: same as kd."
    )]
    pub ky: Option<f64>,

    /// Proportion of effect to reduce (q)
    #[schemars(
        description = "Proportion of the effect to explain away. q=1 (default) means nullify; q=0.5 means reduce by half."
    )]
    pub q: Option<f64>,

    /// Significance level for RV_alpha
    #[schemars(
        description = "Significance level for robustness value calculation. Default: 0.05."
    )]
    pub alpha: Option<f64>,

    /// Generate contour plot data
    #[schemars(
        description = "Whether to generate data for sensitivity contour plots. Default: false."
    )]
    pub contour_data: Option<bool>,
}

/// Request for average marginal effects computation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MarginalEffectsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variable column names
    #[schemars(description = "Names of the independent variable columns.")]
    pub x: Vec<String>,

    /// Model type
    #[schemars(
        description = "Model: 'ols' (default; 'linear'), 'logit' ('logistic') or 'probit'. Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub model: Option<String>,
}

/// Request for E-value sensitivity analysis for unmeasured confounding.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EValueRequest {
    /// Type of effect measure
    #[schemars(
        description = "Effect measure: 'rr' (risk ratio), 'or' (odds ratio), 'hr' (hazard ratio), 'smd' (standardized mean difference) or 'rd' (risk difference; needs baseline_risk). Case and separators ('-', '_', '.', ' ') are ignored; other values are an error."
    )]
    pub effect_type: String,

    /// Point estimate of the effect
    #[schemars(
        description = "Point estimate of the effect measure (e.g., RR=2.5, OR=3.0, SMD=0.5)."
    )]
    pub point: f64,

    /// Lower bound of 95% confidence interval
    #[schemars(
        description = "Lower bound of the 95% confidence interval. Optional but recommended."
    )]
    pub ci_lower: Option<f64>,

    /// Upper bound of 95% confidence interval
    #[schemars(
        description = "Upper bound of the 95% confidence interval. Optional but recommended."
    )]
    pub ci_upper: Option<f64>,

    /// Standard error (for SMD)
    #[schemars(
        description = "Standard error of the estimate. Required for SMD if CI not provided."
    )]
    pub se: Option<f64>,

    /// Whether outcome is rare (for OR/HR)
    #[schemars(
        description = "Whether the outcome is rare (<15% prevalence). If true, OR/HR used as RR approximation. If false, sqrt transformation applied. Default: true for OR/HR."
    )]
    pub rare: Option<bool>,

    /// Baseline risk (for RD)
    #[schemars(
        description = "Baseline risk (probability in unexposed group), required for risk difference. Must be between 0 and 1."
    )]
    pub baseline_risk: Option<f64>,
}
