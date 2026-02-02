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
pub struct SynthPredictorSpec {
    /// Column name of the predictor variable
    #[schemars(description = "Column name of the predictor variable.")]
    pub column: String,

    /// How to aggregate the predictor over time
    #[schemars(description = "Aggregation method: 'mean' (default), 'first', 'last', or 'sum'.")]
    pub aggregation: Option<String>,

    /// Optional time window (start, end) for aggregation
    #[schemars(
        description = "Time window for predictor aggregation as [start, end]. If omitted, uses all pre-treatment periods."
    )]
    pub time_window: Option<(i64, i64)>,
}

/// Request for general GMM IV estimation (Hansen 1982).
#[derive(Debug, Deserialize, JsonSchema)]
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
        description = "Estimation method: 'twostep' (default), 'iterative', or 'cue' (continuously updated)."
    )]
    pub method: Option<String>,

    /// Variance-covariance type
    #[schemars(
        description = "Vcov type: 'hac' (default, robust to serial correlation), 'iid', or 'fixed'."
    )]
    pub vcov: Option<String>,

    /// HAC kernel type
    #[schemars(
        description = "Kernel for HAC weighting: 'bartlett' (default), 'parzen', 'qs', 'truncated'."
    )]
    pub kernel: Option<String>,

    /// HAC bandwidth
    #[schemars(description = "Bandwidth for HAC estimation. None for automatic selection.")]
    pub bandwidth: Option<usize>,
}

/// Request for IV/2SLS regression.
#[derive(Debug, Deserialize, JsonSchema)]
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
    pub treatment: String,

    /// Outcome column (binary 0/1)
    #[schemars(description = "Name of the binary outcome column (Y). Example: recovery status.")]
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
    #[schemars(description = "Optional covariate column names to include in the second stage.")]
    pub x: Option<Vec<String>>,

    /// Polynomial degree for MTE curve
    #[schemars(
        description = "Polynomial degree for MTE approximation. Higher degrees allow more flexible MTE shapes but may overfit. Default is 2."
    )]
    pub mte_degree: Option<usize>,

    /// Propensity score model type
    #[schemars(
        description = "Propensity score model: 'probit' (default), 'logit', or 'linear'. Probit is standard in the MTE literature."
    )]
    pub propensity_model: Option<String>,

    /// Number of grid points for MTE curve
    #[schemars(description = "Number of grid points for evaluating MTE curve. Default is 100.")]
    pub n_grid: Option<usize>,
}

/// Request for Difference-in-Differences estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiDRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable column name
    #[schemars(description = "Name of the outcome/dependent variable column.")]
    pub dep_var: String,

    /// Treatment group indicator column (0/1)
    #[schemars(description = "Column indicating treatment group (1 = treated, 0 = control).")]
    pub treatment_var: String,

    /// Post-treatment period indicator column (0/1)
    #[schemars(description = "Column indicating post-treatment period (1 = post, 0 = pre).")]
    pub post_var: String,
}

/// Request for Callaway-Sant'Anna staggered DiD estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StaggeredDiDRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
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
    pub time_col: String,

    /// Unit identifier column
    #[schemars(
        description = "Column containing the unit/individual identifier (e.g., state_id, firm_id)."
    )]
    pub unit_col: String,

    /// Covariate columns (optional)
    #[schemars(description = "Optional covariates for conditional parallel trends assumption.")]
    pub covariates: Option<Vec<String>>,

    /// Comparison group strategy
    #[schemars(
        description = "Comparison group: 'never_treated' (default) uses only never-treated units, 'not_yet_treated' uses units not yet treated by that period."
    )]
    pub comparison_group: Option<String>,

    /// Estimation method
    #[schemars(
        description = "Estimation method: 'outcome_regression' (default), 'ipw', or 'doubly_robust'."
    )]
    pub estimation_method: Option<String>,

    /// Base period relative to treatment
    #[schemars(
        description = "Base period for pre-treatment comparison, relative to g. Default is -1 (one period before treatment)."
    )]
    pub base_period: Option<i32>,

    /// Number of bootstrap replications
    #[schemars(
        description = "Number of bootstrap replications for standard errors. Default is 999."
    )]
    pub bootstrap: Option<usize>,
}

/// Request for Goodman-Bacon decomposition of staggered DiD.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BaconDecompRequest {
    /// Name/ID of the dataset
    #[schemars(
        description = "Name or ID of a previously loaded panel dataset with staggered treatment."
    )]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Unit identifier column
    #[schemars(
        description = "Column containing the unit/individual identifier (e.g., state_id, firm_id)."
    )]
    pub unit_col: String,

    /// Time period column
    #[schemars(
        description = "Column containing the time period identifier (e.g., year, quarter)."
    )]
    pub time_col: String,

    /// Treatment indicator column
    #[schemars(
        description = "Binary treatment indicator column (0 = untreated, 1 = treated). Should be 0 before treatment and 1 after for each unit."
    )]
    pub treatment_col: String,
}

/// Request for Extended Two-Way Fixed Effects (ETWFE) estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EtwfeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Unit identifier column
    #[schemars(
        description = "Column containing the unit/individual identifier (e.g., state_id, firm_id)."
    )]
    pub unit_col: String,

    /// Time period column
    #[schemars(
        description = "Column containing the time period identifier (e.g., year, quarter)."
    )]
    pub time_col: String,

    /// Treatment indicator column
    #[schemars(
        description = "Column indicating treatment status (1 = currently treated, 0 = not treated). Binary indicator."
    )]
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
        description = "Control group: 'notyet' (default) uses not-yet-treated units, 'never' uses only never-treated units."
    )]
    pub cgroup: Option<String>,
}

/// Request for Inverse Probability Weighting (IPW) treatment effect estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IpwRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for propensity score model
    #[schemars(description = "Names of covariate columns to include in propensity score model.")]
    pub covariates: Vec<String>,

    /// Estimand: 'ate' (Average Treatment Effect) or 'att' (Average Treatment Effect on Treated)
    #[schemars(
        description = "Treatment effect estimand: 'ate' for Average Treatment Effect (default), 'att' for Average Treatment Effect on Treated."
    )]
    pub estimand: Option<String>,

    /// Trimming threshold for extreme propensity scores
    #[schemars(
        description = "Trim observations with propensity scores below trim or above 1-trim. Default is 0.05."
    )]
    pub trim: Option<f64>,

    /// Number of bootstrap replications for standard errors
    #[schemars(
        description = "Number of bootstrap replications for standard error estimation. Default is 999."
    )]
    pub bootstrap: Option<usize>,
}

/// Request for Doubly Robust (AIPW) treatment effect estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DoublyRobustRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for propensity score and outcome models
    #[schemars(
        description = "Names of covariate columns to include in both propensity score and outcome models."
    )]
    pub covariates: Vec<String>,

    /// Estimation method: 'aipw' (default), 'ipw', or 'regression'
    #[schemars(
        description = "Estimation method: 'aipw' for Augmented IPW (default, doubly robust), 'ipw' for IPW only, 'regression' for outcome regression only."
    )]
    pub method: Option<String>,

    /// Estimand: 'ate' (Average Treatment Effect) or 'att' (Average Treatment Effect on Treated)
    #[schemars(
        description = "Treatment effect estimand: 'ate' for Average Treatment Effect (default), 'att' for Average Treatment Effect on Treated."
    )]
    pub estimand: Option<String>,

    /// Trimming threshold for extreme propensity scores
    #[schemars(
        description = "Trim observations with propensity scores below trim or above 1-trim. Default is 0.05."
    )]
    pub trim: Option<f64>,

    /// Number of bootstrap replications for standard errors
    #[schemars(
        description = "Number of bootstrap replications for standard error estimation. Default is 999."
    )]
    pub bootstrap: Option<usize>,
}

/// Request for Double/Debiased Machine Learning (DoubleML) estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DoubleMLRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column (Y).")]
    pub outcome: String,

    /// Treatment variable column name
    #[schemars(
        description = "Name of the treatment variable column (D). Can be continuous or binary."
    )]
    pub treatment: String,

    /// Covariate columns for nuisance model estimation
    #[schemars(description = "Names of covariate columns (X) for nuisance model estimation.")]
    pub covariates: Vec<String>,

    /// Model type: 'plr' (Partially Linear Regression, default) or 'irm' (Interactive Regression Model)
    #[schemars(
        description = "DML model type: 'plr' for Partially Linear Regression (default, Y = theta*D + g(X) + eps), 'irm' for Interactive Regression Model (binary treatment, heterogeneous effects)."
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
pub struct CbpsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for propensity score model
    #[schemars(description = "Names of covariate columns to include in propensity score model.")]
    pub covariates: Vec<String>,

    /// CBPS method: 'exact' (default), 'over', or 'just'
    #[schemars(
        description = "CBPS method: 'exact' for exact balance (default, overidentified), 'over' for over-balanced, 'just' for just-identified (standard logit)."
    )]
    pub method: Option<String>,

    /// Standardized difference threshold for balance
    #[schemars(
        description = "Threshold for standardized difference to consider a covariate balanced. Default is 0.1."
    )]
    pub balance_threshold: Option<f64>,
}

/// Request for flexible inverse probability weighting (WeightIt).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct WeightItRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for balance
    #[schemars(description = "Names of covariate columns to balance between treatment groups.")]
    pub covariates: Vec<String>,

    /// Weighting method: 'logistic' (default), 'entropy', 'energy', or 'stable'
    #[schemars(
        description = "Weighting method: 'logistic' (standard PS, default), 'entropy' (entropy balancing), 'energy' (energy distance), 'stable' (stable weights)."
    )]
    pub method: Option<String>,

    /// Target estimand: 'ate' (default), 'att', or 'atc'
    #[schemars(
        description = "Target estimand: 'ate' (average treatment effect, default), 'att' (on treated), 'atc' (on control)."
    )]
    pub estimand: Option<String>,

    /// Whether to stabilize weights
    #[schemars(
        description = "Whether to stabilize weights by multiplying by marginal treatment probability. Default is false."
    )]
    pub stabilize: Option<bool>,

    /// Trimming quantile for extreme weights
    #[schemars(
        description = "Quantile for trimming extreme weights (e.g., 0.99 trims at 1st and 99th percentile). Default is 1.0 (no trimming)."
    )]
    pub trim_quantile: Option<f64>,
}

/// Request for entropy balancing.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EntropyBalanceRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for balance
    #[schemars(description = "Names of covariate columns to balance exactly on means.")]
    pub covariates: Vec<String>,

    /// Optional target means (defaults to treated group means for ATT)
    #[schemars(
        description = "Optional target means for covariates. If not provided, uses treated group means (ATT)."
    )]
    pub target_means: Option<Vec<f64>>,
}

/// Request for stable balancing weights (SBW).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SBWRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for balance
    #[schemars(description = "Names of covariate columns to balance between treatment groups.")]
    pub covariates: Vec<String>,

    /// Target estimand: 'att' (default), 'ate', or 'atc'
    #[schemars(
        description = "Target estimand: 'att' (effect on treated, default), 'ate' (average treatment effect), 'atc' (effect on control)."
    )]
    pub estimand: Option<String>,

    /// Balance tolerance for approximate balance (0 = exact balance)
    #[schemars(
        description = "Tolerance for approximate balance. 0 means exact balance (default), positive values allow some deviation."
    )]
    pub balance_tol: Option<f64>,

    /// Minimum weight allowed (default 0 for non-negativity)
    #[schemars(description = "Minimum weight allowed. Default is 0 (non-negativity constraint).")]
    pub min_weight: Option<f64>,

    /// Penalty parameter for approximate balance (higher = stricter balance)
    #[schemars(
        description = "Penalty parameter for approximate balance. Higher values enforce stricter balance. Default is 1000."
    )]
    pub balance_penalty: Option<f64>,
}

/// Request for twang GBM propensity score estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TwangRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for propensity model
    #[schemars(
        description = "Names of covariate columns to include in the GBM propensity score model."
    )]
    pub covariates: Vec<String>,

    /// Stopping rule: 'es.mean' (default), 'es.max', 'ks.mean', 'ks.max'
    #[schemars(
        description = "Stopping rule for selecting optimal iterations: 'es.mean' (mean standardized effect size, default), 'es.max' (max effect size), 'ks.mean' (mean KS statistic), 'ks.max' (max KS statistic)."
    )]
    pub stop_method: Option<String>,

    /// Target estimand: 'att' (default), 'ate', or 'atc'
    #[schemars(
        description = "Target estimand: 'att' (effect on treated, default), 'ate' (average treatment effect), 'atc' (effect on control)."
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
pub struct MatchItRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for matching
    #[schemars(description = "Names of covariate columns to use for matching.")]
    pub covariates: Vec<String>,

    /// Matching method: 'nearest' (default), 'cem', 'full', or 'subclass'
    #[schemars(
        description = "Matching method: 'nearest' (nearest neighbor, default), 'cem' (coarsened exact matching), 'full' (full/optimal matching), 'subclass' (propensity score subclassification)."
    )]
    pub method: Option<String>,

    /// Distance metric: 'logit' (default), 'probit', 'mahalanobis', or 'euclidean'
    #[schemars(
        description = "Distance metric: 'logit' (propensity score via logit, default), 'probit', 'mahalanobis', 'euclidean'."
    )]
    pub distance: Option<String>,

    /// Matching ratio (1:k matching, default k=1)
    #[schemars(
        description = "For nearest neighbor: number of controls per treated unit (1:k matching). Default is 1."
    )]
    pub ratio: Option<usize>,

    /// Caliper width (in SD of propensity score)
    #[schemars(
        description = "For nearest neighbor: maximum distance for a valid match, in SD of propensity score. Default is no caliper."
    )]
    pub caliper: Option<f64>,

    /// Whether to sample with replacement
    #[schemars(
        description = "For nearest neighbor: whether to sample controls with replacement. Default is false."
    )]
    pub replace: Option<bool>,

    /// Number of subclasses for subclassification
    #[schemars(
        description = "For subclassification: number of subclasses to create. Default is 5."
    )]
    pub n_subclasses: Option<usize>,
}

/// Request for Targeted Maximum Likelihood Estimation (TMLE).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TmleRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column. Can be binary or continuous.")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for propensity score and outcome models
    #[schemars(
        description = "Names of covariate columns to include in both propensity score and outcome models."
    )]
    pub covariates: Vec<String>,

    /// Outcome model type: 'logistic' (default) or 'linear'
    #[schemars(
        description = "Outcome model type: 'logistic' for binary outcomes (default), 'linear' for continuous outcomes."
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
pub struct CTmleRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column. Can be binary or continuous.")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns (candidates for propensity score selection)
    #[schemars(
        description = "Names of candidate covariate columns. C-TMLE will select which ones to include in the propensity score model via cross-validation."
    )]
    pub covariates: Vec<String>,

    /// Outcome model type: 'logistic' (default) or 'linear'
    #[schemars(
        description = "Outcome model type: 'logistic' for binary outcomes (default), 'linear' for continuous outcomes."
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
        description = "Stopping rule for selection: 'cv_minimum' (stop at minimum CV risk, default), 'one_se' (one-standard-error rule for parsimony), 'max_covariates' (use max_covariates parameter)."
    )]
    pub stopping_rule: Option<String>,

    /// Lower bound for propensity score truncation (default: 0.025)
    #[schemars(description = "Lower bound for propensity score truncation. Default is 0.025.")]
    pub ps_lower: Option<f64>,

    /// Upper bound for propensity score truncation (default: 0.975)
    #[schemars(description = "Upper bound for propensity score truncation. Default is 0.975.")]
    pub ps_upper: Option<f64>,
}

/// Request for Longitudinal Targeted Maximum Likelihood Estimation (LTMLE).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LtmleRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome column names at each time point (chronological order)
    #[schemars(
        description = "Names of outcome variable columns at each time point. For survival outcomes, only the last time point may have the actual outcome; earlier can be zeros."
    )]
    pub outcomes: Vec<String>,

    /// Treatment column names at each time point (chronological order)
    #[schemars(
        description = "Names of treatment indicator columns at each time point. Must be binary (0 or 1)."
    )]
    pub treatments: Vec<String>,

    /// Covariate column names at each time point (chronological order, comma-separated within each time point)
    #[schemars(
        description = "Names of covariate columns at each time point. Each element is a comma-separated list of column names for that time point."
    )]
    pub covariates: Vec<String>,

    /// Outcome model type: 'linear' (default) or 'logistic'
    #[schemars(
        description = "Outcome model type: 'linear' for continuous outcomes (default), 'logistic' for binary outcomes."
    )]
    pub q_model: Option<String>,

    /// Lower bound for propensity score truncation
    #[schemars(description = "Lower bound for propensity score truncation. Default is 0.01.")]
    pub ps_lower: Option<f64>,

    /// Upper bound for propensity score truncation
    #[schemars(description = "Upper bound for propensity score truncation. Default is 0.99.")]
    pub ps_upper: Option<f64>,
}

/// Request for Regression Standardization (G-computation).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StdRegRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column. Can be continuous or binary.")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
    pub treatment: String,

    /// Covariate columns for the outcome model
    #[schemars(description = "Names of covariate columns to include in the outcome model.")]
    pub covariates: Vec<String>,

    /// Outcome model type: 'linear' (default), 'logistic', or 'poisson'
    #[schemars(
        description = "Outcome model type: 'linear' for continuous outcomes (default), 'logistic' for binary outcomes, 'poisson' for count outcomes."
    )]
    pub model_type: Option<String>,

    /// Estimand: 'ate' (default), 'att', 'atc', or 'levels'
    #[schemars(
        description = "Target estimand: 'ate' (Average Treatment Effect, default), 'att' (ATT on Treated), 'atc' (ATC on Controls), 'levels' (E[Y(1)] and E[Y(0)] separately)."
    )]
    pub estimand: Option<String>,

    /// SE method: 'bootstrap' (default), 'delta', or 'sandwich'
    #[schemars(
        description = "Standard error method: 'bootstrap' (default, recommended), 'delta' (analytical), 'sandwich' (robust)."
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
pub struct GFormulaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name (observed at final time point)
    #[schemars(description = "Name of the outcome variable column.")]
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
        description = "Intervention type: 'always_treat' (default), 'never_treat', 'natural' (observed patterns), or 'threshold'."
    )]
    pub intervention: Option<String>,

    /// For threshold intervention: variable index (0-indexed into time-varying covariates)
    #[schemars(
        description = "For threshold intervention: index of the time-varying covariate to check (0-indexed)."
    )]
    pub threshold_variable: Option<usize>,

    /// For threshold intervention: cutoff value
    #[schemars(
        description = "For threshold intervention: threshold value for treatment decision."
    )]
    pub threshold_cutoff: Option<f64>,

    /// For threshold intervention: treat if above (true) or below (false)
    #[schemars(
        description = "For threshold intervention: if true, treat when variable > cutoff; if false, treat when variable <= cutoff."
    )]
    pub threshold_above: Option<bool>,

    /// Outcome type: 'continuous' (default), 'binary', or 'survival'
    #[schemars(
        description = "Outcome type: 'continuous' for linear model (default), 'binary' for logistic model, 'survival' for discrete hazard model."
    )]
    pub outcome_type: Option<String>,

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
pub struct MediationRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary."
    )]
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
        description = "Number of bootstrap replications for standard error estimation. Default is 999."
    )]
    pub bootstrap: Option<usize>,
}

/// Request for Natural Effect Models (medflex) mediation analysis.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct NaturalEffectsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column (continuous).")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(
        description = "Column indicating treatment status (1 = treated, 0 = control). Typically binary."
    )]
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
        description = "Number of bootstrap replications for confidence intervals. Default is 1000. Set to 0 to use delta method instead."
    )]
    pub n_bootstrap: Option<usize>,

    /// Confidence level for intervals
    #[schemars(
        description = "Confidence level for intervals (e.g., 0.95 for 95% CI). Default is 0.95."
    )]
    pub confidence_level: Option<f64>,

    /// Effect scale (difference, ratio, odds_ratio)
    #[schemars(
        description = "Scale for reporting effects: 'difference' (default for continuous outcomes), 'ratio' (for log-link), 'odds_ratio' (for logit)."
    )]
    pub scale: Option<String>,
}

/// Request for Synthetic Control Method.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SyntheticControlRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Unit identifier column name
    #[schemars(description = "Name of the column identifying units (e.g., 'state', 'country').")]
    pub unit_col: String,

    /// Time period column name
    #[schemars(
        description = "Name of the column identifying time periods (must be integer, e.g., 'year')."
    )]
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
        description = "Method for predictor importance weights: 'datadriven' (default), 'equal', or 'custom'."
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
pub struct GsynthRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column (Y).")]
    pub outcome: String,

    /// Treatment indicator column name
    #[schemars(
        description = "Name of the treatment indicator column (D, binary 0/1). Treatment can start at different times for different units."
    )]
    pub treatment: String,

    /// Unit identifier column name
    #[schemars(description = "Name of the column identifying units (e.g., 'state', 'country').")]
    pub unit_col: String,

    /// Time period column name
    #[schemars(description = "Name of the column identifying time periods (must be numeric).")]
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
        description = "Estimator: 'ife' (interactive fixed effects, default) or 'mc' (matrix completion)."
    )]
    pub estimator: Option<String>,

    /// Fixed effects specification
    #[schemars(description = "Fixed effects: 'none', 'unit' (default), 'time', or 'twoWay'.")]
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
pub struct ScpiRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded panel dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column (Y).")]
    pub outcome: String,

    /// Unit identifier column name
    #[schemars(description = "Name of the column identifying units (e.g., 'state', 'country').")]
    pub unit_col: String,

    /// Time period column name
    #[schemars(description = "Name of the column identifying time periods (must be numeric).")]
    pub time_col: String,

    /// Treated unit identifier
    #[schemars(description = "Identifier of the treated unit (must match a value in unit_col).")]
    pub treated_unit: String,

    /// Treatment time period
    #[schemars(description = "First post-treatment time period (treatment starts at this time).")]
    pub treatment_time: i64,

    /// Constraint type
    #[schemars(
        description = "Weight constraint: 'simplex' (default, sum=1, non-negative), 'lasso', 'ridge', or 'lasso_simplex'."
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
        description = "Out-of-sample variance method: 'subgaussian' (default, more conservative), 'gaussian', 'loo_cv', or 'kfold_cv'."
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
pub struct RdEstimateRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable (Y) column.")]
    pub outcome: String,

    /// Running variable column name
    #[schemars(description = "Name of the running (forcing) variable (X) column.")]
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
        description = "Bandwidth selection: 'mserd' (MSE-optimal, default), 'msetwo' (separate left/right), 'cerrd', or 'certwo'."
    )]
    pub bwselect: Option<String>,

    /// Main bandwidth (overrides automatic selection)
    #[schemars(
        description = "Main bandwidth h for estimation. If not specified, uses automatic MSE-optimal selection."
    )]
    pub h: Option<f64>,

    /// Bias bandwidth (overrides automatic selection)
    #[schemars(description = "Bias bandwidth b. Default is rho * h where rho = 1.")]
    pub b: Option<f64>,

    /// Confidence level
    #[schemars(description = "Confidence level for intervals. Default is 0.95.")]
    pub level: Option<f64>,
}

/// Request for RD bandwidth selection only.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RdBandwidthRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable (Y) column.")]
    pub outcome: String,

    /// Running variable column name
    #[schemars(description = "Name of the running (forcing) variable (X) column.")]
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
        description = "Bandwidth selection: 'mserd' (MSE-optimal, default), 'msetwo', 'cerrd', or 'certwo'."
    )]
    pub bwselect: Option<String>,
}

/// Request for Fuzzy Regression Discontinuity estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FuzzyRdRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable (Y) column.")]
    pub outcome: String,

    /// Running variable column name
    #[schemars(description = "Name of the running (forcing) variable (X) column.")]
    pub running_var: String,

    /// Treatment indicator column name
    #[schemars(
        description = "Name of the treatment indicator column (actual treatment received, 0/1)."
    )]
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
        description = "Bandwidth selection: 'mserd' (default), 'msetwo', 'cerrd', or 'certwo'."
    )]
    pub bwselect: Option<String>,

    /// Main bandwidth
    #[schemars(description = "Main bandwidth h. If not specified, uses automatic selection.")]
    pub h: Option<f64>,

    /// Confidence level
    #[schemars(description = "Confidence level for intervals. Default is 0.95.")]
    pub level: Option<f64>,
}

/// Request for Multi-Cutoff Regression Discontinuity estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RdMultiRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable (Y) column.")]
    pub outcome: String,

    /// Running variable column name
    #[schemars(description = "Name of the running (forcing) variable (X) column.")]
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
        description = "Weighting scheme for pooling: 'sample_size' (default), 'inverse_variance', or 'equal'."
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
}

/// Request for Causal Forest estimation (Wager & Athey 2018).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CausalForestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome column name
    #[schemars(description = "Name of the outcome variable (Y).")]
    pub outcome: String,

    /// Treatment column name
    #[schemars(description = "Name of the binary treatment variable (W). Must be 0/1.")]
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
pub struct BartCausalRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome column name
    #[schemars(description = "Name of the outcome variable (Y).")]
    pub outcome: String,

    /// Treatment column name
    #[schemars(description = "Name of the binary treatment variable (W). Must be 0/1.")]
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
pub struct HetTxRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome column name
    #[schemars(description = "Name of the outcome variable (Y).")]
    pub outcome: String,

    /// Treatment column name
    #[schemars(description = "Name of the binary treatment indicator (0/1).")]
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
        description = "Model type for marginal effects: 'ols' (linear regression), 'logit' (logistic regression), 'probit' (probit regression). Default: 'ols'."
    )]
    pub model: Option<String>,
}

/// Request for E-value sensitivity analysis for unmeasured confounding.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EValueRequest {
    /// Type of effect measure
    #[schemars(
        description = "Type of effect measure: 'rr' (risk ratio), 'or' (odds ratio), 'hr' (hazard ratio), 'smd' (standardized mean difference), 'rd' (risk difference)."
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
