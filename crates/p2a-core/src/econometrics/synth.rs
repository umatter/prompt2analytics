//! Synthetic Control Method for comparative case studies.
//!
//! Pure Rust implementation of the synthetic control method as developed by
//! Abadie, Diamond, and Hainmueller (2010). This method constructs a weighted
//! combination of control units to estimate the counterfactual outcome for a
//! treated unit.
//!
//! # References
//!
//! - Abadie, A. & Gardeazabal, J. (2003). "The Economic Costs of Conflict: A Case Study
//!   of the Basque Country." *American Economic Review*, 93(1), 112-132.
//! - Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for
//!   Comparative Case Studies: Estimating the Effect of California's Tobacco Control
//!   Program." *Journal of the American Statistical Association*, 105(490), 493-505.
//! - Abadie, A. (2021). "Using Synthetic Controls: Feasibility, Data Requirements, and
//!   Methodological Aspects." *Journal of Economic Literature*, 59(2), 391-425.
//!
//! Implementation inspired by:
//! - R package `Synth` (Abadie, Diamond, Hainmueller)
//!   Source: <https://cran.r-project.org/package=Synth>
//! - R package `tidysynth` (Eric Dunford)
//!   Source: <https://cran.r-project.org/package=tidysynth>

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, NaPolicy, numeric_options};
use crate::errors::{EconError, EconResult};
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};
use crate::traits::estimator::SignificanceLevel;

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Method for optimizing the V matrix (predictor importance weights).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum VOptimization {
    /// Data-driven: minimize pre-treatment MSPE using Nelder-Mead
    #[default]
    DataDriven,
    /// Equal weights for all predictors
    Equal,
    /// User-specified weights (normalized to sum to 1)
    Custom(Vec<f64>),
}

/// How to aggregate predictor values over time.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
pub enum TimeAggregation {
    /// Use the mean over the time window
    #[default]
    Mean,
    /// Use the first observation in the time window
    First,
    /// Use the last observation in the time window
    Last,
    /// Use the sum over the time window
    Sum,
}

crate::option_enum!(
    TimeAggregation,
    crate::options::Choices {
        option: "predictor aggregation",
        values: &[
            (&["mean", "average"], TimeAggregation::Mean),
            (&["first"], TimeAggregation::First),
            (&["last"], TimeAggregation::Last),
            (&["sum", "total"], TimeAggregation::Sum),
        ],
    }
);

/// The spellings of the V methods (`VOptimization::from_options`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VMethodName {
    DataDriven,
    Equal,
    Custom,
}

crate::option_enum!(
    VMethodName,
    crate::options::Choices {
        option: "V method",
        values: &[
            (
                &["datadriven", "data_driven", "optimal", "nested"],
                VMethodName::DataDriven,
            ),
            (&["equal", "uniform"], VMethodName::Equal),
            (&["custom", "user"], VMethodName::Custom),
        ],
    },
    no_display
);

impl VOptimization {
    /// MCP/CLI spellings: "datadriven" (aliases "data_driven", "optimal", "nested"; the
    /// default when `None`), "equal" ("uniform"), "custom" ("user"; needs `custom_weights`).
    /// Unknown names and a custom method without weights are `InvalidSpecification` listing
    /// the accepted names. Custom weights are validated (and normalised) by the estimator.
    pub fn from_options(
        method: Option<&str>,
        custom_weights: Option<Vec<f64>>,
    ) -> EconResult<VOptimization> {
        match crate::options::parse_or(method, VMethodName::DataDriven)? {
            VMethodName::DataDriven => Ok(VOptimization::DataDriven),
            VMethodName::Equal => Ok(VOptimization::Equal),
            VMethodName::Custom => match custom_weights {
                Some(w) => Ok(VOptimization::Custom(w)),
                None => Err(EconError::InvalidSpecification {
                    message: format!(
                        "V method 'custom' needs custom V weights (custom_v_weights);                          expected one of: {}",
                        VMethodName::choices().help()
                    ),
                }),
            },
        }
    }
}

/// Specification for a predictor variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorSpec {
    /// Column name of the predictor
    pub column: String,
    /// How to aggregate over time
    pub aggregation: TimeAggregation,
    /// Optional time window (start, end) - if None, uses all pre-treatment periods
    pub time_window: Option<(i64, i64)>,
}

impl PredictorSpec {
    /// Create a new predictor specification with mean aggregation.
    pub fn new(column: &str) -> Self {
        Self {
            column: column.to_string(),
            aggregation: TimeAggregation::Mean,
            time_window: None,
        }
    }

    /// Create a predictor using mean over a specific time window.
    pub fn with_window(column: &str, start: i64, end: i64) -> Self {
        Self {
            column: column.to_string(),
            aggregation: TimeAggregation::Mean,
            time_window: Some((start, end)),
        }
    }
}

/// Configuration for synthetic control estimation.
#[derive(Debug, Clone)]
pub struct SynthConfig {
    /// Time period when treatment begins (first post-treatment period)
    pub treatment_time: i64,
    /// Unit identifier of the treated unit
    pub treated_unit: String,
    /// Optional time window for optimization (start, end)
    /// If None, uses all pre-treatment periods
    pub optimization_window: Option<(i64, i64)>,
    /// Method for V matrix optimization
    pub v_method: VOptimization,
    /// Tolerance for optimization convergence
    pub tolerance: f64,
    /// Maximum iterations for V optimization
    pub max_iter: usize,
    /// Whether to run placebo tests for inference
    pub run_placebos: bool,
    /// Minimum weight to report (for sparsity in output)
    pub weight_threshold: f64,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            treatment_time: 0,
            treated_unit: String::new(),
            optimization_window: None,
            v_method: VOptimization::DataDriven,
            tolerance: 1e-6,
            max_iter: 1000,
            run_placebos: false,
            weight_threshold: 0.001,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Balance statistics for a predictor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorBalance {
    /// Predictor name
    pub predictor: String,
    /// Value for the treated unit
    pub treated_value: f64,
    /// Value for the synthetic control
    pub synthetic_value: f64,
    /// Absolute difference
    pub difference: f64,
    /// Percentage difference
    pub percent_diff: f64,
}

/// Treatment effect at a specific time period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeEffect {
    /// Time period
    pub time: i64,
    /// Estimated treatment effect (actual - synthetic)
    pub effect: f64,
    /// Actual outcome for treated unit
    pub actual: f64,
    /// Synthetic control outcome
    pub synthetic: f64,
}

/// Results from placebo (permutation) inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaceboResults {
    /// RMSPE ratios for all units (unit_name, ratio)
    pub rmspe_ratios: Vec<(String, f64)>,
    /// Rank of the treated unit (1 = highest ratio)
    pub treated_rank: usize,
    /// Exact p-value (rank / n_units)
    pub p_value: f64,
    /// Number of units used in placebo tests
    pub n_units: usize,
    /// Significance level
    pub significance: SignificanceLevel,
}

/// Result from synthetic control estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthResult {
    // Basic info
    /// Name/ID of the treated unit
    pub treated_unit: String,
    /// Time period when treatment began
    pub treatment_time: i64,
    /// Number of donor units in the pool
    pub n_donors: usize,
    /// Number of pre-treatment periods
    pub n_pre_periods: usize,
    /// Number of post-treatment periods
    pub n_post_periods: usize,

    // Weights
    /// Unit weights (unit_name, weight) - only non-zero weights above threshold
    pub unit_weights: Vec<(String, f64)>,
    /// All unit weights including zeros
    #[serde(skip)]
    pub all_unit_weights: Vec<(String, f64)>,
    /// Predictor weights (V diagonal)
    pub predictor_weights: Vec<(String, f64)>,

    // Fit diagnostics
    /// Predictor balance comparison
    pub predictor_balance: Vec<PredictorBalance>,
    /// Pre-treatment MSPE (mean squared prediction error)
    pub pre_treatment_mspe: f64,
    /// Pre-treatment RMSPE (root mean squared prediction error)
    pub pre_treatment_rmspe: f64,

    // Treatment effects
    /// Effect at each post-treatment period
    pub treatment_effects: Vec<TimeEffect>,
    /// Average treatment effect over all post-periods
    pub average_effect: f64,
    /// Cumulative treatment effect
    pub cumulative_effect: f64,

    // Time series data
    /// Actual outcomes for treated unit (time, value)
    pub actual_outcome: Vec<(i64, f64)>,
    /// Synthetic control outcomes (time, value)
    pub synthetic_outcome: Vec<(i64, f64)>,

    // Inference
    /// Placebo test results (if run_placebos = true)
    pub placebo_results: Option<PlaceboResults>,

    // Diagnostics
    /// Number of V optimization iterations
    pub v_iterations: usize,
    /// Final loss value from optimization: the pre-treatment MSPE of the reported weights
    pub final_loss: f64,
    /// Any warnings generated
    pub warnings: Vec<String>,
    /// `(X1 - X0 W)' V (X1 - X0 W)` at the reported V and W on the standardised predictors
    /// (Synth's `loss.w`; the QP objective without the tie-breaking ridge).
    #[serde(default)]
    pub predictor_loss: f64,
    /// The Nelder-Mead V search of the start that was kept (data-driven V only).
    #[serde(default)]
    pub convergence: Option<Convergence>,
    /// Final pre-treatment MSPE of each V-search start (equal V, then the regression-based
    /// V); empty for equal and custom V.
    #[serde(default)]
    pub v_start_losses: Vec<f64>,
}

impl ResultNotes for SynthResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        self.convergence.clone()
    }
}

impl fmt::Display for SynthResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Synthetic Control Method Results")?;
        writeln!(f, "=================================")?;
        writeln!(f)?;
        writeln!(f, "Treated Unit: {}", self.treated_unit)?;
        writeln!(f, "Treatment Time: {}", self.treatment_time)?;
        writeln!(f, "Donor Pool Size: {} units", self.n_donors)?;
        writeln!(f, "Pre-Treatment Periods: {}", self.n_pre_periods)?;
        writeln!(f, "Post-Treatment Periods: {}", self.n_post_periods)?;
        writeln!(f)?;

        // Unit weights
        writeln!(f, "SYNTHETIC CONTROL WEIGHTS")?;
        writeln!(f, "-------------------------")?;
        for (unit, weight) in &self.unit_weights {
            writeln!(f, "  {:<25} {:>8.4}", unit, weight)?;
        }
        writeln!(f)?;

        // Predictor weights
        writeln!(f, "PREDICTOR WEIGHTS (V; standardised predictors)")?;
        writeln!(f, "----------------------------------------------")?;
        for (name, weight) in &self.predictor_weights {
            writeln!(f, "  {:<25} {:>8.4}", name, weight)?;
        }
        if self.v_start_losses.len() == 2 {
            writeln!(
                f,
                "  V search: 2 starts (equal, regression), losses {:.6e}, {:.6e}",
                self.v_start_losses[0], self.v_start_losses[1]
            )?;
        }
        writeln!(f)?;

        // Predictor balance
        writeln!(f, "PREDICTOR BALANCE")?;
        writeln!(
            f,
            "{:<25} {:>12} {:>12} {:>12}",
            "Predictor", "Treated", "Synthetic", "Diff %"
        )?;
        writeln!(f, "{}", "-".repeat(63))?;
        for pb in &self.predictor_balance {
            writeln!(
                f,
                "{:<25} {:>12.4} {:>12.4} {:>12.2}%",
                pb.predictor, pb.treated_value, pb.synthetic_value, pb.percent_diff
            )?;
        }
        writeln!(f)?;

        // Fit statistics
        writeln!(f, "PRE-TREATMENT FIT")?;
        writeln!(f, "  MSPE:  {:.6}", self.pre_treatment_mspe)?;
        writeln!(f, "  RMSPE: {:.6}", self.pre_treatment_rmspe)?;
        writeln!(f, "  Predictor loss: {:.6e}", self.predictor_loss)?;
        writeln!(f)?;

        // Treatment effects
        writeln!(f, "TREATMENT EFFECTS")?;
        writeln!(f, "  Average Effect: {:.4}", self.average_effect)?;
        writeln!(f, "  Cumulative Effect: {:.4}", self.cumulative_effect)?;
        writeln!(f)?;

        writeln!(
            f,
            "{:<10} {:>12} {:>12} {:>12}",
            "Time", "Actual", "Synthetic", "Effect"
        )?;
        writeln!(f, "{}", "-".repeat(50))?;
        for te in &self.treatment_effects {
            writeln!(
                f,
                "{:<10} {:>12.4} {:>12.4} {:>12.4}",
                te.time, te.actual, te.synthetic, te.effect
            )?;
        }

        // Placebo inference
        if let Some(ref placebo) = self.placebo_results {
            writeln!(f)?;
            writeln!(f, "PLACEBO INFERENCE")?;
            writeln!(
                f,
                "  Treated Unit Rank: {} / {}",
                placebo.treated_rank, placebo.n_units
            )?;
            writeln!(
                f,
                "  Exact P-Value: {:.4}{}",
                placebo.p_value,
                placebo.significance.stars()
            )?;
        }

        let notes = notes_text(self);
        if !notes.is_empty() {
            writeln!(f)?;
            write!(f, "{notes}")?;
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal Data Structure
// ═══════════════════════════════════════════════════════════════════════════════

/// Prepared data for synthetic control optimization.
struct SynthData {
    /// Treated unit predictor values (k × 1)
    x1: Array1<f64>,
    /// Donor predictor values (k × J)
    x0: Array2<f64>,
    /// Treated pre-treatment outcomes (T0 × 1)
    z1: Array1<f64>,
    /// Donor pre-treatment outcomes (T0 × J)
    z0: Array2<f64>,
    /// Treated post-treatment outcomes
    y1_post: Array1<f64>,
    /// Donor post-treatment outcomes (T_post × J)
    y0_post: Array2<f64>,
    /// Donor unit names
    donor_units: Vec<String>,
    /// Predictor names
    predictor_names: Vec<String>,
    /// Pre-treatment time periods
    pre_times: Vec<i64>,
    /// Post-treatment time periods
    post_times: Vec<i64>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Function
// ═══════════════════════════════════════════════════════════════════════════════

/// Run the Synthetic Control Method.
///
/// Constructs a weighted combination of control (donor) units to estimate
/// the counterfactual outcome for a treated unit. The weights are chosen
/// to minimize the distance between the treated unit and synthetic control
/// on pre-treatment characteristics.
///
/// # Arguments
/// * `dataset` - Panel dataset with outcome, unit, and time columns
/// * `outcome` - Name of the outcome variable column
/// * `unit_col` - Name of the unit identifier column
/// * `time_col` - Name of the time period column
/// * `predictors` - Specifications for predictor variables
/// * `config` - Configuration options
///
/// # Mathematical Model
///
/// The synthetic control weights W* minimize:
/// ```text
/// ||X₁ - X₀W||_V = √[(X₁ - X₀W)' V (X₁ - X₀W)]
/// ```
/// Subject to: w_j ≥ 0, Σw_j = 1
///
/// Where V is chosen to minimize pre-treatment MSPE:
/// ```text
/// MSPE = (1/T₀) Σₜ (Y₁ₜ - Σⱼ wⱼ*Yⱼₜ)²
/// ```
///
/// # References
///
/// - Abadie, Diamond, Hainmueller (2010), JASA 105(490), 493-505.
/// - Equation numbers reference that paper.
///
/// # Data requirements
///
/// One row per (unit, time): a duplicate (unit, time) row is an error, and so is a
/// missing (null) unit or time label. The outcome must be present in every
/// optimisation-window and post-treatment cell of the treated unit and the donors; a
/// missing value there is named ("Outcome 'y' is missing (null or NaN) for unit 'B' at
/// time 2."). Predictors are aggregated over the values present in their window. Integer
/// outcomes and predictors are read as numbers; unit ids may be text, integers or floats
/// (`treated_unit` is matched against their label, e.g. "3"), times integers,
/// whole-number floats or dates. Donors are in sorted order (R `factor()` order).
///
/// # Example
/// ```ignore
/// let predictors = vec![
///     PredictorSpec::new("gdp"),
///     PredictorSpec::with_window("population", 1980, 1990),
/// ];
/// let config = SynthConfig {
///     treatment_time: 1990,
///     treated_unit: "California".to_string(),
///     run_placebos: true,
///     ..Default::default()
/// };
/// let result = run_synthetic_control(&dataset, "cigsale", "state", "year", &predictors, config)?;
/// ```
pub fn run_synthetic_control(
    dataset: &Dataset,
    outcome: &str,
    unit_col: &str,
    time_col: &str,
    predictors: &[PredictorSpec],
    config: SynthConfig,
) -> EconResult<SynthResult> {
    let mut warnings = Vec::new();

    // Validate inputs
    if config.treated_unit.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "treated_unit must be specified in config".to_string(),
        });
    }

    if predictors.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "At least one predictor must be specified".to_string(),
        });
    }

    // Nelder-Mead stops only on a spread below the tolerance, so a zero or
    // negative tolerance would always run all max_iter iterations.
    let tol = config.tolerance;
    if !(tol.is_finite() && tol > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("tolerance must be a finite number greater than 0 (got {tol})"),
        });
    }
    if config.max_iter == 0 {
        return Err(EconError::InvalidSpecification {
            message: "max_iter must be at least 1 (got 0)".to_string(),
        });
    }

    // Prepare data matrices
    let synth_data = prepare_synth_data(
        dataset,
        outcome,
        unit_col,
        time_col,
        predictors,
        &config.treated_unit,
        config.treatment_time,
        config.optimization_window,
    )?;

    if synth_data.donor_units.is_empty() {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: 0,
            context: "No donor units available".to_string(),
        });
    }

    if synth_data.z1.len() < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: synth_data.z1.len(),
            context: "Insufficient pre-treatment periods".to_string(),
        });
    }

    // Optimize V and W
    let fit = optimize_synth_weights(
        &synth_data,
        &config.v_method,
        config.tolerance,
        config.max_iter,
    )?;
    if let Some(note) = fit
        .convergence
        .as_ref()
        .and_then(|c| c.note("synthetic-control V search"))
    {
        warnings.push(note);
    }
    if !fit.w.unique {
        warnings.push(format!(
            "the synthetic-control weights are not unique ({} donors fit the predictors \
             equally well); the minimum-norm weights are reported",
            fit.w.free.len()
        ));
    }
    let v_weights = &fit.v;
    let w_weights = &fit.w.weights;

    // Calculate pre-treatment fit
    let (pre_mspe, pre_rmspe) =
        calculate_pre_treatment_fit(&synth_data.z1, &synth_data.z0, w_weights);

    // Calculate predictor balance
    let predictor_balance = calculate_predictor_balance(
        &synth_data.x1,
        &synth_data.x0,
        w_weights,
        &synth_data.predictor_names,
    );

    // Calculate treatment effects
    let treatment_effects = calculate_treatment_effects(&synth_data, w_weights);

    let average_effect = if !treatment_effects.is_empty() {
        treatment_effects.iter().map(|te| te.effect).sum::<f64>() / treatment_effects.len() as f64
    } else {
        0.0
    };

    let cumulative_effect: f64 = treatment_effects.iter().map(|te| te.effect).sum();

    // Build time series data
    let (actual_outcome, synthetic_outcome) = build_time_series(&synth_data, w_weights);

    // Format weights for output
    let all_unit_weights: Vec<(String, f64)> = synth_data
        .donor_units
        .iter()
        .zip(w_weights.iter())
        .map(|(name, &w)| (name.clone(), w))
        .collect();

    let unit_weights: Vec<(String, f64)> = all_unit_weights
        .iter()
        .filter(|(_, w)| *w >= config.weight_threshold)
        .cloned()
        .collect();

    let predictor_weights: Vec<(String, f64)> = synth_data
        .predictor_names
        .iter()
        .zip(v_weights.iter())
        .map(|(name, &v)| (name.clone(), v))
        .collect();

    // Check weight concentration
    let max_weight = w_weights.iter().cloned().fold(0.0_f64, f64::max);
    if max_weight > 0.9 {
        warnings.push(format!(
            "High weight concentration: one unit has {:.1}% of weight",
            max_weight * 100.0
        ));
    }

    let n_nonzero_weights = w_weights.iter().filter(|&&w| w > 0.001).count();
    if n_nonzero_weights == 1 {
        warnings.push("Only one donor unit has significant weight".to_string());
    }

    // Run placebo tests if requested
    let placebo_results = if config.run_placebos {
        match run_placebo_inference(
            dataset, outcome, unit_col, time_col, predictors, &config, pre_rmspe,
        ) {
            Ok(results) => Some(results),
            Err(e @ EconError::Cancelled(_)) => return Err(e),
            Err(e) => {
                warnings.push(format!("Placebo tests failed: {}", e));
                None
            }
        }
    } else {
        None
    };

    Ok(SynthResult {
        treated_unit: config.treated_unit.clone(),
        treatment_time: config.treatment_time,
        n_donors: synth_data.donor_units.len(),
        n_pre_periods: synth_data.pre_times.len(),
        n_post_periods: synth_data.post_times.len(),
        unit_weights,
        all_unit_weights,
        predictor_weights,
        predictor_balance,
        pre_treatment_mspe: pre_mspe,
        pre_treatment_rmspe: pre_rmspe,
        treatment_effects,
        average_effect,
        cumulative_effect,
        actual_outcome,
        synthetic_outcome,
        placebo_results,
        v_iterations: fit.iterations,
        final_loss: fit.final_loss,
        warnings,
        predictor_loss: fit.predictor_loss,
        convergence: fit.convergence,
        v_start_losses: fit.v_start_losses,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Data Preparation
// ═══════════════════════════════════════════════════════════════════════════════

/// Prepare panel data for synthetic control optimization.
fn prepare_synth_data(
    dataset: &Dataset,
    outcome: &str,
    unit_col: &str,
    time_col: &str,
    predictors: &[PredictorSpec],
    treated_unit: &str,
    treatment_time: i64,
    optimization_window: Option<(i64, i64)>,
) -> EconResult<SynthData> {
    let df = dataset.df();

    // The (unit, time) index, read once: units in R factor() order, missing unit or
    // time labels and duplicate cells refused.
    let panel = read_synth_panel(df, unit_col, time_col)?;
    let times = &panel.times;

    // Validate treated unit exists
    let Some(treated) = panel.units.iter().position(|u| u == treated_unit) else {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treated unit '{}' not found in data. Available units: {:?}",
                treated_unit,
                panel.units.iter().take(10).collect::<Vec<_>>()
            ),
        });
    };

    // Split into pre and post treatment periods
    let pre_times: Vec<i64> = times
        .iter()
        .copied()
        .filter(|&t| t < treatment_time)
        .collect();
    let post_times: Vec<i64> = times
        .iter()
        .copied()
        .filter(|&t| t >= treatment_time)
        .collect();

    if pre_times.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "No pre-treatment periods found. Treatment time {} but earliest data is {}",
                treatment_time,
                times.first().unwrap_or(&0)
            ),
        });
    }

    // Donor units: every unit but the treated one, in level order
    let donors: Vec<usize> = (0..panel.units.len()).filter(|&u| u != treated).collect();
    let donor_units: Vec<String> = donors.iter().map(|&u| panel.units[u].clone()).collect();

    // Determine optimization window
    let opt_times: Vec<i64> = match optimization_window {
        Some((start, end)) => pre_times
            .iter()
            .copied()
            .filter(|&t| t >= start && t <= end)
            .collect(),
        None => pre_times.clone(),
    };

    if opt_times.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "Optimization window contains no pre-treatment periods".to_string(),
        });
    }

    // The outcome and each distinct predictor column, read once (strict numeric:
    // integers and booleans accepted, null or NaN as a missing value, infinite values
    // refused).
    let mut values: HashMap<&str, Vec<Option<f64>>> = HashMap::new();
    values.insert(outcome, numeric_options(df, outcome)?);
    for spec in predictors {
        if !values.contains_key(spec.column.as_str()) {
            values.insert(spec.column.as_str(), numeric_options(df, &spec.column)?);
        }
    }

    // Build predictor matrices
    let k = predictors.len();
    let j = donor_units.len();

    let mut x1 = Array1::zeros(k);
    let mut x0 = crate::resource::try_zeros2(
        k,
        j,
        "synthetic-control predictor matrix (predictors x donors)",
    )?;
    let mut predictor_names = Vec::with_capacity(k);

    // Each cell is a lookup in the panel index, so building these matrices is
    // O(cells); the budget checks let a deadline stop it.
    for (p_idx, pred_spec) in predictors.iter().enumerate() {
        crate::resource::checkpoint()?;

        let pred_times = match pred_spec.time_window {
            Some((start, end)) => pre_times
                .iter()
                .copied()
                .filter(|&t| t >= start && t <= end)
                .collect(),
            None => pre_times.clone(),
        };

        let pred_name = format!(
            "{}{}",
            pred_spec.column,
            match &pred_spec.time_window {
                Some((s, e)) => format!(" [{}-{}]", s, e),
                None => String::new(),
            }
        );
        predictor_names.push(pred_name);

        let column = &values[pred_spec.column.as_str()];

        // Get treated unit predictor value
        x1[p_idx] = aggregate_predictor(
            &panel,
            column,
            &pred_spec.column,
            treated,
            &pred_times,
            &pred_spec.aggregation,
        )?;

        // Get donor unit predictor values
        for (d_idx, &donor) in donors.iter().enumerate() {
            crate::resource::checkpoint()?;
            x0[[p_idx, d_idx]] = aggregate_predictor(
                &panel,
                column,
                &pred_spec.column,
                donor,
                &pred_times,
                &pred_spec.aggregation,
            )?;
        }
    }

    // Build outcome matrices
    let t0 = opt_times.len();
    let t_post = post_times.len();

    let mut z1 = Array1::zeros(t0);
    let mut z0 = crate::resource::try_zeros2(
        t0,
        j,
        "synthetic-control outcome matrix (pre-treatment periods x donors)",
    )?;
    let mut y1_post = Array1::zeros(t_post);
    let mut y0_post = crate::resource::try_zeros2(
        t_post,
        j,
        "synthetic-control outcome matrix (post-treatment periods x donors)",
    )?;

    let y = &values[outcome];

    // Pre-treatment outcomes
    for (t_idx, &time) in opt_times.iter().enumerate() {
        crate::resource::checkpoint()?;
        z1[t_idx] = get_outcome(&panel, y, outcome, treated, time)?;

        for (d_idx, &donor) in donors.iter().enumerate() {
            crate::resource::checkpoint()?;
            z0[[t_idx, d_idx]] = get_outcome(&panel, y, outcome, donor, time)?;
        }
    }

    // Post-treatment outcomes
    for (t_idx, &time) in post_times.iter().enumerate() {
        crate::resource::checkpoint()?;
        y1_post[t_idx] = get_outcome(&panel, y, outcome, treated, time)?;

        for (d_idx, &donor) in donors.iter().enumerate() {
            crate::resource::checkpoint()?;
            y0_post[[t_idx, d_idx]] = get_outcome(&panel, y, outcome, donor, time)?;
        }
    }

    Ok(SynthData {
        x1,
        x0,
        z1,
        z0,
        y1_post,
        y0_post,
        donor_units,
        predictor_names,
        pre_times: opt_times,
        post_times,
    })
}

/// The (unit, time) index of a synthetic-control panel.
struct SynthPanel {
    /// Unit labels in R `factor()` order.
    units: Vec<String>,
    /// The distinct time periods, ascending.
    times: Vec<i64>,
    /// The frame row of each (unit level, time) cell.
    cell: HashMap<(usize, i64), usize>,
}

/// Index a synthetic-control panel in one pass over the rows: the unit as a factor, the
/// time as a whole-number index (or date), a missing unit or time label refused with
/// `NullValues`, and a duplicate (unit, time) row an `InvalidSpecification` naming the
/// first repeated cell.
fn read_synth_panel(df: &DataFrame, unit_col: &str, time_col: &str) -> EconResult<SynthPanel> {
    let cols = ColumnRequest::new(df)
        .factor(unit_col)
        .time_index(time_col)
        .na(NaPolicy::Fail)
        .extract()?;
    let unit = cols.factor(unit_col);
    let time = cols.time_index(time_col);
    let n = cols.n();

    let what = "synthetic-control panel index";
    let mut cell: HashMap<(usize, i64), usize> = HashMap::new();
    cell.try_reserve(n).map_err(|_| {
        crate::resource::reserve_failed(what, n, std::mem::size_of::<((usize, i64), usize)>())
    })?;
    let mut duplicates = 0usize;
    let mut first_duplicate = None;
    for (i, (&code, &t)) in unit.codes.iter().zip(time).enumerate() {
        match cell.entry((code as usize, t)) {
            Entry::Occupied(_) => {
                duplicates += 1;
                first_duplicate.get_or_insert(i);
            }
            Entry::Vacant(slot) => {
                slot.insert(cols.rows()[i]);
            }
        }
    }
    if let Some(i) = first_duplicate {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "synthetic control needs one row per (unit, time): {duplicates} duplicate \
                 row(s) (first: unit '{}', time {}). Remove the duplicates first.",
                unit.label(i),
                time[i]
            ),
        });
    }

    let mut times: Vec<i64> =
        crate::resource::try_with_capacity(n, "synthetic-control time periods")?;
    times.extend_from_slice(time);
    times.sort_unstable();
    times.dedup();

    Ok(SynthPanel {
        units: unit.levels.clone(),
        times,
        cell,
    })
}

/// Aggregate a predictor over `times` for unit level `unit`. Absent cells and missing
/// values are skipped (the aggregate is over the present values).
fn aggregate_predictor(
    panel: &SynthPanel,
    values: &[Option<f64>],
    column: &str,
    unit: usize,
    times: &[i64],
    aggregation: &TimeAggregation,
) -> EconResult<f64> {
    let (mut count, mut sum, mut first, mut last) = (0usize, 0.0, f64::NAN, f64::NAN);
    for &time in times {
        if let Some(&row) = panel.cell.get(&(unit, time))
            && let Some(v) = values[row]
        {
            if count == 0 {
                first = v;
            }
            last = v;
            sum += v;
            count += 1;
        }
    }

    if count == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "No valid values for predictor '{}' for unit '{}' in specified time window",
                column, panel.units[unit]
            ),
        });
    }

    let result = match aggregation {
        TimeAggregation::Mean => sum / count as f64,
        TimeAggregation::First => first,
        TimeAggregation::Last => last,
        TimeAggregation::Sum => sum,
    };

    Ok(result)
}

/// The outcome of unit level `unit` at `time`: `values` holds one entry per frame row.
fn get_outcome(
    panel: &SynthPanel,
    values: &[Option<f64>],
    outcome: &str,
    unit: usize,
    time: i64,
) -> EconResult<f64> {
    let label = &panel.units[unit];
    let row = *panel
        .cell
        .get(&(unit, time))
        .ok_or_else(|| EconError::InvalidSpecification {
            message: format!("No data for unit '{label}' at time {time}"),
        })?;
    values[row].ok_or_else(|| EconError::InvalidSpecification {
        message: format!(
            "Outcome '{outcome}' is missing (null or NaN) for unit '{label}' at time {time}."
        ),
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Optimization
// ═══════════════════════════════════════════════════════════════════════════════

/// Tie-breaking ridge of the simplex QP, relative to `trace(H) / J`.
///
/// `H = X0' V X0` has rank at most k + 1 < J whenever there are more donors than
/// predictors, so the bare problem can have a face of optima. The ridge
/// `SIMPLEX_QP_RIDGE * trace(H) / J * I` makes it strictly convex: the minimiser is then
/// unique, it is the minimum-norm point of that face, and it does not change when every
/// predictor (or V) is multiplied by a common factor. The ridge is part of the pinned
/// definition of the weights (`synth_weights_qp`): solving without it, or with another
/// value, moves weights on a nearly flat face by more than the R pins allow.
pub const SIMPLEX_QP_RIDGE: f64 = 1e-10;

/// Label of the donor-by-donor buffers of the weight solver (allocation ceiling).
const QP_WHAT: &str = "synthetic-control donor-by-donor matrices";

/// The solution of [`simplex_qp`].
#[derive(Debug, Clone)]
pub(crate) struct SimplexQp {
    /// The weights: non-negative, summing to 1.
    pub weights: Array1<f64>,
    /// Indices of the positive weights, ascending.
    pub free: Vec<usize>,
    /// Active-set iterations.
    pub iterations: usize,
    /// Whether the unregularised problem has this single optimum (false: a face of optima,
    /// of which the ridge picked the minimum-norm point).
    pub unique: bool,
}

#[cfg(test)]
thread_local! {
    /// Test hook: solve every KKT system by the full bordered LU.
    static FORCE_BORDERED_KKT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with the reduced KKT path disabled (tests compare the two paths).
#[cfg(test)]
pub(crate) fn with_bordered_kkt<R>(f: impl FnOnce() -> R) -> R {
    FORCE_BORDERED_KKT.with(|c| c.set(true));
    let r = f();
    FORCE_BORDERED_KKT.with(|c| c.set(false));
    r
}

fn force_bordered_kkt() -> bool {
    #[cfg(test)]
    {
        FORCE_BORDERED_KKT.with(|c| c.get())
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// `min_w 1/2 ||a w - b||^2` subject to `w >= 0`, `sum(w) = 1`, exactly (the simplex QP
/// with `H = a'a`, `c = a'b`; `a` is k x J).
///
/// The problem is normalised by `s = ||a||_F^2 / J` (= `trace(H) / J`) and given the
/// tie-breaking ridge [`SIMPLEX_QP_RIDGE`]: `G = a'a / s + ridge I`, `q = a'b / s`, so `(a, b)`
/// and `(t a, t b)` give the same weights (bitwise when `t` is a power of 2). A primal
/// active-set method starts at the best vertex and solves the equality-constrained problem
/// on the free set exactly (a bordered KKT system, or, when the free set is larger than
/// k + 1, the same system reduced to `span(a_F', 1)`, where its solution lies), stepping
/// back to the feasible region when a weight would turn negative and adding the donor with
/// the most negative multiplier until every multiplier is non-negative. A singular KKT
/// system or a run past `10 J + 10` iterations is an error, never a fallback.
pub(crate) fn simplex_qp(a: ArrayView2<f64>, b: ArrayView1<f64>) -> EconResult<SimplexQp> {
    simplex_qp_from(a, b, None, true)
}

/// [`simplex_qp`] from a feasible warm start (the weights and free set of a previous
/// solution of a problem with the same constraints); `check_unique = false` skips the
/// uniqueness test (`unique` is then false).
fn simplex_qp_from(
    a: ArrayView2<f64>,
    b: ArrayView1<f64>,
    warm: Option<&SimplexQp>,
    check_unique: bool,
) -> EconResult<SimplexQp> {
    let (k, j) = a.dim();
    if j == 0 {
        return Err(EconError::InvalidSpecification {
            message: "simplex QP: no donors".to_string(),
        });
    }
    if b.len() != k {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "simplex QP: the target has {} entries for {k} rows of the donor matrix",
                b.len()
            ),
        });
    }
    if let Some(((m, i), v)) = a.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!("simplex QP: donor matrix entry ({m}, {i}) is {v}"),
        });
    }
    if let Some((m, v)) = b.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!("simplex QP: target entry {m} is {v}"),
        });
    }
    if j == 1 {
        return Ok(SimplexQp {
            weights: Array1::from_elem(1, 1.0),
            free: vec![0],
            iterations: 0,
            unique: true,
        });
    }

    let s = a.iter().map(|v| v * v).sum::<f64>() / j as f64;
    if !(s.is_finite() && s > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: "synthetic control: the donors' predictor values carry no variation \
                      (trace of X0'VX0 is 0); the weights are not identified"
                .to_string(),
        });
    }
    let root = s.sqrt();
    let mut at = crate::resource::try_zeros2(k, j, QP_WHAT)?;
    at.zip_mut_with(&a, |t, &v| *t = v / root);
    let bt = b.mapv(|v| v / root);
    // G_jj and q_j
    let colsq: Vec<f64> = (0..j).map(|i| at.column(i).dot(&at.column(i))).collect();
    let q: Vec<f64> = (0..j).map(|i| at.column(i).dot(&bt)).collect();
    let gmax = colsq.iter().cloned().fold(0.0_f64, f64::max) + SIMPLEX_QP_RIDGE;
    let dual_tol = 1e-12 * gmax.max(1.0);

    let mut w = Array1::<f64>::zeros(j);
    let mut free: Vec<usize> = match warm {
        Some(start) if start.weights.len() == j && !start.free.is_empty() => {
            w.assign(&start.weights);
            start.free.clone()
        }
        _ => {
            let mut j0 = 0;
            let mut best = f64::INFINITY;
            for i in 0..j {
                let v = 0.5 * (colsq[i] + SIMPLEX_QP_RIDGE) - q[i];
                if v < best {
                    best = v;
                    j0 = i;
                }
            }
            w[j0] = 1.0;
            vec![j0]
        }
    };

    let cap = 10 * j + 10;
    let mut iterations = 0;
    loop {
        if iterations == cap {
            return Err(EconError::ConvergenceFailure {
                iterations,
                last_change: f64::NAN,
                suggestion: format!(
                    "simplex QP found no optimal active set after {iterations} iterations"
                ),
            });
        }
        crate::resource::checkpoint()?;
        iterations += 1;
        let (z, mu) = kkt_on_free_set(&at, &bt, &free)?;
        if z.iter().all(|&v| v > 0.0) {
            for (&i, &v) in free.iter().zip(z.iter()) {
                w[i] = v;
            }
            // multipliers of the zero weights: a_j'(a w - b) / s + mu (w_j = 0)
            let rho = at.dot(&w) - &bt;
            let mut entering = None;
            let mut min_eta = f64::INFINITY;
            let mut fi = 0;
            for i in 0..j {
                if fi < free.len() && free[fi] == i {
                    fi += 1;
                    continue;
                }
                let eta = at.column(i).dot(&rho) + mu;
                if eta < min_eta {
                    min_eta = eta;
                    entering = Some(i);
                }
            }
            match entering {
                Some(i) if min_eta < -dual_tol => {
                    let pos = free.partition_point(|&f| f < i);
                    free.insert(pos, i);
                }
                _ => break,
            }
        } else {
            // step from w towards z until the first weight reaches 0
            let mut alpha = f64::INFINITY;
            let mut blocking = 0;
            for (pos, (&i, &zi)) in free.iter().zip(z.iter()).enumerate() {
                if zi <= 0.0 {
                    let ratio = w[i] / (w[i] - zi);
                    if ratio < alpha {
                        alpha = ratio;
                        blocking = pos;
                    }
                }
            }
            let alpha = alpha.clamp(0.0, 1.0);
            for (&i, &zi) in free.iter().zip(z.iter()) {
                w[i] += alpha * (zi - w[i]);
            }
            let blocked = free[blocking];
            w[blocked] = 0.0;
            free.retain(|&i| {
                if w[i] <= 0.0 {
                    w[i] = 0.0;
                    false
                } else {
                    true
                }
            });
            if free.is_empty() {
                return Err(EconError::Computation(
                    "simplex QP: the active-set step left no positive weight".to_string(),
                ));
            }
        }
    }

    let unique = check_unique && simplex_qp_unique(&at, &free)?;
    Ok(SimplexQp {
        weights: w,
        free,
        iterations,
        unique,
    })
}

/// The optimum of the unregularised problem is unique iff the free set has at most k + 1
/// donors and `[a_F; 1']` has full column rank.
fn simplex_qp_unique(at: &Array2<f64>, free: &[usize]) -> EconResult<bool> {
    let k = at.nrows();
    let f = free.len();
    if f > k + 1 {
        return Ok(false);
    }
    let mut m = crate::resource::try_zeros2(k + 1, f, QP_WHAT)?;
    for (c, &i) in free.iter().enumerate() {
        for r in 0..k {
            m[[r, c]] = at[[r, i]];
        }
        m[[k, c]] = 1.0;
    }
    let qr = crate::linalg::solve::Qr::decompose(m.view(), crate::linalg::solve::LM_QR_TOL)?;
    Ok(qr.rank() == f)
}

/// Solve `min 1/2 z'(a_F'a_F + ridge I) z - (a_F'b)'z` subject to `1'z = 1` on the free set
/// (`at`, `bt` already divided by `sqrt(s)`): `z` and the multiplier `mu` of the bordered
/// system `[[G_FF, 1], [1', 0]] [z; mu] = [q_F; 1]`.
fn kkt_on_free_set(
    at: &Array2<f64>,
    bt: &Array1<f64>,
    free: &[usize],
) -> EconResult<(Array1<f64>, f64)> {
    let k = at.nrows();
    if free.len() > k + 1 && !force_bordered_kkt() {
        if let Some(sol) = kkt_reduced(at, bt, free)? {
            return Ok(sol);
        }
    }
    kkt_bordered(at, bt, free)
}

/// Solve a bordered system by partial-pivot LU and check its residual.
fn solve_checked(k: &Array2<f64>, rhs: &Array1<f64>) -> Option<Array1<f64>> {
    let n = rhs.len();
    let km = crate::linalg::matrix_ops::ndarray_to_faer(&k.view());
    let mut rm = faer::Mat::<f64>::zeros(n, 1);
    for i in 0..n {
        rm[(i, 0)] = rhs[i];
    }
    let sol = faer::linalg::solvers::Solve::solve(&km.partial_piv_lu(), &rm);
    let x = Array1::from_shape_fn(n, |i| sol[(i, 0)]);
    if !x.iter().all(|v| v.is_finite()) {
        return None;
    }
    let res = k.dot(&x) - rhs;
    let res_max = res.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let rhs_max = rhs.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    (res_max <= 1e-8 * (1.0 + rhs_max)).then_some(x)
}

fn kkt_bordered(
    at: &Array2<f64>,
    bt: &Array1<f64>,
    free: &[usize],
) -> EconResult<(Array1<f64>, f64)> {
    let f = free.len();
    let mut km = crate::resource::try_zeros2(f + 1, f + 1, QP_WHAT)?;
    let mut rhs = Array1::<f64>::zeros(f + 1);
    for (r, &i) in free.iter().enumerate() {
        let ci = at.column(i);
        for (c, &l) in free.iter().enumerate().skip(r) {
            let g = ci.dot(&at.column(l));
            km[[r, c]] = g;
            km[[c, r]] = g;
        }
        km[[r, r]] += SIMPLEX_QP_RIDGE;
        km[[r, f]] = 1.0;
        km[[f, r]] = 1.0;
        rhs[r] = ci.dot(bt);
    }
    rhs[f] = 1.0;
    let x = solve_checked(&km, &rhs)
        .ok_or_else(|| EconError::Computation("simplex QP: singular KKT system".to_string()))?;
    Ok((x.slice(ndarray::s![..f]).to_owned(), x[f]))
}

/// The KKT system reduced to `span(a_F', 1)`: with Q (f x r) an orthonormal basis of that
/// span and `B = a_F Q`, solve `[[B'B + ridge I, Q'1], [1'Q, 0]] [y; mu] = [B'b; 1]` and set
/// `z = Q y`. The optimum lies in that span (the stationarity condition writes `ridge z` as
/// a combination of `a_F'b`, `a_F'a_F z` and `1`), so this is the solution of the full
/// bordered system at O(f k^2) instead of O(f^3). `None` when the reduced system fails its
/// residual check (the caller then solves the full system).
fn kkt_reduced(
    at: &Array2<f64>,
    bt: &Array1<f64>,
    free: &[usize],
) -> EconResult<Option<(Array1<f64>, f64)>> {
    let k = at.nrows();
    let f = free.len();
    // modified Gram-Schmidt (twice) over the rows of [a_F; 1']
    let mut basis: Vec<Array1<f64>> = Vec::with_capacity(k + 1);
    for r in 0..=k {
        let row: Array1<f64> = if r < k {
            Array1::from_shape_fn(f, |c| at[[r, free[c]]])
        } else {
            Array1::ones(f)
        };
        let norm0 = row.dot(&row).sqrt();
        if norm0 == 0.0 {
            continue;
        }
        let mut v = row;
        for _ in 0..2 {
            for e in &basis {
                let d = e.dot(&v);
                v.scaled_add(-d, e);
            }
        }
        let norm = v.dot(&v).sqrt();
        if norm > crate::linalg::solve::LM_QR_TOL * norm0 {
            basis.push(v / norm);
        }
    }
    let r = basis.len();
    // B = a_F Q (k x r), Q'1, B'b
    let mut bmat = Array2::<f64>::zeros((k, r));
    for (c, e) in basis.iter().enumerate() {
        for m in 0..k {
            let mut acc = 0.0;
            for (p, &i) in free.iter().enumerate() {
                acc += at[[m, i]] * e[p];
            }
            bmat[[m, c]] = acc;
        }
    }
    let mut km = Array2::<f64>::zeros((r + 1, r + 1));
    let mut rhs = Array1::<f64>::zeros(r + 1);
    for c1 in 0..r {
        for c2 in c1..r {
            let g = bmat.column(c1).dot(&bmat.column(c2));
            km[[c1, c2]] = g;
            km[[c2, c1]] = g;
        }
        km[[c1, c1]] += SIMPLEX_QP_RIDGE;
        let q1 = basis[c1].sum();
        km[[c1, r]] = q1;
        km[[r, c1]] = q1;
        rhs[c1] = bmat.column(c1).dot(bt);
    }
    rhs[r] = 1.0;
    let Some(x) = solve_checked(&km, &rhs) else {
        return Ok(None);
    };
    let mut z = Array1::<f64>::zeros(f);
    for (c, e) in basis.iter().enumerate() {
        z.scaled_add(x[c], e);
    }
    Ok(Some((z, x[r])))
}

/// Each predictor divided by its standard deviation across the treated unit and the donors
/// (n - 1 denominator), as R Synth's `synth()` does before any V is applied. A predictor
/// whose donor values do not vary is refused, as Synth refuses it.
fn standardized_predictors(data: &SynthData) -> EconResult<(Array1<f64>, Array2<f64>)> {
    let (k, j) = data.x0.dim();
    let mut xs1 = data.x1.clone();
    let mut xs0 = data.x0.clone();
    for m in 0..k {
        let row = data.x0.row(m);
        if j >= 2 {
            let mean = row.sum() / j as f64;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
            if !(var > 0.0) {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "synthetic control: predictor '{}' has no variation across the donor \
                         units; drop it",
                        data.predictor_names[m]
                    ),
                });
            }
        }
        let n = (j + 1) as f64;
        let mean = (row.sum() + data.x1[m]) / n;
        let ss = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() + (data.x1[m] - mean).powi(2);
        let sd = (ss / (n - 1.0)).sqrt();
        if sd > 0.0 && sd.is_finite() {
            xs1[m] /= sd;
            xs0.row_mut(m).mapv_inplace(|v| v / sd);
        }
    }
    Ok((xs1, xs0))
}

/// Synth's regression-based start of the V search (SV2): regress each pre-period's outcomes
/// of the treated unit and the donors on an intercept and the standardised predictors; the
/// start is the squared slope coefficients summed over the periods, normalised to sum 1.
/// Equal V when the regression is not identified (fewer units than predictors plus one, or
/// collinear predictors) or every slope is 0.
fn regression_v_start(
    xs1: &Array1<f64>,
    xs0: &Array2<f64>,
    z1: &Array1<f64>,
    z0: &Array2<f64>,
) -> EconResult<Array1<f64>> {
    let (k, j) = xs0.dim();
    let equal = Array1::from_elem(k, 1.0 / k as f64);
    let mut design = crate::resource::try_zeros2(j + 1, k + 1, "synthetic-control V start design")?;
    for u in 0..=j {
        design[[u, 0]] = 1.0;
        for m in 0..k {
            design[[u, m + 1]] = if u == 0 { xs1[m] } else { xs0[[m, u - 1]] };
        }
    }
    if j + 1 < k + 1 {
        return Ok(equal);
    }
    let qr = crate::linalg::solve::Qr::decompose(design.view(), crate::linalg::solve::LM_QR_TOL)?;
    if qr.rank() < k + 1 {
        return Ok(equal);
    }
    let mut d = Array1::<f64>::zeros(k);
    let mut y = Array1::<f64>::zeros(j + 1);
    for t in 0..z1.len() {
        crate::resource::checkpoint()?;
        y[0] = z1[t];
        for u in 0..j {
            y[u + 1] = z0[[t, u]];
        }
        let beta = qr.coefficients(y.view())?;
        for m in 0..k {
            d[m] += beta[m + 1] * beta[m + 1];
        }
    }
    let total = d.sum();
    if !(total > 0.0 && total.is_finite()) {
        return Ok(equal);
    }
    Ok(d / total)
}

/// Custom V: length k, finite, non-negative, positive sum; returned normalised to sum 1.
fn validate_custom_v(weights: &[f64], k: usize) -> EconResult<Array1<f64>> {
    if weights.len() != k {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Custom V weights length ({}) doesn't match number of predictors ({})",
                weights.len(),
                k
            ),
        });
    }
    let bad = |i: usize, v: f64| EconError::InvalidSpecification {
        message: format!(
            "custom V weights (custom_v_weights) must be finite and non-negative with a \
             positive sum; entry {i} is {v}"
        ),
    };
    if let Some((i, &v)) = weights
        .iter()
        .enumerate()
        .find(|(_, v)| !(v.is_finite() && **v >= 0.0))
    {
        return Err(bad(i, v));
    }
    let sum: f64 = weights.iter().sum();
    if !(sum > 0.0 && sum.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "custom V weights (custom_v_weights) must be finite and non-negative with a \
                 positive sum; their sum is {sum}"
            ),
        });
    }
    Ok(Array1::from_iter(weights.iter().map(|&w| w / sum)))
}

/// Unit weights for predictor weights `v` on standardised predictors: the simplex QP with
/// `a = diag(sqrt v) X0`, `b = diag(sqrt v) X1`.
fn solve_weights_qp(x0: &Array2<f64>, x1: &Array1<f64>, v: &Array1<f64>) -> EconResult<SimplexQp> {
    solve_weights_qp_from(x0, x1, v, None, true)
}

fn solve_weights_qp_from(
    x0: &Array2<f64>,
    x1: &Array1<f64>,
    v: &Array1<f64>,
    warm: Option<&SimplexQp>,
    check_unique: bool,
) -> EconResult<SimplexQp> {
    let (k, j) = x0.dim();
    if j == 0 {
        return Err(EconError::InvalidSpecification {
            message: "No donor units".to_string(),
        });
    }
    let mut a = crate::resource::try_zeros2(k, j, QP_WHAT)?;
    let mut b = Array1::<f64>::zeros(k);
    for m in 0..k {
        let sv = v[m].sqrt();
        a.row_mut(m).zip_mut_with(&x0.row(m), |t, &x| *t = sv * x);
        b[m] = sv * x1[m];
    }
    simplex_qp_from(a.view(), b.view(), warm, check_unique)
}

/// `(X1 - X0 w)' V (X1 - X0 w)` (standardised predictors).
fn predictor_loss(x0: &Array2<f64>, x1: &Array1<f64>, v: &Array1<f64>, w: &Array1<f64>) -> f64 {
    let gap = x1 - &x0.dot(w);
    gap.iter().zip(v.iter()).map(|(g, vm)| vm * g * g).sum()
}

/// V and W with what the search reports.
struct WeightFit {
    v: Array1<f64>,
    w: SimplexQp,
    iterations: usize,
    final_loss: f64,
    predictor_loss: f64,
    convergence: Option<Convergence>,
    v_start_losses: Vec<f64>,
}

/// Raw Nelder-Mead coordinates to V: `max(x, 1e-6)`, normalised to sum 1.
fn v_from_raw(raw: &[f64]) -> Array1<f64> {
    let pos: Vec<f64> = raw.iter().map(|&x| x.max(1e-6)).collect();
    let sum: f64 = pos.iter().sum();
    Array1::from_iter(pos.into_iter().map(|x| x / sum))
}

/// Optimize V (predictor weights) and W (unit weights).
///
/// Nested as R's Synth: predictors standardised by their SD across units; for a given V the
/// unit weights are the exact simplex QP ([`simplex_qp`]); a data-driven V minimises the
/// pre-treatment MSPE by Nelder-Mead from two starts, equal V and Synth's regression-based
/// V ([`regression_v_start`]), keeping the start with the lower loss (equal on ties).
///
/// # References
///
/// - Nelder, J. A., & Mead, R. (1965). "A Simplex Method for Function Minimization."
///   The Computer Journal, 7(4), 308-313.
/// - R package `Synth` uses `optimx(method = "Nelder-Mead")` from both starts.
fn optimize_synth_weights(
    data: &SynthData,
    v_method: &VOptimization,
    tolerance: f64,
    max_iter: usize,
) -> EconResult<WeightFit> {
    let k = data.x1.len();

    // The weight solver's buffers are at most donor-by-donor (the bordered KKT system); the
    // data-driven search keeps one per predictor and the warm start. Their total is checked
    // against the allocation ceiling before any is built.
    let j = data.x0.ncols();
    let n_matrices = if matches!(v_method, VOptimization::DataDriven) {
        k.saturating_add(1)
    } else {
        1
    };
    crate::resource::check_alloc::<f64>(
        crate::resource::checked_len(&[n_matrices, j, j], QP_WHAT)?,
        QP_WHAT,
    )?;

    let (xs1, xs0) = standardized_predictors(data)?;

    let fixed = match v_method {
        VOptimization::Equal => Some(Array1::from_elem(k, 1.0 / k as f64)),
        VOptimization::Custom(weights) => Some(validate_custom_v(weights, k)?),
        VOptimization::DataDriven if k == 1 => Some(Array1::from_elem(1, 1.0)),
        VOptimization::DataDriven => None,
    };
    if let Some(v) = fixed {
        let w = solve_weights_qp(&xs0, &xs1, &v)?;
        let final_loss = calculate_v_loss(data, &w.weights);
        let ploss = predictor_loss(&xs0, &xs1, &v, &w.weights);
        return Ok(WeightFit {
            v,
            w,
            iterations: 0,
            final_loss,
            predictor_loss: ploss,
            convergence: None,
            v_start_losses: Vec::new(),
        });
    }

    // One search per start; each Nelder-Mead evaluation solves the QP from the previous
    // evaluation's weights (a feasible start for the same constraints).
    let warm: std::cell::RefCell<Option<SimplexQp>> = std::cell::RefCell::new(None);
    let evaluate_v = |raw: &[f64]| -> EconResult<f64> {
        let v = v_from_raw(raw);
        let prev = warm.borrow_mut().take();
        let w = solve_weights_qp_from(&xs0, &xs1, &v, prev.as_ref(), false)?;
        let loss = calculate_v_loss(data, &w.weights);
        *warm.borrow_mut() = Some(w);
        Ok(loss)
    };
    let starts = [
        Array1::from_elem(k, 1.0 / k as f64),
        regression_v_start(&xs1, &xs0, &data.z1, &data.z0)?,
    ];
    let mut runs = Vec::with_capacity(2);
    for start in &starts {
        *warm.borrow_mut() = None;
        runs.push(nelder_mead_optimize(
            start.as_slice().expect("contiguous"),
            &evaluate_v,
            tolerance,
            max_iter,
        )?);
    }
    // R's which.min over the two losses: the equal start on ties
    let chosen = if runs[1].value < runs[0].value { 1 } else { 0 };
    let iterations = runs.iter().map(|r| r.iterations).sum();
    let v_start_losses: Vec<f64> = runs.iter().map(|r| r.value).collect();
    let run = &runs[chosen];
    let v = v_from_raw(&run.point);
    // the reported weights are re-solved from a cold start at the final V
    let w = solve_weights_qp(&xs0, &xs1, &v)?;
    let final_loss = calculate_v_loss(data, &w.weights);
    let ploss = predictor_loss(&xs0, &xs1, &v, &w.weights);
    let convergence = Convergence::new(
        run.converged,
        run.iterations,
        max_iter,
        "relative spread of the Nelder-Mead simplex values",
        run.spread,
        tolerance,
    );
    Ok(WeightFit {
        v,
        w,
        iterations,
        final_loss,
        predictor_loss: ploss,
        convergence: Some(convergence),
        v_start_losses,
    })
}

/// The result of [`nelder_mead_optimize`].
struct NelderMead {
    point: Vec<f64>,
    value: f64,
    iterations: usize,
    converged: bool,
    /// `|f_worst - f_best| / (|f_best| + tol)` at the last iteration.
    spread: f64,
}

/// Nelder-Mead simplex optimizer for V-matrix weights.
///
/// Standard Nelder-Mead with reflection, expansion, contraction, and shrink steps.
/// Parameters follow the standard settings from Nelder & Mead (1965):
/// - alpha (reflection) = 1.0
/// - gamma (expansion) = 2.0
/// - rho (contraction) = 0.5
/// - sigma (shrink) = 0.5
///
/// Stops on R `optim`'s `reltol` rule, `|f_worst - f_best| <= tol (|f_best| + tol)`, or when
/// the simplex has collapsed (diameter below `tol * 1e-4`). An error of the objective ends
/// the search; so does a stopped resource budget (`Err(Cancelled)`, checked once per
/// iteration).
fn nelder_mead_optimize<F>(
    x0: &[f64],
    objective: &F,
    tol: f64,
    max_iter: usize,
) -> EconResult<NelderMead>
where
    F: Fn(&[f64]) -> EconResult<f64>,
{
    let n = x0.len();

    // Nelder-Mead standard parameters (Nelder & Mead, 1965)
    let alpha = 1.0; // Reflection coefficient
    let gamma = 2.0; // Expansion coefficient
    let rho = 0.5; // Contraction coefficient
    let sigma = 0.5; // Shrink coefficient

    // Initialize simplex with n+1 vertices
    // Each vertex perturbs one coordinate of x0
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    let mut values: Vec<f64> = Vec::with_capacity(n + 1);

    // First vertex is x0 itself
    simplex.push(x0.to_vec());
    values.push(objective(x0)?);

    // Remaining vertices: perturb each coordinate
    for i in 0..n {
        let mut vertex = x0.to_vec();
        let perturbation = if vertex[i].abs() > 1e-10 {
            0.05 * vertex[i] // 5% perturbation
        } else {
            0.00025 // Small absolute perturbation for near-zero values
        };
        vertex[i] += perturbation;
        values.push(objective(&vertex)?);
        simplex.push(vertex);
    }

    let mut iterations = 0;
    let mut converged = false;
    let mut spread = f64::NAN;

    for iter in 0..max_iter {
        crate::resource::checkpoint()?;
        iterations = iter + 1;

        // Sort vertices by function value (ascending)
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));

        let sorted_simplex: Vec<Vec<f64>> = order.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_values: Vec<f64> = order.iter().map(|&i| values[i]).collect();
        simplex = sorted_simplex;
        values = sorted_values;

        // Convergence: R's reltol rule on the spread of the function values
        let f_best = values[0];
        let f_worst = values[n];
        spread = (f_worst - f_best).abs() / (f_best.abs() + tol);
        if iter > 0 && (f_worst - f_best).abs() <= tol * (f_best.abs() + tol) {
            converged = true;
            break;
        }

        // Also check convergence on simplex diameter
        let mut max_dist = 0.0_f64;
        for i in 1..=n {
            let dist: f64 = simplex[i]
                .iter()
                .zip(simplex[0].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            max_dist = max_dist.max(dist);
        }
        if max_dist < tol * 1e-4 && iter > 0 {
            converged = true;
            break;
        }

        // Compute centroid of all vertices except the worst
        let mut centroid = vec![0.0; n];
        for vertex in simplex.iter().take(n) {
            for (c, v) in centroid.iter_mut().zip(vertex) {
                *c += v;
            }
        }
        for c in centroid.iter_mut() {
            *c /= n as f64;
        }

        // Reflection: x_r = centroid + alpha * (centroid - worst)
        let worst = simplex[n].clone();
        let x_r: Vec<f64> = centroid
            .iter()
            .zip(worst.iter())
            .map(|(&c, &w)| c + alpha * (c - w))
            .collect();
        let f_r = objective(&x_r)?;

        if f_r < values[0] {
            // Reflection is best so far: try expansion
            // x_e = centroid + gamma * (x_r - centroid)
            let x_e: Vec<f64> = centroid
                .iter()
                .zip(x_r.iter())
                .map(|(&c, &r)| c + gamma * (r - c))
                .collect();
            let f_e = objective(&x_e)?;

            if f_e < f_r {
                // Accept expansion
                simplex[n] = x_e;
                values[n] = f_e;
            } else {
                // Accept reflection
                simplex[n] = x_r;
                values[n] = f_r;
            }
        } else if f_r < values[n - 1] {
            // Reflection is better than second-worst: accept it
            simplex[n] = x_r;
            values[n] = f_r;
        } else {
            // Contraction: outside x_c = centroid + rho (x_r - centroid) when the reflection
            // beats the worst vertex, inside x_cc = centroid - rho (centroid - worst) otherwise
            let (x_c, f_bound) = if f_r < values[n] {
                let x: Vec<f64> = centroid
                    .iter()
                    .zip(x_r.iter())
                    .map(|(&c, &r)| c + rho * (r - c))
                    .collect();
                (x, f_r)
            } else {
                let x: Vec<f64> = centroid
                    .iter()
                    .zip(worst.iter())
                    .map(|(&c, &w)| c - rho * (c - w))
                    .collect();
                (x, values[n])
            };
            let f_c = objective(&x_c)?;
            let accept = if f_r < values[n] {
                f_c <= f_bound
            } else {
                f_c < f_bound
            };
            if accept {
                simplex[n] = x_c;
                values[n] = f_c;
            } else {
                // Shrink: move all vertices toward the best
                for i in 1..=n {
                    for j in 0..n {
                        simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                    }
                    values[i] = objective(&simplex[i])?;
                }
            }
        }
    }

    // Return the best vertex
    let mut best_idx = 0;
    for i in 1..=n {
        if values[i] < values[best_idx] {
            best_idx = i;
        }
    }

    Ok(NelderMead {
        point: simplex[best_idx].clone(),
        value: values[best_idx],
        iterations,
        converged,
        spread,
    })
}

/// Calculate the V-loss (pre-treatment MSPE) for given W weights.
///
/// This is the objective function for V optimization.
/// Loss = (1/T₀) Σₜ (Z₁ₜ - Z₀ₜ W)²
///
/// Uses vectorized operations for better performance:
/// - z0.dot(w) computes all synthetic values in one SIMD-optimized operation
/// - Element-wise subtraction and squaring are also vectorized
fn calculate_v_loss(data: &SynthData, w: &Array1<f64>) -> f64 {
    let t0 = data.z1.len();
    if t0 == 0 {
        return 0.0;
    }

    // Vectorized computation: synthetic = Z₀ × W (T0 × J) · (J × 1) = (T0 × 1)
    let synthetic = data.z0.dot(w);

    // Vectorized error computation: errors = z1 - synthetic
    let errors = &data.z1 - &synthetic;

    // Vectorized SSE: sum of squared errors
    let sse: f64 = errors.iter().map(|&e| e * e).sum();

    sse / t0 as f64
}

// ═══════════════════════════════════════════════════════════════════════════════
// Results Calculation
// ═══════════════════════════════════════════════════════════════════════════════

/// Calculate pre-treatment fit statistics (MSPE and RMSPE).
///
/// Measures how well the synthetic control matches the treated unit in the
/// pre-treatment period. Lower values indicate better fit.
///
/// # Arguments
/// * `z1` - Treated unit outcomes in pre-treatment period (T₀ × 1)
/// * `z0` - Donor unit outcomes in pre-treatment period (T₀ × J)
/// * `w` - Unit weights (J × 1)
///
/// # Returns
/// Tuple of (MSPE, RMSPE) where:
/// - MSPE = Mean Squared Prediction Error = (1/T₀) Σₜ (Z₁ₜ - Z₀ₜ W)²
/// - RMSPE = Root Mean Squared Prediction Error = √MSPE
fn calculate_pre_treatment_fit(z1: &Array1<f64>, z0: &Array2<f64>, w: &Array1<f64>) -> (f64, f64) {
    let t0 = z1.len();
    let mut sse = 0.0;

    for t in 0..t0 {
        let synthetic: f64 = z0.row(t).iter().zip(w.iter()).map(|(&z, &w)| z * w).sum();
        let error = z1[t] - synthetic;
        sse += error * error;
    }

    let mspe = sse / t0 as f64;
    let rmspe = mspe.sqrt();

    (mspe, rmspe)
}

/// Calculate predictor balance between treated and synthetic control.
///
/// Computes how well the synthetic control matches the treated unit on each
/// predictor variable. This is a key diagnostic for assessing synthetic control quality.
///
/// # Arguments
/// * `x1` - Treated unit predictor values (K × 1)
/// * `x0` - Donor unit predictor values (K × J)
/// * `w` - Unit weights (J × 1)
/// * `predictor_names` - Names of predictor variables
///
/// # Returns
/// Vector of `PredictorBalance` structs containing:
/// - Predictor name
/// - Treated unit value
/// - Synthetic control value (= X₀ × W)
/// - Absolute difference
/// - Percent difference (relative to treated value)
fn calculate_predictor_balance(
    x1: &Array1<f64>,
    x0: &Array2<f64>,
    w: &Array1<f64>,
    predictor_names: &[String],
) -> Vec<PredictorBalance> {
    let k = x1.len();
    let mut balance = Vec::with_capacity(k);

    for i in 0..k {
        let treated_value = x1[i];
        let synthetic_value: f64 = x0.row(i).iter().zip(w.iter()).map(|(&x, &w)| x * w).sum();
        let difference = (treated_value - synthetic_value).abs();
        let percent_diff = if treated_value.abs() > 1e-10 {
            100.0 * difference / treated_value.abs()
        } else {
            0.0
        };

        balance.push(PredictorBalance {
            predictor: predictor_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("X{}", i)),
            treated_value,
            synthetic_value,
            difference,
            percent_diff,
        });
    }

    balance
}

/// Calculate treatment effects for each post-treatment period.
fn calculate_treatment_effects(data: &SynthData, w: &Array1<f64>) -> Vec<TimeEffect> {
    let mut effects = Vec::with_capacity(data.post_times.len());

    for (t_idx, &time) in data.post_times.iter().enumerate() {
        let actual = data.y1_post[t_idx];
        let synthetic: f64 = data
            .y0_post
            .row(t_idx)
            .iter()
            .zip(w.iter())
            .map(|(&y, &w)| y * w)
            .sum();
        let effect = actual - synthetic;

        effects.push(TimeEffect {
            time,
            effect,
            actual,
            synthetic,
        });
    }

    effects
}

/// Build complete time series for actual and synthetic outcomes.
fn build_time_series(data: &SynthData, w: &Array1<f64>) -> (Vec<(i64, f64)>, Vec<(i64, f64)>) {
    let mut actual = Vec::new();
    let mut synthetic = Vec::new();

    // Pre-treatment periods
    for (t_idx, &time) in data.pre_times.iter().enumerate() {
        actual.push((time, data.z1[t_idx]));

        let synth_val: f64 = data
            .z0
            .row(t_idx)
            .iter()
            .zip(w.iter())
            .map(|(&z, &w)| z * w)
            .sum();
        synthetic.push((time, synth_val));
    }

    // Post-treatment periods
    for (t_idx, &time) in data.post_times.iter().enumerate() {
        actual.push((time, data.y1_post[t_idx]));

        let synth_val: f64 = data
            .y0_post
            .row(t_idx)
            .iter()
            .zip(w.iter())
            .map(|(&y, &w)| y * w)
            .sum();
        synthetic.push((time, synth_val));
    }

    (actual, synthetic)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Placebo Inference
// ═══════════════════════════════════════════════════════════════════════════════

/// Run placebo tests for inference.
///
/// Applies synthetic control to each donor unit as if it were treated,
/// then compares RMSPE ratios to compute exact p-values.
fn run_placebo_inference(
    dataset: &Dataset,
    outcome: &str,
    unit_col: &str,
    time_col: &str,
    predictors: &[PredictorSpec],
    config: &SynthConfig,
    treated_rmspe_pre: f64,
) -> EconResult<PlaceboResults> {
    // Get all units, in level order
    let units = read_synth_panel(dataset.df(), unit_col, time_col)?.units;

    // Calculate RMSPE ratio for treated unit
    // We need the post-treatment RMSPE too
    let treated_config = SynthConfig {
        run_placebos: false,
        ..config.clone()
    };

    let treated_result = run_synthetic_control(
        dataset,
        outcome,
        unit_col,
        time_col,
        predictors,
        treated_config,
    )?;

    let treated_ratio = rmspe_ratio(treated_rmspe_pre, &treated_result.treatment_effects);

    // One placebo fit per donor unit, one after another on this thread. The
    // fits read the caller's budget (a thread-local) at their own checkpoints,
    // so a deadline or a cancellation stops the inference inside a fit. A
    // placebo whose fit fails is skipped; a stopped budget ends the inference.
    let mut rmspe_ratios: Vec<(String, f64)> = vec![(config.treated_unit.clone(), treated_ratio)];
    for donor_unit in units.iter().filter(|unit| *unit != &config.treated_unit) {
        crate::resource::checkpoint()?;
        if let Some(ratio) = placebo_rmspe_ratio(
            dataset, outcome, unit_col, time_col, predictors, config, donor_unit,
        )? {
            rmspe_ratios.push((donor_unit.clone(), ratio));
        }
    }

    // Sort by ratio (descending)
    rmspe_ratios.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Find rank of treated unit
    let treated_rank = rmspe_ratios
        .iter()
        .position(|(unit, _)| unit == &config.treated_unit)
        .map(|pos| pos + 1)
        .unwrap_or(1);

    let n_units = rmspe_ratios.len();
    let p_value = treated_rank as f64 / n_units as f64;
    let significance = SignificanceLevel::from_p_value(p_value);

    Ok(PlaceboResults {
        rmspe_ratios,
        treated_rank,
        p_value,
        n_units,
        significance,
    })
}

/// Post- over pre-treatment RMSPE ratio of the placebo fit that treats `donor`
/// in place of the treated unit. A fit that fails gives `None`, and the placebo
/// is skipped; a stopped budget (`Cancelled`) is passed on and ends the inference.
fn placebo_rmspe_ratio(
    dataset: &Dataset,
    outcome: &str,
    unit_col: &str,
    time_col: &str,
    predictors: &[PredictorSpec],
    config: &SynthConfig,
    donor: &str,
) -> EconResult<Option<f64>> {
    let placebo_config = SynthConfig {
        treated_unit: donor.to_string(),
        run_placebos: false,
        ..config.clone()
    };
    let placebo_result = match run_synthetic_control(
        dataset,
        outcome,
        unit_col,
        time_col,
        predictors,
        placebo_config,
    ) {
        Ok(r) => r,
        Err(e @ EconError::Cancelled(_)) => return Err(e),
        Err(_) => return Ok(None),
    };

    Ok(Some(rmspe_ratio(
        placebo_result.pre_treatment_rmspe,
        &placebo_result.treatment_effects,
    )))
}

/// Post- over pre-treatment RMSPE ratio, the statistic placebo inference ranks
/// units by. The post-treatment RMSPE of no effects is 0; a pre-treatment
/// RMSPE that is not above 1e-10 (NaN included) gives an infinite ratio. The
/// treated unit and every placebo go through this one function so that their
/// ratios stay comparable.
fn rmspe_ratio(pre_rmspe: f64, effects: &[TimeEffect]) -> f64 {
    let post_rmspe = if !effects.is_empty() {
        let mse: f64 =
            effects.iter().map(|te| te.effect * te.effect).sum::<f64>() / effects.len() as f64;
        mse.sqrt()
    } else {
        0.0
    };

    if pre_rmspe > 1e-10 {
        post_rmspe / pre_rmspe
    } else {
        f64::INFINITY
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Generalized Synthetic Control (gsynth)
// ═══════════════════════════════════════════════════════════════════════════════

/// Method for estimating the generalized synthetic control model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GsynthEstimator {
    /// Interactive fixed effects (IFE) - default method
    #[default]
    Ife,
    /// Matrix completion (MC) - alternative method
    MatrixCompletion,
}

/// Fixed effects specification for gsynth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GsynthForce {
    /// Unit fixed effects only
    #[default]
    Unit,
    /// Time fixed effects only
    Time,
    /// Two-way fixed effects
    TwoWay,
    /// No fixed effects
    None,
}

/// Configuration for generalized synthetic control.
#[derive(Debug, Clone)]
pub struct GsynthConfig {
    /// Number of latent factors (0 = auto-select via cross-validation)
    pub n_factors: usize,
    /// Maximum number of factors to try in cross-validation
    pub max_factors: usize,
    /// Use cross-validation for factor selection
    pub cross_validate: bool,
    /// Number of folds for cross-validation
    pub cv_folds: usize,
    /// Fixed effects specification
    pub force: GsynthForce,
    /// Estimation method
    pub estimator: GsynthEstimator,
    /// Compute bootstrap standard errors
    pub bootstrap_se: bool,
    /// Number of bootstrap replications
    pub n_bootstrap: usize,
    /// Minimum pre-treatment periods required
    pub min_pre_periods: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum iterations for EM algorithm
    pub max_iter: usize,
}

impl Default for GsynthConfig {
    fn default() -> Self {
        Self {
            n_factors: 0, // Auto-select
            max_factors: 10,
            cross_validate: true,
            cv_folds: 5,
            force: GsynthForce::TwoWay,
            estimator: GsynthEstimator::Ife,
            bootstrap_se: false,
            n_bootstrap: 200,
            min_pre_periods: 5,
            tolerance: 1e-5,
            max_iter: 1000,
        }
    }
}

/// Treatment effect for a single unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitEffect {
    /// Unit identifier
    pub unit: String,
    /// Treatment period
    pub treatment_time: i64,
    /// Treatment effects by time (time -> effect)
    pub effects: Vec<(i64, f64)>,
    /// Average treatment effect for this unit
    pub att: f64,
    /// Standard error (if bootstrapped)
    pub se: Option<f64>,
}

/// Result from generalized synthetic control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GsynthResult {
    /// Overall average treatment effect on treated
    pub att: f64,
    /// Standard error of ATT (if bootstrapped)
    pub att_se: Option<f64>,
    /// 95% confidence interval (if bootstrapped)
    pub att_ci: Option<(f64, f64)>,
    /// P-value (if bootstrapped)
    pub p_value: Option<f64>,
    /// Number of treated units
    pub n_treated: usize,
    /// Number of control units
    pub n_control: usize,
    /// Number of pre-treatment periods
    pub n_pre_periods: usize,
    /// Number of post-treatment periods
    pub n_post_periods: usize,
    /// Selected number of factors
    pub n_factors: usize,
    /// Cross-validation MSPE by number of factors (if CV used)
    pub cv_mspe: Vec<(usize, f64)>,
    /// Unit-specific treatment effects
    pub unit_effects: Vec<UnitEffect>,
    /// Estimated factors (time x n_factors)
    #[serde(skip)]
    pub factors: Option<Array2<f64>>,
    /// Estimated factor loadings (n_units x n_factors)
    #[serde(skip)]
    pub loadings: Option<Array2<f64>>,
    /// Covariate coefficients (if covariates included)
    pub beta: Vec<f64>,
    /// Covariate names
    pub covariate_names: Vec<String>,
    /// Pre-treatment MSPE
    pub pre_mspe: f64,
    /// Estimator used
    pub estimator: GsynthEstimator,
    /// Fixed effects specification
    pub force: GsynthForce,
    /// Dynamic treatment effects (time relative to treatment -> effect)
    pub dynamic_effects: Vec<(i64, f64)>,
}

impl fmt::Display for GsynthResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Generalized Synthetic Control (gsynth)")?;
        writeln!(f, "=======================================")?;
        writeln!(
            f,
            "Treated units: {}, Control units: {}",
            self.n_treated, self.n_control
        )?;
        writeln!(
            f,
            "Pre-treatment periods: {}, Post-treatment periods: {}",
            self.n_pre_periods, self.n_post_periods
        )?;
        writeln!(f, "Factors: {}", self.n_factors)?;
        writeln!(
            f,
            "Estimator: {:?}, Fixed Effects: {:?}",
            self.estimator, self.force
        )?;
        writeln!(f)?;

        writeln!(f, "AVERAGE TREATMENT EFFECT ON TREATED (ATT)")?;
        writeln!(f, "-----------------------------------------")?;
        write!(f, "ATT: {:.4}", self.att)?;
        if let Some(se) = self.att_se {
            write!(f, " (SE: {:.4})", se)?;
        }
        if let Some((lo, hi)) = self.att_ci {
            write!(f, " [95% CI: {:.4}, {:.4}]", lo, hi)?;
        }
        writeln!(f)?;

        if let Some(p) = self.p_value {
            writeln!(f, "P-value: {:.4}", p)?;
        }
        writeln!(f)?;

        if !self.beta.is_empty() {
            writeln!(f, "COVARIATE COEFFICIENTS")?;
            writeln!(f, "----------------------")?;
            for (i, name) in self.covariate_names.iter().enumerate() {
                writeln!(f, "{:<15} {:>10.4}", name, self.beta[i])?;
            }
            writeln!(f)?;
        }

        writeln!(f, "DYNAMIC EFFECTS (relative to treatment)")?;
        writeln!(f, "---------------------------------------")?;
        writeln!(f, "{:<10} {:>10}", "Period", "Effect")?;
        for (period, effect) in &self.dynamic_effects {
            writeln!(f, "{:<10} {:>10.4}", period, effect)?;
        }

        Ok(())
    }
}

/// Run generalized synthetic control for panel data with multiple treated units.
///
/// # Mathematical Background
///
/// The gsynth method uses an interactive fixed effects (IFE) model:
///
/// Y_it = α_i + λ_i'f_t + X_it'β + τ_it D_it + ε_it
///
/// Where:
/// - α_i: unit fixed effects
/// - λ_i: factor loadings (unit-specific)
/// - f_t: common factors (time-varying)
/// - X_it: observed covariates
/// - D_it: treatment indicator
/// - τ_it: treatment effect
///
/// The method estimates factors and loadings from control units during the
/// pre-treatment period, then uses them to impute counterfactual outcomes
/// for treated units.
///
/// # Arguments
///
/// * `dataset` - Panel dataset
/// * `y_col` - Outcome variable column
/// * `d_col` - Treatment indicator column (0/1)
/// * `unit_col` - Unit identifier column
/// * `time_col` - Time period column
/// * `x_cols` - Optional covariate columns
/// * `config` - Configuration options
///
/// # Returns
///
/// `GsynthResult` with estimated treatment effects and diagnostics.
///
/// # Data requirements
///
/// One row per (unit, time) of a balanced panel, with no missing value (null or NaN) in
/// the outcome, treatment, unit, time or covariate columns: a missing value is
/// `EconError::NullValues` naming the column, and a duplicate (unit, time) row or a
/// missing cell an `InvalidSpecification` naming the first one. Times are whole numbers
/// or dates (a fractional time is refused, naming the row); units may be of any
/// categorical dtype (text, integers, floats, booleans, dates) and are ordered as R's
/// `factor()` sorts them, which fixes the order of `unit_effects` and of the
/// cross-validation folds. Integer and boolean treatments are read as numbers. R's
/// gsynth imputes missing cells by EM, which p2a does not implement.
///
/// # References
///
/// - Xu, Y. (2017). "Generalized Synthetic Control Method: Causal Inference with
///   Interactive Fixed Effects Models." *Political Analysis*, 25(1), 57-76.
///
/// R equivalent: `gsynth::gsynth()`
pub fn run_gsynth(
    dataset: &Dataset,
    y_col: &str,
    d_col: &str,
    unit_col: &str,
    time_col: &str,
    x_cols: &[&str],
    config: GsynthConfig,
) -> EconResult<GsynthResult> {
    // One balanced panel, units in R factor() order (missing values,
    // missing cells and duplicate cells are refused).
    let GsynthPanel {
        units: unique_units,
        times: unique_times,
        y: y_mat,
        d: d_mat,
        x: x_panels,
    } = read_gsynth_panel(dataset.df(), y_col, d_col, unit_col, time_col, x_cols)?;

    let n_times = unique_times.len();
    let time_idx: std::collections::HashMap<i64, usize> = unique_times
        .iter()
        .enumerate()
        .map(|(i, &t)| (t, i))
        .collect();

    // Identify treated and control units
    let mut treated_units: Vec<usize> = Vec::new();
    let mut control_units: Vec<usize> = Vec::new();
    let mut treatment_times: Vec<i64> = Vec::new();

    for (ui, _unit) in unique_units.iter().enumerate() {
        let ever_treated = (0..n_times).any(|ti| d_mat[[ui, ti]] > 0.5);
        if ever_treated {
            treated_units.push(ui);
            // Find first treatment time
            for ti in 0..n_times {
                if d_mat[[ui, ti]] > 0.5 {
                    treatment_times.push(unique_times[ti]);
                    break;
                }
            }
        } else {
            control_units.push(ui);
        }
    }

    let n_treated = treated_units.len();
    let n_control = control_units.len();

    if n_treated == 0 {
        return Err(EconError::InvalidSpecification {
            message: "No treated units found".to_string(),
        });
    }
    if n_control < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Need at least 2 control units".to_string(),
        });
    }

    // Find common pre-treatment period (before any treatment)
    let first_treatment = treatment_times.iter().min().copied().unwrap();
    let first_treatment_idx = time_idx[&first_treatment];

    if first_treatment_idx < config.min_pre_periods {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Need at least {} pre-treatment periods, found {}",
                config.min_pre_periods, first_treatment_idx
            ),
        });
    }

    let n_pre_periods = first_treatment_idx;
    let n_post_periods = n_times - first_treatment_idx;

    // min_pre_periods = 0 lets a panel treated from its first period through;
    // the factor model needs at least one pre-treatment period.
    if n_pre_periods == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Need at least 1 pre-treatment period before the first treated period"
                .to_string(),
        });
    }

    // More factors than min(control units, pre-periods) cannot be estimated.
    if config.n_factors > 0 {
        let bound = n_control.min(n_pre_periods);
        if config.n_factors > bound {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "n_factors = {} is more than the data can identify: at most {bound} (the \
                     smaller of {n_control} control units and {n_pre_periods} pre-treatment \
                     periods). Use fewer factors.",
                    config.n_factors
                ),
            });
        }
    }

    // Cross-validation needs two folds; folds beyond the number of control
    // units were empty, so the fold count is clamped to it.
    let cv_folds = if config.cross_validate && config.n_factors == 0 {
        if config.cv_folds < 2 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "cv_folds must be at least 2 for cross-validation (got {})",
                    config.cv_folds
                ),
            });
        }
        config.cv_folds.min(n_control)
    } else {
        config.cv_folds
    };

    // Covariate panels, read with the outcome
    let (x_mat, covariate_names) = if !x_cols.is_empty() {
        (
            Some(x_panels),
            x_cols.iter().map(|s| s.to_string()).collect(),
        )
    } else {
        (None, Vec::new())
    };

    // Select number of factors via cross-validation if requested
    let (n_factors, cv_mspe) = if config.cross_validate && config.n_factors == 0 {
        select_factors_cv(
            &y_mat,
            &control_units,
            n_pre_periods,
            config.max_factors,
            cv_folds,
            &config,
        )?
    } else if config.n_factors > 0 {
        (config.n_factors, Vec::new())
    } else {
        (2, Vec::new()) // Default to 2 factors
    };

    // Estimate IFE model using control units in pre-treatment period
    let (factors_pre, loadings, beta, _residuals) = estimate_ife(
        &y_mat,
        x_mat.as_ref(),
        &control_units,
        n_pre_periods,
        n_factors,
        &config,
    )?;

    // Downstream code uses the factors actually estimated: the SVD caps the
    // rank at min(control units, pre-periods), which a CV choice or the
    // default of 2 can exceed. For an explicit n_factors (checked above) the
    // two agree.
    let n_factors = factors_pre.ncols();

    // Extend factors to all time periods using control units
    // For post-treatment periods, project control outcomes onto factor space
    let factors = extend_factors_to_all_periods(
        &y_mat,
        &factors_pre,
        &loadings,
        &control_units,
        n_pre_periods,
        n_times,
        n_factors,
        &config,
    )?;

    // Compute counterfactuals and treatment effects for treated units
    let mut unit_effects: Vec<UnitEffect> = Vec::new();
    let mut all_effects: Vec<f64> = Vec::new();
    let mut dynamic_effect_map: std::collections::HashMap<i64, Vec<f64>> =
        std::collections::HashMap::new();

    for (idx, &ui) in treated_units.iter().enumerate() {
        let treatment_time = treatment_times[idx];
        let treatment_idx = time_idx[&treatment_time];

        // Estimate loadings for this treated unit from pre-treatment data
        let unit_loadings =
            estimate_unit_loadings(&y_mat.row(ui).to_owned(), &factors, treatment_idx)?;

        // Compute counterfactual and effects
        let mut effects: Vec<(i64, f64)> = Vec::new();
        let mut post_effects: Vec<f64> = Vec::new();

        // Compute pre-treatment mean for this unit (for fixed effects)
        let pre_mean: f64 = (0..treatment_idx)
            .map(|t| y_mat[[ui, t]])
            .filter(|v| !v.is_nan())
            .sum::<f64>()
            / treatment_idx.max(1) as f64;

        // Compute mean of fitted values in pre-treatment
        let pre_fitted_mean: f64 = if n_factors > 0 {
            (0..treatment_idx)
                .map(|t| {
                    (0..n_factors)
                        .map(|r| unit_loadings[r] * factors[[t, r]])
                        .sum::<f64>()
                })
                .sum::<f64>()
                / treatment_idx.max(1) as f64
        } else {
            0.0
        };

        for ti in 0..n_times {
            let time = unique_times[ti];
            let y_actual = y_mat[[ui, ti]];

            // Counterfactual: λ_i' * f_t + intercept adjustment
            let y_fitted: f64 = (0..n_factors)
                .map(|r| unit_loadings[r] * factors[[ti, r]])
                .sum();

            // Adjust for fixed effects
            let y_counter = if matches!(config.force, GsynthForce::Unit | GsynthForce::TwoWay) {
                y_fitted + pre_mean - pre_fitted_mean
            } else {
                y_fitted
            };

            if !y_actual.is_nan() {
                let effect = y_actual - y_counter;

                if ti >= treatment_idx {
                    // Post-treatment
                    effects.push((time, effect));
                    post_effects.push(effect);
                    all_effects.push(effect);

                    let rel_time = (ti as i64) - (treatment_idx as i64);
                    dynamic_effect_map.entry(rel_time).or_default().push(effect);
                }
            }
        }

        let att_unit = if post_effects.is_empty() {
            0.0
        } else {
            post_effects.iter().sum::<f64>() / post_effects.len() as f64
        };

        unit_effects.push(UnitEffect {
            unit: unique_units[ui].clone(),
            treatment_time,
            effects,
            att: att_unit,
            se: None,
        });
    }

    // Overall ATT
    let att = if all_effects.is_empty() {
        0.0
    } else {
        all_effects.iter().sum::<f64>() / all_effects.len() as f64
    };

    // Dynamic effects (average by relative time)
    let mut dynamic_effects: Vec<(i64, f64)> = dynamic_effect_map
        .iter()
        .map(|(&rel_t, effects)| (rel_t, effects.iter().sum::<f64>() / effects.len() as f64))
        .collect();
    dynamic_effects.sort_by_key(|(t, _)| *t);

    // Compute pre-treatment MSPE
    let pre_mspe = compute_pre_mspe(&y_mat, &factors, &loadings, &control_units, n_pre_periods);

    // Bootstrap for standard errors if requested
    let (att_se, att_ci, p_value) = if config.bootstrap_se {
        bootstrap_gsynth(
            &y_mat,
            &d_mat,
            &treated_units,
            &control_units,
            n_pre_periods,
            n_factors,
            &config,
        )?
    } else {
        (None, None, None)
    };

    Ok(GsynthResult {
        att,
        att_se,
        att_ci,
        p_value,
        n_treated,
        n_control,
        n_pre_periods,
        n_post_periods,
        n_factors,
        cv_mspe,
        unit_effects,
        factors: Some(factors),
        loadings: Some(loadings),
        beta,
        covariate_names,
        pre_mspe,
        estimator: config.estimator,
        force: config.force,
        dynamic_effects,
    })
}

/// A gsynth panel read through the shared extractor: balanced, one row per (unit, time).
struct GsynthPanel {
    /// Unit labels in R `factor()` order.
    units: Vec<String>,
    /// The distinct time periods, ascending.
    times: Vec<i64>,
    /// Outcome panel (units x periods).
    y: Array2<f64>,
    /// Treatment panel (units x periods).
    d: Array2<f64>,
    /// One panel (units x periods) per covariate, in `x_cols` order.
    x: Vec<Array2<f64>>,
}

/// Read a gsynth panel with one `ColumnRequest` (`NaPolicy::Fail`): a missing value in
/// any column is `NullValues` naming it, a fractional or text time an
/// `InvalidSpecification`. Duplicate (unit, time) rows and missing cells are refused
/// before any units x periods buffer is allocated; every step is O(n log n).
fn read_gsynth_panel(
    df: &DataFrame,
    y_col: &str,
    d_col: &str,
    unit_col: &str,
    time_col: &str,
    x_cols: &[&str],
) -> EconResult<GsynthPanel> {
    let cols = ColumnRequest::new(df)
        .factor(unit_col)
        .time_index(time_col)
        .numeric(y_col)
        .numeric(d_col)
        .numerics(x_cols)
        .na(NaPolicy::Fail)
        .extract()?;
    let n = cols.n();
    let unit = cols.factor(unit_col);
    let time = cols.time_index(time_col);
    let units = unit.levels.clone();

    let mut times: Vec<i64> = crate::resource::try_with_capacity(n, "gsynth time periods")?;
    times.extend_from_slice(time);
    times.sort_unstable();
    times.dedup();

    // The rows in (unit level, time) order; equal neighbours are duplicates.
    let key = |r: usize| (unit.codes[r], time[r]);
    let mut order: Vec<usize> = crate::resource::try_with_capacity(n, "gsynth panel order")?;
    order.extend(0..n);
    order.sort_unstable_by_key(|&r| key(r));
    let mut duplicates = 0usize;
    let mut first_duplicate = None;
    for pair in order.windows(2) {
        if key(pair[0]) == key(pair[1]) {
            duplicates += 1;
            first_duplicate.get_or_insert(pair[1]);
        }
    }
    if let Some(r) = first_duplicate {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "gsynth needs one row per (unit, time): {duplicates} duplicate row(s) \
                 (first: unit '{}', time {}). Remove the duplicates first.",
                unit.label(r),
                time[r]
            ),
        });
    }

    let n_units = units.len();
    let n_times = times.len();
    // Without duplicates every row is its own cell, so n <= units x periods.
    let missing = (n_units as u128 * n_times as u128).saturating_sub(n as u128);
    if missing > 0 {
        // The first absent cell in (unit level, time) order. Every cell before it
        // consumes one row of `order`, so the walk is O(n).
        let mut next = order.iter().peekable();
        let mut first = None;
        'units: for u in 0..n_units {
            crate::resource::checkpoint()?;
            for &t in &times {
                match next.peek() {
                    Some(&&r) if unit.codes[r] as usize == u && time[r] == t => {
                        next.next();
                    }
                    _ => {
                        first = Some((u, t));
                        break 'units;
                    }
                }
            }
        }
        let (u, t) = first.ok_or_else(|| {
            EconError::Internal("gsynth panel: no missing cell found".to_string())
        })?;
        return Err(EconError::InvalidSpecification {
            message: format!(
                "gsynth needs a balanced panel: {missing} (unit, time) cell(s) are missing \
                 (first: unit '{}', time {t}). p2a's gsynth has no estimator for \
                 unbalanced panels (R's gsynth imputes missing cells); fill or drop them \
                 first.",
                units[u]
            ),
        });
    }

    // Balanced: order[u * n_times + j] is the row of (units[u], times[j]).
    let panel = |values: &[f64], what: &str| -> EconResult<Array2<f64>> {
        let mut m = crate::resource::try_zeros2(n_units, n_times, what)?;
        for (cell, &r) in m.iter_mut().zip(&order) {
            *cell = values[r];
        }
        Ok(m)
    };
    let y = panel(
        cols.numeric(y_col),
        "gsynth outcome panel (units x periods)",
    )?;
    let d = panel(
        cols.numeric(d_col),
        "gsynth treatment panel (units x periods)",
    )?;
    let mut x = Vec::with_capacity(x_cols.len());
    for col in x_cols {
        x.push(panel(
            cols.numeric(col),
            "gsynth covariate panel (units x periods)",
        )?);
    }
    Ok(GsynthPanel {
        units,
        times,
        y,
        d,
        x,
    })
}

/// Select optimal number of factors via k-fold cross-validation.
///
/// Implements the cross-validation procedure from Xu (2017) for selecting
/// the number of latent factors in the interactive fixed effects model.
///
/// # Algorithm
/// 1. Randomly split control units into k folds
/// 2. For each candidate number of factors r = 0, 1, ..., max_factors:
///    - For each fold:
///      - Estimate factors using training units
///      - Compute prediction MSPE on test units
///    - Average MSPE across folds
/// 3. Select r with minimum average MSPE
///
/// # Arguments
/// * `y_mat` - Outcome matrix (N × T)
/// * `control_units` - Indices of control units
/// * `n_pre` - Number of pre-treatment periods
/// * `max_factors` - Maximum number of factors to consider
/// * `n_folds` - Number of cross-validation folds (default: 5)
/// * `config` - GSynth configuration
///
/// # Returns
/// Tuple of (optimal_r, cv_results) where cv_results contains (r, MSPE) pairs
///
/// # References
/// Xu, Y. (2017). "Generalized Synthetic Control Method: Causal Inference with
/// Interactive Fixed Effects Models." Political Analysis, 25(1), 57-76.
fn select_factors_cv(
    y_mat: &Array2<f64>,
    control_units: &[usize],
    n_pre: usize,
    max_factors: usize,
    n_folds: usize,
    config: &GsynthConfig,
) -> EconResult<(usize, Vec<(usize, f64)>)> {
    use rand::SeedableRng;
    use rand::seq::SliceRandom;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut indices: Vec<usize> = (0..control_units.len()).collect();
    indices.shuffle(&mut rng);

    let fold_size = control_units.len() / n_folds;
    let mut cv_results: Vec<(usize, f64)> = Vec::new();

    for r in 0..=max_factors.min(n_pre - 1) {
        let mut fold_mspe: Vec<f64> = Vec::new();

        for fold in 0..n_folds {
            crate::resource::checkpoint()?;

            let start = fold * fold_size;
            let end = if fold == n_folds - 1 {
                control_units.len()
            } else {
                (fold + 1) * fold_size
            };

            let test_indices: Vec<usize> = indices[start..end].to_vec();
            let train_indices: Vec<usize> = indices
                .iter()
                .enumerate()
                .filter(|&(i, _)| i < start || i >= end)
                .map(|(_, &idx)| idx)
                .collect();

            let train_units: Vec<usize> = train_indices.iter().map(|&i| control_units[i]).collect();

            // Estimate on training set
            let (factors, _, _, _) = match estimate_ife(y_mat, None, &train_units, n_pre, r, config)
            {
                Ok(res) => res,
                Err(e @ EconError::Cancelled(_)) => return Err(e),
                Err(_) => continue,
            };

            // Evaluate on test set
            let mut mspe = 0.0;
            let mut count = 0;

            // Actual number of factors extracted may be less than requested
            let actual_r = factors.ncols();

            for &test_idx in &test_indices {
                let ui = control_units[test_idx];
                let unit_loadings =
                    match estimate_unit_loadings(&y_mat.row(ui).to_owned(), &factors, n_pre) {
                        Ok(l) => l,
                        Err(_) => continue,
                    };

                for ti in 0..n_pre {
                    let y_actual = y_mat[[ui, ti]];
                    if !y_actual.is_nan() {
                        let y_pred: f64 = (0..actual_r)
                            .map(|k| unit_loadings[k] * factors[[ti, k]])
                            .sum();
                        mspe += (y_actual - y_pred).powi(2);
                        count += 1;
                    }
                }
            }

            if count > 0 {
                fold_mspe.push(mspe / count as f64);
            }
        }

        if !fold_mspe.is_empty() {
            let avg_mspe = fold_mspe.iter().sum::<f64>() / fold_mspe.len() as f64;
            cv_results.push((r, avg_mspe));
        }
    }

    // Select r with minimum MSPE
    let best_r = cv_results
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|&(r, _)| r)
        .unwrap_or(2);

    Ok((best_r, cv_results))
}

/// Estimate interactive fixed effects (IFE) model on control units.
///
/// Implements the factor model: Y_it = λ_i' f_t + ε_it
/// where λ_i are unit-specific loadings and f_t are time-varying factors.
///
/// # Model
/// The interactive fixed effects model captures unobserved heterogeneity through
/// latent factors that vary over time and affect units differently:
/// - f_t ∈ ℝʳ: r-dimensional factor at time t
/// - λ_i ∈ ℝʳ: unit i's factor loadings (how sensitive it is to each factor)
///
/// # Algorithm
/// 1. Apply fixed effects transformation based on `config.force`:
///    - None: No demeaning
///    - Unit: Subtract unit means (within transformation)
///    - Time: Subtract time means
///    - TwoWay: Subtract unit means, time means, add grand mean
/// 2. Extract factors and loadings via SVD/PCA
/// 3. Compute residuals
///
/// # Arguments
/// * `y_mat` - Outcome matrix (N × T)
/// * `x_mat` - Optional covariate matrices (reserved for future use)
/// * `control_units` - Indices of control units to use for estimation
/// * `n_pre` - Number of pre-treatment periods
/// * `n_factors` - Number of factors to extract
/// * `config` - GSynth configuration
///
/// # Returns
/// Tuple of (factors, loadings, beta_coefficients, residuals)
/// - factors: T × r matrix of time factors
/// - loadings: N × r matrix of unit loadings
/// - beta: Covariate coefficients (empty if no covariates)
/// - residuals: N × T matrix of residuals
fn estimate_ife(
    y_mat: &Array2<f64>,
    _x_mat: Option<&Vec<Array2<f64>>>,
    control_units: &[usize],
    n_pre: usize,
    n_factors: usize,
    config: &GsynthConfig,
) -> EconResult<(Array2<f64>, Array2<f64>, Vec<f64>, Array2<f64>)> {
    // Extract control outcomes for pre-treatment period
    let n_control = control_units.len();
    let mut y_control = Array2::<f64>::zeros((n_control, n_pre));

    for (i, &ui) in control_units.iter().enumerate() {
        for t in 0..n_pre {
            y_control[[i, t]] = y_mat[[ui, t]];
        }
    }

    // Handle fixed effects
    let y_demean = match config.force {
        GsynthForce::None => y_control.clone(),
        GsynthForce::Unit => {
            let mut y_dm = y_control.clone();
            for i in 0..n_control {
                let row_mean: f64 =
                    y_dm.row(i).iter().filter(|&&v| !v.is_nan()).sum::<f64>() / n_pre as f64;
                for t in 0..n_pre {
                    y_dm[[i, t]] -= row_mean;
                }
            }
            y_dm
        }
        GsynthForce::Time => {
            let mut y_dm = y_control.clone();
            for t in 0..n_pre {
                let col_mean: f64 = (0..n_control)
                    .map(|i| y_dm[[i, t]])
                    .filter(|&v| !v.is_nan())
                    .sum::<f64>()
                    / n_control as f64;
                for i in 0..n_control {
                    y_dm[[i, t]] -= col_mean;
                }
            }
            y_dm
        }
        GsynthForce::TwoWay => {
            let mut y_dm = y_control.clone();
            // Grand mean
            let grand_mean: f64 =
                y_dm.iter().filter(|&&v| !v.is_nan()).sum::<f64>() / (n_control * n_pre) as f64;
            // Row means
            let row_means: Vec<f64> = (0..n_control)
                .map(|i| y_dm.row(i).sum() / n_pre as f64)
                .collect();
            // Col means
            let col_means: Vec<f64> = (0..n_pre)
                .map(|t| (0..n_control).map(|i| y_dm[[i, t]]).sum::<f64>() / n_control as f64)
                .collect();
            // Demean
            for i in 0..n_control {
                for t in 0..n_pre {
                    y_dm[[i, t]] -= row_means[i] + col_means[t] - grand_mean;
                }
            }
            y_dm
        }
    };

    // SVD for factor extraction
    let (factors, loadings) = if n_factors > 0 {
        extract_factors_svd(&y_demean, n_factors)?
    } else {
        // No factors - return empty matrices
        (Array2::zeros((n_pre, 0)), Array2::zeros((n_control, 0)))
    };

    // Actual number of factors extracted (may be less than requested)
    let actual_n_factors = factors.ncols();

    // Compute residuals
    let mut residuals = y_control.clone();
    for i in 0..n_control {
        for t in 0..n_pre {
            let fitted: f64 = (0..actual_n_factors)
                .map(|r| loadings[[i, r]] * factors[[t, r]])
                .sum();
            residuals[[i, t]] -= fitted;
        }
    }

    Ok((factors, loadings, Vec::new(), residuals))
}

/// Extract latent factors and loadings using principal components analysis.
///
/// Implements factor extraction via eigenvalue decomposition of Y'Y.
/// This is the core of the interactive fixed effects model.
///
/// # Algorithm
/// 1. Compute Y'Y (T × T covariance matrix of time points)
/// 2. Use power iteration with deflation to extract top r eigenvectors
/// 3. These eigenvectors become the time factors f_t
/// 4. Loadings λ_i are computed as Y × factors / eigenvalue
///
/// # Normalization
/// Uses PC1 normalization: factors have unit variance, loadings capture scale.
/// This follows the convention in Bai (2009) for factor models.
///
/// # Arguments
/// * `y` - Demeaned outcome matrix (N × T), where N = control units, T = pre-periods
/// * `n_factors` - Number of factors to extract (actual may be less if rank-deficient)
///
/// # Returns
/// Tuple of (factors, loadings):
/// - factors: T × r matrix of time-varying factors
/// - loadings: N × r matrix of unit-specific factor loadings
///
/// # References
/// - Bai, J. (2009). "Panel Data Models with Interactive Fixed Effects."
///   Econometrica, 77(4), 1229-1279.
fn extract_factors_svd(
    y: &Array2<f64>,
    n_factors: usize,
) -> EconResult<(Array2<f64>, Array2<f64>)> {
    let (n_rows, n_cols) = y.dim(); // N x T (control_units x pre_periods)
    let r = n_factors.min(n_rows.min(n_cols));

    if r == 0 {
        return Ok((Array2::zeros((n_cols, 0)), Array2::zeros((n_rows, 0))));
    }

    // For the interactive fixed effects model:
    // Y = Λ F' + ε where Λ is N x r (loadings), F is T x r (factors)
    //
    // We perform PCA on Y to extract factors and loadings.
    // The PC1 normalization convention: factors have unit variance, loadings capture scale.

    // Compute Y'Y (T x T) for factor extraction; T can be large next to N, so
    // the T x T matrix (and its working copy) is checked first.
    let what = "gsynth factor covariance (pre-treatment periods x pre-treatment periods)";
    crate::resource::check_alloc::<f64>(
        crate::resource::checked_len(&[n_cols, n_cols], what)?,
        what,
    )?;
    let yty = y.t().dot(y); // T x T

    // Simple eigenvalue decomposition via power iteration with deflation
    let mut factors = Array2::<f64>::zeros((n_cols, r));
    let mut eigenvalues = Vec::with_capacity(r);

    let mut mat = yty.clone();

    for k in 0..r {
        // Each factor costs up to 200 power-iteration steps and a deflation on
        // the T x T matrix; an explicit n_factors reaches this loop without the
        // per-fold checkpoints of select_factors_cv.
        crate::resource::checkpoint()?;

        // Power iteration for dominant eigenvector
        let mut v = Array1::<f64>::from_elem(n_cols, 1.0 / (n_cols as f64).sqrt());

        for _ in 0..200 {
            let v_new = mat.dot(&v);
            let norm = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break;
            }
            let v_normalized = &v_new / norm;

            // Check convergence
            let diff: f64 = v
                .iter()
                .zip(v_normalized.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            v = v_normalized;

            if diff < 1e-10 {
                break;
            }
        }

        // Normalize eigenvector
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            v = &v / norm;
        }

        // Get eigenvalue
        let eigenvalue = v.dot(&mat.dot(&v));
        eigenvalues.push(eigenvalue.max(0.0));

        factors.column_mut(k).assign(&v);

        // Deflate: subtract the contribution of this eigenvector
        for i in 0..n_cols {
            for j in 0..n_cols {
                mat[[i, j]] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    // Normalize factors to have sqrt(T) norm (standard PCA convention)
    // F_normalized = F * sqrt(T), so F'F / T = I
    let t_sqrt = (n_cols as f64).sqrt();
    for k in 0..r {
        for t in 0..n_cols {
            factors[[t, k]] *= t_sqrt;
        }
    }

    // Compute loadings: Λ = Y * F / T (N x T) * (T x r) / T = N x r
    // This gives us Λ such that Y ≈ Λ * F'
    let loadings = y.dot(&factors) / (n_cols as f64);

    Ok((factors, loadings))
}

/// Estimate factor loadings for a single unit via OLS.
///
/// Given factors f_t and a unit's outcome time series y_it,
/// estimates the unit's factor loadings λ_i by OLS regression:
///
///   y_i = F λ_i + ε_i
///
/// where F is the T × r factor matrix and λ_i is the r × 1 loading vector.
///
/// # Arguments
/// * `y_unit` - Unit's outcome time series
/// * `factors` - T × r matrix of time factors
/// * `n_pre` - Number of pre-treatment periods (uses only these for estimation)
///
/// # Returns
/// Vector of r factor loadings for this unit
fn estimate_unit_loadings(
    y_unit: &Array1<f64>,
    factors: &Array2<f64>,
    n_pre: usize,
) -> EconResult<Vec<f64>> {
    use crate::linalg::matrix_ops::safe_inverse;

    let n_factors = factors.ncols();
    if n_factors == 0 {
        return Ok(Vec::new());
    }

    // Estimate λ_i by OLS: y = F * λ + ε
    // Pre-treatment only
    let f_pre = factors.slice(ndarray::s![0..n_pre, ..]).to_owned();
    let y_pre: Vec<f64> = (0..n_pre).map(|t| y_unit[t]).collect();

    let ftf = f_pre.t().dot(&f_pre);

    // Add small regularization to avoid numerical issues
    let mut ftf_reg = ftf.clone();
    for k in 0..n_factors {
        ftf_reg[[k, k]] += 1e-8;
    }

    let (ftf_inv, _) = safe_inverse(&ftf_reg.view())?;

    let y_arr = Array1::from_vec(y_pre);
    let fty = f_pre.t().dot(&y_arr);
    let loadings = ftf_inv.dot(&fty);

    Ok(loadings.to_vec())
}

/// Extend estimated factors from pre-treatment to post-treatment periods.
///
/// Factors are initially estimated using only pre-treatment control outcomes.
/// This function extrapolates factors to post-treatment periods using control
/// unit outcomes, which are unaffected by treatment.
///
/// # Algorithm
/// For each post-treatment period t:
/// 1. Extract control unit outcomes y_t = [y_{1t}, ..., y_{Jt}]'
/// 2. Apply fixed effects adjustment based on config.force
/// 3. Estimate factor f_t = (Λ'Λ)⁻¹ Λ' y_t
///
/// # Arguments
/// * `y_mat` - Full outcome matrix (N × T)
/// * `factors_pre` - Pre-treatment factors (T₀ × r)
/// * `loadings` - Unit loadings estimated from controls (J × r)
/// * `control_units` - Indices of control units
/// * `n_pre` - Number of pre-treatment periods
/// * `n_times` - Total number of time periods
/// * `n_factors` - Number of factors
/// * `config` - GSynth configuration
///
/// # Returns
/// Extended factors matrix (T × r) covering all time periods
fn extend_factors_to_all_periods(
    y_mat: &Array2<f64>,
    factors_pre: &Array2<f64>,
    loadings: &Array2<f64>,
    control_units: &[usize],
    n_pre: usize,
    n_times: usize,
    n_factors: usize,
    config: &GsynthConfig,
) -> EconResult<Array2<f64>> {
    use crate::linalg::matrix_ops::safe_inverse;

    if n_factors == 0 {
        return Ok(Array2::zeros((n_times, 0)));
    }

    // Full factors matrix
    let mut factors = Array2::<f64>::zeros((n_times, n_factors));

    // Copy pre-treatment factors
    for t in 0..n_pre {
        for r in 0..n_factors {
            factors[[t, r]] = factors_pre[[t, r]];
        }
    }

    // For post-treatment periods, estimate factors from control outcomes
    // f_t = (Λ'Λ)^-1 Λ' y_t for control units
    let ltl = loadings.t().dot(loadings);

    // Add regularization
    let mut ltl_reg = ltl.clone();
    for r in 0..n_factors {
        ltl_reg[[r, r]] += 1e-8;
    }

    let (ltl_inv, _) = safe_inverse(&ltl_reg.view())?;

    // Compute pre-treatment means for unit FE adjustment
    let pre_means: Vec<f64> = control_units
        .iter()
        .map(|&ui| (0..n_pre).map(|s| y_mat[[ui, s]]).sum::<f64>() / n_pre as f64)
        .collect();

    for t in n_pre..n_times {
        // Extract control outcomes at time t
        let y_t: Array1<f64> = Array1::from_iter(control_units.iter().map(|&ui| y_mat[[ui, t]]));

        // Handle fixed effects
        let y_t_adj = match config.force {
            GsynthForce::Unit | GsynthForce::TwoWay => {
                // Subtract unit means from pre-treatment
                let adj: Array1<f64> = Array1::from_iter(
                    control_units
                        .iter()
                        .enumerate()
                        .map(|(i, &ui)| y_mat[[ui, t]] - pre_means[i]),
                );
                adj
            }
            _ => y_t.clone(),
        };

        // f_t = (Λ'Λ)^-1 Λ' y_t
        let lt_y = loadings.t().dot(&y_t_adj);
        let f_t = ltl_inv.dot(&lt_y);

        for r in 0..n_factors {
            factors[[t, r]] = f_t[r];
        }
    }

    Ok(factors)
}

/// Compute pre-treatment MSPE.
fn compute_pre_mspe(
    y_mat: &Array2<f64>,
    factors: &Array2<f64>,
    loadings: &Array2<f64>,
    control_units: &[usize],
    n_pre: usize,
) -> f64 {
    let n_factors = factors.ncols();
    let mut mse = 0.0;
    let mut count = 0;

    for (i, &ui) in control_units.iter().enumerate() {
        for t in 0..n_pre {
            let y_actual = y_mat[[ui, t]];
            if !y_actual.is_nan() {
                let y_fitted: f64 = (0..n_factors)
                    .map(|r| loadings[[i, r]] * factors[[t, r]])
                    .sum();
                mse += (y_actual - y_fitted).powi(2);
                count += 1;
            }
        }
    }

    if count > 0 { mse / count as f64 } else { 0.0 }
}

/// Bootstrap for uncertainty quantification.
///
/// When `config.bootstrap_se == false` this is a no-op and returns `None`
/// for every output, matching the documented contract. When `bootstrap_se`
/// is requested but not yet implemented, the function returns
/// `EconError::NotImplemented` so callers can surface the gap instead of
/// silently dropping the user's request and handing back `None` SEs.
fn bootstrap_gsynth(
    _y_mat: &Array2<f64>,
    _d_mat: &Array2<f64>,
    _treated_units: &[usize],
    _control_units: &[usize],
    _n_pre: usize,
    _n_factors: usize,
    config: &GsynthConfig,
) -> EconResult<(Option<f64>, Option<(f64, f64)>, Option<f64>)> {
    if !config.bootstrap_se {
        return Ok((None, None, None));
    }

    // Bootstrap requested but the resample+re-estimate loop is not yet
    // wired in. Surface this clearly rather than silently producing `None`
    // confidence intervals that look like a successful run.
    Err(EconError::NotImplemented(
        "gsynth bootstrap standard errors are not yet implemented; \
         set bootstrap_se=false or use the analytical placeholder"
            .to_string(),
    ))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple panel dataset for testing.
    fn create_test_panel() -> Dataset {
        // Create a panel with:
        // - 1 treated unit (A)
        // - 2 donor units (B, C)
        // - 6 time periods (1-6), treatment at time 4
        // - Treatment effect of ~5 units

        // Pre-treatment: A is 0.5*B + 0.5*C
        // Post-treatment: A diverges by +5

        let df = df! {
            "unit" => ["A", "A", "A", "A", "A", "A",
                       "B", "B", "B", "B", "B", "B",
                       "C", "C", "C", "C", "C", "C"],
            "time" => [1i64, 2, 3, 4, 5, 6,
                       1, 2, 3, 4, 5, 6,
                       1, 2, 3, 4, 5, 6],
            "outcome" => [
                // Unit A: pre = 0.5*B + 0.5*C, post = pre + 5
                10.0, 11.0, 12.0, 18.0, 19.0, 20.0,  // A (treatment effect = 5 starting at t=4)
                // Unit B
                8.0, 10.0, 12.0, 14.0, 16.0, 18.0,   // B
                // Unit C
                12.0, 12.0, 12.0, 12.0, 12.0, 12.0,  // C
            ],
            "x1" => [
                // Predictor that matches A = 0.5*B + 0.5*C
                5.0, 5.0, 5.0, 5.0, 5.0, 5.0,  // A: mean = 5
                4.0, 4.0, 4.0, 4.0, 4.0, 4.0,  // B: mean = 4
                6.0, 6.0, 6.0, 6.0, 6.0, 6.0,  // C: mean = 6
            ],
        }
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_basic_synth() {
        let dataset = create_test_panel();

        let predictors = vec![PredictorSpec::new("x1")];

        let config = SynthConfig {
            treatment_time: 4,
            treated_unit: "A".to_string(),
            v_method: VOptimization::Equal,
            run_placebos: false,
            ..Default::default()
        };

        let result =
            run_synthetic_control(&dataset, "outcome", "unit", "time", &predictors, config)
                .unwrap();

        // Check basic structure
        assert_eq!(result.treated_unit, "A");
        assert_eq!(result.treatment_time, 4);
        assert_eq!(result.n_donors, 2);
        assert_eq!(result.n_pre_periods, 3); // times 1, 2, 3
        assert_eq!(result.n_post_periods, 3); // times 4, 5, 6

        // Weights should sum to 1
        let weight_sum: f64 = result.all_unit_weights.iter().map(|(_, w)| w).sum();
        assert!(
            (weight_sum - 1.0).abs() < 0.01,
            "Weights should sum to 1, got {}",
            weight_sum
        );

        // With equal V weights and predictor x1, optimal weights should be ~0.5 each
        // since A's x1 = 5 = 0.5 * 4 + 0.5 * 6
        for (unit, weight) in &result.all_unit_weights {
            assert!(
                (*weight - 0.5).abs() < 0.2,
                "Unit {} weight should be ~0.5, got {}",
                unit,
                weight
            );
        }

        // Treatment effect should be positive (around 5)
        assert!(
            result.average_effect > 0.0,
            "Average effect should be positive, got {}",
            result.average_effect
        );

        println!("{}", result);
    }

    #[test]
    fn test_synth_with_data_driven_v() {
        let dataset = create_test_panel();

        let predictors = vec![PredictorSpec::new("x1")];

        let config = SynthConfig {
            treatment_time: 4,
            treated_unit: "A".to_string(),
            v_method: VOptimization::DataDriven,
            run_placebos: false,
            ..Default::default()
        };

        let result =
            run_synthetic_control(&dataset, "outcome", "unit", "time", &predictors, config)
                .unwrap();

        // Should still produce valid results
        assert_eq!(result.n_donors, 2);

        let weight_sum: f64 = result.all_unit_weights.iter().map(|(_, w)| w).sum();
        assert!((weight_sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_synth_predictor_balance() {
        let dataset = create_test_panel();

        let predictors = vec![PredictorSpec::new("x1")];

        let config = SynthConfig {
            treatment_time: 4,
            treated_unit: "A".to_string(),
            v_method: VOptimization::Equal,
            run_placebos: false,
            ..Default::default()
        };

        let result =
            run_synthetic_control(&dataset, "outcome", "unit", "time", &predictors, config)
                .unwrap();

        // Check predictor balance
        assert!(!result.predictor_balance.is_empty());

        let balance = &result.predictor_balance[0];
        assert!((balance.treated_value - 5.0).abs() < 0.1);
        // Synthetic should be close to 5.0 too (since weights ≈ 0.5, 0.5 and values are 4, 6)
        assert!(
            (balance.synthetic_value - 5.0).abs() < 0.5,
            "Synthetic x1 should be ~5.0, got {}",
            balance.synthetic_value
        );
    }

    #[test]
    fn test_synth_invalid_unit() {
        let dataset = create_test_panel();
        let predictors = vec![PredictorSpec::new("x1")];

        let config = SynthConfig {
            treatment_time: 4,
            treated_unit: "NonexistentUnit".to_string(),
            ..Default::default()
        };

        let result =
            run_synthetic_control(&dataset, "outcome", "unit", "time", &predictors, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_synth_no_pre_periods() {
        let dataset = create_test_panel();
        let predictors = vec![PredictorSpec::new("x1")];

        // Treatment time before any data
        let config = SynthConfig {
            treatment_time: 0,
            treated_unit: "A".to_string(),
            ..Default::default()
        };

        let result =
            run_synthetic_control(&dataset, "outcome", "unit", "time", &predictors, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_predictor_spec() {
        let spec = PredictorSpec::new("gdp");
        assert_eq!(spec.column, "gdp");
        assert!(spec.time_window.is_none());

        let spec_window = PredictorSpec::with_window("population", 1980, 1990);
        assert_eq!(spec_window.column, "population");
        assert_eq!(spec_window.time_window, Some((1980, 1990)));
    }

    #[test]
    fn test_display() {
        let dataset = create_test_panel();
        let predictors = vec![PredictorSpec::new("x1")];

        let config = SynthConfig {
            treatment_time: 4,
            treated_unit: "A".to_string(),
            v_method: VOptimization::Equal,
            run_placebos: false,
            ..Default::default()
        };

        let result =
            run_synthetic_control(&dataset, "outcome", "unit", "time", &predictors, config)
                .unwrap();

        // Test Display trait
        let output = format!("{}", result);
        assert!(output.contains("Synthetic Control"));
        assert!(output.contains("Treated Unit: A"));
        assert!(output.contains("PREDICTOR BALANCE"));
    }

    // =========================================================================
    // Tolerance, iteration bounds and cancellation (T0-WP11)
    // =========================================================================

    /// Runs `f` under a budget whose token is already cancelled. Shared with
    /// `gsynth_tests`.
    pub(super) fn cancelled<R>(f: impl FnOnce() -> R) -> R {
        use crate::resource::{Budget, CancelToken, with_budget};
        let token = CancelToken::new();
        token.cancel();
        with_budget(Budget::new(None, token), f)
    }

    #[test]
    fn test_rmspe_ratio_rule() {
        // The treated unit and the placebos share this rule: post-RMSPE over
        // pre-RMSPE, 0 for no effects, infinite for a pre-RMSPE that is not
        // above 1e-10 (NaN included)
        let effect = |time, effect| TimeEffect {
            time,
            effect,
            actual: 10.0 + effect,
            synthetic: 10.0,
        };
        let effects = [effect(5, 3.1), effect(6, -3.9), effect(7, 0.4)];
        let post = ((3.1f64 * 3.1 + 3.9 * 3.9 + 0.4 * 0.4) / 3.0).sqrt();
        assert!((rmspe_ratio(0.7, &effects) - post / 0.7).abs() < 1e-12);
        assert_eq!(rmspe_ratio(0.7, &[]), 0.0);
        assert_eq!(rmspe_ratio(1e-10, &effects), f64::INFINITY);
        assert_eq!(rmspe_ratio(0.0, &[]), f64::INFINITY);
        assert_eq!(rmspe_ratio(f64::NAN, &effects), f64::INFINITY);
    }

    fn synth_cfg() -> SynthConfig {
        SynthConfig {
            treatment_time: 4,
            treated_unit: "A".to_string(),
            v_method: VOptimization::DataDriven,
            max_iter: 50,
            ..Default::default()
        }
    }

    #[test]
    fn test_synth_rejects_non_positive_tolerance_and_zero_max_iter() {
        let ds = create_test_panel();
        let preds = vec![PredictorSpec::new("x1")];
        for tol in [0.0, -1e-6, f64::NAN, f64::INFINITY] {
            let r = run_synthetic_control(
                &ds,
                "outcome",
                "unit",
                "time",
                &preds,
                SynthConfig {
                    tolerance: tol,
                    ..synth_cfg()
                },
            );
            assert!(
                matches!(r, Err(EconError::InvalidSpecification { ref message }) if message.contains("tolerance")),
                "tol={tol}: ok={}",
                r.is_ok()
            );
        }
        let r = run_synthetic_control(
            &ds,
            "outcome",
            "unit",
            "time",
            &preds,
            SynthConfig {
                max_iter: 0,
                ..synth_cfg()
            },
        );
        assert!(
            matches!(r, Err(EconError::InvalidSpecification { ref message }) if message.contains("max_iter"))
        );
    }

    #[test]
    fn test_synth_stops_when_cancelled() {
        let ds = create_test_panel();
        let preds = vec![PredictorSpec::new("x1")];
        let r = cancelled(|| {
            run_synthetic_control(&ds, "outcome", "unit", "time", &preds, synth_cfg())
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))));
        let r = cancelled(|| {
            run_synthetic_control(
                &ds,
                "outcome",
                "unit",
                "time",
                &preds,
                SynthConfig {
                    run_placebos: true,
                    ..synth_cfg()
                },
            )
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))));
    }

    #[test]
    fn test_synth_placebos_propagate_cancellation() {
        // The placebo inference as a whole stops with Cancelled under a
        // cancelled budget. The treated unit's own fit, which runs first, is the
        // one that sees it here; how a placebo fit's Cancelled is handled is
        // tested by test_placebo_fit_passes_on_cancellation_and_skips_failures.
        let ds = create_test_panel();
        let preds = vec![PredictorSpec::new("x1")];
        let r = cancelled(|| {
            run_placebo_inference(&ds, "outcome", "unit", "time", &preds, &synth_cfg(), 1.0)
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))));
    }

    #[test]
    fn test_placebo_fit_passes_on_cancellation_and_skips_failures() {
        // A placebo fit stopped by the budget must end the inference, not be
        // skipped like a fit that fails
        let ds = create_test_panel();
        let preds = vec![PredictorSpec::new("x1")];
        let fit = |donor: &str| {
            placebo_rmspe_ratio(&ds, "outcome", "unit", "time", &preds, &synth_cfg(), donor)
        };
        assert!(matches!(
            cancelled(|| fit("B")),
            Err(EconError::Cancelled(_))
        ));
        assert!(fit("B").unwrap().is_some_and(|ratio| ratio > 0.0));
        // a unit that is not in the data cannot be fitted: that placebo is skipped
        assert_eq!(fit("Z").unwrap(), None);
    }

    #[test]
    fn test_nelder_mead_stops_when_cancelled() {
        let objective = |v: &[f64]| -> EconResult<f64> { Ok(v.iter().map(|x| x * x).sum::<f64>()) };
        let r = cancelled(|| nelder_mead_optimize(&[1.0, 2.0], &objective, 1e-12, 1000));
        assert!(matches!(r, Err(EconError::Cancelled(_))));
        let nm = nelder_mead_optimize(&[1.0, 2.0], &objective, 1e-12, 1000).unwrap();
        assert!(nm.value < 1e-6 && nm.iterations >= 1 && nm.point.len() == 2);
        // an objective error ends the search
        let failing = |_: &[f64]| -> EconResult<f64> {
            Err(EconError::Computation("objective failed".to_string()))
        };
        assert!(matches!(
            nelder_mead_optimize(&[1.0, 2.0], &failing, 1e-12, 1000),
            Err(EconError::Computation(_))
        ));
    }

    /// Unit "A" treated from t = 3 plus `n_donors` donors over t = 1..=3.
    fn wide_panel(n_donors: usize) -> Dataset {
        let (mut u, mut t, mut y, mut x) = (vec![], vec![], vec![], vec![]);
        for d in 0..=n_donors {
            let name = if d == 0 {
                "A".to_string()
            } else {
                format!("D{d}")
            };
            for time in 1..=3i64 {
                u.push(name.clone());
                t.push(time);
                let noise = 0.3 * ((d * 7 + time as usize * 3) as f64).sin();
                y.push(10.0 + d as f64 * 0.1 + time as f64 + noise);
                x.push(5.0 + d as f64 * 0.05 + noise);
            }
        }
        Dataset::new(df! { "unit" => u, "time" => t, "outcome" => y, "x1" => x }.unwrap())
    }

    #[test]
    fn test_synth_matrices_respect_allocation_ceiling() {
        let preds = vec![PredictorSpec::new("x1")];
        // the extractor's 18-row buffers are over a 16-byte ceiling before the 3 x 2
        // outcome matrix
        let ds = create_test_panel();
        let r = crate::resource::with_alloc_ceiling(Some(16), || {
            run_synthetic_control(&ds, "outcome", "unit", "time", &preds, synth_cfg())
        });
        assert!(
            matches!(r, Err(EconError::ResourceLimit(_))),
            "ok={}",
            r.is_ok()
        );
        // 20 donors: the extractor's row buffers and the outcome matrices fit under this
        // ceiling, the donor-by-donor QP matrices do not
        let ds = wide_panel(20);
        let cfg = SynthConfig {
            treatment_time: 3,
            ..synth_cfg()
        };
        let r = crate::resource::with_alloc_ceiling(Some(1100), || {
            run_synthetic_control(&ds, "outcome", "unit", "time", &preds, cfg.clone())
        });
        assert!(
            matches!(r, Err(EconError::ResourceLimit(ref m)) if m.contains("donor-by-donor")),
            "ok={}",
            r.is_ok()
        );
        assert!(run_synthetic_control(&ds, "outcome", "unit", "time", &preds, cfg).is_ok());
    }

    // =========================================================================
    // Known-DGP property tests
    // =========================================================================

    /// Create a larger synthetic control dataset with a known treatment effect.
    ///
    /// DGP:
    /// - 1 treated unit (T), 4 donors (D1-D4)
    /// - 10 pre-treatment periods, 5 post-treatment periods
    /// - T = 0.4*D1 + 0.6*D2 in pre-treatment (known weights)
    /// - Treatment effect = 10 in post-treatment
    fn create_validate_synth_dataset() -> Dataset {
        let mut units = Vec::new();
        let mut times = Vec::new();
        let mut outcomes = Vec::new();
        let mut x1_vals = Vec::new();

        let n_pre = 10;
        let n_post = 5;
        let treatment_time = (n_pre + 1) as i64;
        let true_effect = 10.0;

        // Donor D1: linear trend
        for t in 1..=(n_pre + n_post) {
            units.push("D1");
            times.push(t as i64);
            outcomes.push(20.0 + 1.5 * t as f64 + 0.1 * ((t as f64) * 0.5).sin());
            x1_vals.push(3.0);
        }

        // Donor D2: different slope
        for t in 1..=(n_pre + n_post) {
            units.push("D2");
            times.push(t as i64);
            outcomes.push(25.0 + 1.0 * t as f64 + 0.1 * ((t as f64) * 0.3).sin());
            x1_vals.push(7.0);
        }

        // Donor D3: unrelated
        for t in 1..=(n_pre + n_post) {
            units.push("D3");
            times.push(t as i64);
            outcomes.push(30.0 + 0.5 * t as f64);
            x1_vals.push(1.0);
        }

        // Donor D4: unrelated
        for t in 1..=(n_pre + n_post) {
            units.push("D4");
            times.push(t as i64);
            outcomes.push(15.0 + 2.0 * t as f64);
            x1_vals.push(9.0);
        }

        // Treated unit T: in pre-treatment = 0.4*D1 + 0.6*D2
        // In post-treatment, add true_effect
        for t in 1..=(n_pre + n_post) {
            units.push("T");
            times.push(t as i64);
            let d1_val = 20.0 + 1.5 * t as f64 + 0.1 * ((t as f64) * 0.5).sin();
            let d2_val = 25.0 + 1.0 * t as f64 + 0.1 * ((t as f64) * 0.3).sin();
            let synth_val = 0.4 * d1_val + 0.6 * d2_val;
            let effect = if t > n_pre { true_effect } else { 0.0 };
            outcomes.push(synth_val + effect);
            x1_vals.push(0.4 * 3.0 + 0.6 * 7.0); // 5.4
        }

        let df = df! {
            "unit" => units,
            "time" => times,
            "outcome" => outcomes,
            "x1" => x1_vals,
        }
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_synth_placebo_inference_ranks_every_unit() {
        // every donor gets a placebo fit next to the treated unit's own; T fits
        // its donors almost exactly before t = 11 and gains 10 after, so its
        // post/pre RMSPE ratio ranks first
        let ds = create_validate_synth_dataset();
        let preds = vec![PredictorSpec::new("x1")];
        let cfg = SynthConfig {
            treatment_time: 11,
            treated_unit: "T".to_string(),
            v_method: VOptimization::Equal,
            run_placebos: true,
            ..Default::default()
        };
        let res = run_synthetic_control(&ds, "outcome", "unit", "time", &preds, cfg).unwrap();
        let p = res.placebo_results.expect("placebo results");
        let mut units: Vec<&str> = p.rmspe_ratios.iter().map(|(u, _)| u.as_str()).collect();
        units.sort_unstable();
        assert_eq!(units, ["D1", "D2", "D3", "D4", "T"]);
        assert_eq!(p.n_units, 5);
        assert_eq!(p.treated_rank, 1);
        assert_eq!(p.p_value, 0.2);
        assert!(p.rmspe_ratios.windows(2).all(|w| w[0].1 >= w[1].1));
    }

    /// Validate synthetic control recovers known weights and treatment effect.
    #[test]
    fn test_synth_weight_recovery_recovers_dgp() {
        let dataset = create_validate_synth_dataset();
        let predictors = vec![PredictorSpec::new("x1")];

        let config = SynthConfig {
            treatment_time: 11,
            treated_unit: "T".to_string(),
            v_method: VOptimization::Equal,
            run_placebos: false,
            ..Default::default()
        };

        let result =
            run_synthetic_control(&dataset, "outcome", "unit", "time", &predictors, config)
                .unwrap();

        // Weights should be non-negative
        for (unit, weight) in &result.all_unit_weights {
            assert!(
                *weight >= -1e-6,
                "Weight for {} = {:.4} should be non-negative",
                unit,
                weight
            );
        }

        // Weights should sum to ~1
        let weight_sum: f64 = result.all_unit_weights.iter().map(|(_, w)| w).sum();
        assert!(
            (weight_sum - 1.0).abs() < 0.02,
            "Weights should sum to 1, got {:.6}",
            weight_sum
        );

        // Pre-treatment RMSPE should be small (good fit)
        assert!(
            result.pre_treatment_rmspe < 2.0,
            "Pre-treatment RMSPE = {:.4} should be small",
            result.pre_treatment_rmspe
        );
    }

    /// Validate synthetic control detects treatment effect direction and magnitude.
    #[test]
    fn test_synth_treatment_effect_recovers_dgp() {
        let dataset = create_validate_synth_dataset();
        let predictors = vec![PredictorSpec::new("x1")];
        let true_effect = 10.0;

        let config = SynthConfig {
            treatment_time: 11,
            treated_unit: "T".to_string(),
            v_method: VOptimization::Equal,
            run_placebos: false,
            ..Default::default()
        };

        let result =
            run_synthetic_control(&dataset, "outcome", "unit", "time", &predictors, config)
                .unwrap();

        // Average effect should be close to true effect (within 3 for this DGP)
        assert!(
            (result.average_effect - true_effect).abs() < 3.0,
            "Average effect = {:.4} should be close to true effect = {}",
            result.average_effect,
            true_effect
        );

        // All post-treatment effects should be positive
        for te in &result.treatment_effects {
            assert!(
                te.effect > 0.0,
                "Treatment effect at time {} = {:.4} should be positive",
                te.time,
                te.effect
            );
        }

        // Cumulative effect should be positive and substantial
        assert!(
            result.cumulative_effect > 0.0,
            "Cumulative effect should be positive"
        );

        // Structural checks
        assert_eq!(result.n_pre_periods, 10);
        assert_eq!(result.n_post_periods, 5);
        assert_eq!(result.n_donors, 4);
    }

    /// Validate synthetic control time series outputs are correctly aligned.
    #[test]
    fn test_synth_time_series_alignment_recovers_dgp() {
        let dataset = create_validate_synth_dataset();
        let predictors = vec![PredictorSpec::new("x1")];

        let config = SynthConfig {
            treatment_time: 11,
            treated_unit: "T".to_string(),
            v_method: VOptimization::Equal,
            run_placebos: false,
            ..Default::default()
        };

        let result =
            run_synthetic_control(&dataset, "outcome", "unit", "time", &predictors, config)
                .unwrap();

        // Actual and synthetic outcomes should have the same length
        assert_eq!(result.actual_outcome.len(), result.synthetic_outcome.len());

        // In pre-treatment period, actual and synthetic should be close
        for i in 0..result.n_pre_periods {
            if i < result.actual_outcome.len() && i < result.synthetic_outcome.len() {
                let (_, actual) = result.actual_outcome[i];
                let (_, synth) = result.synthetic_outcome[i];
                let diff = (actual - synth).abs();
                // Pre-treatment fit should be reasonable (within 5 units)
                assert!(
                    diff < 5.0,
                    "Pre-treatment period {}: actual={:.2}, synth={:.2}, diff={:.2}",
                    i,
                    actual,
                    synth,
                    diff
                );
            }
        }
    }

    // =========================================================================
    // T2-WP13: the panel through the shared extractor (X-2, X-8)
    // =========================================================================

    /// The test_basic_synth configuration.
    fn na_cfg() -> SynthConfig {
        SynthConfig {
            treatment_time: 4,
            treated_unit: "A".to_string(),
            v_method: VOptimization::Equal,
            run_placebos: false,
            ..Default::default()
        }
    }

    fn na_synth(ds: &Dataset, unit_col: &str, cfg: SynthConfig) -> EconResult<SynthResult> {
        let preds = [PredictorSpec::new("x1")];
        run_synthetic_control(ds, "outcome", unit_col, "time", &preds, cfg)
    }

    fn weight_names(r: &SynthResult) -> Vec<&str> {
        r.all_unit_weights.iter().map(|(n, _)| n.as_str()).collect()
    }

    fn weight_bits(r: &SynthResult) -> Vec<u64> {
        r.all_unit_weights
            .iter()
            .map(|(_, w)| w.to_bits())
            .collect()
    }

    fn with_column(ds: &Dataset, column: Column) -> Dataset {
        let mut df = ds.df().clone();
        df.with_column(column).unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_na_synth_integer_outcome_and_unit_ids() {
        let ds = create_test_panel();
        let base = na_synth(&ds, "unit", na_cfg()).unwrap();
        // donors in sorted order
        assert_eq!(weight_names(&base), ["B", "C"]);
        // an Int64 outcome is read as numbers
        let outcome = ds
            .df()
            .column("outcome")
            .unwrap()
            .cast(&DataType::Int64)
            .unwrap();
        let int_outcome = na_synth(&with_column(&ds, outcome), "unit", na_cfg()).unwrap();
        assert_eq!(weight_names(&int_outcome), weight_names(&base));
        assert_eq!(weight_bits(&int_outcome), weight_bits(&base));
        assert_eq!(
            int_outcome.average_effect.to_bits(),
            base.average_effect.to_bits()
        );
        // integer unit ids: A -> 1, B -> 2, C -> 3
        let ids: Vec<i64> = (0..ds.nrows()).map(|r| 1 + (r / 6) as i64).collect();
        let int_ids = with_column(&ds, Column::new("unit_id".into(), ids));
        let r = na_synth(
            &int_ids,
            "unit_id",
            SynthConfig {
                treated_unit: "1".to_string(),
                ..na_cfg()
            },
        )
        .unwrap();
        assert_eq!(weight_names(&r), ["2", "3"]);
        assert_eq!(weight_bits(&r), weight_bits(&base));
    }

    #[test]
    fn test_na_synth_duplicate_cell() {
        // (B, 2) is row 7; its copy has another outcome
        let ds = create_test_panel();
        let mut extra = ds.df().slice(7, 1);
        extra
            .with_column(Column::new("outcome".into(), [99.0]))
            .unwrap();
        let mut df = ds.df().clone();
        df.vstack_mut(&extra).unwrap();
        match na_synth(&Dataset::new(df), "unit", na_cfg()) {
            Err(EconError::InvalidSpecification { message }) => assert_eq!(
                message,
                "synthetic control needs one row per (unit, time): 1 duplicate row(s) \
                 (first: unit 'B', time 2). Remove the duplicates first."
            ),
            other => panic!("duplicate cell: {other:?}"),
        }
    }

    #[test]
    fn test_na_synth_missing_outcome_cell() {
        let ds = create_test_panel();
        let outcome: Vec<Option<f64>> = (0..ds.nrows())
            .map(|r| {
                (r != 7).then(|| {
                    ds.df()
                        .column("outcome")
                        .unwrap()
                        .get(r)
                        .unwrap()
                        .try_extract::<f64>()
                        .unwrap()
                })
            })
            .collect();
        let data = with_column(&ds, Column::new("outcome".into(), outcome));
        match na_synth(&data, "unit", na_cfg()) {
            Err(EconError::InvalidSpecification { message }) => assert_eq!(
                message,
                "Outcome 'outcome' is missing (null or NaN) for unit 'B' at time 2."
            ),
            other => panic!("missing outcome cell: {other:?}"),
        }
    }

    #[test]
    fn test_na_synth_null_unit_label() {
        // (A, 6) is row 5
        let ds = create_test_panel();
        let units: Vec<Option<&str>> = (0..ds.nrows())
            .map(|r| (r != 5).then_some(["A", "B", "C"][r / 6]))
            .collect();
        let data = with_column(&ds, Column::new("unit".into(), units));
        match na_synth(&data, "unit", na_cfg()) {
            Err(EconError::NullValues { column, count }) => {
                assert_eq!((column.as_str(), count), ("unit", 1));
            }
            other => panic!("null unit label: {other:?}"),
        }
    }
    // =========================================================================
    // The exact simplex QP and the V search (T3: CAUS-40, CAUS-88, CAUS-116)
    // =========================================================================

    use p2a_validation::{Check, Tol};

    const SMALL: [&str; 3] = ["x1", "x2", "x3"];
    const BIG: [&str; 3] = ["x1_big", "x2_big", "x3_big"];

    /// The planner panel `synth_qp_panel` without the case (property tests stay out of the
    /// registry).
    fn qp_panel() -> Dataset {
        let path = p2a_validation::repo_path("validation/datasets/generated/synth_qp_panel.csv");
        Dataset::new(crate::data::DataLoader::load_csv(&path).expect("synth_qp_panel.csv"))
    }

    fn qp_config(v: VOptimization) -> SynthConfig {
        SynthConfig {
            treatment_time: 15,
            treated_unit: "TR".to_string(),
            v_method: v,
            weight_threshold: 0.0,
            ..Default::default()
        }
    }

    fn qp_fit(ds: &Dataset, v: VOptimization, cols: &[&str]) -> EconResult<SynthResult> {
        let preds: Vec<PredictorSpec> = cols.iter().map(|c| PredictorSpec::new(c)).collect();
        run_synthetic_control(ds, "y", "unit", "time", &preds, qp_config(v))
    }

    fn weights(r: &SynthResult) -> (Vec<String>, Vec<f64>) {
        r.all_unit_weights
            .iter()
            .map(|(n, w)| (n.clone(), *w))
            .unzip()
    }

    /// `ds` with column `name` = `f(row)`.
    fn with_values(ds: &Dataset, name: &str, f: impl Fn(usize) -> f64) -> Dataset {
        let v: Vec<f64> = (0..ds.nrows()).map(f).collect();
        with_column(ds, Column::new(name.into(), v))
    }

    fn column_values(ds: &Dataset, name: &str) -> Vec<f64> {
        let c = ds
            .df()
            .column(name)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap();
        c.f64().unwrap().into_iter().map(|v| v.unwrap()).collect()
    }

    #[test]
    fn test_validate_synth_weights_qp() {
        let (case, ds) = crate::test_support::load_case("synth_weights_qp");
        let r = qp_fit(&ds, VOptimization::Equal, &SMALL).unwrap();
        let n = r.n_pre_periods;
        assert_eq!(n, 14);
        let (names, w) = weights(&r);
        let effects: Vec<f64> = r.treatment_effects.iter().map(|e| e.effect).collect();
        let synthetic: Vec<f64> = r.synthetic_outcome.iter().map(|(_, v)| *v).collect();
        let balance: Vec<f64> = r
            .predictor_balance
            .iter()
            .map(|b| b.synthetic_value)
            .collect();
        let positive = w.iter().filter(|&&x| x > 0.0).count() as f64;
        Check::new(&case)
            .named("equal_w", &names, &w, Tol::coef(n))
            .scalar("equal_predictor_loss", r.predictor_loss, Tol::stat(n))
            .scalar("equal_pre_mspe", r.pre_treatment_mspe, Tol::stat(n))
            .vector("equal_effects", &effects, Tol::coef(n))
            .scalar("equal_average_effect", r.average_effect, Tol::coef(n))
            .vector("equal_synthetic", &synthetic, Tol::coef(n))
            .vector("equal_balance_synthetic", &balance, Tol::coef(n))
            .scalar("equal_n_positive", positive, Tol::exact())
            .finish();
        assert!(r.convergence.is_none() && r.v_start_losses.is_empty());
        assert!(r.warnings.iter().all(|w| !w.contains("not unique")));
    }

    #[test]
    fn test_validate_synth_weights_qp_custom() {
        let (case, ds) = crate::test_support::load_case("synth_weights_qp");
        let r = qp_fit(&ds, VOptimization::Custom(vec![0.6, 0.3, 0.1]), &SMALL).unwrap();
        let n = r.n_pre_periods;
        let (names, w) = weights(&r);
        Check::new(&case)
            .named("custom_w", &names, &w, Tol::coef(n))
            .scalar("custom_predictor_loss", r.predictor_loss, Tol::stat(n))
            .scalar("custom_pre_mspe", r.pre_treatment_mspe, Tol::stat(n))
            .scalar("custom_average_effect", r.average_effect, Tol::coef(n))
            .finish();
        // V is normalised to sum 1 (6, 3, 1 is the same V)
        let r2 = qp_fit(&ds, VOptimization::Custom(vec![6.0, 3.0, 1.0]), &SMALL).unwrap();
        for (a, b) in weights(&r).1.iter().zip(weights(&r2).1.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_validate_synth_weights_qp_invariant_to_predictor_units() {
        let (case, ds) = crate::test_support::load_case("synth_weights_qp");
        let big = qp_fit(&ds, VOptimization::Equal, &BIG).unwrap();
        let small = qp_fit(&ds, VOptimization::Equal, &SMALL).unwrap();
        let (names, w) = weights(&big);
        Check::new(&case)
            .named("equal_big_w", &names, &w, Tol::coef(14))
            .scalar(
                "equal_big_predictor_loss",
                big.predictor_loss,
                Tol::stat(14),
            )
            .finish();
        let x2 = qp_fit(&ds, VOptimization::Equal, &["x1", "x2_big", "x3"]).unwrap();
        for other in [&big, &x2] {
            for (a, b) in weights(other).1.iter().zip(weights(&small).1.iter()) {
                assert!((a - b).abs() < 1e-10, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn test_validate_synth_regression_start() {
        let (case, ds) = crate::test_support::load_case("synth_weights_qp");
        let preds: Vec<PredictorSpec> = SMALL.iter().map(|c| PredictorSpec::new(c)).collect();
        let data = prepare_synth_data(&ds, "y", "unit", "time", &preds, "TR", 15, None).unwrap();
        let (xs1, xs0) = standardized_predictors(&data).unwrap();
        let sv2 = regression_v_start(&xs1, &xs0, &data.z1, &data.z0).unwrap();
        Check::new(&case)
            .named("sv2_start", &SMALL, sv2.as_slice().unwrap(), Tol::coef(14))
            .finish();
    }

    #[test]
    fn test_synth_regression_start_falls_back_to_equal_v() {
        // 2 donors and 3 predictors: 3 units < 4 regression columns, the regression is
        // not identified and the start is equal V
        let df = df! {
            "unit" => ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "time" => [1i64, 2, 3, 1, 2, 3, 1, 2, 3],
            "y" => [1.0, 1.4, 2.9, 0.7, 1.1, 1.9, 1.6, 2.2, 3.1],
            "p" => [1.0, 1.2, 1.1, 0.4, 0.5, 0.45, 2.0, 2.1, 1.9],
            "q" => [3.0, 3.3, 2.8, 1.0, 1.4, 1.2, 4.0, 4.4, 4.1],
            "r" => [0.2, 0.25, 0.3, 0.9, 0.8, 0.85, 0.1, 0.05, 0.12],
        }
        .unwrap();
        let ds = Dataset::new(df);
        let preds: Vec<PredictorSpec> = ["p", "q", "r"]
            .iter()
            .map(|c| PredictorSpec::new(c))
            .collect();
        let data = prepare_synth_data(&ds, "y", "unit", "time", &preds, "A", 3, None).unwrap();
        let (xs1, xs0) = standardized_predictors(&data).unwrap();
        let sv2 = regression_v_start(&xs1, &xs0, &data.z1, &data.z0).unwrap();
        assert_eq!(sv2.to_vec(), vec![1.0 / 3.0; 3]);
    }

    #[test]
    fn test_synth_datadriven_keeps_the_better_start() {
        let r = qp_fit(&qp_panel(), VOptimization::DataDriven, &SMALL).unwrap();
        assert_eq!(r.v_start_losses.len(), 2);
        let best = r
            .v_start_losses
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        assert!(
            r.final_loss <= best * (1.0 + 1e-12),
            "{} vs {best}",
            r.final_loss
        );
        // the reported V reproduces the reported loss
        let v: Vec<f64> = r.predictor_weights.iter().map(|(_, v)| *v).collect();
        let again = qp_fit(&qp_panel(), VOptimization::Custom(v), &SMALL).unwrap();
        assert!((again.pre_treatment_mspe - r.final_loss).abs() <= 1e-9 * r.final_loss);
        assert!((r.pre_treatment_mspe - r.final_loss).abs() <= 1e-12 * r.final_loss);
        assert!(
            r.to_string()
                .contains("V search: 2 starts (equal, regression)")
        );
        let c = r.convergence.as_ref().expect("convergence of the V search");
        assert!(c.converged);
    }

    #[test]
    fn test_synth_invalid_v_is_error() {
        let ds = qp_panel();
        for v in [
            vec![1.0, -1.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![f64::NAN, 1.0, 1.0],
            vec![f64::INFINITY, 1.0, 1.0],
        ] {
            match qp_fit(&ds, VOptimization::Custom(v.clone()), &SMALL) {
                Err(EconError::InvalidSpecification { message }) => {
                    assert!(message.contains("custom V"), "{v:?}: {message}")
                }
                other => panic!("{v:?}: {other:?}"),
            }
        }
        match qp_fit(&ds, VOptimization::Custom(vec![1.0, 1.0]), &SMALL) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(
                    message.contains("doesn't match number of predictors"),
                    "{message}"
                )
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn test_synth_predictors_without_variation_is_error() {
        let ds = with_values(&create_test_panel(), "z", |_| 0.0);
        let preds = [PredictorSpec::new("z")];
        let r = run_synthetic_control(&ds, "outcome", "unit", "time", &preds, na_cfg());
        assert!(
            matches!(r, Err(EconError::InvalidSpecification { ref message }) if message.contains("no variation") && message.contains("'z'")),
            "{r:?}"
        );
    }

    /// `min_j eta_j` over the zero weights, as S1 defines the dual check (recomputed here).
    fn min_eta(a: &Array2<f64>, b: &Array1<f64>, w: &Array1<f64>) -> f64 {
        let j = a.ncols();
        let s = a.iter().map(|v| v * v).sum::<f64>() / j as f64;
        let g = a.t().dot(a) / s + Array2::<f64>::eye(j) * SIMPLEX_QP_RIDGE;
        let q = a.t().dot(b) / s;
        let grad = g.dot(w) - &q;
        let free: Vec<usize> = (0..j).filter(|&i| w[i] > 0.0).collect();
        let mu = -free.iter().map(|&i| grad[i]).sum::<f64>() / free.len() as f64;
        (0..j)
            .filter(|&i| w[i] == 0.0)
            .map(|i| grad[i] + mu)
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn test_simplex_qp_exact_cases() {
        // (a) one predictor, the treated inside the hull: a face of optima, minimum norm
        let a = Array2::from_shape_vec((1, 3), vec![0.0, 1.0, 3.0]).unwrap();
        let b = Array1::from(vec![2.0]);
        let r = simplex_qp(a.view(), b.view()).unwrap();
        for (x, e) in r.weights.iter().zip([1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0]) {
            assert!((x - e).abs() < 1e-8, "{:?}", r.weights);
        }
        assert!(!r.unique);
        // (b) outside the hull: a unique vertex-edge optimum
        let a2 = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        let b2 = Array1::from(vec![0.9, 0.3]);
        let r2 = simplex_qp(a2.view(), b2.view()).unwrap();
        for (x, e) in r2.weights.iter().zip([0.8, 0.2, 0.0]) {
            assert!((x - e).abs() < 1e-9, "{:?}", r2.weights);
        }
        assert!(r2.unique);
        assert_eq!(r2.free, vec![0, 1]);
        // (c) a common power-of-2 scale gives bitwise identical weights
        let r3 = simplex_qp((&a2 * 1024.0).view(), (&b2 * 1024.0).view()).unwrap();
        let bits = |w: &Array1<f64>| w.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&r3.weights), bits(&r2.weights));
        // (d) KKT
        assert!(min_eta(&a, &b, &r.weights) >= -1e-12);
        assert!(min_eta(&a2, &b2, &r2.weights) >= -1e-12);
        // (e) one donor
        let r1 = simplex_qp(Array2::from_elem((2, 1), 0.5).view(), b2.view()).unwrap();
        assert_eq!(r1.weights.to_vec(), vec![1.0]);
        // (f) a non-finite entry
        let mut bad = a2.clone();
        bad[[0, 1]] = f64::NAN;
        assert!(matches!(
            simplex_qp(bad.view(), b2.view()),
            Err(EconError::InvalidSpecification { .. })
        ));
        // (g) k = 2, J = 40, the treated inside the hull: the reduced KKT path equals the
        // bordered one and meets KKT; the optimum is not unique
        let (k, j) = (2, 40);
        let a4 = Array2::from_shape_fn((k, j), |(m, i)| {
            ((m * 31 + i * 17) as f64 * 0.37).sin() + 0.1 * (i as f64 * 0.11).cos()
        });
        let w0 = Array1::from_shape_fn(j, |i| 1.0 + 0.5 * (i as f64 * 0.7).sin());
        let w0 = &w0 / w0.sum();
        let b4 = a4.dot(&w0);
        let reduced = simplex_qp(a4.view(), b4.view()).unwrap();
        let bordered = with_bordered_kkt(|| simplex_qp(a4.view(), b4.view())).unwrap();
        assert!(reduced.free.len() > k + 1);
        // The bordered system has condition number about 1 / SIMPLEX_QP_RIDGE on a face of
        // optima, so its solution carries errors near 1e-6; the reduced system has no
        // near-null directions. Both reach the same regularised objective, and the weights
        // agree to the bordered solve's accuracy.
        let objective = |w: &Array1<f64>| {
            let s = a4.iter().map(|v| v * v).sum::<f64>() / j as f64;
            let r = a4.dot(w) - &b4;
            0.5 * r.dot(&r) / s + 0.5 * SIMPLEX_QP_RIDGE * w.dot(w)
        };
        let (fr, fb) = (objective(&reduced.weights), objective(&bordered.weights));
        assert!(fr <= fb * (1.0 + 1e-9) + 1e-300, "{fr} vs {fb}");
        for (x, y) in reduced.weights.iter().zip(bordered.weights.iter()) {
            assert!((x - y).abs() <= 1e-4 * y.abs().max(1e-3), "{x} vs {y}");
        }
        assert!(min_eta(&a4, &b4, &reduced.weights) >= -1e-12);
        assert!(!reduced.unique);
        assert!((reduced.weights.sum() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_synth_duplicate_donors_split_and_warn() {
        // D09 copies D07: the optimum is a face, the minimum-norm weights split D07's
        // weight equally between the two
        let ds = qp_panel();
        let df = ds.df();
        let mask = df.column("unit").unwrap().str().unwrap().equal("D07");
        let mut copy = df.filter(&mask).unwrap();
        copy.with_column(Column::new("unit".into(), vec!["D09"; copy.height()]))
            .unwrap();
        let mut both = df.clone();
        both.vstack_mut(&copy).unwrap();
        let r = qp_fit(&Dataset::new(both), VOptimization::Equal, &SMALL).unwrap();
        let w = |u: &str| r.all_unit_weights.iter().find(|(n, _)| n == u).unwrap().1;
        assert!(
            w("D07") + w("D09") > 0.5 && (w("D07") - w("D09")).abs() < 1e-8,
            "{} {}",
            w("D07"),
            w("D09")
        );
        assert!((r.all_unit_weights.iter().map(|(_, v)| v).sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(
            r.warnings.iter().any(|m| m.contains("not unique")),
            "{:?}",
            r.warnings
        );
    }

    #[test]
    fn test_synth_datadriven_is_unit_invariant_and_not_worse_than_equal() {
        let ds = qp_panel();
        let base = qp_fit(&ds, VOptimization::DataDriven, &SMALL).unwrap();
        let mut scaled = ds.clone();
        for c in SMALL {
            let v = column_values(&ds, c);
            scaled = with_values(&scaled, c, |i| v[i] * 8192.0);
        }
        let r = qp_fit(&scaled, VOptimization::DataDriven, &SMALL).unwrap();
        let bits = |r: &SynthResult| weights(r).1.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&r), bits(&base));
        assert_eq!(r.v_iterations, base.v_iterations);
        let equal = qp_fit(&ds, VOptimization::Equal, &SMALL).unwrap();
        assert!(base.final_loss <= equal.pre_treatment_mspe * (1.0 + 1e-12));
        assert!(base.convergence.is_some());
    }

    #[test]
    fn test_synth_many_donors_inside_hull() {
        // the comprehensive benchmark's DGP at 100 units x 100 periods: two uniform
        // predictors, the treated unit inside the donors' hull, so the optimum is a face
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let (n_units, n_periods) = (100usize, 100usize);
        let (mut u, mut t, mut y, mut p1, mut p2) = (vec![], vec![], vec![], vec![], vec![]);
        for unit in 0..n_units {
            for time in 0..n_periods {
                u.push(format!("unit_{unit}"));
                t.push(time as i64);
                let base = unit as f64 * 0.5 + time as f64 * 0.1;
                let te = if unit == 0 && time >= 70 { 2.0 } else { 0.0 };
                y.push(base + te + rng.gen_range(-0.3..0.3));
                p1.push(rng.gen_range(0.0..1.0));
                p2.push(rng.gen_range(0.0..1.0));
            }
        }
        let ds = Dataset::new(
            df! { "unit" => u, "time" => t, "outcome" => y, "pred1" => p1, "pred2" => p2 }.unwrap(),
        );
        let preds = [PredictorSpec::new("pred1"), PredictorSpec::new("pred2")];
        let cfg = SynthConfig {
            treatment_time: 70,
            treated_unit: "unit_0".to_string(),
            ..Default::default()
        };
        let r = run_synthetic_control(&ds, "outcome", "unit", "time", &preds, cfg).unwrap();
        assert!(
            r.warnings.iter().any(|m| m.contains("not unique")),
            "{:?}",
            r.warnings
        );
        // KKT of the reported weights at the reported V on the standardised predictors
        let data =
            prepare_synth_data(&ds, "outcome", "unit", "time", &preds, "unit_0", 70, None).unwrap();
        let (xs1, xs0) = standardized_predictors(&data).unwrap();
        let v: Vec<f64> = r.predictor_weights.iter().map(|(_, v)| v.sqrt()).collect();
        let a = Array2::from_shape_fn(xs0.dim(), |(m, i)| v[m] * xs0[[m, i]]);
        let b = Array1::from_shape_fn(xs1.len(), |m| v[m] * xs1[m]);
        let w = Array1::from(weights(&r).1);
        assert!(min_eta(&a, &b, &w) >= -1e-10);
        assert!((w.sum() - 1.0).abs() < 1e-12 && w.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_time_aggregation_and_v_method_parse() {
        assert_eq!(
            "Average".parse::<TimeAggregation>().unwrap(),
            TimeAggregation::Mean
        );
        assert_eq!(
            "total".parse::<TimeAggregation>().unwrap(),
            TimeAggregation::Sum
        );
        assert_eq!(TimeAggregation::Last.to_string(), "last");
        assert!("median".parse::<TimeAggregation>().is_err());
        TimeAggregation::choices().validate().unwrap();
        assert!(matches!(
            VOptimization::from_options(None, None),
            Ok(VOptimization::DataDriven)
        ));
        assert!(matches!(
            VOptimization::from_options(Some("Data-Driven"), None),
            Ok(VOptimization::DataDriven)
        ));
        assert!(matches!(
            VOptimization::from_options(Some("uniform"), None),
            Ok(VOptimization::Equal)
        ));
        assert!(matches!(
            VOptimization::from_options(Some("custom"), Some(vec![1.0, 2.0])),
            Ok(VOptimization::Custom(ref v)) if v == &vec![1.0, 2.0]
        ));
        for bad in [
            VOptimization::from_options(Some("custom"), None),
            VOptimization::from_options(Some("bayes"), None),
        ] {
            match bad {
                Err(EconError::InvalidSpecification { message }) => {
                    assert!(
                        message.contains("datadriven") && message.contains("custom"),
                        "{message}"
                    )
                }
                other => panic!("{other:?}"),
            }
        }
    }
}

#[cfg(test)]
mod gsynth_tests {
    use super::tests::cancelled;
    use super::*;

    fn create_gsynth_dataset() -> Dataset {
        // Panel data: 3 control units (C1, C2, C3), 2 treated units (T1, T2)
        // 10 time periods, treatment starts at t=7 for T1, t=8 for T2
        // This gives 6 pre-treatment periods (1-6) for T1
        let mut units: Vec<&str> = Vec::new();
        let mut times: Vec<i32> = Vec::new();
        let mut outcomes: Vec<f64> = Vec::new();
        let mut treatment: Vec<f64> = Vec::new();

        // Control units (never treated)
        for unit in ["C1", "C2", "C3"] {
            for t in 1..=10 {
                units.push(unit);
                times.push(t);
                outcomes.push(10.0 + t as f64 + if unit == "C2" { 2.0 } else { 0.0 });
                treatment.push(0.0);
            }
        }

        // Treated unit 1: treatment at t=7
        for t in 1..=10 {
            units.push("T1");
            times.push(t);
            let base = 10.0 + t as f64 + 1.0;
            let effect = if t >= 7 { 5.0 } else { 0.0 };
            outcomes.push(base + effect);
            treatment.push(if t >= 7 { 1.0 } else { 0.0 });
        }

        // Treated unit 2: treatment at t=8
        for t in 1..=10 {
            units.push("T2");
            times.push(t);
            let base = 10.0 + t as f64 - 0.5;
            let effect = if t >= 8 { 3.0 } else { 0.0 };
            outcomes.push(base + effect);
            treatment.push(if t >= 8 { 1.0 } else { 0.0 });
        }

        let df = df! {
            "unit" => units,
            "time" => times,
            "outcome" => outcomes,
            "treated" => treatment
        }
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_gsynth_basic() {
        let dataset = create_gsynth_dataset();
        let config = GsynthConfig {
            n_factors: 1,
            cross_validate: false,
            min_pre_periods: 3, // Lower for test
            ..Default::default()
        };

        let result =
            run_gsynth(&dataset, "outcome", "treated", "unit", "time", &[], config).unwrap();

        // Should identify 2 treated and 3 control units
        assert_eq!(result.n_treated, 2);
        assert_eq!(result.n_control, 3);

        // ATT should be positive (treatment has positive effect)
        assert!(
            result.att > 0.0,
            "ATT should be positive, got {}",
            result.att
        );
    }

    #[test]
    fn test_gsynth_unit_effects() {
        let dataset = create_gsynth_dataset();
        let config = GsynthConfig {
            n_factors: 1,
            cross_validate: false,
            min_pre_periods: 3, // Lower for test
            ..Default::default()
        };

        let result =
            run_gsynth(&dataset, "outcome", "treated", "unit", "time", &[], config).unwrap();

        // Should have 2 unit effects
        assert_eq!(result.unit_effects.len(), 2);

        // Both should show positive effects
        for ue in &result.unit_effects {
            assert!(ue.att > 0.0, "Unit {} ATT should be positive", ue.unit);
        }
    }

    #[test]
    fn test_gsynth_cv() {
        let dataset = create_gsynth_dataset();
        let config = GsynthConfig {
            n_factors: 0, // Auto-select
            cross_validate: true,
            max_factors: 3,
            cv_folds: 2,
            min_pre_periods: 2,
            ..Default::default()
        };

        let result =
            run_gsynth(&dataset, "outcome", "treated", "unit", "time", &[], config).unwrap();

        // Should select some number of factors
        assert!(result.n_factors <= 3);
        // CV results should be present
        assert!(!result.cv_mspe.is_empty() || result.n_factors == 0);
    }

    // =========================================================================
    // Factor count and CV fold bounds (T0-WP11)
    // =========================================================================

    fn gs_cfg() -> GsynthConfig {
        GsynthConfig {
            n_factors: 1,
            cross_validate: false,
            min_pre_periods: 3,
            ..Default::default()
        }
    }

    #[test]
    fn test_gsynth_rejects_more_factors_than_the_data_identify() {
        let ds = create_gsynth_dataset();
        // 4 > min(3 controls, 6 pre-periods): index panic at extend_factors_to_all_periods
        // before the fix; usize::MAX: ndarray shape-overflow panic before the fix
        for r in [4usize, usize::MAX] {
            let res = run_gsynth(
                &ds,
                "outcome",
                "treated",
                "unit",
                "time",
                &[],
                GsynthConfig {
                    n_factors: r,
                    ..gs_cfg()
                },
            );
            assert!(
                matches!(res, Err(EconError::InvalidSpecification { ref message })
                    if message.contains("n_factors") && message.contains("at most 3")),
                "r={r}"
            );
        }
        let ok = run_gsynth(
            &ds,
            "outcome",
            "treated",
            "unit",
            "time",
            &[],
            GsynthConfig {
                n_factors: 3,
                ..gs_cfg()
            },
        )
        .unwrap();
        assert_eq!(ok.n_factors, 3);
    }

    #[test]
    fn test_gsynth_cv_folds_bounds() {
        let ds = create_gsynth_dataset();
        let cv = |k| GsynthConfig {
            n_factors: 0,
            cross_validate: true,
            max_factors: 3,
            cv_folds: k,
            ..gs_cfg()
        };
        // 0 divided by zero before the fix
        for k in [0usize, 1] {
            let r = run_gsynth(&ds, "outcome", "treated", "unit", "time", &[], cv(k));
            assert!(
                matches!(r, Err(EconError::InvalidSpecification { ref message }) if message.contains("cv_folds")),
                "k={k}"
            );
        }
        // more folds than control units is clamped to the number of control units, not rejected
        let r = run_gsynth(
            &ds,
            "outcome",
            "treated",
            "unit",
            "time",
            &[],
            cv(1_000_000),
        )
        .unwrap();
        assert!(r.n_factors <= 3);
    }

    #[test]
    fn test_gsynth_requires_a_pre_treatment_period() {
        // T1 treated from the first period: min_pre_periods = 0 used to reach
        // `n_pre - 1` (underflow) in CV and an index panic in the default path
        let df = df! {
            "unit" => ["C1", "C1", "C1", "C2", "C2", "C2", "C3", "C3", "C3", "T1", "T1", "T1"],
            "time" => [1i64, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "outcome" => [1.0, 2.1, 2.9, 2.0, 3.2, 3.9, 0.5, 1.4, 2.6, 4.0, 5.1, 6.2],
            "treated" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        }
        .unwrap();
        let ds = Dataset::new(df);
        for cross_validate in [false, true] {
            let r = run_gsynth(
                &ds,
                "outcome",
                "treated",
                "unit",
                "time",
                &[],
                GsynthConfig {
                    n_factors: 0,
                    cross_validate,
                    min_pre_periods: 0,
                    ..Default::default()
                },
            );
            assert!(
                matches!(r, Err(EconError::InvalidSpecification { ref message }) if message.contains("pre-treatment")),
                "cross_validate={cross_validate}"
            );
        }
    }

    #[test]
    fn test_gsynth_default_factor_count_capped_by_rank() {
        // n_factors = 0 without CV defaults to 2 factors; with one pre-period the
        // SVD returns one, and the requested 2 used to index past it
        let df = df! {
            "unit" => ["C1", "C1", "C1", "C2", "C2", "C2", "C3", "C3", "C3", "T1", "T1", "T1"],
            "time" => [1i64, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "outcome" => [1.0, 2.1, 2.9, 2.0, 3.2, 3.9, 0.5, 1.4, 2.6, 4.0, 7.1, 8.2],
            "treated" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        }
        .unwrap();
        let ds = Dataset::new(df);
        let r = run_gsynth(
            &ds,
            "outcome",
            "treated",
            "unit",
            "time",
            &[],
            GsynthConfig {
                n_factors: 0,
                cross_validate: false,
                min_pre_periods: 1,
                ..Default::default()
            },
        );
        // extract_factors_svd caps r at min(2, 3 controls, 1 pre-period) = 1
        let res = r.unwrap();
        assert_eq!(res.n_factors, 1);
    }

    #[test]
    fn test_gsynth_explicit_factor_count_stops_when_cancelled() {
        // An explicit n_factors does not go through select_factors_cv, whose
        // per-fold checkpoints were the only ones on the gsynth path; the factor
        // extraction checks the budget once per factor
        let ds = create_gsynth_dataset();
        let r = cancelled(|| run_gsynth(&ds, "outcome", "treated", "unit", "time", &[], gs_cfg()));
        assert!(
            matches!(r, Err(EconError::Cancelled(_))),
            "ok={}",
            r.is_ok()
        );
    }

    #[test]
    fn test_gsynth_matrices_respect_allocation_ceiling() {
        // the extractor's 50-row buffers (400 bytes) are over a 64-byte ceiling; a
        // balanced panel's units x periods buffers are no larger
        let ds = create_gsynth_dataset();
        let r = crate::resource::with_alloc_ceiling(Some(64), || {
            run_gsynth(&ds, "outcome", "treated", "unit", "time", &[], gs_cfg())
        });
        assert!(
            matches!(r, Err(EconError::ResourceLimit(_))),
            "ok={}",
            r.is_ok()
        );
        // an unbalanced panel is refused before any per-cell buffer is allocated
        let n = 40usize;
        let quad = Dataset::new(
            df! {
                "unit" => (0..n).map(|i| format!("u{i:02}")).collect::<Vec<_>>(),
                "time" => (1..=n as i64).collect::<Vec<_>>(),
                "outcome" => (0..n).map(|i| i as f64 + 0.5).collect::<Vec<_>>(),
                "treated" => (0..n).map(|i| f64::from(u8::from(i == n - 1))).collect::<Vec<_>>(),
            }
            .unwrap(),
        );
        let r = crate::resource::with_alloc_ceiling(Some(4096), || {
            run_gsynth(&quad, "outcome", "treated", "unit", "time", &[], gs_cfg())
        });
        assert!(
            matches!(r, Err(EconError::InvalidSpecification { ref message }) if message.starts_with(
                "gsynth needs a balanced panel: 1560 (unit, time) cell(s) are missing \
                 (first: unit 'u00', time 2)."
            )),
            "{r:?}"
        );
        // a long panel: the extractor's row buffers and the units-by-periods panels fit
        // under this ceiling, the pre-period factor covariance does not
        let (mut u, mut t, mut y, mut d) = (vec![], vec![], vec![], vec![]);
        for unit in ["C1", "C2", "C3", "T1"] {
            for time in 1..=40i64 {
                let k = unit.as_bytes()[1] as f64;
                u.push(unit);
                t.push(time);
                let treated = unit == "T1" && time > 30;
                y.push(
                    k + 0.1 * time as f64
                        + 0.2 * (k * time as f64).sin()
                        + if treated { 2.0 } else { 0.0 },
                );
                d.push(if treated { 1.0 } else { 0.0 });
            }
        }
        let long =
            Dataset::new(df! { "unit" => u, "time" => t, "outcome" => y, "treated" => d }.unwrap());
        let r = crate::resource::with_alloc_ceiling(Some(3000), || {
            run_gsynth(&long, "outcome", "treated", "unit", "time", &[], gs_cfg())
        });
        assert!(
            matches!(r, Err(EconError::ResourceLimit(ref m)) if m.contains("pre-treatment periods")),
            "ok={}",
            r.is_ok()
        );
        assert!(run_gsynth(&long, "outcome", "treated", "unit", "time", &[], gs_cfg()).is_ok());
    }

    // =========================================================================
    // T2-WP13: missing values, balance, duplicates, time index (CAUS-42)
    // =========================================================================

    fn drop_row(ds: &Dataset, i: usize) -> Dataset {
        let keep = BooleanChunked::from_iter_values("keep".into(), (0..ds.nrows()).map(|r| r != i));
        Dataset::new(ds.df().filter(&keep).unwrap())
    }

    fn dup_row(ds: &Dataset, i: usize) -> Dataset {
        let mut df = ds.df().clone();
        df.vstack_mut(&ds.df().slice(i as i64, 1)).unwrap();
        Dataset::new(df)
    }

    fn null_at(ds: &Dataset, col: &str, i: usize) -> Dataset {
        let mut df = ds.df().clone();
        let c = df.column(col).unwrap();
        let replaced = if c.dtype() == &DataType::String {
            let v: Vec<Option<String>> = c
                .str()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(r, s)| if r == i { None } else { s.map(str::to_string) })
                .collect();
            Column::new(col.into(), v)
        } else {
            let v: Vec<Option<f64>> = c
                .cast(&DataType::Float64)
                .unwrap()
                .f64()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(r, x)| if r == i { None } else { x })
                .collect();
            Column::new(col.into(), v)
        };
        df.with_column(replaced).unwrap();
        Dataset::new(df)
    }

    fn quarterly_time(ds: &Dataset) -> Dataset {
        let mut df = ds.df().clone();
        let t: Vec<Option<f64>> = df
            .column("time")
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .map(|t| t.map(|t| 2000.0 + (t - 1.0) / 4.0))
            .collect();
        df.with_column(Column::new("time".into(), t)).unwrap();
        Dataset::new(df)
    }

    fn with_dtype(ds: &Dataset, col: &str, dtype: DataType) -> Dataset {
        let mut df = ds.df().clone();
        let cast = df.column(col).unwrap().cast(&dtype).unwrap();
        df.with_column(cast).unwrap();
        Dataset::new(df)
    }

    fn gs(ds: &Dataset, config: GsynthConfig) -> EconResult<GsynthResult> {
        run_gsynth(ds, "outcome", "treated", "unit", "time", &[], config)
    }

    #[test]
    fn test_na_gsynth_missing_cell() {
        // (C2, 5) is row 14
        let err = gs(&drop_row(&create_gsynth_dataset(), 14), gs_cfg()).unwrap_err();
        assert!(
            matches!(&err, EconError::InvalidSpecification { message } if message.starts_with(
                "gsynth needs a balanced panel: 1 (unit, time) cell(s) are missing \
                 (first: unit 'C2', time 5)."
            )),
            "{err:?}"
        );
    }

    #[test]
    fn test_na_gsynth_duplicate_cell() {
        let err = gs(&dup_row(&create_gsynth_dataset(), 14), gs_cfg()).unwrap_err();
        assert!(
            matches!(&err, EconError::InvalidSpecification { message } if message.starts_with(
                "gsynth needs one row per (unit, time): 1 duplicate row(s) \
                 (first: unit 'C2', time 5)."
            )),
            "{err:?}"
        );
    }

    #[test]
    fn test_na_gsynth_null_outcome() {
        let ds = create_gsynth_dataset();
        for col in ["outcome", "unit", "treated"] {
            match gs(&null_at(&ds, col, 14), gs_cfg()) {
                Err(EconError::NullValues { column, count }) => {
                    assert_eq!((column.as_str(), count), (col, 1));
                }
                other => panic!("null {col}: expected NullValues, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_na_gsynth_fractional_time() {
        match gs(&quarterly_time(&create_gsynth_dataset()), gs_cfg()) {
            Err(EconError::InvalidSpecification { message }) => assert_eq!(
                message,
                "Time column 'time' must hold whole numbers; row 1 (0-based) is 2000.25. \
                 Round or recode it first."
            ),
            other => panic!("fractional time: {other:?}"),
        }
    }

    #[test]
    fn test_na_gsynth_boolean_treatment() {
        let ds = create_gsynth_dataset();
        let key = |r: &GsynthResult| {
            let mut k = vec![r.att.to_bits()];
            k.extend(r.unit_effects.iter().map(|u| u.att.to_bits()));
            k
        };
        let base = key(&gs(&ds, gs_cfg()).unwrap());
        let boolean = gs(&with_dtype(&ds, "treated", DataType::Boolean), gs_cfg()).unwrap();
        assert_eq!(key(&boolean), base, "Boolean treatment");
        // regression guards: Int64 and whole-number Float64 times
        for dtype in [DataType::Int64, DataType::Float64] {
            let r = gs(&with_dtype(&ds, "time", dtype.clone()), gs_cfg()).unwrap();
            assert_eq!(key(&r), base, "time as {dtype}");
        }
    }

    #[test]
    fn test_gsynth_unit_order_is_deterministic() {
        let ds = create_gsynth_dataset();
        let r = gs(&ds, gs_cfg()).unwrap();
        let labels: Vec<&str> = r.unit_effects.iter().map(|u| u.unit.as_str()).collect();
        assert_eq!(labels, ["T1", "T2"]);
        // the CV folds follow the control-unit order
        let cv = || GsynthConfig {
            n_factors: 0,
            cross_validate: true,
            max_factors: 3,
            cv_folds: 2,
            min_pre_periods: 2,
            ..Default::default()
        };
        let fingerprint = |r: &GsynthResult| {
            let mut k = vec![r.n_factors as u64, r.cv_mspe.len() as u64, r.att.to_bits()];
            for (factors, mspe) in &r.cv_mspe {
                k.push(*factors as u64);
                k.push(mspe.to_bits());
            }
            k
        };
        let a = gs(&ds, cv()).unwrap();
        let b = gs(&ds, cv()).unwrap();
        assert_eq!(fingerprint(&a), fingerprint(&b));
    }
}
