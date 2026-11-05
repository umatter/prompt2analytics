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

crate::option_enum!(
    GsynthEstimator,
    crate::options::Choices {
        option: "gsynth estimator",
        values: &[
            (
                &["ife", "gsynth", "interactive_fixed_effects"],
                GsynthEstimator::Ife,
            ),
            (
                &["mc", "matrix_completion", "matrixcompletion"],
                GsynthEstimator::MatrixCompletion,
            ),
        ],
    }
);

crate::option_enum!(
    GsynthForce,
    crate::options::Choices {
        option: "gsynth fixed effects (force)",
        values: &[
            (&["none"], GsynthForce::None),
            (&["unit", "individual"], GsynthForce::Unit),
            (&["time"], GsynthForce::Time),
            (&["two_way", "twoway"], GsynthForce::TwoWay),
        ],
    }
);

/// Default bootstrap seed when `GsynthConfig::seed` is `None`.
pub const GSYNTH_DEFAULT_SEED: u64 = 42;

/// Configuration for generalized synthetic control.
#[derive(Debug, Clone)]
pub struct GsynthConfig {
    /// Number of latent factors: the fixed r without cross-validation, the smallest r
    /// tried with it (R `r[1]`)
    pub n_factors: usize,
    /// Largest r tried by cross-validation (R `r[2]`); capped at what the data identify
    pub max_factors: usize,
    /// Cross-validate r (IFE: leave one pre-treatment period out) or lambda (MC: k-fold)
    pub cross_validate: bool,
    /// Matrix-completion CV folds over the treated pre-period positions (>= 2 with CV)
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
    /// Relative convergence criterion of the IFE iteration with covariates and of
    /// matrix completion
    pub tolerance: f64,
    /// Maximum iterations per fit
    pub max_iter: usize,
    /// Matrix-completion penalty (R `lambda`): empty = fect's default grid (needs CV), one
    /// value = fixed, several = the CV grid
    pub lambda: Vec<f64>,
    /// Size of the default lambda grid (R `nlambda`, at least 3)
    pub n_lambda: usize,
    /// Bootstrap confidence level
    pub level: f64,
    /// Bootstrap seed (`None`: [`GSYNTH_DEFAULT_SEED`])
    pub seed: Option<u64>,
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
            lambda: Vec::new(),
            n_lambda: 10,
            level: 0.95,
            seed: None,
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
    /// Event-time ATT (gsynth `att`): event time `e = t - T0_j + 1` (first treated period 1,
    /// pre-treatment periods <= 0) -> mean effect over the treated units observed at `e`
    pub dynamic_effects: Vec<(i64, f64)>,
    /// Residual variance of the control fit (gsynth `sigma2`; NaN without residual
    /// degrees of freedom)
    #[serde(default)]
    pub sigma2: f64,
    /// Grand mean of the fixed-effects part (gsynth `mu`)
    #[serde(default)]
    pub mu: f64,
    /// Matrix-completion CV: (lambda, MSPE)
    #[serde(default)]
    pub cv_lambda: Vec<(f64, f64)>,
    /// Matrix-completion penalty used
    #[serde(default)]
    pub lambda: Option<f64>,
    /// Treated units observed at each event time (gsynth `count`)
    #[serde(default)]
    pub dynamic_counts: Vec<(i64, usize)>,
    /// Bootstrap SE per event time (empty without bootstrap)
    #[serde(default)]
    pub dynamic_se: Vec<(i64, f64)>,
    /// Bootstrap settings and outcome
    #[serde(default)]
    pub bootstrap: Option<GsynthBootstrap>,
    /// The main fit's iteration (`None`: closed form)
    #[serde(default)]
    pub convergence: Option<Convergence>,
    /// Warnings
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Which bootstrap produced the standard errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GsynthBootstrapMethod {
    /// gsynth's parametric bootstrap (estimator "gsynth"; empirical errors).
    Parametric,
    /// Unit (nonparametric) bootstrap (gsynth's inference for estimator "mc").
    Nonparametric,
}

/// Bootstrap settings and outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GsynthBootstrap {
    /// Method
    pub method: GsynthBootstrapMethod,
    /// Successful replications
    pub replications: usize,
    /// Replications dropped (failed fit or failed error draw)
    pub failed: usize,
    /// Confidence level
    pub level: f64,
    /// Seed
    pub seed: u64,
}

impl ResultNotes for GsynthResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        self.convergence.clone()
    }
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

        let notes = notes_text(self);
        if !notes.is_empty() {
            writeln!(f)?;
            write!(f, "{notes}")?;
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
    if config.bootstrap_se {
        if config.n_bootstrap < 2 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "n_bootstrap must be at least 2 for bootstrap standard errors (got {})",
                    config.n_bootstrap
                ),
            });
        }
        crate::traits::estimator::validate_level(config.level, "level")?;
    }
    if let Some(l) = config
        .lambda
        .iter()
        .find(|l| !(l.is_finite() && **l >= 0.0))
    {
        return Err(EconError::InvalidSpecification {
            message: format!("gsynth lambda values must be finite and non-negative (got {l})"),
        });
    }
    if config.cross_validate {
        if config.cv_folds < 2 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "cv_folds must be at least 2 for cross-validation (got {})",
                    config.cv_folds
                ),
            });
        }
        if config.n_factors > config.max_factors {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "n_factors = {} (the smallest r cross-validated) is above max_factors = {}",
                    config.n_factors, config.max_factors
                ),
            });
        }
    }

    // One balanced panel, units in R factor() order (missing values,
    // missing cells and duplicate cells are refused).
    let panel = read_gsynth_panel(dataset.df(), y_col, d_col, unit_col, time_col, x_cols)?;
    let m = gsynth_mats(panel)?;
    let n_times = m.times.len();
    let n_treated = m.treated.len();
    let n_control = m.controls.len();
    let t0_min = *m.first_treated.iter().min().expect("treated units");

    if t0_min < config.min_pre_periods {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Need at least {} pre-treatment periods, found {}",
                config.min_pre_periods, t0_min
            ),
        });
    }
    // min_pre_periods = 0 lets a panel treated from its first period through;
    // the factor model needs at least one pre-treatment period.
    if t0_min == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Need at least 1 pre-treatment period before the first treated period"
                .to_string(),
        });
    }
    let n_pre_periods = t0_min;
    let n_post_periods = n_times - t0_min;
    let covariate_names: Vec<String> = x_cols.iter().map(|s| s.to_string()).collect();

    if matches!(config.estimator, GsynthEstimator::MatrixCompletion) {
        if !config.cross_validate && config.lambda.len() != 1 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "gsynth estimator 'mc' without cross-validation needs one lambda \
                     (GsynthConfig::lambda); got {}",
                    config.lambda.len()
                ),
            });
        }
        if config.cross_validate && config.lambda.is_empty() && config.n_lambda < 3 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "gsynth n_lambda must be at least 3 for the default lambda grid (got {})",
                    config.n_lambda
                ),
            });
        }
    }

    if config.bootstrap_se {
        return Err(EconError::NotImplemented(
            "gsynth bootstrap standard errors are not implemented yet; set bootstrap_se = false"
                .to_string(),
        ));
    }

    let mut warnings = Vec::new();
    let est = if matches!(config.estimator, GsynthEstimator::MatrixCompletion) {
        mc_estimate(&m, &config, &covariate_names, &mut warnings)?
    } else {
        let r_id = identified_factors(&m, config.force)?;
        let too_many = |r: usize| EconError::InvalidSpecification {
            message: format!(
                "n_factors = {r} is more than the data can identify: at most {r_id} ({n_control} \
                 control units, {n_times} periods, {t0_min} pre-treatment periods of the earliest \
                 treated unit, force {:?}). Use fewer factors.",
                config.force
            ),
        };
        if config.n_factors > r_id {
            return Err(too_many(config.n_factors));
        }
        let spec = |r: usize| IfeSpec {
            r,
            force: config.force,
            tolerance: config.tolerance,
            max_iter: config.max_iter,
        };

        let (fit, cv_mspe) = if config.cross_validate {
            let hi = config.max_factors.min(r_id);
            if config.max_factors > r_id {
                warnings.push(format!(
                    "max_factors = {} exceeds what the data identify; cross-validated r = {}..{r_id}",
                    config.max_factors, config.n_factors
                ));
            }
            let mut cv_mspe = Vec::with_capacity(hi + 1 - config.n_factors);
            let mut best: Option<(f64, GsynthFit)> = None;
            let (mut not_converged, mut singular) = (0usize, Vec::new());
            for r in config.n_factors..=hi {
                crate::resource::checkpoint()?;
                let (cand, parts) = match gsynth_estimate_parts(&m, &spec(r), &covariate_names) {
                    Ok(v) => v,
                    Err(EconError::InvalidSpecification { message })
                        if message.contains("are collinear over unit") =>
                    {
                        singular.push(r);
                        cv_mspe.push((r, f64::INFINITY));
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                if cand.ife.convergence.as_ref().is_some_and(|c| !c.converged) {
                    not_converged += 1;
                }
                let mspe = match loo_mspe(&m, &parts)? {
                    Some(v) => v,
                    None => {
                        singular.push(r);
                        f64::INFINITY
                    }
                };
                cv_mspe.push((r, mspe));
                if best.as_ref().is_none_or(|(b, _)| mspe < *b) {
                    best = Some((mspe, cand));
                }
            }
            if not_converged > 0 {
                warnings.push(format!(
                    "{not_converged} cross-validation fit(s) did not converge within max_iter = {}",
                    config.max_iter
                ));
            }
            if !singular.is_empty() {
                warnings.push(format!(
                    "cross-validation: the treated units' loadings are not identified for r = {:?}; \
                     their MSPE is infinite",
                    singular
                ));
            }
            match best {
                Some((mspe, fit)) if mspe.is_finite() => (fit, cv_mspe),
                _ => {
                    return Err(EconError::Computation(
                        "gsynth cross-validation: no number of factors gave identified treated \
                         loadings"
                            .to_string(),
                    ));
                }
            }
        } else {
            (
                gsynth_estimate(&m, &spec(config.n_factors), &covariate_names)?,
                Vec::new(),
            )
        };

        if let Some(note) = fit
            .ife
            .convergence
            .as_ref()
            .and_then(|c| c.note("gsynth IFE"))
        {
            warnings.push(note);
        }
        if fit.ife.sigma2.is_nan() {
            warnings.push(
                "the residual variance sigma2 is not identified (no residual degrees of freedom \
                 on the controls); it is reported as NaN"
                    .to_string(),
            );
        }

        Estimate {
            n_factors: fit.ife.factors.ncols(),
            eff: fit.eff,
            att: fit.att,
            mu: fit.ife.mu,
            sigma2: fit.ife.sigma2,
            beta: fit.ife.beta,
            factors: Some(fit.ife.factors),
            loadings: Some(fit.ife.loadings),
            lambda: None,
            cv_lambda: Vec::new(),
            cv_mspe,
            convergence: fit.ife.convergence,
        }
    };

    let (dynamic_effects, dynamic_counts) = event_time_att(&est.eff, &m.first_treated);
    let mut pre_sum = 0.0;
    let mut pre_n = 0usize;
    let mut unit_effects = Vec::with_capacity(n_treated);
    for (j, &t0) in m.first_treated.iter().enumerate() {
        for t in 0..t0 {
            pre_sum += est.eff[[t, j]].powi(2);
            pre_n += 1;
        }
        let effects: Vec<(i64, f64)> = (t0..n_times)
            .map(|t| (m.times[t], est.eff[[t, j]]))
            .collect();
        let att = effects.iter().map(|(_, e)| e).sum::<f64>() / effects.len() as f64;
        unit_effects.push(UnitEffect {
            unit: m.units[m.treated[j]].clone(),
            treatment_time: m.times[t0],
            effects,
            att,
            se: None,
        });
    }
    let pre_mspe = pre_sum / pre_n as f64;

    Ok(GsynthResult {
        att: est.att,
        att_se: None,
        att_ci: None,
        p_value: None,
        n_treated,
        n_control,
        n_pre_periods,
        n_post_periods,
        n_factors: est.n_factors,
        cv_mspe: est.cv_mspe,
        unit_effects,
        factors: est.factors,
        loadings: est.loadings,
        beta: est.beta,
        covariate_names,
        pre_mspe,
        estimator: config.estimator,
        force: config.force,
        dynamic_effects,
        sigma2: est.sigma2,
        mu: est.mu,
        cv_lambda: est.cv_lambda,
        lambda: est.lambda,
        dynamic_counts,
        dynamic_se: Vec::new(),
        bootstrap: None,
        convergence: est.convergence,
        warnings,
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

// ─── gsynth estimation (S3-S6) ──────────────────────────────────────────────────

/// gsynth panel as T x N matrices: rows = periods in time order, columns = units in factor
/// order.
struct GsynthMats {
    y: Array2<f64>,
    d: Array2<f64>,
    x: Vec<Array2<f64>>,
    units: Vec<String>,
    times: Vec<i64>,
    /// Columns of the treated units (any D = 1), in unit order.
    treated: Vec<usize>,
    /// Columns of the never-treated units.
    controls: Vec<usize>,
    /// Per treated unit: T0_j, the index of its first treated period.
    first_treated: Vec<usize>,
}

/// `m` transposed into a `try_zeros2` buffer.
fn transpose_into(m: &Array2<f64>, what: &str) -> EconResult<Array2<f64>> {
    let (r, c) = m.dim();
    let mut out = crate::resource::try_zeros2(c, r, what)?;
    out.assign(&m.t());
    Ok(out)
}

/// The columns `cols` of `m`.
fn select_columns(m: &Array2<f64>, cols: &[usize], what: &str) -> EconResult<Array2<f64>> {
    let mut out = crate::resource::try_zeros2(m.nrows(), cols.len(), what)?;
    for (k, &c) in cols.iter().enumerate() {
        out.column_mut(k).assign(&m.column(c));
    }
    Ok(out)
}

/// S3: the panel as T x N matrices; D must be 0/1 and absorbing.
fn gsynth_mats(panel: GsynthPanel) -> EconResult<GsynthMats> {
    let GsynthPanel {
        units,
        times,
        y,
        d,
        x,
    } = panel;
    let y = transpose_into(&y, "gsynth outcome panel (units x periods)")?;
    let d = transpose_into(&d, "gsynth treatment panel (units x periods)")?;
    let x = x
        .iter()
        .map(|m| transpose_into(m, "gsynth covariate panel (units x periods)"))
        .collect::<EconResult<Vec<_>>>()?;
    let n_t = times.len();
    let mut treated = Vec::new();
    let mut controls = Vec::new();
    let mut first_treated = Vec::new();
    for (i, unit) in units.iter().enumerate() {
        let col = d.column(i);
        if let Some((t, v)) = col
            .iter()
            .enumerate()
            .find(|(_, v)| **v != 0.0 && **v != 1.0)
        {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "gsynth needs a 0/1 treatment: unit '{unit}' has {v} at time {}",
                    times[t]
                ),
            });
        }
        match (0..n_t).find(|&t| col[t] == 1.0) {
            None => controls.push(i),
            Some(t0) => {
                if let Some(t2) = (t0..n_t).find(|&t| col[t] == 0.0) {
                    return Err(EconError::InvalidSpecification {
                        message: format!(
                            "gsynth needs an absorbing treatment: unit '{unit}' is treated at \
                             time {} and untreated at {} (R: \"Gsynth can't be used when \
                             treatments have reversals.\")",
                            times[t0], times[t2]
                        ),
                    });
                }
                treated.push(i);
                first_treated.push(t0);
            }
        }
    }
    if treated.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "No treated units found".to_string(),
        });
    }
    if controls.len() < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Need at least 2 control units".to_string(),
        });
    }
    Ok(GsynthMats {
        y,
        d,
        x,
        units,
        times,
        treated,
        controls,
        first_treated,
    })
}

/// IFE options of one fit.
#[derive(Debug, Clone, Copy)]
struct IfeSpec {
    r: usize,
    force: GsynthForce,
    tolerance: f64,
    max_iter: usize,
}

/// (u, v): unit effects, time effects.
fn force_dims(force: GsynthForce) -> (usize, usize) {
    match force {
        GsynthForce::None => (0, 0),
        GsynthForce::Unit => (1, 0),
        GsynthForce::Time => (0, 1),
        GsynthForce::TwoWay => (1, 1),
    }
}

/// Subtract the grand mean and, as `force` asks, the column (unit) and row (time) effects.
fn gs_demean(m: &Array2<f64>, force: GsynthForce) -> Array2<f64> {
    let (u, v) = force_dims(force);
    let (n_t, n) = m.dim();
    let mu = m.sum() / (n_t * n) as f64;
    let col = m.mean_axis(ndarray::Axis(0)).expect("non-empty");
    let row = m.mean_axis(ndarray::Axis(1)).expect("non-empty");
    Array2::from_shape_fn((n_t, n), |(t, i)| {
        let mut z = m[[t, i]] - mu;
        if u == 1 {
            z -= col[i] - mu;
        }
        if v == 1 {
            z -= row[t] - mu;
        }
        z
    })
}

/// `vec(m)` in column-major order (R's `c(M)`).
fn vec_cm(m: &Array2<f64>) -> Array1<f64> {
    m.t().iter().copied().collect()
}

/// The inverse of [`vec_cm`].
fn unvec_cm(v: &Array1<f64>, n_t: usize, n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n_t, n), |(t, i)| v[t + n_t * i])
}

/// Factors and loadings of `U` (T x N): `F = sqrt(T) U_svd[:, :r]`, `Lambda = U'F / T`,
/// `FE = F Lambda'`.
fn gs_pca(u: &Array2<f64>, r: usize) -> EconResult<(Array2<f64>, Array2<f64>, Array2<f64>)> {
    crate::resource::checkpoint()?;
    let (n_t, n) = u.dim();
    if r == 0 {
        return Ok((
            Array2::zeros((n_t, 0)),
            Array2::zeros((n, 0)),
            Array2::zeros((n_t, n)),
        ));
    }
    let mat = crate::linalg::matrix_ops::ndarray_to_faer(&u.view());
    let svd = mat
        .thin_svd()
        .map_err(|e| EconError::Computation(format!("gsynth factor SVD failed: {e:?}")))?;
    let uu = svd.U();
    if uu.ncols() < r {
        return Err(EconError::Computation(format!(
            "gsynth factor SVD returned {} singular vectors for {r} factors",
            uu.ncols()
        )));
    }
    let scale = (n_t as f64).sqrt();
    let f = Array2::from_shape_fn((n_t, r), |(t, k)| scale * uu[(t, k)]);
    let lambda = u.t().dot(&f) / n_t as f64;
    let fe = f.dot(&lambda.t());
    Ok((f, lambda, fe))
}

/// The IFE fit on the controls (S4).
#[derive(Debug, Clone)]
struct IfeFit {
    beta: Vec<f64>,
    mu: f64,
    xi: Array1<f64>,
    alpha: Array1<f64>,
    factors: Array2<f64>,
    loadings: Array2<f64>,
    fitted: Array2<f64>,
    residuals: Array2<f64>,
    sigma2: f64,
    convergence: Option<Convergence>,
}

/// The stacked demeaned covariates ((T N) x p, column-major cells) and their QR; aliased
/// columns are refused, naming them.
fn stacked_covariates(
    x: &[Array2<f64>],
    force: GsynthForce,
    names: &[String],
) -> EconResult<Option<(Array2<f64>, crate::linalg::solve::Qr)>> {
    if x.is_empty() {
        return Ok(None);
    }
    let (n_t, n) = x[0].dim();
    let cells = crate::resource::checked_len(&[n_t, n], "gsynth stacked covariates")?;
    let mut xx = crate::resource::try_zeros2(
        cells,
        x.len(),
        "gsynth stacked covariates (control cells x covariates)",
    )?;
    for (j, m) in x.iter().enumerate() {
        xx.column_mut(j).assign(&vec_cm(&gs_demean(m, force)));
    }
    let qr = crate::linalg::solve::Qr::decompose(xx.view(), crate::linalg::solve::LM_QR_TOL)?;
    let aliased = qr.aliased();
    if !aliased.is_empty() {
        let bad: Vec<&str> = aliased
            .iter()
            .map(|&j| names.get(j).map(String::as_str).unwrap_or("?"))
            .collect();
        return Err(EconError::InvalidSpecification {
            message: format!(
                "gsynth: covariate(s) {} are collinear with the other covariates or the fixed \
                 effects; drop them",
                bad.join(", ")
            ),
        });
    }
    Ok(Some((xx, qr)))
}

/// S4: Bai's IFE on the controls `y` (T x N_co) with covariates `x`.
fn ife_controls(
    y: &Array2<f64>,
    x: &[Array2<f64>],
    spec: &IfeSpec,
    names: &[String],
) -> EconResult<IfeFit> {
    crate::resource::checkpoint()?;
    let (n_t, n) = y.dim();
    let (u, v) = force_dims(spec.force);
    let r = spec.r;
    let yy = gs_demean(y, spec.force);
    let p = x.len();
    let (beta, convergence) = match stacked_covariates(x, spec.force, names)? {
        None => (Vec::new(), None),
        Some((xx, qr)) => {
            let vy = vec_cm(&yy);
            let norm_y = vy.dot(&vy).sqrt().max(f64::MIN_POSITIVE);
            let mut beta = qr.coefficients(vy.view())?;
            let mut step = f64::NAN;
            let mut converged = false;
            let mut k = 0;
            while k < spec.max_iter {
                crate::resource::checkpoint()?;
                k += 1;
                let resid = unvec_cm(&(&vy - &xx.dot(&beta)), n_t, n);
                let (_, _, fe) = gs_pca(&resid, r)?;
                let target = &vy - &vec_cm(&fe);
                let next = qr.coefficients(target.view())?;
                let change = xx.dot(&(&next - &beta));
                step = change.dot(&change).sqrt() / norm_y;
                beta = next;
                if step < spec.tolerance {
                    converged = true;
                    break;
                }
            }
            let conv = Convergence::new(
                converged,
                k,
                spec.max_iter,
                "relative change of the covariate fit",
                step,
                spec.tolerance,
            );
            (beta.to_vec(), Some(conv))
        }
    };
    let mut resid = y.clone();
    for (j, m) in x.iter().enumerate() {
        resid.scaled_add(-beta[j], m);
    }
    let mu = resid.sum() / (n_t * n) as f64;
    let xi = if v == 1 {
        resid.mean_axis(ndarray::Axis(1)).expect("non-empty") - mu
    } else {
        Array1::zeros(n_t)
    };
    let alpha = if u == 1 {
        resid.mean_axis(ndarray::Axis(0)).expect("non-empty") - mu
    } else {
        Array1::zeros(n)
    };
    let (factors, loadings, fe) = gs_pca(&gs_demean(&resid, spec.force), r)?;
    let fitted = Array2::from_shape_fn((n_t, n), |(t, i)| mu + alpha[i] + xi[t] + fe[[t, i]]);
    let residuals = &resid - &fitted;
    let (n_fe, n_dim) = match spec.force {
        GsynthForce::None => (1, 0),
        GsynthForce::Unit => (n, 1),
        GsynthForce::Time => (n_t, 1),
        GsynthForce::TwoWay => (n + n_t - 1, 2),
    };
    let df = (n_t * n) as i128
        - n_fe as i128
        - p as i128
        - (r as i128) * ((n + n_t) as i128 - n_dim as i128)
        + (r * r) as i128;
    let sse: f64 = residuals.iter().map(|e| e * e).sum();
    let sigma2 = if df > 0 { sse / df as f64 } else { f64::NAN };
    Ok(IfeFit {
        beta,
        mu,
        xi,
        alpha,
        factors,
        loadings,
        fitted,
        residuals,
        sigma2,
        convergence,
    })
}

/// S4.11: `min(N_co - v, T - u, T0_min - u - 1)`.
fn identified_factors(m: &GsynthMats, force: GsynthForce) -> EconResult<usize> {
    let (u, v) = force_dims(force);
    let t0_min = *m.first_treated.iter().min().expect("treated units");
    let n_t = m.times.len();
    let bound = [
        m.controls.len().checked_sub(v),
        n_t.checked_sub(u),
        t0_min.checked_sub(u + 1),
    ];
    match bound.iter().copied().collect::<Option<Vec<usize>>>() {
        Some(b) => Ok(b.into_iter().min().expect("three bounds")),
        None => Err(EconError::InsufficientData {
            required: u + 1,
            provided: t0_min,
            context: format!(
                "gsynth with force {force:?}: pre-treatment periods of the earliest treated unit \
                 (the fixed effects leave no degrees of freedom for a factor model)"
            ),
        }),
    }
}

/// S5 inputs of the treated units: `U_tr` and `F^`.
struct TreatedParts {
    u_tr: Array2<f64>,
    f_hat: Array2<f64>,
}

fn treated_parts(m: &GsynthMats, ife: &IfeFit, force: GsynthForce) -> EconResult<TreatedParts> {
    let (u, v) = force_dims(force);
    let n_t = m.times.len();
    let mut u_tr = select_columns(&m.y, &m.treated, "gsynth treated outcomes")?;
    for (j, x) in m.x.iter().enumerate() {
        let xt = select_columns(x, &m.treated, "gsynth treated covariates")?;
        u_tr.scaled_add(-ife.beta[j], &xt);
    }
    u_tr -= ife.mu;
    if v == 1 {
        for (t, mut row) in u_tr.rows_mut().into_iter().enumerate() {
            row -= ife.xi[t];
        }
    }
    let r = ife.factors.ncols();
    let mut f_hat = Array2::<f64>::ones((n_t, r + u));
    f_hat.slice_mut(ndarray::s![.., ..r]).assign(&ife.factors);
    Ok(TreatedParts { u_tr, f_hat })
}

fn loadings_error(m: &GsynthMats, j: usize, r: usize) -> EconError {
    EconError::InvalidSpecification {
        message: format!(
            "gsynth: the {r} factors (and intercept) are collinear over unit '{}''s {} \
             pre-treatment periods; use fewer factors",
            m.units[m.treated[j]], m.first_treated[j]
        ),
    }
}

/// The rows `rows` of `a`.
fn select_rows(a: &Array2<f64>, rows: &[usize]) -> Array2<f64> {
    Array2::from_shape_fn((rows.len(), a.ncols()), |(i, k)| a[[rows[i], k]])
}

/// The base fit (S5).
#[derive(Debug, Clone)]
struct GsynthFit {
    /// T x N_tr treatment effects (pre periods: prediction errors).
    eff: Array2<f64>,
    att: f64,
    ife: IfeFit,
}

/// Effects of the treated units given `U_tr` and `F^`.
fn treated_effects(m: &GsynthMats, parts: &TreatedParts, r: usize) -> EconResult<Array2<f64>> {
    let mut eff = parts.u_tr.clone();
    if parts.f_hat.ncols() == 0 {
        return Ok(eff);
    }
    for (j, &t0) in m.first_treated.iter().enumerate() {
        let pre: Vec<usize> = (0..t0).collect();
        let fp = select_rows(&parts.f_hat, &pre);
        let yp = Array1::from_iter(pre.iter().map(|&t| parts.u_tr[[t, j]]));
        let ls = crate::linalg::solve::least_squares(fp.view(), yp.view())?;
        if ls.rank < fp.ncols() {
            return Err(loadings_error(m, j, r));
        }
        let pred = parts.f_hat.dot(&ls.coefficients);
        let mut col = eff.column_mut(j);
        col -= &pred;
    }
    Ok(eff)
}

/// Mean effect over the treated post-period cells.
fn att_of(eff: &Array2<f64>, first_treated: &[usize]) -> f64 {
    let (mut sum, mut count) = (0.0, 0usize);
    for (j, &t0) in first_treated.iter().enumerate() {
        for t in t0..eff.nrows() {
            sum += eff[[t, j]];
            count += 1;
        }
    }
    sum / count as f64
}

/// S5: the IFE fit on the controls and the effects of the treated units.
fn gsynth_estimate(m: &GsynthMats, spec: &IfeSpec, names: &[String]) -> EconResult<GsynthFit> {
    gsynth_estimate_parts(m, spec, names).map(|(fit, _)| fit)
}

fn gsynth_estimate_parts(
    m: &GsynthMats,
    spec: &IfeSpec,
    names: &[String],
) -> EconResult<(GsynthFit, TreatedParts)> {
    let y_co = select_columns(&m.y, &m.controls, "gsynth control outcomes")?;
    let x_co =
        m.x.iter()
            .map(|x| select_columns(x, &m.controls, "gsynth control covariates"))
            .collect::<EconResult<Vec<_>>>()?;
    let ife = ife_controls(&y_co, &x_co, spec, names)?;
    let parts = treated_parts(m, &ife, spec.force)?;
    let eff = treated_effects(m, &parts, spec.r)?;
    let att = att_of(&eff, &m.first_treated);
    Ok((GsynthFit { eff, att, ife }, parts))
}

/// S6: leave-one-pre-period-out MSPE of the treated units, by PRESS (`e / (1 - h)`) from
/// each unit's pre-period fit; a pre-period whose leverage is 1 is refitted explicitly.
/// `None` when a regression is rank-deficient.
fn loo_mspe(m: &GsynthMats, parts: &TreatedParts) -> EconResult<Option<f64>> {
    let (mut sum, mut count) = (0.0, 0usize);
    let k = parts.f_hat.ncols();
    for (j, &t0) in m.first_treated.iter().enumerate() {
        crate::resource::checkpoint()?;
        if k == 0 {
            for t in 0..t0 {
                sum += parts.u_tr[[t, j]].powi(2);
                count += 1;
            }
            continue;
        }
        let pre: Vec<usize> = (0..t0).collect();
        let fp = select_rows(&parts.f_hat, &pre);
        let yp = Array1::from_iter(pre.iter().map(|&t| parts.u_tr[[t, j]]));
        let ls = crate::linalg::solve::least_squares(fp.view(), yp.view())?;
        if ls.rank < k {
            return Ok(None);
        }
        for s in 0..t0 {
            let row = fp.row(s);
            let h = row.dot(&ls.unscaled_cov.dot(&row));
            let err = if 1.0 - h > 1e-10 {
                ls.residuals[s] / (1.0 - h)
            } else {
                let keep: Vec<usize> = (0..t0).filter(|&t| t != s).collect();
                let fk = select_rows(&parts.f_hat, &keep);
                let yk = Array1::from_iter(keep.iter().map(|&t| parts.u_tr[[t, j]]));
                if keep.len() < k {
                    return Ok(None);
                }
                let lk = crate::linalg::solve::least_squares(fk.view(), yk.view())?;
                if lk.rank < k {
                    return Ok(None);
                }
                yp[s] - row.dot(&lk.coefficients)
            };
            sum += err * err;
            count += 1;
        }
    }
    Ok(Some(sum / count as f64))
}

// ─── Matrix completion (S7, S8) ───────────────────────────────────────────────

/// Relative criterion and iteration cap of the lambda = +inf start fit (fect's `initialFit`).
const MC_START_TOL: f64 = 1e-14;
const MC_START_MAX_ITER: usize = 10_000;

/// Options of one matrix-completion fit.
#[derive(Debug, Clone, Copy)]
struct McSpec {
    lambda: f64,
    force: GsynthForce,
    tolerance: f64,
    max_iter: usize,
}

/// A matrix-completion fit.
#[derive(Debug, Clone)]
struct McFit {
    fitted: Array2<f64>,
    beta: Vec<f64>,
    mu: f64,
    rank: usize,
    convergence: Convergence,
}

/// The stacked demeaned covariates of the whole panel, decomposed once.
struct McPrep {
    cov: Option<(Array2<f64>, crate::linalg::solve::Qr)>,
}

fn mc_prep(m: &GsynthMats, force: GsynthForce, names: &[String]) -> EconResult<McPrep> {
    Ok(McPrep {
        cov: stacked_covariates(&m.x, force, names)?,
    })
}

/// `U diag((d - tau)_+) V'` and the number of positive thresholded values.
fn svt(e: &Array2<f64>, tau: f64) -> EconResult<(Array2<f64>, usize)> {
    let (n_t, n) = e.dim();
    let mat = crate::linalg::matrix_ops::ndarray_to_faer(&e.view());
    let svd = mat.thin_svd().map_err(|err| {
        EconError::Computation(format!("gsynth matrix-completion SVD failed: {err:?}"))
    })?;
    let d = svd.S().column_vector();
    let (u, v) = (svd.U(), svd.V());
    let mut out = Array2::<f64>::zeros((n_t, n));
    let mut rank = 0;
    for k in 0..d.nrows() {
        let s = d[k] - tau;
        if s > 0.0 {
            rank += 1;
            for t in 0..n_t {
                let a = s * u[(t, k)];
                for i in 0..n {
                    out[[t, i]] += a * v[(i, k)];
                }
            }
        }
    }
    Ok((out, rank))
}

/// S7: soft-impute for `min 1/2 sum_O (Y - mu - alpha - xi - X beta - L)^2 + lambda T N ||L||_*`
/// from `start`.
fn mc_fit(
    m: &GsynthMats,
    prep: &McPrep,
    observed: &Array2<f64>,
    spec: &McSpec,
    start: &Array2<f64>,
) -> EconResult<McFit> {
    let (n_t, n) = m.y.dim();
    let (u, v) = force_dims(spec.force);
    let tau = spec.lambda * (n_t * n) as f64;
    let mut fit = start.clone();
    let mut l = Array2::<f64>::zeros((n_t, n));
    let mut beta: Vec<f64> = Vec::new();
    let (mut mu, mut rank, mut change, mut converged, mut it) = (0.0, 0usize, f64::NAN, false, 0);
    while it < spec.max_iter {
        crate::resource::checkpoint()?;
        it += 1;
        let z = Array2::from_shape_fn((n_t, n), |(t, i)| {
            if observed[[t, i]] == 1.0 {
                m.y[[t, i]]
            } else {
                fit[[t, i]]
            }
        });
        let r = &z - &l;
        let mut xb = Array2::<f64>::zeros((n_t, n));
        if let Some((_, qr)) = &prep.cov {
            let b = qr.coefficients(vec_cm(&gs_demean(&r, spec.force)).view())?;
            beta = b.to_vec();
            for (j, x) in m.x.iter().enumerate() {
                xb.scaled_add(beta[j], x);
            }
        }
        let rb = &r - &xb;
        mu = rb.sum() / (n_t * n) as f64;
        let alpha = if u == 1 {
            rb.mean_axis(ndarray::Axis(0)).expect("non-empty") - mu
        } else {
            Array1::zeros(n)
        };
        let xi = if v == 1 {
            rb.mean_axis(ndarray::Axis(1)).expect("non-empty") - mu
        } else {
            Array1::zeros(n_t)
        };
        let fe = Array2::from_shape_fn((n_t, n), |(t, i)| mu + alpha[i] + xi[t]);
        if spec.lambda.is_finite() {
            let (lk, rk) = svt(&(&z - &fe - &xb), tau)?;
            l = lk;
            rank = rk;
        } else {
            l.fill(0.0);
            rank = 0;
        }
        let new = &fe + &xb + &l;
        let diff = &new - &fit;
        let num = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
        let den = fit.iter().map(|x| x * x).sum::<f64>().sqrt();
        change = num / den;
        fit = new;
        if change < spec.tolerance {
            converged = true;
            break;
        }
    }
    Ok(McFit {
        fitted: fit,
        beta,
        mu,
        rank,
        convergence: Convergence::new(
            converged,
            it,
            spec.max_iter,
            "relative change of the fitted values",
            change,
            spec.tolerance,
        ),
    })
}

/// The untreated-cell mask `1 - D`.
fn mc_observed(m: &GsynthMats) -> Array2<f64> {
    m.d.mapv(|d| 1.0 - d)
}

/// The lambda = +inf fit (two-way effects and covariates on the observed cells), from
/// `from` or from the mean of the observed cells.
fn mc_start(
    m: &GsynthMats,
    prep: &McPrep,
    observed: &Array2<f64>,
    force: GsynthForce,
    from: Option<&Array2<f64>>,
) -> EconResult<Array2<f64>> {
    let start = match from {
        Some(f) => f.clone(),
        None => {
            let (mut s, mut c) = (0.0, 0usize);
            for (y, o) in m.y.iter().zip(observed.iter()) {
                if *o == 1.0 {
                    s += y;
                    c += 1;
                }
            }
            Array2::from_elem(m.y.dim(), s / c as f64)
        }
    };
    let spec = McSpec {
        lambda: f64::INFINITY,
        force,
        tolerance: MC_START_TOL,
        max_iter: MC_START_MAX_ITER,
    };
    Ok(mc_fit(m, prep, observed, &spec, &start)?.fitted)
}

/// S8: fect's default grid from the largest singular value of `O (Y - start) / (T N)`.
fn mc_lambda_grid(
    m: &GsynthMats,
    prep: &McPrep,
    start: &Array2<f64>,
    n_lambda: usize,
) -> EconResult<(Vec<f64>, f64)> {
    if n_lambda < 3 {
        return Err(EconError::InvalidSpecification {
            message: format!("gsynth n_lambda must be at least 3 (got {n_lambda})"),
        });
    }
    let _ = prep;
    let (n_t, n) = m.y.dim();
    let observed = mc_observed(m);
    let scale = (n_t * n) as f64;
    let e = Array2::from_shape_fn((n_t, n), |(t, i)| {
        observed[[t, i]] * (m.y[[t, i]] - start[[t, i]]) / scale
    });
    let mat = crate::linalg::matrix_ops::ndarray_to_faer(&e.view());
    let d = mat
        .singular_values()
        .map_err(|err| EconError::Computation(format!("gsynth lambda grid SVD failed: {err:?}")))?;
    let d_max = d.iter().cloned().fold(0.0_f64, f64::max);
    let lmax = d_max.log10();
    let by = 3.0 / (n_lambda - 2) as f64;
    let mut grid: Vec<f64> = (0..n_lambda - 1)
        .map(|i| 10f64.powf(lmax - i as f64 * by))
        .collect();
    grid.push(0.0);
    Ok((grid, d_max))
}

/// S8: k-fold CV MSPE of each lambda over the treated units' pre-period cells (position
/// `p mod k == f` in fold f), each fold's lambdas fitted in grid order from the previous
/// lambda's fit. Returns the MSPEs and the number of non-converged fits.
#[allow(clippy::too_many_arguments)]
fn mc_cv_mspe(
    m: &GsynthMats,
    prep: &McPrep,
    full_start: &Array2<f64>,
    grid: &[f64],
    folds: usize,
    force: GsynthForce,
    tolerance: f64,
    max_iter: usize,
) -> EconResult<(Vec<f64>, usize)> {
    let observed = mc_observed(m);
    let mut sse = vec![0.0; grid.len()];
    let mut count = 0usize;
    let mut not_converged = 0usize;
    for f in 0..folds {
        let mut o_f = observed.clone();
        let mut held: Vec<(usize, usize)> = Vec::new();
        for (j, &t0) in m.first_treated.iter().enumerate() {
            let col = m.treated[j];
            for p in (0..t0).filter(|p| p % folds == f) {
                o_f[[p, col]] = 0.0;
                held.push((p, col));
            }
        }
        let start = mc_start(m, prep, &o_f, force, Some(full_start))?;
        let mut warm = start;
        for (g, &lambda) in grid.iter().enumerate() {
            crate::resource::checkpoint()?;
            let spec = McSpec {
                lambda,
                force,
                tolerance,
                max_iter,
            };
            let fit = mc_fit(m, prep, &o_f, &spec, &warm)?;
            if !fit.convergence.converged {
                not_converged += 1;
            }
            for &(t, i) in &held {
                sse[g] += (m.y[[t, i]] - fit.fitted[[t, i]]).powi(2);
            }
            warm = fit.fitted;
        }
        count += held.len();
    }
    Ok((
        sse.into_iter().map(|s| s / count as f64).collect(),
        not_converged,
    ))
}

/// One estimator's outputs, before the shared summaries.
struct Estimate {
    eff: Array2<f64>,
    att: f64,
    mu: f64,
    sigma2: f64,
    beta: Vec<f64>,
    n_factors: usize,
    factors: Option<Array2<f64>>,
    loadings: Option<Array2<f64>>,
    lambda: Option<f64>,
    cv_lambda: Vec<(f64, f64)>,
    cv_mspe: Vec<(usize, f64)>,
    convergence: Option<Convergence>,
}

/// S7/S8 and S5.11: the matrix-completion estimate, lambda fixed or cross-validated.
fn mc_estimate(
    m: &GsynthMats,
    config: &GsynthConfig,
    names: &[String],
    warnings: &mut Vec<String>,
) -> EconResult<Estimate> {
    let prep = mc_prep(m, config.force, names)?;
    let observed = mc_observed(m);
    let full_start = mc_start(m, &prep, &observed, config.force, None)?;
    let (lambda, cv_lambda) = if config.cross_validate {
        let grid = if config.lambda.is_empty() {
            mc_lambda_grid(m, &prep, &full_start, config.n_lambda)?.0
        } else {
            config.lambda.clone()
        };
        let t0_min = *m.first_treated.iter().min().expect("treated units");
        let folds = config.cv_folds.min(t0_min);
        if folds < 2 {
            return Err(EconError::InsufficientData {
                required: 2,
                provided: t0_min,
                context: "gsynth matrix-completion CV folds over the earliest treated unit's \
                          pre-treatment periods"
                    .to_string(),
            });
        }
        let (mspe, not_converged) = mc_cv_mspe(
            m,
            &prep,
            &full_start,
            &grid,
            folds,
            config.force,
            config.tolerance,
            config.max_iter,
        )?;
        if not_converged > 0 {
            warnings.push(format!(
                "{not_converged} matrix-completion cross-validation fit(s) did not converge \
                 within max_iter = {}",
                config.max_iter
            ));
        }
        let mut best = 0;
        for (g, v) in mspe.iter().enumerate() {
            if v.total_cmp(&mspe[best]).is_lt() {
                best = g;
            }
        }
        (grid[best], grid.iter().copied().zip(mspe).collect())
    } else {
        (config.lambda[0], Vec::new())
    };
    let spec = McSpec {
        lambda,
        force: config.force,
        tolerance: config.tolerance,
        max_iter: config.max_iter,
    };
    let fit = mc_fit(m, &prep, &observed, &spec, &full_start)?;
    if let Some(note) = fit.convergence.note("gsynth matrix completion") {
        warnings.push(note);
    }
    let (n_t, _) = m.y.dim();
    let mut eff = crate::resource::try_zeros2(n_t, m.treated.len(), "gsynth treated effects")?;
    for (j, &c) in m.treated.iter().enumerate() {
        for t in 0..n_t {
            eff[[t, j]] = m.y[[t, c]] - fit.fitted[[t, c]];
        }
    }
    let (mut sse, mut n_obs) = (0.0, 0usize);
    for ((y, f), o) in m.y.iter().zip(fit.fitted.iter()).zip(observed.iter()) {
        if *o == 1.0 {
            sse += (y - f).powi(2);
            n_obs += 1;
        }
    }
    let att = att_of(&eff, &m.first_treated);
    Ok(Estimate {
        eff,
        att,
        mu: fit.mu,
        sigma2: sse / n_obs as f64,
        beta: fit.beta,
        n_factors: fit.rank,
        factors: None,
        loadings: None,
        lambda: Some(lambda),
        cv_lambda,
        cv_mspe: Vec::new(),
        convergence: Some(fit.convergence),
    })
}

/// Event-time ATT (`e = t - T0_j + 1`) over the treated units, and the unit counts.
fn event_time_att(
    eff: &Array2<f64>,
    first_treated: &[usize],
) -> (Vec<(i64, f64)>, Vec<(i64, usize)>) {
    let mut acc: std::collections::BTreeMap<i64, (f64, usize)> = std::collections::BTreeMap::new();
    for (j, &t0) in first_treated.iter().enumerate() {
        for t in 0..eff.nrows() {
            let e = t as i64 - t0 as i64 + 1;
            let a = acc.entry(e).or_insert((0.0, 0));
            a.0 += eff[[t, j]];
            a.1 += 1;
        }
    }
    let effects = acc.iter().map(|(&e, &(s, c))| (e, s / c as f64)).collect();
    let counts = acc.iter().map(|(&e, &(_, c))| (e, c)).collect();
    (effects, counts)
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
        // r_id = min(3 controls - 1, 10 periods - 1, 6 pre-periods - 2) = 2 under two-way
        // effects; usize::MAX was an ndarray shape-overflow panic before T0-WP11
        for r in [3usize, usize::MAX] {
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
                    if message.contains("n_factors") && message.contains("at most 2")),
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
                n_factors: 2,
                ..gs_cfg()
            },
        )
        .unwrap();
        assert_eq!(ok.n_factors, 2);
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
        // n_factors = 0 without CV is r = 0; with one pre-period the two-way effects
        // leave the treated unit's intercept no pre-period to spare (r_id < 0)
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
        assert!(
            matches!(r, Err(EconError::InsufficientData { .. })),
            "ok={}",
            r.is_ok()
        );
    }

    #[test]
    fn test_gsynth_explicit_factor_count_stops_when_cancelled() {
        // An explicit n_factors does not go through the cross-validation; the IFE
        // fit checks the budget once per iteration and per factor extraction
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
        // a long panel with four covariates: the extractor's buffers (at most 2560
        // bytes) and every 40 x 4 panel (1280 bytes) fit under this ceiling, the stacked
        // control covariates (120 x 4, 3840 bytes) do not
        let (mut u, mut t, mut y, mut d) = (vec![], vec![], vec![], vec![]);
        let (mut x1, mut x2, mut x3, mut x4) = (vec![], vec![], vec![], vec![]);
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
                x1.push((0.7 * k + 0.3 * time as f64).sin());
                x2.push((0.2 * k * time as f64).cos());
                x3.push((0.37 * k * time as f64 + 0.5).sin());
                x4.push((0.5 * k + 0.011 * (time * time) as f64).cos());
            }
        }
        let long = Dataset::new(
            df! {
                "unit" => u, "time" => t, "outcome" => y, "treated" => d,
                "x1" => x1, "x2" => x2, "x3" => x3, "x4" => x4
            }
            .unwrap(),
        );
        let xs = ["x1", "x2", "x3", "x4"];
        let r = crate::resource::with_alloc_ceiling(Some(3000), || {
            run_gsynth(&long, "outcome", "treated", "unit", "time", &xs, gs_cfg())
        });
        assert!(
            matches!(r, Err(EconError::ResourceLimit(ref m)) if m.contains("stacked covariates")),
            "{r:?}"
        );
        assert!(run_gsynth(&long, "outcome", "treated", "unit", "time", &xs, gs_cfg()).is_ok());
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

    // =========================================================================
    // gsynth 1.4.0 pins and properties (T3-WP31: CAUS-41, CAUS-43, CAUS-44)
    // =========================================================================

    use p2a_validation::{Check, Tol};

    const SIM_N: usize = 1500; // 50 x 30 cells (gsynth_simdata)
    const STAG_N: usize = 720; // 36 x 20 cells (gsynth_staggered)

    fn tight(force: GsynthForce, r: usize) -> GsynthConfig {
        GsynthConfig {
            n_factors: r,
            cross_validate: false,
            force,
            tolerance: 1e-12,
            max_iter: 5000,
            ..Default::default()
        }
    }

    fn event(r: &GsynthResult) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            r.dynamic_effects.iter().map(|(t, _)| *t as f64).collect(),
            r.dynamic_effects.iter().map(|(_, e)| *e).collect(),
            r.dynamic_counts.iter().map(|(_, c)| *c as f64).collect(),
        )
    }

    /// Pins `<p>_{att_avg, att, time, count, mu, sigma2, rmse, att_unit}` (+ `_beta`).
    fn check_fit(chk: &mut Check, p: &str, r: &GsynthResult, tol: Tol, var_tol: Tol) {
        let (t, att, cnt) = event(r);
        let ua: Vec<f64> = r.unit_effects.iter().map(|u| u.att).collect();
        chk.scalar(&format!("{p}_att_avg"), r.att, tol)
            .vector(&format!("{p}_att"), &att, tol)
            .vector(&format!("{p}_time"), &t, Tol::exact())
            .vector(&format!("{p}_count"), &cnt, Tol::exact())
            .scalar(&format!("{p}_mu"), r.mu, tol)
            .scalar(&format!("{p}_sigma2"), r.sigma2, var_tol)
            .scalar(&format!("{p}_rmse"), r.pre_mspe.sqrt(), var_tol)
            .vector(&format!("{p}_att_unit"), &ua, tol);
        if !r.beta.is_empty() {
            chk.vector(&format!("{p}_beta"), &r.beta, tol);
        }
    }

    fn sim(ds: &Dataset, x: &[&str], c: GsynthConfig) -> EconResult<GsynthResult> {
        run_gsynth(ds, "Y", "D", "id", "time", x, c)
    }

    /// The case's simdata without the case (property tests stay out of the registry).
    fn simdata() -> Dataset {
        let path = p2a_validation::repo_path("validation/datasets/generated/gsynth_simdata.csv");
        Dataset::new(crate::data::DataLoader::load_csv(&path).expect("gsynth_simdata.csv"))
    }

    fn set_col(ds: &Dataset, name: &str, v: Vec<f64>) -> Dataset {
        let mut df = ds.df().clone();
        df.with_column(Column::new(name.into(), v)).unwrap();
        Dataset::new(df)
    }

    fn col(ds: &Dataset, name: &str) -> Vec<f64> {
        let c = ds
            .df()
            .column(name)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap();
        c.f64().unwrap().into_iter().map(|v| v.unwrap()).collect()
    }

    #[test]
    fn test_validate_gsynth_ife_r2() {
        let (case, ds) = crate::test_support::load_case("gsynth_simdata");
        let r = sim(&ds, &[], tight(GsynthForce::TwoWay, 2)).unwrap();
        let mut chk = Check::new(&case);
        check_fit(
            &mut chk,
            "ife_r2_nocov",
            &r,
            Tol::coef(SIM_N),
            Tol::stat(SIM_N),
        );
        chk.scalar("ife_r2_nocov_r", r.n_factors as f64, Tol::exact())
            .finish();
        assert!(r.convergence.is_none() && r.beta.is_empty());
    }

    #[test]
    fn test_validate_gsynth_ife_cov() {
        let (case, ds) = crate::test_support::load_case("gsynth_simdata");
        let r = sim(&ds, &["X1", "X2"], tight(GsynthForce::TwoWay, 2)).unwrap();
        let it = Tol::iterative(1e-8);
        let mut chk = Check::new(&case);
        check_fit(&mut chk, "ife_r2_cov", &r, it, it);
        let d = sim(
            &ds,
            &["X1", "X2"],
            GsynthConfig {
                tolerance: GsynthConfig::default().tolerance,
                ..tight(GsynthForce::TwoWay, 2)
            },
        )
        .unwrap();
        // gsynth stops on an absolute change of beta at its default 1e-3; this package's
        // scale-free rule at 1e-5 lands within 5e-4 of both
        chk.scalar("ife_r2_cov_default_tol_att_avg", d.att, Tol::new(5e-4, 0.0))
            .scalar("ife_r2_cov_att_avg", d.att, Tol::new(5e-4, 0.0))
            .finish();
        assert!(r.convergence.as_ref().unwrap().converged);
    }

    #[test]
    fn test_validate_gsynth_ife_unit() {
        let (case, ds) = crate::test_support::load_case("gsynth_simdata");
        let r = sim(&ds, &[], tight(GsynthForce::Unit, 2)).unwrap();
        let mut chk = Check::new(&case);
        check_fit(
            &mut chk,
            "ife_r2_unit",
            &r,
            Tol::coef(SIM_N),
            Tol::stat(SIM_N),
        );
        chk.finish();
    }

    #[test]
    fn test_validate_gsynth_staggered() {
        let (case, ds) = crate::test_support::load_case("gsynth_staggered");
        let it = Tol::iterative(1e-8);
        let tw = sim(&ds, &["X"], tight(GsynthForce::TwoWay, 2)).unwrap();
        let tw0 = sim(&ds, &[], tight(GsynthForce::TwoWay, 2)).unwrap();
        let un = sim(&ds, &["X"], tight(GsynthForce::Unit, 2)).unwrap();
        let mut chk = Check::new(&case);
        check_fit(&mut chk, "tw_r2_cov", &tw, it, it);
        check_fit(
            &mut chk,
            "tw_r2_nocov",
            &tw0,
            Tol::coef(STAG_N),
            Tol::stat(STAG_N),
        );
        check_fit(&mut chk, "unit_r2_cov", &un, it, it);
        chk.finish();
    }

    #[test]
    fn test_validate_gsynth_ife_cv() {
        let (case, ds) = crate::test_support::load_case("gsynth_simdata");
        let cfg = GsynthConfig {
            cross_validate: true,
            max_factors: 5,
            ..tight(GsynthForce::TwoWay, 0)
        };
        let r = sim(&ds, &["X1", "X2"], cfg.clone()).unwrap();
        let rs: Vec<usize> = r.cv_mspe.iter().map(|(r, _)| *r).collect();
        assert_eq!(rs, vec![0, 1, 2, 3, 4, 5]);
        let mspe: Vec<f64> = r.cv_mspe.iter().map(|(_, m)| *m).collect();
        let r0 = sim(&ds, &[], cfg).unwrap();
        let mspe0: Vec<f64> = r0.cv_mspe.iter().map(|(_, m)| *m).collect();
        let it = Tol::iterative(1e-8);
        Check::new(&case)
            .vector("ife_loo_cov_mspe", &mspe, it)
            // fect fits the CV models at 1e-3
            .vector("ife_loo_cov_mspe_fect", &mspe, Tol::new(1e-3, 0.0))
            .scalar("ife_loo_cov_r", r.n_factors as f64, Tol::exact())
            .scalar("ife_loo_cov_att_avg", r.att, it)
            .vector("ife_loo_nocov_mspe", &mspe0, Tol::coef(SIM_N))
            .vector("ife_loo_nocov_mspe_fect", &mspe0, Tol::coef(SIM_N))
            .scalar("ife_loo_nocov_r", r0.n_factors as f64, Tol::exact())
            .scalar("ife_loo_nocov_att_avg", r0.att, Tol::coef(SIM_N))
            .finish();
    }

    #[test]
    fn test_validate_gsynth_staggered_cv() {
        let (case, ds) = crate::test_support::load_case("gsynth_staggered");
        let cfg = GsynthConfig {
            cross_validate: true,
            max_factors: 5,
            ..tight(GsynthForce::TwoWay, 0)
        };
        let r = sim(&ds, &["X"], cfg).unwrap();
        let mspe: Vec<f64> = r.cv_mspe.iter().map(|(_, m)| *m).collect();
        Check::new(&case)
            .vector("loo_cov_mspe", &mspe, Tol::iterative(1e-8))
            .vector("loo_cov_mspe_fect", &mspe, Tol::new(1e-3, 0.0))
            .scalar("loo_cov_r", r.n_factors as f64, Tol::exact())
            .finish();
    }

    #[test]
    fn test_gsynth_loo_press_equals_refits() {
        // the PRESS errors e / (1 - h) equal explicit leave-one-out refits
        let ds = simdata();
        let panel = read_gsynth_panel(ds.df(), "Y", "D", "id", "time", &["X1", "X2"]).unwrap();
        let m = gsynth_mats(panel).unwrap();
        let names = vec!["X1".to_string(), "X2".to_string()];
        for r in [0usize, 1, 3] {
            let spec = IfeSpec {
                r,
                force: GsynthForce::TwoWay,
                tolerance: 1e-10,
                max_iter: 1000,
            };
            let (_, parts) = gsynth_estimate_parts(&m, &spec, &names).unwrap();
            let press = loo_mspe(&m, &parts).unwrap().unwrap();
            let (mut sum, mut n) = (0.0, 0usize);
            for (j, &t0) in m.first_treated.iter().enumerate() {
                for s in 0..t0 {
                    let keep: Vec<usize> = (0..t0).filter(|&t| t != s).collect();
                    let fk = select_rows(&parts.f_hat, &keep);
                    let yk = Array1::from_iter(keep.iter().map(|&t| parts.u_tr[[t, j]]));
                    let pred = if fk.ncols() == 0 {
                        0.0
                    } else {
                        let ls = crate::linalg::solve::least_squares(fk.view(), yk.view()).unwrap();
                        parts.f_hat.row(s).dot(&ls.coefficients)
                    };
                    sum += (parts.u_tr[[s, j]] - pred).powi(2);
                    n += 1;
                }
            }
            let refit = sum / n as f64;
            assert!(
                (press - refit).abs() <= 1e-10 * refit,
                "r={r}: {press} vs {refit}"
            );
        }
    }

    #[test]
    fn test_gsynth_options_change_the_estimate() {
        let ds = simdata();
        let cov = sim(&ds, &["X1", "X2"], tight(GsynthForce::TwoWay, 2)).unwrap();
        let tw = sim(&ds, &[], tight(GsynthForce::TwoWay, 2)).unwrap();
        let unit = sim(&ds, &[], tight(GsynthForce::Unit, 2)).unwrap();
        assert!((cov.att - tw.att).abs() > 0.5, "{} {}", cov.att, tw.att);
        assert!((unit.att - tw.att).abs() > 0.1, "{} {}", unit.att, tw.att);
    }

    #[test]
    fn test_gsynth_ife_convergence_is_scale_free() {
        let ds = simdata();
        let cfg = || GsynthConfig {
            tolerance: 1e-8,
            ..tight(GsynthForce::TwoWay, 2)
        };
        let base = sim(&ds, &["X1", "X2"], cfg()).unwrap();
        let k0 = base.convergence.as_ref().unwrap().iterations as i64;
        let y = col(&ds, "Y");
        let big = set_col(&ds, "Y", y.iter().map(|v| v * 1e6).collect());
        let rb = sim(&big, &["X1", "X2"], cfg()).unwrap();
        let kb = rb.convergence.as_ref().unwrap().iterations as i64;
        assert!((kb - k0).abs() <= 1, "{kb} vs {k0}");
        assert!((rb.att / 1e6 - base.att).abs() <= 1e-6 * base.att.abs());
        let mut small = ds.clone();
        for c in ["X1", "X2"] {
            let x = col(&ds, c);
            small = set_col(&small, c, x.iter().map(|v| v * 1e-6).collect());
        }
        let rs = sim(&small, &["X1", "X2"], cfg()).unwrap();
        let ks = rs.convergence.as_ref().unwrap().iterations as i64;
        assert!((ks - k0).abs() <= 1, "{ks} vs {k0}");
        for (a, b) in rs.beta.iter().zip(base.beta.iter()) {
            assert!((a * 1e-6 - b).abs() <= 1e-6 * b.abs(), "{a} {b}");
        }
    }

    #[test]
    fn test_gsynth_reports_non_convergence() {
        let ds = simdata();
        let r = sim(
            &ds,
            &["X1", "X2"],
            GsynthConfig {
                max_iter: 2,
                ..tight(GsynthForce::TwoWay, 2)
            },
        )
        .unwrap();
        let c = r.convergence.as_ref().unwrap();
        assert!(!c.converged && c.iterations == 2);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("did not converge after 2 iterations")),
            "{:?}",
            r.warnings
        );
        assert!(r.to_string().contains("did not converge"));
    }

    #[test]
    fn test_gsynth_rejects_invalid_inputs() {
        let ds = simdata();
        let d = col(&ds, "D");
        let id = col(&ds, "id");
        let time = col(&ds, "time");
        let msg = |r: EconResult<GsynthResult>| match r {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got ok={}", other.is_ok()),
        };
        // a 2 in one treated row
        let first = d.iter().position(|&v| v == 1.0).unwrap();
        let mut d2 = d.clone();
        d2[first] = 2.0;
        let m = msg(sim(
            &set_col(&ds, "D", d2),
            &[],
            tight(GsynthForce::TwoWay, 2),
        ));
        assert!(m.contains("0/1"), "{m}");
        // a treated unit untreated again at its last period
        let unit = id[first];
        let tmax = time.iter().cloned().fold(f64::MIN, f64::max);
        let last = (0..d.len())
            .find(|&i| id[i] == unit && time[i] == tmax)
            .unwrap();
        let mut d3 = d.clone();
        d3[last] = 0.0;
        let m = msg(sim(
            &set_col(&ds, "D", d3),
            &[],
            tight(GsynthForce::TwoWay, 2),
        ));
        assert!(m.contains("absorbing"), "{m}");
        // a collinear covariate
        let x3: Vec<f64> = col(&ds, "X1")
            .iter()
            .zip(col(&ds, "X2"))
            .map(|(a, b)| a + b)
            .collect();
        let m = msg(sim(
            &set_col(&ds, "X3", x3),
            &["X1", "X2", "X3"],
            tight(GsynthForce::TwoWay, 2),
        ));
        assert!(m.contains("X3") && m.contains("collinear"), "{m}");
        // a time-invariant covariate under unit effects
        let m = msg(sim(
            &set_col(&ds, "Z", id.clone()),
            &["Z"],
            tight(GsynthForce::Unit, 2),
        ));
        assert!(m.contains("Z"), "{m}");
        // r_id = min(45 - 1, 30 - 1, 20 - 2) = 18
        let m = msg(sim(&ds, &[], tight(GsynthForce::TwoWay, 19)));
        assert!(m.contains("at most 18"), "{m}");
    }

    #[test]
    fn test_gsynth_cv_caps_r_at_the_identified_bound() {
        let ds = create_gsynth_dataset();
        let r = gs(
            &ds,
            GsynthConfig {
                n_factors: 0,
                cross_validate: true,
                max_factors: 3,
                ..gs_cfg()
            },
        )
        .unwrap();
        assert_eq!(r.cv_mspe.len(), 3);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("max_factors = 3 exceeds")),
            "{:?}",
            r.warnings
        );
    }

    #[test]
    fn test_gsynth_estimator_and_force_parse() {
        assert_eq!(
            "gsynth".parse::<GsynthEstimator>().unwrap(),
            GsynthEstimator::Ife
        );
        for s in ["mc", "Matrix-Completion", "matrixcompletion"] {
            assert_eq!(
                s.parse::<GsynthEstimator>().unwrap(),
                GsynthEstimator::MatrixCompletion
            );
        }
        assert_eq!(GsynthEstimator::MatrixCompletion.to_string(), "mc");
        assert!("em".parse::<GsynthEstimator>().is_err());
        for s in ["two-way", "twoWay", "TWO_WAY"] {
            assert_eq!(s.parse::<GsynthForce>().unwrap(), GsynthForce::TwoWay);
        }
        assert_eq!(
            "individual".parse::<GsynthForce>().unwrap(),
            GsynthForce::Unit
        );
        assert!("both".parse::<GsynthForce>().is_err());
        GsynthEstimator::choices().validate().unwrap();
        GsynthForce::choices().validate().unwrap();
    }

    fn mc(lambda: Vec<f64>) -> GsynthConfig {
        GsynthConfig {
            estimator: GsynthEstimator::MatrixCompletion,
            lambda,
            ..tight(GsynthForce::TwoWay, 0)
        }
    }

    #[test]
    fn test_validate_gsynth_mc() {
        let (case, ds) = crate::test_support::load_case("gsynth_simdata");
        let r = sim(
            &ds,
            &["X1", "X2"],
            GsynthConfig {
                max_iter: 100_000,
                ..mc(vec![0.01])
            },
        )
        .unwrap();
        let r3 = sim(
            &ds,
            &["X1", "X2"],
            GsynthConfig {
                max_iter: 100_000,
                ..mc(vec![0.3])
            },
        )
        .unwrap();
        let mut chk = Check::new(&case);
        let i7 = Tol::iterative(1e-7);
        check_fit(&mut chk, "mc_l001_cov", &r, i7, i7);
        chk.scalar("mc_l001_cov_rank", r.n_factors as f64, Tol::exact());
        let i8 = Tol::iterative(1e-8);
        check_fit(&mut chk, "mc_l03_cov", &r3, i8, i8);
        chk.finish();
        assert_eq!(r.lambda, Some(0.01));
        assert!(r.factors.is_none() && r.loadings.is_none());
        assert_eq!(r3.n_factors, 0);
        assert!(r.convergence.as_ref().unwrap().converged);
    }

    #[test]
    fn test_validate_gsynth_mc_lambda_grid() {
        let (case, ds) = crate::test_support::load_case("gsynth_simdata");
        let panel = read_gsynth_panel(ds.df(), "Y", "D", "id", "time", &["X1", "X2"]).unwrap();
        let m = gsynth_mats(panel).unwrap();
        let names = vec!["X1".to_string(), "X2".to_string()];
        let prep = mc_prep(&m, GsynthForce::TwoWay, &names).unwrap();
        let start = mc_start(&m, &prep, &mc_observed(&m), GsynthForce::TwoWay, None).unwrap();
        let (grid, d_max) = mc_lambda_grid(&m, &prep, &start, 10).unwrap();
        Check::new(&case)
            .vector("mc_grid", &grid, Tol::coef(SIM_N))
            .scalar("mc_eigen_max", d_max, Tol::coef(SIM_N))
            .finish();
    }

    #[test]
    fn test_validate_gsynth_mc_cv() {
        let (case, ds) = crate::test_support::load_case("gsynth_simdata");
        let grid = case.vector("mc_cv_user_grid");
        let r = sim(
            &ds,
            &["X1", "X2"],
            GsynthConfig {
                cross_validate: true,
                cv_folds: 5,
                tolerance: 1e-8,
                max_iter: 1000,
                ..mc(grid.clone())
            },
        )
        .unwrap();
        let lambdas: Vec<f64> = r.cv_lambda.iter().map(|(l, _)| *l).collect();
        assert_eq!(lambdas, grid);
        let mspe: Vec<f64> = r.cv_lambda.iter().map(|(_, v)| *v).collect();
        // the tolerance-1e-8 fits are this far from the solution (mc_cv_user_gap = 1.15e-6)
        let gap = case.scalar("mc_cv_user_gap");
        let tol = Tol::iterative(1e-6_f64.max(10.0 * gap));
        Check::new(&case)
            .vector("mc_cv_user_mspe", &mspe, tol)
            .scalar("mc_cv_user_lambda", r.lambda.unwrap(), Tol::exact())
            .scalar(
                "mc_cv_user_att_avg",
                r.att,
                Tol::iterative(1e-5_f64.max(10.0 * gap)),
            )
            .finish();
    }

    #[test]
    fn test_gsynth_mc_requires_lambda_without_cv() {
        let ds = simdata();
        let msg = |c: GsynthConfig| match sim(&ds, &[], c) {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got ok={}", other.is_ok()),
        };
        for l in [vec![], vec![-0.1], vec![f64::NAN], vec![0.1, 0.2]] {
            let m = msg(mc(l.clone()));
            assert!(m.contains("lambda"), "{l:?}: {m}");
        }
        let m = msg(GsynthConfig {
            cross_validate: true,
            n_lambda: 2,
            ..mc(vec![])
        });
        assert!(m.contains("n_lambda"), "{m}");
    }

    #[test]
    fn test_gsynth_mc_convergence_is_scale_free() {
        let ds = simdata();
        let cfg = |l: f64| GsynthConfig {
            tolerance: 1e-8,
            max_iter: 100_000,
            ..mc(vec![l])
        };
        let base = sim(&ds, &["X1", "X2"], cfg(0.01)).unwrap();
        let mut big = ds.clone();
        for c in ["Y", "X1", "X2"] {
            let v = col(&ds, c);
            big = set_col(&big, c, v.iter().map(|x| x * 1e6).collect());
        }
        let rb = sim(&big, &["X1", "X2"], cfg(0.01 * 1e6)).unwrap();
        let (k0, kb) = (
            base.convergence.as_ref().unwrap().iterations as i64,
            rb.convergence.as_ref().unwrap().iterations as i64,
        );
        assert!((kb - k0).abs() <= 1, "{kb} vs {k0}");
        assert!((rb.att / 1e6 - base.att).abs() <= 1e-6 * base.att.abs());
    }
}
