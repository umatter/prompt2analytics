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

use ndarray::{Array1, Array2};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
    /// Final loss value from optimization
    pub final_loss: f64,
    /// Any warnings generated
    pub warnings: Vec<String>,
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

        // Predictor balance
        writeln!(f, "PREDICTOR BALANCE")?;
        writeln!(f, "{:<25} {:>12} {:>12} {:>12}", "Predictor", "Treated", "Synthetic", "Diff %")?;
        writeln!(f, "{}", "-".repeat(63))?;
        for pb in &self.predictor_balance {
            writeln!(f, "{:<25} {:>12.4} {:>12.4} {:>12.2}%",
                     pb.predictor, pb.treated_value, pb.synthetic_value, pb.percent_diff)?;
        }
        writeln!(f)?;

        // Fit statistics
        writeln!(f, "PRE-TREATMENT FIT")?;
        writeln!(f, "  MSPE:  {:.6}", self.pre_treatment_mspe)?;
        writeln!(f, "  RMSPE: {:.6}", self.pre_treatment_rmspe)?;
        writeln!(f)?;

        // Treatment effects
        writeln!(f, "TREATMENT EFFECTS")?;
        writeln!(f, "  Average Effect: {:.4}", self.average_effect)?;
        writeln!(f, "  Cumulative Effect: {:.4}", self.cumulative_effect)?;
        writeln!(f)?;

        writeln!(f, "{:<10} {:>12} {:>12} {:>12}", "Time", "Actual", "Synthetic", "Effect")?;
        writeln!(f, "{}", "-".repeat(50))?;
        for te in &self.treatment_effects {
            writeln!(f, "{:<10} {:>12.4} {:>12.4} {:>12.4}",
                     te.time, te.actual, te.synthetic, te.effect)?;
        }

        // Placebo inference
        if let Some(ref placebo) = self.placebo_results {
            writeln!(f)?;
            writeln!(f, "PLACEBO INFERENCE")?;
            writeln!(f, "  Treated Unit Rank: {} / {}", placebo.treated_rank, placebo.n_units)?;
            writeln!(f, "  Exact P-Value: {:.4}{}", placebo.p_value, placebo.significance.stars())?;
        }

        if !self.warnings.is_empty() {
            writeln!(f)?;
            writeln!(f, "Warnings:")?;
            for w in &self.warnings {
                writeln!(f, "  - {}", w)?;
            }
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
    let (v_weights, w_weights, v_iterations, final_loss) = optimize_synth_weights(
        &synth_data,
        &config.v_method,
        config.tolerance,
        config.max_iter,
    )?;

    // Check for convergence issues
    if v_iterations >= config.max_iter {
        warnings.push(format!(
            "V optimization reached maximum iterations ({}); may not have converged",
            config.max_iter
        ));
    }

    // Calculate pre-treatment fit
    let (pre_mspe, pre_rmspe) = calculate_pre_treatment_fit(&synth_data.z1, &synth_data.z0, &w_weights);

    // Calculate predictor balance
    let predictor_balance = calculate_predictor_balance(
        &synth_data.x1,
        &synth_data.x0,
        &w_weights,
        &synth_data.predictor_names,
    );

    // Calculate treatment effects
    let treatment_effects = calculate_treatment_effects(&synth_data, &w_weights);

    let average_effect = if !treatment_effects.is_empty() {
        treatment_effects.iter().map(|te| te.effect).sum::<f64>() / treatment_effects.len() as f64
    } else {
        0.0
    };

    let cumulative_effect: f64 = treatment_effects.iter().map(|te| te.effect).sum();

    // Build time series data
    let (actual_outcome, synthetic_outcome) = build_time_series(&synth_data, &w_weights);

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
            dataset,
            outcome,
            unit_col,
            time_col,
            predictors,
            &config,
            pre_rmspe,
        ) {
            Ok(results) => Some(results),
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
        v_iterations,
        final_loss,
        warnings,
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

    // Get unique units and times
    let units: Vec<String> = df
        .column(unit_col)
        .map_err(|e| EconError::ColumnNotFound {
            column: unit_col.to_string(),
            available: vec![format!("{:?}", e)],
        })?
        .str()
        .map_err(|_| EconError::InvalidSpecification { message: format!("Unit column '{}' must be string type", unit_col) })?
        .into_iter()
        .filter_map(|s: Option<&str>| s.map(|s| s.to_string()))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let times: Vec<i64> = df
        .column(time_col)
        .map_err(|e| EconError::ColumnNotFound {
            column: time_col.to_string(),
            available: vec![format!("{:?}", e)],
        })?
        .i64()
        .map_err(|_| EconError::InvalidSpecification { message: format!("Time column '{}' must be integer type", time_col) })?
        .into_iter()
        .filter_map(|t| t)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let mut times = times;
    times.sort();

    // Validate treated unit exists
    if !units.contains(&treated_unit.to_string()) {
        return Err(EconError::InvalidSpecification { message: format!(
            "Treated unit '{}' not found in data. Available units: {:?}",
            treated_unit,
            units.iter().take(10).collect::<Vec<_>>()
        ) });
    }

    // Split into pre and post treatment periods
    let pre_times: Vec<i64> = times.iter().copied().filter(|&t| t < treatment_time).collect();
    let post_times: Vec<i64> = times.iter().copied().filter(|&t| t >= treatment_time).collect();

    if pre_times.is_empty() {
        return Err(EconError::InvalidSpecification { message: format!(
            "No pre-treatment periods found. Treatment time {} but earliest data is {}",
            treatment_time,
            times.first().unwrap_or(&0)
        ) });
    }

    // Get donor units (all except treated)
    let donor_units: Vec<String> = units.into_iter().filter(|u| u != treated_unit).collect();

    // Determine optimization window
    let opt_times: Vec<i64> = match optimization_window {
        Some((start, end)) => pre_times.iter().copied().filter(|&t| t >= start && t <= end).collect(),
        None => pre_times.clone(),
    };

    if opt_times.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "Optimization window contains no pre-treatment periods".to_string(),
        });
    }

    // Build predictor matrices
    let k = predictors.len();
    let j = donor_units.len();

    let mut x1 = Array1::zeros(k);
    let mut x0 = Array2::zeros((k, j));
    let mut predictor_names = Vec::with_capacity(k);

    for (p_idx, pred_spec) in predictors.iter().enumerate() {
        let pred_times = match pred_spec.time_window {
            Some((start, end)) => pre_times.iter().copied().filter(|&t| t >= start && t <= end).collect(),
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

        // Get treated unit predictor value
        x1[p_idx] = aggregate_predictor(df, &pred_spec.column, unit_col, time_col, treated_unit, &pred_times, &pred_spec.aggregation)?;

        // Get donor unit predictor values
        for (d_idx, donor) in donor_units.iter().enumerate() {
            x0[[p_idx, d_idx]] = aggregate_predictor(df, &pred_spec.column, unit_col, time_col, donor, &pred_times, &pred_spec.aggregation)?;
        }
    }

    // Build outcome matrices
    let t0 = opt_times.len();
    let t_post = post_times.len();

    let mut z1 = Array1::zeros(t0);
    let mut z0 = Array2::zeros((t0, j));
    let mut y1_post = Array1::zeros(t_post);
    let mut y0_post = Array2::zeros((t_post, j));

    // Pre-treatment outcomes
    for (t_idx, &time) in opt_times.iter().enumerate() {
        z1[t_idx] = get_outcome(df, outcome, unit_col, time_col, treated_unit, time)?;

        for (d_idx, donor) in donor_units.iter().enumerate() {
            z0[[t_idx, d_idx]] = get_outcome(df, outcome, unit_col, time_col, donor, time)?;
        }
    }

    // Post-treatment outcomes
    for (t_idx, &time) in post_times.iter().enumerate() {
        y1_post[t_idx] = get_outcome(df, outcome, unit_col, time_col, treated_unit, time)?;

        for (d_idx, donor) in donor_units.iter().enumerate() {
            y0_post[[t_idx, d_idx]] = get_outcome(df, outcome, unit_col, time_col, donor, time)?;
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

/// Aggregate predictor values over time for a specific unit.
fn aggregate_predictor(
    df: &DataFrame,
    column: &str,
    unit_col: &str,
    time_col: &str,
    unit: &str,
    times: &[i64],
    aggregation: &TimeAggregation,
) -> EconResult<f64> {
    let mut values = Vec::new();

    for &time in times {
        if let Ok(val) = get_outcome(df, column, unit_col, time_col, unit, time) {
            if val.is_finite() {
                values.push(val);
            }
        }
    }

    if values.is_empty() {
        return Err(EconError::InvalidSpecification { message: format!(
            "No valid values for predictor '{}' for unit '{}' in specified time window",
            column, unit
        ) });
    }

    let result = match aggregation {
        TimeAggregation::Mean => values.iter().sum::<f64>() / values.len() as f64,
        TimeAggregation::First => values[0],
        TimeAggregation::Last => *values.last().unwrap(),
        TimeAggregation::Sum => values.iter().sum(),
    };

    Ok(result)
}

/// Get outcome value for a specific unit at a specific time.
fn get_outcome(
    df: &DataFrame,
    outcome: &str,
    unit_col: &str,
    time_col: &str,
    unit: &str,
    time: i64,
) -> EconResult<f64> {
    // Filter to the specific unit and time
    let mask = df
        .column(unit_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: unit_col.to_string(),
            available: vec![],
        })?
        .str()
        .map_err(|_| EconError::InvalidSpecification { message: "Unit column must be string".to_string() })?
        .equal(unit);

    let time_mask = df
        .column(time_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: time_col.to_string(),
            available: vec![],
        })?
        .i64()
        .map_err(|_| EconError::InvalidSpecification { message: "Time column must be integer".to_string() })?
        .equal(time);

    let combined_mask = &mask & &time_mask;

    let filtered = df.filter(&combined_mask).map_err(|e| {
        EconError::InvalidSpecification { message: format!("Filter error: {:?}", e) }
    })?;

    if filtered.height() == 0 {
        return Err(EconError::InvalidSpecification { message: format!(
            "No data for unit '{}' at time {}",
            unit, time
        ) });
    }

    let val = filtered
        .column(outcome)
        .map_err(|_| EconError::ColumnNotFound {
            column: outcome.to_string(),
            available: vec![],
        })?
        .f64()
        .map_err(|_| EconError::InvalidSpecification { message: format!("Outcome '{}' must be numeric", outcome) })?
        .get(0)
        .ok_or_else(|| EconError::InvalidSpecification { message: "Missing outcome value".to_string() })?;

    Ok(val)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Optimization
// ═══════════════════════════════════════════════════════════════════════════════

/// Optimize V (predictor weights) and W (unit weights).
///
/// Uses nested optimization:
/// - Outer: Optimize V to minimize pre-treatment MSPE
/// - Inner: For given V, solve constrained QP for W
fn optimize_synth_weights(
    data: &SynthData,
    v_method: &VOptimization,
    tolerance: f64,
    max_iter: usize,
) -> EconResult<(Array1<f64>, Array1<f64>, usize, f64)> {
    let k = data.x1.len();

    // Initialize V weights
    let v = match v_method {
        VOptimization::Equal => Array1::from_elem(k, 1.0 / k as f64),
        VOptimization::Custom(weights) => {
            if weights.len() != k {
                return Err(EconError::InvalidSpecification { message: format!(
                    "Custom V weights length ({}) doesn't match number of predictors ({})",
                    weights.len(),
                    k
                ) });
            }
            let sum: f64 = weights.iter().sum();
            Array1::from_vec(weights.iter().map(|&w| w / sum).collect())
        }
        VOptimization::DataDriven => {
            // Start with equal weights, then optimize
            Array1::from_elem(k, 1.0 / k as f64)
        }
    };

    // For non-data-driven methods, just solve for W once
    if !matches!(v_method, VOptimization::DataDriven) {
        let w = solve_weights_qp(&data.x0, &data.x1, &v)?;
        let loss = calculate_v_loss(data, &w);
        return Ok((v, w, 0, loss));
    }

    // Data-driven V optimization using coordinate descent / Nelder-Mead style
    let mut best_v = v.clone();
    let mut best_w = solve_weights_qp(&data.x0, &data.x1, &v)?;
    let mut best_loss = calculate_v_loss(data, &best_w);

    let mut iterations = 0;

    // Simple coordinate descent for V optimization
    for iter in 0..max_iter {
        iterations = iter + 1;
        let mut improved = false;

        for i in 0..k {
            // Try increasing and decreasing this V component
            for delta in &[0.1, -0.1, 0.05, -0.05, 0.01, -0.01] {
                let mut v_new = best_v.clone();
                v_new[i] = (v_new[i] + delta).max(0.001);

                // Normalize
                let sum: f64 = v_new.sum();
                v_new.mapv_inplace(|x| x / sum);

                // Solve for W with new V
                if let Ok(w_new) = solve_weights_qp(&data.x0, &data.x1, &v_new) {
                    let loss_new = calculate_v_loss(data, &w_new);

                    if loss_new < best_loss - tolerance {
                        best_v = v_new;
                        best_w = w_new;
                        best_loss = loss_new;
                        improved = true;
                    }
                }
            }
        }

        if !improved {
            break;
        }
    }

    Ok((best_v, best_w, iterations, best_loss))
}

/// Calculate the V-loss (pre-treatment MSPE) for given W weights.
///
/// This is the objective function for V optimization.
/// Loss = (1/T₀) Σₜ (Z₁ₜ - Z₀ₜ W)²
fn calculate_v_loss(data: &SynthData, w: &Array1<f64>) -> f64 {
    let t0 = data.z1.len();
    let mut sse = 0.0;

    for t in 0..t0 {
        let z1_t = data.z1[t];
        let synthetic_t: f64 = (0..data.donor_units.len())
            .map(|j| w[j] * data.z0[[t, j]])
            .sum();
        let error = z1_t - synthetic_t;
        sse += error * error;
    }

    sse / t0 as f64
}

/// Solve for unit weights W using quadratic programming.
///
/// Minimizes: (X₁ - X₀W)' V (X₁ - X₀W)
/// Subject to: Σwⱼ = 1, wⱼ ≥ 0
///
/// This is converted to standard QP form:
/// Minimize: (1/2) W' H W + c' W
/// Where H = X₀' V X₀ and c = -X₀' V X₁
fn solve_weights_qp(
    x0: &Array2<f64>,  // k × J
    x1: &Array1<f64>,  // k × 1
    v: &Array1<f64>,   // k × 1 (diagonal of V)
) -> EconResult<Array1<f64>> {
    let k = x1.len();
    let j = x0.ncols();

    if j == 0 {
        return Err(EconError::InvalidSpecification { message: "No donor units".to_string() });
    }

    // Build V as diagonal matrix (we work with diagonal directly for efficiency)
    // H = X₀' V X₀ (J × J)
    // c = -X₀' V X₁ (J × 1)

    let mut h = Array2::zeros((j, j));
    let mut c = Array1::zeros(j);

    for i in 0..j {
        for l in 0..j {
            let mut sum = 0.0;
            for m in 0..k {
                sum += x0[[m, i]] * v[m] * x0[[m, l]];
            }
            h[[i, l]] = sum;
        }

        let mut sum_c = 0.0;
        for m in 0..k {
            sum_c += x0[[m, i]] * v[m] * x1[m];
        }
        c[i] = -sum_c;
    }

    // Add small regularization to ensure positive definiteness
    for i in 0..j {
        h[[i, i]] += 1e-8;
    }

    // Solve QP using our custom implementation
    // (The quadprog crate requires specific input format)
    solve_simplex_constrained_qp(&h, &c, j)
}

/// Solve constrained QP with simplex constraints (Σw = 1, w ≥ 0).
///
/// Uses active set method / projected gradient descent.
fn solve_simplex_constrained_qp(
    h: &Array2<f64>,
    c: &Array1<f64>,
    n: usize,
) -> EconResult<Array1<f64>> {
    // Initialize with uniform weights
    let mut w = Array1::from_elem(n, 1.0 / n as f64);

    let max_iter = 10000;
    let tolerance = 1e-10;

    // Projected gradient descent with line search
    for _ in 0..max_iter {
        // Gradient: H * w + c
        let grad = h.dot(&w) + c;

        // Compute step direction (negative gradient projected onto simplex)
        // For simplex, this means we need to project onto Σw = 1, w ≥ 0

        // Frank-Wolfe style: find vertex minimizing gradient
        let mut min_idx = 0;
        let mut min_val = grad[0];
        for i in 1..n {
            if grad[i] < min_val {
                min_val = grad[i];
                min_idx = i;
            }
        }

        // Direction: e_min - w (move toward best vertex)
        let mut direction = Array1::zeros(n);
        direction[min_idx] = 1.0;
        let direction = &direction - &w;

        // Line search: find optimal step size
        // f(w + α*d) = 0.5*(w+αd)'H(w+αd) + c'(w+αd)
        // df/dα = d'Hw + α*d'Hd + c'd = 0
        // α* = -(d'Hw + c'd) / (d'Hd)

        let hd = h.dot(&direction);
        let d_h_d: f64 = direction.iter().zip(hd.iter()).map(|(&di, &hi)| di * hi).sum();
        let d_h_w: f64 = direction.iter().zip(h.dot(&w).iter()).map(|(&di, &hi)| di * hi).sum();
        let c_d: f64 = c.iter().zip(direction.iter()).map(|(&ci, &di)| ci * di).sum();

        let alpha = if d_h_d.abs() > 1e-12 {
            (-(d_h_w + c_d) / d_h_d).max(0.0).min(1.0)
        } else {
            0.0
        };

        // Update
        let w_new = &w + &(&direction * alpha);

        // Check convergence
        let change: f64 = w_new.iter().zip(w.iter()).map(|(&a, &b)| (a - b).abs()).sum();
        w = w_new;

        if change < tolerance {
            break;
        }
    }

    // Project to ensure constraints are satisfied
    w.mapv_inplace(|x| x.max(0.0));
    let sum: f64 = w.sum();
    if sum > 0.0 {
        w.mapv_inplace(|x| x / sum);
    }

    Ok(w)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Results Calculation
// ═══════════════════════════════════════════════════════════════════════════════

/// Calculate pre-treatment fit statistics.
fn calculate_pre_treatment_fit(
    z1: &Array1<f64>,
    z0: &Array2<f64>,
    w: &Array1<f64>,
) -> (f64, f64) {
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

/// Calculate predictor balance between treated and synthetic.
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
            predictor: predictor_names.get(i).cloned().unwrap_or_else(|| format!("X{}", i)),
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
    // Get all units
    let df = dataset.df();
    let units: Vec<String> = df
        .column(unit_col)
        .map_err(|e| EconError::ColumnNotFound {
            column: unit_col.to_string(),
            available: vec![format!("{:?}", e)],
        })?
        .str()
        .map_err(|_| EconError::InvalidSpecification { message: "Unit column must be string".to_string() })?
        .into_iter()
        .filter_map(|s: Option<&str>| s.map(|s| s.to_string()))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

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

    let treated_rmspe_post = if !treated_result.treatment_effects.is_empty() {
        let mse: f64 = treated_result
            .treatment_effects
            .iter()
            .map(|te| te.effect * te.effect)
            .sum::<f64>()
            / treated_result.treatment_effects.len() as f64;
        mse.sqrt()
    } else {
        0.0
    };

    let treated_ratio = if treated_rmspe_pre > 1e-10 {
        treated_rmspe_post / treated_rmspe_pre
    } else {
        f64::INFINITY
    };

    // Run placebo for each donor unit
    let mut rmspe_ratios: Vec<(String, f64)> = vec![(config.treated_unit.clone(), treated_ratio)];

    for donor_unit in &units {
        if donor_unit == &config.treated_unit {
            continue;
        }

        // Run synth with this donor as treated
        let placebo_config = SynthConfig {
            treated_unit: donor_unit.clone(),
            run_placebos: false,
            ..config.clone()
        };

        if let Ok(placebo_result) = run_synthetic_control(
            dataset,
            outcome,
            unit_col,
            time_col,
            predictors,
            placebo_config,
        ) {
            let rmspe_pre = placebo_result.pre_treatment_rmspe;
            let rmspe_post = if !placebo_result.treatment_effects.is_empty() {
                let mse: f64 = placebo_result
                    .treatment_effects
                    .iter()
                    .map(|te| te.effect * te.effect)
                    .sum::<f64>()
                    / placebo_result.treatment_effects.len() as f64;
                mse.sqrt()
            } else {
                0.0
            };

            let ratio = if rmspe_pre > 1e-10 {
                rmspe_post / rmspe_pre
            } else {
                f64::INFINITY
            };

            rmspe_ratios.push((donor_unit.clone(), ratio));
        }
    }

    // Sort by ratio (descending)
    rmspe_ratios.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

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
        }.unwrap();

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

        let result = run_synthetic_control(&dataset, "outcome", "unit", "time", &predictors, config)
            .unwrap();

        // Check basic structure
        assert_eq!(result.treated_unit, "A");
        assert_eq!(result.treatment_time, 4);
        assert_eq!(result.n_donors, 2);
        assert_eq!(result.n_pre_periods, 3); // times 1, 2, 3
        assert_eq!(result.n_post_periods, 3); // times 4, 5, 6

        // Weights should sum to 1
        let weight_sum: f64 = result.all_unit_weights.iter().map(|(_, w)| w).sum();
        assert!((weight_sum - 1.0).abs() < 0.01, "Weights should sum to 1, got {}", weight_sum);

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

        let result = run_synthetic_control(&dataset, "outcome", "unit", "time", &predictors, config)
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

        let result = run_synthetic_control(&dataset, "outcome", "unit", "time", &predictors, config)
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

        let result = run_synthetic_control(&dataset, "outcome", "unit", "time", &predictors, config);
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

        let result = run_synthetic_control(&dataset, "outcome", "unit", "time", &predictors, config);
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
    fn test_qp_solver() {
        // Simple test: minimize ||x||² subject to Σx = 1, x ≥ 0
        // Solution should be uniform: x = [0.5, 0.5]
        let h = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).unwrap();
        let c = Array1::from(vec![0.0, 0.0]);

        let w = solve_simplex_constrained_qp(&h, &c, 2).unwrap();

        assert!((w[0] - 0.5).abs() < 0.01);
        assert!((w[1] - 0.5).abs() < 0.01);
        assert!((w.sum() - 1.0).abs() < 0.01);
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

        let result = run_synthetic_control(&dataset, "outcome", "unit", "time", &predictors, config)
            .unwrap();

        // Test Display trait
        let output = format!("{}", result);
        assert!(output.contains("Synthetic Control"));
        assert!(output.contains("Treated Unit: A"));
        assert!(output.contains("PREDICTOR BALANCE"));
    }
}
