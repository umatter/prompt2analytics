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
use rayon::prelude::*;
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
    let (pre_mspe, pre_rmspe) =
        calculate_pre_treatment_fit(&synth_data.z1, &synth_data.z0, &w_weights);

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
            dataset, outcome, unit_col, time_col, predictors, &config, pre_rmspe,
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
        .map_err(|_| EconError::InvalidSpecification {
            message: format!("Unit column '{}' must be string type", unit_col),
        })?
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
        .map_err(|_| EconError::InvalidSpecification {
            message: format!("Time column '{}' must be integer type", time_col),
        })?
        .into_iter()
        .flatten()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let mut times = times;
    times.sort();

    // Validate treated unit exists
    if !units.contains(&treated_unit.to_string()) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treated unit '{}' not found in data. Available units: {:?}",
                treated_unit,
                units.iter().take(10).collect::<Vec<_>>()
            ),
        });
    }

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

    // Get donor units (all except treated)
    let donor_units: Vec<String> = units.into_iter().filter(|u| u != treated_unit).collect();

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

    // Build predictor matrices
    let k = predictors.len();
    let j = donor_units.len();

    let mut x1 = Array1::zeros(k);
    let mut x0 = Array2::zeros((k, j));
    let mut predictor_names = Vec::with_capacity(k);

    for (p_idx, pred_spec) in predictors.iter().enumerate() {
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

        // Get treated unit predictor value
        x1[p_idx] = aggregate_predictor(
            df,
            &pred_spec.column,
            unit_col,
            time_col,
            treated_unit,
            &pred_times,
            &pred_spec.aggregation,
        )?;

        // Get donor unit predictor values
        for (d_idx, donor) in donor_units.iter().enumerate() {
            x0[[p_idx, d_idx]] = aggregate_predictor(
                df,
                &pred_spec.column,
                unit_col,
                time_col,
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
        return Err(EconError::InvalidSpecification {
            message: format!(
                "No valid values for predictor '{}' for unit '{}' in specified time window",
                column, unit
            ),
        });
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
        .map_err(|_| EconError::InvalidSpecification {
            message: "Unit column must be string".to_string(),
        })?
        .equal(unit);

    let time_mask = df
        .column(time_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: time_col.to_string(),
            available: vec![],
        })?
        .i64()
        .map_err(|_| EconError::InvalidSpecification {
            message: "Time column must be integer".to_string(),
        })?
        .equal(time);

    let combined_mask = &mask & &time_mask;

    let filtered = df
        .filter(&combined_mask)
        .map_err(|e| EconError::InvalidSpecification {
            message: format!("Filter error: {:?}", e),
        })?;

    if filtered.height() == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!("No data for unit '{}' at time {}", unit, time),
        });
    }

    let val = filtered
        .column(outcome)
        .map_err(|_| EconError::ColumnNotFound {
            column: outcome.to_string(),
            available: vec![],
        })?
        .f64()
        .map_err(|_| EconError::InvalidSpecification {
            message: format!("Outcome '{}' must be numeric", outcome),
        })?
        .get(0)
        .ok_or_else(|| EconError::InvalidSpecification {
            message: "Missing outcome value".to_string(),
        })?;

    Ok(val)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Optimization
// ═══════════════════════════════════════════════════════════════════════════════

/// Precomputed data for H matrix computation (thread-safe, read-only after creation).
///
/// Precomputes structures that allow O(j²) H matrix construction instead of O(j²k).
/// Since X0 doesn't change during V optimization, we can precompute:
/// - outer_products[m] = X0[m,:]' * X0[m,:] for each predictor m
/// - x0_x1[m][i] = X0[m,i] * X1[m] for the c vector
struct QpPrecomputed {
    /// Outer products per predictor: outer_products[m][i,l] = X0[m,i] * X0[m,l]
    outer_products: Vec<Array2<f64>>,
    /// X0 * X1 products per predictor: x0_x1[m][i] = X0[m,i] * X1[m]
    x0_x1_products: Vec<Array1<f64>>,
    /// Number of donors
    j: usize,
    /// Number of predictors
    k: usize,
}

impl QpPrecomputed {
    /// Create precomputed data from X0 and X1.
    fn new(x0: &Array2<f64>, x1: &Array1<f64>) -> Self {
        let k = x0.nrows();
        let j = x0.ncols();

        // Precompute outer products for each predictor dimension
        let mut outer_products = Vec::with_capacity(k);
        let mut x0_x1_products = Vec::with_capacity(k);

        for m in 0..k {
            // outer_products[m][i,l] = X0[m,i] * X0[m,l]
            let mut outer = Array2::zeros((j, j));
            let mut x0_x1 = Array1::zeros(j);

            for i in 0..j {
                let x0_mi = x0[[m, i]];
                // Exploit symmetry: only compute upper triangle
                for l in i..j {
                    let val = x0_mi * x0[[m, l]];
                    outer[[i, l]] = val;
                    outer[[l, i]] = val;
                }
                x0_x1[i] = x0_mi * x1[m];
            }

            outer_products.push(outer);
            x0_x1_products.push(x0_x1);
        }

        QpPrecomputed {
            outer_products,
            x0_x1_products,
            j,
            k,
        }
    }
}

/// Working buffers for QP solving (one per thread).
struct QpWorkspace {
    h_buffer: Array2<f64>,
    c_buffer: Array1<f64>,
}

impl QpWorkspace {
    fn new(j: usize) -> Self {
        QpWorkspace {
            h_buffer: Array2::zeros((j, j)),
            c_buffer: Array1::zeros(j),
        }
    }

    /// Build H and c matrices using precomputed data and given V weights.
    fn build_h_and_c<'a>(
        &'a mut self,
        precomputed: &QpPrecomputed,
        v: &Array1<f64>,
    ) -> (&'a Array2<f64>, &'a Array1<f64>) {
        let j = precomputed.j;

        // Reset buffers
        self.h_buffer.fill(0.0);
        self.c_buffer.fill(0.0);

        // H = Σ_m V[m] * outer_products[m]
        // c = -Σ_m V[m] * x0_x1[m]
        for m in 0..precomputed.k {
            let vm = v[m];
            if vm.abs() < 1e-12 {
                continue; // Skip near-zero weights
            }

            // Add weighted outer product to H
            let outer = &precomputed.outer_products[m];
            for i in 0..j {
                for l in 0..j {
                    self.h_buffer[[i, l]] += vm * outer[[i, l]];
                }
            }

            // Add weighted x0_x1 to c
            let x0_x1 = &precomputed.x0_x1_products[m];
            for i in 0..j {
                self.c_buffer[i] -= vm * x0_x1[i];
            }
        }

        // Add regularization for numerical stability
        for i in 0..j {
            self.h_buffer[[i, i]] += 1e-8;
        }

        (&self.h_buffer, &self.c_buffer)
    }

    /// Solve for unit weights W using precomputed data.
    fn solve(&mut self, precomputed: &QpPrecomputed, v: &Array1<f64>) -> EconResult<Array1<f64>> {
        let (h, c) = self.build_h_and_c(precomputed, v);
        solve_simplex_constrained_qp(h, c, precomputed.j)
    }
}

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
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Custom V weights length ({}) doesn't match number of predictors ({})",
                        weights.len(),
                        k
                    ),
                });
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

    // Create precomputed QP data for efficient repeated solves
    let precomputed = QpPrecomputed::new(&data.x0, &data.x1);
    let j = precomputed.j;

    // Initial solve with starting V
    let mut workspace = QpWorkspace::new(j);
    let mut best_v = v.clone();
    let mut best_w = workspace.solve(&precomputed, &v)?;
    let mut best_loss = calculate_v_loss(data, &best_w);

    let mut iterations = 0;

    // Track loss at start of each outer iteration for convergence check
    let mut loss_at_iter_start = best_loss;
    let convergence_threshold = tolerance * 10.0; // Relative improvement threshold

    // Parallel coordinate descent for V optimization
    // Generate all (dimension, delta) candidates to evaluate in parallel
    let deltas: [f64; 6] = [0.1, -0.1, 0.05, -0.05, 0.01, -0.01];

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Evaluate all coordinate directions in parallel
        // Each thread gets its own workspace but shares the precomputed data
        let candidates: Vec<(usize, f64)> = (0..k)
            .flat_map(|i| deltas.iter().map(move |&d| (i, d)))
            .collect();

        // Parallel evaluation of all candidates
        let results: Vec<(f64, Array1<f64>, Array1<f64>)> = candidates
            .par_iter()
            .filter_map(|&(dim, delta)| {
                // Build candidate V vector
                let mut v_new = best_v.clone();
                v_new[dim] = (v_new[dim] + delta).max(0.001);

                // Normalize
                let sum: f64 = v_new.sum();
                if sum <= 0.0 {
                    return None;
                }
                v_new.mapv_inplace(|x| x / sum);

                // Each thread creates its own workspace
                let mut local_workspace = QpWorkspace::new(j);

                // Solve QP for this candidate
                let w_new = local_workspace.solve(&precomputed, &v_new).ok()?;
                let loss_new = calculate_v_loss(data, &w_new);

                Some((loss_new, v_new, w_new))
            })
            .collect();

        // Find the best result among all candidates
        let mut improved = false;
        for (loss_new, v_new, w_new) in results {
            if loss_new < best_loss - tolerance {
                best_v = v_new;
                best_w = w_new;
                best_loss = loss_new;
                improved = true;
            }
        }

        // Check for convergence based on relative improvement this iteration
        if iter > 5 {
            let relative_improvement =
                (loss_at_iter_start - best_loss) / loss_at_iter_start.max(1e-10);
            if relative_improvement < convergence_threshold && relative_improvement >= 0.0 {
                break; // Converged - improvements too small to continue
            }
        }
        loss_at_iter_start = best_loss;

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

/// Solve for unit weights W using quadratic programming.
///
/// Minimizes: (X₁ - X₀W)' V (X₁ - X₀W)
/// Subject to: Σwⱼ = 1, wⱼ ≥ 0
///
/// This is converted to standard QP form:
/// Minimize: (1/2) W' H W + c' W
/// Where H = X₀' V X₀ and c = -X₀' V X₁
fn solve_weights_qp(
    x0: &Array2<f64>, // k × J
    x1: &Array1<f64>, // k × 1
    v: &Array1<f64>,  // k × 1 (diagonal of V)
) -> EconResult<Array1<f64>> {
    let k = x1.len();
    let j = x0.ncols();

    if j == 0 {
        return Err(EconError::InvalidSpecification {
            message: "No donor units".to_string(),
        });
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
        let d_h_d: f64 = direction
            .iter()
            .zip(hd.iter())
            .map(|(&di, &hi)| di * hi)
            .sum();
        let d_h_w: f64 = direction
            .iter()
            .zip(h.dot(&w).iter())
            .map(|(&di, &hi)| di * hi)
            .sum();
        let c_d: f64 = c
            .iter()
            .zip(direction.iter())
            .map(|(&ci, &di)| ci * di)
            .sum();

        let alpha = if d_h_d.abs() > 1e-12 {
            (-(d_h_w + c_d) / d_h_d).max(0.0).min(1.0)
        } else {
            0.0
        };

        // Update
        let w_new = &w + &(&direction * alpha);

        // Check convergence
        let change: f64 = w_new
            .iter()
            .zip(w.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
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
    // Get all units
    let df = dataset.df();
    let units: Vec<String> = df
        .column(unit_col)
        .map_err(|e| EconError::ColumnNotFound {
            column: unit_col.to_string(),
            available: vec![format!("{:?}", e)],
        })?
        .str()
        .map_err(|_| EconError::InvalidSpecification {
            message: "Unit column must be string".to_string(),
        })?
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

    // Run placebo for each donor unit IN PARALLEL
    // Each placebo test is independent, making this embarrassingly parallel
    let donor_ratios: Vec<(String, f64)> = units
        .par_iter()
        .filter(|unit| *unit != &config.treated_unit)
        .filter_map(|donor_unit| {
            // Run synth with this donor as treated
            let placebo_config = SynthConfig {
                treated_unit: donor_unit.clone(),
                run_placebos: false,
                ..config.clone()
            };

            run_synthetic_control(
                dataset,
                outcome,
                unit_col,
                time_col,
                predictors,
                placebo_config,
            )
            .ok()
            .map(|placebo_result| {
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

                (donor_unit.clone(), ratio)
            })
        })
        .collect();

    // Combine treated unit ratio with donor ratios
    let mut rmspe_ratios: Vec<(String, f64)> = vec![(config.treated_unit.clone(), treated_ratio)];
    rmspe_ratios.extend(donor_ratios);

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

        let result =
            run_synthetic_control(&dataset, "outcome", "unit", "time", &predictors, config)
                .unwrap();

        // Test Display trait
        let output = format!("{}", result);
        assert!(output.contains("Synthetic Control"));
        assert!(output.contains("Treated Unit: A"));
        assert!(output.contains("PREDICTOR BALANCE"));
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
    let df = dataset.df();

    // Extract data
    let units = extract_id_column(df, unit_col)?;
    let times = extract_time_column(df, time_col)?;
    let y = extract_numeric_column(df, y_col)?;
    let d = extract_numeric_column(df, d_col)?;

    // Build panel structure
    let unique_units: Vec<String> = units
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let mut unique_times: Vec<i64> = times
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    unique_times.sort();

    let n_units = unique_units.len();
    let n_times = unique_times.len();

    // Build index mappings
    let unit_idx: std::collections::HashMap<&str, usize> = unique_units
        .iter()
        .enumerate()
        .map(|(i, u)| (u.as_str(), i))
        .collect();
    let time_idx: std::collections::HashMap<i64, usize> = unique_times
        .iter()
        .enumerate()
        .map(|(i, &t)| (t, i))
        .collect();

    // Build Y and D matrices (units x times)
    let mut y_mat = Array2::<f64>::from_elem((n_units, n_times), f64::NAN);
    let mut d_mat = Array2::<f64>::zeros((n_units, n_times));

    for (row_idx, (unit, time)) in units.iter().zip(times.iter()).enumerate() {
        let ui = unit_idx[unit.as_str()];
        let ti = time_idx[time];
        y_mat[[ui, ti]] = y[row_idx];
        d_mat[[ui, ti]] = d[row_idx];
    }

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

    // Extract covariates if specified
    let (x_mat, covariate_names) = if !x_cols.is_empty() {
        let mut x_data: Vec<Array2<f64>> = Vec::new();
        for col in x_cols {
            let x_vec = extract_numeric_column(df, col)?;
            let mut x_panel = Array2::<f64>::from_elem((n_units, n_times), f64::NAN);
            for (row_idx, (unit, time)) in units.iter().zip(times.iter()).enumerate() {
                let ui = unit_idx[unit.as_str()];
                let ti = time_idx[time];
                x_panel[[ui, ti]] = x_vec[row_idx];
            }
            x_data.push(x_panel);
        }
        (Some(x_data), x_cols.iter().map(|s| s.to_string()).collect())
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
            config.cv_folds,
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

/// Extract ID column as strings.
fn extract_id_column(df: &DataFrame, col: &str) -> EconResult<Vec<String>> {
    let series = df.column(col).map_err(|_| EconError::ColumnNotFound {
        column: col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    if let Ok(ca) = series.str() {
        Ok(ca.into_no_null_iter().map(|s| s.to_string()).collect())
    } else if let Ok(ca) = series.i64() {
        Ok(ca.into_no_null_iter().map(|v| v.to_string()).collect())
    } else if let Ok(ca) = series.i32() {
        Ok(ca.into_no_null_iter().map(|v| v.to_string()).collect())
    } else {
        Err(EconError::InvalidSpecification {
            message: format!("Column '{}' must be string or integer", col),
        })
    }
}

/// Extract time column as i64.
fn extract_time_column(df: &DataFrame, col: &str) -> EconResult<Vec<i64>> {
    let series = df.column(col).map_err(|_| EconError::ColumnNotFound {
        column: col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    if let Ok(ca) = series.i64() {
        Ok(ca.into_no_null_iter().collect())
    } else if let Ok(ca) = series.i32() {
        Ok(ca.into_no_null_iter().map(|v| v as i64).collect())
    } else if let Ok(ca) = series.f64() {
        Ok(ca.into_no_null_iter().map(|v| v as i64).collect())
    } else {
        Err(EconError::InvalidSpecification {
            message: format!("Column '{}' must be numeric", col),
        })
    }
}

/// Extract numeric column.
fn extract_numeric_column(df: &DataFrame, col: &str) -> EconResult<Vec<f64>> {
    let series = df.column(col).map_err(|_| EconError::ColumnNotFound {
        column: col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    if let Ok(ca) = series.f64() {
        Ok(ca.into_no_null_iter().collect())
    } else if let Ok(ca) = series.i64() {
        Ok(ca.into_no_null_iter().map(|v| v as f64).collect())
    } else if let Ok(ca) = series.i32() {
        Ok(ca.into_no_null_iter().map(|v| v as f64).collect())
    } else {
        Err(EconError::InvalidSpecification {
            message: format!("Column '{}' must be numeric", col),
        })
    }
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

    // Compute Y'Y (T x T) for factor extraction
    let yty = y.t().dot(y); // T x T

    // Simple eigenvalue decomposition via power iteration with deflation
    let mut factors = Array2::<f64>::zeros((n_cols, r));
    let mut eigenvalues = Vec::with_capacity(r);

    let mut mat = yty.clone();

    for k in 0..r {
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
fn bootstrap_gsynth(
    _y_mat: &Array2<f64>,
    _d_mat: &Array2<f64>,
    _treated_units: &[usize],
    _control_units: &[usize],
    _n_pre: usize,
    _n_factors: usize,
    config: &GsynthConfig,
) -> EconResult<(Option<f64>, Option<(f64, f64)>, Option<f64>)> {
    // Placeholder - full bootstrap would resample and re-estimate
    // For now, return None to indicate bootstrap not performed
    if !config.bootstrap_se {
        return Ok((None, None, None));
    }

    // TODO: Implement full bootstrap
    Ok((None, None, None))
}

#[cfg(test)]
mod gsynth_tests {
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
}
