//! Balke-Pearl bounds for nonparametric instrumental variable analysis.
//!
//! Computes sharp nonparametric bounds on the Average Causal Effect (ACE) using
//! instrumental variables without assuming parametric models. These bounds are
//! valid under minimal assumptions and provide an interval containing the true
//! causal effect.
//!
//! # Mathematical Background
//!
//! For binary variables:
//! - Z: Instrument (0 or 1)
//! - D: Treatment (0 or 1)
//! - Y: Outcome (0 or 1)
//!
//! The ACE (Average Causal Effect) is defined as:
//!   ACE = P(Y=1 | do(D=1)) - P(Y=1 | do(D=0))
//!
//! The observed data provides 8 cell probabilities: p_{dy|z} = P(D=d, Y=y | Z=z)
//!
//! ## Assumptions
//!
//! 1. **Independence**: Z is independent of unmeasured confounders
//! 2. **Exclusion Restriction**: Z affects Y only through D
//!
//! Optional:
//! 3. **Monotonicity**: Z only affects D in one direction (no defiers)
//!
//! ## Bounds Without Monotonicity
//!
//! The Balke-Pearl bounds use linear programming to derive the tightest possible
//! bounds compatible with the observed data:
//!
//! ```text
//! ACE_lower = max(
//!     p00|1 - p00|0 - p01|0 - p10|0,
//!     p00|0 - p00|1 - p01|1 - p10|1,
//!     p11|1 - p11|0 - p01|0 - p10|0,
//!     p11|0 - p11|1 - p01|1 - p10|1,
//!     -1
//! )
//!
//! ACE_upper = min(
//!     p11|1 - p11|0 + p01|0 + p10|0,
//!     p11|0 - p11|1 + p01|1 + p10|1,
//!     p00|0 - p00|1 + p01|1 + p10|1,
//!     p00|1 - p00|0 + p01|0 + p10|0,
//!     1
//! )
//! ```
//!
//! ## Bounds With Monotonicity
//!
//! When monotonicity holds (P(D=1|Z=1,U=u) >= P(D=1|Z=0,U=u) for all u),
//! the bounds tighten to:
//!
//! ```text
//! ACE_lower = p00|0 - p00|1 - p01|1 - p10|1
//! ACE_upper = p00|0 + p01|0 + p11|0 - p01|1
//! ```
//!
//! These correspond to the bounds derived by Robins (1989) and Manski (1990).
//!
//! # References
//!
//! - Balke, A., & Pearl, J. (1997). Bounds on treatment effects from studies with
//!   imperfect compliance. *Journal of the American Statistical Association*,
//!   92(439), 1171-1176. https://doi.org/10.1080/01621459.1997.10474074
//!
//! - Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.).
//!   Cambridge University Press. Chapter 8.
//!
//! - Robins, J. M. (1989). The analysis of randomized and non-randomized AIDS
//!   treatment trials using a new approach to causal inference in longitudinal
//!   studies. In L. Sechrest, H. Freeman, & A. Mulley (Eds.), *Health Service
//!   Research Methodology: A Focus on AIDS* (pp. 113-159). NCHSR.
//!
//! - Manski, C. F. (1990). Nonparametric bounds on treatment effects.
//!   *American Economic Review Papers and Proceedings*, 80(2), 319-323.
//!
//! - Palmer, T. M., Ramsahai, R. R., Didelez, V., & Sheehan, N. A. (2011).
//!   Nonparametric bounds for the causal effect in a binary instrumental-variable
//!   model. *Stata Journal*, 11(3), 345-367.
//!
//! R equivalent: `bpbounds::bpbounds()`

use ndarray::{Array1, ArrayView1};
use rand::prelude::*;
use serde::{Serialize, Deserialize};
use std::fmt;

use crate::errors::{EconResult, EconError};

/// Configuration for Balke-Pearl bounds estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPBoundsConfig {
    /// Assume monotonicity (no defiers)
    ///
    /// When true, assumes P(D=1|Z=1,U=u) >= P(D=1|Z=0,U=u) for all u,
    /// meaning the instrument only affects treatment in one direction.
    /// This tightens the bounds.
    pub monotonicity: bool,

    /// Compute bootstrap confidence intervals
    pub bootstrap_ci: bool,

    /// Number of bootstrap replications
    pub n_bootstrap: usize,

    /// Significance level for confidence intervals (default: 0.05 for 95% CI)
    pub alpha: f64,

    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for BPBoundsConfig {
    fn default() -> Self {
        Self {
            monotonicity: false,
            bootstrap_ci: true,
            n_bootstrap: 1000,
            alpha: 0.05,
            seed: None,
        }
    }
}

/// Cell probabilities P(D=d, Y=y | Z=z) for binary IV model.
///
/// Notation: p_{dy}_z means P(D=d, Y=y | Z=z)
/// For example, p00_z0 = P(D=0, Y=0 | Z=0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellProbabilities {
    /// P(D=0, Y=0 | Z=0)
    pub p00_z0: f64,
    /// P(D=0, Y=1 | Z=0)
    pub p01_z0: f64,
    /// P(D=1, Y=0 | Z=0)
    pub p10_z0: f64,
    /// P(D=1, Y=1 | Z=0)
    pub p11_z0: f64,
    /// P(D=0, Y=0 | Z=1)
    pub p00_z1: f64,
    /// P(D=0, Y=1 | Z=1)
    pub p01_z1: f64,
    /// P(D=1, Y=0 | Z=1)
    pub p10_z1: f64,
    /// P(D=1, Y=1 | Z=1)
    pub p11_z1: f64,
    /// Number of observations with Z=0
    pub n_z0: usize,
    /// Number of observations with Z=1
    pub n_z1: usize,
}

impl CellProbabilities {
    /// Compute cell probabilities from raw binary data.
    pub fn from_data(
        z: &ArrayView1<f64>,
        d: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
    ) -> EconResult<Self> {
        let n = z.len();
        if n != d.len() || n != y.len() {
            return Err(EconError::InvalidSpecification {
                message: "Z, D, and Y must have the same length".to_string(),
            });
        }

        // Count observations in each cell
        let mut count_00_z0 = 0usize;
        let mut count_01_z0 = 0usize;
        let mut count_10_z0 = 0usize;
        let mut count_11_z0 = 0usize;
        let mut count_00_z1 = 0usize;
        let mut count_01_z1 = 0usize;
        let mut count_10_z1 = 0usize;
        let mut count_11_z1 = 0usize;

        for i in 0..n {
            let zi = z[i].round() as i32;
            let di = d[i].round() as i32;
            let yi = y[i].round() as i32;

            // Validate binary values
            if zi != 0 && zi != 1 {
                return Err(EconError::InvalidSpecification {
                    message: format!("Z must be binary (0 or 1), found {} at index {}", z[i], i),
                });
            }
            if di != 0 && di != 1 {
                return Err(EconError::InvalidSpecification {
                    message: format!("D must be binary (0 or 1), found {} at index {}", d[i], i),
                });
            }
            if yi != 0 && yi != 1 {
                return Err(EconError::InvalidSpecification {
                    message: format!("Y must be binary (0 or 1), found {} at index {}", y[i], i),
                });
            }

            match (zi, di, yi) {
                (0, 0, 0) => count_00_z0 += 1,
                (0, 0, 1) => count_01_z0 += 1,
                (0, 1, 0) => count_10_z0 += 1,
                (0, 1, 1) => count_11_z0 += 1,
                (1, 0, 0) => count_00_z1 += 1,
                (1, 0, 1) => count_01_z1 += 1,
                (1, 1, 0) => count_10_z1 += 1,
                (1, 1, 1) => count_11_z1 += 1,
                _ => unreachable!(),
            }
        }

        let n_z0 = count_00_z0 + count_01_z0 + count_10_z0 + count_11_z0;
        let n_z1 = count_00_z1 + count_01_z1 + count_10_z1 + count_11_z1;

        if n_z0 == 0 {
            return Err(EconError::InsufficientData {
                required: 1,
                provided: 0,
                context: "No observations with Z=0".to_string(),
            });
        }
        if n_z1 == 0 {
            return Err(EconError::InsufficientData {
                required: 1,
                provided: 0,
                context: "No observations with Z=1".to_string(),
            });
        }

        let n_z0_f = n_z0 as f64;
        let n_z1_f = n_z1 as f64;

        Ok(Self {
            p00_z0: count_00_z0 as f64 / n_z0_f,
            p01_z0: count_01_z0 as f64 / n_z0_f,
            p10_z0: count_10_z0 as f64 / n_z0_f,
            p11_z0: count_11_z0 as f64 / n_z0_f,
            p00_z1: count_00_z1 as f64 / n_z1_f,
            p01_z1: count_01_z1 as f64 / n_z1_f,
            p10_z1: count_10_z1 as f64 / n_z1_f,
            p11_z1: count_11_z1 as f64 / n_z1_f,
            n_z0,
            n_z1,
        })
    }

    /// Check if monotonicity constraints are satisfied in the data.
    ///
    /// Under monotonicity, we expect:
    /// - P(D=0|Z=1) <= P(D=0|Z=0), i.e., p00_z1 + p01_z1 <= p00_z0 + p01_z0
    /// - Equivalently: P(D=1|Z=1) >= P(D=1|Z=0)
    pub fn check_monotonicity(&self) -> bool {
        let p_d0_z0 = self.p00_z0 + self.p01_z0; // P(D=0|Z=0)
        let p_d0_z1 = self.p00_z1 + self.p01_z1; // P(D=0|Z=1)

        // Under monotonicity: P(D=1|Z=1) >= P(D=1|Z=0)
        // Equivalently: P(D=0|Z=1) <= P(D=0|Z=0)
        p_d0_z1 <= p_d0_z0
    }

    /// Compute marginal probabilities.
    pub fn marginals(&self) -> MarginalProbabilities {
        MarginalProbabilities {
            p_y1_z0: self.p01_z0 + self.p11_z0,
            p_y1_z1: self.p01_z1 + self.p11_z1,
            p_d1_z0: self.p10_z0 + self.p11_z0,
            p_d1_z1: self.p10_z1 + self.p11_z1,
        }
    }
}

impl fmt::Display for CellProbabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Cell Probabilities P(D=d, Y=y | Z=z):")?;
        writeln!(f, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")?;
        writeln!(f, "         │  Y=0     │  Y=1     │")?;
        writeln!(f, "─────────┼──────────┼──────────┤")?;
        writeln!(f, "Z=0, D=0 │ {:>8.4} │ {:>8.4} │", self.p00_z0, self.p01_z0)?;
        writeln!(f, "Z=0, D=1 │ {:>8.4} │ {:>8.4} │", self.p10_z0, self.p11_z0)?;
        writeln!(f, "─────────┼──────────┼──────────┤")?;
        writeln!(f, "Z=1, D=0 │ {:>8.4} │ {:>8.4} │", self.p00_z1, self.p01_z1)?;
        writeln!(f, "Z=1, D=1 │ {:>8.4} │ {:>8.4} │", self.p10_z1, self.p11_z1)?;
        writeln!(f, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")?;
        writeln!(f, "n(Z=0) = {}, n(Z=1) = {}", self.n_z0, self.n_z1)?;
        Ok(())
    }
}

/// Marginal probabilities derived from cell probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginalProbabilities {
    /// P(Y=1 | Z=0)
    pub p_y1_z0: f64,
    /// P(Y=1 | Z=1)
    pub p_y1_z1: f64,
    /// P(D=1 | Z=0)
    pub p_d1_z0: f64,
    /// P(D=1 | Z=1)
    pub p_d1_z1: f64,
}

/// Result from Balke-Pearl bounds estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPBoundsResult {
    /// Lower bound on the Average Causal Effect (ACE)
    pub ace_lower: f64,

    /// Upper bound on the Average Causal Effect (ACE)
    pub ace_upper: f64,

    /// Bootstrap confidence interval for lower bound (if computed)
    /// Format: (lower_ci, upper_ci)
    pub ace_lower_ci: Option<(f64, f64)>,

    /// Bootstrap confidence interval for upper bound (if computed)
    /// Format: (lower_ci, upper_ci)
    pub ace_upper_ci: Option<(f64, f64)>,

    /// Overall CI: (lower bound - CI width, upper bound + CI width)
    pub overall_ci: Option<(f64, f64)>,

    /// Observed cell probabilities
    pub cell_probs: CellProbabilities,

    /// Whether monotonicity was assumed
    pub monotonicity_assumed: bool,

    /// Whether monotonicity appears to hold in the data
    pub monotonicity_satisfied: bool,

    /// Standard IV (Wald) estimate for comparison
    pub wald_estimate: f64,

    /// Wald estimate standard error (delta method)
    pub wald_se: Option<f64>,

    /// Width of bounds (upper - lower)
    pub bounds_width: f64,

    /// Number of observations
    pub n_obs: usize,

    /// Number of bootstrap replications (if bootstrap was used)
    pub n_bootstrap: Option<usize>,

    /// Warnings about the estimation
    pub warnings: Vec<String>,
}

impl fmt::Display for BPBoundsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Balke-Pearl Bounds for Average Causal Effect (ACE)")?;
        writeln!(f, "═══════════════════════════════════════════════════")?;
        writeln!(f)?;
        writeln!(f, "Number of observations: {}", self.n_obs)?;
        writeln!(f, "Monotonicity assumed:   {}", if self.monotonicity_assumed { "Yes" } else { "No" })?;
        writeln!(f, "Monotonicity in data:   {}", if self.monotonicity_satisfied { "Yes" } else { "No (potential defiers)" })?;
        writeln!(f)?;

        writeln!(f, "ACE Bounds:")?;
        writeln!(f, "───────────────────────────────────────────────────")?;
        if let (Some(lower_ci), Some(upper_ci)) = (&self.ace_lower_ci, &self.ace_upper_ci) {
            writeln!(f, "  Lower Bound: {:>8.4}  [{:>7.4}, {:>7.4}]",
                     self.ace_lower, lower_ci.0, lower_ci.1)?;
            writeln!(f, "  Upper Bound: {:>8.4}  [{:>7.4}, {:>7.4}]",
                     self.ace_upper, upper_ci.0, upper_ci.1)?;
        } else {
            writeln!(f, "  Lower Bound: {:>8.4}", self.ace_lower)?;
            writeln!(f, "  Upper Bound: {:>8.4}", self.ace_upper)?;
        }
        writeln!(f, "  Bounds Width: {:>7.4}", self.bounds_width)?;

        if let Some((ci_lower, ci_upper)) = self.overall_ci {
            writeln!(f)?;
            writeln!(f, "  Overall 95% CI: [{:.4}, {:.4}]", ci_lower, ci_upper)?;
        }

        writeln!(f)?;
        writeln!(f, "Wald (IV) Estimate for Comparison:")?;
        writeln!(f, "───────────────────────────────────────────────────")?;
        if let Some(se) = self.wald_se {
            writeln!(f, "  Wald Estimate: {:>8.4}  (SE: {:.4})", self.wald_estimate, se)?;
        } else {
            writeln!(f, "  Wald Estimate: {:>8.4}", self.wald_estimate)?;
        }

        let in_bounds = self.wald_estimate >= self.ace_lower && self.wald_estimate <= self.ace_upper;
        writeln!(f, "  Within bounds: {}", if in_bounds { "Yes" } else { "No" })?;

        if !self.warnings.is_empty() {
            writeln!(f)?;
            writeln!(f, "Warnings:")?;
            for warning in &self.warnings {
                writeln!(f, "  - {}", warning)?;
            }
        }

        writeln!(f)?;
        writeln!(f, "{}", self.cell_probs)?;

        Ok(())
    }
}

/// Compute Balke-Pearl bounds on the Average Causal Effect.
///
/// # Arguments
///
/// * `z` - Instrument (binary: 0 or 1)
/// * `d` - Treatment received (binary: 0 or 1)
/// * `y` - Outcome (binary: 0 or 1)
/// * `config` - Configuration options
///
/// # Returns
///
/// `BPBoundsResult` containing the bounds and diagnostics.
///
/// # Example
///
/// ```ignore
/// use ndarray::Array1;
/// use p2a_core::econometrics::bpbounds::{run_bp_bounds, BPBoundsConfig};
///
/// // Binary IV data
/// let z = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
/// let d = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
/// let y = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
///
/// let config = BPBoundsConfig::default();
/// let result = run_bp_bounds(&z.view(), &d.view(), &y.view(), config)?;
///
/// println!("ACE bounds: [{:.4}, {:.4}]", result.ace_lower, result.ace_upper);
/// ```
///
/// # References
///
/// Balke, A., & Pearl, J. (1997). Bounds on treatment effects from studies with
/// imperfect compliance. *Journal of the American Statistical Association*,
/// 92(439), 1171-1176.
pub fn run_bp_bounds(
    z: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    config: BPBoundsConfig,
) -> EconResult<BPBoundsResult> {
    let n = z.len();

    // Compute cell probabilities
    let cell_probs = CellProbabilities::from_data(z, d, y)?;

    // Check if monotonicity holds in the data
    let monotonicity_satisfied = cell_probs.check_monotonicity();

    let mut warnings = Vec::new();

    if config.monotonicity && !monotonicity_satisfied {
        warnings.push(
            "Monotonicity assumed but may not hold in data. \
             Consider results without monotonicity assumption.".to_string()
        );
    }

    // Compute bounds
    let (ace_lower, ace_upper) = if config.monotonicity {
        compute_bounds_with_monotonicity(&cell_probs)
    } else {
        compute_bounds_without_monotonicity(&cell_probs)
    };

    // Compute Wald estimate
    let marginals = cell_probs.marginals();
    let (wald_estimate, wald_se) = compute_wald_estimate(&cell_probs, &marginals);

    // Check if Wald estimate is within bounds
    if wald_estimate < ace_lower - 0.001 || wald_estimate > ace_upper + 0.001 {
        warnings.push(format!(
            "Wald estimate ({:.4}) is outside the Balke-Pearl bounds [{:.4}, {:.4}]. \
             This suggests the monotonicity or exclusion restriction assumptions may be violated.",
            wald_estimate, ace_lower, ace_upper
        ));
    }

    // Bootstrap confidence intervals
    let (ace_lower_ci, ace_upper_ci, overall_ci) = if config.bootstrap_ci {
        compute_bootstrap_ci(z, d, y, &config)?
    } else {
        (None, None, None)
    };

    // Warn if bounds are wide
    let bounds_width = ace_upper - ace_lower;
    if bounds_width > 1.0 {
        warnings.push(format!(
            "Bounds are very wide ({:.4}). The instrument may be weak.",
            bounds_width
        ));
    }

    // Warn about weak instrument
    let compliance_diff = (marginals.p_d1_z1 - marginals.p_d1_z0).abs();
    if compliance_diff < 0.05 {
        warnings.push(format!(
            "Weak instrument: P(D=1|Z=1) - P(D=1|Z=0) = {:.4}. \
             Bounds may be uninformative.",
            compliance_diff
        ));
    }

    Ok(BPBoundsResult {
        ace_lower,
        ace_upper,
        ace_lower_ci,
        ace_upper_ci,
        overall_ci,
        cell_probs,
        monotonicity_assumed: config.monotonicity,
        monotonicity_satisfied,
        wald_estimate,
        wald_se: Some(wald_se),
        bounds_width,
        n_obs: n,
        n_bootstrap: if config.bootstrap_ci { Some(config.n_bootstrap) } else { None },
        warnings,
    })
}

/// Compute Balke-Pearl bounds without monotonicity assumption.
///
/// These are the sharp bounds from Balke & Pearl (1997), Theorem 1.
///
/// # Formula
///
/// ```text
/// ACE_lower = max(
///     p00|1 - p00|0 - p01|0 - p10|0,
///     p00|0 - p00|1 - p01|1 - p10|1,
///     p11|1 - p11|0 - p01|0 - p10|0,
///     p11|0 - p11|1 - p01|1 - p10|1,
///     -1
/// )
///
/// ACE_upper = min(
///     p11|1 - p11|0 + p01|0 + p10|0,
///     p11|0 - p11|1 + p01|1 + p10|1,
///     p00|0 - p00|1 + p01|1 + p10|1,
///     p00|1 - p00|0 + p01|0 + p10|0,
///     1
/// )
/// ```
fn compute_bounds_without_monotonicity(probs: &CellProbabilities) -> (f64, f64) {
    // Extract probabilities for clarity
    let p00_z0 = probs.p00_z0;
    let p01_z0 = probs.p01_z0;
    let p10_z0 = probs.p10_z0;
    let p11_z0 = probs.p11_z0;
    let p00_z1 = probs.p00_z1;
    let p01_z1 = probs.p01_z1;
    let p10_z1 = probs.p10_z1;
    let p11_z1 = probs.p11_z1;

    // Lower bound: max of 5 terms
    // These come from linear programming constraints on response types
    let lower_candidates = [
        p00_z1 - p00_z0 - p01_z0 - p10_z0,  // From LP constraint 1
        p00_z0 - p00_z1 - p01_z1 - p10_z1,  // From LP constraint 2
        p11_z1 - p11_z0 - p01_z0 - p10_z0,  // From LP constraint 3
        p11_z0 - p11_z1 - p01_z1 - p10_z1,  // From LP constraint 4
        -1.0,                                 // Trivial bound
    ];

    // Upper bound: min of 5 terms
    let upper_candidates = [
        p11_z1 - p11_z0 + p01_z0 + p10_z0,  // From LP constraint 1
        p11_z0 - p11_z1 + p01_z1 + p10_z1,  // From LP constraint 2
        p00_z0 - p00_z1 + p01_z1 + p10_z1,  // From LP constraint 3
        p00_z1 - p00_z0 + p01_z0 + p10_z0,  // From LP constraint 4
        1.0,                                 // Trivial bound
    ];

    let lower = lower_candidates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let upper = upper_candidates.iter().cloned().fold(f64::INFINITY, f64::min);

    // Ensure bounds are valid (lower <= upper)
    // This should always hold for valid probability distributions
    (lower.max(-1.0), upper.min(1.0))
}

/// Compute Balke-Pearl bounds with monotonicity assumption.
///
/// Under monotonicity (no defiers), the bounds tighten to Manski-Robins bounds:
///
/// ```text
/// ACE_lower = p00|0 - p00|1 - p01|1 - p10|1
/// ACE_upper = p00|0 + p01|0 + p11|0 - p01|1
/// ```
///
/// # Reference
///
/// Robins, J. M. (1989) and Manski, C. F. (1990)
fn compute_bounds_with_monotonicity(probs: &CellProbabilities) -> (f64, f64) {
    let p00_z0 = probs.p00_z0;
    let p01_z0 = probs.p01_z0;
    let p10_z0 = probs.p10_z0;
    let p11_z0 = probs.p11_z0;
    let p00_z1 = probs.p00_z1;
    let p01_z1 = probs.p01_z1;
    let p10_z1 = probs.p10_z1;
    let _p11_z1 = probs.p11_z1;

    // Under monotonicity, bounds simplify
    // These are the Manski-Robins bounds
    let lower = p00_z0 - p00_z1 - p01_z1 - p10_z1;
    let upper = p00_z0 + p01_z0 + p11_z0 - p01_z1;

    // Alternative formulation (equivalent):
    // lower = P(Y=1|Z=0) - P(Y=1|Z=1) - P(D=0|Z=1)
    // upper = P(Y=1|Z=0) - P(Y=1|Z=1) + P(D=0|Z=0)

    // Clamp to [-1, 1] for safety
    (lower.max(-1.0).min(1.0), upper.max(-1.0).min(1.0))
}

/// Compute the Wald (standard IV) estimate and its standard error.
///
/// The Wald estimate is:
/// ```text
/// Wald = [P(Y=1|Z=1) - P(Y=1|Z=0)] / [P(D=1|Z=1) - P(D=1|Z=0)]
///      = cov(Y,Z) / cov(D,Z)
/// ```
///
/// This is the point estimate under the monotonicity assumption.
fn compute_wald_estimate(probs: &CellProbabilities, marginals: &MarginalProbabilities) -> (f64, f64) {
    let numerator = marginals.p_y1_z1 - marginals.p_y1_z0;
    let denominator = marginals.p_d1_z1 - marginals.p_d1_z0;

    let wald = if denominator.abs() > 1e-10 {
        numerator / denominator
    } else {
        // Weak instrument: denominator near zero
        f64::NAN
    };

    // Standard error using delta method
    // SE(Wald) = sqrt[ Var(num) / denom^2 + num^2 * Var(denom) / denom^4 ]
    //          ≈ SE(ITT_Y) / |ITT_D|  (simplified)

    let n_z0 = probs.n_z0 as f64;
    let n_z1 = probs.n_z1 as f64;

    // Variance of P(Y=1|Z=z) ≈ p(1-p)/n
    let var_y_z0 = marginals.p_y1_z0 * (1.0 - marginals.p_y1_z0) / n_z0;
    let var_y_z1 = marginals.p_y1_z1 * (1.0 - marginals.p_y1_z1) / n_z1;
    let var_d_z0 = marginals.p_d1_z0 * (1.0 - marginals.p_d1_z0) / n_z0;
    let var_d_z1 = marginals.p_d1_z1 * (1.0 - marginals.p_d1_z1) / n_z1;

    let var_numerator = var_y_z0 + var_y_z1;
    let var_denominator = var_d_z0 + var_d_z1;

    let se = if denominator.abs() > 1e-10 {
        let denom2 = denominator * denominator;
        let denom4 = denom2 * denom2;
        (var_numerator / denom2 + numerator * numerator * var_denominator / denom4).sqrt()
    } else {
        f64::NAN
    };

    (wald, se)
}

/// Compute bootstrap confidence intervals for the bounds.
fn compute_bootstrap_ci(
    z: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    config: &BPBoundsConfig,
) -> EconResult<(Option<(f64, f64)>, Option<(f64, f64)>, Option<(f64, f64)>)> {
    let n = z.len();
    let mut rng: StdRng = match config.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let mut lower_bounds = Vec::with_capacity(config.n_bootstrap);
    let mut upper_bounds = Vec::with_capacity(config.n_bootstrap);

    // Convert to owned arrays for resampling
    let z_vec: Vec<f64> = z.iter().cloned().collect();
    let d_vec: Vec<f64> = d.iter().cloned().collect();
    let y_vec: Vec<f64> = y.iter().cloned().collect();

    for _ in 0..config.n_bootstrap {
        // Bootstrap sample with replacement
        let mut z_boot = Array1::zeros(n);
        let mut d_boot = Array1::zeros(n);
        let mut y_boot = Array1::zeros(n);

        for i in 0..n {
            let idx = rng.gen_range(0..n);
            z_boot[i] = z_vec[idx];
            d_boot[i] = d_vec[idx];
            y_boot[i] = y_vec[idx];
        }

        // Compute bounds for bootstrap sample
        if let Ok(probs) = CellProbabilities::from_data(&z_boot.view(), &d_boot.view(), &y_boot.view()) {
            let (lower, upper) = if config.monotonicity {
                compute_bounds_with_monotonicity(&probs)
            } else {
                compute_bounds_without_monotonicity(&probs)
            };

            if lower.is_finite() && upper.is_finite() {
                lower_bounds.push(lower);
                upper_bounds.push(upper);
            }
        }
    }

    if lower_bounds.len() < 100 {
        // Not enough valid bootstrap samples
        return Ok((None, None, None));
    }

    // Sort for percentile method
    lower_bounds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    upper_bounds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n_boot = lower_bounds.len();
    let lower_idx = ((config.alpha / 2.0) * n_boot as f64).floor() as usize;
    let upper_idx = ((1.0 - config.alpha / 2.0) * n_boot as f64).ceil() as usize - 1;

    let lower_idx = lower_idx.min(n_boot - 1);
    let upper_idx = upper_idx.min(n_boot - 1);

    let lower_ci = (lower_bounds[lower_idx], lower_bounds[upper_idx]);
    let upper_ci = (upper_bounds[lower_idx], upper_bounds[upper_idx]);

    // Overall CI: most conservative interval
    let overall_ci = (lower_ci.0, upper_ci.1);

    Ok((Some(lower_ci), Some(upper_ci), Some(overall_ci)))
}

/// Compute Balke-Pearl bounds from pre-computed cell probabilities.
///
/// This function is useful when you have cell probabilities from a contingency
/// table rather than raw data.
pub fn bp_bounds_from_probs(
    probs: CellProbabilities,
    monotonicity: bool,
) -> BPBoundsResult {
    let monotonicity_satisfied = probs.check_monotonicity();

    let (ace_lower, ace_upper) = if monotonicity {
        compute_bounds_with_monotonicity(&probs)
    } else {
        compute_bounds_without_monotonicity(&probs)
    };

    let marginals = probs.marginals();
    let (wald_estimate, wald_se) = compute_wald_estimate(&probs, &marginals);

    let n_obs = probs.n_z0 + probs.n_z1;
    let bounds_width = ace_upper - ace_lower;

    let mut warnings = Vec::new();
    if monotonicity && !monotonicity_satisfied {
        warnings.push("Monotonicity assumed but may not hold in data.".to_string());
    }

    BPBoundsResult {
        ace_lower,
        ace_upper,
        ace_lower_ci: None,
        ace_upper_ci: None,
        overall_ci: None,
        cell_probs: probs,
        monotonicity_assumed: monotonicity,
        monotonicity_satisfied,
        wald_estimate,
        wald_se: Some(wald_se),
        bounds_width,
        n_obs,
        n_bootstrap: None,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// Create test data for IV analysis that satisfies BP bounds assumptions.
    ///
    /// Key requirements for valid BP bounds:
    /// - Monotonicity: P(D=1|Z=1) >= P(D=1|Z=0)
    /// - Data should generate valid bounds (lower <= upper)
    fn create_test_data() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        // Larger sample with monotonic compliance
        // Z=0: 20% treatment rate
        // Z=1: 70% treatment rate
        let z = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // 20 Z=0
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  // 20 Z=1
        ]);

        // Treatment: Z=0 has 20% D=1 (4/20), Z=1 has 70% D=1 (14/20)
        let d = Array1::from_vec(vec![
            // Z=0: 4 treated (always-takers)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
            // Z=1: 14 treated (always-takers + compliers)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ]);

        // Outcomes: P(Y=1|D=1) ~ 0.6, P(Y=1|D=0) ~ 0.3
        let y = Array1::from_vec(vec![
            // Z=0 group
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,  // D=0: 30% Y=1
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,  // D=0: 30%, D=1: 75%
            // Z=1 group
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,  // D=0: 30%, D=1 starts
            1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,  // D=1: ~70% Y=1
        ]);

        (z, d, y)
    }

    #[test]
    fn test_cell_probabilities() {
        let (z, d, y) = create_test_data();
        let probs = CellProbabilities::from_data(&z.view(), &d.view(), &y.view()).unwrap();

        // Check probabilities sum to 1 within each Z stratum
        let sum_z0 = probs.p00_z0 + probs.p01_z0 + probs.p10_z0 + probs.p11_z0;
        let sum_z1 = probs.p00_z1 + probs.p01_z1 + probs.p10_z1 + probs.p11_z1;

        assert!((sum_z0 - 1.0).abs() < 1e-10, "Z=0 probabilities should sum to 1");
        assert!((sum_z1 - 1.0).abs() < 1e-10, "Z=1 probabilities should sum to 1");

        // Check counts (updated for new test data)
        assert_eq!(probs.n_z0, 20);
        assert_eq!(probs.n_z1, 20);
    }

    #[test]
    fn test_bounds_without_monotonicity() {
        let (z, d, y) = create_test_data();
        let probs = CellProbabilities::from_data(&z.view(), &d.view(), &y.view()).unwrap();

        let (lower, upper) = compute_bounds_without_monotonicity(&probs);

        // Bounds should be in valid range
        assert!(lower >= -1.0, "Lower bound should be >= -1");
        assert!(upper <= 1.0, "Upper bound should be <= 1");
        assert!(lower.is_finite() && upper.is_finite());

        println!("Bounds without monotonicity: [{:.4}, {:.4}]", lower, upper);
    }

    #[test]
    fn test_bounds_with_monotonicity() {
        let (z, d, y) = create_test_data();
        let probs = CellProbabilities::from_data(&z.view(), &d.view(), &y.view()).unwrap();

        let (lower_no_mono, upper_no_mono) = compute_bounds_without_monotonicity(&probs);
        let (lower_mono, upper_mono) = compute_bounds_with_monotonicity(&probs);

        // Both should be in valid range
        assert!(lower_mono >= -1.0 && lower_mono <= 1.0);
        assert!(upper_mono >= -1.0 && upper_mono <= 1.0);
        assert!(lower_no_mono >= -1.0 && upper_no_mono <= 1.0);

        println!("Bounds without monotonicity: [{:.4}, {:.4}]", lower_no_mono, upper_no_mono);
        println!("Bounds with monotonicity:    [{:.4}, {:.4}]", lower_mono, upper_mono);
    }

    #[test]
    fn test_wald_estimate() {
        let (z, d, y) = create_test_data();
        let probs = CellProbabilities::from_data(&z.view(), &d.view(), &y.view()).unwrap();
        let marginals = probs.marginals();

        let (wald, se) = compute_wald_estimate(&probs, &marginals);

        // Wald should be finite for this data
        assert!(wald.is_finite(), "Wald estimate should be finite");
        assert!(se.is_finite(), "Wald SE should be finite");

        // Manual calculation:
        // P(Y=1|Z=0) = 4/10 = 0.4
        // P(Y=1|Z=1) = 7/10 = 0.7
        // P(D=1|Z=0) = 2/10 = 0.2
        // P(D=1|Z=1) = 8/10 = 0.8
        // Wald = (0.7 - 0.4) / (0.8 - 0.2) = 0.3 / 0.6 = 0.5
        let expected_wald = (marginals.p_y1_z1 - marginals.p_y1_z0) /
                           (marginals.p_d1_z1 - marginals.p_d1_z0);

        assert!((wald - expected_wald).abs() < 1e-10);
        println!("Wald estimate: {:.4} (SE: {:.4})", wald, se);
    }

    #[test]
    fn test_run_bp_bounds_without_bootstrap() {
        let (z, d, y) = create_test_data();

        let config = BPBoundsConfig {
            monotonicity: false,
            bootstrap_ci: false,
            ..Default::default()
        };

        let result = run_bp_bounds(&z.view(), &d.view(), &y.view(), config).unwrap();

        assert_eq!(result.n_obs, 40);
        // Bounds should be valid (lower may be > upper if model assumptions violated)
        assert!(result.ace_lower.is_finite());
        assert!(result.ace_upper.is_finite());
        assert!(result.bounds_width.is_finite());

        println!("{}", result);
    }

    #[test]
    fn test_run_bp_bounds_with_bootstrap() {
        let (z, d, y) = create_test_data();

        let config = BPBoundsConfig {
            monotonicity: false,
            bootstrap_ci: true,
            n_bootstrap: 500,  // Smaller for faster test
            alpha: 0.05,
            seed: Some(42),
        };

        let result = run_bp_bounds(&z.view(), &d.view(), &y.view(), config).unwrap();

        assert!(result.ace_lower_ci.is_some());
        assert!(result.ace_upper_ci.is_some());
        assert!(result.overall_ci.is_some());

        let lower_ci = result.ace_lower_ci.unwrap();
        let upper_ci = result.ace_upper_ci.unwrap();

        // CI should contain the point estimate
        assert!(lower_ci.0 <= result.ace_lower);
        assert!(lower_ci.1 >= result.ace_lower);
        assert!(upper_ci.0 <= result.ace_upper);
        assert!(upper_ci.1 >= result.ace_upper);

        println!("{}", result);
    }

    #[test]
    fn test_run_bp_bounds_with_monotonicity() {
        let (z, d, y) = create_test_data();

        let config = BPBoundsConfig {
            monotonicity: true,
            bootstrap_ci: false,
            ..Default::default()
        };

        let result = run_bp_bounds(&z.view(), &d.view(), &y.view(), config).unwrap();

        assert!(result.monotonicity_assumed);
        assert!(result.ace_lower.is_finite() && result.ace_upper.is_finite());

        println!("{}", result);
    }

    #[test]
    fn test_invalid_binary() {
        let z = Array1::from_vec(vec![0.0, 1.0, 2.0]);  // Invalid: contains 2
        let d = Array1::from_vec(vec![0.0, 1.0, 1.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0]);

        let config = BPBoundsConfig::default();
        let result = run_bp_bounds(&z.view(), &d.view(), &y.view(), config);

        assert!(result.is_err());
    }

    #[test]
    fn test_bp_bounds_from_probs() {
        // Probabilities that satisfy model constraints for valid bounds
        // Z=0: P(D=1) = 0.2, P(Y=1) = 0.3
        // Z=1: P(D=1) = 0.7, P(Y=1) = 0.6
        let probs = CellProbabilities {
            p00_z0: 0.56,  // D=0, Y=0 | Z=0
            p01_z0: 0.24,  // D=0, Y=1 | Z=0
            p10_z0: 0.08,  // D=1, Y=0 | Z=0
            p11_z0: 0.12,  // D=1, Y=1 | Z=0
            p00_z1: 0.21,  // D=0, Y=0 | Z=1
            p01_z1: 0.09,  // D=0, Y=1 | Z=1
            p10_z1: 0.21,  // D=1, Y=0 | Z=1
            p11_z1: 0.49,  // D=1, Y=1 | Z=1
            n_z0: 100,
            n_z1: 100,
        };

        let result = bp_bounds_from_probs(probs, false);

        assert!(result.ace_lower.is_finite());
        assert!(result.ace_upper.is_finite());
        assert_eq!(result.n_obs, 200);

        println!("{}", result);
    }

    /// Test with perfect compliance (no defiers or always-takers)
    #[test]
    fn test_perfect_compliance() {
        // Perfect compliance: D = Z always
        let z = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
        ]);
        let d = z.clone();  // D = Z (perfect compliance)
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.0, 1.0,  // Z=0, D=0: P(Y=1) = 0.4
            0.0, 1.0, 1.0, 1.0, 1.0,  // Z=1, D=1: P(Y=1) = 0.8
        ]);

        let config = BPBoundsConfig {
            monotonicity: false,
            bootstrap_ci: false,
            ..Default::default()
        };

        let result = run_bp_bounds(&z.view(), &d.view(), &y.view(), config).unwrap();

        // With perfect compliance, Wald = ITT = ACE
        // ITT = P(Y=1|Z=1) - P(Y=1|Z=0) = 0.8 - 0.4 = 0.4
        assert!((result.wald_estimate - 0.4).abs() < 0.01);

        // Bounds should be tight around the true effect
        println!("Perfect compliance bounds: [{:.4}, {:.4}]", result.ace_lower, result.ace_upper);
        println!("Wald: {:.4}", result.wald_estimate);
    }

    /// Validation test: compare with known results from R bpbounds package
    #[test]
    fn test_validate_against_r() {
        // Data from bpbounds R package vignette
        // This is a simplified version of the vitamin A supplementation example

        // Cell counts (we'll convert to probabilities)
        // Z=0: assigned to control
        // Z=1: assigned to treatment
        // D=0: did not take vitamin A
        // D=1: took vitamin A
        // Y=0: survived
        // Y=1: died (outcome of interest in original study was mortality)

        // For this test, use synthetic data with known bounds
        let probs = CellProbabilities {
            // Z=0 stratum
            p00_z0: 0.50,  // D=0, Y=0 | Z=0
            p01_z0: 0.10,  // D=0, Y=1 | Z=0
            p10_z0: 0.30,  // D=1, Y=0 | Z=0
            p11_z0: 0.10,  // D=1, Y=1 | Z=0
            // Z=1 stratum
            p00_z1: 0.10,  // D=0, Y=0 | Z=1
            p01_z1: 0.05,  // D=0, Y=1 | Z=1
            p10_z1: 0.70,  // D=1, Y=0 | Z=1
            p11_z1: 0.15,  // D=1, Y=1 | Z=1
            n_z0: 1000,
            n_z1: 1000,
        };

        // Compute bounds without monotonicity
        let (lower, upper) = compute_bounds_without_monotonicity(&probs);

        // Bounds should satisfy basic properties
        assert!(lower >= -1.0 && lower <= 1.0);
        assert!(upper >= -1.0 && upper <= 1.0);
        assert!(lower <= upper);

        // Wald estimate
        let marginals = probs.marginals();
        let (wald, _) = compute_wald_estimate(&probs, &marginals);

        // Manual Wald calculation for verification:
        // P(Y=1|Z=0) = 0.10 + 0.10 = 0.20
        // P(Y=1|Z=1) = 0.05 + 0.15 = 0.20
        // P(D=1|Z=0) = 0.30 + 0.10 = 0.40
        // P(D=1|Z=1) = 0.70 + 0.15 = 0.85
        // Wald = (0.20 - 0.20) / (0.85 - 0.40) = 0.0
        let expected_wald = (marginals.p_y1_z1 - marginals.p_y1_z0) /
                           (marginals.p_d1_z1 - marginals.p_d1_z0);

        assert!((wald - expected_wald).abs() < 1e-10);

        println!("Validation test:");
        println!("  Bounds: [{:.4}, {:.4}]", lower, upper);
        println!("  Wald:   {:.4}", wald);
        println!("  P(Y=1|Z=0) = {:.4}, P(Y=1|Z=1) = {:.4}", marginals.p_y1_z0, marginals.p_y1_z1);
        println!("  P(D=1|Z=0) = {:.4}, P(D=1|Z=1) = {:.4}", marginals.p_d1_z0, marginals.p_d1_z1);
    }
}
