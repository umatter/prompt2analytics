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
//! The Balke-Pearl bounds are the optimum of the linear program over the 16 response
//! types; in closed form (Balke & Pearl 1997; Pearl 2009, eq. 8.14-8.15), with
//! `p_{dy|z}` written `pdy|z`:
//!
//! ```text
//! ACE_lower = max(p11|1 + p00|0 - 1, p11|0 + p00|1 - 1,
//!                 p11|0 - p11|1 - p01|1 - p10|0 - p01|0, p11|1 - p11|0 - p01|0 - p10|1 - p01|1,
//!                 -p10|1 - p01|1, -p10|0 - p01|0,
//!                 p00|1 - p10|1 - p01|1 - p10|0 - p00|0, p00|0 - p10|0 - p01|0 - p10|1 - p00|1)
//! ACE_upper = min(1 - p10|1 - p01|0, 1 - p10|0 - p01|1,
//!                 -p10|0 + p10|1 + p00|1 + p11|0 + p00|0, -p10|1 + p11|1 + p00|1 + p10|0 + p00|0,
//!                 p11|1 + p00|1, p11|0 + p00|0,
//!                 -p01|1 + p11|1 + p00|1 + p11|0 + p01|0, -p01|0 + p11|0 + p00|0 + p11|1 + p01|1)
//! ```
//!
//! ## Bounds With Monotonicity
//!
//! When monotonicity holds (no defiers), the bounds are
//!
//! ```text
//! ACE_lower = p11|1 + p00|0 - 1
//! ACE_upper = 1 - p10|1 - p01|0
//! ```
//!
//! These correspond to the bounds derived by Robins (1989) and Manski (1990).
//!
//! The Wald SE is the delta method with the within-arm covariance of P(Y=1|Z=z) and
//! P(D=1|Z=z) (equal to 2SLS with HC0 on the expanded data).
//!
//! Both paths, with and without monotonicity, test Pearl's instrumental inequality: for each
//! `d`, the result warns when `sum_y max_z P(Y = y, D = d | Z = z) > 1` ("the instrumental
//! inequality fails ..."). The test is made directly, not inferred from the bounds, which need
//! not cross when it fails.
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
//! R equivalent: `bpbounds::bpbounds()` (not installed on the reference machine; validated
//! against a hand-computed table checked with the response-type LP,
//! `validation/econometrics/bpbounds.md`)

use ndarray::{Array1, ArrayView1};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::stats::robust::percentile_interval;

/// Seed of the bootstrap resampling when [`BPBoundsConfig::seed`] is `None`.
pub const DEFAULT_BOOTSTRAP_SEED: u64 = 42;

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

    /// Number of bootstrap replications.
    /// Must be at least 1 when `bootstrap_ci` is set. Fewer than 100 valid
    /// replicates give no confidence intervals (the CI fields are `None`).
    pub n_bootstrap: usize,

    /// Significance level for confidence intervals (default: 0.05 for 95% CI).
    /// Must lie strictly between 0 and 1 when `bootstrap_ci` is set.
    pub alpha: f64,

    /// Seed of the bootstrap resampling (None: [`DEFAULT_BOOTSTRAP_SEED`])
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
        writeln!(
            f,
            "Z=0, D=0 │ {:>8.4} │ {:>8.4} │",
            self.p00_z0, self.p01_z0
        )?;
        writeln!(
            f,
            "Z=0, D=1 │ {:>8.4} │ {:>8.4} │",
            self.p10_z0, self.p11_z0
        )?;
        writeln!(f, "─────────┼──────────┼──────────┤")?;
        writeln!(
            f,
            "Z=1, D=0 │ {:>8.4} │ {:>8.4} │",
            self.p00_z1, self.p01_z1
        )?;
        writeln!(
            f,
            "Z=1, D=1 │ {:>8.4} │ {:>8.4} │",
            self.p10_z1, self.p11_z1
        )?;
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
        writeln!(
            f,
            "Monotonicity assumed:   {}",
            if self.monotonicity_assumed {
                "Yes"
            } else {
                "No"
            }
        )?;
        writeln!(
            f,
            "Monotonicity in data:   {}",
            if self.monotonicity_satisfied {
                "Yes"
            } else {
                "No (potential defiers)"
            }
        )?;
        writeln!(f)?;

        writeln!(f, "ACE Bounds:")?;
        writeln!(f, "───────────────────────────────────────────────────")?;
        if let (Some(lower_ci), Some(upper_ci)) = (&self.ace_lower_ci, &self.ace_upper_ci) {
            writeln!(
                f,
                "  Lower Bound: {:>8.4}  [{:>7.4}, {:>7.4}]",
                self.ace_lower, lower_ci.0, lower_ci.1
            )?;
            writeln!(
                f,
                "  Upper Bound: {:>8.4}  [{:>7.4}, {:>7.4}]",
                self.ace_upper, upper_ci.0, upper_ci.1
            )?;
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
            writeln!(
                f,
                "  Wald Estimate: {:>8.4}  (SE: {:.4})",
                self.wald_estimate, se
            )?;
        } else {
            writeln!(f, "  Wald Estimate: {:>8.4}", self.wald_estimate)?;
        }

        let in_bounds =
            self.wald_estimate >= self.ace_lower && self.wald_estimate <= self.ace_upper;
        writeln!(
            f,
            "  Within bounds: {}",
            if in_bounds { "Yes" } else { "No" }
        )?;

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

    if config.bootstrap_ci {
        validate_bootstrap_config(&config)?;
    }

    // Compute cell probabilities
    let cell_probs = CellProbabilities::from_data(z, d, y)?;

    // Check if monotonicity holds in the data
    let monotonicity_satisfied = cell_probs.check_monotonicity();

    let mut warnings = Vec::new();

    if config.monotonicity && !monotonicity_satisfied {
        warnings.push(
            "Monotonicity assumed but may not hold in data. \
             Consider results without monotonicity assumption."
                .to_string(),
        );
    }

    // Compute bounds
    let (ace_lower, ace_upper) = if config.monotonicity {
        compute_bounds_with_monotonicity(&cell_probs)
    } else {
        compute_bounds_without_monotonicity(&cell_probs)
    };

    warnings.extend(instrumental_inequality_warnings(&cell_probs));

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
        n_bootstrap: if config.bootstrap_ci {
            Some(config.n_bootstrap)
        } else {
            None
        },
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
/// ACE_lower = max(p11|1 + p00|0 - 1, p11|0 + p00|1 - 1,
///                 p11|0 - p11|1 - p01|1 - p10|0 - p01|0, p11|1 - p11|0 - p01|0 - p10|1 - p01|1,
///                 -p10|1 - p01|1, -p10|0 - p01|0,
///                 p00|1 - p10|1 - p01|1 - p10|0 - p00|0, p00|0 - p10|0 - p01|0 - p10|1 - p00|1)
/// ACE_upper = min(1 - p10|1 - p01|0, 1 - p10|0 - p01|1,
///                 -p10|0 + p10|1 + p00|1 + p11|0 + p00|0, -p10|1 + p11|1 + p00|1 + p10|0 + p00|0,
///                 p11|1 + p00|1, p11|0 + p00|0,
///                 -p01|1 + p11|1 + p00|1 + p11|0 + p01|0, -p01|0 + p11|0 + p00|0 + p11|1 + p01|1)
/// ```
///
/// (Balke & Pearl 1997; Pearl 2009, eq. 8.14-8.15), equal to the response-type LP.
fn compute_bounds_without_monotonicity(probs: &CellProbabilities) -> (f64, f64) {
    let CellProbabilities {
        p00_z0,
        p01_z0,
        p10_z0,
        p11_z0,
        p00_z1,
        p01_z1,
        p10_z1,
        p11_z1,
        ..
    } = *probs;
    let lower_terms = [
        p11_z1 + p00_z0 - 1.0,
        p11_z0 + p00_z1 - 1.0,
        p11_z0 - p11_z1 - p01_z1 - p10_z0 - p01_z0,
        p11_z1 - p11_z0 - p01_z0 - p10_z1 - p01_z1,
        -p10_z1 - p01_z1,
        -p10_z0 - p01_z0,
        p00_z1 - p10_z1 - p01_z1 - p10_z0 - p00_z0,
        p00_z0 - p10_z0 - p01_z0 - p10_z1 - p00_z1,
    ];
    let upper_terms = [
        1.0 - p10_z1 - p01_z0,
        1.0 - p10_z0 - p01_z1,
        -p10_z0 + p10_z1 + p00_z1 + p11_z0 + p00_z0,
        -p10_z1 + p11_z1 + p00_z1 + p10_z0 + p00_z0,
        p11_z1 + p00_z1,
        p11_z0 + p00_z0,
        -p01_z1 + p11_z1 + p00_z1 + p11_z0 + p01_z0,
        -p01_z0 + p11_z0 + p00_z0 + p11_z1 + p01_z1,
    ];
    let lower = lower_terms
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let upper = upper_terms.iter().copied().fold(f64::INFINITY, f64::min);
    (lower.max(-1.0), upper.min(1.0))
}

/// Compute Balke-Pearl bounds with monotonicity assumption.
///
/// Under monotonicity (no defiers), the bounds tighten to Manski-Robins bounds:
///
/// ```text
/// ACE_lower = p00|0 - p00|1 - p01|1 - p10|1
/// ACE_upper = p00|1 + p01|1 + p11|1 - p01|0 = 1 - p10|1 - p01|0
/// ```
///
/// # Reference
///
/// Robins, J. M. (1989) and Manski, C. F. (1990)
fn compute_bounds_with_monotonicity(probs: &CellProbabilities) -> (f64, f64) {
    let p00_z0 = probs.p00_z0;
    let p01_z0 = probs.p01_z0;
    let p00_z1 = probs.p00_z1;
    let p01_z1 = probs.p01_z1;
    let p10_z1 = probs.p10_z1;
    let p11_z1 = probs.p11_z1;

    // Under monotonicity, bounds simplify
    // These are the Manski-Robins bounds
    let lower = p00_z0 - p00_z1 - p01_z1 - p10_z1; // = p11|1 + p00|0 - 1
    let upper = p00_z1 + p01_z1 + p11_z1 - p01_z0; // = 1 - p10|1 - p01|0

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
fn compute_wald_estimate(
    probs: &CellProbabilities,
    marginals: &MarginalProbabilities,
) -> (f64, f64) {
    let numerator = marginals.p_y1_z1 - marginals.p_y1_z0;
    let denominator = marginals.p_d1_z1 - marginals.p_d1_z0;

    let wald = if denominator.abs() > 1e-10 {
        numerator / denominator
    } else {
        // Weak instrument: denominator near zero
        f64::NAN
    };

    // Standard error by the delta method:
    // Var = Var(num) / den^2 + num^2 Var(den) / den^4 - 2 num Cov(num, den) / den^3

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
        // within-arm covariance of P(Y=1|Z=z) and P(D=1|Z=z): (p11|z - p_y p_d) / n_z
        let cov = (probs.p11_z0 - marginals.p_y1_z0 * marginals.p_d1_z0) / n_z0
            + (probs.p11_z1 - marginals.p_y1_z1 * marginals.p_d1_z1) / n_z1;
        let var = var_numerator / denom2 + numerator * numerator * var_denominator / denom4
            - 2.0 * numerator * cov / (denom2 * denominator);
        var.max(0.0).sqrt()
    } else {
        f64::NAN
    };

    (wald, se)
}

/// Check the bootstrap controls, which only matter when `bootstrap_ci` is set.
fn validate_bootstrap_config(config: &BPBoundsConfig) -> EconResult<()> {
    if config.n_bootstrap == 0 {
        return Err(EconError::InvalidSpecification {
            message: "n_bootstrap must be at least 1 when bootstrap_ci is set (got 0)".to_string(),
        });
    }
    if !(config.alpha.is_finite() && config.alpha > 0.0 && config.alpha < 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "alpha must lie strictly between 0 and 1 (got {}); the confidence level is 1 - alpha",
                config.alpha
            ),
        });
    }
    Ok(())
}

/// The type-7 percentile interval at level `1 - alpha` (R's `quantile(type = 7)`) of
/// bootstrap draws; NaN is refused.
fn bootstrap_interval(draws: &[f64], alpha: f64) -> EconResult<(f64, f64)> {
    // percentile_interval sorts its own copy and refuses NaN draws
    percentile_interval(draws, 1.0 - alpha)
}

/// Pearl's instrumental inequality for binary Z, D and Y: for each `d`,
/// `sum_y max_z P(Y = y, D = d | Z = z) <= 1`. Returns one warning per failing `d` (with
/// a `1e-12` tolerance). A violation means the exclusion restriction or the randomization
/// of Z fails; the Balke-Pearl bounds may then cross, but they need not, so the
/// inequality is tested directly rather than inferred from the bounds.
fn instrumental_inequality_warnings(probs: &CellProbabilities) -> Vec<String> {
    let sums = [
        (
            0,
            probs.p00_z0.max(probs.p00_z1) + probs.p01_z0.max(probs.p01_z1),
        ),
        (
            1,
            probs.p10_z0.max(probs.p10_z1) + probs.p11_z0.max(probs.p11_z1),
        ),
    ];
    sums.iter()
        .filter(|&&(_, sum)| sum > 1.0 + 1e-12)
        .map(|&(d, sum)| {
            format!(
                "the instrumental inequality fails (d = {d}: {sum:.4} > 1): the exclusion \
                 restriction or randomization of Z is violated; the Balke-Pearl bounds are \
                 not valid"
            )
        })
        .collect()
}

/// Compute bootstrap confidence intervals for the bounds (seeded; `seed = None` uses
/// [`DEFAULT_BOOTSTRAP_SEED`]).
fn compute_bootstrap_ci(
    z: &ArrayView1<f64>,
    d: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    config: &BPBoundsConfig,
) -> EconResult<(Option<(f64, f64)>, Option<(f64, f64)>, Option<(f64, f64)>)> {
    let n = z.len();
    let mut rng = StdRng::seed_from_u64(config.seed.unwrap_or(DEFAULT_BOOTSTRAP_SEED));

    let mut lower_bounds =
        resource::try_with_capacity::<f64>(config.n_bootstrap, "bootstrap lower bounds")?;
    let mut upper_bounds =
        resource::try_with_capacity::<f64>(config.n_bootstrap, "bootstrap upper bounds")?;

    // Convert to owned arrays for resampling
    let z_vec: Vec<f64> = z.iter().cloned().collect();
    let d_vec: Vec<f64> = d.iter().cloned().collect();
    let y_vec: Vec<f64> = y.iter().cloned().collect();

    // Resample buffers, overwritten by every replicate
    let mut z_boot = Array1::zeros(n);
    let mut d_boot = Array1::zeros(n);
    let mut y_boot = Array1::zeros(n);

    for _ in 0..config.n_bootstrap {
        resource::checkpoint()?;

        // Bootstrap sample with replacement
        for i in 0..n {
            let idx = rng.gen_range(0..n);
            z_boot[i] = z_vec[idx];
            d_boot[i] = d_vec[idx];
            y_boot[i] = y_vec[idx];
        }

        // Compute bounds for bootstrap sample
        if let Ok(probs) =
            CellProbabilities::from_data(&z_boot.view(), &d_boot.view(), &y_boot.view())
        {
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

    let lower_ci = bootstrap_interval(&lower_bounds, config.alpha)?;
    let upper_ci = bootstrap_interval(&upper_bounds, config.alpha)?;

    // Overall CI: most conservative interval
    let overall_ci = (lower_ci.0, upper_ci.1);

    Ok((Some(lower_ci), Some(upper_ci), Some(overall_ci)))
}

/// Compute Balke-Pearl bounds from pre-computed cell probabilities.
///
/// This function is useful when you have cell probabilities from a contingency
/// table rather than raw data.
pub fn bp_bounds_from_probs(probs: CellProbabilities, monotonicity: bool) -> BPBoundsResult {
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
    warnings.extend(instrumental_inequality_warnings(&probs));

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
    use crate::resource::{Budget, CancelToken, with_budget};
    use ndarray::Array1;

    fn cancelled() -> Budget {
        let token = CancelToken::new();
        token.cancel();
        Budget::new(None, token)
    }

    fn expect_invalid<T>(res: EconResult<T>, needle: &str) {
        match res {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(
                    message.contains(needle),
                    "message {message:?} should name `{needle}`"
                )
            }
            Err(e) => panic!("expected InvalidSpecification naming `{needle}`, got: {e}"),
            Ok(_) => panic!("expected InvalidSpecification naming `{needle}`, got Ok"),
        }
    }

    fn expect_resource_limit<T>(res: EconResult<T>) {
        match res {
            Err(EconError::ResourceLimit(_)) => {}
            Err(e) => panic!("expected ResourceLimit, got: {e}"),
            Ok(_) => panic!("expected ResourceLimit, got Ok"),
        }
    }

    fn expect_cancelled<T>(res: EconResult<T>) {
        match res {
            Err(EconError::Cancelled(_)) => {}
            Err(e) => panic!("expected Cancelled, got: {e}"),
            Ok(_) => panic!("expected Cancelled, got Ok"),
        }
    }

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
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, // 20 Z=0
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, // 20 Z=1
        ]);

        // Treatment: Z=0 has 20% D=1 (4/20), Z=1 has 70% D=1 (14/20)
        let d = Array1::from_vec(vec![
            // Z=0: 4 treated (always-takers)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 1.0, // Z=1: 14 treated (always-takers + compliers)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]);

        // Outcomes: P(Y=1|D=1) ~ 0.6, P(Y=1|D=0) ~ 0.3
        let y = Array1::from_vec(vec![
            // Z=0 group
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // D=0: 30% Y=1
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, // D=0: 30%, D=1: 75%
            // Z=1 group
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, // D=0: 30%, D=1 starts
            1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, // D=1: ~70% Y=1
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

        assert!(
            (sum_z0 - 1.0).abs() < 1e-10,
            "Z=0 probabilities should sum to 1"
        );
        assert!(
            (sum_z1 - 1.0).abs() < 1e-10,
            "Z=1 probabilities should sum to 1"
        );

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
        assert!((-1.0..=1.0).contains(&lower_mono));
        assert!((-1.0..=1.0).contains(&upper_mono));
        assert!(lower_no_mono >= -1.0 && upper_no_mono <= 1.0);

        println!(
            "Bounds without monotonicity: [{:.4}, {:.4}]",
            lower_no_mono, upper_no_mono
        );
        println!(
            "Bounds with monotonicity:    [{:.4}, {:.4}]",
            lower_mono, upper_mono
        );
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
        let expected_wald =
            (marginals.p_y1_z1 - marginals.p_y1_z0) / (marginals.p_d1_z1 - marginals.p_d1_z0);

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
            n_bootstrap: 500, // Smaller for faster test
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
        let z = Array1::from_vec(vec![0.0, 1.0, 2.0]); // Invalid: contains 2
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
            p00_z0: 0.56, // D=0, Y=0 | Z=0
            p01_z0: 0.24, // D=0, Y=1 | Z=0
            p10_z0: 0.08, // D=1, Y=0 | Z=0
            p11_z0: 0.12, // D=1, Y=1 | Z=0
            p00_z1: 0.21, // D=0, Y=0 | Z=1
            p01_z1: 0.09, // D=0, Y=1 | Z=1
            p10_z1: 0.21, // D=1, Y=0 | Z=1
            p11_z1: 0.49, // D=1, Y=1 | Z=1
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
        let z = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let d = z.clone(); // D = Z (perfect compliance)
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.0, 1.0, // Z=0, D=0: P(Y=1) = 0.4
            0.0, 1.0, 1.0, 1.0, 1.0, // Z=1, D=1: P(Y=1) = 0.8
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
        println!(
            "Perfect compliance bounds: [{:.4}, {:.4}]",
            result.ace_lower, result.ace_upper
        );
        println!("Wald: {:.4}", result.wald_estimate);
    }

    fn probs(c: [usize; 8]) -> CellProbabilities {
        let (n0, n1) = (c[..4].iter().sum::<usize>(), c[4..].iter().sum::<usize>());
        let (a, b) = (n0 as f64, n1 as f64);
        CellProbabilities {
            p00_z0: c[0] as f64 / a,
            p01_z0: c[1] as f64 / a,
            p10_z0: c[2] as f64 / a,
            p11_z0: c[3] as f64 / a,
            p00_z1: c[4] as f64 / b,
            p01_z1: c[5] as f64 / b,
            p10_z1: c[6] as f64 / b,
            p11_z1: c[7] as f64 / b,
            n_z0: n0,
            n_z1: n1,
        }
    }

    fn close(a: f64, b: f64, what: &str) {
        assert!(
            (a - b).abs() <= 1e-12 * b.abs().max(1.0),
            "{what}: {a} vs {b}"
        );
    }

    /// Expand a 2x2x2 count table (order as in `probs`) to raw rows and run the raw-data path.
    fn bounds_from_rows(counts: [usize; 8], monotonicity: bool) -> BPBoundsResult {
        let mut rows = Vec::new();
        for (cell, &c) in counts.iter().enumerate() {
            let (zv, dv, yv) = (
                (cell / 4) as f64,
                ((cell / 2) % 2) as f64,
                (cell % 2) as f64,
            );
            rows.extend(std::iter::repeat_n([zv, dv, yv], c));
        }
        let col = |j: usize| Array1::from_iter(rows.iter().map(|r| r[j]));
        let (z, d, y) = (col(0), col(1), col(2));
        let cfg = BPBoundsConfig {
            monotonicity,
            bootstrap_ci: false,
            ..Default::default()
        };
        run_bp_bounds(&z.view(), &d.view(), &y.view(), cfg).unwrap()
    }

    fn warns_inequality(r: &BPBoundsResult) -> bool {
        r.warnings
            .iter()
            .any(|w| w.contains("instrumental inequality fails"))
    }

    /// A table that violates the instrumental inequality (P(Y=0, D=0 | Z=0) = 0.9 and
    /// P(Y=1, D=0 | Z=1) = 0.9 sum to 1.8 > 1) gives crossing bounds, and the result says why.
    /// Fails before: no warning named the cause.
    #[test]
    fn test_bpbounds_crossing_bounds_warn() {
        let counts = [18, 1, 0, 1, 1, 18, 1, 0];
        let free = bp_bounds_from_probs(probs(counts), false);
        assert!(free.ace_lower > free.ace_upper && free.bounds_width < 0.0);
        assert!(warns_inequality(&free), "{:?}", free.warnings);
        let r = bounds_from_rows(counts, false);
        assert!(r.ace_lower > r.ace_upper);
        assert!(warns_inequality(&r), "{:?}", r.warnings);
        // a table satisfying it (vitamin A) carries no such warning
        let ok = bp_bounds_from_probs(probs([74, 11514, 0, 0, 34, 2385, 12, 9663]), false);
        assert!(!warns_inequality(&ok), "{:?}", ok.warnings);
    }

    /// The inequality can fail while the bounds do not cross: z0 = (0, 0, 0, 5),
    /// z1 = (0, 0, 1, 4) gives, for d = 1, max_z P(y=0, d=1 | z) + max_z P(y=1, d=1 | z)
    /// = 0.2 + 1.0 = 1.2 > 1, yet bounds [0.2, 0.6]. The warning must still appear, on both
    /// paths and in both monotonicity modes. Fails before: only crossing bounds warned.
    #[test]
    fn test_bpbounds_inequality_fails_without_crossing() {
        let counts = [0, 0, 0, 5, 0, 0, 1, 4];
        for monotonicity in [false, true] {
            let from_probs = bp_bounds_from_probs(probs(counts), monotonicity);
            let from_rows = bounds_from_rows(counts, monotonicity);
            for r in [&from_probs, &from_rows] {
                assert!(r.ace_lower <= r.ace_upper, "mono {monotonicity}");
                assert!(warns_inequality(r), "mono {monotonicity}: {:?}", r.warnings);
                assert!(
                    r.warnings.iter().any(|w| w.contains("d = 1: 1.2000 > 1")),
                    "{:?}",
                    r.warnings
                );
            }
        }
        let free = bp_bounds_from_probs(probs(counts), false);
        close(free.ace_lower, 0.2, "lower");
        close(free.ace_upper, 0.6, "upper");
    }

    /// Balke-Pearl closed form on two hand-computed tables (bpbounds is not installed;
    /// checked during planning against the exact response-type LP, lpSolve 5.6.23, to
    /// 5.5e-17). Derivation of Table 2 in validation/econometrics/bpbounds.md. Replaces
    /// test_validate_against_r, which compared with no R value.
    #[test]
    fn test_bpbounds_closed_form_tables() {
        for (label, counts, wald, mono_lower) in [
            (
                "vitamin A",
                [74, 11514, 0, 0, 34, 2385, 12, 9663],
                0.0032280386285733,
                -0.1946228482113902,
            ),
            ("table 2", [12, 10, 8, 10, 6, 4, 10, 20], 1.0 / 3.0, -0.2),
        ] {
            let mono = bp_bounds_from_probs(probs(counts), true);
            close(mono.wald_estimate, wald, label);
            close(mono.ace_lower, mono_lower, label);
            assert!(mono.monotonicity_satisfied, "{label}");
            let (lower, upper, mono_upper, wald_se) = match label {
                "vitamin A" => (
                    -0.1946228482113902,
                    0.00539368891445724,
                    0.00539368891445724,
                    0.0011591629284325984,
                ),
                _ => (-0.2, 0.5, 0.5, 0.3683044231992716),
            };
            let free = bp_bounds_from_probs(probs(counts), false);
            close(free.ace_lower, lower, label);
            close(free.ace_upper, upper, label);
            close(mono.ace_upper, mono_upper, label);
            close(mono.wald_se.unwrap(), wald_se, label);
            // never inverted on data the IV model can generate
            assert!(free.bounds_width >= 0.0, "{label}");
        }
    }

    /// The interval is the type-7 percentile interval (R quantile(type = 7)). Fails before:
    /// the index rule floor(alpha/2 B) / ceil((1 - alpha/2) B) - 1 gave 0.1 and 1.0 on draws
    /// 0.1, ..., 1.0 (type 7: 0.1225, 0.9775).
    #[test]
    fn test_bp_bounds_percentile_rule() {
        let draws: Vec<f64> = (1..=10).rev().map(|i| i as f64 / 10.0).collect(); // unsorted on purpose
        let (lo, hi) = bootstrap_interval(&draws, 0.05).unwrap();
        assert!(
            (lo - 0.1225).abs() < 1e-15 && (hi - 0.9775).abs() < 1e-15,
            "{lo} {hi}"
        );
    }

    /// No seed means DEFAULT_BOOTSTRAP_SEED (42), not OS entropy, and the reported intervals
    /// are the type-7 intervals of the replayed replicate bounds.
    #[test]
    fn test_bp_bounds_bootstrap_ci_seeded() {
        let (z, d, y) = create_test_data();
        let run = |seed: Option<u64>| {
            let c = BPBoundsConfig {
                n_bootstrap: 500,
                seed,
                ..Default::default()
            };
            run_bp_bounds(&z.view(), &d.view(), &y.view(), c).unwrap()
        };
        let (a, b) = (run(None), run(Some(DEFAULT_BOOTSTRAP_SEED)));
        assert_eq!(
            (a.ace_lower_ci, a.ace_upper_ci),
            (b.ace_lower_ci, b.ace_upper_ci)
        );
        let mut rng = StdRng::seed_from_u64(DEFAULT_BOOTSTRAP_SEED);
        let n = z.len();
        let (mut lows, mut ups) = (Vec::new(), Vec::new());
        for _ in 0..500 {
            let idx: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
            let pick = |v: &Array1<f64>| Array1::from_iter(idx.iter().map(|&i| v[i]));
            if let Ok(p) =
                CellProbabilities::from_data(&pick(&z).view(), &pick(&d).view(), &pick(&y).view())
            {
                let (l, u) = compute_bounds_without_monotonicity(&p);
                lows.push(l);
                ups.push(u);
            }
        }
        assert_eq!(
            a.ace_lower_ci,
            Some(bootstrap_interval(&lows, 0.05).unwrap())
        );
        assert_eq!(
            a.ace_upper_ci,
            Some(bootstrap_interval(&ups, 0.05).unwrap())
        );
    }

    #[test]
    fn test_bp_bounds_rejects_bad_bootstrap_controls() {
        let (z, d, y) = create_test_data();
        let run = |config: BPBoundsConfig| run_bp_bounds(&z.view(), &d.view(), &y.view(), config);
        expect_invalid(
            run(BPBoundsConfig {
                n_bootstrap: 0,
                seed: Some(1),
                ..Default::default()
            }),
            "n_bootstrap",
        );
        for alpha in [f64::NAN, 0.0, 1.0, -0.5, 2.5] {
            expect_invalid(
                run(BPBoundsConfig {
                    alpha,
                    n_bootstrap: 200,
                    seed: Some(1),
                    ..Default::default()
                }),
                "alpha",
            );
        }
        // without a bootstrap neither field is used
        assert!(
            run(BPBoundsConfig {
                bootstrap_ci: false,
                n_bootstrap: 0,
                alpha: f64::NAN,
                ..Default::default()
            })
            .is_ok()
        );
    }

    #[test]
    fn test_bp_bounds_rejects_unallocatable_bootstrap() {
        let (z, d, y) = create_test_data();
        expect_resource_limit(run_bp_bounds(
            &z.view(),
            &d.view(),
            &y.view(),
            BPBoundsConfig {
                n_bootstrap: usize::MAX / 4,
                seed: Some(1),
                ..Default::default()
            },
        ));
    }

    #[test]
    fn test_bp_bounds_stops_when_cancelled() {
        let (z, d, y) = create_test_data();
        expect_cancelled(with_budget(cancelled(), || {
            run_bp_bounds(
                &z.view(),
                &d.view(),
                &y.view(),
                BPBoundsConfig {
                    n_bootstrap: 200,
                    seed: Some(1),
                    ..Default::default()
                },
            )
        }));
    }
}
