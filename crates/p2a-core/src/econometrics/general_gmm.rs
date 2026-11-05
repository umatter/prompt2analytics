//! General Method of Moments (GMM) estimation.
//!
//! Pure Rust implementation of the generalized method of moments estimator of Hansen (1982)
//! for linear IV moments `E[z (y - x'theta)] = 0` and for the normal-distribution moments
//! `E[x - mu] = E[(x - mu)^2 - s2] = E[(x - mu)^3] = 0`. Every configuration computes what
//! `gmm::gmm` computes for the corresponding call:
//!
//! | p2a configuration | `gmm::gmm` call |
//! |---|---|
//! | `method = TwoStep, vcov = MDS` | `type = "twoStep", vcov = "MDS"` |
//! | `method = Iterative, vcov = MDS` | `type = "iterative", vcov = "MDS", crit = tolerance` |
//! | `method = CUE, vcov = MDS` | `type = "cue", vcov = "MDS"` |
//! | `vcov = IID` | `vcov = "iid"` (two-step equals 2SLS) |
//! | `vcov = HAC` (default: Bartlett, `floor(4 (n/100)^(2/9))` lags) | `vcov = "HAC", kernel = "Bartlett", bw = lags + 1, prewhite = 0` |
//! | `vcov = HAC, prewhite = true` | `.., prewhite = 1` (VAR(1) prewhitening) |
//! | `vcov = Fixed` | `wmatrix = "ident"` (one-step GMM, sandwich variance) |
//! | `weight_matrix = Some(W)` | `weightsMatrix = W` (one-step GMM, sandwich variance) |
//! | `centered = false` | `centeredVcov = FALSE` |
//!
//! - **Two-step**: 2SLS (linear) or the identity-weight minimiser (nonlinear) first, then
//!   `W = Omega(theta_1)^-1`. **Iterative**: weight updates until the relative change in the
//!   coefficients is below `tolerance`. **CUE**: `gbar' Omega(theta)^-1 gbar` minimised by
//!   BFGS from the two-step estimate (for linear IV without prewhitening the objective is an
//!   exact quadratic form with an analytic gradient).
//! - The variance is `(G' Omega(theta)^-1 G)^-1 / n`, `G = d gbar / d theta`, with Omega
//!   re-estimated at the estimate; for a fixed W it is the sandwich
//!   `(G'WG)^-1 G'W Omega W G (G'WG)^-1 / n`, and no J statistic is reported (it is
//!   chi-squared(q - k) only for the optimal W).
//! - Moment covariances are centred by default (gmm's `centeredVcov = TRUE`).
//! - p2a's default HAC weight differs from gmm's default (Quadratic Spectral kernel with the
//!   Andrews (1991) bandwidth and VAR(1) prewhitening), which is not implemented: call
//!   `gmm(.., kernel = "Bartlett", bw = lags + 1, prewhite = 0)` to reproduce p2a's.
//! - Nonlinear (normal-distribution) moments are minimised by Gauss-Newton.
//!
//! # References
//!
//! - Hansen, L.P. (1982). "Large Sample Properties of Generalized Method of
//!   Moments Estimators." Econometrica, 50(4), 1029-1054.
//!
//! - Hansen, L.P., Heaton, J., & Yaron, A. (1996). "Finite-Sample Properties
//!   of Some Alternative GMM Estimators." Journal of Business & Economic
//!   Statistics, 14(3), 262-280.
//!
//! - Andrews, D.W.K., & Monahan, J.C. (1992). "An Improved Heteroskedasticity and
//!   Autocorrelation Consistent Covariance Matrix Estimator." Econometrica, 60(4), 953-966.
//!
//! R equivalent: `gmm::gmm()` (validated against it: `validation/econometrics/general_gmm.md`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{eig_symmetric, matrix_inverse};
use crate::linalg::solve::{LM_QR_TOL, Qr, SYM_RANK_TOL, SingularPolicy, sym_inverse};
use crate::regression::HacKernel;
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes, notes_text, relative_change_vec};
use crate::traits::estimator::{SignificanceLevel, chi_squared_p_value, normal_p_value_two_sided};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// GMM estimation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GmmMethod {
    /// Two-step estimation (Hansen 1982): 2SLS (linear) or the identity-weight minimiser
    /// (nonlinear) first, then the inverse of the moment covariance at that estimate
    #[default]
    TwoStep,
    /// Iterative GMM - iterate until convergence
    Iterative,
    /// Continuously updated estimator (Hansen, Heaton & Yaron 1996): minimises
    /// `gbar' Omega(theta)^-1 gbar` by BFGS from the two-step estimate
    CUE,
    /// One-step GMM with a fixed weighting matrix (`weight_matrix`, or the identity with
    /// `vcov = Fixed`); reported for every fit with a fixed W
    OneStep,
}

impl fmt::Display for GmmMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GmmMethod::TwoStep => write!(f, "Two-Step"),
            GmmMethod::Iterative => write!(f, "Iterative"),
            GmmMethod::CUE => write!(f, "Continuously Updated (CUE)"),
            GmmMethod::OneStep => write!(f, "One-Step (fixed weighting matrix)"),
        }
    }
}

crate::option_enum!(
    GmmMethod,
    crate::options::Choices {
        option: "method",
        values: &[
            (&["twostep", "two_step"], GmmMethod::TwoStep),
            (&["iterative", "iter", "iterated"], GmmMethod::Iterative),
            (&["cue", "continuously_updated"], GmmMethod::CUE),
            (&["onestep", "one_step"], GmmMethod::OneStep),
        ],
    },
    no_display
);

/// Variance-covariance estimation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GmmVcov {
    /// HAC (heteroskedasticity and autocorrelation consistent), lag-count kernel weights
    #[default]
    HAC,
    /// i.i.d. errors: `sigma^2 Z'Z / n` for linear IV (two-step = 2SLS)
    IID,
    /// The identity weighting matrix (one-step GMM) with the MDS moment covariance in the
    /// sandwich variance
    Fixed,
    /// Heteroskedasticity-robust (martingale difference) moment covariance
    MDS,
}

crate::option_enum!(
    GmmVcov,
    crate::options::Choices {
        option: "vcov",
        values: &[
            (&["hac"], GmmVcov::HAC),
            (&["iid", "homoskedastic"], GmmVcov::IID),
            (&["fixed"], GmmVcov::Fixed),
            (&["mds", "robust", "heteroskedastic"], GmmVcov::MDS),
        ],
    }
);

/// The weighting matrix a fit used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GmmWeighting {
    /// The inverse of the estimated moment covariance
    #[default]
    Optimal,
    /// The identity
    Identity,
    /// A matrix the caller supplied
    User,
}

/// Configuration for general GMM estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralGmmConfig {
    /// Estimation method
    pub method: GmmMethod,
    /// Variance-covariance type
    pub vcov: GmmVcov,
    /// HAC kernel for weighting matrix
    pub kernel: HacKernel,
    /// HAC bandwidth (None for automatic).
    /// With HAC weighting and linear IV moments, must be less than the
    /// number of observations.
    pub bandwidth: Option<usize>,
    /// Convergence tolerance.
    /// With linear IV moments, must be finite and greater than 0.
    pub tolerance: f64,
    /// Maximum iterations (for iterative/CUE).
    /// With linear IV moments, must be at least 1.
    pub max_iter: usize,
    /// Whether to prewhiten for HAC
    pub prewhite: bool,
    /// Centre the moments before their covariance is estimated
    #[serde(default = "default_centered")]
    pub centered: bool,
    /// A fixed q x q weighting matrix (symmetric positive definite): one-step GMM with
    /// the sandwich variance and no J statistic (gmm's `weightsMatrix`). `None` with
    /// `vcov = Fixed` is the identity (gmm's `wmatrix = "ident"`).
    #[serde(default, with = "opt_matrix")]
    pub weight_matrix: Option<Array2<f64>>,
}

/// `Option<Array2<f64>>` as an optional list of rows (square).
mod opt_matrix {
    use ndarray::Array2;
    use serde::de::Error;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(m: &Option<Array2<f64>>, s: S) -> Result<S::Ok, S::Error> {
        m.as_ref()
            .map(|m| {
                m.outer_iter()
                    .map(|r| r.to_vec())
                    .collect::<Vec<Vec<f64>>>()
            })
            .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Array2<f64>>, D::Error> {
        let Some(rows) = Option::<Vec<Vec<f64>>>::deserialize(d)? else {
            return Ok(None);
        };
        let q = rows.len();
        if rows.iter().any(|r| r.len() != q) {
            return Err(D::Error::custom(
                "weight_matrix must be a square list of rows",
            ));
        }
        Array2::from_shape_vec((q, q), rows.into_iter().flatten().collect())
            .map(Some)
            .map_err(D::Error::custom)
    }
}

fn default_centered() -> bool {
    true
}

impl Default for GeneralGmmConfig {
    fn default() -> Self {
        Self {
            method: GmmMethod::TwoStep,
            vcov: GmmVcov::HAC,
            kernel: HacKernel::Bartlett,
            bandwidth: None,
            tolerance: 1e-7,
            max_iter: 100,
            prewhite: false,
            centered: true,
            weight_matrix: None,
        }
    }
}

/// Predefined moment condition specification.
#[derive(Debug, Clone)]
pub enum MomentCondition {
    /// Linear IV: E[z(y - xβ)] = 0
    /// y = outcome, X = regressors, Z = instruments
    LinearIV {
        y: Array1<f64>,
        x: Array2<f64>,
        z: Array2<f64>,
    },
    /// Linear model with overidentification
    /// Same as LinearIV but X ⊂ Z
    LinearOveridentified {
        y: Array1<f64>,
        x: Array2<f64>,
        z: Array2<f64>,
    },
    /// Normal distribution moments: E[x-μ] = 0, E[(x-μ)²-σ²] = 0
    NormalDistribution { data: Array1<f64> },
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from general GMM estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralGmmResult {
    /// Parameter estimates
    pub coefficients: Vec<f64>,
    /// Parameter names
    pub names: Vec<String>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// t-statistics
    pub t_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// Covariance matrix of estimates
    pub vcov: Vec<Vec<f64>>,
    /// J-statistic (Hansen test for overidentifying restrictions)
    pub j_stat: f64,
    /// J-test p-value
    pub j_pvalue: f64,
    /// Degrees of freedom for J-test
    pub j_df: usize,
    /// Final objective function value
    pub objective: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of parameters
    pub n_params: usize,
    /// Number of moment conditions
    pub n_moments: usize,
    /// Convergence achieved
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Estimation method used
    pub method: GmmMethod,
    /// Rows dropped because of missing values (`run_gmm_iv`; the array-level
    /// estimators drop nothing).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
    /// Moment covariance type
    #[serde(default)]
    pub vcov_type: GmmVcov,
    /// Weighting matrix used
    #[serde(default)]
    pub weighting: GmmWeighting,
    /// Whether the moments were centred
    #[serde(default)]
    pub centered: bool,
    /// HAC lags with non-zero weight
    #[serde(default)]
    pub hac_lags: Option<usize>,
    /// Convergence record
    #[serde(default)]
    pub convergence: Option<crate::traits::convergence::Convergence>,
    /// Warnings
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ResultNotes for GeneralGmmResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        self.convergence.clone()
    }
}

impl fmt::Display for GeneralGmmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Generalized Method of Moments (GMM) Estimation")?;
        writeln!(f, "===============================================")?;
        writeln!(f)?;
        writeln!(f, "Method: {}", self.method)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Parameters: {}", self.n_params)?;
        writeln!(f, "Moment conditions: {}", self.n_moments)?;
        writeln!(
            f,
            "Overidentification: {} (q - k = {})",
            if self.n_moments > self.n_params {
                "Yes"
            } else {
                "No (just-identified)"
            },
            self.n_moments.saturating_sub(self.n_params)
        )?;
        let weighting = match self.weighting {
            GmmWeighting::Optimal => "optimal (inverse of the moment covariance)",
            GmmWeighting::Identity => "identity",
            GmmWeighting::User => "user-supplied",
        };
        writeln!(f, "Weighting matrix: {weighting}")?;
        let centring = if self.centered {
            "centred"
        } else {
            "uncentred"
        };
        match (self.vcov_type, self.hac_lags) {
            (GmmVcov::HAC, Some(lags)) => {
                writeln!(f, "Moment covariance: HAC ({lags} lags), {centring}")?
            }
            (GmmVcov::IID, _) => writeln!(f, "Moment covariance: iid (sigma^2 Z'Z / n)")?,
            _ => writeln!(
                f,
                "Moment covariance: heteroskedasticity-robust (MDS), {centring}"
            )?,
        }
        writeln!(f)?;
        writeln!(f, "Coefficients:")?;
        writeln!(
            f,
            "{:>15} {:>12} {:>12} {:>10} {:>10}",
            "Parameter", "Estimate", "Std.Err", "z-stat", "P>|z|"
        )?;
        writeln!(f, "{}", "-".repeat(65))?;

        for i in 0..self.n_params {
            let sig = SignificanceLevel::from_p_value(self.p_values[i]);
            writeln!(
                f,
                "{:>15} {:>12.6} {:>12.6} {:>10.3} {:>10.4}{}",
                self.names[i],
                self.coefficients[i],
                self.std_errors[i],
                self.t_stats[i],
                self.p_values[i],
                sig.stars()
            )?;
        }
        writeln!(f, "{}", "-".repeat(65))?;

        if self.n_moments > self.n_params {
            writeln!(f)?;
            if self.j_stat.is_nan() {
                writeln!(
                    f,
                    "J-Test: not reported for a fixed weighting matrix (n g' Omega^-1 g is \
                     chi-squared(q - k) only for the optimal W)"
                )?;
            } else {
                writeln!(f, "J-Test for Overidentifying Restrictions:")?;
                writeln!(f, "  J-statistic: {:.4} (df = {})", self.j_stat, self.j_df)?;
                writeln!(f, "  p-value: {:.4}", self.j_pvalue)?;
                if self.j_pvalue < 0.05 {
                    writeln!(f, "  Instruments may be invalid (p < 0.05)")?;
                } else {
                    writeln!(
                        f,
                        "  Cannot reject validity of overidentifying restrictions"
                    )?;
                }
            }
        }

        let notes = notes_text(self);
        if !notes.is_empty() {
            writeln!(f)?;
            write!(f, "{notes}")?;
        }
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Estimation Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Run GMM estimation with predefined moment conditions.
///
/// # Arguments
///
/// * `moments` - Predefined moment condition specification
/// * `config` - Configuration options
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::general_gmm::{run_general_gmm, MomentCondition, GeneralGmmConfig};
///
/// // IV estimation: y = X β + ε, with instruments Z
/// let moments = MomentCondition::LinearIV { y, x, z };
/// let result = run_general_gmm(moments, GeneralGmmConfig::default())?;
/// ```
pub fn run_general_gmm(
    moments: MomentCondition,
    config: GeneralGmmConfig,
) -> EconResult<GeneralGmmResult> {
    match moments {
        MomentCondition::LinearIV { y, x, z } => estimate_linear_iv(&y, &x, &z, config),
        MomentCondition::LinearOveridentified { y, x, z } => estimate_linear_iv(&y, &x, &z, config),
        MomentCondition::NormalDistribution { data } => estimate_normal_distribution(&data, config),
    }
}

/// Convenience function for linear IV GMM from dataset.
///
/// # Arguments
///
/// * `dataset` - The dataset
/// * `y_col` - Outcome variable
/// * `x_cols` - Regressors (endogenous and exogenous)
/// * `z_cols` - Instruments (should include exogenous regressors)
/// * `config` - Configuration options
///
/// Rows with a missing value (null or NaN) in `y`, a regressor or an instrument are
/// dropped from every column and reported in `missing`; an infinite value is refused
/// (`NonFiniteValues`), a text column is `NonNumericColumn`.
pub fn run_gmm_iv(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    z_cols: &[&str],
    config: Option<GeneralGmmConfig>,
) -> EconResult<GeneralGmmResult> {
    let config = config.unwrap_or_default();

    // y, X and Z on the rows complete in all of them (incomplete rows are dropped
    // and reported)
    let cols = ColumnRequest::new(dataset.df())
        .numeric(y_col)
        .numerics(x_cols)
        .numerics(z_cols)
        .na(NaPolicy::Omit)
        .extract()?;
    let y = cols.numeric_array(y_col);

    // X and Z with intercept
    let x_dm = cols.matrix(x_cols, true)?;
    let z = cols.matrix(z_cols, true)?.data;

    let moments = MomentCondition::LinearIV { y, x: x_dm.data, z };
    let mut result = run_general_gmm(moments, config)?;

    result.names = x_dm.column_names;
    result.missing = cols.missing().clone();

    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// The GMM engine: moment models, moment covariance, weighted steps
// ═══════════════════════════════════════════════════════════════════════════════

/// p2a's default HAC lag count, `floor(4 (n / 100)^(2 / 9))` (Newey-West).
fn newey_west_lags(n: usize) -> usize {
    (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize
}

/// `(m + m') / 2`: products of symmetric factors are symmetric only up to rounding.
fn symmetrize(m: &Array2<f64>) -> Array2<f64> {
    (m + &m.t()) * 0.5
}

/// A set of moment conditions `E[g_i(theta)] = 0`.
trait MomentModel {
    /// Observations.
    fn n(&self) -> usize;
    /// Moment conditions.
    fn q(&self) -> usize;
    /// Parameters.
    fn k(&self) -> usize;
    /// `g_i(theta)` row by row (n x q).
    fn moments(&self, theta: ArrayView1<f64>) -> EconResult<Array2<f64>>;
    /// `gbar(theta)`, the column means of [`MomentModel::moments`].
    fn gbar(&self, theta: ArrayView1<f64>) -> EconResult<Array1<f64>>;
    /// `G = d gbar / d theta` (q x k).
    fn jacobian(&self, theta: ArrayView1<f64>) -> Array2<f64>;
    /// The estimate every weighted fit starts from (and the first-step estimate).
    fn first_step(
        &self,
        config: &GeneralGmmConfig,
    ) -> EconResult<(Array1<f64>, Option<Convergence>)>;
    /// The start of a fit with a fixed weighting matrix.
    fn start(&self) -> Array1<f64>;
    /// The moment covariance under `vcov = IID` when it differs from the MDS one.
    fn iid_omega(&self, _theta: ArrayView1<f64>) -> Option<Array2<f64>> {
        None
    }
    /// The linear IV model, which has closed-form weighted steps.
    fn as_linear(&self) -> Option<&LinearIv<'_>> {
        None
    }
}

/// Linear IV moments `g_i = z_i (y_i - x_i' theta)`.
struct LinearIv<'a> {
    y: &'a Array1<f64>,
    x: &'a Array2<f64>,
    z: &'a Array2<f64>,
    /// `Z'X / n` (q x k).
    zx: Array2<f64>,
    /// `Z'y / n`.
    zy: Array1<f64>,
    /// `Z'Z / n`, computed on first use (a fixed-W fit with MDS or HAC meat never needs it).
    zz: std::cell::OnceCell<Array2<f64>>,
}

impl<'a> LinearIv<'a> {
    fn new(y: &'a Array1<f64>, x: &'a Array2<f64>, z: &'a Array2<f64>) -> Self {
        let n = y.len() as f64;
        LinearIv {
            y,
            x,
            z,
            zx: z.t().dot(x) / n,
            zy: z.t().dot(y) / n,
            zz: std::cell::OnceCell::new(),
        }
    }

    fn zz(&self) -> &Array2<f64> {
        self.zz
            .get_or_init(|| symmetrize(&(self.z.t().dot(self.z) / self.y.len() as f64)))
    }

    fn residuals(&self, theta: ArrayView1<f64>) -> Array1<f64> {
        self.y - &self.x.dot(&theta)
    }

    /// `argmin gbar' W gbar = (A' W A)^-1 A' W Z'y / n` with `A = Z'X / n`.
    fn weighted_solve(&self, w: &Array2<f64>) -> EconResult<Array1<f64>> {
        let aw = self.zx.t().dot(w);
        let m = symmetrize(&aw.dot(&self.zx));
        let inv =
            sym_inverse(m.view(), SingularPolicy::Error, "X'Z W Z'X").map_err(|e| match e {
                EconError::SingularMatrix { .. } => EconError::InvalidSpecification {
                    message: "the regressors are not identified by the instruments: X'Z W Z'X is \
                          singular (a regressor is collinear with others or not instrumented)"
                        .to_string(),
                },
                other => other,
            })?;
        Ok(inv.inverse.dot(&aw.dot(&self.zy)))
    }
}

impl MomentModel for LinearIv<'_> {
    fn n(&self) -> usize {
        self.y.len()
    }
    fn q(&self) -> usize {
        self.z.ncols()
    }
    fn k(&self) -> usize {
        self.x.ncols()
    }
    fn moments(&self, theta: ArrayView1<f64>) -> EconResult<Array2<f64>> {
        let e = self.residuals(theta);
        Ok(self.z * &e.insert_axis(Axis(1)))
    }
    fn gbar(&self, theta: ArrayView1<f64>) -> EconResult<Array1<f64>> {
        Ok(&self.zy - &self.zx.dot(&theta))
    }
    fn jacobian(&self, _theta: ArrayView1<f64>) -> Array2<f64> {
        -&self.zx
    }
    /// 2SLS: the weighted step with `W = (Z'Z / n)^-1` (gmm's `.tetlin(type = "2sls")`).
    fn first_step(
        &self,
        _config: &GeneralGmmConfig,
    ) -> EconResult<(Array1<f64>, Option<Convergence>)> {
        let zz_inv = match sym_inverse(self.zz().view(), SingularPolicy::Error, "Z'Z / n") {
            Ok(inv) => inv.inverse,
            Err(EconError::SingularMatrix { .. }) => {
                let qr = Qr::decompose(self.z.view(), LM_QR_TOL)?;
                let aliased: Vec<String> = qr
                    .aliased()
                    .iter()
                    .map(|j| format!("instrument column {}", j + 1))
                    .collect();
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "the instruments are collinear ({} is a linear combination of the others)",
                        if aliased.is_empty() {
                            "a column".to_string()
                        } else {
                            aliased.join(", ")
                        }
                    ),
                });
            }
            Err(e) => return Err(e),
        };
        Ok((self.weighted_solve(&zz_inv)?, None))
    }
    fn start(&self) -> Array1<f64> {
        Array1::zeros(self.k())
    }
    /// gmm's `vcov = "iid"`: `sigma^2 Z'Z / n` with `sigma^2 = mean((e - mean e)^2)`.
    fn iid_omega(&self, theta: ArrayView1<f64>) -> Option<Array2<f64>> {
        let e = self.residuals(theta);
        let mean = e.mean().unwrap_or(0.0);
        let sigma2 = e.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / e.len() as f64;
        Some(self.zz() * sigma2)
    }
    fn as_linear(&self) -> Option<&LinearIv<'_>> {
        Some(self)
    }
}

/// An estimated moment covariance `Omega` and the HAC lags it used.
struct MomentCov {
    omega: Array2<f64>,
    lags: Option<usize>,
}

/// `u` with its column means subtracted.
fn centre_columns(u: &mut Array2<f64>) {
    if u.nrows() == 0 {
        return;
    }
    let means = u.mean_axis(Axis(0)).expect("non-empty");
    *u -= &means;
}

/// The moment covariance `Omega(theta)` of `config.vcov` (sandwich's `meatHAC` with
/// `adjust = FALSE` for HAC; `Fixed` uses the MDS covariance).
fn moment_covariance(
    model: &dyn MomentModel,
    theta: ArrayView1<f64>,
    config: &GeneralGmmConfig,
) -> EconResult<MomentCov> {
    let n = model.n();
    if config.vcov == GmmVcov::IID
        && let Some(omega) = model.iid_omega(theta)
    {
        return Ok(MomentCov { omega, lags: None });
    }
    let mut u = model.moments(theta)?;
    if config.centered {
        centre_columns(&mut u);
    }
    match config.vcov {
        GmmVcov::HAC => hac_covariance(u, config, n),
        GmmVcov::IID | GmmVcov::MDS | GmmVcov::Fixed => Ok(MomentCov {
            omega: symmetrize(&(u.t().dot(&u) / n as f64)),
            lags: None,
        }),
    }
}

/// The HAC moment covariance with the lag-count rule: weights `kernel.weight(j, b)` for
/// `j = 1..=min(b, m - 1)`, `b = bandwidth` or `floor(4 (n/100)^(2/9))`; divided by the
/// original `n`.
fn hac_covariance(u: Array2<f64>, config: &GeneralGmmConfig, n: usize) -> EconResult<MomentCov> {
    let (eps, d) = if config.prewhite {
        let (eps, d) = var1_prewhiten(&u)?;
        (eps, Some(d))
    } else {
        (u, None)
    };
    let m = eps.nrows();
    let b = config.bandwidth.unwrap_or_else(|| newey_west_lags(n));
    let lags = b.min(m.saturating_sub(1));
    let mut s = eps.t().dot(&eps);
    for j in 1..=lags {
        resource::checkpoint()?;
        let w = config.kernel.weight(j, b);
        if w.abs() < 1e-15 {
            continue;
        }
        let gamma = eps.slice(s![j.., ..]).t().dot(&eps.slice(s![..m - j, ..]));
        s.scaled_add(w, &gamma);
        s.scaled_add(w, &gamma.t());
    }
    if let Some(d) = d {
        s = d.dot(&s).dot(&d.t());
    }
    Ok(MomentCov {
        omega: symmetrize(&(s / n as f64)),
        lags: Some(lags),
    })
}

/// A user weighting matrix must be q x q, finite, symmetric and positive definite.
fn validate_weight_matrix(w: &Array2<f64>, q: usize) -> EconResult<()> {
    let invalid = |why: String| {
        Err(EconError::InvalidSpecification {
            message: format!("weight_matrix {why}"),
        })
    };
    if w.dim() != (q, q) {
        return invalid(format!(
            "must be {q} x {q} (one row and column per moment condition), got {} x {}",
            w.nrows(),
            w.ncols()
        ));
    }
    if w.iter().any(|v| !v.is_finite()) {
        return invalid("contains non-finite values".to_string());
    }
    let big = w.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    let asym = w
        .indexed_iter()
        .fold(0.0_f64, |a, ((i, j), v)| a.max((v - w[[j, i]]).abs()));
    if asym > 1e-10 * big.max(1.0) {
        return invalid(format!("must be symmetric (max |W - W'| = {asym:.3e})"));
    }
    let (eig, _) = eig_symmetric(&w.view()).map_err(|e| EconError::InvalidSpecification {
        message: format!("weight_matrix: eigenvalues failed ({e})"),
    })?;
    let min = eig.iter().fold(f64::INFINITY, |a, &v| a.min(v));
    let max = eig.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    if min.is_nan() || min <= SYM_RANK_TOL * q as f64 * max {
        return invalid(format!(
            "must be positive definite (smallest eigenvalue {min:.3e})"
        ));
    }
    Ok(())
}

/// The inverse of a moment covariance (the efficient weighting matrix).
fn weight_from(omega: &Array2<f64>) -> EconResult<Array2<f64>> {
    Ok(sym_inverse(
        omega.view(),
        SingularPolicy::Error,
        "moment covariance matrix Omega",
    )?
    .inverse)
}

/// `argmin gbar(theta)' W gbar(theta)` from `start`: closed form for linear IV,
/// Gauss-Newton otherwise.
fn weighted_step(
    model: &dyn MomentModel,
    w: &Array2<f64>,
    start: &Array1<f64>,
    config: &GeneralGmmConfig,
) -> EconResult<(Array1<f64>, Option<Convergence>)> {
    match model.as_linear() {
        Some(lin) => Ok((lin.weighted_solve(w)?, None)),
        None => gauss_newton(model, w, start, config),
    }
}

/// The two-step, iterated or one-step estimate, before the variance.
struct Estimate {
    theta: Array1<f64>,
    /// The weighting matrix of the last estimation step (J and the objective use it).
    w: Array2<f64>,
    method: GmmMethod,
    weighting: GmmWeighting,
    convergence: Option<Convergence>,
    iterations: usize,
    warnings: Vec<String>,
}

fn estimate(model: &dyn MomentModel, config: &GeneralGmmConfig) -> EconResult<Estimate> {
    let mut warnings = Vec::new();
    let fixed = match (&config.weight_matrix, config.vcov) {
        (Some(w), _) => Some((w.clone(), GmmWeighting::User)),
        (None, GmmVcov::Fixed) => Some((Array2::eye(model.q()), GmmWeighting::Identity)),
        (None, _) => None,
    };
    if let Some((w, weighting)) = fixed {
        let (theta, conv) = weighted_step(model, &w, &model.start(), config)?;
        if let Some(note) = conv.as_ref().and_then(|c| c.note("Gauss-Newton step")) {
            warnings.push(note);
        }
        return Ok(Estimate {
            theta,
            w,
            method: GmmMethod::OneStep,
            weighting,
            iterations: conv.as_ref().map_or(1, |c| c.iterations),
            convergence: conv,
            warnings,
        });
    }
    let (theta1, conv1) = model.first_step(config)?;
    if let Some(note) = conv1
        .as_ref()
        .and_then(|c| c.note("Gauss-Newton first step"))
    {
        warnings.push(note);
    }
    match config.method {
        GmmMethod::TwoStep | GmmMethod::OneStep => {
            let w = weight_from(&moment_covariance(model, theta1.view(), config)?.omega)?;
            let (theta, conv) = weighted_step(model, &w, &theta1, config)?;
            if let Some(note) = conv.as_ref().and_then(|c| c.note("Gauss-Newton step")) {
                warnings.push(note);
            }
            Ok(Estimate {
                theta,
                w,
                method: GmmMethod::TwoStep,
                weighting: GmmWeighting::Optimal,
                iterations: 2,
                convergence: conv,
                warnings,
            })
        }
        GmmMethod::CUE => {
            let w1 = weight_from(&moment_covariance(model, theta1.view(), config)?.omega)?;
            let (two_step, conv) = weighted_step(model, &w1, &theta1, config)?;
            if let Some(note) = conv.as_ref().and_then(|c| c.note("Gauss-Newton step")) {
                warnings.push(note);
            }
            let (theta, c) = cue_bfgs(model, config, two_step)?;
            if let Some(note) = c.note("CUE (BFGS)") {
                warnings.push(note);
            }
            let w = weight_from(&moment_covariance(model, theta.view(), config)?.omega)?;
            Ok(Estimate {
                theta,
                w,
                method: GmmMethod::CUE,
                weighting: GmmWeighting::Optimal,
                iterations: c.iterations,
                convergence: Some(c),
                warnings,
            })
        }
        GmmMethod::Iterative => {
            let mut theta = theta1;
            let mut w = Array2::eye(model.q());
            let mut change = f64::NAN;
            let mut converged = false;
            let mut iterations = 0;
            for it in 1..=config.max_iter {
                resource::checkpoint()?;
                iterations = it;
                w = weight_from(&moment_covariance(model, theta.view(), config)?.omega)?;
                let (next, conv) = weighted_step(model, &w, &theta, config)?;
                if let Some(note) = conv.as_ref().and_then(|c| c.note("Gauss-Newton step")) {
                    warnings.push(note);
                }
                change = relative_change_vec(
                    next.as_slice().unwrap_or(&next.to_vec()),
                    theta.as_slice().unwrap_or(&theta.to_vec()),
                );
                theta = next;
                if change < config.tolerance {
                    converged = true;
                    break;
                }
            }
            let c = Convergence::new(
                converged,
                iterations,
                config.max_iter,
                "relative change in coefficients",
                change,
                config.tolerance,
            );
            if let Some(note) = c.note("Iterative GMM") {
                warnings.push(note);
            }
            Ok(Estimate {
                theta,
                w,
                method: GmmMethod::Iterative,
                weighting: GmmWeighting::Optimal,
                iterations,
                convergence: Some(c),
                warnings,
            })
        }
    }
}

/// Variance, J statistic and the reported fields of a fit.
fn finish_fit(
    model: &dyn MomentModel,
    est: Estimate,
    config: &GeneralGmmConfig,
    names: Vec<String>,
) -> EconResult<GeneralGmmResult> {
    let (n, k, q) = (model.n(), model.k(), model.q());
    let nf = n as f64;
    let theta = est.theta;
    let cov = moment_covariance(model, theta.view(), config)?;
    let g = model.jacobian(theta.view());
    let fixed = est.method == GmmMethod::OneStep;
    let vcov_mat = if fixed {
        // sandwich: T1 Omega T1' / n with T1 = (G'WG)^-1 G'W (gmm's FinRes)
        let gw = g.t().dot(&est.w);
        let bread = sym_inverse(
            symmetrize(&gw.dot(&g)).view(),
            SingularPolicy::Error,
            "G' W G",
        )?
        .inverse;
        let t1 = bread.dot(&gw);
        symmetrize(&(t1.dot(&cov.omega).dot(&t1.t()) / nf))
    } else {
        let oinv = weight_from(&cov.omega)?;
        let m = symmetrize(&g.t().dot(&oinv).dot(&g));
        sym_inverse(m.view(), SingularPolicy::Error, "G' Omega^-1 G")?.inverse / nf
    };
    let gbar = model.gbar(theta.view())?;
    let objective = nf * gbar.dot(&est.w.dot(&gbar));
    let j_df = q - k;
    let (j_stat, j_pvalue) = if fixed {
        (f64::NAN, f64::NAN)
    } else if q > k {
        (objective, chi_squared_p_value(objective, j_df as f64))
    } else {
        (0.0, 1.0)
    };
    let mut warnings = est.warnings;
    if q == k {
        warnings.push(
            "the model is just identified (q = k): two-step, iterative and CUE coincide"
                .to_string(),
        );
    }
    let std_errors: Vec<f64> = vcov_mat.diag().iter().map(|&v| v.max(0.0).sqrt()).collect();
    let coefficients = theta.to_vec();
    let t_stats: Vec<f64> = coefficients
        .iter()
        .zip(&std_errors)
        .map(|(&b, &se)| b / se)
        .collect();
    let p_values = t_stats
        .iter()
        .map(|&t| normal_p_value_two_sided(t))
        .collect();
    let vcov = vcov_mat.outer_iter().map(|r| r.to_vec()).collect();
    let converged = est.convergence.as_ref().is_none_or(|c| c.converged);
    Ok(GeneralGmmResult {
        coefficients,
        names,
        std_errors,
        t_stats,
        p_values,
        vcov,
        j_stat,
        j_pvalue,
        j_df,
        objective,
        n_obs: n,
        n_params: k,
        n_moments: q,
        converged,
        iterations: est.iterations,
        method: est.method,
        missing: MissingSummary::none(n),
        vcov_type: config.vcov,
        weighting: est.weighting,
        centered: config.centered,
        hac_lags: cov.lags,
        convergence: est.convergence,
        warnings,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Linear IV GMM
// ═══════════════════════════════════════════════════════════════════════════════

fn estimate_linear_iv(
    y: &Array1<f64>,
    x: &Array2<f64>,
    z: &Array2<f64>,
    config: GeneralGmmConfig,
) -> EconResult<GeneralGmmResult> {
    let n = y.len();
    let k = x.ncols(); // number of parameters
    let q = z.ncols(); // number of moment conditions

    if z.nrows() != n || x.nrows() != n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Dimension mismatch: n={}, X rows={}, Z rows={}",
                n,
                x.nrows(),
                z.nrows()
            ),
        });
    }
    if q < k {
        return Err(EconError::InvalidSpecification {
            message: format!("Underidentified: {} instruments < {} parameters", q, k),
        });
    }
    for (what, finite) in [
        ("y", y.iter().all(|v| v.is_finite())),
        ("x", x.iter().all(|v| v.is_finite())),
        ("z", z.iter().all(|v| v.is_finite())),
    ] {
        if !finite {
            return Err(EconError::InvalidSpecification {
                message: format!("{what} contains non-finite values"),
            });
        }
    }
    validate_config(&config, n, q)?;
    let model = LinearIv::new(y, x, z);
    let est = estimate(&model, &config)?;
    let names = (0..k).map(|i| format!("beta_{}", i + 1)).collect();
    finish_fit(&model, est, &config, names)
}

/// Check the iteration controls and, for HAC weighting, the bandwidth against
/// the sample size.
fn validate_config(config: &GeneralGmmConfig, n: usize, q: usize) -> EconResult<()> {
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "tolerance must be a finite number greater than 0 (got {})",
                config.tolerance
            ),
        });
    }
    if config.max_iter == 0 {
        return Err(EconError::InvalidSpecification {
            message: "max_iter must be at least 1 (got 0)".to_string(),
        });
    }
    if config.vcov == GmmVcov::HAC
        && let Some(b) = config.bandwidth
        && b >= n
    {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "bandwidth = {} must be less than the number of observations ({})",
                b, n
            ),
        });
    }
    let fixed_w = config.weight_matrix.is_some() || config.vcov == GmmVcov::Fixed;
    if fixed_w && matches!(config.method, GmmMethod::Iterative | GmmMethod::CUE) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "method = {} needs the optimal weighting matrix; a fixed weighting matrix \
                 (weight_matrix, or vcov = fixed) gives one-step GMM (method = two_step or one_step)",
                config.method
            ),
        });
    }
    if config.method == GmmMethod::OneStep && !fixed_w {
        return Err(EconError::InvalidSpecification {
            message: "method = one_step needs a fixed weighting matrix: set weight_matrix \
                      (or vcov = fixed for the identity)"
                .to_string(),
        });
    }
    if let Some(w) = &config.weight_matrix {
        validate_weight_matrix(w, q)?;
    }
    if config.vcov == GmmVcov::HAC && config.prewhite && n.saturating_sub(1) <= q {
        return Err(EconError::InsufficientData {
            required: q + 2,
            provided: n,
            context: format!(
                "VAR(1) prewhitening of {q} moment conditions needs more than {q} lagged rows"
            ),
        });
    }
    Ok(())
}

/// VAR(1) prewhitening of the estimating functions (Andrews and Monahan 1992; sandwich's
/// `prewhite = 1`): `u_t = A u_{t-1} + eps_t` by OLS without intercept on t = 2..n; returns
/// the residuals and `D = (I - A)^-1`, so that the meat is `D S(eps) D'`.
fn var1_prewhiten(u: &Array2<f64>) -> EconResult<(Array2<f64>, Array2<f64>)> {
    let (n, q) = u.dim();
    let lagged = u.slice(s![..n - 1, ..]);
    let current = u.slice(s![1.., ..]);
    let ll = symmetrize(&lagged.t().dot(&lagged));
    let ll_inv = sym_inverse(
        ll.view(),
        SingularPolicy::Error,
        "VAR(1) prewhitening cross-product",
    )?;
    // B = A' (q x q): current = lagged B + eps
    let b = ll_inv.inverse.dot(&lagged.t().dot(&current));
    let eps = &current - &lagged.dot(&b);
    let i_minus_a = Array2::<f64>::eye(q) - b.t();
    let d = matrix_inverse(&i_minus_a.view()).map_err(|_| EconError::InvalidSpecification {
        message: "prewhitening: I - A is singular (the VAR(1) of the moments has a unit root)"
            .to_string(),
    })?;
    Ok((eps, d))
}

/// Gauss-Newton on `gbar' W gbar` from `start`: `theta <- theta - lambda (G'WG)^-1 G'W gbar`,
/// `lambda` halved from 1 while the objective increases (down to 2^-30); stops when the
/// relative change is below [`GN_TOLERANCE`] or at `config.max_iter`.
fn gauss_newton(
    model: &dyn MomentModel,
    w: &Array2<f64>,
    start: &Array1<f64>,
    config: &GeneralGmmConfig,
) -> EconResult<(Array1<f64>, Option<Convergence>)> {
    let objective = |theta: &Array1<f64>| -> EconResult<f64> {
        let g = model.gbar(theta.view())?;
        Ok(g.dot(&w.dot(&g)))
    };
    let mut theta = start.clone();
    let mut q_cur = objective(&theta)?;
    let mut change = f64::NAN;
    let mut converged = false;
    let mut iterations = 0;
    for it in 1..=config.max_iter {
        resource::checkpoint()?;
        iterations = it;
        let g = model.jacobian(theta.view());
        let gbar = model.gbar(theta.view())?;
        let gw = g.t().dot(w);
        let m = symmetrize(&gw.dot(&g));
        let step = sym_inverse(m.view(), SingularPolicy::Error, "G' W G")?
            .inverse
            .dot(&gw.dot(&gbar));
        let mut lambda = 1.0;
        let (next, q_next) = loop {
            let trial = &theta - &(&step * lambda);
            let q_trial = objective(&trial)?;
            if q_trial <= q_cur || lambda <= 2f64.powi(-30) {
                break (trial, q_trial);
            }
            lambda *= 0.5;
        };
        if q_next.is_nan() || q_next > q_cur {
            // no step decreases the objective: the minimum up to rounding
            change = 0.0;
            converged = true;
            break;
        }
        change = relative_change_vec(&next.to_vec(), &theta.to_vec());
        theta = next;
        q_cur = q_next;
        if change < GN_TOLERANCE {
            converged = true;
            break;
        }
    }
    Ok((
        theta,
        Some(Convergence::new(
            converged,
            iterations,
            config.max_iter,
            "relative change in coefficients (Gauss-Newton)",
            change,
            GN_TOLERANCE,
        )),
    ))
}

/// Gauss-Newton stopping rule (relative change in coefficients).
const GN_TOLERANCE: f64 = 1e-12;

// ═══════════════════════════════════════════════════════════════════════════════
// Continuously updated GMM (CUE)
// ═══════════════════════════════════════════════════════════════════════════════

/// The CUE objective `Q(theta) = gbar' Omega(theta)^-1 gbar` and its gradient.
enum CueObjective<'m> {
    /// Linear IV without prewhitening: `Omega` is a quadratic form in `c = (1, -theta)`,
    /// so every evaluation costs `O(k^2 q^2 + q^3)` after a one-off meat of the stacked
    /// estimating functions `B_t = [z_t y_t, z_t x_t1, .., z_t x_tk]`.
    Quadratic {
        q: usize,
        /// `bbar_l` (columns of `B` averaged), one per block `l = 0..=k`.
        bbar: Vec<Array1<f64>>,
        /// `Gamma`: the meat of `B` (q(k+1) square), or for iid the centred cross-product
        /// of `[y, X]` divided by n ((k+1) square) and `Z'Z / n`.
        gamma: Array2<f64>,
        iid_zz: Option<Array2<f64>>,
    },
    /// Any other model: `Omega` re-estimated at every theta, central-difference gradient.
    Direct {
        model: &'m dyn MomentModel,
        config: &'m GeneralGmmConfig,
    },
}

impl<'m> CueObjective<'m> {
    fn new(model: &'m dyn MomentModel, config: &'m GeneralGmmConfig) -> EconResult<Self> {
        let Some(lin) = model.as_linear() else {
            return Ok(CueObjective::Direct { model, config });
        };
        if config.vcov == GmmVcov::HAC && config.prewhite {
            return Ok(CueObjective::Direct { model, config });
        }
        let (n, q, k) = (lin.n(), lin.q(), lin.k());
        let mut bbar = Vec::with_capacity(k + 1);
        bbar.push(lin.zy.clone());
        for j in 0..k {
            bbar.push(lin.zx.column(j).to_owned());
        }
        if config.vcov == GmmVcov::IID {
            let mut yx = resource::try_zeros2(n, k + 1, "CUE [y, X]")?;
            yx.column_mut(0).assign(lin.y);
            yx.slice_mut(s![.., 1..]).assign(lin.x);
            centre_columns(&mut yx);
            let c = symmetrize(&(yx.t().dot(&yx) / n as f64));
            return Ok(CueObjective::Quadratic {
                q,
                bbar,
                gamma: c,
                iid_zz: Some(lin.zz().clone()),
            });
        }
        let width = resource::checked_len(&[q, k + 1], "CUE stacked moments")?;
        let mut b = resource::try_zeros2(n, width, "CUE stacked moments")?;
        for l in 0..=k {
            let col = if l == 0 {
                lin.y.view()
            } else {
                lin.x.column(l - 1)
            };
            let mut block = b.slice_mut(s![.., l * q..(l + 1) * q]);
            block.assign(lin.z);
            block *= &col.insert_axis(Axis(1));
        }
        if config.centered {
            centre_columns(&mut b);
        }
        let gamma = match config.vcov {
            GmmVcov::HAC => hac_covariance(b, config, n)?.omega,
            _ => symmetrize(&(b.t().dot(&b) / n as f64)),
        };
        Ok(CueObjective::Quadratic {
            q,
            bbar,
            gamma,
            iid_zz: None,
        })
    }

    /// `(Q, gbar, Omega)` at theta; `Q = +inf` when Omega is singular.
    fn omega_and_gbar(&self, theta: &Array1<f64>) -> EconResult<(Array1<f64>, Array2<f64>)> {
        match self {
            CueObjective::Direct { model, config } => Ok((
                model.gbar(theta.view())?,
                moment_covariance(*model, theta.view(), config)?.omega,
            )),
            CueObjective::Quadratic {
                q,
                bbar,
                gamma,
                iid_zz,
            } => {
                let c = cue_c(theta);
                let mut g = Array1::<f64>::zeros(*q);
                for (cl, bl) in c.iter().zip(bbar) {
                    g.scaled_add(*cl, bl);
                }
                let omega = match iid_zz {
                    Some(zz) => zz * c.dot(&gamma.dot(&c)),
                    None => {
                        let mut o = Array2::<f64>::zeros((*q, *q));
                        for (l, cl) in c.iter().enumerate() {
                            for (m, cm) in c.iter().enumerate() {
                                o.scaled_add(
                                    cl * cm,
                                    &gamma.slice(s![l * q..(l + 1) * q, m * q..(m + 1) * q]),
                                );
                            }
                        }
                        symmetrize(&o)
                    }
                };
                Ok((g, omega))
            }
        }
    }

    fn value(&self, theta: &Array1<f64>) -> EconResult<f64> {
        resource::checkpoint()?;
        let (g, omega) = self.omega_and_gbar(theta)?;
        match sym_inverse(
            omega.view(),
            SingularPolicy::Error,
            "moment covariance matrix Omega",
        ) {
            Ok(inv) => Ok(g.dot(&inv.inverse.dot(&g))),
            Err(EconError::SingularMatrix { .. }) | Err(EconError::InvalidSpecification { .. }) => {
                Ok(f64::INFINITY)
            }
            Err(e) => Err(e),
        }
    }

    fn gradient(&self, theta: &Array1<f64>) -> EconResult<Array1<f64>> {
        match self {
            CueObjective::Direct { .. } => {
                let scale = theta.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
                let mut grad = Array1::zeros(theta.len());
                for j in 0..theta.len() {
                    let h = if scale == 0.0 {
                        1e-5
                    } else {
                        1e-5 * theta[j].abs().max(1e-2 * scale)
                    };
                    let mut up = theta.clone();
                    up[j] += h;
                    let mut down = theta.clone();
                    down[j] -= h;
                    grad[j] = (self.value(&up)? - self.value(&down)?) / (2.0 * h);
                }
                Ok(grad)
            }
            CueObjective::Quadratic {
                q,
                bbar,
                gamma,
                iid_zz,
            } => {
                let (g, omega) = self.omega_and_gbar(theta)?;
                let a = weight_from(&omega)?.dot(&g);
                let c = cue_c(theta);
                let k = theta.len();
                let mut grad = Array1::zeros(k);
                match iid_zz {
                    Some(zz) => {
                        let cc = gamma.dot(&c);
                        let aza = a.dot(&zz.dot(&a));
                        for j in 0..k {
                            let l = j + 1;
                            let dq_dc = 2.0 * bbar[l].dot(&a) - 2.0 * cc[l] * aza;
                            grad[j] = -dq_dc;
                        }
                    }
                    None => {
                        for j in 0..k {
                            let l = j + 1;
                            let mut quad = 0.0;
                            for (m, cm) in c.iter().enumerate() {
                                let blm = gamma.slice(s![l * q..(l + 1) * q, m * q..(m + 1) * q]);
                                let bml = gamma.slice(s![m * q..(m + 1) * q, l * q..(l + 1) * q]);
                                quad += cm * (a.dot(&blm.dot(&a)) + a.dot(&bml.dot(&a)));
                            }
                            grad[j] = -(2.0 * bbar[l].dot(&a) - quad);
                        }
                    }
                }
                Ok(grad)
            }
        }
    }
}

/// `c = (1, -theta)`: the linear IV moments are `g_t = B_t c`.
fn cue_c(theta: &Array1<f64>) -> Array1<f64> {
    let mut c = Array1::zeros(theta.len() + 1);
    c[0] = 1.0;
    for (j, t) in theta.iter().enumerate() {
        c[j + 1] = -t;
    }
    c
}

/// Minimise the CUE objective by BFGS from `start` (the two-step estimate).
fn cue_bfgs(
    model: &dyn MomentModel,
    config: &GeneralGmmConfig,
    start: Array1<f64>,
) -> EconResult<(Array1<f64>, Convergence)> {
    let obj = CueObjective::new(model, config)?;
    let k = start.len();
    let mut theta = start;
    // H0 = (2 G' Omega(theta0)^-1 G)^-1
    let g_jac = model.jacobian(theta.view());
    let omega0 = moment_covariance(model, theta.view(), config)?.omega;
    let h0m = symmetrize(&(g_jac.t().dot(&weight_from(&omega0)?).dot(&g_jac) * 2.0));
    let h0 = sym_inverse(h0m.view(), SingularPolicy::Error, "CUE initial Hessian")?.inverse;
    let mut h = h0.clone();
    let mut q_cur = obj.value(&theta)?;
    let mut grad = obj.gradient(&theta)?;
    let mut change = f64::NAN;
    let mut converged = false;
    let mut iterations = 0;
    for it in 1..=config.max_iter {
        resource::checkpoint()?;
        iterations = it;
        let mut p = -h.dot(&grad);
        let mut slope = grad.dot(&p);
        if slope >= 0.0 {
            h = h0.clone();
            p = -h.dot(&grad);
            slope = grad.dot(&p);
        }
        let mut t = 1.0;
        let mut accepted = None;
        while t >= 2f64.powi(-40) {
            let trial = &theta + &(&p * t);
            let q_trial = obj.value(&trial)?;
            if q_trial <= q_cur + 1e-4 * t * slope {
                accepted = Some((trial, q_trial));
                break;
            }
            t *= 0.5;
        }
        let Some((next, q_next)) = accepted else {
            // no decrease: converged when the predicted decrease is at rounding level
            converged = -slope <= 1e-12 * q_cur.max(f64::MIN_POSITIVE);
            break;
        };
        change = relative_change_vec(&next.to_vec(), &theta.to_vec());
        let grad_next = obj.gradient(&next)?;
        let s_vec = &next - &theta;
        let y_vec = &grad_next - &grad;
        let sy = s_vec.dot(&y_vec);
        let (ns, ny) = (s_vec.dot(&s_vec).sqrt(), y_vec.dot(&y_vec).sqrt());
        if sy > 1e-12 * ns * ny {
            let rho = 1.0 / sy;
            let eye = Array2::<f64>::eye(k);
            let left = &eye - &(outer(&s_vec, &y_vec) * rho);
            let right = &eye - &(outer(&y_vec, &s_vec) * rho);
            h = left.dot(&h).dot(&right) + outer(&s_vec, &s_vec) * rho;
        }
        theta = next;
        q_cur = q_next;
        grad = grad_next;
        if change < config.tolerance {
            converged = true;
            break;
        }
    }
    Ok((
        theta,
        Convergence::new(
            converged,
            iterations,
            config.max_iter,
            "relative change in coefficients (BFGS on the CUE objective)",
            change,
            config.tolerance,
        ),
    ))
}

fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    Array2::from_shape_fn((a.len(), b.len()), |(i, j)| a[i] * b[j])
}

// ═══════════════════════════════════════════════════════════════════════════════
// Normal Distribution GMM
// ═══════════════════════════════════════════════════════════════════════════════

/// Normal-distribution moments `g_i = (x_i - mu, (x_i - mu)^2 - s2, (x_i - mu)^3)`.
struct NormalMoments<'a> {
    x: ArrayView1<'a, f64>,
}

impl MomentModel for NormalMoments<'_> {
    fn n(&self) -> usize {
        self.x.len()
    }
    fn q(&self) -> usize {
        3
    }
    fn k(&self) -> usize {
        2
    }
    fn moments(&self, theta: ArrayView1<f64>) -> EconResult<Array2<f64>> {
        let n = self.x.len();
        let mut u = resource::try_zeros2(n, 3, "normal-distribution moments")?;
        for (i, &v) in self.x.iter().enumerate() {
            let e = v - theta[0];
            u[[i, 0]] = e;
            u[[i, 1]] = e * e - theta[1];
            u[[i, 2]] = e * e * e;
        }
        Ok(u)
    }
    fn gbar(&self, theta: ArrayView1<f64>) -> EconResult<Array1<f64>> {
        let n = self.x.len() as f64;
        let (mut g1, mut g2, mut g3) = (0.0, 0.0, 0.0);
        for &v in self.x {
            let e = v - theta[0];
            g1 += e;
            g2 += e * e - theta[1];
            g3 += e * e * e;
        }
        Ok(Array1::from(vec![g1 / n, g2 / n, g3 / n]))
    }
    fn jacobian(&self, theta: ArrayView1<f64>) -> Array2<f64> {
        let n = self.x.len() as f64;
        let (mut m1, mut m2) = (0.0, 0.0);
        for &v in self.x {
            let e = v - theta[0];
            m1 += e;
            m2 += e * e;
        }
        let (m1, m2) = (m1 / n, m2 / n);
        ndarray::array![[-1.0, 0.0], [-2.0 * m1, -1.0], [-3.0 * m2, 0.0]]
    }
    /// gmm's nonlinear `twoStep` first step: the identity-weight minimiser from the
    /// method-of-moments start.
    fn first_step(
        &self,
        config: &GeneralGmmConfig,
    ) -> EconResult<(Array1<f64>, Option<Convergence>)> {
        gauss_newton(self, &Array2::eye(3), &self.start(), config)
    }
    fn start(&self) -> Array1<f64> {
        let n = self.x.len() as f64;
        let mean = self.x.sum() / n;
        let var = self.x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        Array1::from(vec![mean, var])
    }
}

fn estimate_normal_distribution(
    data: &Array1<f64>,
    config: GeneralGmmConfig,
) -> EconResult<GeneralGmmResult> {
    let n = data.len();
    if n < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n,
            context: "Normal distribution GMM requires at least 3 observations".to_string(),
        });
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "data contains non-finite values".to_string(),
        });
    }
    let model = NormalMoments { x: data.view() };
    let var = model.start()[1];
    if var.is_nan() || var <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "data has no variation (variance 0); the normal moments are not identified"
                .to_string(),
        });
    }
    validate_config(&config, n, 3)?;
    let est = estimate(&model, &config)?;
    finish_fit(
        &model,
        est,
        &config,
        vec!["mu".to_string(), "sigma_sq".to_string()],
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::ols::na_fixtures::{nan_vec, repo_dataset, with_derived};
    use crate::resource::{Budget, CancelToken, with_budget};
    use ndarray::Array1;
    use p2a_validation::{Check, Tol};

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

    fn expect_cancelled<T>(res: EconResult<T>) {
        match res {
            Err(EconError::Cancelled(_)) => {}
            Err(e) => panic!("expected Cancelled, got: {e}"),
            Ok(_) => panic!("expected Cancelled, got Ok"),
        }
    }

    /// The data of case `gmm_linear_iv`, for property tests that compare no R value.
    fn gmm_iv() -> Dataset {
        repo_dataset("validation/datasets/generated/gmm_iv.csv")
    }

    const X_COLS: [&str; 3] = ["x1", "x2", "w"];
    const Z_COLS: [&str; 4] = ["z1", "z2", "z3", "w"];

    fn cfg(method: GmmMethod, vcov: GmmVcov) -> GeneralGmmConfig {
        GeneralGmmConfig {
            method,
            vcov,
            tolerance: 1e-12,
            max_iter: 1000,
            ..Default::default()
        }
    }

    fn fit(ds: &Dataset, c: GeneralGmmConfig) -> GeneralGmmResult {
        run_gmm_iv(ds, "y", &X_COLS, &Z_COLS, Some(c)).unwrap()
    }

    fn pin(
        check: &mut Check<'_>,
        tag: &str,
        r: &GeneralGmmResult,
        coef: Tol,
        se: Tol,
        with_j: bool,
    ) {
        check
            .named(&format!("{tag}_coef"), &r.names, &r.coefficients, coef)
            .named(&format!("{tag}_se"), &r.names, &r.std_errors, se);
        if with_j {
            check.scalar(&format!("{tag}_J"), r.j_stat, se).scalar(
                &format!("{tag}_J_p"),
                r.j_pvalue,
                Tol::new(se.rel, 1e-12),
            );
        }
    }

    /// Case `gmm_linear_iv`: gmm(type = "twoStep", vcov = "MDS"), centredVcov TRUE and FALSE. The
    /// variance is (G' Omega(theta)^-1 G)^-1 / n with G = -Z'X / n and Omega re-evaluated at the
    /// estimate. Fails before: the code returned (X'Z W Z'X)^-1 / n, so every SE was n times too
    /// small (intercept 8.3328492522612659e-05 against 0.0416335537028179; ratio 0.0020015 on all
    /// four), and the weight was uncentred (coefficients up to 2.9e-8 off).
    #[test]
    fn test_validate_gmm_twostep() {
        let (case, ds) = crate::test_support::load_case("gmm_linear_iv");
        let n = case.scalar("n") as usize;
        let r = fit(&ds, cfg(GmmMethod::TwoStep, GmmVcov::MDS));
        let mut c = cfg(GmmMethod::TwoStep, GmmVcov::MDS);
        c.centered = false;
        let u = fit(&ds, c);
        let mut check = Check::new(&case);
        pin(
            &mut check,
            "twostep_mds",
            &r,
            Tol::coef(n),
            Tol::se(n),
            true,
        );
        pin(
            &mut check,
            "twostep_mds_uncentered",
            &u,
            Tol::coef(n),
            Tol::se(n),
            true,
        );
        check.finish();
        assert_eq!((r.n_obs, r.n_params, r.n_moments, r.j_df), (500, 4, 5, 1));
        assert_eq!(
            (r.method, r.weighting, r.vcov_type, r.centered),
            (
                GmmMethod::TwoStep,
                GmmWeighting::Optimal,
                GmmVcov::MDS,
                true
            )
        );
        assert!(r.convergence.is_none() && r.warnings.is_empty());
        // centring is an option R distinguishes (x1: 2.8e-8)
        assert!((r.coefficients[1] - u.coefficients[1]).abs() > 1e-8);
    }

    /// gmm(vcov = "iid"): W = (sigma^2 Z'Z / n)^-1 with sigma^2 = mean((e - mean e)^2), so the
    /// two-step estimate is 2SLS. Fails before: "IID" summed z z' e^2 (intercept
    /// 0.99852862276382393 against 0.9987404878125472).
    #[test]
    fn test_validate_gmm_iid() {
        let (case, ds) = crate::test_support::load_case("gmm_linear_iv");
        let n = case.scalar("n") as usize;
        let r = fit(&ds, cfg(GmmMethod::TwoStep, GmmVcov::IID));
        let mut check = Check::new(&case);
        pin(
            &mut check,
            "twostep_iid",
            &r,
            Tol::coef(n),
            Tol::se(n),
            true,
        );
        check.named(
            "iid_equals_2sls_coef",
            &r.names,
            &r.coefficients,
            Tol::coef(n),
        );
        check.finish();
        let mds = fit(&ds, cfg(GmmMethod::TwoStep, GmmVcov::MDS));
        assert!((r.coefficients[1] - mds.coefficients[1]).abs() > 1e-4); // R: 0.94234 vs 0.94425
    }

    /// gmm(type = "iterative", vcov = "MDS", crit = 1e-12): weight updates from 2SLS until the
    /// relative change is below 1e-12 (the planning transcription took 5 updates; the changes
    /// were 1.91e-3, 7.10e-7, 7.80e-10, 1.49e-12, 2.7e-15). J uses the last update's weight.
    /// Iterated GMM to crit 1e-12 on both sides (planning transcription 1.2e-15), so
    /// Tol::iterative with the README values at n = 500. Fails before: SEs n times too small;
    /// uncentred weights; `convergence` was None.
    #[test]
    fn test_validate_gmm_iterative() {
        let (case, ds) = crate::test_support::load_case("gmm_linear_iv");
        let r = fit(&ds, cfg(GmmMethod::Iterative, GmmVcov::MDS));
        let mut check = Check::new(&case);
        pin(
            &mut check,
            "iter_mds",
            &r,
            Tol::iterative(1e-8),
            Tol::iterative(1e-6),
            true,
        );
        check.finish();
        let c = r
            .convergence
            .clone()
            .expect("iterative GMM reports convergence");
        assert!(c.converged && r.converged);
        assert_eq!(r.iterations, c.iterations);
        assert!(c.final_change < 1e-12);
        assert_eq!(r.method, GmmMethod::Iterative);
    }

    /// R10 scale test: y x 1e-6 and y x 1e6 give coefficients x 1e-6 / x 1e6 and the same
    /// number of weight updates at the default tolerance (the transcription gives 3 / 3 / 3).
    /// Fails before: the absolute rule ||delta beta|| < 1e-7 stopped after 1 update at 1e-6 and
    /// needed 5 at 1e6. CUE is equivariant too.
    #[test]
    fn test_gmm_iterative_scale_free() {
        let ds = gmm_iv();
        let fit_scaled = |factor: f64, method: GmmMethod| {
            let s = with_derived(&ds, "ys", "y", |v| v * factor);
            let c = GeneralGmmConfig {
                method,
                vcov: GmmVcov::MDS,
                ..Default::default()
            };
            run_gmm_iv(&s, "ys", &X_COLS, &Z_COLS, Some(c)).unwrap()
        };
        let base = fit_scaled(1.0, GmmMethod::Iterative);
        for factor in [1e-6, 1e6] {
            let r = fit_scaled(factor, GmmMethod::Iterative);
            assert_eq!(r.iterations, base.iterations, "x {factor}");
            for (a, b) in r.coefficients.iter().zip(&base.coefficients) {
                assert!(
                    (a / factor / b - 1.0).abs() < 1e-9,
                    "x {factor}: {a} vs {b}"
                );
            }
            assert!((r.j_stat / base.j_stat - 1.0).abs() < 1e-9);
        }
        let base = fit_scaled(1.0, GmmMethod::CUE);
        for factor in [1e-6, 1e6] {
            let r = fit_scaled(factor, GmmMethod::CUE);
            assert!(
                (r.iterations as i64 - base.iterations as i64).abs() <= 1,
                "CUE x {factor}: {} vs {}",
                r.iterations,
                base.iterations
            );
            for (a, b) in r.coefficients.iter().zip(&base.coefficients) {
                assert!(
                    (a / factor / b - 1.0).abs() < 1e-6,
                    "CUE x {factor}: {a} vs {b}"
                );
            }
        }
    }

    /// HAC weights. p2a's default (Bartlett, floor(4 (n/100)^(2/9)) = 5 lags, weights 1 - j/6,
    /// no prewhitening) is gmm(kernel = "Bartlett", bw = 6, prewhite = 0); with
    /// `prewhite = true` it is gmm(.., prewhite = 1) (VAR(1) prewhitening of the estimating
    /// functions). Fails before: SEs n times too small; `prewhite` was never read.
    #[test]
    fn test_validate_gmm_hac() {
        let (case, ds) = crate::test_support::load_case("gmm_linear_iv");
        let n = case.scalar("n") as usize;
        let b = fit(&ds, GeneralGmmConfig::default());
        let pw = fit(
            &ds,
            GeneralGmmConfig {
                prewhite: true,
                ..Default::default()
            },
        );
        let mut check = Check::new(&case);
        pin(
            &mut check,
            "twostep_hac_bartlett",
            &b,
            Tol::coef(n),
            Tol::se(n),
            true,
        );
        pin(
            &mut check,
            "twostep_hac_bartlett_pw",
            &pw,
            Tol::coef(n),
            Tol::se(n),
            true,
        );
        check.scalar(
            "hac_nw_lags",
            b.hac_lags.expect("lag count") as f64,
            Tol::exact(),
        );
        // deviation ECON-68: twostep_hac, twostep_hac_bw (gmm's default QS + bwAndrews +
        // prewhite) not implemented
        check.finish();
        assert_eq!(b.hac_lags, Some(5));
        // prewhitening is an option R distinguishes
        assert!((pw.std_errors[1] / b.std_errors[1] - 1.0).abs() > 1e-6);
    }

    /// gmm(type = "cue", vcov = "MDS" | HAC Bartlett with bw = 6, prewhite = 0) with
    /// Nelder-Mead at reltol 1e-15. BFGS to relative change 1e-12 against gmm's Nelder-Mead at
    /// reltol 1e-15; planning gap 1.3e-10 (MDS), 6.3e-10 (HAC), so Tol::iterative with the
    /// README values, and p2a's objective may not exceed R's. Fails before: CUE ran iterative
    /// GMM (x1 0.94424605 against 0.94420560, 4.3e-5 relative).
    #[test]
    fn test_validate_gmm_cue() {
        let (case, ds) = crate::test_support::load_case("gmm_linear_iv");
        let r = fit(&ds, cfg(GmmMethod::CUE, GmmVcov::MDS));
        let h = fit(&ds, cfg(GmmMethod::CUE, GmmVcov::HAC));
        let mut check = Check::new(&case);
        pin(
            &mut check,
            "cue_mds",
            &r,
            Tol::iterative(1e-8),
            Tol::iterative(1e-6),
            true,
        );
        pin(
            &mut check,
            "cue_hac_bartlett",
            &h,
            Tol::iterative(1e-8),
            Tol::iterative(1e-6),
            true,
        );
        // deviation ECON-68: cue_hac, cue_hac_bw (gmm's default HAC) not implemented
        check.finish();
        assert_eq!(r.method, GmmMethod::CUE);
        assert!(r.convergence.as_ref().is_some_and(|c| c.converged));
        assert!(h.convergence.as_ref().is_some_and(|c| c.converged));
        assert!(r.j_stat <= case.scalar("cue_mds_J") * (1.0 + 1e-10));
        assert!(h.j_stat <= case.scalar("cue_hac_bartlett_J") * (1.0 + 1e-10));
        // CUE and iterated GMM differ, as in R
        let it = fit(&ds, cfg(GmmMethod::Iterative, GmmVcov::MDS));
        assert!((r.coefficients[1] / it.coefficients[1] - 1.0).abs() > 1e-6);
    }

    /// The linear-IV CUE objective as a quadratic form in (1, -theta) and its analytic
    /// gradient equal the direct evaluation (Omega re-estimated at theta) and central
    /// differences. Reads the case's data; pins no R value.
    #[test]
    fn test_gmm_cue_quadratic_form_matches_direct() {
        let ds = gmm_iv();
        let y = Array1::from(nan_vec(&ds, "y"));
        let n = y.len();
        let xc: Vec<Vec<f64>> = X_COLS.iter().map(|c| nan_vec(&ds, c)).collect();
        let zc: Vec<Vec<f64>> = Z_COLS.iter().map(|c| nan_vec(&ds, c)).collect();
        let x = Array2::from_shape_fn((n, 4), |(i, j)| if j == 0 { 1.0 } else { xc[j - 1][i] });
        let z = Array2::from_shape_fn((n, 5), |(i, j)| if j == 0 { 1.0 } else { zc[j - 1][i] });
        let model = LinearIv::new(&y, &x, &z);
        let configs = [
            cfg(GmmMethod::CUE, GmmVcov::IID),
            cfg(GmmMethod::CUE, GmmVcov::MDS),
            GeneralGmmConfig {
                centered: false,
                ..cfg(GmmMethod::CUE, GmmVcov::MDS)
            },
            cfg(GmmMethod::CUE, GmmVcov::HAC),
        ];
        let close = |a: f64, b: f64, tol: f64, what: &str| {
            assert!(
                (a - b).abs() <= tol * b.abs().max(1e-300),
                "{what}: {a} vs {b}"
            );
        };
        for config in &configs {
            let (tsls, _) = model.first_step(config).unwrap();
            let w = weight_from(
                &moment_covariance(&model, tsls.view(), config)
                    .unwrap()
                    .omega,
            )
            .unwrap();
            let two = model.weighted_solve(&w).unwrap();
            let shifted = two.mapv(|v| v + 0.1);
            let quad = CueObjective::new(&model, config).unwrap();
            assert!(matches!(quad, CueObjective::Quadratic { .. }));
            let direct = CueObjective::Direct {
                model: &model,
                config,
            };
            let mut rows = Vec::new();
            for theta in [&tsls, &two, &shifted] {
                close(
                    quad.value(theta).unwrap(),
                    direct.value(theta).unwrap(),
                    1e-10,
                    "Q",
                );
                rows.push((
                    quad.gradient(theta).unwrap(),
                    direct.gradient(theta).unwrap(),
                ));
            }
            // central differences carry O(h^2) truncation error, and the gradient vanishes
            // near the two-step estimate: compare on the largest gradient's scale
            let scale = rows
                .iter()
                .flat_map(|(a, _)| a.iter())
                .fold(0.0_f64, |m, v| m.max(v.abs()));
            for (ga, gd) in &rows {
                for (a, d) in ga.iter().zip(gd) {
                    assert!(
                        (a - d).abs() <= 1e-6 * scale,
                        "gradient {a} vs {d} ({config:?})"
                    );
                }
            }
        }
    }

    /// One-step GMM with a fixed W and the sandwich variance T1 Omega T1' / n,
    /// T1 = (G'WG)^-1 G'W (gmm FinRes): gmm(wmatrix = "ident") through vcov = Fixed and through
    /// weight_matrix = I; gmm(weightsMatrix = (Z'Z/n)^-1), whose estimate is 2SLS. gmm prints
    /// n g' Omega^-1 g (ident_mds_J, fixedw_mds_J) as a J statistic; it is not chi2(q - k) unless
    /// W is optimal, so p2a reports none (NaN; deviation ECON-68). Fails before: Fixed ignored any
    /// matrix, and its variance was (X'Z Z'X)^-1 / n (intercept SE 9.0129038162678113e-05
    /// against 0.04167092215444331).
    #[test]
    fn test_validate_gmm_ident() {
        let (case, ds) = crate::test_support::load_case("gmm_linear_iv");
        let n = case.scalar("n") as usize;
        let ident = fit(&ds, cfg(GmmMethod::TwoStep, GmmVcov::Fixed));
        let mut c = cfg(GmmMethod::TwoStep, GmmVcov::MDS);
        c.weight_matrix = Some(Array2::eye(5));
        let user_eye = fit(&ds, c.clone());
        let cols: Vec<Vec<f64>> = Z_COLS.iter().map(|z| nan_vec(&ds, z)).collect();
        let z = Array2::from_shape_fn((n, 5), |(i, j)| if j == 0 { 1.0 } else { cols[j - 1][i] });
        let zz_n = symmetrize(&(z.t().dot(&z) / n as f64));
        c.weight_matrix = Some(
            sym_inverse(zz_n.view(), SingularPolicy::Error, "Z'Z/n")
                .unwrap()
                .inverse,
        );
        let fixed = fit(&ds, c);
        let mut check = Check::new(&case);
        pin(
            &mut check,
            "ident_mds",
            &ident,
            Tol::coef(n),
            Tol::se(n),
            false,
        );
        pin(
            &mut check,
            "ident_mds",
            &user_eye,
            Tol::coef(n),
            Tol::se(n),
            false,
        );
        pin(
            &mut check,
            "fixedw_mds",
            &fixed,
            Tol::coef(n),
            Tol::se(n),
            false,
        );
        check.finish();
        for r in [&ident, &user_eye, &fixed] {
            assert_eq!((r.method, r.j_df), (GmmMethod::OneStep, 1));
            assert!(r.j_stat.is_nan() && r.j_pvalue.is_nan());
        }
        assert_eq!(
            (ident.weighting, user_eye.weighting, fixed.weighting),
            (
                GmmWeighting::Identity,
                GmmWeighting::User,
                GmmWeighting::User
            )
        );
        assert!(format!("{ident}").contains("J-Test: not reported"));
    }

    #[test]
    fn test_gmm_rejects_bad_weight_matrix_and_combinations() {
        let ds = gmm_iv();
        let run = |c: GeneralGmmConfig| run_gmm_iv(&ds, "y", &X_COLS, &Z_COLS, Some(c));
        let with_w = |w: Array2<f64>| GeneralGmmConfig {
            weight_matrix: Some(w),
            ..Default::default()
        };
        expect_invalid(run(with_w(Array2::eye(4))), "weight_matrix"); // not q x q
        let mut asym = Array2::eye(5);
        asym[[0, 1]] = 0.5;
        expect_invalid(run(with_w(asym)), "weight_matrix"); // not symmetric
        let mut indef = Array2::eye(5);
        indef[[2, 2]] = -1.0;
        expect_invalid(run(with_w(indef)), "weight_matrix"); // not positive definite
        let mut nan = Array2::eye(5);
        nan[[0, 0]] = f64::NAN;
        expect_invalid(run(with_w(nan)), "weight_matrix");
        let mut c = with_w(Array2::eye(5));
        c.method = GmmMethod::CUE;
        expect_invalid(run(c), "method");
        let c = GeneralGmmConfig {
            vcov: GmmVcov::Fixed,
            method: GmmMethod::Iterative,
            ..Default::default()
        };
        expect_invalid(run(c), "method");
        let c = GeneralGmmConfig {
            method: GmmMethod::OneStep,
            ..Default::default()
        };
        expect_invalid(run(c), "weight_matrix");
    }

    #[test]
    fn test_gmm_config_serde_round_trip() {
        let c = GeneralGmmConfig {
            prewhite: true,
            weight_matrix: Some(ndarray::array![[2.0, 0.5], [0.5, 1.0]]),
            ..Default::default()
        };
        let back: GeneralGmmConfig =
            serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back.weight_matrix, c.weight_matrix);
        assert_eq!((back.centered, back.prewhite), (true, true));
        let old: GeneralGmmConfig = serde_json::from_str(
            r#"{"method":"TwoStep","vcov":"HAC","kernel":"Bartlett","bandwidth":null,"tolerance":1e-7,"max_iter":100,"prewhite":false}"#,
        )
        .unwrap();
        assert!(old.centered && old.weight_matrix.is_none());
    }

    /// Case `gmm_linear_iv`, normal moments on x = w: gmm(g, w, t0 = method of moments, gradv,
    /// type = "twoStep" | "iterative", vcov = "MDS") with BFGS at reltol 1e-15 (the tight optimum).
    /// Gauss-Newton against R's BFGS at reltol 1e-15; planning gap 1.8e-11 / 6.9e-10, so
    /// Tol::iterative with the README values. Fails before: the estimates were the method of
    /// moments (mu 0.056278592574600002 against 0.048607398751015045), and J used the identity
    /// weight (4.0346296518810849 against 1.4148059570011566), so it scaled with the units
    /// (x 10: 4034629.65).
    #[test]
    fn test_validate_gmm_normal_moments() {
        let (case, ds) = crate::test_support::load_case("gmm_linear_iv");
        let w = Array1::from(nan_vec(&ds, "w"));
        let run = |data: Array1<f64>, method: GmmMethod| {
            run_general_gmm(
                MomentCondition::NormalDistribution { data },
                cfg(method, GmmVcov::MDS),
            )
            .unwrap()
        };
        let two = run(w.clone(), GmmMethod::TwoStep);
        let it = run(w.clone(), GmmMethod::Iterative);
        let mut check = Check::new(&case);
        pin(
            &mut check,
            "normal_twostep",
            &two,
            Tol::iterative(1e-8),
            Tol::iterative(1e-6),
            true,
        );
        pin(
            &mut check,
            "normal_iter",
            &it,
            Tol::iterative(1e-8),
            Tol::iterative(1e-6),
            true,
        );
        check.finish();
        assert_eq!(two.names, ["mu", "sigma_sq"]);
        assert!(two.convergence.as_ref().is_some_and(|c| c.converged));
        // iterated GMM is unit-free: x 10 leaves J and scales mu by 10 and sigma^2 by 100
        // (R transcription: 6.9e-11 and 3.7e-9)
        let it10 = run(w.mapv(|v| 10.0 * v), GmmMethod::Iterative);
        assert!((it10.j_stat / it.j_stat - 1.0).abs() < 1e-7);
        assert!((it10.coefficients[0] / (10.0 * it.coefficients[0]) - 1.0).abs() < 1e-7);
        assert!((it10.coefficients[1] / (100.0 * it.coefficients[1]) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_gmm_method_parses_aliases_and_rejects_unknown() {
        GmmMethod::choices().validate().unwrap();
        for (s, v) in [
            ("twoStep", GmmMethod::TwoStep),
            ("two-step", GmmMethod::TwoStep),
            ("ITERATIVE", GmmMethod::Iterative),
            ("cue", GmmMethod::CUE),
            ("Continuously Updated", GmmMethod::CUE),
            ("one_step", GmmMethod::OneStep),
        ] {
            assert_eq!(s.parse::<GmmMethod>().unwrap(), v, "{s}");
        }
        expect_invalid("cue2".parse::<GmmMethod>(), "method");
    }

    #[test]
    fn test_gmm_vcov_parses_aliases_and_rejects_unknown() {
        GmmVcov::choices().validate().unwrap();
        for (s, v) in [
            ("HAC", GmmVcov::HAC),
            ("iid", GmmVcov::IID),
            ("MDS", GmmVcov::MDS),
            ("robust", GmmVcov::MDS),
            ("Fixed", GmmVcov::Fixed),
        ] {
            assert_eq!(s.parse::<GmmVcov>().unwrap(), v, "{s}");
        }
        assert_eq!(GmmVcov::MDS.to_string(), "mds");
        expect_invalid("TrueFixed".parse::<GmmVcov>(), "vcov");
    }

    #[test]
    fn test_linear_iv_gmm() {
        // Simple IV example
        let n = 50;
        let y = Array1::from_iter((0..n).map(|i| 2.0 + 3.0 * (i as f64) + (i % 5) as f64 * 0.1));
        let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 });
        let z = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => 1.0,
            1 => i as f64,
            2 => (i * i) as f64 / 100.0,
            _ => 0.0,
        });

        let moments = MomentCondition::LinearIV { y, x, z };
        let config = GeneralGmmConfig {
            method: GmmMethod::TwoStep,
            ..Default::default()
        };

        let result = run_general_gmm(moments, config).unwrap();

        // Check structure
        assert_eq!(result.n_params, 2);
        assert_eq!(result.n_moments, 3);
        assert!(result.n_moments > result.n_params); // overidentified

        // Coefficients should be reasonable
        assert!((result.coefficients[0] - 2.0).abs() < 2.0); // intercept
        assert!((result.coefficients[1] - 3.0).abs() < 1.0); // slope

        // J-test should be computed
        assert!(result.j_stat >= 0.0);
        assert!(result.j_pvalue >= 0.0 && result.j_pvalue <= 1.0);
    }

    #[test]
    fn test_gmm_display() {
        let data = Array1::from(vec![1.0, 1.1, 0.9, 1.2, 0.8, 1.15, 0.85, 1.05, 0.95, 1.0]);
        let moments = MomentCondition::NormalDistribution { data };
        let result = run_general_gmm(moments, GeneralGmmConfig::default()).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("GMM"));
        assert!(display.contains("Coefficients"));
        assert!(display.contains("J-Test"));
    }

    /// The y, X and Z of `test_linear_iv_gmm` (n = 50, q = 3).
    fn iv_data() -> (Array1<f64>, Array2<f64>, Array2<f64>) {
        let n = 50;
        let y = Array1::from_iter((0..n).map(|i| 2.0 + 3.0 * (i as f64) + (i % 5) as f64 * 0.1));
        let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 });
        let z = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => 1.0,
            1 => i as f64,
            2 => (i * i) as f64 / 100.0,
            _ => 0.0,
        });
        (y, x, z)
    }

    #[test]
    fn test_gmm_rejects_hac_bandwidth_not_below_n() {
        let (y, x, z) = iv_data();
        let run = |vcov: GmmVcov, bandwidth: Option<usize>| {
            run_general_gmm(
                MomentCondition::LinearIV {
                    y: y.clone(),
                    x: x.clone(),
                    z: z.clone(),
                },
                GeneralGmmConfig {
                    vcov,
                    bandwidth,
                    ..Default::default()
                },
            )
        };
        expect_invalid(run(GmmVcov::HAC, Some(50)), "bandwidth");
        expect_invalid(run(GmmVcov::HAC, Some(usize::MAX)), "bandwidth");
        assert!(!matches!(
            run(GmmVcov::HAC, Some(49)),
            Err(EconError::InvalidSpecification { .. })
        ));
        // the bandwidth only applies to HAC and stays ignored otherwise
        assert!(!matches!(
            run(GmmVcov::IID, Some(1_000_000)),
            Err(EconError::InvalidSpecification { .. })
        ));
    }

    #[test]
    fn test_gmm_rejects_invalid_iteration_controls() {
        let (y, x, z) = iv_data();
        let run = |tolerance: f64, max_iter: usize| {
            run_general_gmm(
                MomentCondition::LinearIV {
                    y: y.clone(),
                    x: x.clone(),
                    z: z.clone(),
                },
                GeneralGmmConfig {
                    method: GmmMethod::Iterative,
                    tolerance,
                    max_iter,
                    ..Default::default()
                },
            )
        };
        expect_invalid(run(0.0, 100), "tolerance");
        expect_invalid(run(f64::NAN, 100), "tolerance");
        expect_invalid(run(1e-7, 0), "max_iter");
    }

    #[test]
    fn test_gmm_stops_when_cancelled() {
        let (y, x, z) = iv_data();
        expect_cancelled(with_budget(cancelled(), || {
            run_general_gmm(
                MomentCondition::LinearIV { y, x, z },
                GeneralGmmConfig {
                    method: GmmMethod::Iterative,
                    ..Default::default()
                },
            )
        }));
    }

    /// The HAC lag loop checks the budget once per lag. `test_gmm_stops_when_cancelled`
    /// stops at the top of the iterative loop, so this calls the covariance directly.
    #[test]
    fn test_hac_weight_stops_when_cancelled() {
        let (y, x, z) = iv_data();
        let theta = Array1::from(vec![2.0, 3.0]);
        let config = GeneralGmmConfig {
            bandwidth: Some(5),
            ..Default::default()
        };
        assert_eq!(config.vcov, GmmVcov::HAC);
        let model = LinearIv::new(&y, &x, &z);
        assert!(moment_covariance(&model, theta.view(), &config).is_ok());
        expect_cancelled(with_budget(cancelled(), || {
            moment_covariance(&model, theta.view(), &config).map(|m| m.omega)
        }));
    }

    #[test]
    fn test_hac_lag_loop_stops_at_sample_size() {
        let (y, x, z) = iv_data();
        let theta = Array1::from(vec![2.0, 3.0]);
        let model = LinearIv::new(&y, &x, &z);
        let w = |bw: usize| {
            moment_covariance(
                &model,
                theta.view(),
                &GeneralGmmConfig {
                    kernel: HacKernel::Truncated,
                    bandwidth: Some(bw),
                    ..Default::default()
                },
            )
            .map(|m| m.omega)
        };
        // Truncated: weight 1 for every lag <= bw, so lags 1..=49 for all of these
        let reference = w(49);
        for bw in [50, 500, usize::MAX] {
            match (&reference, &w(bw)) {
                (Ok(a), Ok(b)) => assert_eq!(a, b, "bandwidth {bw}"),
                (Err(_), Err(_)) => {}
                (a, b) => panic!("bandwidth {bw}: {a:?} vs {b:?}"),
            }
        }
    }

    /// GMM-IV drops rows with a missing outcome, regressor or instrument and equals the
    /// fit on the complete rows; the errors name the column.
    #[test]
    fn test_validate_na_gmm_iv_omit_equals_complete_rows() {
        let (case, ds) = crate::test_support::load_case("iv_na_rows");
        assert_eq!(case.dataset, "validation/datasets/generated/iv_na_rows.csv");
        let x = ["x", "w", "k"];
        let z = ["z1", "z2", "w", "k"];
        let r = run_gmm_iv(&ds, "y", &x, &z, None).unwrap();
        let complete = Dataset::new(ds.df().drop_nulls::<String>(None).unwrap());
        let c = run_gmm_iv(&complete, "y", &x, &z, None).unwrap();
        let tol = p2a_validation::Tol::new(1e-12, 1e-12);
        let va = r
            .coefficients
            .iter()
            .chain(&r.std_errors)
            .chain([&r.j_stat]);
        let vb = c
            .coefficients
            .iter()
            .chain(&c.std_errors)
            .chain([&c.j_stat]);
        for (a, b) in va.zip(vb) {
            assert!(tol.allows(*a, *b), "{a} vs {b}");
        }
        assert_eq!(r.names, c.names);
        assert_eq!(r.n_obs, 294);
        assert!(c.missing.is_empty());
        let note = "6 of 300 rows dropped because of missing values (y: 2, x: 1, w: 1, k: 1, z1: 1, z2: 1)";
        assert_eq!(r.missing.note().as_deref(), Some(note));
        let text = r.to_string();
        assert!(
            text.trim_end()
                .ends_with(&format!("Missing values: {note}")),
            "{text}"
        );
        match run_gmm_iv(&ds, "label", &x, &z, None) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "label"),
            other => panic!("expected NonNumericColumn, got {other:?}"),
        }
        match run_gmm_iv(&ds, "y", &x, &["nope", "w"], None) {
            Err(EconError::ColumnNotFound { column, .. }) => assert_eq!(column, "nope"),
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
    }
}
