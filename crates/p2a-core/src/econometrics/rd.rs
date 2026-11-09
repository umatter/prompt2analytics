//! Regression Discontinuity Design (RDD) estimation with robust bias-corrected inference.
//!
//! Pure Rust implementation of local polynomial RD estimation following the methodology
//! of Calonico, Cattaneo, Titiunik & Farrell (2014-2020).
//!
//! For given bandwidths (h and b, optionally different on the two sides), [`run_rd`]
//! computes what `rdrobust::rdrobust` 4.0.0 computes: the local-polynomial estimate, the
//! CCT (2014) bias-corrected estimate built from the `Q_q` weights, and the conventional
//! and robust sandwich variances with nearest-neighbour residuals (whole tie groups,
//! `nnmatch`) or HC0-HC3 residuals. [`run_fuzzy_rd`] fits the outcome and the treatment
//! on one bandwidth, as `rdrobust(fuzzy = )`. When h is not given, h and b are selected as
//! `rdrobust::rdbwselect` 4.0.0 selects them ([`crate::econometrics::rd_bwselect`]), with b
//! selected separately from h. Covariates, clustering, weights and derivatives (`deriv`)
//! are not supported.
//!
//! # References
//!
//! - Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). "Robust Nonparametric Confidence
//!   Intervals for Regression-Discontinuity Designs". *Econometrica*, 82(6), 2295-2326.
//! - Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2020). "Optimal Bandwidth Choice for
//!   Robust Bias Corrected Inference in Regression Discontinuity Designs". *Econometrics Journal*, 23(2), 192-210.
//! - Imbens, G. & Kalyanaraman, K. (2012). "Optimal Bandwidth Choice for the Regression
//!   Discontinuity Estimator". *Review of Economic Studies*, 79(3), 933-959.
//! - Implementation adapted from R package `rdrobust` (Calonico, Cattaneo, Farrell, Titiunik).
//!   Source: https://cran.r-project.org/package=rdrobust

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

use super::rd_bwselect::{BwSelectOptions, BwSelection, rdbwselect};
use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::safe_inverse;
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::SignificanceLevel;

/// Kernel function types for local polynomial regression.
///
/// The kernel determines how observations are weighted based on their distance
/// from the cutoff point. Closer observations receive higher weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum KernelType {
    /// Triangular kernel: K(u) = (1 - |u|) for |u| ≤ 1
    /// Gives linearly decreasing weights. Recommended for RD (default).
    #[default]
    Triangular,
    /// Epanechnikov kernel: K(u) = 0.75(1 - u²) for |u| ≤ 1
    /// Optimal for MSE in density estimation.
    Epanechnikov,
    /// Uniform kernel: K(u) = 0.5 for |u| ≤ 1
    /// Equal weights within bandwidth.
    Uniform,
}

impl KernelType {
    /// Evaluate the kernel function at point u.
    pub fn evaluate(&self, u: f64) -> f64 {
        if u.abs() > 1.0 {
            return 0.0;
        }
        match self {
            KernelType::Triangular => 1.0 - u.abs(),
            KernelType::Epanechnikov => 0.75 * (1.0 - u * u),
            KernelType::Uniform => 0.5,
        }
    }

    /// rdbwselect's rule-of-thumb constant `C_c` of the pilot bandwidth.
    pub fn constant(&self) -> f64 {
        match self {
            // numeric-ok: rdbwselect's kernel constant C_c, not a critical value
            KernelType::Triangular => 2.576,
            KernelType::Epanechnikov => 2.34,
            KernelType::Uniform => 1.843,
        }
    }

    /// Parse kernel type from string (the spellings of [`KernelType::choices()`]).
    pub fn from_str(s: &str) -> Option<Self> {
        Self::choices().parse(s).ok()
    }

    /// rdrobust's `kernel` names, case-insensitive: "triangular" (also "tri"),
    /// "epanechnikov" ("epa"), "uniform" ("uni"). An unknown name is an
    /// `InvalidSpecification` listing the valid ones.
    pub fn parse(s: &str) -> EconResult<KernelType> {
        s.parse::<KernelType>()
            .map_err(|_| EconError::InvalidSpecification {
                message: format!(
                    "Unknown kernel '{s}'; use \"triangular\" (the default, also \"tri\"), \
                     \"epanechnikov\" (\"epa\") or \"uniform\" (\"uni\")"
                ),
            })
    }
}

crate::option_enum!(
    KernelType,
    crate::options::Choices {
        option: "kernel",
        values: &[
            (&["triangular", "tri"], KernelType::Triangular),
            (&["epanechnikov", "epa"], KernelType::Epanechnikov),
            (&["uniform", "uni"], KernelType::Uniform),
        ],
    },
    no_display
);

impl fmt::Display for KernelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelType::Triangular => write!(f, "Triangular"),
            KernelType::Epanechnikov => write!(f, "Epanechnikov"),
            KernelType::Uniform => write!(f, "Uniform"),
        }
    }
}

/// Bandwidth selection methods: rdrobust::rdbwselect's ten `bwselect` choices, in its row
/// order ([`BandwidthMethod::ALL`]). The MSE methods minimise the asymptotic MSE of the point
/// estimator; each CER method is its MSE counterpart with h scaled by the coverage-error
/// rate `N^(-p / ((3 + p)(3 + 2p)))` (b unchanged). The selector is
/// [`crate::econometrics::rd_bwselect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BandwidthMethod {
    /// `mserd`: MSE-optimal, one common bandwidth for both sides (default).
    #[default]
    MseRd,
    /// `msetwo`: MSE-optimal, separate bandwidths left and right of the cutoff.
    MseTwo,
    /// `msesum`: MSE-optimal for the sum of the two regression estimates, one common bandwidth.
    MseSum,
    /// `msecomb1`: the smaller of mserd and msesum.
    MseComb1,
    /// `msecomb2`: the median of mserd, msesum and msetwo, per side.
    MseComb2,
    /// `cerrd`: CER-optimal, one common bandwidth.
    CerRd,
    /// `certwo`: CER-optimal, separate left and right.
    CerTwo,
    /// `cersum`: CER-optimal for the sum of the two regression estimates, one common bandwidth.
    CerSum,
    /// `cercomb1`: the smaller of cerrd and cersum.
    CerComb1,
    /// `cercomb2`: the median of cerrd, cersum and certwo, per side.
    CerComb2,
}

impl BandwidthMethod {
    /// The ten methods in rdbwselect's row order (`all = TRUE`).
    pub const ALL: [BandwidthMethod; 10] = [
        BandwidthMethod::MseRd,
        BandwidthMethod::MseTwo,
        BandwidthMethod::MseSum,
        BandwidthMethod::MseComb1,
        BandwidthMethod::MseComb2,
        BandwidthMethod::CerRd,
        BandwidthMethod::CerTwo,
        BandwidthMethod::CerSum,
        BandwidthMethod::CerComb1,
        BandwidthMethod::CerComb2,
    ];

    /// rdrobust's name: "mserd", ..., "cercomb2".
    pub fn name(&self) -> &'static str {
        Self::choices().canonical(*self)
    }

    /// What the method selects, in words.
    pub fn description(&self) -> &'static str {
        match self {
            BandwidthMethod::MseRd => "MSE-optimal, one common bandwidth",
            BandwidthMethod::MseTwo => "MSE-optimal, separate left and right",
            BandwidthMethod::MseSum => {
                "MSE-optimal for the sum of the two regression estimates, one common bandwidth"
            }
            BandwidthMethod::MseComb1 => "smaller of mserd and msesum",
            BandwidthMethod::MseComb2 => "median of mserd, msesum and msetwo, per side",
            BandwidthMethod::CerRd => "CER-optimal, one common bandwidth",
            BandwidthMethod::CerTwo => "CER-optimal, separate left and right",
            BandwidthMethod::CerSum => {
                "CER-optimal for the sum of the two regression estimates, one common bandwidth"
            }
            BandwidthMethod::CerComb1 => "smaller of cerrd and cersum",
            BandwidthMethod::CerComb2 => "median of cerrd, cersum and certwo, per side",
        }
    }

    /// A coverage-error-rate (CER) method.
    pub fn is_cer(&self) -> bool {
        matches!(
            self,
            BandwidthMethod::CerRd
                | BandwidthMethod::CerTwo
                | BandwidthMethod::CerSum
                | BandwidthMethod::CerComb1
                | BandwidthMethod::CerComb2
        )
    }

    /// The MSE method a CER method scales (CerRd -> MseRd, ...); an MSE method itself.
    pub fn mse_counterpart(&self) -> BandwidthMethod {
        match self {
            BandwidthMethod::CerRd => BandwidthMethod::MseRd,
            BandwidthMethod::CerTwo => BandwidthMethod::MseTwo,
            BandwidthMethod::CerSum => BandwidthMethod::MseSum,
            BandwidthMethod::CerComb1 => BandwidthMethod::MseComb1,
            BandwidthMethod::CerComb2 => BandwidthMethod::MseComb2,
            other => *other,
        }
    }

    /// Parse bandwidth method from string (the spellings of [`BandwidthMethod::choices()`]).
    pub fn from_str(s: &str) -> Option<Self> {
        Self::choices().parse(s).ok()
    }

    /// rdrobust's `bwselect` names, case-insensitive (also "mse", "mse_two", "cer",
    /// "cer_two"). An unknown name is an `InvalidSpecification` listing the ten valid ones.
    pub fn parse(s: &str) -> EconResult<BandwidthMethod> {
        s.parse::<BandwidthMethod>().map_err(|_| {
            let others: Vec<String> = Self::ALL[1..]
                .iter()
                .map(|m| format!("\"{}\" ({})", m.name(), m.description()))
                .collect();
            EconError::InvalidSpecification {
                message: format!(
                    "Unknown bwselect '{s}'; use \"mserd\" ({}, the default), {}",
                    BandwidthMethod::MseRd.description(),
                    others.join(", ")
                ),
            }
        })
    }

    /// Check if this method can give different bandwidths left and right of the cutoff
    /// (msetwo, certwo, msecomb2, cercomb2).
    pub fn is_separate(&self) -> bool {
        matches!(
            self,
            BandwidthMethod::MseTwo
                | BandwidthMethod::CerTwo
                | BandwidthMethod::MseComb2
                | BandwidthMethod::CerComb2
        )
    }
}

crate::option_enum!(
    BandwidthMethod,
    crate::options::Choices {
        option: "bwselect",
        values: &[
            (&["mserd", "mse"], BandwidthMethod::MseRd),
            (&["msetwo", "mse_two"], BandwidthMethod::MseTwo),
            (&["msesum"], BandwidthMethod::MseSum),
            (&["msecomb1"], BandwidthMethod::MseComb1),
            (&["msecomb2"], BandwidthMethod::MseComb2),
            (&["cerrd", "cer"], BandwidthMethod::CerRd),
            (&["certwo", "cer_two"], BandwidthMethod::CerTwo),
            (&["cersum"], BandwidthMethod::CerSum),
            (&["cercomb1"], BandwidthMethod::CerComb1),
            (&["cercomb2"], BandwidthMethod::CerComb2),
        ],
    },
    no_display
);

impl fmt::Display for BandwidthMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name(), self.description())
    }
}

/// Variance-covariance estimation methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VceType {
    /// Nearest-neighbor variance estimator (default, nnmatch=3).
    /// Robust to heteroskedasticity without specification.
    #[default]
    Nn,
    /// HC0: White's heteroskedasticity-consistent estimator.
    Hc0,
    /// HC1: HC0 with small-sample correction (n/(n-k)).
    Hc1,
    /// HC2: Uses leverage adjustment.
    Hc2,
    /// HC3: Most conservative; inflates residuals by leverage.
    Hc3,
}

impl VceType {
    /// Parse VCE type from string (the spellings of [`VceType::choices()`]).
    pub fn from_str(s: &str) -> Option<Self> {
        Self::choices().parse(s).ok()
    }

    /// rdrobust's `vce` names, case-insensitive: "nn" (also "nearest"), "hc0", "hc1",
    /// "hc2", "hc3". An unknown name is an `InvalidSpecification` listing the valid ones.
    pub fn parse(s: &str) -> EconResult<VceType> {
        s.parse::<VceType>()
            .map_err(|_| EconError::InvalidSpecification {
                message: format!(
                    "Unknown vce '{s}'; use \"nn\" (nearest-neighbour residuals, the default, as \
                     rdrobust), \"hc0\", \"hc1\", \"hc2\" or \"hc3\""
                ),
            })
    }
}

crate::option_enum!(
    VceType,
    crate::options::Choices {
        option: "vce",
        values: &[
            (&["nn", "nearest"], VceType::Nn),
            (&["hc0"], VceType::Hc0),
            (&["hc1"], VceType::Hc1),
            (&["hc2"], VceType::Hc2),
            (&["hc3"], VceType::Hc3),
        ],
    },
    no_display
);

impl fmt::Display for VceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VceType::Nn => write!(f, "Nearest-Neighbor"),
            VceType::Hc0 => write!(f, "HC0"),
            VceType::Hc1 => write!(f, "HC1"),
            VceType::Hc2 => write!(f, "HC2"),
            VceType::Hc3 => write!(f, "HC3"),
        }
    }
}

/// Configuration for RD estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdConfig {
    /// Polynomial order for point estimation (default: 1 = local linear).
    pub p: usize,
    /// Polynomial order for bias correction (default: p + 1).
    pub q: Option<usize>,
    /// Main bandwidth h; `None` selects h with `bwselect` (rdrobust::rdbwselect).
    pub h: Option<f64>,
    /// Bias bandwidth b. `None`: with a given h, `b = h / rho`; with a selected h, b is
    /// selected too (or `h / rho`, see `rho` and `rho_with_selected_h`). A b given with a
    /// selected h is used (rdrobust ignores it).
    pub b: Option<f64>,
    /// rho = h / b, as rdrobust: when b is not given, `b = h / rho` for a given h, and for a
    /// selected h when `rho` is not 1 or `rho_with_selected_h` is set (default 1).
    pub rho: f64,
    /// Kernel function type.
    pub kernel: KernelType,
    /// Bandwidth selection method when h is not given (rdrobust::rdbwselect's `bwselect`).
    pub bwselect: BandwidthMethod,
    /// Variance estimation method.
    pub vce: VceType,
    /// Number of neighbors for NN variance estimation (default: 3).
    pub nnmatch: usize,
    /// Confidence level (default: 0.95).
    pub level: f64,
    /// Weight of the regularisation term of the bandwidth selector (rdbwselect
    /// `scaleregul`, default 1; 0 switches it off). Used only when h is selected.
    pub scaleregul: f64,
    /// Main bandwidth right of the cutoff when it differs from the left one (rdrobust
    /// `h = c(h_l, h_r)`); requires `h`, which is then the left-side bandwidth.
    #[serde(default)]
    pub h_right: Option<f64>,
    /// Bias bandwidth right of the cutoff (rdrobust `b = c(b_l, b_r)`); requires `b`.
    #[serde(default)]
    pub b_right: Option<f64>,
    /// With a selected h and no b: false = b selected by `bwselect` (rdrobust without
    /// rho) when `rho == 1`; true = `b = h / rho` (rdrobust's rho). A `rho` other than 1
    /// always gives `b = h / rho`, whatever this flag says. Ignored when h is given.
    #[serde(default)]
    pub rho_with_selected_h: bool,
}

impl Default for RdConfig {
    fn default() -> Self {
        Self {
            p: 1,     // Local linear (most common)
            q: None,  // Will default to p + 1
            h: None,  // Auto-compute
            b: None,  // Auto-compute
            rho: 1.0, // b = h by default
            kernel: KernelType::default(),
            bwselect: BandwidthMethod::default(),
            vce: VceType::default(),
            nnmatch: 3,
            level: 0.95,
            scaleregul: 1.0,
            h_right: None,
            b_right: None,
            rho_with_selected_h: false,
        }
    }
}

/// Where a bandwidth came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BandwidthOrigin {
    /// Selected by `RdConfig::bwselect`, as rdrobust::rdbwselect.
    #[default]
    Selected,
    /// Given by the user (h / h_right, b / b_right).
    UserSupplied,
    /// b only: `h / rho` (rdrobust's rho; rho = 1 gives b = h).
    RhoTimesH,
}

/// The confidence level of a result deserialized without one.
pub(crate) fn default_rd_level() -> f64 {
    0.95
}

/// rdrobust's default `scaleregul` (the weight of the bandwidth selector's regularisation).
pub(crate) fn default_scaleregul() -> f64 {
    1.0
}

/// rdrobust's default `nnmatch`.
pub(crate) fn default_nnmatch() -> usize {
    3
}

/// `scaleregul` must be finite and >= 0.
pub(crate) fn validate_scaleregul(v: f64) -> EconResult<()> {
    if v.is_finite() && v >= 0.0 {
        return Ok(());
    }
    Err(EconError::InvalidSpecification {
        message: format!(
            "scaleregul (the regularisation scale of the bandwidth selector) must be a finite \
             number >= 0 (got {v}); rdrobust's default is 1 and 0 switches the regularisation off"
        ),
    })
}

/// Bandwidth estimation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdBandwidth {
    /// Main bandwidth (left of cutoff).
    pub h_left: f64,
    /// Main bandwidth (right of cutoff).
    pub h_right: f64,
    /// Bias bandwidth (left of cutoff).
    pub b_left: f64,
    /// Bias bandwidth (right of cutoff).
    pub b_right: f64,
    /// Bandwidth selection method used.
    pub bwselect: BandwidthMethod,
    /// Kernel used.
    pub kernel: KernelType,
    /// Polynomial order.
    pub p: usize,
    /// Sample size left of cutoff.
    pub n_left: usize,
    /// Sample size right of cutoff.
    pub n_right: usize,
    /// Rows dropped because the outcome or the running variable is missing.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
    /// Bias-correction order.
    #[serde(default)]
    pub q: usize,
    /// Residuals of the selector's variance terms.
    #[serde(default)]
    pub vce: VceType,
    /// Nearest neighbours of the `nn` residuals.
    #[serde(default = "default_nnmatch")]
    pub nnmatch: usize,
    /// Regularisation weight of the selector.
    #[serde(default = "default_scaleregul")]
    pub scaleregul: f64,
    /// Observations left of the cutoff with positive kernel weight under h (rdbwselect `N_h`).
    #[serde(default)]
    pub n_h_left: usize,
    /// The same right of the cutoff.
    #[serde(default)]
    pub n_h_right: usize,
    /// Distinct running-variable values left of the cutoff (rdbwselect `M`).
    #[serde(default)]
    pub m_left: usize,
    /// Distinct running-variable values right of the cutoff.
    #[serde(default)]
    pub m_right: usize,
    /// Mass points in the running variable (`1 - M / N >= 0.2` on a side).
    #[serde(default)]
    pub mass_points: bool,
    /// `Some(10)` when the mass-point adjustment applied.
    #[serde(default)]
    pub bwcheck: Option<usize>,
    /// The pilot bandwidth of the selector's variance terms.
    #[serde(default)]
    pub pilot_bandwidth: f64,
    /// Where b came from.
    #[serde(default)]
    pub b_origin: BandwidthOrigin,
    /// The treatment of a fuzzy selection.
    #[serde(default)]
    pub treatment: Option<String>,
    /// Warnings of the selection (and the missing-value note).
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl fmt::Display for RdBandwidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RD Bandwidth Selection (rdrobust::rdbwselect)")?;
        writeln!(f, "{}", "=".repeat(45))?;
        writeln!(f, "Method: {}", self.bwselect)?;
        writeln!(
            f,
            "Kernel: {}, p = {}, q = {}, VCE: {} (nnmatch = {}), scaleregul = {}",
            self.kernel, self.p, self.q, self.vce, self.nnmatch, self.scaleregul
        )?;
        if let Some(t) = &self.treatment {
            writeln!(f, "Fuzzy: treatment {t}")?;
        }
        writeln!(f)?;
        writeln!(f, "Main bandwidth (h):")?;
        writeln!(f, "  Left:  {:.4}", self.h_left)?;
        writeln!(f, "  Right: {:.4}", self.h_right)?;
        writeln!(f)?;
        writeln!(f, "Bias bandwidth (b):")?;
        writeln!(f, "  Left:  {:.4}", self.b_left)?;
        writeln!(f, "  Right: {:.4}", self.b_right)?;
        writeln!(f)?;
        writeln!(
            f,
            "Sample sizes: {} (left), {} (right)",
            self.n_left, self.n_right
        )?;
        writeln!(
            f,
            "Within h: {} (left), {} (right)",
            self.n_h_left, self.n_h_right
        )?;
        writeln!(
            f,
            "Distinct running-variable values: {} (left), {} (right)",
            self.m_left, self.m_right
        )?;
        write_warnings_and_missing(f, &self.warnings, &self.missing)
    }
}

/// Result from Sharp RD estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdResult {
    /// Outcome variable name.
    pub outcome: String,
    /// Running variable name.
    pub running_var: String,
    /// Cutoff value.
    pub cutoff: f64,

    // Sample information
    /// Total observations left of cutoff.
    pub n_left: usize,
    /// Total observations right of cutoff.
    pub n_right: usize,
    /// Observations left of the cutoff with positive kernel weight under h (rdrobust `N_h`).
    pub n_eff_left: usize,
    /// Observations right of the cutoff with positive kernel weight under h.
    pub n_eff_right: usize,
    /// Observations left of the cutoff with positive kernel weight under b (rdrobust `N_b`).
    #[serde(default)]
    pub n_b_left: usize,
    /// Observations right of the cutoff with positive kernel weight under b.
    #[serde(default)]
    pub n_b_right: usize,

    // Point estimates
    /// Conventional RD estimate (local polynomial at cutoff; rdrobust `coef[1]`).
    pub tau_conventional: f64,
    /// CCT (2014) bias-corrected estimate (rdrobust `coef[2]`).
    pub tau_bc: f64,
    /// Robust bias-corrected estimate: equal to `tau_bc`, with the robust SE.
    pub tau_robust: f64,

    // Standard errors
    /// Conventional standard error (rdrobust `se[1]`).
    pub se_conventional: f64,
    /// Standard error of the bias-corrected row: the conventional one (rdrobust `se[2]`).
    pub se_bc: f64,
    /// Robust standard error, accounting for the estimated bias (rdrobust `se[3]`).
    pub se_robust: f64,

    // Confidence intervals
    /// Conventional CI: `tau_conventional -/+ z se_conventional`.
    pub ci_conventional: (f64, f64),
    /// Bias-corrected CI: `tau_bc -/+ z se_conventional`.
    pub ci_bc: (f64, f64),
    /// Robust bias-corrected CI (recommended): `tau_bc -/+ z se_robust`.
    pub ci_robust: (f64, f64),
    /// Confidence level of every interval (`RdConfig::level`).
    #[serde(default = "default_rd_level")]
    pub level: f64,

    // P-values
    /// P-value for conventional estimate.
    pub p_conventional: f64,
    /// P-value for bias-corrected estimate.
    pub p_bc: f64,
    /// P-value for robust estimate.
    pub p_robust: f64,
    /// Significance level (based on robust p-value).
    pub significance: SignificanceLevel,

    // Bandwidth information
    /// Main bandwidth (left).
    pub h_left: f64,
    /// Main bandwidth (right).
    pub h_right: f64,
    /// Bias bandwidth (left).
    pub b_left: f64,
    /// Bias bandwidth (right).
    pub b_right: f64,
    /// Bandwidth selection method (not run when `bandwidth_given`).
    pub bwselect: BandwidthMethod,
    /// The caller gave h (rdrobust's "Manual" bandwidth), so no selector ran
    /// (`h_origin == UserSupplied`).
    #[serde(default)]
    pub bandwidth_given: bool,
    /// Where h came from (never `RhoTimesH`).
    #[serde(default)]
    pub h_origin: BandwidthOrigin,
    /// Where b came from.
    #[serde(default)]
    pub b_origin: BandwidthOrigin,

    // Specification
    /// Polynomial order for estimation.
    pub p: usize,
    /// Polynomial order for bias correction.
    pub q: usize,
    /// Kernel function used.
    pub kernel: KernelType,
    /// Variance estimation method.
    pub vce: VceType,

    // Polynomial coefficients (for diagnostics)
    /// Coefficients from left-side polynomial.
    pub coef_left: Vec<f64>,
    /// Coefficients from right-side polynomial.
    pub coef_right: Vec<f64>,

    /// Estimated bias: `tau_conventional - tau_bc`.
    pub bias: f64,

    /// Warnings generated during estimation.
    pub warnings: Vec<String>,

    /// Rows dropped because the outcome or the running variable is missing (as rdrobust's
    /// `complete.cases`).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

/// The warnings without the missing-value note (which prints on its own line, last).
pub(crate) fn without_missing_note(warnings: &[String], missing: &MissingSummary) -> Vec<String> {
    let note = missing.note();
    warnings
        .iter()
        .filter(|w| Some(w.as_str()) != note.as_deref())
        .cloned()
        .collect()
}

/// Warnings already filtered, for `notes_text`.
struct Shown(Vec<String>);

impl ResultNotes for Shown {
    fn notes_warnings(&self) -> Vec<String> {
        self.0.clone()
    }
}

impl ResultNotes for RdResult {
    fn notes_warnings(&self) -> Vec<String> {
        without_missing_note(&self.warnings, &self.missing)
    }
}

impl ResultNotes for FuzzyRdResult {
    fn notes_warnings(&self) -> Vec<String> {
        without_missing_note(&self.warnings, &self.missing)
    }
}

impl ResultNotes for RdBandwidth {
    fn notes_warnings(&self) -> Vec<String> {
        without_missing_note(&self.warnings, &self.missing)
    }
}

/// The notes block of an RD Display: `Warning: <text>` per warning (as `notes_text`), then
/// `Missing values: <note>` last.
pub(crate) fn write_warnings_and_missing(
    f: &mut fmt::Formatter<'_>,
    warnings: &[String],
    missing: &MissingSummary,
) -> fmt::Result {
    let note = missing.note();
    let shown = without_missing_note(warnings, missing);
    if !shown.is_empty() {
        writeln!(f)?;
        write!(f, "{}", notes_text(&Shown(shown)))?;
    }
    if let Some(note) = note {
        writeln!(f)?;
        writeln!(f, "Missing values: {note}")?;
    }
    Ok(())
}

/// "95", "90", "97.5": a confidence level in percent, for interval labels.
pub(crate) fn level_percent(level: f64) -> String {
    let pct = (level * 100.0 * 1e9).round() / 1e9;
    if pct.fract() == 0.0 {
        format!("{pct:.0}")
    } else {
        format!("{pct}")
    }
}

/// "[95% CI]", "[90% CI]", "[97.5% CI]": the interval label for a confidence level.
fn ci_label(level: f64) -> String {
    format!("[{}% CI]", level_percent(level))
}

/// The estimate table's header, `level` labelling the interval column.
fn write_rd_header(f: &mut fmt::Formatter<'_>, level: f64) -> fmt::Result {
    writeln!(
        f,
        "{:<20} {:>12} {:>12} {:>10} {:>10}  {}",
        "Method",
        "Estimate",
        "Std. Err.",
        "z",
        "P>|z|",
        ci_label(level)
    )?;
    writeln!(f, "{}", "-".repeat(90))
}

/// One row of the estimate table: estimate, SE, z = est / se (inf when se = 0), two-sided
/// p and the interval.
fn write_rd_row(
    f: &mut fmt::Formatter<'_>,
    label: &str,
    est: f64,
    se: f64,
    p: f64,
    ci: (f64, f64),
    stars: &str,
) -> fmt::Result {
    writeln!(
        f,
        "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.4}  [{:>8.4}, {:>8.4}]{}",
        label,
        est,
        se,
        est / se,
        p,
        ci.0,
        ci.1,
        stars
    )
}

impl RdResult {
    /// Where h and b came from, for output: "h and b selected by mserd (...), as
    /// rdrobust::rdbwselect", "h user-supplied; b = 1.0000 h (rho)", "h and b user-supplied".
    pub fn bandwidth_label(&self) -> String {
        use BandwidthOrigin::*;
        let ratio = format!("{:.4}", self.b_left / self.h_left);
        let by = format!("selected by {}, as rdrobust::rdbwselect", self.bwselect);
        match (self.h_origin, self.b_origin) {
            (Selected, Selected) => format!("h and b {by}"),
            (Selected, UserSupplied) => format!("h {by}; b user-supplied"),
            (Selected, RhoTimesH) => format!("h {by}; b = {ratio} h (rho)"),
            (_, RhoTimesH) => format!("h user-supplied; b = {ratio} h (rho)"),
            _ => "h and b user-supplied".to_string(),
        }
    }
}

impl fmt::Display for RdResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Sharp RD Estimation Results")?;
        writeln!(f, "===========================")?;
        writeln!(f, "Outcome: {}", self.outcome)?;
        writeln!(
            f,
            "Running variable: {} (cutoff = {:.4})",
            self.running_var, self.cutoff
        )?;
        writeln!(f)?;

        writeln!(f, "Number of Observations:")?;
        writeln!(
            f,
            "  Total:     {} (left), {} (right)",
            self.n_left, self.n_right
        )?;
        writeln!(
            f,
            "  Effective: {} (left), {} (right)",
            self.n_eff_left, self.n_eff_right
        )?;
        writeln!(
            f,
            "  Bias fit:  {} (left), {} (right)",
            self.n_b_left, self.n_b_right
        )?;
        writeln!(f)?;

        writeln!(
            f,
            "Bandwidth (h): {:.4} (left), {:.4} (right)",
            self.h_left, self.h_right
        )?;
        writeln!(
            f,
            "Bandwidth (b): {:.4} (left), {:.4} (right)",
            self.b_left, self.b_right
        )?;
        writeln!(f, "Bandwidth source: {}", self.bandwidth_label())?;
        writeln!(f)?;

        writeln!(
            f,
            "Specification: Order p={} (estimation), q={} (bias)",
            self.p, self.q
        )?;
        writeln!(f, "Kernel: {}, VCE: {}", self.kernel, self.vce)?;
        writeln!(f)?;

        write_rd_header(f, self.level)?;
        write_rd_row(
            f,
            "Conventional",
            self.tau_conventional,
            self.se_conventional,
            self.p_conventional,
            self.ci_conventional,
            "",
        )?;
        write_rd_row(
            f,
            "Bias-Corrected",
            self.tau_bc,
            self.se_bc,
            self.p_bc,
            self.ci_bc,
            "",
        )?;
        write_rd_row(
            f,
            "Robust",
            self.tau_robust,
            self.se_robust,
            self.p_robust,
            self.ci_robust,
            self.significance.stars(),
        )?;
        writeln!(f, "{}", "-".repeat(90))?;
        writeln!(f)?;

        writeln!(
            f,
            "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '\u{2020}' 0.1"
        )?;
        writeln!(
            f,
            "Note: Robust = bias-corrected estimate with the CCT (2014) robust standard error, \
             as rdrobust's 'Robust' row (recommended); Bias-Corrected uses the conventional \
             standard error."
        )?;

        write_warnings_and_missing(f, &self.warnings, &self.missing)
    }
}

/// Result from Fuzzy RD estimation (rdrobust `rdrobust(y, x, fuzzy = t)`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyRdResult {
    /// Sharp RD of the outcome (reduced form), from the same fit and bandwidths as the
    /// first stage.
    pub outcome_rd: RdResult,
    /// Sharp RD of the treatment (first stage).
    pub first_stage: RdResult,
    /// Conventional Wald ratio `tau_Y,cl / tau_T,cl` (rdrobust `coef[1]`).
    pub tau_fuzzy: f64,
    /// Its joint delta-method standard error, covariance of the two jumps included
    /// (rdrobust `se[1]`).
    pub se_fuzzy: f64,
    /// Two-sided p-value of the conventional estimate.
    pub p_fuzzy: f64,
    /// Conventional CI at `level`.
    pub ci_fuzzy: (f64, f64),
    /// Significance level, from `p_fuzzy_robust` (CCT, as for sharp RD).
    pub significance: SignificanceLevel,
    /// Treatment variable name.
    pub treatment: String,
    /// Bias-corrected Wald estimate `tau_fuzzy - s'(B_Y, B_T)` (rdrobust `coef[2]`).
    #[serde(default)]
    pub tau_fuzzy_bc: f64,
    /// Robust standard error of the bias-corrected estimate (rdrobust `se[3]`).
    #[serde(default)]
    pub se_fuzzy_robust: f64,
    /// Robust CI at `level`: `tau_fuzzy_bc -/+ z se_fuzzy_robust`.
    #[serde(default)]
    pub ci_fuzzy_robust: (f64, f64),
    /// Two-sided p-value of the robust row.
    #[serde(default)]
    pub p_fuzzy_robust: f64,
    /// Conventional first-stage jump (rdrobust `tau_T[1]`).
    #[serde(default)]
    pub first_stage_tau: f64,
    /// First-stage F: `(tau_T / se_T)^2` with the conventional SE.
    #[serde(default)]
    pub first_stage_f: f64,
    /// Warnings (a weak first stage, ill-conditioned fits).
    #[serde(default)]
    pub warnings: Vec<String>,
    /// Confidence level of every interval.
    #[serde(default = "default_rd_level")]
    pub level: f64,
    /// Rows dropped because the outcome, the running variable or the treatment is missing.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl fmt::Display for FuzzyRdResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rd = &self.outcome_rd;
        writeln!(f, "Fuzzy RD Estimation Results")?;
        writeln!(f, "===========================")?;
        writeln!(f, "Outcome: {}", rd.outcome)?;
        writeln!(f, "Treatment: {}", self.treatment)?;
        writeln!(
            f,
            "Running variable: {} (cutoff = {:.4})",
            rd.running_var, rd.cutoff
        )?;
        writeln!(
            f,
            "Bandwidth (h): {:.4} (left), {:.4} (right); (b): {:.4} (left), {:.4} (right); \
             one bandwidth for both stages",
            rd.h_left, rd.h_right, rd.b_left, rd.b_right
        )?;
        writeln!(f, "Bandwidth source: {}", rd.bandwidth_label())?;
        writeln!(f, "Kernel: {}, VCE: {}", rd.kernel, rd.vce)?;
        writeln!(f)?;

        writeln!(
            f,
            "Fuzzy RD Estimate (LATE = jump in outcome / jump in treatment):"
        )?;
        write_rd_header(f, self.level)?;
        write_rd_row(
            f,
            "Conventional",
            self.tau_fuzzy,
            self.se_fuzzy,
            self.p_fuzzy,
            self.ci_fuzzy,
            "",
        )?;
        write_rd_row(
            f,
            "Robust",
            self.tau_fuzzy_bc,
            self.se_fuzzy_robust,
            self.p_fuzzy_robust,
            self.ci_fuzzy_robust,
            self.significance.stars(),
        )?;
        writeln!(f, "{}", "-".repeat(90))?;
        writeln!(f)?;

        writeln!(
            f,
            "First stage (jump in treatment): {:.4} (SE {:.4}), F = {:.2}",
            self.first_stage_tau, self.first_stage.se_conventional, self.first_stage_f
        )?;
        writeln!(
            f,
            "Reduced form (jump in outcome):  {:.4} (SE {:.4})",
            rd.tau_conventional, rd.se_conventional
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '\u{2020}' 0.1"
        )?;
        writeln!(
            f,
            "Note: Robust = bias-corrected Wald estimate with the CCT (2014) robust standard \
             error, as rdrobust(fuzzy = ); Conventional = ratio of the conventional jumps with \
             the joint delta-method standard error."
        )?;

        write_warnings_and_missing(f, &self.warnings, &self.missing)
    }
}

// ============================================================================
// Parameter validation
// ============================================================================

/// Largest local-polynomial order `p` accepted by the RD estimators.
///
/// This is a method-level cap and holds in every mode,
/// not only on the hosted server. rdrobust allows orders 0..=20, but with the
/// uncentred polynomial design used here orders above 4 are already
/// numerically meaningless; 10 leaves ample room while keeping every design
/// matrix and the pilot fit (order at most 12) small. The bias-correction
/// order `q` may be at most `MAX_RD_ORDER + 1`.
pub(crate) const MAX_RD_ORDER: usize = 10;

/// Check the parameters of an [`RdConfig`] that do not depend on the data.
///
/// The checks run in this order: the polynomial order `p`, the bias order `q`
/// (only after `p`, so `p + 1` cannot overflow), the bandwidths `h` and `b`
/// when given, `rho`, the confidence `level`, `nnmatch`, the right-side
/// bandwidths `h_right` and `b_right` when given (each finite and positive,
/// and only together with `h` or `b`), and last `q > p`: rdrobust requires
/// the bias-correction order to exceed the estimation order. Every failure is
/// an `EconError::InvalidSpecification` naming the parameter, the value and
/// the bound.
pub(crate) fn validate_rd_config(config: &RdConfig) -> EconResult<()> {
    let spec = |message: String| Err(EconError::InvalidSpecification { message });
    let p = config.p;
    if p > MAX_RD_ORDER {
        return spec(format!(
            "Polynomial order p = {p} is too large: local polynomial RD supports p from 0 to \
             {MAX_RD_ORDER} (p = 1, local linear, is the standard choice)."
        ));
    }
    let q = config.q.unwrap_or(p + 1);
    if q > MAX_RD_ORDER + 1 {
        return spec(format!(
            "Bias-correction order q = {q} is too large: q can be at most {} (the default is p + 1).",
            MAX_RD_ORDER + 1
        ));
    }
    if let Some(h) = config.h {
        if !(h.is_finite() && h > 0.0) {
            return spec(format!(
                "Main bandwidth h must be a finite number greater than 0 (got {h})."
            ));
        }
    }
    if let Some(b) = config.b {
        if !(b.is_finite() && b > 0.0) {
            return spec(format!(
                "Bias bandwidth b must be a finite number greater than 0 (got {b})."
            ));
        }
    }
    let rho = config.rho;
    if !(rho.is_finite() && rho > 0.0) {
        return spec(format!(
            "rho (the ratio h/b, as rdrobust: b = h / rho) must be a finite number greater \
             than 0 (got {rho})."
        ));
    }
    let level = config.level;
    if !(level.is_finite() && level > 0.0 && level < 1.0) {
        return spec(format!(
            "Confidence level must be strictly between 0 and 1 (got {level}); use 0.95 for a 95% \
             interval."
        ));
    }
    if config.nnmatch == 0 {
        return spec(
            "nnmatch (nearest neighbours for the variance) must be at least 1 (got 0).".to_string(),
        );
    }
    validate_scaleregul(config.scaleregul)?;
    for (name, value) in [("h_right", config.h_right), ("b_right", config.b_right)] {
        if let Some(v) = value
            && !(v.is_finite() && v > 0.0)
        {
            return spec(format!(
                "Right-side bandwidth {name} must be a finite number greater than 0 (got {v})."
            ));
        }
    }
    if config.h_right.is_some() && config.h.is_none() {
        return spec("h_right needs h, the left-side main bandwidth.".to_string());
    }
    if config.b_right.is_some() && config.b.is_none() {
        return spec("b_right needs b, the left-side bias bandwidth.".to_string());
    }
    if q <= p {
        return spec(format!(
            "Bias-correction order q = {q} must be greater than the estimation order p = {p} \
             (rdrobust requires q > p; the default is p + 1)."
        ));
    }
    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Build polynomial design matrix of order p centered at cutoff.
/// Returns matrix with columns [1, (x-c), (x-c)^2, ..., (x-c)^p].
///
/// The `n x (p + 1)` matrix is allocated through `resource::try_zeros2`, so a
/// size above the allocation ceiling is a `ResourceLimit` error.
fn build_polynomial_matrix(x: ArrayView1<f64>, cutoff: f64, p: usize) -> EconResult<Array2<f64>> {
    let n = x.len();
    // A saturated column count cannot be allocated, so try_zeros2 reports it.
    let mut matrix =
        crate::resource::try_zeros2(n, p.saturating_add(1), "RD polynomial design matrix")?;

    for i in 0..n {
        let dx = x[i] - cutoff;
        let mut power = 1.0;
        for j in 0..=p {
            matrix[[i, j]] = power;
            power *= dx;
        }
    }

    Ok(matrix)
}

// ============================================================================
// The rdrobust estimator for given bandwidths (rdrobust 4.0.0, `rdrobust()`)
// ============================================================================

/// The rows of one side of the cutoff that enter the local fits, sorted by the
/// running variable: rdrobust's `ind` (positive kernel weight under max(h, b)).
struct SideData {
    /// "left" or "right", for messages.
    side: &'static str,
    /// Running variable, ascending (stable: ties keep their row order).
    x: Vec<f64>,
    /// Data columns (outcome, and the treatment of a fuzzy fit), m x k, in `x` order.
    d: Array2<f64>,
    /// Kernel weights K((x - c)/h)/h and K((x - c)/b)/b of the kept rows.
    w_h: Vec<f64>,
    w_b: Vec<f64>,
    /// Rows of the side with positive weight under h (rdrobust `N_h`) and under b (`N_b`).
    n_h: usize,
    n_b: usize,
}

/// The local fits of one side, for every data column at once.
struct SideFit {
    /// Order-p local-polynomial coefficients, (p + 1) x k.
    beta_p: Array2<f64>,
    /// Bias-corrected coefficients invG_p Q_q' D, (p + 1) x k.
    beta_bc: Array2<f64>,
    /// First row of invG_p: the intercept's weights.
    a: Array1<f64>,
    /// W_h R_p, m x (p + 1).
    rx: Array2<f64>,
    /// rdrobust's Q_q, m x (p + 1).
    qq: Array2<f64>,
    /// Residuals for the conventional (order p) variance, m x k.
    res_h: Array2<f64>,
    /// Residuals for the robust (order q) variance; `None` when they are `res_h` (the
    /// nearest-neighbour residuals do not depend on the fit).
    res_b: Option<Array2<f64>>,
    n_h: usize,
    n_b: usize,
    /// Ill-conditioning notes from the matrix inversions.
    warnings: Vec<String>,
}

impl SideFit {
    /// The residuals of the robust variance (`res_b`, or `res_h` when they are the same).
    fn robust_residuals(&self) -> &Array2<f64> {
        self.res_b.as_ref().unwrap_or(&self.res_h)
    }
}

/// Select and sort the rows of one side (`left`: x < c, else x >= c) with positive
/// kernel weight under max(h, b), and count those under h and under b.
fn select_side(
    x: ArrayView1<f64>,
    cols: &[ArrayView1<f64>],
    cutoff: f64,
    left: bool,
    h: f64,
    b: f64,
    kernel: KernelType,
) -> EconResult<SideData> {
    crate::resource::checkpoint()?;
    let side = if left { "left" } else { "right" };
    // rdrobust keeps the rows with weight under b, or under h when h > b.
    let by_h = h > b;
    let (mut n_h, mut n_b) = (0usize, 0usize);
    let mut kept: Vec<(usize, f64, f64)> =
        crate::resource::try_with_capacity(x.len(), "RD estimation window")?;
    for (i, &xi) in x.iter().enumerate() {
        let on_side = if left { xi < cutoff } else { xi >= cutoff };
        if !on_side {
            continue;
        }
        let w_h = kernel.evaluate((xi - cutoff) / h) / h;
        let w_b = kernel.evaluate((xi - cutoff) / b) / b;
        n_h += usize::from(w_h > 0.0);
        n_b += usize::from(w_b > 0.0);
        if (by_h && w_h > 0.0) || (!by_h && w_b > 0.0) {
            kept.push((i, w_h, w_b));
        }
    }
    // Stable sort by x (R's order()); -0.0 and 0.0 are one value.
    let key = |v: f64| if v == 0.0 { 0.0 } else { v };
    kept.sort_by(|a, b| key(x[a.0]).total_cmp(&key(x[b.0])));

    let m = kept.len();
    let k = cols.len();
    let mut d = crate::resource::try_zeros2(m, k, "RD side data")?;
    let mut xs: Vec<f64> = crate::resource::try_with_capacity(m, "RD side data")?;
    let mut w_h: Vec<f64> = crate::resource::try_with_capacity(m, "RD kernel weights")?;
    let mut w_b: Vec<f64> = crate::resource::try_with_capacity(m, "RD kernel weights")?;
    for (r, &(i, wh, wb)) in kept.iter().enumerate() {
        xs.push(x[i]);
        w_h.push(wh);
        w_b.push(wb);
        for (c, col) in cols.iter().enumerate() {
            d[[r, c]] = col[i];
        }
    }
    Ok(SideData {
        side,
        x: xs,
        d,
        w_h,
        w_b,
        n_h,
        n_b,
    })
}

/// (R' diag(w) R)^-1 through `safe_inverse`; a failure is `SingularMatrix` naming the
/// side, an ill-conditioned matrix a warning.
fn weighted_gram_inverse(
    r: ArrayView2<f64>,
    w: &[f64],
    cols: usize,
    what: &str,
    side: &str,
    warnings: &mut Vec<String>,
) -> EconResult<Array2<f64>> {
    let mut g = Array2::<f64>::zeros((cols, cols));
    for (i, &wi) in w.iter().enumerate() {
        for j in 0..cols {
            let rij = wi * r[[i, j]];
            for l in 0..cols {
                g[[j, l]] += rij * r[[i, l]];
            }
        }
    }
    match safe_inverse(&g.view()) {
        Ok((inv, None)) => Ok(inv),
        Ok((inv, Some(cond))) => {
            warnings.push(format!(
                "The {what} on the {side} side of the cutoff is ill-conditioned (condition \
                 number {cond:.2e}); a pseudoinverse was used."
            ));
            Ok(inv)
        }
        Err(e) => Err(EconError::SingularMatrix {
            context: format!("{what} on the {side} side of the cutoff"),
            suggestion: format!(
                "Use a larger bandwidth or a lower polynomial order; too few distinct values \
                 of the running variable lie inside the window ({e})."
            ),
        }),
    }
}

/// Residuals `w_i (D_i - R_i beta)` for the HC0-HC3 variances: HC1 scales by
/// sqrt(m / (m - dim)) with m the rows in the window, HC2/HC3 by the leverage
/// h_ii = w_i R_i' invG R_i (floored at 1 - 1e-8, as rdrobust).
pub(crate) fn hc_residuals(
    r: ArrayView2<f64>,
    w: &[f64],
    inv_g: &Array2<f64>,
    beta: &Array2<f64>,
    d: &Array2<f64>,
    vce: VceType,
) -> EconResult<Array2<f64>> {
    let (m, k) = d.dim();
    let dim = r.ncols();
    let mut res = crate::resource::try_zeros2(m, k, "RD residuals")?;
    for i in 0..m {
        let row = r.row(i);
        let scale = match vce {
            VceType::Hc0 | VceType::Nn => 1.0,
            VceType::Hc1 => (m as f64 / m.saturating_sub(dim) as f64).sqrt(),
            VceType::Hc2 | VceType::Hc3 => {
                // R_i' invG R_i without a temporary (rdrobust's
                // rowSums((R %*% invG) * (R * W)))
                let mut quad = 0.0;
                for j in 0..dim {
                    let mut acc = 0.0;
                    for l in 0..dim {
                        acc += inv_g[[j, l]] * row[l];
                    }
                    quad += row[j] * acc;
                }
                let hii = w[i] * quad;
                let one_minus = (1.0 - hii).max(1e-8);
                if vce == VceType::Hc2 {
                    (1.0 / one_minus).sqrt()
                } else {
                    1.0 / one_minus
                }
            }
        };
        for c in 0..k {
            let fitted = row.dot(&beta.column(c));
            res[[i, c]] = scale * (d[[i, c]] - fitted);
        }
    }
    Ok(res)
}

/// rdrobust's nearest-neighbour residuals (`rdrobust_res`, `vce = "nn"`) on sorted `x`,
/// for every column of `d` at once.
///
/// Rows with equal x form a block (rdrobust's `rle`), and every row of a block has the
/// same neighbours: its own block plus whole neighbouring blocks, added on the nearer
/// side (both sides when equally far) until the window holds at least
/// `min(nnmatch, m - 1)` other rows. With `J_i` those rows and `S` their sum (own block
/// included), `res_i = sqrt(J_i / (J_i + 1)) (d_i - (S - d_i) / J_i)`. Block sums are
/// formed once, so the cost is O(m k) plus O(B min(nnmatch, m) k) for B blocks.
pub(crate) fn nn_residuals(x: &[f64], d: &Array2<f64>, nnmatch: usize) -> EconResult<Array2<f64>> {
    let (m, k) = d.dim();
    let mut res = crate::resource::try_zeros2(m, k, "RD nearest-neighbour residuals")?;
    if m == 0 {
        return Ok(res);
    }
    let mut starts: Vec<usize> = crate::resource::try_with_capacity(m, "RD tie blocks")?;
    starts.push(0);
    for i in 1..m {
        if x[i] != x[i - 1] {
            starts.push(i);
        }
    }
    let nb = starts.len();
    let end = |blk: usize| if blk + 1 < nb { starts[blk + 1] } else { m };
    let len = |blk: usize| end(blk) - starts[blk];
    let mut sums = crate::resource::try_zeros2(nb, k, "RD tie blocks")?;
    for blk in 0..nb {
        for i in starts[blk]..end(blk) {
            for c in 0..k {
                sums[[blk, c]] += d[[i, c]];
            }
        }
    }

    let target = nnmatch.min(m - 1);
    let mut total = vec![0.0; k];
    for blk in 0..nb {
        let v = x[starts[blk]];
        let (mut lo, mut hi) = (blk, blk);
        let mut cnt = len(blk) - 1;
        while cnt < target {
            // cnt < m - 1, so at least one side still has a block
            if lo == 0 {
                hi += 1;
                cnt += len(hi);
            } else if hi == nb - 1 {
                lo -= 1;
                cnt += len(lo);
            } else {
                let dl = v - x[starts[lo - 1]];
                let dr = x[starts[hi + 1]] - v;
                if dl > dr {
                    hi += 1;
                    cnt += len(hi);
                } else if dl < dr {
                    lo -= 1;
                    cnt += len(lo);
                } else {
                    hi += 1;
                    lo -= 1;
                    cnt += len(hi) + len(lo);
                }
            }
        }
        if cnt == 0 {
            return Err(EconError::InvalidSpecification {
                message: "Nearest-neighbour residuals need at least two observations inside \
                          the estimation window on each side of the cutoff."
                    .to_string(),
            });
        }
        total.iter_mut().for_each(|t| *t = 0.0);
        for bb in lo..=hi {
            for (c, t) in total.iter_mut().enumerate() {
                *t += sums[[bb, c]];
            }
        }
        let ji = cnt as f64;
        let scale = (ji / (ji + 1.0)).sqrt();
        for i in starts[blk]..end(blk) {
            for c in 0..k {
                res[[i, c]] = scale * (d[[i, c]] - (total[c] - d[[i, c]]) / ji);
            }
        }
    }
    Ok(res)
}

/// The local fits of one side (rdrobust `rdrobust()`, lines for `R_q`, `invG`, `Q_q`,
/// `beta_p`, `beta_bc` and the residuals).
fn fit_side(
    side: &SideData,
    cutoff: f64,
    h: f64,
    p: usize,
    q: usize,
    vce: VceType,
    nnmatch: usize,
) -> EconResult<SideFit> {
    let (m, k) = side.d.dim();
    let mut warnings = Vec::new();

    // R_q = [1, (x - c), ..., (x - c)^q]; R_p is its first p + 1 columns (a view).
    let rq = build_polynomial_matrix(ArrayView1::from(side.x.as_slice()), cutoff, q)?;
    let rp = rq.slice(ndarray::s![.., ..=p]);
    let inv_gp = weighted_gram_inverse(
        rp,
        &side.w_h,
        p + 1,
        &format!("order-{p} X'WX (bandwidth h)"),
        side.side,
        &mut warnings,
    )?;
    let inv_gq = weighted_gram_inverse(
        rq.view(),
        &side.w_b,
        q + 1,
        &format!("order-{q} X'WX (bias bandwidth b)"),
        side.side,
        &mut warnings,
    )?;

    // L = R_p' (W_h .* u^(p+1)), u = (x - c)/h
    let pow = i32::try_from(p + 1).unwrap_or(i32::MAX);
    let mut l_vec = Array1::<f64>::zeros(p + 1);
    let mut rx = crate::resource::try_zeros2(m, p + 1, "RD weighted design")?;
    for i in 0..m {
        let u = (side.x[i] - cutoff) / h;
        let wu = side.w_h[i] * u.powi(pow);
        for j in 0..=p {
            l_vec[j] += rp[[i, j]] * wu;
            rx[[i, j]] = rp[[i, j]] * side.w_h[i];
        }
    }
    // Q_q,i = W_h,i R_p,i - h^(p+1) W_b,i (R_q,i . invG_q[:, p+1]) L
    let hp = h.powi(pow);
    let col = inv_gq.column(p + 1);
    let mut qq = crate::resource::try_zeros2(m, p + 1, "RD bias-correction weights")?;
    for i in 0..m {
        let ci = rq.row(i).dot(&col);
        let shift = hp * side.w_b[i] * ci;
        for j in 0..=p {
            qq[[i, j]] = rx[[i, j]] - shift * l_vec[j];
        }
    }

    // beta_p = invG_p R_p' W_h D, beta_q = invG_q R_q' W_b D, beta_bc = invG_p Q_q' D
    let mut u_q = Array2::<f64>::zeros((q + 1, k));
    for i in 0..m {
        for j in 0..=q {
            let wr = side.w_b[i] * rq[[i, j]];
            for c in 0..k {
                u_q[[j, c]] += wr * side.d[[i, c]];
            }
        }
    }
    let beta_p = inv_gp.dot(&rx.t().dot(&side.d));
    let beta_q = inv_gq.dot(&u_q);
    let beta_bc = inv_gp.dot(&qq.t().dot(&side.d));

    let (res_h, res_b) = match vce {
        VceType::Nn => (nn_residuals(&side.x, &side.d, nnmatch)?, None),
        _ => (
            hc_residuals(rp, &side.w_h, &inv_gp, &beta_p, &side.d, vce)?,
            Some(hc_residuals(
                rq.view(),
                &side.w_b,
                &inv_gq,
                &beta_q,
                &side.d,
                vce,
            )?),
        ),
    };

    Ok(SideFit {
        beta_p,
        beta_bc,
        a: inv_gp.row(0).to_owned(),
        rx,
        qq,
        res_h,
        res_b,
        n_h: side.n_h,
        n_b: side.n_b,
        warnings,
    })
}

/// Variance of the intercept of a combination `s` of the data columns:
/// `(invG_p M invG_p)[0, 0]` with `M = sum_i (res_i . s)^2 RX_i RX_i'`, that is
/// `sum_i (res_i . s)^2 (a . RX_i)^2` (rdrobust `rdrobust_vce` without clusters).
fn intercept_variance(rx: &Array2<f64>, res: &Array2<f64>, a: &Array1<f64>, s: &[f64]) -> f64 {
    let mut v = 0.0;
    for i in 0..rx.nrows() {
        let r: f64 = s.iter().enumerate().map(|(c, sc)| res[[i, c]] * sc).sum();
        let g = rx.row(i).dot(a);
        let t = r * g;
        v += t * t;
    }
    v
}

/// Both sides of one RD fit, with the bandwidths and counts behind it.
struct RdFit {
    left: SideFit,
    right: SideFit,
    n_left: usize,
    n_right: usize,
    h: (f64, f64),
    b: (f64, f64),
    p: usize,
    q: usize,
    /// Where h and b came from (and the selection, when it ran).
    resolved: Resolved,
    warnings: Vec<String>,
}

/// The bandwidths of one fit and where they came from.
struct Resolved {
    h: (f64, f64),
    b: (f64, f64),
    h_origin: BandwidthOrigin,
    b_origin: BandwidthOrigin,
    selection: Option<BwSelection>,
}

/// The bandwidths of a fit: h given or selected by `config.bwselect` (rdbwselect, with the
/// treatment `t` of a fuzzy design); b given, `h / rho`, or selected. The columns are
/// made contiguous only when the selector runs.
fn resolve_bandwidths(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    t: Option<&ArrayView1<f64>>,
    cutoff: f64,
    config: &RdConfig,
) -> EconResult<Resolved> {
    let given_b = config.b.map(|b| (b, config.b_right.unwrap_or(b)));
    let rho = config.rho;
    if let Some(h) = config.h {
        let h = (h, config.h_right.unwrap_or(h));
        let (b, b_origin) = match given_b {
            Some(b) => (b, BandwidthOrigin::UserSupplied),
            None => ((h.0 / rho, h.1 / rho), BandwidthOrigin::RhoTimesH),
        };
        return Ok(Resolved {
            h,
            b,
            h_origin: BandwidthOrigin::UserSupplied,
            b_origin,
            selection: None,
        });
    }
    let xs = as_slice(x, "RD running variable")?;
    let ys = as_slice(y, "RD outcome")?;
    let ts = t.map(|t| as_slice(t, "RD treatment")).transpose()?;
    let sel = rdbwselect(
        &xs,
        &ys,
        ts.as_deref(),
        cutoff,
        config.bwselect,
        &BwSelectOptions::from_config(config),
    )?;
    let bw = sel.bandwidths;
    let h = (bw.h_left, bw.h_right);
    let (b, b_origin) = match given_b {
        Some(b) => (b, BandwidthOrigin::UserSupplied),
        // A rho other than its default can only mean the caller asked for that ratio.
        None if config.rho_with_selected_h || rho != 1.0 => {
            ((h.0 / rho, h.1 / rho), BandwidthOrigin::RhoTimesH)
        }
        None => ((bw.b_left, bw.b_right), BandwidthOrigin::Selected),
    };
    Ok(Resolved {
        h,
        b,
        h_origin: BandwidthOrigin::Selected,
        b_origin,
        selection: Some(sel),
    })
}

/// A contiguous slice of a column view (one copy, under the allocation ceiling, when the
/// view is strided).
fn as_slice<'a>(v: &'a ArrayView1<'_, f64>, what: &str) -> EconResult<std::borrow::Cow<'a, [f64]>> {
    match v.as_slice() {
        Some(s) => Ok(std::borrow::Cow::Borrowed(s)),
        None => {
            let mut owned = crate::resource::try_with_capacity(v.len(), what)?;
            owned.extend(v.iter().copied());
            Ok(std::borrow::Cow::Owned(owned))
        }
    }
}

impl RdFit {
    /// Conventional and bias-corrected jumps of column `c`.
    fn jumps(&self, c: usize) -> (f64, f64) {
        (
            self.right.beta_p[[0, c]] - self.left.beta_p[[0, c]],
            self.right.beta_bc[[0, c]] - self.left.beta_bc[[0, c]],
        )
    }

    /// Conventional and robust SEs of the jump of the combination `s` of the columns.
    fn ses(&self, s: &[f64]) -> (f64, f64) {
        let (l, r) = (&self.left, &self.right);
        let v_cl = intercept_variance(&l.rx, &l.res_h, &l.a, s)
            + intercept_variance(&r.rx, &r.res_h, &r.a, s);
        let v_rb = intercept_variance(&l.qq, l.robust_residuals(), &l.a, s)
            + intercept_variance(&r.qq, r.robust_residuals(), &r.a, s);
        (v_cl.sqrt(), v_rb.sqrt())
    }
}

/// Fit the local polynomials of every column in `cols` (row-aligned with `x`) on both
/// sides of `cutoff`. When `config.h` is not given, rdbwselect runs on `cols[0]` (the
/// outcome), with `cols[1]` as the treatment of a fuzzy design.
fn fit_rd(
    x: ArrayView1<f64>,
    cols: &[ArrayView1<f64>],
    cutoff: f64,
    config: &RdConfig,
) -> EconResult<RdFit> {
    let n = x.len();
    if n < 20 {
        return Err(EconError::InsufficientData {
            required: 20,
            provided: n,
            context: "RD estimation requires at least 20 observations".to_string(),
        });
    }

    // Count observations on each side
    let n_left = x.iter().filter(|&&xi| xi < cutoff).count();
    let n_right = x.iter().filter(|&&xi| xi >= cutoff).count();
    if n_left < 5 || n_right < 5 {
        return Err(EconError::InsufficientData {
            required: 5,
            provided: n_left.min(n_right),
            context: "Need at least 5 observations on each side of the cutoff".to_string(),
        });
    }

    // Polynomial orders (validated: p <= MAX_RD_ORDER, p < q <= MAX_RD_ORDER + 1)
    let p = config.p;
    let q = config.q.unwrap_or(p + 1);

    // Bandwidths: given or selected as rdbwselect; b given, h / rho or selected.
    let resolved = resolve_bandwidths(&x, &cols[0], cols.get(1), cutoff, config)?;
    let ((h_left, h_right), (b_left, b_right)) = (resolved.h, resolved.b);

    let left = select_side(x, cols, cutoff, true, h_left, b_left, config.kernel)?;
    let right = select_side(x, cols, cutoff, false, h_right, b_right, config.kernel)?;

    // A local polynomial of order p needs at least p + 2 points with positive
    // kernel weight on each side; fewer gives a degenerate (exact or singular)
    // fit, so this is an error rather than a warning.
    for s in [&left, &right] {
        if s.n_h < p + 2 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Too few observations inside the bandwidth for polynomial order p = {p}: \
                     {} on the {} side of the cutoff, at least {} are needed. Use a \
                     larger bandwidth h or a lower order p.",
                    s.n_h,
                    s.side,
                    p + 2
                ),
            });
        }
    }
    // The bias fit of order q needs at least q + 1 points inside b on each side.
    for s in [&left, &right] {
        if s.n_b < q + 1 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Too few observations inside the bias bandwidth b for the bias-correction \
                     polynomial of order q = {q}: {} on the {} side, at least {} are \
                     needed. Use a larger b (or a smaller rho).",
                    s.n_b,
                    s.side,
                    q + 1
                ),
            });
        }
    }
    // HC1 scales the order-q residuals by sqrt(m / (m - q - 1)).
    if config.vce == VceType::Hc1 {
        for s in [&left, &right] {
            if s.x.len() <= q + 1 {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Too few observations on the {} side for HC1 with q = {q}: {} inside \
                         the estimation window, more than {} are needed. Use a larger \
                         bandwidth or another vce.",
                        s.side,
                        s.x.len(),
                        q + 1
                    ),
                });
            }
        }
    }

    crate::resource::checkpoint()?;
    let left = fit_side(&left, cutoff, h_left, p, q, config.vce, config.nnmatch)?;
    crate::resource::checkpoint()?;
    let right = fit_side(&right, cutoff, h_right, p, q, config.vce, config.nnmatch)?;
    let mut warnings: Vec<String> = resolved
        .selection
        .as_ref()
        .map(|s| s.info.warnings.clone())
        .unwrap_or_default();
    warnings.extend(left.warnings.iter().cloned());
    warnings.extend(right.warnings.iter().cloned());

    Ok(RdFit {
        left,
        right,
        n_left,
        n_right,
        h: (h_left, h_right),
        b: (b_left, b_right),
        p,
        q,
        resolved,
        warnings,
    })
}

/// The standard normal and its critical value for a two-sided interval at `level`.
fn normal_and_z(level: f64) -> (Normal, f64) {
    // Normal::new(0, 1) cannot fail; level is validated in (0, 1).
    let normal = Normal::standard();
    let z = -normal.inverse_cdf((1.0 - level) / 2.0);
    (normal, z)
}

/// Two-sided normal p-value of `est / se` (upper tail with `sf`, X-17). A zero SE gives
/// p = 0 for a nonzero estimate, as R's `2 * pnorm(-Inf)` (a first stage whose take-up is
/// constant on each side has nearest-neighbour residuals 0), and p = 1 for a zero one.
fn two_sided_p(normal: &Normal, est: f64, se: f64) -> f64 {
    if se == 0.0 {
        return if est == 0.0 { 1.0 } else { 0.0 };
    }
    2.0 * normal.sf((est / se).abs())
}

/// The sharp RD result of data column `c` of a fit.
fn rd_result_from(
    fit: &RdFit,
    c: usize,
    outcome: &str,
    running_var: &str,
    cutoff: f64,
    config: &RdConfig,
) -> RdResult {
    let k = fit.left.beta_p.ncols();
    let mut s = vec![0.0; k];
    s[c] = 1.0;
    let (tau_conventional, tau_bc) = fit.jumps(c);
    let (se_conventional, se_robust) = fit.ses(&s);
    let (normal, z) = normal_and_z(config.level);
    let ci = |est: f64, se: f64| (est - z * se, est + z * se);
    let p_robust = two_sided_p(&normal, tau_bc, se_robust);
    RdResult {
        outcome: outcome.to_string(),
        running_var: running_var.to_string(),
        cutoff,
        n_left: fit.n_left,
        n_right: fit.n_right,
        n_eff_left: fit.left.n_h,
        n_eff_right: fit.right.n_h,
        n_b_left: fit.left.n_b,
        n_b_right: fit.right.n_b,
        tau_conventional,
        tau_bc,
        tau_robust: tau_bc,
        se_conventional,
        se_bc: se_conventional,
        se_robust,
        ci_conventional: ci(tau_conventional, se_conventional),
        ci_bc: ci(tau_bc, se_conventional),
        ci_robust: ci(tau_bc, se_robust),
        level: config.level,
        p_conventional: two_sided_p(&normal, tau_conventional, se_conventional),
        p_bc: two_sided_p(&normal, tau_bc, se_conventional),
        p_robust,
        significance: SignificanceLevel::from_p_value(p_robust),
        h_left: fit.h.0,
        h_right: fit.h.1,
        b_left: fit.b.0,
        b_right: fit.b.1,
        bwselect: config.bwselect,
        bandwidth_given: fit.resolved.h_origin == BandwidthOrigin::UserSupplied,
        h_origin: fit.resolved.h_origin,
        b_origin: fit.resolved.b_origin,
        p: fit.p,
        q: fit.q,
        kernel: config.kernel,
        vce: config.vce,
        coef_left: fit.left.beta_p.column(c).to_vec(),
        coef_right: fit.right.beta_p.column(c).to_vec(),
        bias: tau_conventional - tau_bc,
        warnings: fit.warnings.clone(),
        missing: MissingSummary::none(fit.n_left + fit.n_right),
    }
}

// ============================================================================
// Main Estimation Functions
// ============================================================================

/// Run Sharp Regression Discontinuity estimation.
///
/// For given bandwidths (`config.h`, `config.b`, and `h_right`/`b_right` for unequal
/// sides) the result is what `rdrobust::rdrobust` 4.0.0 reports: on each side the rows
/// with positive kernel weight under max(h, b), sorted by the running variable, are fitted
/// by weighted local polynomials of order p (bandwidth h) and q (bandwidth b); the
/// bias-corrected estimate uses the CCT (2014) `Q_q` weights; the conventional and robust
/// variances are sandwiches of the intercepts with nearest-neighbour residuals (whole tie
/// groups, `nnmatch` neighbours) or HC0-HC3 residuals. When `h` is not given, h and b are
/// `rdrobust::rdbwselect`'s ([`crate::econometrics::rd_bwselect`]), b selected separately;
/// a given h without b gives `b = h / rho`.
/// Covariates, clustering, weights and derivatives are not supported.
///
/// # Arguments
/// * `dataset` - The dataset
/// * `outcome` - Name of the outcome variable (Y)
/// * `running_var` - Name of the running variable (X)
/// * `cutoff` - Cutoff value for the running variable
/// * `config` - RD configuration options
///
/// # Returns
/// RdResult containing point estimates, standard errors, and confidence intervals
/// for conventional, bias-corrected, and robust inference.
///
/// # References
///
/// Calonico, Cattaneo & Titiunik (2014). Robust Nonparametric Confidence Intervals
/// for Regression-Discontinuity Designs. Econometrica 82(6): 2295-2326.
pub fn run_rd(
    dataset: &Dataset,
    outcome: &str,
    running_var: &str,
    cutoff: f64,
    config: RdConfig,
) -> EconResult<RdResult> {
    validate_rd_config(&config)?;

    // Rows with a missing outcome or running variable are dropped, as rdrobust does
    // (complete.cases); infinite values and text columns are refused by name.
    let cols = ColumnRequest::new(dataset.df())
        .numeric(outcome)
        .numeric(running_var)
        .na(NaPolicy::Omit)
        .extract()?;
    let y = ArrayView1::from(cols.numeric(outcome));
    let x = ArrayView1::from(cols.numeric(running_var));

    let fit = fit_rd(x, &[y], cutoff, &config)?;
    let mut result = rd_result_from(&fit, 0, outcome, running_var, cutoff, &config);
    attach_missing(&mut result.warnings, &mut result.missing, cols.missing());
    Ok(result)
}

/// Record the rows the extraction dropped: the summary, and its note among the warnings
/// (so printing paths that only show warnings report it too).
fn attach_missing(warnings: &mut Vec<String>, slot: &mut MissingSummary, missing: &MissingSummary) {
    *slot = missing.clone();
    if let Some(note) = missing.note() {
        warnings.push(note);
    }
}

/// Select the RD bandwidths h and b as `rdrobust::rdbwselect` (sharp design), without
/// running the estimation: `rd_bandwidth_with` with the default configuration and `p`,
/// `kernel` and `bwselect`.
pub fn rd_bandwidth(
    dataset: &Dataset,
    outcome: &str,
    running_var: &str,
    cutoff: f64,
    p: usize,
    kernel: KernelType,
    bwselect: BandwidthMethod,
) -> EconResult<RdBandwidth> {
    let mut config = RdConfig::default();
    config.p = p;
    config.kernel = kernel;
    config.bwselect = bwselect;
    rd_bandwidth_with(dataset, outcome, running_var, None, cutoff, &config)
}

/// Bandwidth selection only, with every selection option of `config` (`bwselect`, p, q,
/// kernel, vce, nnmatch, scaleregul), as `rdrobust::rdbwselect`. `treatment` selects for a
/// fuzzy design, as `rdbwselect(fuzzy = )`. `h`, `h_right`, `b` and `b_right` must be
/// `None`. Rows with a missing outcome, running variable or treatment are dropped.
pub fn rd_bandwidth_with(
    dataset: &Dataset,
    outcome: &str,
    running_var: &str,
    treatment: Option<&str>,
    cutoff: f64,
    config: &RdConfig,
) -> EconResult<RdBandwidth> {
    validate_rd_config(config)?;
    if config.h.is_some()
        || config.h_right.is_some()
        || config.b.is_some()
        || config.b_right.is_some()
    {
        return Err(EconError::InvalidSpecification {
            message: "rd_bandwidth_with selects h and b; remove h, b, h_right and b_right from \
                      the configuration, or call run_rd"
                .to_string(),
        });
    }
    let mut request = ColumnRequest::new(dataset.df())
        .numeric(outcome)
        .numeric(running_var);
    if let Some(t) = treatment {
        request = request.numeric(t);
    }
    let cols = request.na(NaPolicy::Omit).extract()?;
    let y = ArrayView1::from(cols.numeric(outcome));
    let x = ArrayView1::from(cols.numeric(running_var));
    let t = treatment.map(|t| ArrayView1::from(cols.numeric(t)));

    let resolved = resolve_bandwidths(&x, &y, t.as_ref(), cutoff, config)?;
    let sel = resolved.selection.ok_or_else(|| {
        EconError::Internal("rd_bandwidth_with: the bandwidths were not selected".to_string())
    })?;
    let info = sel.info;
    let mut warnings = info.warnings;
    let mut missing = MissingSummary::default();
    attach_missing(&mut warnings, &mut missing, cols.missing());
    Ok(RdBandwidth {
        h_left: resolved.h.0,
        h_right: resolved.h.1,
        b_left: resolved.b.0,
        b_right: resolved.b.1,
        bwselect: config.bwselect,
        kernel: config.kernel,
        p: config.p,
        n_left: info.n_left,
        n_right: info.n_right,
        missing,
        q: info.q,
        vce: info.vce,
        nnmatch: info.nnmatch,
        scaleregul: info.scaleregul,
        n_h_left: sel.n_h_left,
        n_h_right: sel.n_h_right,
        m_left: info.m_left,
        m_right: info.m_right,
        mass_points: info.mass_points,
        bwcheck: info.bwcheck,
        pilot_bandwidth: info.pilot_bandwidth,
        b_origin: resolved.b_origin,
        treatment: treatment.map(str::to_string),
        warnings,
    })
}

/// Run Fuzzy Regression Discontinuity estimation.
///
/// Fuzzy RD is used when treatment is not perfectly determined by the running
/// variable crossing the cutoff. Instead, the probability of treatment jumps
/// at the cutoff.
///
/// As `rdrobust(y, x, fuzzy = t)`: the outcome and the treatment are fitted once, on one
/// bandwidth (when `h` is not given, rdbwselect selects h and b for the fuzzy design, with
/// the treatment, as `rdbwselect(fuzzy = )`), with the same
/// local-polynomial matrices. The conventional estimate is the Wald ratio
/// `tau_Y / tau_T` of the conventional jumps; the bias-corrected one is
/// `tau - s'(tau_Y - tau_Y,bc, tau_T - tau_T,bc)` with `s = (1/tau_T, -tau_Y/tau_T²)`; both
/// variances come from the combined residuals `res_Y / tau_T - tau_Y / tau_T² res_T`, so the
/// covariance of the two jumps is included. Every interval uses `config.level`.
///
/// # Arguments
/// * `dataset` - The dataset
/// * `outcome` - Name of the outcome variable (Y)
/// * `running_var` - Name of the running variable (X)
/// * `treatment` - Name of the treatment variable (D, binary)
/// * `cutoff` - Cutoff value
/// * `config` - RD configuration
pub fn run_fuzzy_rd(
    dataset: &Dataset,
    outcome: &str,
    running_var: &str,
    treatment: &str,
    cutoff: f64,
    config: RdConfig,
) -> EconResult<FuzzyRdResult> {
    validate_rd_config(&config)?;

    // Complete cases of outcome, running variable and treatment (rdrobust's na.ok).
    let cols = ColumnRequest::new(dataset.df())
        .numeric(outcome)
        .numeric(running_var)
        .numeric(treatment)
        .na(NaPolicy::Omit)
        .extract()?;
    let y = ArrayView1::from(cols.numeric(outcome));
    let x = ArrayView1::from(cols.numeric(running_var));
    let t = ArrayView1::from(cols.numeric(treatment));

    let fit = fit_rd(x, &[y, t], cutoff, &config)?;
    let (tau_y, tau_y_bc) = fit.jumps(0);
    let (tau_t, tau_t_bc) = fit.jumps(1);
    if tau_t.abs() < 1e-10 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "First stage is degenerate: the conventional treatment jump at the cutoff is \
                 {tau_t:e}, and |jump| must be at least 1e-10 for the Wald ratio; check the \
                 treatment column '{treatment}' and the bandwidth"
            ),
        });
    }
    let mut outcome_rd = rd_result_from(&fit, 0, outcome, running_var, cutoff, &config);
    let mut first_stage = rd_result_from(&fit, 1, treatment, running_var, cutoff, &config);
    for r in [&mut outcome_rd, &mut first_stage] {
        attach_missing(&mut r.warnings, &mut r.missing, cols.missing());
    }

    // Wald ratio and its bias correction (rdrobust's s_Y and B_F)
    let tau_fuzzy = tau_y / tau_t;
    let s = [1.0 / tau_t, -tau_y / (tau_t * tau_t)];
    let tau_fuzzy_bc = tau_fuzzy - (s[0] * (tau_y - tau_y_bc) + s[1] * (tau_t - tau_t_bc));
    let (se_fuzzy, se_fuzzy_robust) = fit.ses(&s);

    let (normal, z) = normal_and_z(config.level);
    let ci = |est: f64, se: f64| (est - z * se, est + z * se);
    let p_fuzzy = two_sided_p(&normal, tau_fuzzy, se_fuzzy);
    let p_fuzzy_robust = two_sided_p(&normal, tau_fuzzy_bc, se_fuzzy_robust);

    let first_stage_f = (tau_t / first_stage.se_conventional).powi(2);
    let mut warnings = fit.warnings.clone();
    let mut missing = MissingSummary::default();
    attach_missing(&mut warnings, &mut missing, cols.missing());
    if first_stage_f < 10.0 {
        warnings.push(format!(
            "Weak first stage: F = {first_stage_f:.2} (< 10); the fuzzy RD estimate and its \
             intervals are unreliable."
        ));
    }

    Ok(FuzzyRdResult {
        outcome_rd,
        first_stage,
        tau_fuzzy,
        se_fuzzy,
        p_fuzzy,
        ci_fuzzy: ci(tau_fuzzy, se_fuzzy),
        significance: SignificanceLevel::from_p_value(p_fuzzy_robust),
        treatment: treatment.to_string(),
        tau_fuzzy_bc,
        se_fuzzy_robust,
        ci_fuzzy_robust: ci(tau_fuzzy_bc, se_fuzzy_robust),
        p_fuzzy_robust,
        first_stage_tau: tau_t,
        first_stage_f,
        warnings,
        level: config.level,
        missing,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::load_case;
    use p2a_validation::{Check, Tol};
    use polars::prelude::*;

    #[test]
    fn test_kernel_and_bwselect_names_are_parsed_strictly() {
        for (name, kernel) in [
            ("triangular", KernelType::Triangular),
            ("TRI", KernelType::Triangular),
            ("Epanechnikov", KernelType::Epanechnikov),
            ("epa", KernelType::Epanechnikov),
            ("uniform", KernelType::Uniform),
            ("uni", KernelType::Uniform),
        ] {
            assert_eq!(KernelType::parse(name).unwrap(), kernel, "{name}");
        }
        for (name, method) in [
            ("mserd", BandwidthMethod::MseRd),
            ("MSETWO", BandwidthMethod::MseTwo),
            ("cerrd", BandwidthMethod::CerRd),
            ("certwo", BandwidthMethod::CerTwo),
            ("cer_two", BandwidthMethod::CerTwo),
        ] {
            assert_eq!(BandwidthMethod::parse(name).unwrap(), method, "{name}");
        }
        match KernelType::parse("gaussian") {
            Err(EconError::InvalidSpecification { message }) => assert!(
                message.starts_with("Unknown kernel 'gaussian'; use \"triangular\"")
                    && message.contains("\"epa\"")
                    && message.contains("\"uniform\""),
                "{message}"
            ),
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        match BandwidthMethod::parse("ik") {
            Err(EconError::InvalidSpecification { message }) => assert!(
                message.starts_with("Unknown bwselect 'ik'; use \"mserd\"")
                    && message.contains("\"certwo\""),
                "{message}"
            ),
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        assert_eq!(
            BandwidthMethod::parse("msesum").unwrap(),
            BandwidthMethod::MseSum
        );
        assert_eq!(
            BandwidthMethod::parse("cercomb2").unwrap(),
            BandwidthMethod::CerComb2
        );
    }

    fn spec_message<T: std::fmt::Debug>(r: EconResult<T>) -> String {
        match r {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_bandwidth_method_parses_aliases_and_rejects_unknown() {
        use BandwidthMethod::*;
        BandwidthMethod::choices().validate().unwrap();
        for (name, m) in [
            ("mserd", MseRd),
            ("MSERD", MseRd),
            ("mse", MseRd),
            ("msetwo", MseTwo),
            ("mse_two", MseTwo),
            ("mse-two", MseTwo),
            ("msesum", MseSum),
            ("msecomb1", MseComb1),
            ("MSEcomb2", MseComb2),
            ("cerrd", CerRd),
            ("cer", CerRd),
            ("certwo", CerTwo),
            ("cer_two", CerTwo),
            ("cersum", CerSum),
            ("cercomb1", CerComb1),
            ("cercomb2", CerComb2),
        ] {
            assert_eq!(name.parse::<BandwidthMethod>().unwrap(), m, "{name}");
        }
        for bad in ["ik", "cct", "cv", "manual", ""] {
            let msg = spec_message(bad.parse::<BandwidthMethod>());
            assert!(
                msg.contains("bwselect")
                    && msg.contains(
                        "mserd, msetwo, msesum, msecomb1, msecomb2, cerrd, certwo, cersum, \
                         cercomb1, cercomb2"
                    ),
                "{msg}"
            );
            let msg = spec_message(BandwidthMethod::parse(bad));
            assert!(
                msg.starts_with(&format!("Unknown bwselect '{bad}'; use \"mserd\""))
                    && msg.contains("\"cercomb2\""),
                "{msg}"
            );
        }
        assert!(
            BandwidthMethod::ALL
                .iter()
                .all(|m| m.name().parse::<BandwidthMethod>().unwrap() == *m)
        );
        assert_eq!(BandwidthMethod::from_str("certwo"), Some(CerTwo));
        assert_eq!(BandwidthMethod::from_str("ik"), None);
    }

    #[test]
    fn test_kernel_type_parses_aliases_and_rejects_unknown() {
        KernelType::choices().validate().unwrap();
        for (name, k) in [
            ("tri", KernelType::Triangular),
            ("Triangular", KernelType::Triangular),
            ("epa", KernelType::Epanechnikov),
            ("EPANECHNIKOV", KernelType::Epanechnikov),
            ("uni", KernelType::Uniform),
            ("uniform", KernelType::Uniform),
        ] {
            assert_eq!(name.parse::<KernelType>().unwrap(), k, "{name}");
        }
        let msg = spec_message("gaussian".parse::<KernelType>());
        assert!(
            msg.contains("kernel") && msg.contains("triangular, epanechnikov, uniform"),
            "{msg}"
        );
        let msg = spec_message(KernelType::parse("gaussian"));
        assert!(
            msg.starts_with("Unknown kernel 'gaussian'; use \"triangular\""),
            "{msg}"
        );
        assert_eq!(KernelType::from_str("epa"), Some(KernelType::Epanechnikov));
        assert_eq!(KernelType::from_str("gaussian"), None);
    }

    #[test]
    fn test_vce_type_parses_aliases_and_rejects_unknown() {
        VceType::choices().validate().unwrap();
        for (name, v) in [
            ("nn", VceType::Nn),
            ("Nearest", VceType::Nn),
            ("HC0", VceType::Hc0),
            ("hc1", VceType::Hc1),
            ("Hc2", VceType::Hc2),
            ("hc3", VceType::Hc3),
        ] {
            assert_eq!(name.parse::<VceType>().unwrap(), v, "{name}");
        }
        let msg = spec_message("robust".parse::<VceType>());
        assert!(msg.contains("vce") && msg.contains("hc1"), "{msg}");
        let msg = spec_message(VceType::parse("robust"));
        assert!(msg.starts_with("Unknown vce 'robust'; use \"nn\""), "{msg}");
    }

    fn create_sharp_rd_dataset() -> Dataset {
        // Create synthetic RD data with known treatment effect
        // Y = 1 + 0.5*X + 2*I(X >= 0) + noise
        // True treatment effect = 2
        let mut x_vals = Vec::new();
        let mut y_vals = Vec::new();

        // Use deterministic but random-like noise
        // Simple LCG-style pseudorandom: x_{n+1} = (a*x_n + c) mod m
        let mut seed: u64 = 12345;
        let noise = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((*s as f64) / (u64::MAX as f64) - 0.5) * 0.3
        };

        // Left of cutoff
        for i in 0..50 {
            let x = -2.0 + (i as f64) * 0.04;
            let n = noise(&mut seed);
            let y = 1.0 + 0.5 * x + n;
            x_vals.push(x);
            y_vals.push(y);
        }

        // Right of cutoff
        for i in 0..50 {
            let x = (i as f64) * 0.04;
            let n = noise(&mut seed);
            let y = 1.0 + 0.5 * x + 2.0 + n; // +2 is treatment effect
            x_vals.push(x);
            y_vals.push(y);
        }

        let df = df! {
            "outcome" => y_vals,
            "running" => x_vals
        }
        .unwrap();

        Dataset::new(df)
    }

    fn create_fuzzy_rd_dataset() -> Dataset {
        // Fuzzy RD: treatment probability jumps but not perfectly
        // P(D=1|X>=0) = 0.8, P(D=1|X<0) = 0.2
        // Y = 1 + 0.5*X + 3*D + noise
        // True LATE = 3
        let mut x_vals = Vec::new();
        let mut y_vals = Vec::new();
        let mut d_vals = Vec::new();

        // Use deterministic pseudorandom for treatment assignment and noise
        let mut seed: u64 = 54321;
        let rand = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(1103515245).wrapping_add(12345);
            (*s as f64) / (u64::MAX as f64)
        };

        // Left of cutoff: ~20% treated
        for _ in 0..50 {
            let r = rand(&mut seed);
            let x = -2.0 + r * 1.96; // x in [-2, -0.04]
            let d = if rand(&mut seed) < 0.2 { 1.0 } else { 0.0 };
            let noise = (rand(&mut seed) - 0.5) * 0.3;
            let y = 1.0 + 0.5 * x + 3.0 * d + noise;
            x_vals.push(x);
            y_vals.push(y);
            d_vals.push(d);
        }

        // Right of cutoff: ~80% treated
        for _ in 0..50 {
            let r = rand(&mut seed);
            let x = r * 1.96; // x in [0, 1.96]
            let d = if rand(&mut seed) < 0.8 { 1.0 } else { 0.0 };
            let noise = (rand(&mut seed) - 0.5) * 0.3;
            let y = 1.0 + 0.5 * x + 3.0 * d + noise;
            x_vals.push(x);
            y_vals.push(y);
            d_vals.push(d);
        }

        let df = df! {
            "outcome" => y_vals,
            "running" => x_vals,
            "treatment" => d_vals
        }
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_kernel_functions() {
        // Triangular kernel
        assert!((KernelType::Triangular.evaluate(0.0) - 1.0).abs() < 1e-10);
        assert!((KernelType::Triangular.evaluate(0.5) - 0.5).abs() < 1e-10);
        assert!((KernelType::Triangular.evaluate(1.0)).abs() < 1e-10);
        assert_eq!(KernelType::Triangular.evaluate(1.5), 0.0);

        // Epanechnikov kernel
        assert!((KernelType::Epanechnikov.evaluate(0.0) - 0.75).abs() < 1e-10);
        assert_eq!(KernelType::Epanechnikov.evaluate(1.5), 0.0);

        // Uniform kernel
        assert!((KernelType::Uniform.evaluate(0.0) - 0.5).abs() < 1e-10);
        assert!((KernelType::Uniform.evaluate(0.9) - 0.5).abs() < 1e-10);
        assert_eq!(KernelType::Uniform.evaluate(1.5), 0.0);
    }

    #[test]
    fn test_polynomial_matrix() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let cutoff = 1.0;

        let mat = build_polynomial_matrix(x.view(), cutoff, 2).unwrap();

        // Row 0: x=0, (x-c) = -1
        // [1, -1, 1]
        assert!((mat[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((mat[[0, 1]] - (-1.0)).abs() < 1e-10);
        assert!((mat[[0, 2]] - 1.0).abs() < 1e-10);

        // Row 1: x=1, (x-c) = 0
        // [1, 0, 0]
        assert!((mat[[1, 0]] - 1.0).abs() < 1e-10);
        assert!((mat[[1, 1]]).abs() < 1e-10);
        assert!((mat[[1, 2]]).abs() < 1e-10);

        // Row 2: x=2, (x-c) = 1
        // [1, 1, 1]
        assert!((mat[[2, 0]] - 1.0).abs() < 1e-10);
        assert!((mat[[2, 1]] - 1.0).abs() < 1e-10);
        assert!((mat[[2, 2]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sharp_rd_basic() {
        let dataset = create_sharp_rd_dataset();
        let config = RdConfig::default();

        let result = run_rd(&dataset, "outcome", "running", 0.0, config).unwrap();

        // Check structure
        assert_eq!(result.outcome, "outcome");
        assert_eq!(result.running_var, "running");
        assert!((result.cutoff).abs() < 1e-10);
        assert_eq!(result.p, 1);
        assert_eq!(result.q, 2);

        // Check sample sizes
        assert!(result.n_left > 0);
        assert!(result.n_right > 0);

        // Treatment effect should be close to 2.0
        // Allow some tolerance due to noise and estimation error
        assert!(
            (result.tau_conventional - 2.0).abs() < 1.0,
            "Conventional estimate {} too far from 2.0",
            result.tau_conventional
        );
        assert!(
            (result.tau_robust - 2.0).abs() < 1.0,
            "Robust estimate {} too far from 2.0",
            result.tau_robust
        );

        // Standard errors should be positive
        assert!(result.se_conventional > 0.0);
        assert!(result.se_robust > 0.0);

        // Bandwidths should be positive
        assert!(result.h_left > 0.0);
        assert!(result.h_right > 0.0);
    }

    #[test]
    fn test_sharp_rd_different_kernels() {
        let dataset = create_sharp_rd_dataset();

        for kernel in [
            KernelType::Triangular,
            KernelType::Epanechnikov,
            KernelType::Uniform,
        ] {
            let config = RdConfig {
                kernel,
                ..Default::default()
            };

            let result = run_rd(&dataset, "outcome", "running", 0.0, config).unwrap();

            // All kernels should give reasonable estimates
            assert!(
                (result.tau_robust - 2.0).abs() < 1.5,
                "Kernel {:?} gave estimate {} too far from 2.0",
                kernel,
                result.tau_robust
            );
        }
    }

    #[test]
    fn test_bandwidth_selection() {
        let dataset = create_sharp_rd_dataset();

        let bw = rd_bandwidth(
            &dataset,
            "outcome",
            "running",
            0.0,
            1,
            KernelType::Triangular,
            BandwidthMethod::MseRd,
        )
        .unwrap();

        // Bandwidths should be positive and reasonable
        assert!(bw.h_left > 0.0);
        assert!(bw.h_right > 0.0);
        assert!(bw.h_left < 5.0); // Not too large
        assert!(bw.h_right < 5.0);

        // Sample sizes should be tracked
        assert!(bw.n_left > 0);
        assert!(bw.n_right > 0);
    }

    #[test]
    fn test_rd_insufficient_data() {
        // Create very small dataset
        let df = df! {
            "outcome" => [1.0, 2.0, 3.0, 4.0],
            "running" => [-1.0, -0.5, 0.5, 1.0]
        }
        .unwrap();

        let dataset = Dataset::new(df);
        let config = RdConfig::default();

        let result = run_rd(&dataset, "outcome", "running", 0.0, config);

        assert!(result.is_err());
    }

    #[test]
    fn test_rd_missing_column() {
        let dataset = create_sharp_rd_dataset();
        let config = RdConfig::default();

        let result = run_rd(&dataset, "nonexistent", "running", 0.0, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_display_formatting() {
        let dataset = create_sharp_rd_dataset();
        let config = RdConfig::default();

        let result = run_rd(&dataset, "outcome", "running", 0.0, config).unwrap();

        let output = format!("{}", result);
        assert!(output.contains("Sharp RD Estimation Results"));
        assert!(output.contains("Conventional"));
        assert!(output.contains("Robust"));
        assert!(output.contains("Bandwidth"));
    }

    // =========================================================================
    // DGP-recovery tests (no R reference; the rdrobust pins are further below)
    // =========================================================================

    /// A sharp design with a known jump of 0.5: y = 0.5 + 0.3 x + 0.5 1(x >= 0) + noise,
    /// x uniform on (-1, 1), n = 500, from an LCG (not R's random numbers).
    fn create_rd_validation_dataset() -> Dataset {
        // Using a larger sample to ensure stable estimates
        let n = 500;
        let mut x_vals = Vec::with_capacity(n);
        let mut y_vals = Vec::with_capacity(n);

        // Simple LCG for reproducibility
        let mut seed: u64 = 42;
        let a: u64 = 1103515245;
        let c: u64 = 12345;
        let m: u64 = 2_u64.pow(31);

        for _ in 0..n {
            // Generate x ~ Uniform(-1, 1)
            seed = (a.wrapping_mul(seed).wrapping_add(c)) % m;
            let u1 = (seed as f64) / (m as f64);
            let x = -1.0 + 2.0 * u1;

            // Generate noise ~ Normal(0, 0.2) via Box-Muller
            seed = (a.wrapping_mul(seed).wrapping_add(c)) % m;
            let u2 = (seed as f64) / (m as f64);
            seed = (a.wrapping_mul(seed).wrapping_add(c)) % m;
            let u3 = (seed as f64) / (m as f64);
            let noise = 0.2 * (-2.0 * u2.ln()).sqrt() * (2.0 * std::f64::consts::PI * u3).cos();

            // Treatment indicator
            let treatment = if x >= 0.0 { 1.0 } else { 0.0 };

            // Outcome: y = 0.5 + 0.3*x + 0.5*treatment + noise
            let y = 0.5 + 0.3 * x + 0.5 * treatment + noise;

            x_vals.push(x);
            y_vals.push(y);
        }

        let df = df! {
            "outcome" => y_vals,
            "running" => x_vals
        }
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_rd_bandwidth_methods_run() {
        // Test different bandwidth selection methods
        let dataset = create_rd_validation_dataset();

        for bwmethod in [
            BandwidthMethod::MseRd,
            BandwidthMethod::CerRd,
            BandwidthMethod::MseTwo,
        ] {
            let config = RdConfig {
                bwselect: bwmethod,
                ..Default::default()
            };

            let result = run_rd(&dataset, "outcome", "running", 0.0, config);
            assert!(result.is_ok(), "Bandwidth method {:?} failed", bwmethod);

            let r = result.unwrap();
            // All methods should give bandwidths in reasonable range
            assert!(
                r.h_left > 0.05 && r.h_left < 2.0,
                "BW method {:?}: h_left={:.4} out of range",
                bwmethod,
                r.h_left
            );
            assert!(
                r.h_right > 0.05 && r.h_right < 2.0,
                "BW method {:?}: h_right={:.4} out of range",
                bwmethod,
                r.h_right
            );
        }
    }

    #[test]
    fn test_rd_polynomial_orders_recover_dgp() {
        // Test different polynomial orders (p=1, p=2)
        let dataset = create_rd_validation_dataset();

        // p=1 (local linear - default)
        let config_p1 = RdConfig {
            p: 1,
            ..Default::default()
        };
        let result_p1 = run_rd(&dataset, "outcome", "running", 0.0, config_p1).unwrap();

        // p=2 (local quadratic)
        let config_p2 = RdConfig {
            p: 2,
            ..Default::default()
        };
        let result_p2 = run_rd(&dataset, "outcome", "running", 0.0, config_p2).unwrap();

        // Both should give estimates close to the true effect (0.5)
        assert!(
            (result_p1.tau_robust - 0.5).abs() < 0.3,
            "p=1: estimate {:.4} too far from 0.5",
            result_p1.tau_robust
        );
        assert!(
            (result_p2.tau_robust - 0.5).abs() < 0.4, // Quadratic may be less precise
            "p=2: estimate {:.4} too far from 0.5",
            result_p2.tau_robust
        );

        // Higher polynomial order should have higher bias correction order
        assert_eq!(result_p1.q, 2);
        assert_eq!(result_p2.q, 3);
    }

    // =========================================================================
    // Parameter validation and resource guards (T0-WP11)
    // =========================================================================

    fn spec_msg(r: EconResult<RdResult>) -> String {
        match r {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!(
                "expected InvalidSpecification, got {:?}",
                other.map(|r| r.tau_conventional)
            ),
        }
    }

    #[test]
    fn test_rd_rejects_polynomial_order_above_cap() {
        let ds = create_sharp_rd_dataset();
        let m = spec_msg(run_rd(
            &ds,
            "outcome",
            "running",
            0.0,
            RdConfig {
                p: 11,
                ..Default::default()
            },
        ));
        assert!(m.contains("p = 11") && m.contains("10"), "{m}");
        // overflow-sized order: before the fix this panics on `p + 1` / `p + 2`
        let m = spec_msg(run_rd(
            &ds,
            "outcome",
            "running",
            0.0,
            RdConfig {
                p: usize::MAX,
                ..Default::default()
            },
        ));
        assert!(m.contains("Polynomial order"), "{m}");
        let r = run_fuzzy_rd(
            &create_fuzzy_rd_dataset(),
            "outcome",
            "running",
            "treatment",
            0.0,
            RdConfig {
                p: 11,
                ..Default::default()
            },
        );
        assert!(matches!(r, Err(EconError::InvalidSpecification { .. })));
        let r = rd_bandwidth(
            &ds,
            "outcome",
            "running",
            0.0,
            11,
            KernelType::Triangular,
            BandwidthMethod::MseRd,
        );
        assert!(
            matches!(r, Err(EconError::InvalidSpecification { ref message }) if message.contains("p = 11"))
        );
        let m = spec_msg(run_rd(
            &ds,
            "outcome",
            "running",
            0.0,
            RdConfig {
                q: Some(12),
                ..Default::default()
            },
        ));
        assert!(m.contains("q = 12"), "{m}");
    }

    #[test]
    fn test_rd_rejects_invalid_bandwidth_level_and_nnmatch() {
        let ds = create_sharp_rd_dataset();
        for h in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let m = spec_msg(run_rd(
                &ds,
                "outcome",
                "running",
                0.0,
                RdConfig {
                    h: Some(h),
                    ..Default::default()
                },
            ));
            assert!(m.contains("bandwidth h"), "{m}");
        }
        let m = spec_msg(run_rd(
            &ds,
            "outcome",
            "running",
            0.0,
            RdConfig {
                b: Some(0.0),
                ..Default::default()
            },
        ));
        assert!(m.contains("bandwidth b"), "{m}");
        let m = spec_msg(run_rd(
            &ds,
            "outcome",
            "running",
            0.0,
            RdConfig {
                rho: 0.0,
                ..Default::default()
            },
        ));
        assert!(m.contains("rho"), "{m}");
        // level = 95 panics inside statrs before the fix
        for level in [95.0, 0.0, 1.0, f64::NAN] {
            let m = spec_msg(run_rd(
                &ds,
                "outcome",
                "running",
                0.0,
                RdConfig {
                    level,
                    ..Default::default()
                },
            ));
            assert!(m.contains("level"), "{m}");
        }
        let m = spec_msg(run_rd(
            &ds,
            "outcome",
            "running",
            0.0,
            RdConfig {
                nnmatch: 0,
                ..Default::default()
            },
        ));
        assert!(m.contains("nnmatch"), "{m}");
    }

    #[test]
    fn test_rd_requires_enough_observations_inside_bandwidth() {
        let ds = create_sharp_rd_dataset();
        // h = 0.1 leaves 2 left-side points (x = -0.08, -0.04) with positive weight: p = 1 needs 3
        let m = spec_msg(run_rd(
            &ds,
            "outcome",
            "running",
            0.0,
            RdConfig {
                h: Some(0.1),
                ..Default::default()
            },
        ));
        assert!(m.contains("bandwidth") && m.contains("left"), "{m}");
        // h wide enough, but the bias bandwidth b = 0.1 is too narrow for q = 2 (needs 3)
        let m = spec_msg(run_rd(
            &ds,
            "outcome",
            "running",
            0.0,
            RdConfig {
                h: Some(1.0),
                b: Some(0.1),
                ..Default::default()
            },
        ));
        assert!(m.contains("bias bandwidth b"), "{m}");
        // the order cap itself does not reject p = 10 (all 100 points in the window)
        if let Err(EconError::InvalidSpecification { message }) = run_rd(
            &ds,
            "outcome",
            "running",
            0.0,
            RdConfig {
                p: 10,
                h: Some(2.5),
                ..Default::default()
            },
        ) {
            assert!(!message.contains("Polynomial order"), "{message}");
        }
    }

    #[test]
    fn test_rd_design_matrix_respects_allocation_ceiling() {
        let ds = create_sharp_rd_dataset();
        // 100 x 2 design = 1600 bytes > 64
        let r = crate::resource::with_alloc_ceiling(Some(64), || {
            run_rd(
                &ds,
                "outcome",
                "running",
                0.0,
                RdConfig {
                    h: Some(1.0),
                    ..Default::default()
                },
            )
        });
        assert!(
            matches!(r, Err(EconError::ResourceLimit(_))),
            "{:?}",
            r.map(|r| r.tau_conventional)
        );
    }

    /// A strided outcome is copied (and named) only when the selector needs a slice.
    #[test]
    fn test_rd_strided_columns_copied_only_for_selection() {
        let mut s = 7u64;
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * i as f64 / n as f64).collect();
        // Row-major n x 2: column 0 is a strided view.
        let mut m = Array2::<f64>::zeros((n, 2));
        for (i, &xi) in x.iter().enumerate() {
            m[[i, 0]] = 1.0 + 0.5 * xi + if xi >= 0.0 { 0.8 } else { 0.0 } + lcg(&mut s) - 0.5;
            m[[i, 1]] = lcg(&mut s);
        }
        let xv = ArrayView1::from(&x[..]);
        let y = m.column(0);
        assert!(y.as_slice().is_none());

        // Selection: the outcome copy is the first allocation and names its column.
        let r = crate::resource::with_alloc_ceiling(Some(64), || {
            fit_rd(xv, &[y], 0.0, &RdConfig::default()).map(|f| f.h)
        });
        match r {
            Err(EconError::ResourceLimit(m)) => assert!(m.contains("RD outcome"), "{m}"),
            other => panic!("{other:?}"),
        }
        // A given h never copies the columns: the fit is the same as on a contiguous copy.
        let config = RdConfig {
            h: Some(0.5),
            ..Default::default()
        };
        let yc = y.to_owned();
        let a = fit_rd(xv, &[y], 0.0, &config).unwrap();
        let b = fit_rd(xv, &[yc.view()], 0.0, &config).unwrap();
        assert_eq!(a.jumps(0), b.jumps(0));
        let r = crate::resource::with_alloc_ceiling(Some(64), || {
            fit_rd(xv, &[y], 0.0, &config).map(|f| f.h)
        });
        match r {
            Err(EconError::ResourceLimit(m)) => assert!(!m.contains("RD outcome"), "{m}"),
            other => panic!("{other:?}"),
        }
    }

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 53) as f64
    }

    // =========================================================================
    // rdrobust 4.0.0 pins at fixed bandwidths (validation/r/cases/causal_rd.R;
    // T2-WP11: CAUS-1, CAUS-5, CAUS-29)
    // =========================================================================

    const TRI: KernelType = KernelType::Triangular;

    fn fixed(kernel: KernelType, vce: VceType, h: f64, b: f64, p: usize) -> RdConfig {
        RdConfig {
            p,
            h: Some(h),
            b: Some(b),
            kernel,
            vce,
            ..Default::default()
        }
    }

    /// Pins one rdrobust run, `<prefix>_*` keys of the case (`.rd_values` in causal_rd.R).
    fn check_rd(check: &mut Check<'_>, prefix: &str, r: &RdResult, n: usize) {
        let k = |s: &str| format!("{prefix}_{s}");
        check
            .scalar(&k("coef_conv"), r.tau_conventional, Tol::coef(n))
            .scalar(&k("coef_bc"), r.tau_bc, Tol::coef(n))
            .scalar(&k("se_conv"), r.se_conventional, Tol::se(n))
            .scalar(&k("se_robust"), r.se_robust, Tol::se(n))
            .scalar(
                &k("z_conv"),
                r.tau_conventional / r.se_conventional,
                Tol::stat(n),
            )
            .scalar(&k("z_robust"), r.tau_robust / r.se_robust, Tol::stat(n))
            .scalar(&k("p_conv"), r.p_conventional, Tol::p_value(n))
            .scalar(&k("p_robust"), r.p_robust, Tol::p_value(n))
            .vector(
                &k("ci_conv"),
                &[r.ci_conventional.0, r.ci_conventional.1],
                Tol::stat(n),
            )
            .vector(
                &k("ci_robust"),
                &[r.ci_robust.0, r.ci_robust.1],
                Tol::stat(n),
            )
            .vector(
                &k("n_h"),
                &[r.n_eff_left as f64, r.n_eff_right as f64],
                Tol::exact(),
            )
            .vector(
                &k("n_b"),
                &[r.n_b_left as f64, r.n_b_right as f64],
                Tol::exact(),
            )
            .vector(&k("beta_p_left"), &r.coef_left, Tol::coef(n))
            .vector(&k("beta_p_right"), &r.coef_right, Tol::coef(n));
        // rdrobust's second row uses the conventional SE; the clamp of CAUS-5 is gone
        assert_eq!(r.se_bc, r.se_conventional, "{prefix}");
        assert_eq!(r.tau_robust, r.tau_bc, "{prefix}");
        assert!(
            (r.bias - (r.tau_conventional - r.tau_bc)).abs() <= 1e-15 * (1.0 + r.bias.abs()),
            "{prefix}"
        );
    }

    #[test]
    fn test_validate_rd_sharp_fixed_bw() {
        let (case, ds) = load_case("rd_sharp_fixed_bw");
        let n = 1000;
        let mut check = Check::new(&case);
        let kernels = [
            ("tri", KernelType::Triangular),
            ("epa", KernelType::Epanechnikov),
            ("uni", KernelType::Uniform),
        ];
        let vces = [
            ("nn", VceType::Nn),
            ("hc0", VceType::Hc0),
            ("hc1", VceType::Hc1),
            ("hc2", VceType::Hc2),
            ("hc3", VceType::Hc3),
        ];
        for (k, kernel) in kernels {
            for (v, vce) in vces {
                let r = run_rd(&ds, "y", "x", 0.0, fixed(kernel, vce, 0.3, 0.3, 1)).unwrap();
                check_rd(&mut check, &format!("{k}_{v}_hb"), &r, n);
            }
        }
        // h != b both ways (NN neighbours are searched in the larger window), and p = 2
        for (prefix, h, b, p) in [
            ("tri_nn_h03_b05", 0.3, 0.5, 1),
            ("tri_nn_h05_b03", 0.5, 0.3, 1),
            ("tri_nn_p2", 0.5, 0.6, 2),
        ] {
            let r = run_rd(&ds, "y", "x", 0.0, fixed(TRI, VceType::Nn, h, b, p)).unwrap();
            check_rd(&mut check, prefix, &r, n);
        }
        // unequal sides: rdrobust h = c(0.25, 0.35), b = c(0.4, 0.45)
        let asym = RdConfig {
            h: Some(0.25),
            h_right: Some(0.35),
            b: Some(0.4),
            b_right: Some(0.45),
            vce: VceType::Hc1,
            ..Default::default()
        };
        let r = run_rd(&ds, "y", "x", 0.0, asym).unwrap();
        assert_eq!(
            (r.h_left, r.h_right, r.b_left, r.b_right),
            (0.25, 0.35, 0.4, 0.45)
        );
        check_rd(&mut check, "tri_hc1_asym", &r, n);
        let r = run_rd(
            &ds,
            "y",
            "x",
            0.0,
            RdConfig {
                level: 0.90,
                ..fixed(TRI, VceType::Nn, 0.3, 0.3, 1)
            },
        )
        .unwrap();
        check.vector(
            "level90_ci_robust",
            &[r.ci_robust.0, r.ci_robust.1],
            Tol::stat(n),
        );
        check.finish();
    }

    #[test]
    fn test_validate_rd_mass_points_nn() {
        let (case, ds) = load_case("rd_mass_points_nn");
        let mut check = Check::new(&case);
        for m in [3usize, 1, 5] {
            let r = run_rd(
                &ds,
                "ym",
                "xm",
                0.0,
                RdConfig {
                    nnmatch: m,
                    ..fixed(TRI, VceType::Nn, 0.3, 0.4, 1)
                },
            )
            .unwrap();
            check_rd(&mut check, &format!("nn{m}"), &r, 1000);
        }
        check.finish();
    }

    /// One row of the estimate table, in the format `Display` uses.
    fn table_row(label: &str, est: f64, se: f64, p: f64, ci: &[f64]) -> String {
        format!(
            "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.4}  [{:>8.4}, {:>8.4}]",
            label,
            est,
            se,
            est / se,
            p,
            ci[0],
            ci[1]
        )
    }

    #[test]
    fn test_validate_rd_sharp_display_table() {
        let (case, ds) = load_case("rd_sharp_fixed_bw");
        let r = run_rd(&ds, "y", "x", 0.0, fixed(TRI, VceType::Hc1, 0.3, 0.3, 1)).unwrap();
        let text = r.to_string();
        let v = |s: &str| case.scalar(&format!("tri_hc1_hb_{s}"));
        let ci = |s: &str| case.vector(&format!("tri_hc1_hb_{s}"));
        assert!(
            text.contains(
                "Method                   Estimate    Std. Err.          z      P>|z|  [95% CI]"
            ),
            "{text}"
        );
        let conventional = table_row(
            "Conventional",
            v("coef_conv"),
            v("se_conv"),
            v("p_conv"),
            &ci("ci_conv"),
        );
        assert!(text.contains(&conventional), "{conventional}\n{text}");
        let stars = SignificanceLevel::from_p_value(v("p_robust")).stars();
        assert_eq!(stars, "**");
        let robust = table_row(
            "Robust",
            v("coef_bc"),
            v("se_robust"),
            v("p_robust"),
            &ci("ci_robust"),
        ) + stars;
        assert!(text.contains(&robust), "{robust}\n{text}");
        let n_b = ci("n_b");
        let bias_fit = format!(
            "  Bias fit:  {} (left), {} (right)",
            n_b[0] as usize, n_b[1] as usize
        );
        assert!(text.contains(&bias_fit), "{bias_fit}\n{text}");
        assert!(
            text.contains("Note: Robust = bias-corrected estimate"),
            "{text}"
        );

        let r90 = run_rd(
            &ds,
            "y",
            "x",
            0.0,
            RdConfig {
                level: 0.90,
                ..fixed(TRI, VceType::Hc1, 0.3, 0.3, 1)
            },
        )
        .unwrap();
        let text90 = r90.to_string();
        assert!(text90.contains("[90% CI]"), "{text90}");
        assert!(!text90.contains("[95% CI]"), "{text90}");
    }

    #[test]
    fn test_rd_rejects_q_not_above_p_and_lonely_right_bandwidths() {
        let ds = create_sharp_rd_dataset();
        let run = |config: RdConfig| spec_msg(run_rd(&ds, "outcome", "running", 0.0, config));
        let m = run(RdConfig {
            q: Some(1),
            ..Default::default()
        });
        assert!(m.contains("q = 1") && m.contains("greater than"), "{m}");
        let m = run(RdConfig {
            h_right: Some(0.5),
            ..Default::default()
        });
        assert!(m.contains("h_right"), "{m}");
        let m = run(RdConfig {
            h: Some(1.0),
            b_right: Some(0.5),
            ..Default::default()
        });
        assert!(m.contains("b_right"), "{m}");
        let m = run(RdConfig {
            h: Some(1.0),
            h_right: Some(-1.0),
            ..Default::default()
        });
        assert!(m.contains("h_right"), "{m}");
    }

    /// Direct transcription of rdrobust 4.0.0's `rdrobust_res(vce = "nn")` loop
    /// (0-based), for one column on sorted `x`.
    fn rdrobust_res_naive(x: &[f64], y: &[f64], matches: usize) -> Vec<f64> {
        let n = x.len();
        // dups / dupsid: length of the run of equal x and the position in it (rle)
        let (mut dups, mut dupsid) = (vec![0usize; n], vec![0usize; n]);
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j + 1 < n && x[j + 1] == x[i] {
                j += 1;
            }
            for (t, k) in (i..=j).enumerate() {
                dups[k] = j - i + 1;
                dupsid[k] = t + 1;
            }
            i = j + 1;
        }
        let mut res = vec![0.0; n];
        for pos in 0..n {
            let mut rpos = dups[pos] - dupsid[pos];
            let mut lpos = dupsid[pos] - 1;
            while lpos + rpos < matches.min(n - 1) {
                if pos < lpos + 1 {
                    rpos += dups[pos + rpos + 1];
                } else if pos + rpos + 1 >= n {
                    lpos += dups[pos - lpos - 1];
                } else if x[pos] - x[pos - lpos - 1] > x[pos + rpos + 1] - x[pos] {
                    rpos += dups[pos + rpos + 1];
                } else if x[pos] - x[pos - lpos - 1] < x[pos + rpos + 1] - x[pos] {
                    lpos += dups[pos - lpos - 1];
                } else {
                    rpos += dups[pos + rpos + 1];
                    lpos += dups[pos - lpos - 1];
                }
            }
            let ind_j = (pos - lpos)..=(pos + rpos);
            let y_j: f64 = y[ind_j.clone()].iter().sum::<f64>() - y[pos];
            let ji = (ind_j.count() - 1) as f64;
            res[pos] = (ji / (ji + 1.0)).sqrt() * (y[pos] - y_j / ji);
        }
        res
    }

    #[test]
    fn test_nn_residuals_follow_rdrobust_rule() {
        let close = |label: &str, a: &[f64], b: &[f64]| {
            assert_eq!(a.len(), b.len(), "{label}");
            for (i, (u, v)) in a.iter().zip(b).enumerate() {
                assert!(
                    (u - v).abs() <= 1e-12 * (1.0 + v.abs()),
                    "{label} row {i}: {u} vs {v}"
                );
            }
        };
        let one = |x: &[f64], y: &[f64], j: usize| -> Vec<f64> {
            let d = Array1::from_vec(y.to_vec()).insert_axis(ndarray::Axis(1));
            nn_residuals(x, &d, j).unwrap().column(0).to_vec()
        };
        let key = |v: f64| if v == 0.0 { 0.0 } else { v };
        let mut s = 2026u64;
        let n = 600;
        // (a) continuous x, noisy y
        let mut x: Vec<f64> = (0..n).map(|_| -1.0 + 2.0 * lcg(&mut s)).collect();
        x.sort_by(|a, b| key(*a).total_cmp(&key(*b)));
        let y: Vec<f64> = x
            .iter()
            .map(|v| 1.0 + 0.5 * v + lcg(&mut s) - 0.5)
            .collect();
        for j in [1, 3, 7, 40, 599, 1000] {
            close(
                &format!("continuous j={j}"),
                &one(&x, &y, j),
                &rdrobust_res_naive(&x, &y, j),
            );
        }
        // (b) mass points: 21 distinct values, blocks larger than j
        let xm: Vec<f64> = x.iter().map(|v| (v * 10.0).round() / 10.0).collect();
        for j in [1, 3, 50] {
            close(
                &format!("mass points j={j}"),
                &one(&xm, &y, j),
                &rdrobust_res_naive(&xm, &y, j),
            );
        }
        // (c) integer grid with signed zeros: equal left and right distances
        let mut xg: Vec<f64> = (0..n)
            .map(|i| match i % 9 {
                0 => -0.0,
                1 => 0.0,
                k => k as f64 - 4.0,
            })
            .collect();
        xg.sort_by(|a, b| key(*a).total_cmp(&key(*b)));
        close(
            "grid j=3",
            &one(&xg, &y, 3),
            &rdrobust_res_naive(&xg, &y, 3),
        );
        // (d) tiny windows: m = 2 and m = 3 with j = 3
        for m in [2usize, 3] {
            close(
                &format!("m={m}"),
                &one(&x[..m], &y[..m], 3),
                &rdrobust_res_naive(&x[..m], &y[..m], 3),
            );
        }
        // two columns at once (outcome and treatment) equal the one-column runs
        let y2: Vec<f64> = y.iter().map(|v| 2.0 * v + 1.0).collect();
        let mut d = Array2::zeros((n, 2));
        for i in 0..n {
            d[[i, 0]] = y[i];
            d[[i, 1]] = y2[i];
        }
        let both = nn_residuals(&xm, &d, 3).unwrap();
        close("column 0", &both.column(0).to_vec(), &one(&xm, &y, 3));
        close("column 1", &both.column(1).to_vec(), &one(&xm, &y2, 3));
    }

    // =========================================================================
    // Fuzzy RD: rdrobust(fuzzy = ) at fixed bandwidths (T2-WP11: CAUS-30)
    // =========================================================================

    /// Pins one fuzzy rdrobust run, `<prefix>_*` keys of `rd_fuzzy_fixed_bw`.
    fn check_fuzzy(
        check: &mut Check<'_>,
        case: &p2a_validation::Case,
        prefix: &str,
        f: &FuzzyRdResult,
    ) {
        let n = 1000;
        let k = |s: &str| format!("{prefix}_{s}");
        check
            .scalar(&k("coef_conv"), f.tau_fuzzy, Tol::coef(n))
            .scalar(&k("coef_bc"), f.tau_fuzzy_bc, Tol::coef(n))
            .scalar(&k("se_conv"), f.se_fuzzy, Tol::se(n))
            .scalar(&k("se_robust"), f.se_fuzzy_robust, Tol::se(n))
            .scalar(&k("z_conv"), f.tau_fuzzy / f.se_fuzzy, Tol::stat(n))
            .scalar(
                &k("z_robust"),
                f.tau_fuzzy_bc / f.se_fuzzy_robust,
                Tol::stat(n),
            )
            .scalar(&k("p_conv"), f.p_fuzzy, Tol::p_value(n))
            .scalar(&k("p_robust"), f.p_fuzzy_robust, Tol::p_value(n))
            .vector(&k("ci_conv"), &[f.ci_fuzzy.0, f.ci_fuzzy.1], Tol::stat(n))
            .vector(
                &k("ci_robust"),
                &[f.ci_fuzzy_robust.0, f.ci_fuzzy_robust.1],
                Tol::stat(n),
            )
            .vector(
                &k("n_h"),
                &[
                    f.outcome_rd.n_eff_left as f64,
                    f.outcome_rd.n_eff_right as f64,
                ],
                Tol::exact(),
            )
            .vector(
                &k("n_b"),
                &[f.outcome_rd.n_b_left as f64, f.outcome_rd.n_b_right as f64],
                Tol::exact(),
            )
            .vector(&k("beta_p_left"), &f.outcome_rd.coef_left, Tol::coef(n))
            .vector(&k("beta_p_right"), &f.outcome_rd.coef_right, Tol::coef(n))
            .scalar(&k("first_stage_conv"), f.first_stage_tau, Tol::coef(n))
            .scalar(
                &k("first_stage_conv"),
                f.first_stage.tau_conventional,
                Tol::coef(n),
            )
            .scalar(&k("first_stage_bc"), f.first_stage.tau_bc, Tol::coef(n))
            .scalar(
                &k("first_stage_se_conv"),
                f.first_stage.se_conventional,
                Tol::se(n),
            )
            .scalar(
                &k("first_stage_se_robust"),
                f.first_stage.se_robust,
                Tol::se(n),
            );
        let f_case =
            (case.scalar(&k("first_stage_conv")) / case.scalar(&k("first_stage_se_conv"))).powi(2);
        assert!(
            Tol::stat(n).allows(f.first_stage_f, f_case),
            "{prefix}: first-stage F {} vs {f_case}",
            f.first_stage_f
        );
        // one bandwidth for both stages
        let bw = |r: &RdResult| (r.h_left, r.h_right, r.b_left, r.b_right);
        assert_eq!(bw(&f.outcome_rd), bw(&f.first_stage), "{prefix}");
    }

    #[test]
    fn test_validate_rd_fuzzy_fixed_bw() {
        let (case, ds) = load_case("rd_fuzzy_fixed_bw");
        let n = 1000;
        let mut check = Check::new(&case);
        for (prefix, cfg) in [
            ("fuzzy_nn", fixed(TRI, VceType::Nn, 0.4, 0.4, 1)),
            ("fuzzy_hc1", fixed(TRI, VceType::Hc1, 0.4, 0.4, 1)),
            ("fuzzy_nn_h03_b05", fixed(TRI, VceType::Nn, 0.3, 0.5, 1)),
        ] {
            let f = run_fuzzy_rd(&ds, "yf", "x", "d", 0.0, cfg).unwrap();
            check_fuzzy(&mut check, &case, prefix, &f);
            if prefix == "fuzzy_nn" {
                check.scalar(
                    "reduced_form_coef_conv",
                    f.outcome_rd.tau_conventional,
                    Tol::coef(n),
                );
            }
        }
        let f = run_fuzzy_rd(
            &ds,
            "yf",
            "x",
            "d",
            0.0,
            RdConfig {
                level: 0.90,
                ..fixed(TRI, VceType::Nn, 0.4, 0.4, 1)
            },
        )
        .unwrap();
        assert_eq!(f.level, 0.90);
        check
            .vector(
                "fuzzy_level90_ci_conv",
                &[f.ci_fuzzy.0, f.ci_fuzzy.1],
                Tol::stat(n),
            )
            .vector(
                "fuzzy_level90_ci_robust",
                &[f.ci_fuzzy_robust.0, f.ci_fuzzy_robust.1],
                Tol::stat(n),
            );
        check.finish();
    }

    /// The first stage of a fuzzy fit is the sharp RD of the treatment, here an Int64 column.
    #[test]
    fn test_validate_rd_sharp_int64_outcome() {
        let (case, ds) = load_case("rd_fuzzy_fixed_bw");
        assert_eq!(ds.df().column("d").unwrap().dtype(), &DataType::Int64);
        let r = run_rd(&ds, "d", "x", 0.0, fixed(TRI, VceType::Nn, 0.4, 0.4, 1)).unwrap();
        let n = 1000;
        Check::new(&case)
            .scalar(
                "fuzzy_nn_first_stage_conv",
                r.tau_conventional,
                Tol::coef(n),
            )
            .scalar("fuzzy_nn_first_stage_bc", r.tau_bc, Tol::coef(n))
            .scalar(
                "fuzzy_nn_first_stage_se_conv",
                r.se_conventional,
                Tol::se(n),
            )
            .scalar("fuzzy_nn_first_stage_se_robust", r.se_robust, Tol::se(n))
            .finish();
    }

    /// Fuzzy design on which p2a's selector picks different bandwidths for the outcome
    /// (curved: 0.5 + 10 x^2 + d + N(0, 0.3^2)) and the treatment (take-up 0.2 left of
    /// the cutoff, 0.8 right); n = 2000, x uniform on [-1, 1].
    fn create_fuzzy_rd_curved_dataset() -> Dataset {
        let mut s: u64 = 20261030;
        let n = 2000;
        let (mut xs, mut ys, mut ds) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            let x = 2.0 * lcg(&mut s) - 1.0;
            let take_up = if x >= 0.0 { 0.8 } else { 0.2 };
            let d = if lcg(&mut s) < take_up { 1.0 } else { 0.0 };
            // Box-Muller; 1 - u lies in (0, 1], so the log is finite
            let (u1, u2) = (1.0 - lcg(&mut s), lcg(&mut s));
            let noise = 0.3 * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            xs.push(x);
            ys.push(0.5 + 10.0 * x * x + d + noise);
            ds.push(d);
        }
        Dataset::new(df! { "outcome" => ys, "running" => xs, "treatment" => ds }.unwrap())
    }

    #[test]
    fn test_fuzzy_rd_one_bandwidth_for_both_stages() {
        let ds = create_fuzzy_rd_curved_dataset();
        // Precondition: run on its own, the selector picks another h for the treatment, so
        // two separate sharp fits (the old code) would give the stages different bandwidths.
        let sharp = run_rd(&ds, "outcome", "running", 0.0, RdConfig::default()).unwrap();
        let sharp_t = run_rd(&ds, "treatment", "running", 0.0, RdConfig::default()).unwrap();
        assert_ne!(sharp_t.h_left, sharp.h_left);
        assert_ne!(sharp_t.h_right, sharp.h_right);
        let fuzzy_bw = rd_bandwidth_with(
            &ds,
            "outcome",
            "running",
            Some("treatment"),
            0.0,
            &RdConfig::default(),
        )
        .unwrap();

        let f = run_fuzzy_rd(
            &ds,
            "outcome",
            "running",
            "treatment",
            0.0,
            RdConfig::default(),
        )
        .unwrap();
        let bw = |r: &RdResult| (r.h_left, r.h_right, r.b_left, r.b_right);
        assert_eq!(bw(&f.outcome_rd), bw(&f.first_stage));
        // the selection is rdbwselect(fuzzy = ): on the ratio, with the treatment, not on the
        // outcome alone (rdrobust: 0.64194 fuzzy, 0.53810 sharp on this dataset)
        assert_eq!(
            bw(&f.outcome_rd),
            (
                fuzzy_bw.h_left,
                fuzzy_bw.h_right,
                fuzzy_bw.b_left,
                fuzzy_bw.b_right
            )
        );
        assert_ne!(f.outcome_rd.h_left, sharp.h_left);
    }

    /// `validation/datasets/generated/rd_sim.csv` as data only (no R value is compared).
    fn rd_sim() -> Dataset {
        let path = p2a_validation::repo_path("validation/datasets/generated/rd_sim.csv");
        Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap())
    }

    /// Pins one rdrobust run at selected bandwidths: the `.rd_sel` keys `<prefix>_*` of
    /// causal_rdbw.R.
    fn check_selected(check: &mut Check<'_>, prefix: &str, r: &RdResult, n: usize) {
        let k = |s: &str| format!("{prefix}_{s}");
        check
            .vector(&k("h"), &[r.h_left, r.h_right], Tol::coef(n))
            .vector(&k("b"), &[r.b_left, r.b_right], Tol::coef(n))
            .scalar(&k("coef_conv"), r.tau_conventional, Tol::coef(n))
            .scalar(&k("coef_bc"), r.tau_bc, Tol::coef(n))
            .scalar(&k("se_conv"), r.se_conventional, Tol::se(n))
            .scalar(&k("se_robust"), r.se_robust, Tol::se(n))
            .scalar(&k("p_robust"), r.p_robust, Tol::p_value(n))
            .vector(
                &k("ci_robust"),
                &[r.ci_robust.0, r.ci_robust.1],
                Tol::stat(n),
            )
            .vector(
                &k("n_h"),
                &[r.n_eff_left as f64, r.n_eff_right as f64],
                Tol::exact(),
            )
            .vector(
                &k("n_b"),
                &[r.n_b_left as f64, r.n_b_right as f64],
                Tol::exact(),
            );
    }

    fn sel(f: impl FnOnce(&mut RdConfig)) -> RdConfig {
        let mut c = RdConfig::default();
        f(&mut c);
        c
    }

    #[test]
    fn test_validate_rd_auto_bandwidth() {
        use BandwidthOrigin::*;
        let (case, ds) = load_case("rd_bwselect_sharp");
        let n = 1000;
        let run = |y: &str, x: &str, c: RdConfig| run_rd(&ds, y, x, 0.0, c).unwrap();
        let mut check = Check::new(&case);

        let auto = run("y", "x", RdConfig::default());
        check_selected(&mut check, "auto", &auto, n);
        assert_eq!((auto.h_origin, auto.b_origin), (Selected, Selected));
        assert!(!auto.bandwidth_given);
        let r = run("y", "x", sel(|c| c.bwselect = BandwidthMethod::MseTwo));
        check_selected(&mut check, "msetwo_auto", &r, n);
        let rho1 = run("y", "x", sel(|c| c.rho_with_selected_h = true));
        check_selected(&mut check, "rho1_auto", &rho1, n);
        assert_eq!((rho1.h_origin, rho1.b_origin), (Selected, RhoTimesH));
        let r = run("y", "x", sel(|c| c.scaleregul = 0.0));
        check_selected(&mut check, "noreg_auto", &r, n);
        let r = run("y", "x", sel(|c| c.vce = VceType::Hc1));
        check_selected(&mut check, "hc1_auto", &r, n);
        let b05 = run("y", "x", sel(|c| c.b = Some(0.5)));
        check_selected(&mut check, "hsel_b05", &b05, n);
        assert_eq!((b05.h_origin, b05.b_origin), (Selected, UserSupplied));
        let r = run("ym", "xm", RdConfig::default());
        check_selected(&mut check, "mass_auto", &r, n);

        let cer = run("y", "x", sel(|c| c.bwselect = BandwidthMethod::CerRd));
        check
            .vector("cerrd_h", &[cer.h_left, cer.h_right], Tol::coef(n))
            .vector("cerrd_b", &[cer.b_left, cer.b_right], Tol::coef(n))
            .scalar("cerrd_coef_conv", cer.tau_conventional, Tol::coef(n))
            .scalar("cerrd_coef_bc", cer.tau_bc, Tol::coef(n))
            .scalar("cerrd_se_conv", cer.se_conventional, Tol::se(n))
            .scalar("cerrd_se_robust", cer.se_robust, Tol::se(n))
            .vector(
                "cerrd_n_h",
                &[cer.n_eff_left as f64, cer.n_eff_right as f64],
                Tol::exact(),
            );

        // origins of given bandwidths
        let r = run("y", "x", sel(|c| c.h = Some(0.3)));
        assert_eq!((r.h_origin, r.b_origin), (UserSupplied, RhoTimesH));
        assert!(r.bandwidth_given);
        let r = run(
            "y",
            "x",
            sel(|c| {
                c.h = Some(0.3);
                c.b = Some(0.3)
            }),
        );
        assert_eq!((r.h_origin, r.b_origin), (UserSupplied, UserSupplied));
        // a rho other than 1 with a selected h is honoured: b = h / rho (rdrobust's rho)
        let r = run("y", "x", sel(|c| c.rho = 2.0));
        assert_eq!((r.h_origin, r.b_origin), (Selected, RhoTimesH));
        assert_eq!(r.b_left, r.h_left / 2.0);
        assert_eq!(r.b_right, r.h_right / 2.0);
        assert_eq!((r.h_left, r.h_right), (auto.h_left, auto.h_right));

        // selection only
        let bw = rd_bandwidth(
            &ds,
            "y",
            "x",
            0.0,
            1,
            KernelType::Triangular,
            BandwidthMethod::MseRd,
        )
        .unwrap();
        check
            .vector("auto_h", &[bw.h_left, bw.h_right], Tol::coef(n))
            .vector("auto_b", &[bw.b_left, bw.b_right], Tol::coef(n))
            .vector(
                "auto_n_h",
                &[bw.n_h_left as f64, bw.n_h_right as f64],
                Tol::exact(),
            )
            .vector(
                "tri_p1_nn_counts_n",
                &[bw.n_left as f64, bw.n_right as f64],
                Tol::exact(),
            )
            .vector(
                "tri_p1_nn_counts_m",
                &[bw.m_left as f64, bw.m_right as f64],
                Tol::exact(),
            );
        assert!(!bw.mass_points);
        // CAUS-28: rd_bandwidth returned b = h
        assert_ne!(bw.b_left, bw.h_left);
        assert_eq!(bw.b_origin, Selected);
        let bw = rd_bandwidth_with(
            &ds,
            "y",
            "x",
            None,
            0.0,
            &sel(|c| c.bwselect = BandwidthMethod::CerRd),
        )
        .unwrap();
        check
            .vector("cerrd_h", &[bw.h_left, bw.h_right], Tol::coef(n))
            .vector("cerrd_b", &[bw.b_left, bw.b_right], Tol::coef(n));
        let bw = rd_bandwidth(
            &ds,
            "ym",
            "xm",
            0.0,
            1,
            KernelType::Triangular,
            BandwidthMethod::MseRd,
        )
        .unwrap();
        assert!(bw.mass_points);
        assert_eq!(bw.bwcheck, Some(10));
        check.vector(
            "mass_counts_m",
            &[bw.m_left as f64, bw.m_right as f64],
            Tol::exact(),
        );
        assert!(
            bw.warnings
                .iter()
                .any(|w| w.starts_with("Mass points detected")),
            "{:?}",
            bw.warnings
        );
        check.finish();
    }

    #[test]
    fn test_validate_rd_fuzzy_auto_bandwidth() {
        let (case, ds) = load_case("rd_bwselect_fuzzy");
        let n = 1000;
        let mut check = Check::new(&case);
        let f = run_fuzzy_rd(&ds, "yf", "x", "d", 0.0, RdConfig::default()).unwrap();
        let o = &f.outcome_rd;
        check
            .vector("fuzzy_auto_h", &[o.h_left, o.h_right], Tol::coef(n))
            .vector("fuzzy_auto_b", &[o.b_left, o.b_right], Tol::coef(n))
            .scalar("fuzzy_auto_coef_conv", f.tau_fuzzy, Tol::coef(n))
            .scalar("fuzzy_auto_coef_bc", f.tau_fuzzy_bc, Tol::coef(n))
            .scalar("fuzzy_auto_se_conv", f.se_fuzzy, Tol::se(n))
            .scalar("fuzzy_auto_se_robust", f.se_fuzzy_robust, Tol::se(n))
            .scalar("fuzzy_auto_p_robust", f.p_fuzzy_robust, Tol::p_value(n))
            .vector(
                "fuzzy_auto_ci_robust",
                &[f.ci_fuzzy_robust.0, f.ci_fuzzy_robust.1],
                Tol::stat(n),
            )
            .vector(
                "fuzzy_auto_n_h",
                &[o.n_eff_left as f64, o.n_eff_right as f64],
                Tol::exact(),
            )
            .scalar(
                "fuzzy_auto_first_stage_conv",
                f.first_stage_tau,
                Tol::coef(n),
            );
        assert_eq!(f.first_stage.h_left, f.outcome_rd.h_left);
        assert_eq!(f.first_stage.b_origin, BandwidthOrigin::Selected);

        let c = run_fuzzy_rd(
            &ds,
            "yf",
            "x",
            "d",
            0.0,
            sel(|c| c.bwselect = BandwidthMethod::CerRd),
        )
        .unwrap();
        let o = &c.outcome_rd;
        check
            .vector("fuzzy_cerrd_h", &[o.h_left, o.h_right], Tol::coef(n))
            .vector("fuzzy_cerrd_b", &[o.b_left, o.b_right], Tol::coef(n))
            .scalar("fuzzy_cerrd_coef_bc", c.tau_fuzzy_bc, Tol::coef(n))
            .scalar("fuzzy_cerrd_se_robust", c.se_fuzzy_robust, Tol::se(n));

        let bw = rd_bandwidth_with(&ds, "yf", "x", Some("d"), 0.0, &RdConfig::default()).unwrap();
        check.vector(
            "fuzzy_tri_p1_nn_mserd",
            &[bw.h_left, bw.h_right, bw.b_left, bw.b_right],
            Tol::coef(n),
        );
        assert_eq!(bw.treatment.as_deref(), Some("d"));

        // one-sided compliance: no take-up left of the cutoff
        let x = ds.df().column("x").unwrap().f64().unwrap().clone();
        let d = ds
            .df()
            .column("d")
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap();
        let d = d.f64().unwrap().clone();
        let d1: Vec<f64> = x
            .into_iter()
            .zip(d.into_iter())
            .map(|(x, d)| match (x, d) {
                (Some(x), Some(d)) => d * if x >= 0.0 { 1.0 } else { 0.0 },
                _ => f64::NAN,
            })
            .collect();
        let mut df = ds.df().clone();
        df.with_column(Series::new("d1".into(), d1)).unwrap();
        let ds1 = Dataset::new(df);
        let f1 = run_fuzzy_rd(&ds1, "yf", "x", "d1", 0.0, RdConfig::default()).unwrap();
        let o = &f1.outcome_rd;
        check
            .vector(
                "fuzzy_onesided_mserd",
                &[o.h_left, o.h_right, o.b_left, o.b_right],
                Tol::coef(n),
            )
            .scalar("fuzzy_onesided_coef_conv", f1.tau_fuzzy, Tol::coef(n))
            .scalar("fuzzy_onesided_coef_bc", f1.tau_fuzzy_bc, Tol::coef(n))
            .scalar("fuzzy_onesided_se_robust", f1.se_fuzzy_robust, Tol::se(n));
        assert!(
            f1.warnings
                .iter()
                .any(|w| w.contains("one-sided compliance")),
            "{:?}",
            f1.warnings
        );
        check.finish();
    }

    #[test]
    fn test_rd_bandwidth_label_names_the_source() {
        let ds = rd_sim();
        let sel_by = |m: &str| format!("selected by {m}, as rdrobust::rdbwselect");
        let mserd = "mserd (MSE-optimal, one common bandwidth)";
        let cases: Vec<(RdConfig, String)> = vec![
            (RdConfig::default(), format!("h and b {}", sel_by(mserd))),
            (
                sel(|c| c.bwselect = BandwidthMethod::CerTwo),
                format!(
                    "h and b {}",
                    sel_by("certwo (CER-optimal, separate left and right)")
                ),
            ),
            (
                sel(|c| c.b = Some(0.5)),
                format!("h {}; b user-supplied", sel_by(mserd)),
            ),
            (
                sel(|c| c.rho_with_selected_h = true),
                format!("h {}; b = 1.0000 h (rho)", sel_by(mserd)),
            ),
            (
                sel(|c| c.h = Some(0.3)),
                "h user-supplied; b = 1.0000 h (rho)".to_string(),
            ),
            (
                sel(|c| {
                    c.h = Some(0.3);
                    c.b = Some(0.5)
                }),
                "h and b user-supplied".to_string(),
            ),
        ];
        for (config, label) in cases {
            let r = run_rd(&ds, "y", "x", 0.0, config).unwrap();
            assert_eq!(r.bandwidth_label(), label);
            let text = r.to_string();
            assert!(
                text.contains(&format!("Bandwidth source: {label}")),
                "{text}"
            );
            assert!(!text.contains("Bandwidth Method"), "{text}");
        }
        let f = run_fuzzy_rd(&ds, "yf", "x", "d", 0.0, RdConfig::default()).unwrap();
        assert!(
            f.to_string()
                .contains("Bandwidth source: h and b selected by mserd"),
            "{f}"
        );
    }

    #[test]
    fn test_rd_cer_differs_from_mse() {
        use BandwidthMethod::*;
        let ds = rd_sim();
        let run = |p: usize, m: BandwidthMethod| {
            run_rd(
                &ds,
                "y",
                "x",
                0.0,
                sel(|c| {
                    c.p = p;
                    c.bwselect = m
                }),
            )
            .unwrap()
        };
        let pairs = [
            (MseRd, CerRd),
            (MseTwo, CerTwo),
            (MseSum, CerSum),
            (MseComb1, CerComb1),
            (MseComb2, CerComb2),
        ];
        for (p, list) in [(1usize, &pairs[..]), (2, &pairs[..1]), (0, &pairs[..1])] {
            let f = 1000f64.powf(-(p as f64) / ((3 + p) * (3 + 2 * p)) as f64);
            for &(m, c) in list {
                let (mse, cer) = (run(p, m), run(p, c));
                for (cs, ms) in [(cer.h_left, mse.h_left), (cer.h_right, mse.h_right)] {
                    assert!((cs - ms * f).abs() <= 1e-12 * ms, "{m:?} p={p}: {cs} {ms}");
                }
                assert_eq!((cer.b_left, cer.b_right), (mse.b_left, mse.b_right));
                if p > 0 {
                    assert!(cer.h_left < 0.75 * mse.h_left, "{m:?} p={p}");
                    assert_ne!(cer.tau_bc, mse.tau_bc);
                } else {
                    assert_eq!(cer.tau_bc, mse.tau_bc);
                }
            }
        }
    }

    #[test]
    fn test_rd_rejects_invalid_scaleregul() {
        let ds = create_sharp_rd_dataset();
        let fz = create_fuzzy_rd_dataset();
        for v in [-1.0, f64::NAN] {
            let c = sel(|c| c.scaleregul = v);
            let m = spec_msg(run_rd(&ds, "outcome", "running", 0.0, c.clone()));
            assert!(m.contains("scaleregul"), "{m}");
            let m = match run_fuzzy_rd(&fz, "outcome", "running", "treatment", 0.0, c.clone()) {
                Err(EconError::InvalidSpecification { message }) => message,
                other => panic!("expected InvalidSpecification, got {:?}", other.is_ok()),
            };
            assert!(m.contains("scaleregul"), "{m}");
            let m = spec_message(rd_bandwidth_with(&ds, "outcome", "running", None, 0.0, &c));
            assert!(m.contains("scaleregul"), "{m}");
        }
        let m = spec_message(rd_bandwidth_with(
            &ds,
            "outcome",
            "running",
            None,
            0.0,
            &sel(|c| c.h = Some(0.5)),
        ));
        assert!(m.contains("rd_bandwidth_with selects h and b"), "{m}");
    }

    /// `create_sharp_rd_dataset` plus `t`: 0 left of the cutoff, 1 right of it.
    fn sharp_with_constant_sides() -> Dataset {
        let ds = create_sharp_rd_dataset();
        let x = ds.df().column("running").unwrap().f64().unwrap().clone();
        let t: Vec<f64> = x
            .into_no_null_iter()
            .map(|v| if v >= 0.0 { 1.0 } else { 0.0 })
            .collect();
        let mut df = ds.df().clone();
        df.with_column(Series::new("t".into(), t)).unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_rd_zero_se_gives_p_zero_for_a_nonzero_jump() {
        // Constant on each side: the nearest-neighbour residuals are 0, so the SEs are 0
        // and, as R's 2 * pnorm(-Inf), every p-value of the (nonzero) jump is 0.
        let ds = sharp_with_constant_sides();
        let cfg = || RdConfig {
            h: Some(1.0),
            ..Default::default()
        };
        let r = run_rd(&ds, "t", "running", 0.0, cfg()).unwrap();
        assert!(
            (r.tau_conventional - 1.0).abs() < 1e-8,
            "{}",
            r.tau_conventional
        );
        assert_eq!((r.se_conventional, r.se_robust), (0.0, 0.0));
        assert_eq!((r.p_conventional, r.p_bc, r.p_robust), (0.0, 0.0, 0.0));
        assert_eq!(r.significance, SignificanceLevel::TenthPercent);
        let text = r.to_string();
        let row = text
            .lines()
            .find(|l| l.starts_with("Conventional"))
            .unwrap();
        assert!(row.contains("inf"), "{row}");

        // the first stage of a fuzzy fit on that column
        let f = run_fuzzy_rd(&ds, "outcome", "running", "t", 0.0, cfg()).unwrap();
        assert_eq!(f.first_stage.se_conventional, 0.0);
        assert_eq!(f.first_stage.p_conventional, 0.0);
        assert!(f.first_stage_f.is_infinite());
        assert!(!f.warnings.iter().any(|w| w.starts_with("Weak first stage")));

        // a zero estimate with a zero SE is not significant
        assert_eq!(two_sided_p(&Normal::standard(), 0.0, 0.0), 1.0);
    }

    #[test]
    fn test_fuzzy_rd_degenerate_first_stage_names_the_jump() {
        let mut df = create_sharp_rd_dataset().df().clone();
        df.with_column(Series::new("t".into(), vec![0.5; 100]))
            .unwrap();
        let ds = Dataset::new(df);
        let cfg = RdConfig {
            h: Some(1.0),
            ..Default::default()
        };
        match run_fuzzy_rd(&ds, "outcome", "running", "t", 0.0, cfg) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(
                    message.starts_with("First stage is degenerate")
                        && message.contains("at least 1e-10")
                        && message.contains("'t'"),
                    "{message}"
                );
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_rd_display_names_a_given_bandwidth() {
        let ds = create_sharp_rd_dataset();
        let given = run_rd(
            &ds,
            "outcome",
            "running",
            0.0,
            RdConfig {
                h: Some(1.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(given.bandwidth_given);
        assert!(
            given
                .to_string()
                .contains("Bandwidth source: h user-supplied; b = 1.0000 h (rho)"),
            "{given}"
        );
        let selected = run_rd(&ds, "outcome", "running", 0.0, RdConfig::default()).unwrap();
        assert!(!selected.bandwidth_given);
        assert!(
            selected
                .to_string()
                .contains("Bandwidth source: h and b selected by mserd"),
            "{selected}"
        );
        // a result serialised before the field existed reads as selected
        let mut json = serde_json::to_value(&given).unwrap();
        for key in ["bandwidth_given", "h_origin", "b_origin"] {
            json.as_object_mut().unwrap().remove(key);
        }
        let old: RdResult = serde_json::from_value(json).unwrap();
        assert!(!old.bandwidth_given);
        assert_eq!(
            (old.h_origin, old.b_origin),
            (BandwidthOrigin::Selected, BandwidthOrigin::Selected)
        );
    }

    #[test]
    fn test_validate_rd_fuzzy_weak_first_stage_warning() {
        let (case, ds) = load_case("rd_fuzzy_fixed_bw");
        let n = 1000;
        let f = run_fuzzy_rd(
            &ds,
            "yf",
            "x",
            "d",
            0.0,
            fixed(TRI, VceType::Nn, 0.2, 0.2, 1),
        )
        .unwrap();
        Check::new(&case)
            .scalar(
                "fuzzy_nn_h02_first_stage_conv",
                f.first_stage.tau_conventional,
                Tol::coef(n),
            )
            .scalar(
                "fuzzy_nn_h02_first_stage_se_conv",
                f.first_stage.se_conventional,
                Tol::se(n),
            )
            .finish();
        let f_case = (case.scalar("fuzzy_nn_h02_first_stage_conv")
            / case.scalar("fuzzy_nn_h02_first_stage_se_conv"))
        .powi(2);
        assert!(f_case < 10.0);
        assert!(
            Tol::stat(n).allows(f.first_stage_f, f_case),
            "F {} vs {f_case}",
            f.first_stage_f
        );
        let expected = format!("Weak first stage: F = {f_case:.2}");
        assert!(
            f.warnings.iter().any(|w| w.starts_with(&expected)),
            "{expected} in {:?}",
            f.warnings
        );
        assert!(f.to_string().contains(&expected), "{f}");
        let strong = run_fuzzy_rd(
            &ds,
            "yf",
            "x",
            "d",
            0.0,
            fixed(TRI, VceType::Nn, 0.4, 0.4, 1),
        )
        .unwrap();
        assert!(
            !strong
                .warnings
                .iter()
                .any(|w| w.starts_with("Weak first stage")),
            "{:?}",
            strong.warnings
        );
    }

    // =========================================================================
    // Missing values and column types (T2-WP11: CAUS-76)
    // =========================================================================

    /// `rd_sim.csv` as data only (no R value is compared).
    fn rd_data() -> Dataset {
        load_case("rd_sharp_fixed_bw").1
    }

    /// `col` cast to Float64 with the listed rows set to `fill` (null when `None`).
    fn with_cells(ds: &Dataset, col: &str, rows: &[usize], fill: Option<f64>) -> Dataset {
        let mut df = ds.df().clone();
        let cast = df.column(col).unwrap().cast(&DataType::Float64).unwrap();
        let mut vals: Vec<Option<f64>> = cast.f64().unwrap().into_iter().collect();
        for &r in rows {
            vals[r] = fill;
        }
        df.with_column(Series::new(col.into(), vals)).unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_validate_rd_sharp_fixed_bw_missing_rows() {
        let (case, ds) = load_case("rd_sharp_fixed_bw");
        let rows = |k: &str| -> Vec<usize> { case.vector(k).iter().map(|&v| v as usize).collect() };
        let (na_y, na_x) = (rows("na_rows_y"), rows("na_rows_x"));
        let mut check = Check::new(&case);
        for fill in [None, Some(f64::NAN)] {
            let dn = with_cells(&with_cells(&ds, "y", &na_y, fill), "x", &na_x, fill);
            let r = run_rd(&dn, "y", "x", 0.0, fixed(TRI, VceType::Nn, 0.3, 0.3, 1)).unwrap();
            check_rd(&mut check, "na_tri_nn_hb", &r, 996);
            assert_eq!(
                r.missing,
                MissingSummary {
                    n_total: 1000,
                    n_used: 996,
                    n_dropped: 4,
                    by_column: vec![("y".into(), 2), ("x".into(), 2)],
                },
                "{fill:?}"
            );
            let text = r.to_string();
            let line =
                "Missing values: 4 of 1000 rows dropped because of missing values (y: 2, x: 2)";
            assert_eq!(text.trim_end().lines().last(), Some(line), "{text}");
            assert_eq!(text.matches("4 of 1000 rows dropped").count(), 1, "{text}");
        }
        check.finish();
    }

    #[test]
    fn test_na_rd_refuses_infinite_text_and_absent_columns() {
        let ds = rd_data();
        let mut df = ds.df().clone();
        let text = df.column("y").unwrap().cast(&DataType::String).unwrap();
        df.with_column(
            text.as_materialized_series()
                .clone()
                .with_name("y_text".into()),
        )
        .unwrap();
        let ds_text = Dataset::new(df);
        let ds_inf = with_cells(&ds, "x", &[0], Some(f64::INFINITY));
        let cfg = || fixed(TRI, VceType::Nn, 0.3, 0.3, 1);

        let inf = |r: EconResult<()>| {
            assert!(
                matches!(r, Err(EconError::NonFiniteValues { ref column, count: 1 }) if column == "x"),
                "{r:?}"
            )
        };
        let text_col = |r: EconResult<()>| {
            assert!(
                matches!(r, Err(EconError::NonNumericColumn { ref column }) if column == "y_text"),
                "{r:?}"
            )
        };
        let absent = |r: EconResult<()>| {
            assert!(
                matches!(r, Err(EconError::ColumnNotFound { ref column, .. }) if column == "nope"),
                "{r:?}"
            )
        };
        // run_rd
        inf(run_rd(&ds_inf, "y", "x", 0.0, cfg()).map(|_| ()));
        text_col(run_rd(&ds_text, "y_text", "x", 0.0, cfg()).map(|_| ()));
        absent(run_rd(&ds, "nope", "x", 0.0, cfg()).map(|_| ()));
        // rd_bandwidth
        let bw = |d: &Dataset, y: &str| {
            rd_bandwidth(d, y, "x", 0.0, 1, TRI, BandwidthMethod::MseRd).map(|_| ())
        };
        inf(bw(&ds_inf, "y"));
        text_col(bw(&ds_text, "y_text"));
        absent(bw(&ds, "nope"));
        // run_fuzzy_rd (treatment column)
        let fz = |d: &Dataset, t: &str| run_fuzzy_rd(d, "yf", "x", t, 0.0, cfg()).map(|_| ());
        inf(fz(&ds_inf, "d"));
        text_col(fz(&ds_text, "y_text"));
        absent(fz(&ds, "nope"));
    }

    #[test]
    fn test_na_rd_bandwidth_and_fuzzy_drop_missing_rows() {
        let ds = rd_data();
        let n = ds.df().height();
        let keep = |dropped: &[usize]| -> Dataset {
            let mask: BooleanChunked = (0..n).map(|i| !dropped.contains(&i)).collect();
            Dataset::new(ds.df().filter(&mask).unwrap())
        };
        // rd_bandwidth: two missing outcomes
        let dn = with_cells(&ds, "y", &[3, 400], None);
        let bw =
            |d: &Dataset| rd_bandwidth(d, "y", "x", 0.0, 1, TRI, BandwidthMethod::MseRd).unwrap();
        let (a, b) = (bw(&dn), bw(&keep(&[3, 400])));
        assert_eq!(
            (
                a.h_left, a.h_right, a.b_left, a.b_right, a.n_left, a.n_right
            ),
            (
                b.h_left, b.h_right, b.b_left, b.b_right, b.n_left, b.n_right
            )
        );
        assert_eq!(a.missing.n_dropped, 2);
        assert!(
            a.to_string()
                .contains("Missing values: 2 of 1000 rows dropped"),
            "{a}"
        );
        // run_fuzzy_rd: two missing treatments
        let dn = with_cells(&ds, "d", &[5, 700], None);
        let cfg = || fixed(TRI, VceType::Nn, 0.4, 0.4, 1);
        let f = run_fuzzy_rd(&dn, "yf", "x", "d", 0.0, cfg()).unwrap();
        let g = run_fuzzy_rd(&keep(&[5, 700]), "yf", "x", "d", 0.0, cfg()).unwrap();
        assert_eq!(
            (f.tau_fuzzy, f.se_fuzzy, f.tau_fuzzy_bc, f.se_fuzzy_robust),
            (g.tau_fuzzy, g.se_fuzzy, g.tau_fuzzy_bc, g.se_fuzzy_robust)
        );
        assert_eq!(f.missing.by_column, vec![("d".to_string(), 2)]);
        assert!(
            f.to_string()
                .contains("Missing values: 2 of 1000 rows dropped"),
            "{f}"
        );
    }
}
