//! Generalized Least Squares (GLS) regression.
//!
//! GLS extends OLS to handle correlated and heteroscedastic errors.
//! The model is Y = Xβ + ε where Var(ε) = σ²Ω.
//!
//! # Structures
//!
//! - Identity (OLS), AR(1) (`Ω[i, j] = ρ^|i - j|`), compound symmetry (`Ω[i, j] = ρ` off
//!   the diagonal), and a known Ω ([`gls()`] only).
//! - **Groups** (nlme `form = ~ 1 | g`): with [`GlsSpec::groups`] / [`GlsOptions::group_col`]
//!   the AR(1) or compound-symmetry correlation acts within each group, in data order, and
//!   rows of different groups are independent (Ω is block diagonal).
//! - **ρ fixed or estimated.** A given ρ is used as is (nlme `fixed = TRUE`); without one,
//!   ρ maximises nlme's restricted (REML, the default) or full (ML) profile log-likelihood,
//!   by Brent's method on the open range of ρ.
//!
//! # Log-likelihood (nlme's)
//!
//! With the whitened data `(P y, P X)`, `P'P = Ω⁻¹`, and `SSE* = |P y - P X β|²`:
//!
//! - ML: `l = -N/2 (ln 2π + 1 + ln(SSE*/N)) - ½ ln det Ω`
//! - REML: `l = -(N-p)/2 (ln 2π + 1 + ln(SSE*/(N-p))) - ½ ln det Ω - ½ ln det(X*'X*)`
//!
//! σ is `sqrt(SSE*/(N-p))` for REML and `sqrt(SSE*/N)` for ML; the standard errors use
//! `SSE*/(N-p)` for both, as nlme's `vcov`. AIC and BIC count `p + 1` parameters plus an
//! estimated ρ; BIC uses `ln(N - p)` for REML and `ln N` for ML.
//!
//! # Identification
//!
//! Compound-symmetry ρ is estimated only when it is identified: if every group indicator
//! lies in the column space of X (one group with an intercept, or group dummies among the
//! regressors), the profile likelihood is flat in ρ, and the fit is refused (give ρ).
//! A rank-deficient (whitened) design is an error, as nlme's "computed gls fit is singular".

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::{LM_QR_TOL, Qr, least_squares};
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes};
use crate::traits::estimator::t_test_p_value;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

/// Correlation structure for GLS.
#[derive(Debug, Clone)]
pub enum CorrelationStructure {
    /// AR(1) correlation: Ω[i,j] = ρ^|i-j|
    AR1 { rho: f64 },
    /// Compound symmetry: Ω[i,j] = ρ for i≠j, 1 for i=j
    CompoundSymmetry { rho: f64 },
    /// Known correlation matrix (user-provided)
    Known { omega: Array2<f64> },
    /// Identity (equivalent to OLS)
    Identity,
}

/// The likelihood GLS reports and, when ρ is estimated, maximises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum GlsMethod {
    /// Restricted maximum likelihood (nlme's default).
    #[default]
    Reml,
    /// Maximum likelihood.
    Ml,
}

impl GlsMethod {
    /// "REML" or "ML", as nlme prints it.
    pub fn label(self) -> &'static str {
        match self {
            GlsMethod::Reml => "REML",
            GlsMethod::Ml => "ML",
        }
    }
}

crate::option_enum!(
    GlsMethod,
    crate::options::Choices {
        option: "method",
        values: &[
            (&["reml", "restricted"], GlsMethod::Reml),
            (&["ml", "maximum_likelihood"], GlsMethod::Ml),
        ],
    }
);

/// A parametric correlation structure of [`gls_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlsCorrelation {
    /// Independent errors (OLS).
    #[default]
    Identity,
    /// AR(1) within groups (nlme `corAR1`).
    Ar1,
    /// Compound symmetry within groups (nlme `corCompSymm`).
    CompoundSymmetry,
}

crate::option_enum!(
    GlsCorrelation,
    crate::options::Choices {
        option: "correlation",
        values: &[
            (&["identity", "ols", "none"], GlsCorrelation::Identity),
            (&["ar1", "ar(1)", "corar1"], GlsCorrelation::Ar1),
            (
                &["compound_symmetry", "cs", "exchangeable", "corcompsymm"],
                GlsCorrelation::CompoundSymmetry
            ),
        ],
    }
);

/// A GLS model for [`gls_with`].
#[derive(Debug, Clone, Default)]
pub struct GlsSpec {
    /// The correlation structure.
    pub correlation: GlsCorrelation,
    /// Fixed correlation parameter; `None` estimates it by `method` (nlme without
    /// `fixed = TRUE`).
    pub rho: Option<f64>,
    /// Group code per row (nlme `form = ~ 1 | g`): the correlation acts within groups, in row
    /// order; rows of different groups are independent. `None` = one group of all rows.
    pub groups: Option<Vec<u32>>,
    /// REML (default) or ML.
    pub method: GlsMethod,
}

/// Options of [`run_gls_with`]: [`GlsSpec`] with the group given as a column.
#[derive(Debug, Clone, Default)]
pub struct GlsOptions {
    /// The correlation structure.
    pub correlation: GlsCorrelation,
    /// Fixed correlation parameter; `None` estimates it.
    pub rho: Option<f64>,
    /// Grouping column (any type, read as a factor); `None` = one group.
    pub group_col: Option<String>,
    /// REML (default) or ML.
    pub method: GlsMethod,
}

/// Result of GLS estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlsResult {
    /// Coefficient estimates
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// t-statistics
    pub t_values: Vec<f64>,
    /// p-values (two-sided)
    pub p_values: Vec<f64>,
    /// Residuals
    #[serde(skip)]
    pub residuals: Vec<f64>,
    /// Fitted values
    #[serde(skip)]
    pub fitted: Vec<f64>,
    /// Residual standard error (σ; REML: SSE*/(n - p), ML: SSE*/n)
    pub sigma: f64,
    /// R-squared (generalized)
    pub r_squared: f64,
    /// Adjusted R-squared
    pub adj_r_squared: f64,
    /// Log-likelihood (nlme's, by `method`)
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC (`ln(n - p)` for REML, `ln n` for ML)
    pub bic: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Degrees of freedom for residuals
    pub df_residual: usize,
    /// Number of parameters (coefficients, σ and an estimated ρ)
    pub n_params: usize,
    /// Correlation structure used
    pub correlation: String,
    /// Correlation parameter (given or estimated)
    pub correlation_param: Option<f64>,
    /// ρ was estimated (not given).
    #[serde(default)]
    pub rho_estimated: bool,
    /// The likelihood reported (and maximised when ρ was estimated).
    #[serde(default)]
    pub method: GlsMethod,
    /// Number of groups, when groups were given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_groups: Option<usize>,
    /// The ρ search, when ρ was estimated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub convergence: Option<Convergence>,
    /// Notes on the fit (boundary estimates, non-convergence); absent from JSON when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl ResultNotes for GlsResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        self.convergence.clone()
    }
}

/// Fit a GLS model.
///
/// Estimates Y = Xβ + ε where Var(ε) = σ²Ω using generalized least squares, at the given
/// structure (no estimation), and reports nlme's REML log-likelihood, AIC and BIC.
///
/// The data are whitened with a matrix `P` such that `P'P = Ω⁻¹` and OLS is fit
/// to `(P y, P X)`, which gives exactly the GLS estimator, its standard errors
/// and σ. `P` is the Prais-Winsten transform for AR(1), a closed-form symmetric
/// square root of `Ω⁻¹` for compound symmetry (both O(n p)), no transform for
/// the identity, and `L⁻¹` applied by forward substitution, where `Ω = L L'`, for
/// a known `Ω` (an O(n³) Cholesky factor, allocated fallibly, then O(n² p)
/// whitening). A rank-deficient design is an error.
///
/// # Arguments
///
/// * `y` - Response vector (n × 1)
/// * `x` - Design matrix (n × p), should include intercept if needed
/// * `correlation` - Correlation structure for errors
///
/// # Returns
///
/// A `GlsResult` containing coefficient estimates and diagnostics.
pub fn gls(
    y: &ndarray::ArrayView1<f64>,
    x: &ArrayView2<f64>,
    correlation: CorrelationStructure,
) -> EconResult<GlsResult> {
    check_shapes(y, x)?;
    let blocks = Blocks::single(y.len());
    let (structure, kind) = match &correlation {
        CorrelationStructure::Identity => (Structure::Identity, GlsCorrelation::Identity),
        CorrelationStructure::AR1 { rho } => (Structure::Ar1(*rho), GlsCorrelation::Ar1),
        CorrelationStructure::CompoundSymmetry { rho } => {
            (Structure::Cs(*rho), GlsCorrelation::CompoundSymmetry)
        }
        CorrelationStructure::Known { omega } => {
            (Structure::Known(omega), GlsCorrelation::Identity)
        }
    };
    let engine = Engine {
        y: y.view(),
        x: x.view(),
        blocks: &blocks,
        method: GlsMethod::Reml,
        names: None,
        intercept: None,
    };
    let fit = engine.fit_at(&structure)?;
    let mut r = engine.result(&fit, &structure, kind, false)?;
    if matches!(correlation, CorrelationStructure::Known { .. }) {
        r.correlation = "User-specified".to_string();
        r.correlation_param = None;
    }
    Ok(r)
}

/// Fit GLS with an AR(1), compound-symmetry or identity structure, optionally within groups,
/// with ρ fixed or estimated by REML or ML (nlme's `gls`).
///
/// `x` is the full design (include the intercept column). With `spec.rho = None` and a
/// correlation structure, ρ maximises the profile log-likelihood of `spec.method`;
/// compound-symmetry ρ is refused when it is not identified (see the module doc).
pub fn gls_with(y: &ArrayView1<f64>, x: &ArrayView2<f64>, spec: &GlsSpec) -> EconResult<GlsResult> {
    gls_engine(y, x, spec, None, None)
}

fn check_shapes(y: &ArrayView1<f64>, x: &ArrayView2<f64>) -> EconResult<()> {
    let n = y.len();
    let p = x.ncols();
    if x.nrows() != n {
        return Err(EconError::InvalidSpecification {
            message: "X and y must have the same number of observations".to_string(),
        });
    }
    if n <= p {
        return Err(EconError::InsufficientData {
            required: p + 1,
            provided: n,
            context: "GLS regression".to_string(),
        });
    }
    Ok(())
}

fn gls_engine(
    y: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    spec: &GlsSpec,
    names: Option<&[String]>,
    intercept: Option<bool>,
) -> EconResult<GlsResult> {
    resource::checkpoint()?;
    check_shapes(y, x)?;
    let n = y.len();
    if spec.correlation == GlsCorrelation::Identity && (spec.rho.is_some() || spec.groups.is_some())
    {
        return Err(EconError::InvalidSpecification {
            message: "rho and groups are used only by ar1 and compound_symmetry".to_string(),
        });
    }
    let blocks = match &spec.groups {
        None => Blocks::single(n),
        Some(codes) => Blocks::from_codes(codes, n)?,
    };
    let engine = Engine {
        y: y.view(),
        x: x.view(),
        blocks: &blocks,
        method: spec.method,
        names,
        intercept,
    };
    let structure_at = |rho: f64| match spec.correlation {
        GlsCorrelation::Identity => Structure::Identity,
        GlsCorrelation::Ar1 => Structure::Ar1(rho),
        GlsCorrelation::CompoundSymmetry => Structure::Cs(rho),
    };
    if spec.correlation != GlsCorrelation::Identity && blocks.max_size() < 2 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "{} needs a group with at least two observations",
                spec.correlation
            ),
        });
    }
    let mut result = match (spec.correlation, spec.rho) {
        (GlsCorrelation::Identity, _) => {
            let fit = engine.fit_at(&Structure::Identity)?;
            engine.result(&fit, &Structure::Identity, spec.correlation, false)?
        }
        (_, Some(rho)) => {
            let s = structure_at(rho);
            let fit = engine.fit_at(&s)?;
            engine.result(&fit, &s, spec.correlation, false)?
        }
        (corr, None) => {
            let (lo, hi) = match corr {
                GlsCorrelation::CompoundSymmetry => {
                    engine.check_cs_identified()?;
                    (-1.0 / (blocks.max_size() as f64 - 1.0), 1.0)
                }
                _ => (-1.0, 1.0),
            };
            // fail early (rank, shapes) at an interior rho
            // and take its SSE* as the reference of the centred objective
            let sse_ref = engine
                .fit_at(&structure_at(0.5 * (lo + hi)))?
                .sse
                .max(f64::MIN_POSITIVE);
            let objective = |rho: f64| -> EconResult<f64> {
                resource::checkpoint()?;
                let v = engine
                    .fit_at(&structure_at(rho))?
                    .profile_objective(sse_ref);
                Ok(if v.is_finite() { v } else { f64::INFINITY })
            };
            let b = brent_fmin(lo + 1e-8, hi - 1e-8, objective, RHO_TOL, RHO_MAX_EVAL)?;
            let s = structure_at(b.x);
            let fit = engine.fit_at(&s)?;
            let mut r = engine.result(&fit, &s, corr, true)?;
            let conv = Convergence::new(
                b.converged,
                b.evaluations,
                RHO_MAX_EVAL,
                "Brent interval in rho",
                b.final_width,
                RHO_TOL,
            );
            if let Some(note) = conv.note("GLS rho") {
                r.warnings.push(note);
            }
            if b.x - lo < 1e-6 || hi - b.x < 1e-6 {
                r.warnings.push(format!(
                    "rho estimate {} is at the boundary of ({lo}, {hi}); the likelihood has no \
                     interior maximum",
                    b.x
                ));
            }
            r.convergence = Some(conv);
            r
        }
    };
    if spec.groups.is_some() {
        result.n_groups = Some(blocks.count());
    }
    Ok(result)
}

/// Brent tolerance and evaluation limit of the ρ search.
const RHO_TOL: f64 = 1e-12;
const RHO_MAX_EVAL: usize = 200;

/// A structure at a given ρ (the known Ω is used by [`gls()`] only).
enum Structure<'a> {
    Identity,
    Ar1(f64),
    Cs(f64),
    Known(&'a Array2<f64>),
}

/// Rows grouped by code: `order[starts[g]..starts[g + 1]]` are the rows of block g, in data
/// order. `order` is `None` for a single block of all rows in order.
struct Blocks {
    order: Option<Vec<usize>>,
    starts: Vec<usize>,
}

impl Blocks {
    fn single(n: usize) -> Self {
        Blocks {
            order: None,
            starts: vec![0, n],
        }
    }

    fn from_codes(codes: &[u32], n: usize) -> EconResult<Self> {
        if codes.len() != n {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "groups has {} codes but the data have {n} rows",
                    codes.len()
                ),
            });
        }
        let mut order: Vec<usize> = resource::try_with_capacity(n, "GLS group order")?;
        order.extend(0..n);
        order.sort_by_key(|&i| codes[i]); // stable: data order within a group
        let mut starts = vec![0];
        for r in 1..n {
            if codes[order[r]] != codes[order[r - 1]] {
                starts.push(r);
            }
        }
        starts.push(n);
        Ok(Blocks {
            order: Some(order),
            starts,
        })
    }

    fn count(&self) -> usize {
        self.starts.len() - 1
    }

    fn max_size(&self) -> usize {
        self.starts
            .windows(2)
            .map(|w| w[1] - w[0])
            .max()
            .unwrap_or(0)
    }

    /// Row of the original data at position `r` of the grouped order.
    fn row(&self, r: usize) -> usize {
        self.order.as_ref().map_or(r, |o| o[r])
    }
}

/// One fit at a fixed structure.
struct FitAt {
    beta: Array1<f64>,
    unscaled: Array2<f64>,
    sse: f64,
    loglik: f64,
    /// N (ML) or N - p (REML).
    df_lik: f64,
    /// ½ ln det Ω, plus ½ ln det(X*'X*) for REML.
    penalty: f64,
}

impl FitAt {
    /// `-l` up to a constant, centred on `sse_ref`: `df/2 ln(SSE*/sse_ref) + penalty`. Its
    /// values do not grow with the scale of y, so the ρ search takes the same path at any scale.
    fn profile_objective(&self, sse_ref: f64) -> f64 {
        0.5 * self.df_lik * (self.sse / sse_ref).ln() + self.penalty
    }
}

/// The data and settings shared by every fit of one GLS call.
struct Engine<'a, 'd> {
    y: ArrayView1<'d, f64>,
    x: ArrayView2<'d, f64>,
    blocks: &'a Blocks,
    method: GlsMethod,
    /// Column names for the aliasing message (`None`: "column j").
    names: Option<&'a [String]>,
    /// Whether X has an intercept, for R²; `None` detects a constant column.
    intercept: Option<bool>,
}

impl Engine<'_, '_> {
    /// Whitened `(P y, P X)` in grouped row order and `ln det Ω`.
    fn whiten(&self, s: &Structure) -> EconResult<(Array1<f64>, Array2<f64>, f64)> {
        resource::checkpoint()?;
        let (n, p) = self.x.dim();
        let b = self.blocks;
        let m_max = b.max_size();
        match s {
            Structure::Ar1(rho) => {
                let rho = *rho;
                if !rho.is_finite() || rho.abs() >= 1.0 {
                    return Err(EconError::InvalidSpecification {
                        message: format!("AR(1) correlation |rho| must be < 1 (got {rho})"),
                    });
                }
            }
            Structure::Cs(rho) => {
                let rho = *rho;
                let lo = -1.0 / (m_max as f64 - 1.0);
                if !rho.is_finite() || rho <= lo || rho >= 1.0 {
                    let size = if b.count() == 1 {
                        format!("n = {m_max}")
                    } else {
                        format!("the largest group, n = {m_max}")
                    };
                    return Err(EconError::InvalidSpecification {
                        message: format!(
                            "compound symmetry rho must lie in (-1/(n - 1), 1) = ({lo:.6}, 1) \
                             for {size} (got {rho})"
                        ),
                    });
                }
            }
            Structure::Known(omega) => {
                if omega.nrows() != n || omega.ncols() != n {
                    return Err(EconError::InvalidSpecification {
                        message: format!("Known Omega must be {}x{}", n, n),
                    });
                }
            }
            Structure::Identity => {}
        }
        let mut y_star = resource::try_vec(n, 0.0, "GLS whitened response")?;
        let mut x_star = resource::try_zeros2(n, p, WHITENED_DESIGN)?;
        for r in 0..n {
            let i = b.row(r);
            y_star[r] = self.y[i];
            for j in 0..p {
                x_star[[r, j]] = self.x[[i, j]];
            }
        }
        let mut col = vec![0.0; n];
        let mut logdet = 0.0;
        let apply = |v: &mut [f64]| -> EconResult<()> {
            match s {
                Structure::Identity => {}
                Structure::Ar1(rho) => {
                    let scale = (1.0 - rho * rho).sqrt();
                    for g in b.starts.windows(2) {
                        let blk = &mut v[g[0]..g[1]];
                        for t in (1..blk.len()).rev() {
                            blk[t] = (blk[t] - rho * blk[t - 1]) / scale;
                        }
                    }
                }
                Structure::Cs(rho) => {
                    // Ω_g = (1 - ρ) I + ρ 11' has eigenvalue 1 + (m - 1)ρ on 1 and 1 - ρ on
                    // its complement, so P_g = c (I - a 11'/m) is symmetric with P_g² = Ω_g⁻¹.
                    let c = 1.0 / (1.0 - rho).sqrt();
                    for g in b.starts.windows(2) {
                        let blk = &mut v[g[0]..g[1]];
                        let m = blk.len() as f64;
                        let a = 1.0 - ((1.0 - rho) / (1.0 + (m - 1.0) * rho)).sqrt();
                        let mean = blk.iter().sum::<f64>() / m;
                        for e in blk.iter_mut() {
                            *e = c * (*e - a * mean);
                        }
                    }
                }
                Structure::Known(_) => unreachable!("handled with its factor"),
            }
            Ok(())
        };
        match s {
            Structure::Known(omega) => {
                // Ω = L L', P = L⁻¹: solve L z = v by forward substitution
                let l = cholesky_lower(omega)?;
                logdet = 2.0 * (0..n).map(|i| l[[i, i]].ln()).sum::<f64>();
                forward_substitute(&l, y_star.as_mut_slice())?;
                for j in 0..p {
                    resource::checkpoint()?;
                    col.iter_mut()
                        .zip(x_star.column(j))
                        .for_each(|(c, &v)| *c = v);
                    forward_substitute(&l, &mut col)?;
                    x_star
                        .column_mut(j)
                        .iter_mut()
                        .zip(&col)
                        .for_each(|(d, &v)| *d = v);
                }
            }
            _ => {
                apply(y_star.as_mut_slice())?;
                for j in 0..p {
                    resource::checkpoint()?;
                    col.iter_mut()
                        .zip(x_star.column(j))
                        .for_each(|(c, &v)| *c = v);
                    apply(&mut col)?;
                    x_star
                        .column_mut(j)
                        .iter_mut()
                        .zip(&col)
                        .for_each(|(d, &v)| *d = v);
                }
                for g in b.starts.windows(2) {
                    let m = (g[1] - g[0]) as f64;
                    logdet += match s {
                        Structure::Ar1(rho) => (m - 1.0) * (1.0 - rho * rho).ln(),
                        Structure::Cs(rho) => {
                            (m - 1.0) * (1.0 - rho).ln() + (1.0 + (m - 1.0) * rho).ln()
                        }
                        _ => 0.0,
                    };
                }
            }
        }
        Ok((Array1::from_vec(y_star), x_star, logdet))
    }

    /// Fit at `s`: coefficients, `(X*'X*)^-1`, SSE* and the log-likelihood of `method`.
    fn fit_at(&self, s: &Structure) -> EconResult<FitAt> {
        let (n, p) = self.x.dim();
        let (ys, xs, logdet) = self.whiten(s)?;
        let sol = least_squares(xs.view(), ys.view())?;
        if sol.rank < p {
            let cols: Vec<String> = sol
                .aliased
                .iter()
                .map(|&j| match self.names {
                    Some(nm) if j < nm.len() => format!("'{}'", nm[j]),
                    _ => format!("column {}", j + 1),
                })
                .collect();
            return Err(EconError::SingularMatrix {
                context: format!(
                    "GLS: the whitened design is rank deficient (rank {} of {p}); aliased: {}",
                    sol.rank,
                    cols.join(", ")
                ),
                suggestion: "drop collinear regressors".to_string(),
            });
        }
        let sse = sol.residuals.dot(&sol.residuals);
        let unscaled = sol.unscaled_cov;
        let (nf, pf) = (n as f64, p as f64);
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        // l = -df/2 (ln 2π + 1 + ln(SSE*/df)) - penalty, with df = N (ML) or N - p (REML)
        let (df_lik, penalty) = match self.method {
            GlsMethod::Ml => (nf, 0.5 * logdet),
            GlsMethod::Reml => {
                // ½ ln det(X*'X*) = -½ ln det((X*'X*)^-1) = -Σ ln L_ii of its Cholesky factor
                let l = cholesky_lower(&unscaled)?;
                let half_logdet_xtx = -(0..p).map(|i| l[[i, i]].ln()).sum::<f64>();
                (nf - pf, 0.5 * logdet + half_logdet_xtx)
            }
        };
        let loglik = -0.5 * df_lik * (ln2pi + 1.0 + (sse / df_lik).ln()) - penalty;
        Ok(FitAt {
            beta: sol.coefficients,
            unscaled,
            sse,
            loglik,
            df_lik,
            penalty,
        })
    }

    /// The result at a fit.
    fn result(
        &self,
        fit: &FitAt,
        s: &Structure,
        kind: GlsCorrelation,
        estimated: bool,
    ) -> EconResult<GlsResult> {
        let (n, p) = self.x.dim();
        let (nf, pf) = (n as f64, p as f64);
        let df_residual = n - p;
        let beta = &fit.beta;
        let fitted: Array1<f64> = self.x.dot(beta);
        let residuals: Array1<f64> = self.y.to_owned() - &fitted;
        let s2_unbiased = fit.sse / (nf - pf);
        let sigma = match self.method {
            GlsMethod::Reml => s2_unbiased.sqrt(),
            GlsMethod::Ml => (fit.sse / nf).sqrt(),
        };
        let std_errors: Array1<f64> = fit
            .unscaled
            .diag()
            .mapv(|v| (v * s2_unbiased).max(0.0).sqrt());
        let t_values: Array1<f64> = beta / &std_errors;
        let p_values: Array1<f64> = t_values.mapv(|t| t_test_p_value(t, df_residual as f64));

        // R-squared (using original residuals). As summary.lm (REG-19): the total sum of
        // squares is about the mean when X has an intercept and about zero otherwise, and
        // adjusted R² uses n - 1 or n accordingly.
        let has_intercept = self
            .intercept
            .unwrap_or_else(|| super::ols::has_constant_column(self.x));
        let sst: f64 = if has_intercept {
            let y_mean = self.y.sum() / nf;
            self.y.iter().map(|yi| (yi - y_mean).powi(2)).sum()
        } else {
            self.y.dot(&self.y)
        };
        let ss_res: f64 = residuals.dot(&residuals);
        let r_squared = if sst > 0.0 { 1.0 - ss_res / sst } else { 0.0 };
        let adj_r_squared =
            1.0 - (1.0 - r_squared) * (n - usize::from(has_intercept)) as f64 / df_residual as f64;

        let n_params = p + 1 + usize::from(estimated);
        let k = n_params as f64;
        let aic = -2.0 * fit.loglik + 2.0 * k;
        let bic = -2.0 * fit.loglik
            + k * match self.method {
                GlsMethod::Reml => (nf - pf).ln(),
                GlsMethod::Ml => nf.ln(),
            };
        let (corr_name, corr_param) = match (kind, s) {
            (GlsCorrelation::Ar1, Structure::Ar1(rho)) => ("AR(1)".to_string(), Some(*rho)),
            (GlsCorrelation::CompoundSymmetry, Structure::Cs(rho)) => {
                ("Compound Symmetry".to_string(), Some(*rho))
            }
            _ => ("Identity (OLS)".to_string(), None),
        };
        Ok(GlsResult {
            coefficients: beta.to_vec(),
            std_errors: std_errors.to_vec(),
            t_values: t_values.to_vec(),
            p_values: p_values.to_vec(),
            residuals: residuals.to_vec(),
            fitted: fitted.to_vec(),
            sigma,
            r_squared,
            adj_r_squared,
            log_likelihood: fit.loglik,
            aic,
            bic,
            n_obs: n,
            df_residual,
            n_params,
            correlation: corr_name,
            correlation_param: corr_param,
            rho_estimated: estimated,
            method: self.method,
            n_groups: None,
            convergence: None,
            warnings: Vec::new(),
        })
    }

    /// Compound-symmetry ρ is identified unless every group indicator lies in col(X).
    fn check_cs_identified(&self) -> EconResult<()> {
        let (n, p) = self.x.dim();
        let b = self.blocks;
        let mut wx = resource::try_zeros2(n, p, "GLS within-group design")?;
        for j in 0..p {
            let mut norm_x = 0.0f64;
            let mut norm_w = 0.0f64;
            for g in b.starts.windows(2) {
                let m = (g[1] - g[0]) as f64;
                let mean = (g[0]..g[1]).map(|r| self.x[[b.row(r), j]]).sum::<f64>() / m;
                for r in g[0]..g[1] {
                    let v = self.x[[b.row(r), j]];
                    let w = v - mean;
                    wx[[r, j]] = w;
                    norm_x += v * v;
                    norm_w += w * w;
                }
            }
            // a group-constant column leaves rounding noise only
            if norm_w.sqrt() <= LM_QR_TOL * norm_x.sqrt() {
                wx.column_mut(j).fill(0.0);
            }
        }
        let rank_w = Qr::decompose(wx.view(), LM_QR_TOL)?.rank();
        let rank_x = Qr::decompose(self.x, LM_QR_TOL)?.rank();
        if b.count() + rank_w == rank_x {
            return Err(EconError::InvalidSpecification {
                message: "compound-symmetry rho is not identified: every group indicator lies in \
                          the column space of X (one group with an intercept, or group dummies \
                          among the regressors); give rho, or groups whose effects are not in X"
                    .to_string(),
            });
        }
        Ok(())
    }
}

/// Outcome of [`brent_fmin`].
struct BrentMin {
    x: f64,
    evaluations: usize,
    converged: bool,
    final_width: f64,
}

/// Brent's minimiser on `[ax, bx]` (a port of R's `optimize.c` `Brent_fmin`), with at most
/// `max_eval` evaluations of `f`.
fn brent_fmin(
    ax: f64,
    bx: f64,
    mut f: impl FnMut(f64) -> EconResult<f64>,
    tol: f64,
    max_eval: usize,
) -> EconResult<BrentMin> {
    let c = (3.0 - 5.0f64.sqrt()) * 0.5;
    let eps = f64::EPSILON.sqrt();
    let (mut a, mut b) = (ax, bx);
    let mut v = a + c * (b - a);
    let mut w = v;
    let mut x = v;
    let mut d = 0.0f64;
    let mut e = 0.0f64;
    let mut fx = f(x)?;
    let mut evaluations = 1;
    let mut fv = fx;
    let mut fw = fx;
    let tol3 = tol / 3.0;
    let converged;
    loop {
        let xm = (a + b) * 0.5;
        let tol1 = eps * x.abs() + tol3;
        let t2 = tol1 * 2.0;
        if (x - xm).abs() <= t2 - (b - a) * 0.5 {
            converged = true;
            break;
        }
        if evaluations >= max_eval {
            converged = false;
            break;
        }
        let (mut p, mut q, mut r) = (0.0f64, 0.0f64, 0.0f64);
        if e.abs() > tol1 {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = (q - r) * 2.0;
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            r = e;
            e = d;
        }
        if p.abs() >= (q * 0.5 * r).abs() || p <= q * (a - x) || p >= q * (b - x) {
            // golden-section step
            e = if x < xm { b - x } else { a - x };
            d = c * e;
        } else {
            // parabolic-interpolation step
            d = p / q;
            let u = x + d;
            if u - a < t2 || b - u < t2 {
                d = if x < xm { tol1 } else { -tol1 };
            }
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };
        let fu = f(u)?;
        evaluations += 1;
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    Ok(BrentMin {
        x,
        evaluations,
        converged,
        final_width: b - a,
    })
}

/// Allocation label for the whitened design `X*`, shared by every structure.
const WHITENED_DESIGN: &str = "GLS whitened design";

/// Solve `L z = v` in place for a lower-triangular `L` (forward substitution).
fn forward_substitute(l: &Array2<f64>, v: &mut [f64]) -> EconResult<()> {
    for i in 0..v.len() {
        let diag = l[[i, i]];
        if diag.abs() < 1e-15 {
            return Err(EconError::SingularMatrix {
                context: "Lower triangular matrix has zero on diagonal".to_string(),
                suggestion: "Check matrix conditioning".to_string(),
            });
        }
        let mut sum = v[i];
        for k in 0..i {
            sum -= l[[i, k]] * v[k];
        }
        v[i] = sum / diag;
    }
    Ok(())
}

/// Cholesky decomposition (lower triangular L such that A = LL').
fn cholesky_lower(a: &Array2<f64>) -> EconResult<Array2<f64>> {
    let n = a.nrows();
    let mut l = resource::try_zeros2(n, n, "GLS Cholesky factor")?;

    for i in 0..n {
        resource::checkpoint()?;
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }

            if i == j {
                let diag = a[[i, i]] - sum;
                if diag <= 0.0 {
                    return Err(EconError::SingularMatrix {
                        context: "Cholesky decomposition failed: matrix not positive definite"
                            .to_string(),
                        suggestion: "Check that the correlation matrix is positive definite"
                            .to_string(),
                    });
                }
                l[[i, j]] = diag.sqrt();
            } else {
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Ok(l)
}

/// Fit GLS with an AR(1) correlation whose ρ is nlme's REML estimate (`gls(correlation =
/// corAR1())`), on one group of all rows in order.
pub fn gls_ar1_auto(y: &ndarray::ArrayView1<f64>, x: &ArrayView2<f64>) -> EconResult<GlsResult> {
    let spec = GlsSpec {
        correlation: GlsCorrelation::Ar1,
        ..GlsSpec::default()
    };
    gls_with(y, x, &spec)
}

/// Run GLS from raw flat slices.
///
/// Lower-level entry point that takes a flat row-major X buffer. Most callers
/// should prefer [`run_gls`], which extracts columns from a `Dataset`.
pub fn run_gls_raw(
    y: &[f64],
    x: &[f64],
    n_cols: usize,
    correlation_type: &str,
    correlation_param: Option<f64>,
) -> EconResult<GlsResult> {
    let n = y.len();
    let y_arr = Array1::from_vec(y.to_vec());
    let x_arr = Array2::from_shape_vec((n, n_cols), x.to_vec()).map_err(|e| {
        EconError::InvalidSpecification {
            message: format!("Invalid X matrix shape: {}", e),
        }
    })?;
    let spec = spec_from_strings(correlation_type, correlation_param)?;
    gls_engine(&y_arr.view(), &x_arr.view(), &spec, None, None)
}

/// Run GLS against a `Dataset`, matching the column-based pattern used by
/// [`crate::run_ols`] and the rest of the regression family.
///
/// # Arguments
/// * `dataset` - The source dataset.
/// * `y_col` - Name of the dependent variable column.
/// * `x_cols` - Names of the regressor columns.
/// * `intercept` - If `true`, a leading intercept column of ones is added to X.
/// * `correlation_type` - `"identity"` (aliases `"ols"`, `"none"`), `"ar1"`,
///   `"compound_symmetry"` (aliases `"cs"`, `"exchangeable"`), or `"ar1_auto"` (alias
///   `"auto"`: AR(1) with ρ estimated).
/// * `correlation_param` - A fixed ρ for `"ar1"` and `"compound_symmetry"`; `None` estimates
///   ρ by REML (nlme's default). Ignored by `"identity"` and `"ar1_auto"`.
///
/// Use [`run_gls_with`] for groups and ML.
pub fn run_gls(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    intercept: bool,
    correlation_type: &str,
    correlation_param: Option<f64>,
) -> EconResult<GlsResult> {
    let spec = spec_from_strings(correlation_type, correlation_param)?;
    let options = GlsOptions {
        correlation: spec.correlation,
        rho: spec.rho,
        group_col: None,
        method: spec.method,
    };
    run_gls_with(dataset, y_col, x_cols, intercept, &options)
}

/// GLS against a `Dataset` with groups, a fixed or estimated ρ, and REML or ML (nlme's
/// `gls(y ~ x, correlation = corAR1(form = ~ 1 | g), method = ...)`).
///
/// Rows are used in data order (the AR(1) chain follows it within each group); a missing
/// value in any model column or the group column is an error, since dropping a row would
/// change the correlation structure.
pub fn run_gls_with(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    intercept: bool,
    options: &GlsOptions,
) -> EconResult<GlsResult> {
    let df = dataset.df();
    let mut request = ColumnRequest::new(df).numeric(y_col).numerics(x_cols);
    if let Some(g) = &options.group_col {
        request = request.factor(g);
    }
    let cols = request.na(NaPolicy::Fail).extract()?;
    let design = cols.matrix(x_cols, intercept)?;
    let y = cols.numeric_array(y_col);
    let groups = options
        .group_col
        .as_ref()
        .map(|g| cols.factor(g).codes.clone());
    let spec = GlsSpec {
        correlation: options.correlation,
        rho: options.rho,
        groups,
        method: options.method,
    };
    gls_engine(
        &y.view(),
        &design.data.view(),
        &spec,
        Some(&design.column_names),
        Some(intercept),
    )
}

/// The [`GlsSpec`] (REML, one group) of the string entry points.
fn spec_from_strings(correlation_type: &str, rho: Option<f64>) -> EconResult<GlsSpec> {
    let lower = correlation_type.trim().to_lowercase();
    let (correlation, rho) = if lower == "ar1_auto" || lower == "auto" {
        (GlsCorrelation::Ar1, None)
    } else {
        match correlation_type.parse::<GlsCorrelation>() {
            Ok(GlsCorrelation::Identity) => (GlsCorrelation::Identity, None),
            Ok(c) => (c, rho),
            Err(_) => {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "unknown correlation type '{correlation_type}': expected identity | ar1 | \
                         compound_symmetry, or ar1_auto (AR(1) with rho estimated)"
                    ),
                });
            }
        }
    };
    Ok(GlsSpec {
        correlation,
        rho,
        groups: None,
        method: GlsMethod::Reml,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::matrix_ops::safe_inverse;
    use ndarray::array;

    /// GLS with identity correlation is OLS: pinned to lm without intercept (ols_no_intercept_grunfeld)
    /// and, for the centered R² with intercept, to ols_grunfeld. REG-19.
    #[test]
    fn test_validate_gls_identity_no_intercept_grunfeld() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("ols_no_intercept_grunfeld");
        let n = case.scalar("n_obs") as usize;
        let r = run_gls(&ds, "inv", &["value", "capital"], false, "identity", None).unwrap();
        let names = ["value", "capital"];
        Check::new(&case)
            .named("coef", &names, &r.coefficients, Tol::coef(n))
            .named("se_classical", &names, &r.std_errors, Tol::se(n))
            .named("t_classical", &names, &r.t_values, Tol::stat(n))
            .named("p_classical", &names, &r.p_values, Tol::p_value(n))
            .scalar("r2", r.r_squared, Tol::stat(n))
            .scalar("adj_r2", r.adj_r_squared, Tol::stat(n))
            .scalar("sigma", r.sigma, Tol::stat(n))
            .finish();
        let (with, ds) = crate::test_support::load_case("ols_grunfeld");
        let r = run_gls(&ds, "inv", &["value", "capital"], true, "identity", None).unwrap();
        Check::new(&with)
            .scalar("r2", r.r_squared, Tol::stat(n))
            .scalar("adj_r2", r.adj_r_squared, Tol::stat(n))
            .finish();
    }

    #[test]
    fn test_gls_identity() {
        // With identity correlation, GLS should equal OLS
        let y = array![1.0, 2.1, 2.9, 4.1, 5.0];
        let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]];

        let result = gls(&y.view(), &x.view(), CorrelationStructure::Identity).unwrap();

        // Slope should be close to 1
        assert!((result.coefficients[1] - 1.0).abs() < 0.2);
        assert!(result.r_squared > 0.9);
    }

    #[test]
    fn test_gls_ar1() {
        let y = array![1.0, 2.1, 2.9, 4.1, 5.0, 6.0, 7.1, 8.0, 9.0, 10.1];
        let x = array![
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0],
            [1.0, 6.0],
            [1.0, 7.0],
            [1.0, 8.0],
            [1.0, 9.0],
            [1.0, 10.0]
        ];

        let result = gls(&y.view(), &x.view(), CorrelationStructure::AR1 { rho: 0.5 }).unwrap();

        // Should still get reasonable estimates
        assert!((result.coefficients[1] - 1.0).abs() < 0.3);
        assert!(result.r_squared > 0.9);
    }

    #[test]
    fn test_gls_compound_symmetry() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]];

        let result = gls(
            &y.view(),
            &x.view(),
            CorrelationStructure::CompoundSymmetry { rho: 0.3 },
        )
        .unwrap();

        assert!((result.coefficients[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_gls_ar1_auto() {
        let y = array![1.0, 2.1, 2.9, 4.1, 5.0, 6.0, 7.1, 8.0, 9.0, 10.1];
        let x = array![
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0],
            [1.0, 6.0],
            [1.0, 7.0],
            [1.0, 8.0],
            [1.0, 9.0],
            [1.0, 10.0]
        ];

        let result = gls_ar1_auto(&y.view(), &x.view()).unwrap();

        // Should estimate rho automatically
        assert!(result.correlation_param.is_some());
        assert!(result.coefficients[1] > 0.5);
    }

    /// Dense n x n correlation matrix (the pre-T0-WP14 `build_correlation_matrix`),
    /// kept as a test reference for the closed-form transforms.
    fn dense_correlation_matrix(n: usize, structure: &CorrelationStructure) -> Array2<f64> {
        match structure {
            CorrelationStructure::AR1 { rho } => {
                let mut omega = Array2::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        let lag = (i as i64 - j as i64).unsigned_abs() as i32;
                        omega[[i, j]] = rho.powi(lag);
                    }
                }
                omega
            }
            CorrelationStructure::CompoundSymmetry { rho } => {
                let mut omega = Array2::from_elem((n, n), *rho);
                for i in 0..n {
                    omega[[i, i]] = 1.0;
                }
                omega
            }
            CorrelationStructure::Known { omega } => omega.clone(),
            CorrelationStructure::Identity => Array2::eye(n),
        }
    }

    /// Textbook GLS with an explicit inverse: beta = (X'O^-1 X)^-1 X'O^-1 y,
    /// sigma^2 = e'O^-1 e / (n - p). Returns (beta, standard errors, sigma).
    fn dense_gls_reference(
        y: &Array1<f64>,
        x: &Array2<f64>,
        omega: &Array2<f64>,
    ) -> (Array1<f64>, Array1<f64>, f64) {
        let (omega_inv, _) = safe_inverse(&omega.view()).unwrap();
        let xto = x.t().dot(&omega_inv);
        let (a_inv, _) = safe_inverse(&xto.dot(x).view()).unwrap();
        let beta = a_inv.dot(&xto.dot(y));
        let e = y - &x.dot(&beta);
        let (n, p) = x.dim();
        let sigma2 = e.dot(&omega_inv.dot(&e)) / (n - p) as f64;
        let se = a_inv.diag().mapv(|v| (v * sigma2).sqrt());
        (beta, se, sigma2.sqrt())
    }

    const NLME_X: [f64; 30] = [
        5.637, 8.080, 7.150, 8.753, 6.938, 6.813, 8.448, 7.718, 7.320, 6.537, 2.562, 2.370, 2.185,
        0.415, 6.587, 0.424, 9.626, 7.693, 8.135, 7.260, 3.562, 6.383, 3.659, 6.179, 5.827, 9.956,
        1.355, 9.545, 1.014, 3.836,
    ];
    const NLME_Y: [f64; 30] = [
        6.5899, 7.4509, 8.7175, 8.8798, 7.3817, 6.2093, 7.3718, 8.4644, 8.4831, 8.5294, 4.1718,
        3.3226, 3.6097, 2.7661, 8.2699, 2.9248, 9.6094, 8.4130, 8.1559, 7.5124, 5.1513, 7.7384,
        4.7780, 7.3021, 5.5900, 9.2503, 1.8103, 8.8721, 2.9977, 4.4985,
    ];

    fn nlme_xy() -> (Array1<f64>, Array2<f64>) {
        let x = Array2::from_shape_fn((30, 2), |(i, j)| if j == 0 { 1.0 } else { NLME_X[i] });
        (Array1::from_vec(NLME_Y.to_vec()), x)
    }

    /// GLS with a fixed correlation vs `nlme::gls` (nlme 3.1.164; REML sigma).
    ///
    /// ```r
    /// library(nlme); set.seed(20261021); n <- 30
    /// x <- round(runif(n, 0, 10), 3); e <- numeric(n); e[1] <- rnorm(1)
    /// for (i in 2:n) e[i] <- 0.6 * e[i - 1] + rnorm(1, sd = 0.8)
    /// y <- round(1.5 + 0.8 * x + e, 4); d <- data.frame(y = y, x = x)
    /// gls(y ~ x, data = d, correlation = corAR1(0.6, fixed = TRUE))
    /// # and corAR1(-0.4), corCompSymm(0.3), corCompSymm(-0.02), no correlation
    /// # reported: coef(fit), summary(fit)$tTable[, 2], fit$sigma
    /// ```
    #[test]
    fn test_validate_gls_fixed_correlation_vs_nlme() {
        let (y, x) = nlme_xy();
        let cases = [
            (
                CorrelationStructure::AR1 { rho: 0.6 },
                [
                    1.941977840640,
                    0.791557641587,
                    0.387453755473,
                    0.040120184694,
                    0.903229314501,
                ],
            ),
            (
                CorrelationStructure::AR1 { rho: -0.4 },
                [
                    2.213916554461,
                    0.747378376163,
                    0.307612193194,
                    0.049986850996,
                    0.906511290861,
                ],
            ),
            (
                CorrelationStructure::CompoundSymmetry { rho: 0.3 },
                [
                    2.093925805991,
                    0.767614285417,
                    0.552217510279,
                    0.046135388346,
                    0.852538446062,
                ],
            ),
            (
                CorrelationStructure::CompoundSymmetry { rho: -0.02 },
                [
                    2.093925805991,
                    0.767614285417,
                    0.277347530251,
                    0.046135388346,
                    0.706257230683,
                ],
            ),
            (
                CorrelationStructure::Identity,
                [
                    2.093925805991,
                    0.767614285417,
                    0.294784053256,
                    0.046135388346,
                    0.713284838904,
                ],
            ),
        ];
        for (corr, r) in cases {
            let label = format!("{corr:?}");
            let g = gls(&y.view(), &x.view(), corr).unwrap();
            let got = [
                g.coefficients[0],
                g.coefficients[1],
                g.std_errors[0],
                g.std_errors[1],
                g.sigma,
            ];
            for (a, e) in got.iter().zip(r) {
                assert!((a - e).abs() < 1e-8, "{label}: got {got:?}, nlme {r:?}");
            }
        }
    }

    #[test]
    fn test_gls_closed_forms_match_dense_reference() {
        let n = 25;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => 1.0,
            1 => (i as f64 * 0.7).sin() * 3.0 + i as f64 * 0.1,
            _ => ((i * 13) % 7) as f64 - 3.0,
        });
        let y = Array1::from_shape_fn(n, |i| {
            1.0 + 0.5 * x[[i, 1]] - 0.3 * x[[i, 2]] + (i as f64 * 1.9).cos() * 0.4
        });
        for corr in [
            CorrelationStructure::AR1 { rho: 0.9 },
            CorrelationStructure::AR1 { rho: 0.3 },
            CorrelationStructure::AR1 { rho: -0.7 },
            CorrelationStructure::CompoundSymmetry { rho: 0.5 },
            CorrelationStructure::CompoundSymmetry { rho: -0.03 },
            CorrelationStructure::Identity,
        ] {
            let omega = dense_correlation_matrix(n, &corr);
            let (b_ref, se_ref, sigma_ref) = dense_gls_reference(&y, &x, &omega);
            let g = gls(&y.view(), &x.view(), corr.clone()).unwrap();
            for j in 0..3 {
                assert!(
                    (g.coefficients[j] - b_ref[j]).abs() < 1e-10
                        && (g.std_errors[j] - se_ref[j]).abs() < 1e-10,
                    "{corr:?}: {:?} vs {b_ref}, {:?} vs {se_ref}",
                    g.coefficients,
                    g.std_errors
                );
            }
            assert!((g.sigma - sigma_ref).abs() < 1e-10, "{corr:?}");
            let known = gls(&y.view(), &x.view(), CorrelationStructure::Known { omega }).unwrap();
            for j in 0..3 {
                assert!(
                    (known.coefficients[j] - g.coefficients[j]).abs() < 1e-10,
                    "Known vs closed form {corr:?}"
                );
            }
        }
    }

    #[test]
    fn test_gls_compound_symmetry_rho_error_names_value_and_bound() {
        // n = 5: rho must lie in (-0.25, 1)
        let y = array![1.0, 2.1, 2.9, 4.1, 5.0];
        let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]];
        let e = gls(
            &y.view(),
            &x.view(),
            CorrelationStructure::CompoundSymmetry { rho: -0.3 },
        )
        .unwrap_err();
        match e {
            EconError::InvalidSpecification { message } => assert!(
                message.contains("(-0.250000, 1)")
                    && message.contains("n = 5")
                    && message.contains("got -0.3"),
                "{message}"
            ),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn test_gls_rejects_non_finite_rho() {
        let (y, x) = nlme_xy();
        for corr in [
            CorrelationStructure::AR1 { rho: f64::NAN },
            CorrelationStructure::CompoundSymmetry { rho: f64::NAN },
        ] {
            let e = gls(&y.view(), &x.view(), corr).unwrap_err();
            assert!(matches!(e, EconError::InvalidSpecification { .. }), "{e:?}");
        }
    }

    #[test]
    fn test_gls_whitened_design_respects_alloc_ceiling() {
        // X* is 2000 x 3 x 8 B = 48,000 B, above a 32 KiB ceiling
        let n = 2000;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => 1.0,
            1 => (i as f64 * 0.37).sin() * 5.0,
            _ => ((i * 13) % 7) as f64 - 3.0,
        });
        let y = Array1::from_shape_fn(n, |i| 2.0 + 0.5 * x[[i, 1]] + (i as f64 * 2.1).cos());
        for corr in [
            CorrelationStructure::AR1 { rho: 0.5 },
            CorrelationStructure::CompoundSymmetry { rho: 0.3 },
            CorrelationStructure::Identity,
        ] {
            let label = format!("{corr:?}");
            let e =
                resource::with_alloc_ceiling(Some(32 << 10), || gls(&y.view(), &x.view(), corr))
                    .unwrap_err();
            assert!(
                matches!(&e, EconError::ResourceLimit(m) if m.contains("GLS whitened design")),
                "{label}: {e:?}"
            );
        }
    }

    fn cancelled() -> resource::Budget {
        let t = resource::CancelToken::new();
        t.cancel();
        resource::Budget::new(None, t)
    }

    #[test]
    fn test_gls_stops_when_cancelled() {
        let (y, x) = nlme_xy();
        let omega = dense_correlation_matrix(y.len(), &CorrelationStructure::AR1 { rho: 0.5 });
        for corr in [
            CorrelationStructure::AR1 { rho: 0.5 },
            CorrelationStructure::CompoundSymmetry { rho: 0.3 },
            CorrelationStructure::Known { omega },
            CorrelationStructure::Identity,
        ] {
            let label = match &corr {
                CorrelationStructure::Known { .. } => "Known".to_string(),
                c => format!("{c:?}"),
            };
            let e =
                resource::with_budget(cancelled(), || gls(&y.view(), &x.view(), corr)).unwrap_err();
            assert!(matches!(e, EconError::Cancelled(_)), "{label}: {e:?}");
        }
        // The string entry points, including the identity aliases
        let x_flat: Vec<f64> = x.iter().copied().collect();
        for kind in ["identity", "ols", "ar1_auto"] {
            let e = resource::with_budget(cancelled(), || {
                run_gls_raw(y.as_slice().unwrap(), &x_flat, 2, kind, None)
            })
            .unwrap_err();
            assert!(matches!(e, EconError::Cancelled(_)), "{kind}: {e:?}");
        }
    }

    #[test]
    fn test_gls_known_omega_respects_alloc_ceiling() {
        let n = 400;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                1.0
            } else {
                (i as f64 * 0.37).sin() * 5.0
            }
        });
        let y = Array1::from_shape_fn(n, |i| 2.0 + 0.5 * x[[i, 1]] + (i as f64 * 2.1).cos());
        let omega = dense_correlation_matrix(n, &CorrelationStructure::AR1 { rho: 0.5 });
        // L is 400*400*8 = 1.28 MB
        let e = crate::resource::with_alloc_ceiling(Some(256 << 10), || {
            gls(&y.view(), &x.view(), CorrelationStructure::Known { omega })
        })
        .unwrap_err();
        assert!(matches!(e, EconError::ResourceLimit(_)), "{e:?}");
    }

    #[test]
    fn test_gls_ar1_large_n_is_linear() {
        // Post-fix guard. Do NOT run against the dense code: it allocates
        // several 320 GB matrices.
        let n = 200_000;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                1.0
            } else {
                (i % 1000) as f64 / 100.0
            }
        });
        let y = Array1::from_shape_fn(n, |i| {
            1.0 + 0.8 * x[[i, 1]] + ((i * 7919) % 1009) as f64 / 1009.0
        });
        let g = gls(&y.view(), &x.view(), CorrelationStructure::AR1 { rho: 0.5 }).unwrap();
        assert_eq!(g.n_obs, n);
        assert!((g.coefficients[1] - 0.8).abs() < 0.05);
    }

    #[test]
    fn test_build_ar1_matrix() {
        let omega = dense_correlation_matrix(4, &CorrelationStructure::AR1 { rho: 0.5 });

        // Check structure: Ω[i,j] = 0.5^|i-j|
        assert!((omega[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((omega[[0, 1]] - 0.5).abs() < 1e-10);
        assert!((omega[[0, 2]] - 0.25).abs() < 1e-10);
        assert!((omega[[0, 3]] - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky() {
        let a = array![[4.0, 2.0], [2.0, 5.0]];
        let l = cholesky_lower(&a).unwrap();

        // Verify LL' = A
        let reconstructed = l.dot(&l.t());
        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-10);
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════════
    // VALIDATION TESTS - Comparing against R reference implementations
    // ════════════════════════════════════════════════════════════════════════════

    /// Case `gls_ar1` (nlme::gls REML on the legacy Test 2 DGP; validation/r/cases/regression_legacy.R):
    /// coefficients, SEs, t, p and sigma at nlme's phi.
    #[test]
    fn test_validate_gls_ar1_nlme() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("gls_ar1");
        let n = case.scalar("n_obs") as usize;
        let phi = case.scalar("phi");
        let names = ["(Intercept)", "x"];
        let g = run_gls(&ds, "y", &["x"], true, "ar1", Some(phi)).unwrap();
        Check::new(&case)
            .named("coef", &names, &g.coefficients, Tol::coef(n))
            .named("se", &names, &g.std_errors, Tol::se(n))
            .named("t", &names, &g.t_values, Tol::stat(n))
            .named("p", &names, &g.p_values, Tol::p_value(n))
            .scalar("sigma", g.sigma, Tol::stat(n))
            .scalar("df_resid", g.df_residual as f64, Tol::exact())
            .scalar("n_obs", g.n_obs as f64, Tol::exact())
            .finish();
        // REG-9 (moved into T3-WP11): at nlme's phi the log-likelihood is nlme's REML logLik
        // (fixed phi: 3 parameters, so AIC/BIC differ from nlme's estimated-phi values by 2 and ln(48)).
        Check::new(&case)
            .scalar("log_lik_reml", g.log_likelihood, Tol::stat(n))
            .finish();
        assert_eq!(g.n_params, 3);
        assert!(g.correlation.contains("AR(1)") && g.correlation_param == Some(phi));
        // REG-43 (moved in): "ar1_auto" estimates phi by REML as nlme (it was the lag-1 residual
        // correlation, 0.6542)
        let auto = run_gls(&ds, "y", &["x"], true, "ar1_auto", None).unwrap();
        Check::new(&case)
            .scalar("phi", auto.correlation_param.unwrap(), Tol::iterative(1e-6))
            .named("coef", &names, &auto.coefficients, Tol::coef(n))
            .named("se", &names, &auto.std_errors, Tol::se(n))
            .scalar("log_lik_reml", auto.log_likelihood, Tol::stat(n))
            .scalar("aic_reml", auto.aic, Tol::stat(n))
            .scalar("bic_reml", auto.bic, Tol::stat(n))
            .finish();
        let ml = GlsOptions {
            correlation: GlsCorrelation::Ar1,
            method: GlsMethod::Ml,
            ..GlsOptions::default()
        };
        let gml = run_gls_with(&ds, "y", &["x"], true, &ml).unwrap();
        Check::new(&case)
            .scalar(
                "phi_ml",
                gml.correlation_param.unwrap(),
                Tol::iterative(1e-6),
            )
            .named("coef_ml", &names, &gml.coefficients, Tol::coef(n))
            .scalar("log_lik_ml", gml.log_likelihood, Tol::stat(n))
            .finish();
        assert!(auto.rho_estimated && auto.n_params == 4 && !g.rho_estimated && g.n_params == 3);
    }

    #[test]
    fn test_reg32_gls_errors_name_the_column() {
        let ds = Dataset::new(
            polars::prelude::df! {
                "y" => [1.1, 1.9, 3.2, 3.8, 5.1, 6.3],
                "x1" => [1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0],
            }
            .unwrap(),
        );
        match run_gls(&ds, "y", &["x1"], true, "identity", None) {
            Err(EconError::NullValues { column, count }) => {
                assert_eq!(column, "x1");
                assert_eq!(count, 1);
            }
            other => panic!("expected NullValues for x1, got {other:?}"),
        }
        match run_gls(&ds, "y", &["x9"], true, "identity", None) {
            Err(EconError::ColumnNotFound { column, available }) => {
                assert_eq!(column, "x9");
                assert_eq!(available, ["y", "x1"]);
            }
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
    }

    fn cs_options(method: GlsMethod, rho: Option<f64>) -> GlsOptions {
        GlsOptions {
            correlation: GlsCorrelation::CompoundSymmetry,
            group_col: Some("g".into()),
            method,
            rho,
        }
    }
    const NAMES: [&str; 3] = ["(Intercept)", "x", "z"];

    /// The data of case `gls_compsymm` (for tests that do not compare its values).
    fn gls_cs_data() -> Dataset {
        crate::regression::ols::na_fixtures::repo_dataset(
            "validation/datasets/generated/gls_cs.csv",
        )
    }

    /// `y` and `[1, x, z]` of the `gls_compsymm` data.
    fn gls_cs_arrays(ds: &Dataset) -> (Array1<f64>, Array2<f64>) {
        use crate::data::columns::numeric_column;
        let df = ds.df();
        let y = numeric_column(df, "y", NaPolicy::Fail).unwrap().0;
        let x = numeric_column(df, "x", NaPolicy::Fail).unwrap().0;
        let z = numeric_column(df, "z", NaPolicy::Fail).unwrap().0;
        let n = y.len();
        let design = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => 1.0,
            1 => x[i],
            _ => z[i],
        });
        (Array1::from_vec(y), design)
    }

    fn groups_of(ds: &Dataset) -> Vec<u32> {
        let cols = ColumnRequest::new(ds.df()).factor("g").extract().unwrap();
        cols.factor("g").codes.clone()
    }

    /// Case `gls_compsymm`: nlme::gls(y ~ x + z, correlation = corCompSymm(form = ~ 1 | g)) with
    /// rho estimated. Pins: nlme's profile likelihood maximised by optimize(tol = 1e-12)
    /// (`<m>_profile_*`, the tight refit); cross-check: nlme's own fit (`<m>_*`), whose optimiser
    /// stops up to 3e-8 relative short in rho.
    fn check_compsymm(case_id: &str, method: GlsMethod, m: &str) {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case(case_id);
        let n = case.scalar("n") as usize;
        let g = run_gls_with(&ds, "y", &["x", "z"], true, &cs_options(method, None)).unwrap();
        let rho = g.correlation_param.unwrap();
        let k = |s: &str| format!("{m}_{s}");
        let p = |s: &str| format!("{m}_profile_{s}");
        Check::new(&case)
            // flat maximum: Brent resolves rho to about 1.5e-8 |rho|
            .scalar(&p("rho"), rho, Tol::iterative(1e-6))
            .named(&p("coef"), &NAMES, &g.coefficients, Tol::coef(n))
            .named(&p("se"), &NAMES, &g.std_errors, Tol::se(n))
            .scalar(&p("sigma"), g.sigma, Tol::stat(n))
            .scalar(&p("loglik"), g.log_likelihood, Tol::stat(n))
            .scalar(&k("rho"), rho, Tol::iterative(1e-6))
            .named(&k("coef"), &NAMES, &g.coefficients, Tol::iterative(1e-6))
            .named(&k("se"), &NAMES, &g.std_errors, Tol::iterative(1e-6))
            .scalar(&k("sigma"), g.sigma, Tol::iterative(1e-6))
            .scalar(&k("loglik"), g.log_likelihood, Tol::stat(n))
            .scalar(&k("aic"), g.aic, Tol::stat(n))
            .scalar(&k("bic"), g.bic, Tol::stat(n))
            .scalar("n_groups", g.n_groups.unwrap() as f64, Tol::exact())
            .finish();
        assert!(
            g.rho_estimated && g.method == method && g.n_params == 5,
            "{g:?}"
        );
        assert!(
            g.convergence.as_ref().is_some_and(|c| c.converged),
            "{:?}",
            g.convergence
        );
    }

    #[test]
    fn test_validate_gls_compsymm_reml() {
        check_compsymm("gls_compsymm", GlsMethod::Reml, "reml")
    }

    #[test]
    fn test_validate_gls_compsymm_ml() {
        check_compsymm("gls_compsymm", GlsMethod::Ml, "ml")
    }

    /// Case `gls_compsymm`: corCompSymm(0.3, form = ~ 1 | g, fixed = TRUE), REML and ML (no
    /// optimisation).
    #[test]
    fn test_validate_gls_compsymm_fixed_rho_grouped() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("gls_compsymm");
        let n = case.scalar("n") as usize;
        for (method, m) in [(GlsMethod::Reml, "reml"), (GlsMethod::Ml, "ml")] {
            let g =
                run_gls_with(&ds, "y", &["x", "z"], true, &cs_options(method, Some(0.3))).unwrap();
            let k = |s: &str| format!("cs03_{m}_{s}");
            Check::new(&case)
                .named(&k("coef"), &NAMES, &g.coefficients, Tol::coef(n))
                .named(&k("se"), &NAMES, &g.std_errors, Tol::se(n))
                .scalar(&k("sigma"), g.sigma, Tol::stat(n))
                .scalar(&k("loglik"), g.log_likelihood, Tol::stat(n))
                .scalar(&k("aic"), g.aic, Tol::stat(n))
                .scalar(&k("bic"), g.bic, Tol::stat(n))
                .finish();
            assert!(!g.rho_estimated && g.n_params == 4 && g.convergence.is_none());
        }
    }

    /// Case `gls_compsymm`: corAR1(form = ~ 1 | g) by REML. nlme stops 2.2e-6 relative short in
    /// rho here (its logLik is 4.3e-11 below the profile maximum), so its keys are a looser
    /// cross-check.
    #[test]
    fn test_validate_gls_ar1_grouped_reml() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("gls_compsymm");
        let n = case.scalar("n") as usize;
        let o = GlsOptions {
            correlation: GlsCorrelation::Ar1,
            group_col: Some("g".into()),
            ..GlsOptions::default()
        };
        let g = run_gls_with(&ds, "y", &["x", "z"], true, &o).unwrap();
        let rho = g.correlation_param.unwrap();
        Check::new(&case)
            .scalar("ar1g_profile_rho", rho, Tol::iterative(1e-6))
            .named("ar1g_profile_coef", &NAMES, &g.coefficients, Tol::coef(n))
            .named("ar1g_profile_se", &NAMES, &g.std_errors, Tol::se(n))
            .scalar("ar1g_profile_sigma", g.sigma, Tol::stat(n))
            .scalar("ar1g_profile_loglik", g.log_likelihood, Tol::stat(n))
            .scalar("ar1g_rho", rho, Tol::iterative(1e-5))
            .named("ar1g_coef", &NAMES, &g.coefficients, Tol::iterative(1e-6))
            .named("ar1g_se", &NAMES, &g.std_errors, Tol::iterative(1e-5))
            .scalar("ar1g_loglik", g.log_likelihood, Tol::stat(n))
            .finish();
    }

    /// One block with an intercept: the common within-block shock is the intercept; rho is not
    /// identified.
    #[test]
    fn test_gls_compsymm_rho_not_identified() {
        let ds = gls_cs_data();
        let mut o = cs_options(GlsMethod::Reml, None);
        o.group_col = None;
        match run_gls_with(&ds, "y", &["x", "z"], true, &o).unwrap_err() {
            EconError::InvalidSpecification { message } => assert!(
                message.contains("rho") && message.contains("not identified"),
                "{message}"
            ),
            e => panic!("{e:?}"),
        }
        // the old string API reaches the same check
        assert!(run_gls(&ds, "y", &["x", "z"], true, "cs", None).is_err());
        // groups with one row each give no within-group pairs
        let (y, x) = gls_cs_arrays(&ds);
        let spec = GlsSpec {
            correlation: GlsCorrelation::CompoundSymmetry,
            rho: None,
            groups: Some((0..y.len() as u32).collect()),
            method: GlsMethod::Reml,
        };
        assert!(matches!(
            gls_with(&y.view(), &x.view(), &spec),
            Err(EconError::InvalidSpecification { .. })
        ));
        // rho or groups with identity
        let spec = GlsSpec {
            correlation: GlsCorrelation::Identity,
            rho: Some(0.3),
            ..GlsSpec::default()
        };
        assert!(matches!(
            gls_with(&y.view(), &x.view(), &spec),
            Err(EconError::InvalidSpecification { .. })
        ));
    }

    /// An aliased column is an error, as nlme's "computed gls fit is singular".
    #[test]
    fn test_gls_rejects_rank_deficient_design() {
        let ds = gls_cs_data();
        let (y, x) = gls_cs_arrays(&ds);
        let n = y.len();
        let mut x2 = Array2::zeros((n, 4));
        x2.slice_mut(ndarray::s![.., ..3]).assign(&x);
        x2.column_mut(3).assign(&(&x.column(1) * 2.0));
        let spec = GlsSpec {
            correlation: GlsCorrelation::CompoundSymmetry,
            rho: Some(0.3),
            groups: Some(groups_of(&ds)),
            method: GlsMethod::Reml,
        };
        match gls_with(&y.view(), &x2.view(), &spec).unwrap_err() {
            EconError::SingularMatrix { context, .. } => assert!(
                context.contains("rank 3 of 4") && context.contains("column 4"),
                "{context}"
            ),
            e => panic!("{e:?}"),
        }
    }

    /// The profile maximisation is scale-free (y * 1e-6 and y * 1e6).
    #[test]
    fn test_gls_rho_estimate_is_scale_invariant() {
        let ds = gls_cs_data();
        let (y, x) = gls_cs_arrays(&ds);
        let spec = GlsSpec {
            correlation: GlsCorrelation::CompoundSymmetry,
            rho: None,
            groups: Some(groups_of(&ds)),
            method: GlsMethod::Reml,
        };
        let base = gls_with(&y.view(), &x.view(), &spec).unwrap();
        let (rho0, it0) = (
            base.correlation_param.unwrap(),
            base.convergence.as_ref().unwrap().iterations as i64,
        );
        for s in [1e-6, 1e6] {
            let ys = &y * s;
            let g = gls_with(&ys.view(), &x.view(), &spec).unwrap();
            assert!(
                (g.correlation_param.unwrap() - rho0).abs() <= 1e-6 * rho0.abs(),
                "y*{s}: rho {:?} vs {rho0}",
                g.correlation_param
            );
            for j in 0..3 {
                assert!(
                    (g.coefficients[j] / s - base.coefficients[j]).abs()
                        <= 1e-6 * base.coefficients[j].abs()
                );
            }
            let it = g.convergence.as_ref().unwrap().iterations as i64;
            assert!((it - it0).abs() <= 5, "y*{s}: {it} vs {it0} evaluations");
        }
    }

    #[test]
    fn test_gls_options_parse_aliases_and_reject_unknown() {
        assert_eq!("REML".parse::<GlsMethod>().unwrap(), GlsMethod::Reml);
        assert_eq!(
            "maximum-likelihood".parse::<GlsMethod>().unwrap(),
            GlsMethod::Ml
        );
        for s in ["cs", "Compound Symmetry", "exchangeable", "corCompSymm"] {
            assert_eq!(
                s.parse::<GlsCorrelation>().unwrap(),
                GlsCorrelation::CompoundSymmetry,
                "{s}"
            );
        }
        assert_eq!(
            "AR1".parse::<GlsCorrelation>().unwrap(),
            GlsCorrelation::Ar1
        );
        match "gmm".parse::<GlsMethod>().unwrap_err() {
            EconError::InvalidSpecification { message } => assert!(
                message.contains("method") && message.contains("reml") && message.contains("ml"),
                "{message}"
            ),
            e => panic!("{e:?}"),
        }
        let ds = gls_cs_data();
        match run_gls(&ds, "y", &["x"], true, "toeplitz", None).unwrap_err() {
            EconError::InvalidSpecification { message } => assert!(
                message.contains("compound_symmetry") && message.contains("ar1"),
                "{message}"
            ),
            e => panic!("{e:?}"),
        }
    }

    /// The explicit `intercept` decides the R² rule: a constant non-zero regressor with
    /// `intercept = false` is not an intercept (summary.lm keys on the formula's intercept).
    #[test]
    fn test_gls_r_squared_follows_explicit_intercept() {
        let ds = Dataset::new(
            polars::prelude::df! {
                "y" => [1.1, 1.9, 3.2, 3.8, 5.1, 6.3, 6.8, 8.2],
                "c" => [2.0; 8],
                "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            }
            .unwrap(),
        );
        let r = run_gls(&ds, "y", &["c", "x"], false, "identity", None).unwrap();
        let y = [1.1, 1.9, 3.2, 3.8, 5.1, 6.3, 6.8, 8.2];
        let ss_res: f64 = r.residuals.iter().map(|e| e * e).sum();
        let uncentred = 1.0 - ss_res / y.iter().map(|v| v * v).sum::<f64>();
        assert!((r.r_squared - uncentred).abs() < 1e-12, "{}", r.r_squared);
        // the raw-matrix entry point still detects the constant column
        let x = Array2::from_shape_fn((8, 2), |(i, j)| if j == 0 { 2.0 } else { i as f64 + 1.0 });
        let g = gls(
            &Array1::from_vec(y.to_vec()).view(),
            &x.view(),
            CorrelationStructure::Identity,
        )
        .unwrap();
        assert!((g.r_squared - uncentred).abs() > 1e-6);
    }
}
