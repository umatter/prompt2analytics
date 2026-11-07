//! Gaussian mixture models fitted by EM ([`gaussian_mixture`], [`gaussian_mixture_with`]); R
//! analogue `mclust`.
//!
//! The four covariance types are mclust's models: `Full` is VVV, `Tied` is EEE (one pooled
//! covariance), `Diagonal` is VVI and `Spherical` is VII, with mclust's parameter counts. Every
//! component density goes through a Cholesky factor, so its log-determinant and Mahalanobis term
//! are exact in any dimension.
//!
//! Differences from mclust, recorded in `validation/ml/gmm.md`: the default start is a k-means
//! partition, not mclust's hierarchical one; the stopping rule is
//! `relative_change(l, l_prev) < tolerance` (mclust's is `|dl| / (1 + |l|)`); a singular
//! covariance gets a small relative ridge and a warning (mclust reports `NA`).
//!
//! ```
//! use ndarray::array;
//! use p2a_core::ml::advanced_clustering_mod::gmm::{GmmConfig, gaussian_mixture_with};
//! use p2a_core::ml::CovarianceType;
//! let x = array![
//!     [0.0, 0.1],
//!     [0.2, -0.1],
//!     [0.1, 0.3],
//!     [-0.2, 0.0],
//!     [5.0, 5.2],
//!     [5.3, 4.9],
//!     [4.8, 5.1],
//!     [5.1, 4.7]
//! ];
//! let mut cfg = GmmConfig::new(2);
//! cfg.covariance_type = CovarianceType::Spherical;
//! let fit = gaussian_mixture_with(x.view(), &cfg).unwrap();
//! assert_eq!(fit.model_name, "VII");
//! ```

use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::cholesky;
use crate::linalg::solve::SYM_RANK_TOL;
use crate::options::Choices;
use crate::resource::{checkpoint, try_zeros2};
use crate::traits::convergence::{Convergence, ResultNotes, notes_text, relative_change};
use ndarray::{Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

/// Seed of the k-means start when `seed` is `None`.
pub const GMM_DEFAULT_SEED: u64 = 42;
/// Default stopping rule: `relative_change(l, l_prev) < 1e-8`. mclust stops at
/// `|dl| / (1 + |l|) < 1e-5`, which leaves the parameters far from the optimum on moderate n; pass
/// 1e-5 to come close to it.
pub const GMM_DEFAULT_TOLERANCE: f64 = 1e-8;
/// Ridge added to a singular component covariance, as a multiple of the data's column variances.
pub const GMM_SINGULAR_RIDGE: f64 = 1e-10;

// =============================================================================
// Gaussian Mixture Model
// =============================================================================

/// Result of Gaussian Mixture Model clustering.
///
/// # References
///
/// - McLachlan, G.J. and Peel, D. (2000). "Finite Mixture Models". Wiley.
/// - R mclust package documentation
///   Source: <https://www.rdocumentation.org/packages/mclust/>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianMixtureResult {
    /// Hard cluster assignments (most likely cluster for each point)
    pub labels: Vec<usize>,
    /// Soft assignments (n x k matrix of responsibilities)
    pub responsibilities: Vec<Vec<f64>>,
    /// Mixture weights (k weights summing to 1)
    pub weights: Vec<f64>,
    /// Component means (k x d matrix)
    pub means: Vec<Vec<f64>>,
    /// Component covariances (k matrices, each d x d)
    pub covariances: Vec<Vec<Vec<f64>>>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// BIC (`-2 logLik + df ln n`, lower is better; mclust reports the negative of this)
    pub bic: f64,
    /// AIC (`-2 logLik + 2 df`, lower is better)
    pub aic: f64,
    /// Number of E-steps run
    pub n_iterations: usize,
    /// Whether the stopping rule was met
    pub converged: bool,
    /// Number of components
    pub n_components: usize,
    /// Number of observations
    pub n: usize,
    /// Number of features
    pub n_features: usize,
    /// mclust's name of the fitted model: "VVV", "EEE", "VVI" or "VII"
    #[serde(default)]
    pub model_name: String,
    /// Number of free parameters (mclust's `df`)
    #[serde(default)]
    pub n_parameters: usize,
    /// What the fit had to do or could not do (a ridge, no convergence)
    #[serde(default)]
    pub warnings: Vec<String>,
    /// The stopping rule's final state
    #[serde(default)]
    pub convergence: Convergence,
}

impl ResultNotes for GaussianMixtureResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        Some(self.convergence.clone())
    }
}

impl std::fmt::Display for GaussianMixtureResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Gaussian Mixture Model Results")?;
        writeln!(f, "==============================")?;
        writeln!(f, "Number of components: {}", self.n_components)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Number of features: {}", self.n_features)?;
        let family = match self.model_name.as_str() {
            "VVV" => "full",
            "EEE" => "tied",
            "VVI" => "diagonal",
            "VII" => "spherical",
            _ => "unknown",
        };
        writeln!(f, "Model: {} ({family} covariances)", self.model_name)?;
        writeln!(f, "Parameters: {}", self.n_parameters)?;
        writeln!(f)?;
        writeln!(f, "Log-likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "BIC: {:.4}", self.bic)?;
        writeln!(f, "AIC: {:.4}", self.aic)?;
        writeln!(f)?;
        writeln!(f, "Component weights:")?;
        for (i, w) in self.weights.iter().enumerate() {
            writeln!(f, "  Component {}: {:.4}", i, w)?;
        }

        let mut cluster_counts = vec![0usize; self.n_components];
        for &label in &self.labels {
            if let Some(c) = cluster_counts.get_mut(label) {
                *c += 1;
            }
        }
        writeln!(f)?;
        writeln!(f, "Cluster sizes (hard assignment):")?;
        for (i, count) in cluster_counts.iter().enumerate() {
            writeln!(f, "  Cluster {}: {} points", i, count)?;
        }
        write!(f, "{}", notes_text(self))
    }
}

/// Covariance type for Gaussian mixture models, named as in mclust in parentheses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CovarianceType {
    /// Full covariance matrix for each component (VVV)
    Full,
    /// Diagonal covariance per component (VVI)
    Diagonal,
    /// Spherical: one variance per component (VII)
    Spherical,
    /// Tied: one pooled covariance for all components (EEE)
    Tied,
}

crate::option_enum!(
    CovarianceType,
    Choices {
        option: "covariance type",
        values: &[
            (&["full", "VVV"], CovarianceType::Full),
            (&["diagonal", "diag", "VVI"], CovarianceType::Diagonal),
            (&["spherical", "VII"], CovarianceType::Spherical),
            (&["tied", "EEE"], CovarianceType::Tied),
        ],
    }
);

impl CovarianceType {
    /// mclust's model name: "VVV", "VVI", "VII" or "EEE".
    pub fn mclust_name(self) -> &'static str {
        match self {
            CovarianceType::Full => "VVV",
            CovarianceType::Diagonal => "VVI",
            CovarianceType::Spherical => "VII",
            CovarianceType::Tied => "EEE",
        }
    }

    /// `None` when the count does not fit in a `usize`.
    fn checked_parameters(self, d: usize, g: usize) -> Option<usize> {
        let triangle = d.checked_mul(d.checked_add(1)?)? / 2;
        let means = g.checked_mul(d)?;
        let covariances = match self {
            CovarianceType::Full => g.checked_mul(triangle)?,
            CovarianceType::Tied => triangle,
            CovarianceType::Diagonal => g.checked_mul(d)?,
            CovarianceType::Spherical => g,
        };
        means
            .checked_add(covariances)?
            .checked_add(g.checked_sub(1)?)
    }

    /// mclust's `df` for `g >= 1` components in `d` dimensions: means, covariances and the
    /// `g - 1` free proportions. Saturates at `usize::MAX`, which no fit can reach
    /// ([`gaussian_mixture_with`] refuses a count that does not fit).
    pub fn n_parameters(self, d: usize, g: usize) -> usize {
        self.checked_parameters(d, g).unwrap_or(usize::MAX)
    }
}

/// Where the EM iterations start.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum GmmInit {
    /// A k-means partition (`crate::ml::kmeans`, 20 iterations, 5 starts) of the data divided by
    /// the root mean column variance, then one M-step on it (mclust starts with an M-step from a
    /// classification).
    #[default]
    KMeans,
    /// A hard partition, labels in `0..n_components`, each component non-empty: one M-step first.
    Classification(Vec<usize>),
    /// Proportions, means (`n_components` x d) and covariances (`n_components` x d x d): the first
    /// E-step uses them (mclust `estep(parameters)`). Covariances must be symmetric positive
    /// definite and obey the covariance type (equal for Tied, diagonal for Diagonal, scalar for
    /// Spherical).
    Parameters {
        /// Mixing proportions, positive, summing to 1
        weights: Vec<f64>,
        /// Component means
        means: Vec<Vec<f64>>,
        /// Component covariance matrices
        covariances: Vec<Vec<Vec<f64>>>,
    },
}

/// Gaussian mixture configuration ([`gaussian_mixture`] keeps its positional form).
#[derive(Debug, Clone, PartialEq)]
pub struct GmmConfig {
    /// Number of mixture components.
    pub n_components: usize,
    /// Covariance model (default `Full`).
    pub covariance_type: CovarianceType,
    /// Start (default a k-means partition).
    pub init: GmmInit,
    /// Maximum number of E-steps (default 100).
    pub max_iterations: usize,
    /// Stopping rule on the relative change of the log-likelihood (default
    /// [`GMM_DEFAULT_TOLERANCE`]).
    pub tolerance: f64,
    /// k-means seed; `None` means [`GMM_DEFAULT_SEED`].
    pub seed: Option<u64>,
}

impl GmmConfig {
    /// The defaults for `n_components` components.
    pub fn new(n_components: usize) -> Self {
        GmmConfig {
            n_components,
            covariance_type: CovarianceType::Full,
            init: GmmInit::default(),
            max_iterations: 100,
            tolerance: GMM_DEFAULT_TOLERANCE,
            seed: None,
        }
    }
}

fn spec(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

/// One Gaussian component with its Cholesky factor, shared with `mixtures.rs`.
#[derive(Debug, Clone)]
pub(crate) struct Component {
    pub(crate) mean: Vec<f64>,
    pub(crate) cov: Array2<f64>,
    /// Lower factor, row-major (d x d).
    chol: Vec<f64>,
    /// `d ln(2 pi) + ln det(cov)`.
    log_const: f64,
}

impl Component {
    /// `None` when `cov` is not positive definite: the Cholesky factorisation fails, or a pivot is
    /// negligible against its own diagonal entry (`L_ii^2 <= SYM_RANK_TOL * d * cov_ii`, which does
    /// not depend on the units of the columns).
    pub(crate) fn new(mean: Vec<f64>, cov: Array2<f64>) -> Option<Component> {
        let d = cov.nrows();
        let l = cholesky(&cov.view()).ok()?;
        let mut log_det = 0.0;
        for i in 0..d {
            let (pivot, diag) = (l[[i, i]], cov[[i, i]]);
            if !pivot.is_finite() || pivot <= 0.0 {
                return None;
            }
            if pivot * pivot <= SYM_RANK_TOL * d as f64 * diag {
                return None;
            }
            log_det += 2.0 * pivot.ln();
        }
        let mut chol = Vec::with_capacity(d * d);
        for row in l.rows() {
            chol.extend(row.iter().copied());
        }
        let log_const = d as f64 * (2.0 * std::f64::consts::PI).ln() + log_det;
        Some(Component {
            mean,
            cov,
            chol,
            log_const,
        })
    }

    /// The normal log-density at `x`: `-0.5 (d ln 2 pi + ln det + |L^-1 (x - mu)|^2)`. `u` is a
    /// d-vector of scratch space.
    #[inline]
    fn log_pdf_with(&self, x: ArrayView1<'_, f64>, u: &mut [f64]) -> f64 {
        let d = self.mean.len();
        let mut mahal = 0.0;
        for (r, (xr, mr)) in x.iter().zip(&self.mean).enumerate() {
            let row = &self.chol[r * d..r * d + r + 1];
            let mut s = xr - mr;
            for (l, v) in row[..r].iter().zip(u.iter()) {
                s -= l * v;
            }
            let v = s / row[r];
            u[r] = v;
            mahal += v * v;
        }
        -0.5 * (self.log_const + mahal)
    }

    /// The normal log-density at `x`.
    #[cfg(test)]
    pub(crate) fn log_pdf(&self, x: ArrayView1<'_, f64>) -> f64 {
        let mut u = vec![0.0; self.mean.len()];
        self.log_pdf_with(x, &mut u)
    }
}

/// Fills `resp` (n x k) with the posteriors (log-sum-exp) and returns the log-likelihood.
pub(crate) fn e_step(
    data: ArrayView2<'_, f64>,
    weights: &[f64],
    comps: &[Component],
    resp: &mut Array2<f64>,
) -> f64 {
    let d = data.ncols();
    let k = comps.len();
    let log_weights: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
    let mut u = vec![0.0; d];
    let mut lp = vec![0.0; k];
    let mut ll = 0.0;
    for (row, mut post) in data.rows().into_iter().zip(resp.rows_mut()) {
        let mut top = f64::NEG_INFINITY;
        for j in 0..k {
            let v = log_weights[j] + comps[j].log_pdf_with(row, &mut u);
            lp[j] = v;
            top = top.max(v);
        }
        let sum_exp: f64 = lp.iter().map(|v| (v - top).exp()).sum();
        let lse = top + sum_exp.ln();
        ll += lse;
        for (p, v) in post.iter_mut().zip(&lp) {
            *p = (v - lse).exp();
        }
    }
    ll
}

/// Proportions, means and covariances of `cov_type` from the posteriors: `W_k / n_k` (Full),
/// `sum_k W_k / n` (Tied), `diag(W_k) / n_k` (Diagonal), `tr(W_k) / (d n_k) I` (Spherical), where
/// `W_k = sum_i resp_ik (x_i - mu_k)(x_i - mu_k)'`. No ridge and no floor. `InsufficientData` when a
/// component has no posterior mass.
pub(crate) fn m_step(
    data: ArrayView2<'_, f64>,
    resp: ArrayView2<'_, f64>,
    cov_type: CovarianceType,
) -> EconResult<(Vec<f64>, Vec<Vec<f64>>, Vec<Array2<f64>>)> {
    let (n, d) = data.dim();
    let k = resp.ncols();
    let mut weights = Vec::with_capacity(k);
    let mut means: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut covs: Vec<Array2<f64>> = Vec::with_capacity(k);
    let mut pooled = try_zeros2(d, d, "Gaussian mixture covariance")?;
    let mut diff = vec![0.0; d];
    for j in 0..k {
        let post = resp.column(j);
        let nk: f64 = post.sum();
        if nk <= n as f64 * f64::EPSILON {
            return Err(EconError::InsufficientData {
                required: 1,
                provided: 0,
                context: format!(
                    "Gaussian mixture: component {} has no posterior mass; reduce n_components",
                    j + 1
                ),
            });
        }
        let mut mean = vec![0.0; d];
        for (row, &r) in data.rows().into_iter().zip(post.iter()) {
            for (m, x) in mean.iter_mut().zip(row.iter()) {
                *m += r * x;
            }
        }
        for m in &mut mean {
            *m /= nk;
        }
        // Upper triangle of W_k (the diagonal only where the model needs no more).
        let mut w = try_zeros2(d, d, "Gaussian mixture covariance")?;
        let full = matches!(cov_type, CovarianceType::Full | CovarianceType::Tied);
        for (row, &r) in data.rows().into_iter().zip(post.iter()) {
            for ((v, x), m) in diff.iter_mut().zip(row.iter()).zip(&mean) {
                *v = x - m;
            }
            if full {
                for a in 0..d {
                    let ra = r * diff[a];
                    for b in a..d {
                        w[[a, b]] += ra * diff[b];
                    }
                }
            } else {
                for a in 0..d {
                    w[[a, a]] += r * diff[a] * diff[a];
                }
            }
        }
        match cov_type {
            CovarianceType::Full | CovarianceType::Diagonal => {
                for a in 0..d {
                    for b in a..d {
                        let v = w[[a, b]] / nk;
                        w[[a, b]] = v;
                        w[[b, a]] = v;
                    }
                }
            }
            CovarianceType::Tied => {
                for a in 0..d {
                    for b in a..d {
                        pooled[[a, b]] += w[[a, b]];
                    }
                }
            }
            CovarianceType::Spherical => {
                let var = w.diag().sum() / (d as f64 * nk);
                w.fill(0.0);
                w.diag_mut().fill(var);
            }
        }
        weights.push(nk / n as f64);
        means.push(mean);
        covs.push(w);
    }
    if cov_type == CovarianceType::Tied {
        for a in 0..d {
            for b in a..d {
                let v = pooled[[a, b]] / n as f64;
                pooled[[a, b]] = v;
                pooled[[b, a]] = v;
            }
        }
        for c in &mut covs {
            c.assign(&pooled);
        }
    }
    Ok((weights, means, covs))
}

/// What the fit knows about the data's scale, for the relative ridge.
struct RidgeScale {
    column_variances: Vec<f64>,
    mean_variance: f64,
}

/// Population variance of each column and their mean.
fn ridge_scale(data: ArrayView2<'_, f64>) -> EconResult<RidgeScale> {
    let (n, d) = data.dim();
    let column_variances: Vec<f64> = data
        .columns()
        .into_iter()
        .map(|c| {
            let mean = c.sum() / n as f64;
            c.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64
        })
        .collect();
    if column_variances.iter().any(|v| !v.is_finite()) {
        return Err(spec(
            "Gaussian mixture: the data are too large in magnitude for their variance to be \
             finite; rescale them"
                .into(),
        ));
    }
    let mean_variance = column_variances.iter().sum::<f64>() / d as f64;
    Ok(RidgeScale {
        column_variances,
        mean_variance,
    })
}

/// Factors the M-step's covariances, adding the relative ridge to a singular one (with one warning
/// per component).
fn factor_components(
    means: Vec<Vec<f64>>,
    covs: Vec<Array2<f64>>,
    cov_type: CovarianceType,
    scale: &RidgeScale,
    warned: &mut [bool],
    warnings: &mut Vec<String>,
) -> EconResult<Vec<Component>> {
    let mut comps: Vec<Component> = Vec::with_capacity(covs.len());
    for (j, (mean, cov)) in means.into_iter().zip(covs).enumerate() {
        if cov_type == CovarianceType::Tied {
            if let Some(first) = comps.first() {
                comps.push(Component {
                    mean,
                    ..first.clone()
                });
                continue;
            }
        }
        if let Some(c) = Component::new(mean.clone(), cov.clone()) {
            comps.push(c);
            continue;
        }
        let what = if cov_type == CovarianceType::Tied {
            "the pooled covariance".to_string()
        } else {
            format!("component {} covariance", j + 1)
        };
        let mut ridged = cov;
        for (a, var) in scale.column_variances.iter().enumerate() {
            // A constant column has no variance of its own to be relative to.
            let base = if *var > 0.0 {
                *var
            } else {
                scale.mean_variance
            };
            ridged[[a, a]] += GMM_SINGULAR_RIDGE * base;
        }
        let Some(c) = Component::new(mean, ridged) else {
            return Err(EconError::SingularMatrix {
                context: format!(
                    "Gaussian mixture: {what} is rank deficient after a relative ridge"
                ),
                suggestion: "reduce n_components or remove duplicated points".into(),
            });
        };
        if !warned[j] {
            warned[j] = true;
            warnings.push(format!(
                "{what} is singular (rank deficient); a ridge of 1e-10 x the variance of each \
                 data column was added"
            ));
        }
        comps.push(c);
    }
    Ok(comps)
}

fn validate_parameters_start(
    k: usize,
    d: usize,
    cov_type: CovarianceType,
    weights: &[f64],
    means: &[Vec<f64>],
    covariances: &[Vec<Vec<f64>>],
) -> EconResult<()> {
    if weights.len() != k || means.len() != k || covariances.len() != k {
        return Err(spec(format!(
            "start parameters must have {k} weights, means and covariances, got {}, {} and {}",
            weights.len(),
            means.len(),
            covariances.len()
        )));
    }
    if weights.iter().any(|w| !w.is_finite() || *w <= 0.0) {
        return Err(spec("start weights must be finite and positive".into()));
    }
    let total: f64 = weights.iter().sum();
    if (total - 1.0).abs() > 1e-8 {
        return Err(spec(format!("start weights must sum to 1, got {total}")));
    }
    for (j, (m, c)) in means.iter().zip(covariances).enumerate() {
        if m.len() != d || m.iter().any(|v| !v.is_finite()) {
            return Err(spec(format!("start mean {j} must have {d} finite values")));
        }
        if c.len() != d
            || c.iter()
                .any(|r| r.len() != d || r.iter().any(|v| !v.is_finite()))
        {
            return Err(spec(format!(
                "start covariance {j} must be a finite {d} x {d} matrix"
            )));
        }
        let big = c.iter().flatten().fold(1.0f64, |a, v| a.max(v.abs()));
        let sym_tol = 1e-10 * big;
        for a in 0..d {
            for b in 0..a {
                if (c[a][b] - c[b][a]).abs() > sym_tol {
                    return Err(spec(format!("start covariance {j} is not symmetric")));
                }
            }
        }
        let obeys = match cov_type {
            CovarianceType::Full => true,
            CovarianceType::Tied => c
                .iter()
                .flatten()
                .zip(covariances[0].iter().flatten())
                .all(|(x, y)| (x - y).abs() <= sym_tol),
            CovarianceType::Diagonal | CovarianceType::Spherical => {
                let scalar = cov_type == CovarianceType::Diagonal
                    || (0..d).all(|a| (c[a][a] - c[0][0]).abs() <= sym_tol);
                scalar && (0..d).all(|a| (0..d).all(|b| a == b || c[a][b].abs() <= sym_tol))
            }
        };
        if !obeys {
            return Err(spec(format!(
                "start covariance {j} does not obey the {} model",
                cov_type.mclust_name()
            )));
        }
    }
    Ok(())
}

/// One-hot posteriors of a hard partition, checked to leave no component empty.
fn one_hot(labels: &[usize], n: usize, k: usize) -> EconResult<Array2<f64>> {
    if labels.len() != n {
        return Err(spec(format!(
            "start classification has length {}, the data have {n} rows",
            labels.len()
        )));
    }
    let mut z = try_zeros2(n, k, "Gaussian mixture responsibilities")?;
    let mut count = vec![0usize; k];
    for (i, &l) in labels.iter().enumerate() {
        if l >= k {
            return Err(spec(format!(
                "start classification label {l} (row {i}) must be below {k}"
            )));
        }
        z[[i, l]] = 1.0;
        count[l] += 1;
    }
    if let Some(j) = count.iter().position(|c| *c == 0) {
        return Err(spec(format!(
            "start classification leaves component {j} empty"
        )));
    }
    Ok(z)
}

/// Gaussian mixture by EM, as mclust's `me()` for VVV / EEE / VVI / VII.
///
/// Each iteration is an E-step followed by an M-step; the fit stops when
/// `relative_change(l, l_prev) < tolerance`. At the iteration limit the last M-step is skipped, so
/// the returned parameters, responsibilities and log-likelihood always belong together. The data
/// divided by a power of two give the same partition, so the stopping rule and the ridge do not
/// depend on the data's units; see `validation/ml/gmm.md`.
pub fn gaussian_mixture_with(
    data: ArrayView2<'_, f64>,
    config: &GmmConfig,
) -> EconResult<GaussianMixtureResult> {
    let (n, d) = data.dim();
    let k = config.n_components;
    let cov_type = config.covariance_type;
    if n == 0 || d == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: n.min(d),
            context: "Gaussian mixture: the data have no rows or no columns".into(),
        });
    }
    crate::ml::extract::require_finite(&data, "Gaussian mixture").map_err(spec)?;
    if k == 0 || k > n {
        return Err(spec(format!(
            "n_components must be between 1 and {n} (the number of rows), got {k}"
        )));
    }
    if config.max_iterations == 0 {
        return Err(spec("max_iterations must be at least 1, got 0".into()));
    }
    if !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err(spec(format!(
            "tolerance must be finite and positive, got {}",
            config.tolerance
        )));
    }
    let Some(n_parameters) = cov_type.checked_parameters(d, k) else {
        return Err(EconError::ResourceLimit(format!(
            "Gaussian mixture: the parameter count of {k} components in {d} dimensions overflows"
        )));
    };

    let scale = ridge_scale(data)?;
    let mut warnings = Vec::new();
    let mut warned = vec![false; k];
    let mut resp = try_zeros2(n, k, "Gaussian mixture responsibilities")?;

    let (mut weights, mut comps) = match &config.init {
        GmmInit::Parameters {
            weights,
            means,
            covariances,
        } => {
            validate_parameters_start(k, d, cov_type, weights, means, covariances)?;
            let total: f64 = weights.iter().sum();
            let mut comps = Vec::with_capacity(k);
            for (j, (m, c)) in means.iter().zip(covariances).enumerate() {
                let mut cov = try_zeros2(d, d, "Gaussian mixture covariance")?;
                for (a, row) in c.iter().enumerate() {
                    for (b, v) in row.iter().enumerate() {
                        cov[[a, b]] = *v;
                    }
                }
                let Some(comp) = Component::new(m.clone(), cov) else {
                    return Err(spec(format!(
                        "start covariance {j} is not positive definite"
                    )));
                };
                comps.push(comp);
            }
            (weights.iter().map(|w| w / total).collect(), comps)
        }
        init => {
            let z = match init {
                GmmInit::Classification(labels) => one_hot(labels, n, k)?,
                _ => {
                    if scale.mean_variance <= 0.0 {
                        return Err(spec("Gaussian mixture: data have zero variance".into()));
                    }
                    let s = scale.mean_variance.sqrt();
                    let scaled = data.mapv(|v| v / s);
                    let km = match crate::ml::kmeans(
                        scaled.view(),
                        k,
                        Some(20),
                        Some(1e-4),
                        Some(5),
                        Some(config.seed.unwrap_or(GMM_DEFAULT_SEED)),
                    ) {
                        Ok(km) => km,
                        Err(e) => {
                            // k-means reports a cancelled budget as text; keep it a cancellation.
                            checkpoint()?;
                            return Err(EconError::Computation(format!(
                                "Gaussian mixture: the k-means start failed: {e}"
                            )));
                        }
                    };
                    let mut z = try_zeros2(n, k, "Gaussian mixture responsibilities")?;
                    for (i, &l) in km.labels.iter().enumerate() {
                        z[[i, l]] = 1.0;
                    }
                    z
                }
            };
            let (w, means, covs) = m_step(data, z.view(), cov_type)?;
            let comps =
                factor_components(means, covs, cov_type, &scale, &mut warned, &mut warnings)?;
            (w, comps)
        }
    };

    let mut ll = f64::NEG_INFINITY;
    let mut last_change = f64::INFINITY;
    let mut iterations = 0;
    let mut converged = false;
    for iter in 1..=config.max_iterations {
        checkpoint()?;
        iterations = iter;
        let ll_prev = ll;
        ll = e_step(data, &weights, &comps, &mut resp);
        if !ll.is_finite() {
            return Err(EconError::Computation(format!(
                "Gaussian mixture: the log-likelihood is not finite at iteration {iter}"
            )));
        }
        if iter > 1 {
            last_change = relative_change(ll, ll_prev);
            if last_change < config.tolerance {
                converged = true;
                break;
            }
        }
        if iter == config.max_iterations {
            break;
        }
        let (w, means, covs) = m_step(data, resp.view(), cov_type)?;
        comps = factor_components(means, covs, cov_type, &scale, &mut warned, &mut warnings)?;
        weights = w;
    }

    let convergence = Convergence::new(
        converged,
        iterations,
        config.max_iterations,
        "relative change in log-likelihood",
        last_change,
        config.tolerance,
    );
    if let Some(note) = convergence.note("Gaussian mixture EM") {
        warnings.push(note);
    }

    let df = n_parameters as f64;
    let labels: Vec<usize> = resp
        .rows()
        .into_iter()
        .map(|row| {
            let mut best = 0;
            for (j, v) in row.iter().enumerate() {
                if *v > row[best] {
                    best = j;
                }
            }
            best
        })
        .collect();

    Ok(GaussianMixtureResult {
        labels,
        responsibilities: resp.rows().into_iter().map(|r| r.to_vec()).collect(),
        weights,
        means: comps.iter().map(|c| c.mean.clone()).collect(),
        covariances: comps
            .iter()
            .map(|c| c.cov.rows().into_iter().map(|r| r.to_vec()).collect())
            .collect(),
        log_likelihood: ll,
        bic: -2.0 * ll + df * (n as f64).ln(),
        aic: -2.0 * ll + 2.0 * df,
        n_iterations: iterations,
        converged,
        n_components: k,
        n,
        n_features: d,
        model_name: cov_type.mclust_name().to_string(),
        n_parameters,
        warnings,
        convergence,
    })
}

/// Run Gaussian Mixture Model clustering using the EM algorithm ([`gaussian_mixture_with`] with the
/// positional arguments).
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_components` - Number of mixture components
/// * `covariance_type` - Type of covariance parameters (default: Full)
/// * `max_iterations` - Maximum EM iterations (default: 100)
/// * `tolerance` - Stopping rule on the relative change of the log-likelihood (default:
///   [`GMM_DEFAULT_TOLERANCE`])
/// * `seed` - Seed of the k-means start (default: [`GMM_DEFAULT_SEED`])
///
/// # Returns
/// * `GaussianMixtureResult` containing cluster assignments and model parameters
///
/// # References
///
/// - McLachlan, G.J. and Peel, D. (2000). "Finite Mixture Models".
pub fn gaussian_mixture(
    data: ArrayView2<f64>,
    n_components: usize,
    covariance_type: Option<CovarianceType>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    seed: Option<u64>,
) -> Result<GaussianMixtureResult, String> {
    let mut config = GmmConfig::new(n_components);
    if let Some(c) = covariance_type {
        config.covariance_type = c;
    }
    if let Some(m) = max_iterations {
        config.max_iterations = m;
    }
    if let Some(t) = tolerance {
        config.tolerance = t;
    }
    config.seed = seed;
    gaussian_mixture_with(data, &config).map_err(|e| e.to_string())
}

/// Convenience wrapper for gaussian_mixture.
pub fn run_gaussian_mixture(
    data: ArrayView2<f64>,
    n_components: usize,
    seed: Option<u64>,
) -> Result<GaussianMixtureResult, String> {
    gaussian_mixture(data, n_components, None, None, None, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    use crate::data::columns::{NaPolicy, numeric_column};
    use p2a_validation::{Case, Check, Tol};

    fn case_matrix(ds: &crate::data::Dataset, cols: &[&str]) -> Array2<f64> {
        let mut m = Array2::zeros((ds.nrows(), cols.len()));
        for (j, c) in cols.iter().enumerate() {
            let (v, _) = numeric_column(ds.df(), c, NaPolicy::Fail).unwrap();
            m.column_mut(j).assign(&ndarray::Array1::from_vec(v));
        }
        m
    }
    /// Positions that sort `keys` ascending (the case orders components this way).
    fn order_by(keys: &[f64]) -> Vec<usize> {
        let mut o: Vec<usize> = (0..keys.len()).collect();
        o.sort_by(|&a, &b| keys[a].total_cmp(&keys[b]));
        o
    }
    /// 1-based position of each label in `order` (R's `match(cluster, o)`).
    fn relabel(labels: &[usize], order: &[usize]) -> Vec<f64> {
        let mut pos = vec![0usize; order.len()];
        for (p, &k) in order.iter().enumerate() {
            pos[k] = p;
        }
        labels.iter().map(|&l| (pos[l] + 1) as f64).collect()
    }

    const EM_PARAM: Tol = Tol::new(1e-5, 1e-5);
    const EM_LOGLIK: Tol = Tol::new(1e-12, 0.0);
    const FIXED_POINT: Tol = Tol::new(1e-10, 1e-12);
    const MODELS: [(&str, CovarianceType); 4] = [
        ("VVV", CovarianceType::Full),
        ("EEE", CovarianceType::Tied),
        ("VVI", CovarianceType::Diagonal),
        ("VII", CovarianceType::Spherical),
    ];

    /// The mixture_3d data for the property tests.
    fn mixture_3d() -> Array2<f64> {
        let path = p2a_validation::repo_path("validation/datasets/generated/mixture_3d.csv");
        let df = crate::data::DataLoader::load_csv(&path).unwrap();
        case_matrix(&crate::data::Dataset::new(df), &["x1", "x2", "x3"])
    }

    /// Proportions, means, covariances (component-major) and 1-based classification, components
    /// ordered by the first coordinate of the mean (the case's order).
    fn gmm_values(r: &GaussianMixtureResult) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let o = order_by(&r.means.iter().map(|m| m[0]).collect::<Vec<_>>());
        let pro = o.iter().map(|&k| r.weights[k]).collect();
        let mean = o.iter().flat_map(|&k| r.means[k].clone()).collect();
        let sigma = o
            .iter()
            .flat_map(|&k| {
                r.covariances[k]
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect();
        (pro, mean, sigma, relabel(&r.labels, &o))
    }

    fn check_tight(case: &Case, m: &str, r: &GaussianMixtureResult) {
        let (pro, mean, sigma, cls) = gmm_values(r);
        let k = |s: &str| format!("{m}_{s}");
        // AIC = -2 logLik + 2 df gives the parameter count from fields that exist before the fix.
        let df = (r.aic + 2.0 * r.log_likelihood) / 2.0;
        Check::new(case)
            .scalar(&k("tight_loglik"), r.log_likelihood, EM_LOGLIK)
            .scalar(&k("tight_bic"), -r.bic, EM_LOGLIK) // mclust's BIC is 2 logLik - df ln n
            .scalar(&k("df"), df, Tol::new(0.0, 1e-9))
            .vector(&k("tight_pro"), &pro, EM_PARAM)
            .vector(&k("tight_mean"), &mean, EM_PARAM)
            .vector(&k("tight_sigma"), &sigma, EM_PARAM)
            .vector(&k("tight_classification"), &cls, Tol::exact())
            .scalar(&k("loglik"), r.log_likelihood, Tol::new(1e-12, 2e-10)) // Mclust's own stop, 1.5e-10 below
            .finish();
    }

    fn fit(x: &Array2<f64>, ct: CovarianceType) -> GaussianMixtureResult {
        gaussian_mixture(x.view(), 3, Some(ct), Some(10_000), Some(1e-13), Some(42)).unwrap()
    }

    #[test]
    fn test_validate_gmm_mclust_vvv() {
        let (case, ds) = crate::test_support::load_case("gmm_mclust_3d");
        let x = case_matrix(&ds, &["x1", "x2", "x3"]);
        let r = fit(&x, CovarianceType::Full);
        assert!(r.converged);
        check_tight(&case, "VVV", &r);
    }

    #[test]
    fn test_validate_gmm_mclust_eee() {
        let (case, ds) = crate::test_support::load_case("gmm_mclust_3d");
        let x = case_matrix(&ds, &["x1", "x2", "x3"]);
        let r = fit(&x, CovarianceType::Tied);
        assert!(r.converged);
        check_tight(&case, "EEE", &r);
    }

    #[test]
    fn test_validate_gmm_mclust_vvi_vii() {
        let (case, ds) = crate::test_support::load_case("gmm_mclust_3d");
        let x = case_matrix(&ds, &["x1", "x2", "x3"]);
        for (m, ct) in [
            ("VVI", CovarianceType::Diagonal),
            ("VII", CovarianceType::Spherical),
        ] {
            let r = fit(&x, ct);
            assert!(r.converged, "{m}");
            check_tight(&case, m, &r);
        }
    }

    #[test]
    fn test_gaussian_mixture_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = gaussian_mixture(data.view(), 2, None, Some(50), None, Some(42)).unwrap();

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.n_components, 2);
        assert_eq!(result.weights.len(), 2);
        assert!(result.weights.iter().sum::<f64>() - 1.0 < 1e-6);
    }

    #[test]
    fn test_validate_gmm_mclust_fixed_point() {
        let (case, ds) = crate::test_support::load_case("gmm_mclust_3d");
        let x = case_matrix(&ds, &["x1", "x2", "x3"]);
        for (m, ct) in MODELS {
            let means: Vec<Vec<f64>> = case
                .vector(&format!("{m}_tight_mean"))
                .chunks(3)
                .map(<[f64]>::to_vec)
                .collect();
            let sigma = case.vector(&format!("{m}_tight_sigma"));
            let covariances: Vec<Vec<Vec<f64>>> = sigma
                .chunks(9)
                .map(|c| c.chunks(3).map(<[f64]>::to_vec).collect())
                .collect();
            let mut cfg = GmmConfig::new(3);
            cfg.covariance_type = ct;
            cfg.tolerance = 1e-13;
            cfg.init = GmmInit::Parameters {
                weights: case.vector(&format!("{m}_tight_pro")),
                means,
                covariances,
            };
            let r = gaussian_mixture_with(x.view(), &cfg).unwrap();
            assert!(
                r.converged && r.n_iterations <= 3,
                "{m}: {} iterations",
                r.n_iterations
            );
            assert_eq!(r.model_name, m);
            assert!(r.warnings.is_empty(), "{m}: {:?}", r.warnings);
            let (pro, mean, sigma, cls) = gmm_values(&r);
            let k = |s: &str| format!("{m}_{s}");
            Check::new(&case)
                .scalar(&k("tight_loglik"), r.log_likelihood, Tol::new(1e-13, 0.0))
                .scalar(&k("df"), r.n_parameters as f64, Tol::exact())
                .vector(&k("tight_pro"), &pro, FIXED_POINT)
                .vector(&k("tight_mean"), &mean, FIXED_POINT)
                .vector(&k("tight_sigma"), &sigma, FIXED_POINT)
                .vector(&k("tight_classification"), &cls, Tol::exact())
                .finish();
        }
    }

    #[test]
    fn test_gmm_component_log_pdf_matches_the_closed_form() {
        // A correlated 3 x 3 covariance, with the determinant and the quadratic form by hand.
        let cov = array![[2.0, 0.6, 0.0], [0.6, 1.0, 0.0], [0.0, 0.0, 0.5]];
        let comp = Component::new(vec![1.0, -1.0, 0.0], cov).unwrap();
        let x = array![2.0, 0.0, 1.0];
        // det = (2 - 0.36) * 0.5 = 0.82; inv of the 2 x 2 block = [1, -0.6; -0.6, 2] / 1.64.
        let quad = (1.0 - 1.2 + 2.0) / 1.64 + 2.0;
        let want = -0.5 * (3.0 * (2.0 * std::f64::consts::PI).ln() + 0.82f64.ln() + quad);
        assert!((comp.log_pdf(x.view()) - want).abs() < 1e-14);
        assert!(Component::new(vec![0.0; 2], array![[1.0, 1.0], [1.0, 1.0]]).is_none());
    }

    #[test]
    fn test_gmm_tied_pools_one_covariance() {
        let x = mixture_3d();
        let (full, tied) = (fit(&x, CovarianceType::Full), fit(&x, CovarianceType::Tied));
        assert!(
            tied.covariances.windows(2).all(|w| w[0] == w[1]),
            "tied covariances differ"
        );
        assert!(full.covariances.windows(2).any(|w| w[0] != w[1]));
        // -1442.6 vs -1482.2 (mclust VVV vs EEE); before the fix both were the same full fit
        assert!(full.log_likelihood - tied.log_likelihood > 30.0);
        assert_eq!((full.n_parameters, tied.n_parameters), (29, 17));
    }

    #[test]
    fn test_gmm_is_scale_equivariant() {
        // ML-8: absolute ridges and thresholds made the fit depend on the units. Powers of two
        // scale exactly.
        let x = mixture_3d();
        let base = fit(&x, CovarianceType::Full);
        let (bpro, bmean, _, bcls) = gmm_values(&base);
        for c in [2f64.powi(-10), 2f64.powi(10)] {
            let r = fit(&(&x * c), CovarianceType::Full);
            let (pro, mean, _, cls) = gmm_values(&r);
            assert_eq!(cls, bcls, "classification at scale {c}");
            let want = base.log_likelihood - 300.0 * 3.0 * c.ln();
            assert!(
                (r.log_likelihood - want).abs() <= 1e-11 * want.abs(),
                "logLik {} vs {want}",
                r.log_likelihood
            );
            for (a, b) in mean.iter().zip(&bmean) {
                assert!((a / c - b).abs() <= 1e-5, "mean {a} / {c} vs {b}");
            }
            for (a, b) in pro.iter().zip(&bpro) {
                assert!((a - b).abs() <= 1e-5);
            }
            assert!(
                r.n_iterations.abs_diff(base.n_iterations) <= 3 + base.n_iterations / 4,
                "{} vs {} iterations",
                r.n_iterations,
                base.n_iterations
            );
        }
    }

    #[test]
    fn test_gmm_singular_component_gets_a_relative_ridge() {
        // 24 noisy points near the origin, 4 points exactly on a line far away (rank-1 covariance).
        let x = Array2::from_shape_fn((28, 3), |(i, j)| {
            let t = i as f64;
            if i < 24 {
                [(t * 0.37).sin(), (t * 0.91).cos(), 0.5 * (t * 1.3).sin()][j]
            } else {
                let s = (i - 24) as f64;
                [20.0 + s, 20.0 + 2.0 * s, 20.0 - s][j]
            }
        });
        let r = gaussian_mixture(
            x.view(),
            2,
            Some(CovarianceType::Full),
            Some(200),
            None,
            Some(1),
        )
        .unwrap();
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("singular") && w.contains("ridge")),
            "{:?}",
            r.warnings
        );
        assert!(r.log_likelihood.is_finite());
        let small = gaussian_mixture(
            (&x * 2f64.powi(-16)).view(),
            2,
            Some(CovarianceType::Full),
            Some(200),
            None,
            Some(1),
        )
        .unwrap();
        assert_eq!(small.labels, r.labels); // the ridge is relative to the data's variance
    }

    #[test]
    fn test_gmm_singularity_check_does_not_depend_on_column_units() {
        // Two well-separated correlated clusters, one column in units 1e8 times the other.
        let x = Array2::from_shape_fn((40, 2), |(i, j)| {
            let t = i as f64;
            let c = if i < 20 { 0.0 } else { 10.0 };
            let v = c + [(t * 0.7).sin(), (t * 1.9).cos()][j];
            if j == 1 { v * 1e8 } else { v }
        });
        let r = gaussian_mixture(
            x.view(),
            2,
            Some(CovarianceType::Full),
            Some(50),
            None,
            Some(1),
        )
        .unwrap();
        assert!(
            !r.warnings.iter().any(|w| w.contains("singular")),
            "{:?}",
            r.warnings
        );
    }

    #[test]
    fn test_gmm_rejects_bad_parameters() {
        let x = Array2::from_shape_fn((10, 2), |(i, j)| {
            (i * 7 + j * 3) as f64 * 0.1 + (i as f64).sin()
        });
        let e = |r: Result<GaussianMixtureResult, String>| r.unwrap_err();
        assert!(e(gaussian_mixture(x.view(), 0, None, None, None, None)).contains("n_components"));
        assert!(e(gaussian_mixture(x.view(), 11, None, None, None, None)).contains("n_components"));
        assert!(
            e(gaussian_mixture(x.view(), 2, None, Some(0), None, None)).contains("max_iterations")
        );
        for tol in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                e(gaussian_mixture(x.view(), 2, None, None, Some(tol), None)).contains("tolerance"),
                "{tol}"
            );
        }
        let mut bad = x.clone();
        bad[[3, 1]] = f64::NAN;
        assert!(e(gaussian_mixture(bad.view(), 2, None, None, None, None)).contains("finite"));
        let mut cfg = GmmConfig::new(2);
        cfg.init = GmmInit::Classification(vec![0; 10]); // component 1 empty
        assert!(
            gaussian_mixture_with(x.view(), &cfg)
                .unwrap_err()
                .to_string()
                .contains("component")
        );
    }

    #[test]
    fn test_covariance_type_parses_aliases_and_rejects_unknown() {
        CovarianceType::choices().validate().unwrap();
        for (s, v) in [
            ("full", CovarianceType::Full),
            ("VVV", CovarianceType::Full),
            ("Tied", CovarianceType::Tied),
            ("EEE", CovarianceType::Tied),
            ("diag", CovarianceType::Diagonal),
            ("VVI", CovarianceType::Diagonal),
            ("spherical", CovarianceType::Spherical),
            ("vii", CovarianceType::Spherical),
        ] {
            assert_eq!(s.parse::<CovarianceType>().unwrap(), v, "{s}");
        }
        let e = "bogus".parse::<CovarianceType>().unwrap_err().to_string();
        assert!(
            e.contains("expected one of: full, diagonal, spherical, tied"),
            "{e}"
        );
        assert_eq!(CovarianceType::Tied.mclust_name(), "EEE");
        assert_eq!(CovarianceType::Diagonal.n_parameters(3, 3), 20);
    }

    #[test]
    fn test_gmm_display_names_the_model_and_notes() {
        let x = mixture_3d();
        let r = gaussian_mixture(
            x.view(),
            3,
            Some(CovarianceType::Tied),
            Some(2),
            None,
            Some(42),
        )
        .unwrap();
        let s = format!("{r}");
        assert!(
            s.contains("Model: EEE") && s.contains("Parameters: 17"),
            "{s}"
        );
        assert!(
            !r.converged && s.contains("Converged: no (2 iterations"),
            "{s}"
        );
        assert!(r.warnings.iter().any(|w| w.contains("did not converge")));
    }
}
