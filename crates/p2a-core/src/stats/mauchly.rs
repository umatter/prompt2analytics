//! Mauchly's test for sphericity.
//!
//! Implements Mauchly's test of sphericity for repeated measures designs.
//! This test evaluates whether a covariance matrix (or transformation thereof)
//! is proportional to the identity matrix, an assumption required for univariate
//! F-tests in repeated measures ANOVA.
//!
//! # Mathematical Background
//!
//! For p repeated measures, the sphericity assumption requires that the
//! variance-covariance matrix of all pairwise differences be proportional
//! to the identity matrix.
//!
//! ## Mauchly's W Statistic
//!
//! With T the pp x p transformation (Helmert contrasts by default, R's `X = ~1`), S the sample
//! covariance, B = T S T' and Psi = T Sigma T' (Sigma = I unless given), l are the eigenvalues of
//! Psi^-1 B (R's `U = solve(Psi, B)`), computed as those of the symmetric Psi^-1/2 B Psi^-1/2:
//!
//! W = prod(l) / mean(l)^pp
//!
//! With nd = n - 1 and rho = 1 - (2 pp^2 + pp + 2) / (6 pp nd), z = -nd rho ln(W) is
//! approximately chi-square with f = pp (pp + 1) / 2 - 1 degrees of freedom; the p-value adds
//! Anderson's (2003, Sec. 10.7.4) second-order term
//!
//! p = Pr(chi2_f > z) + w2 (Pr(chi2_{f+4} > z) - Pr(chi2_f > z)),
//! w2 = (pp + 2)(pp - 1)(pp - 2)(2 pp^3 + 6 pp^2 + 3 pp + 2) / (288 (nd pp rho)^2),
//!
//! as R's `mauchly.test`, except that R writes `3 * p` (the number of measures) for Anderson's
//! `3 pp` (see `validation/stats/mauchly.md`).
//!
//! ## Epsilon Corrections
//!
//! When sphericity is violated, degrees of freedom are corrected:
//!
//! - **Greenhouse-Geisser (ε̂_GG)**: Conservative lower bound
//!   ε̂_GG = (Σ l)² / (pp Σ l²)
//!
//! - **Huynh-Feldt (ε̂_HF)**: Less conservative adjustment
//!   ε̂_HF = (n pp ε̂_GG - 2) / (pp (n - 1 - pp ε̂_GG)), capped at 1
//!
//! - **Lower bound**: ε_LB = 1/pp
//!
//! # References
//!
//! - Mauchly, J.W. (1940). Significance test for sphericity of a normal n-variate
//!   distribution. *The Annals of Mathematical Statistics*, 11(2), 204-209.
//!   https://doi.org/10.1214/aoms/1177731915
//!
//! - Greenhouse, S.W., & Geisser, S. (1959). On methods in the analysis of profile
//!   data. *Psychometrika*, 24(2), 95-112. https://doi.org/10.1007/BF02289823
//!
//! - Huynh, H., & Feldt, L.S. (1976). Estimation of the Box correction for degrees
//!   of freedom from sample data in randomized block and split-plot designs.
//!   *Journal of Educational Statistics*, 1(1), 69-82.
//!   https://doi.org/10.3102/10769986001001069
//!
//! - Anderson, T.W. (2003). *An Introduction to Multivariate Statistical Analysis* (3rd ed.),
//!   Section 10.7.4. Wiley.
//!
//! - Box, G.E.P. (1954). Some theorems on quadratic forms applied in the study
//!   of analysis of variance problems, I. Effect of inequality of variance in
//!   the one-way classification. *The Annals of Mathematical Statistics*, 25(2),
//!   290-302. https://doi.org/10.1214/aoms/1177728786
//!
//! R equivalent: `stats::mauchly.test()`

use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::eig_symmetric;
use crate::linalg::solve::SYM_RANK_TOL;
use crate::traits::estimator::chi_squared_p_value;
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

// ============================================================================
// Mauchly Test Result
// ============================================================================

/// Result of Mauchly's test for sphericity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MauchlyResult {
    /// Mauchly's W statistic
    pub w: f64,
    /// Chi-squared test statistic (R's z = -nd rho ln W)
    pub chi_squared: f64,
    /// Degrees of freedom for chi-squared test
    pub df: f64,
    /// P-value, with Anderson's second-order term
    pub p_value: f64,
    /// Number of levels (p)
    pub p_levels: usize,
    /// Number of subjects/observations (n)
    pub n: usize,
    /// Greenhouse-Geisser epsilon correction
    pub epsilon_gg: f64,
    /// Huynh-Feldt epsilon correction (capped at 1)
    pub epsilon_hf: f64,
    /// Lower bound epsilon (1 / n_contrasts)
    pub epsilon_lb: f64,
    /// The first-order chi-square tail Pr(chi2_df > chi_squared)
    #[serde(default)]
    pub p_value_first_order: f64,
    /// Number of rows of the transformation T (R's pp)
    #[serde(default)]
    pub n_contrasts: usize,
    /// The transformation, in words
    #[serde(default)]
    pub transform: String,
}

impl std::fmt::Display for MauchlyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Mauchly's Test of Sphericity")?;
        writeln!(f, "=============================")?;
        if !self.transform.is_empty() {
            writeln!(f, "Transformation: {}", self.transform)?;
        }
        writeln!(f, "W statistic: {:.6}", self.w)?;
        writeln!(f, "Chi-squared: {:.4}", self.chi_squared)?;
        writeln!(f, "df: {:.0}", self.df)?;
        writeln!(
            f,
            "p-value: {:.6} (first-order chi-square {:.6})",
            self.p_value, self.p_value_first_order
        )?;
        writeln!(f)?;

        if self.p_value < 0.05 {
            writeln!(f, "Sphericity assumption is violated (p < 0.05)")?;
            writeln!(f, "Consider using epsilon corrections:")?;
        } else {
            writeln!(f, "Sphericity assumption is met (p >= 0.05)")?;
        }
        writeln!(f)?;

        writeln!(f, "Epsilon Corrections:")?;
        writeln!(f, "  Greenhouse-Geisser: {:.4}", self.epsilon_gg)?;
        writeln!(f, "  Huynh-Feldt: {:.4}", self.epsilon_hf)?;
        writeln!(f, "  Lower bound: {:.4}", self.epsilon_lb)?;

        Ok(())
    }
}

/// The transformation T applied to the p measures before testing sphericity.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum MauchlyTransform {
    /// Orthonormal contrasts orthogonal to the constant (R's `X = ~1`; the p - 1 Helmert rows)
    #[default]
    Contrasts,
    /// The measures themselves (R's default `T = I`)
    Identity,
    /// A user-supplied pp x p matrix (R's `T`). W, the epsilons and the p-value depend only on its
    /// row space, so any `idata` / `M` / `X` design of R is tested by passing its T.
    Matrix(Array2<f64>),
}

/// The options of [`mauchly_test_with`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MauchlyConfig {
    /// The transformation T (default: contrasts, R's `X = ~1`)
    pub transform: MauchlyTransform,
    /// The covariance proportionality is tested against (R's `Sigma`; identity when absent)
    pub sigma: Option<Array2<f64>>,
}

fn invalid(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

// ============================================================================
// Core Implementation
// ============================================================================

/// Mauchly's test that the covariance of the transformed measures `T y` is proportional to
/// `T Sigma T'` (R's `mauchly.test(object, M, X, idata, T, Sigma)` on an intercept-only mlm).
///
/// # Errors
///
/// `InvalidSpecification` for fewer than 2 subjects or measures, a non-finite cell, a `sigma`
/// that is not p x p, finite, symmetric and positive definite, a `T` without p columns, rows or
/// finite entries, or a `T Sigma T'` that is not positive definite (T without full row rank);
/// `InsufficientData` when there are no more subjects than contrasts or the transformed sample
/// covariance is singular (R returns W = 0 there).
pub fn mauchly_test_with(data: &Array2<f64>, config: &MauchlyConfig) -> EconResult<MauchlyResult> {
    let (n, p) = data.dim();
    if n < 2 {
        return Err(invalid(format!(
            "Mauchly's test needs at least 2 observations, got {n}"
        )));
    }
    if p < 2 {
        return Err(invalid(format!(
            "Mauchly's test needs at least 2 repeated measures, got {p}"
        )));
    }
    let non_finite = data.iter().filter(|v| !v.is_finite()).count();
    if non_finite > 0 {
        return Err(invalid(format!(
            "data must be finite; {non_finite} cell(s) are NaN or infinite"
        )));
    }
    if let Some(sig) = &config.sigma {
        if sig.dim() != (p, p) {
            return Err(invalid(format!(
                "sigma must be {p} x {p}, got {} x {}",
                sig.nrows(),
                sig.ncols()
            )));
        }
        if sig.iter().any(|v| !v.is_finite()) {
            return Err(invalid("sigma must be finite".to_string()));
        }
        let scale = sig.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for i in 0..p {
            for j in 0..i {
                if (sig[[i, j]] - sig[[j, i]]).abs() > 1e-12 * scale.max(f64::MIN_POSITIVE) {
                    return Err(invalid(format!(
                        "sigma must be symmetric (sigma[{i}, {j}] = {} but sigma[{j}, {i}] = {})",
                        sig[[i, j]],
                        sig[[j, i]]
                    )));
                }
            }
        }
        let (lam, _) = eig_symmetric(&sig.view())?;
        let (lo, hi) = min_max(&lam);
        if !(hi > 0.0 && lo > SYM_RANK_TOL * p as f64 * hi) {
            return Err(invalid(format!(
                "sigma must be positive definite (smallest eigenvalue {lo:e}, largest {hi:e})"
            )));
        }
    }

    let (t, mut transform) = match &config.transform {
        MauchlyTransform::Contrasts => (
            create_orthogonal_contrasts(p).t().to_owned(),
            "contrasts orthogonal to the constant (R: X = ~1)".to_string(),
        ),
        MauchlyTransform::Identity => (Array2::eye(p), "identity (R: T = I)".to_string()),
        MauchlyTransform::Matrix(t) => {
            if t.ncols() != p {
                return Err(invalid(format!(
                    "T must have {p} columns (one per measure), got {} x {}",
                    t.nrows(),
                    t.ncols()
                )));
            }
            if t.nrows() == 0 {
                return Err(invalid("T must have at least one row".to_string()));
            }
            if t.iter().any(|v| !v.is_finite()) {
                return Err(invalid("T must be finite".to_string()));
            }
            (t.clone(), "user-supplied contrast matrix T".to_string())
        }
    };
    if config.sigma.is_some() {
        transform.push_str(", tested against Sigma");
    }
    let pp = t.nrows();

    // B = T S T', Psi = T Sigma T'
    let cov = compute_covariance(data)?;
    let b = t.dot(&cov).dot(&t.t());
    let psi = match &config.sigma {
        Some(sig) => t.dot(sig).dot(&t.t()),
        None => t.dot(&t.t()),
    };
    let psi = symmetrise(&psi);
    let (lam, v) = eig_symmetric(&psi.view())?;
    let (lo, hi) = min_max(&lam);
    if !(hi > 0.0 && lo > SYM_RANK_TOL * pp as f64 * hi) {
        return Err(invalid(
            "T Sigma T' must be positive definite (T needs full row rank)".to_string(),
        ));
    }
    let inv_sqrt = Array1::from_iter(lam.iter().map(|l| 1.0 / l.sqrt()));
    let psi_inv_half = (&v * &inv_sqrt).dot(&v.t());
    let c = symmetrise(&psi_inv_half.dot(&b).dot(&psi_inv_half));
    let (l, _) = eig_symmetric(&c.view())?;

    let nd = (n - 1) as f64;
    let (l_lo, l_hi) = min_max(&l);
    if n - 1 < pp || !(l_hi > 0.0 && l_lo > SYM_RANK_TOL * pp as f64 * l_hi) {
        return Err(EconError::InsufficientData {
            required: pp + 1,
            provided: n,
            context: "Mauchly's test needs more subjects than contrasts and a non-singular \
                      transformed covariance"
                .to_string(),
        });
    }

    let ppf = pp as f64;
    let sum_l: f64 = l.sum();
    let log_w = l.iter().map(|x| x.ln()).sum::<f64>() - ppf * (sum_l / ppf).ln();
    let w = log_w.exp();
    let rho = 1.0 - (2.0 * ppf * ppf + ppf + 2.0) / (6.0 * ppf * nd);
    let z = -nd * rho * log_w;
    let df = ppf * (ppf + 1.0) / 2.0 - 1.0;
    // Anderson (2003, Sec. 10.7.4), with pp throughout (R's code has 3 * p for 3 * pp)
    let w2 = (ppf + 2.0)
        * (ppf - 1.0)
        * (ppf - 2.0)
        * (2.0 * ppf.powi(3) + 6.0 * ppf * ppf + 3.0 * ppf + 2.0)
        / (288.0 * (nd * ppf * rho).powi(2));
    let tail = |k: f64| {
        if z <= 0.0 {
            1.0
        } else {
            chi_squared_p_value(z, k)
        }
    };
    let (p_value, p_value_first_order) = if df == 0.0 {
        (1.0, 1.0)
    } else {
        let pr1 = tail(df);
        (pr1 + w2 * (tail(df + 4.0) - pr1), pr1)
    };

    // Epsilons
    let epsilon_lb = 1.0 / ppf;
    let sum_l2: f64 = l.iter().map(|x| x * x).sum();
    let epsilon_gg = (sum_l * sum_l / (ppf * sum_l2)).clamp(epsilon_lb, 1.0);
    let denominator = ppf * (nd - ppf * epsilon_gg);
    let epsilon_hf = if denominator > 0.0 {
        (((nd + 1.0) * ppf * epsilon_gg - 2.0) / denominator).clamp(epsilon_gg, 1.0)
    } else {
        1.0
    };

    Ok(MauchlyResult {
        w,
        chi_squared: z,
        df,
        p_value,
        p_levels: p,
        n,
        epsilon_gg,
        epsilon_hf,
        epsilon_lb,
        p_value_first_order,
        n_contrasts: pp,
        transform,
    })
}

/// Mauchly's test with R's `X = ~1` contrasts, against `sigma` when given (identity otherwise).
///
/// # Arguments
///
/// * `data` - Data matrix (n × p) where n = subjects, p = repeated measures
/// * `sigma` - The covariance sphericity is tested against (None = identity)
///
/// # Errors
///
/// As [`mauchly_test_with`].
pub fn mauchly_test(data: &Array2<f64>, sigma: Option<&Array2<f64>>) -> EconResult<MauchlyResult> {
    mauchly_test_with(
        data,
        &MauchlyConfig {
            transform: MauchlyTransform::Contrasts,
            sigma: sigma.cloned(),
        },
    )
}

fn min_max(v: &Array1<f64>) -> (f64, f64) {
    v.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        })
}

fn symmetrise(m: &Array2<f64>) -> Array2<f64> {
    (m + &m.t()) / 2.0
}

/// Compute sample covariance matrix.
fn compute_covariance(data: &Array2<f64>) -> EconResult<Array2<f64>> {
    let (n, _) = data.dim();
    let means = data.mean_axis(Axis(0)).ok_or(EconError::EmptyDataset)?;
    let centered = data - &means;
    Ok(centered.t().dot(&centered) / (n - 1) as f64)
}

/// Create orthogonal contrast matrix for sphericity test.
///
/// Creates a p × (p-1) matrix where each column represents an orthogonal contrast.
/// The simplest choice is successive differences, but we use Helmert contrasts
/// which are orthonormal.
fn create_orthogonal_contrasts(p: usize) -> Array2<f64> {
    let d = p - 1;
    let mut m = Array2::zeros((p, d));

    // Helmert contrasts (orthonormal)
    for j in 0..d {
        let k = j + 1;
        // First k elements get 1/sqrt(k*(k+1))
        let scale1 = 1.0 / ((k * (k + 1)) as f64).sqrt();
        for i in 0..k {
            m[[i, j]] = scale1;
        }
        // (k+1)th element gets -k/sqrt(k*(k+1))
        m[[k, j]] = -(k as f64) / ((k * (k + 1)) as f64).sqrt();
    }

    m
}

// ============================================================================
// Simplified interface
// ============================================================================

/// Reshape row-major `data` (n × p).
fn reshape(data: &[f64], n: usize, p: usize) -> EconResult<Array2<f64>> {
    if n.checked_mul(p) != Some(data.len()) {
        return Err(invalid(format!(
            "Data length {} doesn't match {} × {}",
            data.len(),
            n,
            p
        )));
    }
    Array2::from_shape_vec((n, p), data.to_vec())
        .map_err(|e| invalid(format!("Failed to reshape data: {e}")))
}

/// Perform Mauchly's test from flat data.
///
/// # Arguments
///
/// * `data` - Flat data in row-major order (n × p)
/// * `n` - Number of subjects
/// * `p` - Number of repeated measures
pub fn mauchly_test_from_slice(data: &[f64], n: usize, p: usize) -> EconResult<MauchlyResult> {
    mauchly_test(&reshape(data, n, p)?, None)
}

/// Run Mauchly's test (MCP wrapper).
pub fn run_mauchly_test(data: &[f64], n: usize, p: usize) -> EconResult<MauchlyResult> {
    mauchly_test_from_slice(data, n, p)
}

/// Run Mauchly's test against `sigma` (row-major p × p), with R's `X = ~1` contrasts.
pub fn run_mauchly_test_with_sigma(
    data: &[f64],
    n: usize,
    p: usize,
    sigma: &[f64],
) -> EconResult<MauchlyResult> {
    if p.checked_mul(p) != Some(sigma.len()) {
        return Err(invalid(format!(
            "sigma must be {p} x {p} ({} values), got {} values",
            p.saturating_mul(p),
            sigma.len()
        )));
    }
    let sigma = Array2::from_shape_vec((p, p), sigma.to_vec())
        .map_err(|e| invalid(format!("Failed to reshape sigma: {e}")))?;
    mauchly_test(&reshape(data, n, p)?, Some(&sigma))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};
    use p2a_validation::{Case, Check, Tol};

    #[test]
    fn test_mauchly_spherical() {
        // Data with approximate sphericity (different noise patterns)
        // Measurements have similar variance differences
        let data = array![
            [1.0, 1.5, 2.1],
            [2.0, 2.4, 2.9],
            [1.5, 2.1, 2.5],
            [2.5, 2.9, 3.4],
            [1.2, 1.8, 2.3],
            [2.2, 2.7, 3.1],
            [1.8, 2.2, 2.8],
            [2.8, 3.3, 3.7],
            [1.4, 1.9, 2.4],
            [2.4, 2.8, 3.3],
            [3.1, 3.5, 4.0],
            [0.8, 1.3, 1.8],
        ];

        let result = mauchly_test(&data, None).unwrap();

        // For approximately spherical data, W should be reasonably close to 1
        // And p-value should be non-significant (> 0.05)
        assert!(result.w.is_finite(), "W should be finite");
        assert!(result.p_value.is_finite(), "p-value should be finite");
    }

    #[test]
    fn test_mauchly_non_spherical() {
        // Data with clear violation of sphericity
        // Variance of first difference >> variance of second difference
        let data = array![
            [1.0, 10.0, 11.0], // Large jump then small
            [2.0, 12.0, 12.5],
            [1.5, 15.0, 15.2],
            [2.5, 8.0, 8.3],
            [1.2, 11.0, 11.1],
            [2.2, 14.0, 14.2],
            [1.8, 9.0, 9.5],
            [2.8, 13.0, 13.1],
            [1.4, 16.0, 16.3],
            [2.4, 7.0, 7.2],
            [3.0, 5.0, 5.8],
            [0.5, 18.0, 18.1],
        ];

        let result = mauchly_test(&data, None).unwrap();

        // W should be small for non-spherical data
        assert!(
            result.w < 0.9 || result.p_value < 0.05,
            "W = {} should be smaller or p < 0.05",
            result.w
        );
        // Epsilon corrections should be < 1
        assert!(result.epsilon_gg <= 1.0);
        assert!(result.epsilon_hf <= 1.0);
    }

    #[test]
    fn test_mauchly_epsilons() {
        // Test epsilon correction bounds with varied data
        let data = array![
            [1.0, 2.2, 3.1, 4.3],
            [2.0, 3.1, 4.5, 5.2],
            [1.5, 2.8, 3.6, 4.9],
            [2.5, 3.4, 4.8, 5.7],
            [1.2, 2.5, 3.3, 4.6],
            [2.2, 3.0, 4.2, 5.4],
            [1.8, 2.6, 3.9, 4.7],
            [2.8, 3.7, 4.6, 5.9],
            [1.4, 2.3, 3.4, 4.5],
            [2.4, 3.2, 4.4, 5.6],
            [3.1, 4.0, 5.1, 6.2],
            [0.9, 2.1, 3.0, 4.2],
        ];

        let result = mauchly_test(&data, None).unwrap();

        // Lower bound should be 1/(p-1) = 1/3
        assert!((result.epsilon_lb - 1.0 / 3.0).abs() < 0.01);

        // GG should be >= lower bound
        assert!(result.epsilon_gg >= result.epsilon_lb);

        // HF should be >= GG
        assert!(result.epsilon_hf >= result.epsilon_gg - 0.001); // Small tolerance

        // All epsilons should be <= 1
        assert!(result.epsilon_gg <= 1.0);
        assert!(result.epsilon_hf <= 1.0);
    }

    #[test]
    fn test_mauchly_from_slice() {
        // More varied data
        let data = vec![
            1.0, 2.1, 3.3, 2.0, 3.2, 4.1, 1.5, 2.4, 3.8, 2.5, 3.6, 4.5, 1.2, 2.6, 3.2, 3.0, 4.1,
            5.2,
        ];

        let result = mauchly_test_from_slice(&data, 6, 3).unwrap();

        assert!(result.w.is_finite());
        assert!(result.df > 0.0);
    }

    #[test]
    fn test_orthogonal_contrasts() {
        let m = create_orthogonal_contrasts(4);

        // Check dimensions
        assert_eq!(m.dim(), (4, 3));

        // Check orthogonality: M'M should be close to identity
        let mtm = m.t().dot(&m);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (mtm[[i, j]] - expected).abs() < 1e-10,
                    "M'M[{},{}] = {} (expected {})",
                    i,
                    j,
                    mtm[[i, j]],
                    expected
                );
            }
        }
    }

    fn mauchly_case() -> (Case, Array2<f64>) {
        let (case, ds) = crate::test_support::load_case("stat_mauchly_sigma");
        let cols: Vec<Vec<f64>> = ["V1", "V2", "V3", "V4"]
            .iter()
            .map(|c| {
                crate::data::columns::numeric_column(
                    ds.df(),
                    c,
                    crate::data::columns::NaPolicy::Fail,
                )
                .unwrap()
                .0
            })
            .collect();
        let n = cols[0].len();
        (case, Array2::from_shape_fn((n, 4), |(i, j)| cols[j][i]))
    }

    fn matrix(case: &Case, key: &str, rows: usize) -> Array2<f64> {
        let v = case.vector(key);
        let cols = v.len() / rows;
        Array2::from_shape_vec((rows, cols), v).unwrap()
    }

    #[test]
    fn test_validate_mauchly_sigma() {
        // R: mauchly.test(lm(Y ~ 1), ...) on mauchly_sigma (15 subjects, 4 measures) with T = I,
        // X = ~1, idata f = (a, a, b, b) with X = ~f (passed as R's T), and Sigma = diag(1:4) with
        // T = I and with X = ~1.
        let (case, y) = mauchly_case();
        let n = y.nrows(); // 15
        let psi = Array2::from_diag(&Array1::from(case.vector("psi_diag")));
        let designs = [
            (
                "identity",
                MauchlyConfig {
                    transform: MauchlyTransform::Identity,
                    sigma: None,
                },
            ),
            ("contrasts", MauchlyConfig::default()),
            (
                "within_f",
                MauchlyConfig {
                    transform: MauchlyTransform::Matrix(matrix(&case, "within_f_T", 2)),
                    sigma: None,
                },
            ),
            (
                "psi_identity",
                MauchlyConfig {
                    transform: MauchlyTransform::Identity,
                    sigma: Some(psi.clone()),
                },
            ),
            (
                "psi_contrasts",
                MauchlyConfig {
                    transform: MauchlyTransform::Contrasts,
                    sigma: Some(psi.clone()),
                },
            ),
        ];
        let mut c = Check::new(&case);
        for (k, cfg) in &designs {
            let r = mauchly_test_with(&y, cfg).unwrap();
            // deviation STAT-82: p uses Anderson's w2 with 3 pp (R's code has 3 * p), pinned to
            // the case's p_anderson; R's own p is within 1e-3 relative
            c.scalar(&format!("{k}_W"), r.w, Tol::stat(n))
                .scalar(&format!("{k}_p_anderson"), r.p_value, Tol::p_value(n))
                .scalar(&format!("{k}_p"), r.p_value, Tol::new(1e-3, 1e-12))
                .scalar(
                    &format!("{k}_p_first_order"),
                    r.p_value_first_order,
                    Tol::p_value(n),
                )
                .scalar(&format!("{k}_chisq"), r.chi_squared, Tol::stat(n))
                .scalar(&format!("{k}_df"), r.df, Tol::exact())
                .scalar(&format!("{k}_pp"), r.n_contrasts as f64, Tol::exact())
                .scalar(&format!("{k}_gg"), r.epsilon_gg, Tol::stat(n));
            // stats:::sphericity prints HF uncapped (within_f: 1.059110549795335); anova.mlm and
            // p2a use min(1, HF)
            let hf = case.scalar(&format!("{k}_hf")).min(1.0);
            assert!(
                Tol::stat(n).allows(r.epsilon_hf, hf),
                "{k}_hf rust={} R={hf}",
                r.epsilon_hf
            );
            assert!((r.epsilon_lb - 1.0 / r.n_contrasts as f64).abs() < 1e-15);
        }
        // R's own non-orthonormal basis Thin.row(Proj(I) - Proj(1)): the test is basis-invariant
        let r = mauchly_test_with(
            &y,
            &MauchlyConfig {
                transform: MauchlyTransform::Matrix(matrix(&case, "contrasts_T", 3)),
                sigma: None,
            },
        )
        .unwrap();
        c.scalar("contrasts_W", r.w, Tol::stat(n)).scalar(
            "contrasts_p_anderson",
            r.p_value,
            Tol::p_value(n),
        );
        // the legacy entry point honours sigma (STAT-56: it was transformed and discarded)
        let r = mauchly_test(&y, Some(&psi)).unwrap();
        c.scalar("psi_contrasts_W", r.w, Tol::stat(n)).scalar(
            "psi_contrasts_p_anderson",
            r.p_value,
            Tol::p_value(n),
        );
        let flat: Vec<f64> = y.iter().copied().collect();
        let psi_flat: Vec<f64> = psi.iter().copied().collect();
        let r = run_mauchly_test_with_sigma(&flat, n, 4, &psi_flat).unwrap();
        c.scalar("psi_contrasts_W", r.w, Tol::stat(n));
        c.finish();
    }

    #[test]
    fn test_validate_mauchly_two_columns() {
        // R: mauchly.test(lm(Y[, 1:2] ~ 1), X = ~1): one contrast, W = 1, p = 1 (was p = NaN)
        let (case, y) = mauchly_case();
        let r = mauchly_test(&y.slice(s![.., 0..2]).to_owned(), None).unwrap();
        Check::new(&case)
            .scalar("two_col_W", r.w, Tol::exact())
            .scalar("two_col_p", r.p_value, Tol::exact())
            .finish();
        assert_eq!((r.df, r.n_contrasts), (0.0, 1));
    }

    #[test]
    fn test_mauchly_rejects_invalid_sigma_and_transform() {
        let (_case, y) = mauchly_case();
        let msg = |c: MauchlyConfig| mauchly_test_with(&y, &c).unwrap_err().to_string();
        let sig = |m: Array2<f64>| MauchlyConfig {
            sigma: Some(m),
            ..Default::default()
        };
        assert!(msg(sig(Array2::eye(3))).contains("sigma must be 4 x 4"));
        let mut asym = Array2::eye(4);
        asym[[0, 1]] = 0.5;
        assert!(msg(sig(asym)).contains("symmetric"));
        assert!(msg(sig(-Array2::<f64>::eye(4))).contains("positive definite"));
        let mut nan = Array2::eye(4);
        nan[[2, 2]] = f64::NAN;
        assert!(msg(sig(nan)).contains("finite"));
        let t = |m: Array2<f64>| MauchlyConfig {
            transform: MauchlyTransform::Matrix(m),
            sigma: None,
        };
        assert!(msg(t(Array2::zeros((2, 3)))).contains("T must have 4 columns"));
        // rank-deficient T: two equal rows
        let dup = array![[1.0, -1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]];
        assert!(msg(t(dup)).contains("full row rank"));
        // 3 subjects, 4 measures with T = I: B is singular (R returns W = 0)
        let few = y.slice(s![0..3, ..]).to_owned();
        assert!(matches!(
            mauchly_test_with(
                &few,
                &MauchlyConfig {
                    transform: MauchlyTransform::Identity,
                    sigma: None
                }
            ),
            Err(EconError::InsufficientData { .. })
        ));
        let mut bad = y.clone();
        bad[[0, 0]] = f64::NAN;
        assert!(
            mauchly_test(&bad, None)
                .unwrap_err()
                .to_string()
                .contains("finite")
        );
        let e = run_mauchly_test_with_sigma(&[0.0; 8], 2, 4, &[1.0; 4]).unwrap_err();
        assert!(e.to_string().contains("sigma must be 4 x 4"), "{e}");
    }

    #[test]
    fn test_mauchly_display_names_transform_and_both_p_values() {
        let (_case, y) = mauchly_case();
        let text = format!("{}", mauchly_test(&y, None).unwrap());
        assert!(text.contains("Transformation: contrasts orthogonal to the constant"));
        assert!(text.contains("(first-order chi-square 0.353699)"), "{text}");
    }
}
