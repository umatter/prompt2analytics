//! Vector autoregression VAR(p) by equation-wise OLS ([`run_var`]) and impulse responses ([`compute_irf`], [`run_var_irf`]).

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fmt;

use super::common::{array2_to_vec, df_to_array2};
use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::cholesky;
use crate::linalg::solve::{LM_QR_TOL, Qr};
use crate::resource;
use crate::traits::estimator::t_test_p_value;

/// Result from a VAR model estimation (`vars::VAR(type = "const")`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarResult {
    /// Number of lags p
    pub lags: usize,
    /// Number of variables K
    pub n_vars: usize,
    /// Observations used in each equation: T = n - p.
    pub n_obs: usize,
    /// Variable names
    pub var_names: Vec<String>,
    /// stats::AIC of a vars::VAR fit: -2 logLik + 2 K (1 + K p).
    pub aic: f64,
    /// stats::BIC: -2 logLik + ln(T) K (1 + K p).
    pub bic: f64,
    /// (K, 1 + K p); row = equation, columns [const, y1.l1, .., yK.l1, .., yK.lp] (names in
    /// `coef_names`; vars puts const last).
    pub coefficients: Vec<Vec<f64>>,
    /// Residual covariance U'U / (T - K p - 1), as vars' summary()$covres; used for the SEs
    /// and the orthogonalised IRF. (Before Tier 3 this held U'U / T, now `sigma_u_ml`.)
    pub sigma_u: Vec<Vec<f64>>,
    /// vars::logLik: -T/2 (K ln 2 pi + ln det sigma_u_ml + K).
    pub log_likelihood: f64,
    /// sqrt(sigma_u[eq][eq] * (Z'Z)^-1_jj), as lm inside vars::VAR.
    pub std_errors: Vec<Vec<f64>>,
    /// Coefficient names in the column order of `coefficients`: "const", "<var>.l<lag>".
    #[serde(default)]
    pub coef_names: Vec<String>,
    /// t values, coefficient / SE.
    #[serde(default)]
    pub t_values: Vec<Vec<f64>>,
    /// Two-sided p-values on t(T - K p - 1).
    #[serde(default)]
    pub p_values: Vec<Vec<f64>>,
    /// T - K p - 1, the residual df of each equation (>= K).
    #[serde(default)]
    pub df_residual: usize,
    /// Maximum-likelihood residual covariance U'U / T.
    #[serde(default)]
    pub sigma_u_ml: Vec<Vec<f64>>,
    /// ln det `sigma_u_ml`.
    #[serde(default)]
    pub log_det_sigma_ml: f64,
}

/// A p-value as a coefficient table prints it.
fn format_p(p: f64) -> String {
    if p.is_nan() {
        "NA".to_string()
    } else if p < 1e-4 {
        format!("{p:.2e}")
    } else {
        format!("{p:.4}")
    }
}

/// Entry `[i][j]` of a nested vector, NaN when absent (a deserialised old result).
fn cell(m: &[Vec<f64>], i: usize, j: usize) -> f64 {
    m.get(i).and_then(|r| r.get(j)).copied().unwrap_or(f64::NAN)
}

impl fmt::Display for VarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Vector Autoregression VAR({}) Results", self.lags)?;
        writeln!(f, "===========================================")?;
        writeln!(f, "No. Variables: {}", self.n_vars)?;
        writeln!(
            f,
            "Observations: {} (the first {} rows are used as lags)",
            self.n_obs, self.lags
        )?;
        writeln!(f, "Variables: {}", self.var_names.join(", "))?;
        writeln!(f, "Deterministic terms: constant")?;
        writeln!(
            f,
            "Residual df per equation: {} (T - Kp - 1)",
            self.df_residual
        )?;
        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "AIC: {:.4}", self.aic)?;
        writeln!(f, "BIC: {:.4}", self.bic)?;
        writeln!(
            f,
            "  (AIC and BIC use the T = {} observations of this fit; fits with other lag orders use other samples)",
            self.n_obs
        )?;
        writeln!(f)?;

        let width = self
            .coef_names
            .iter()
            .map(|n| n.chars().count())
            .max()
            .unwrap_or(0)
            .max(4)
            + 2;
        for (eq, name) in self.var_names.iter().enumerate() {
            writeln!(f, "Equation {name}:")?;
            writeln!(
                f,
                "  {:<width$}{:>14}{:>14}{:>11}{:>11}",
                "Term", "Estimate", "Std. Error", "t value", "Pr(>|t|)"
            )?;
            for (j, term) in self.coef_names.iter().enumerate() {
                writeln!(
                    f,
                    "  {:<width$}{:>14.6}{:>14.6}{:>11.3}{:>11}",
                    term,
                    cell(&self.coefficients, eq, j),
                    cell(&self.std_errors, eq, j),
                    cell(&self.t_values, eq, j),
                    format_p(cell(&self.p_values, eq, j))
                )?;
            }
            writeln!(f)?;
        }

        writeln!(
            f,
            "Residual Covariance Matrix (Sigma_u, divisor T - Kp - 1 = {}):",
            self.df_residual
        )?;
        for (i, row) in self.sigma_u.iter().enumerate() {
            write!(
                f,
                "  {}: [",
                self.var_names.get(i).unwrap_or(&format!("Var{}", i))
            )?;
            for (j, val) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:.4}", val)?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

/// Result from VAR Impulse Response Function computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarIrfResult {
    /// Variable names
    pub var_names: Vec<String>,
    /// Number of steps
    pub steps: usize,
    /// Number of lags in the VAR
    pub lags: usize,
    /// IRF tensor: `[step][response_var][shock_var]`
    pub irf: Vec<Vec<Vec<f64>>>,
    /// Orthogonalized IRF (Cholesky decomposition)
    pub orthogonalized: bool,
}

impl fmt::Display for VarIrfResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let orth_str = if self.orthogonalized {
            " (Orthogonalized)"
        } else {
            ""
        };
        writeln!(
            f,
            "VAR({}) Impulse Response Functions{}",
            self.lags, orth_str
        )?;
        writeln!(f, "=======================================")?;
        writeln!(
            f,
            "Steps: {} (horizons 0 to {})",
            self.steps,
            self.steps.saturating_sub(1)
        )?;
        writeln!(f, "Variables: {}", self.var_names.join(", "))?;
        if self.orthogonalized {
            writeln!(
                f,
                "Impact: lower Cholesky factor of Sigma_u, as vars::irf(ortho = TRUE)"
            )?;
        } else {
            writeln!(
                f,
                "Impact: identity (unit shocks), as vars::irf(ortho = FALSE)"
            )?;
        }
        writeln!(
            f,
            "Layout: one table per impulse; rows are horizons, columns responses"
        )?;

        for (shock, impulse) in self.var_names.iter().enumerate() {
            writeln!(f)?;
            writeln!(f, "Impulse: {impulse}")?;
            write!(f, "{:>6}", "h")?;
            for response in &self.var_names {
                write!(f, "{response:>14}")?;
            }
            writeln!(f)?;
            for (h, phi) in self.irf.iter().enumerate() {
                write!(f, "{h:>6}")?;
                for resp in 0..self.var_names.len() {
                    write!(f, "{:>14.6}", cell(phi, resp, shock))?;
                }
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

/// Run a Vector Autoregression (VAR) model.
///
/// # Arguments
/// * `dataset` - The dataset containing the time series data
/// * `columns` - Names of the columns to include in the VAR
/// * `lags` - Number of lags to include
///
/// # Model
/// VAR(p) model: y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + e_t, estimated equation by
/// equation on the common design Z = [1, y_{t-1}', .., y_{t-p}'] (T = n - p rows,
/// m = 1 + K p columns) by pivoted QR.
///
/// # Requirements
/// `lags >= 1`; `n > p + 1` (`InsufficientTimePeriods`); `n >= (K + 1)(p + 1)`, so that each
/// equation keeps at least K residual degrees of freedom (`InsufficientData`); lags of full
/// rank (an aliased lag, e.g. a series proportional to another, is `SingularMatrix` naming
/// it).
///
/// R equivalent: `vars::VAR(y, p = lags, type = "const")`.
///
/// Integer and Boolean columns are accepted; a missing value (null or NaN) is
/// `EconError::NullValues` naming the column, and text is `NonNumericColumn` (as for
/// VARMA, VECM and the IRF, which read their columns the same way).
pub fn run_var(dataset: &Dataset, columns: &[&str], lags: usize) -> EconResult<VarResult> {
    if lags == 0 {
        return Err(EconError::InvalidSpecification {
            message: "VAR requires at least 1 lag".to_string(),
        });
    }

    let (data, var_names) = df_to_array2(dataset.df(), columns)?;
    let (t_full, k) = data.dim();

    if t_full <= lags.saturating_add(1) {
        return Err(EconError::InsufficientTimePeriods {
            lags,
            required: lags.saturating_add(2),
            provided: t_full,
        });
    }

    let t = t_full - lags; // Usable observations

    // Build the design matrix Z: [1, y_{t-1}, y_{t-2}, ..., y_{t-p}]
    // Shape: (t, 1 + k*p). lags < t_full, so k * lags is below the size of the
    // data matrix and the + 1 cannot overflow.
    let n_regressors = 1 + resource::checked_len(&[k, lags], "VAR design")?;

    // ECON-79: each equation has m = 1 + K p coefficients on T = n - p rows, and the
    // K x K residual covariance (likelihood, orthogonalised IRF) needs T - m >= K.
    // vars::VAR returns NA coefficients, NaN SEs or a degenerate logLik below this.
    let df_residual = match t.checked_sub(n_regressors) {
        Some(d) if d >= k => d,
        _ => {
            let required = k
                .checked_add(1)
                .zip(lags.checked_add(1))
                .and_then(|(a, b)| a.checked_mul(b))
                .unwrap_or(usize::MAX);
            return Err(EconError::InsufficientData {
                required,
                provided: t_full,
                context: format!(
                    "VAR({lags}) of {k} series: each equation has 1 + K p = {n_regressors} \
                     coefficients on n - p = {t} rows, and the K x K residual covariance needs \
                     at least K = {k} residual degrees of freedom (n >= (K + 1)(p + 1) = \
                     {required}); vars::VAR returns NaN standard errors or a degenerate logLik here"
                ),
            });
        }
    };

    let mut z = resource::try_zeros2(t, n_regressors, "VAR design")?;
    let mut y = Array2::zeros((t, k));

    for i in 0..t {
        let t_idx = i + lags;

        // Intercept
        z[[i, 0]] = 1.0;

        // Lagged values
        for lag in 0..lags {
            let lag_idx = t_idx - lag - 1;
            for var in 0..k {
                z[[i, 1 + lag * k + var]] = data[[lag_idx, var]];
            }
        }

        // Current values (response)
        for var in 0..k {
            y[[i, var]] = data[[t_idx, var]];
        }
    }

    // Every equation has the same design, so one pivoted QR (dqrdc2, as lm.fit inside
    // vars::VAR) serves all K of them.
    let coef_names = var_coef_names(&var_names, lags);
    let qr = Qr::decompose(z.view(), LM_QR_TOL)?;
    if qr.rank() < n_regressors {
        let names: Vec<&str> = qr
            .aliased()
            .iter()
            .map(|&j| coef_names[j].as_str())
            .collect();
        return Err(EconError::SingularMatrix {
            context: format!(
                "VAR design has rank {} of {n_regressors}: {} aliased (coefficients not defined \
                 because of singularities)",
                qr.rank(),
                names.join(", ")
            ),
            suggestion:
                "Drop the series that is a linear combination of the others (or constant), \
                         or lower the lag order"
                    .to_string(),
        });
    }
    let unscaled = qr.unscaled_covariance()?;

    let mut coefficients = Array2::zeros((k, n_regressors));
    for eq in 0..k {
        coefficients
            .row_mut(eq)
            .assign(&qr.coefficients(y.column(eq))?);
    }
    // U = Y - Z B' in one matrix product (cheaper than K more passes over the reflectors)
    let residuals = &y - &z.dot(&coefficients.t());

    // Residual covariances: U'U / (T - K p - 1) for the SEs and the orthogonalised IRF
    // (vars' summary()$covres, ECON-90), U'U / T for the likelihood.
    let cross = residuals.t().dot(&residuals);
    let sigma_u = &cross / df_residual as f64;
    let sigma_u_ml = &cross / t as f64;

    // Standard errors, t and p of each equation (lm with T - K p - 1 residual df)
    let mut std_errors = Array2::zeros((k, n_regressors));
    let mut t_values = Array2::zeros((k, n_regressors));
    let mut p_values = Array2::zeros((k, n_regressors));
    for eq in 0..k {
        let sigma_eq = sigma_u[[eq, eq]];
        for j in 0..n_regressors {
            let se = (sigma_eq * unscaled[[j, j]]).max(0.0).sqrt();
            let tv = coefficients[[eq, j]] / se;
            std_errors[[eq, j]] = se;
            t_values[[eq, j]] = tv;
            p_values[[eq, j]] = t_test_p_value(tv, df_residual as f64);
        }
    }

    // ECON-50: log det of the ML Sigma through its Cholesky factor (vars' logLik.varest)
    let l = cholesky(&sigma_u_ml.view()).map_err(|_| EconError::SingularMatrix {
        context: "VAR residual covariance (divisor T) is singular: one equation's residuals \
                  are a linear combination of the others'"
            .to_string(),
        suggestion: "Drop the redundant series".to_string(),
    })?;
    let log_det_sigma_ml = 2.0 * (0..k).map(|i| l[[i, i]].ln()).sum::<f64>();
    let (tf, kf) = (t as f64, k as f64);
    let log_likelihood =
        -0.5 * tf * (kf * (2.0 * std::f64::consts::PI).ln() + log_det_sigma_ml + kf);

    // stats::AIC / BIC of a varest: df = sum of the coefficient counts, K (1 + K p)
    let n_params = (k * n_regressors) as f64;
    let aic = -2.0 * log_likelihood + 2.0 * n_params;
    let bic = -2.0 * log_likelihood + n_params * tf.ln();

    Ok(VarResult {
        lags,
        n_vars: k,
        n_obs: t,
        var_names,
        aic,
        bic,
        coefficients: array2_to_vec(&coefficients),
        sigma_u: array2_to_vec(&sigma_u),
        log_likelihood,
        std_errors: array2_to_vec(&std_errors),
        coef_names,
        t_values: array2_to_vec(&t_values),
        p_values: array2_to_vec(&p_values),
        df_residual,
        sigma_u_ml: array2_to_vec(&sigma_u_ml),
        log_det_sigma_ml,
    })
}

/// vars' coefficient names in the column order of the design: `const`, then `<var>.l<lag>`
/// for lag 1..=p and each variable.
fn var_coef_names(var_names: &[String], lags: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(1 + var_names.len() * lags);
    names.push("const".to_string());
    for lag in 1..=lags {
        for v in var_names {
            names.push(format!("{v}.l{lag}"));
        }
    }
    names
}

/// Refuse a [`VarResult`] whose vectors do not have the shapes `n_vars` and `lags` imply
/// (one built by hand or deserialised), instead of indexing past them.
fn check_var_shapes(v: &VarResult) -> EconResult<()> {
    let k = v.n_vars;
    let m = v.lags.checked_mul(k).and_then(|x| x.checked_add(1));
    let ok = k >= 1
        && v.var_names.len() == k
        && v.coefficients.len() == k
        && m.is_some_and(|m| v.coefficients.iter().all(|row| row.len() == m))
        && v.sigma_u.len() == k
        && v.sigma_u.iter().all(|row| row.len() == k);
    if ok {
        return Ok(());
    }
    let p = v.lags;
    Err(EconError::InvalidSpecification {
        message: format!(
            "VarResult is inconsistent: n_vars = {k} and lags = {p} need {k} variable names, {k} \
             coefficient rows of length 1 + K p and a {k} x {k} sigma_u; found {} names, {} \
             coefficient rows (lengths {:?}) and {} sigma_u rows",
            v.var_names.len(),
            v.coefficients.len(),
            v.coefficients.iter().map(Vec::len).collect::<Vec<_>>(),
            v.sigma_u.len()
        ),
    })
}

/// Compute Impulse Response Functions for a VAR model.
///
/// # Arguments
/// * `var_result` - The fitted VAR model
/// * `steps` - Number of periods for IRF
/// * `orthogonalize` - Whether to use Cholesky orthogonalization
pub fn compute_irf(
    var_result: &VarResult,
    steps: usize,
    orthogonalize: bool,
) -> EconResult<VarIrfResult> {
    if steps == 0 {
        return Err(EconError::InvalidSpecification {
            message: "IRF requires at least 1 step".to_string(),
        });
    }
    check_var_shapes(var_result)?;

    let k = var_result.n_vars;
    let p = var_result.lags;

    // The output holds steps k x k matrices; refuse an impossible horizon up front
    let cells = resource::checked_len(&[steps, k, k], "IRF horizons x variables^2")?;
    resource::check_alloc::<f64>(cells, "impulse responses")?;

    // Extract coefficient matrices A_1, A_2, ..., A_p from the VAR result
    // coefficients shape: (k, 1 + k*p) - first column is intercept
    let mut a_matrices: Vec<Array2<f64>> =
        resource::try_with_capacity(p, "VAR coefficient matrices")?;

    for lag in 0..p {
        let mut a_lag = Array2::zeros((k, k));
        for eq in 0..k {
            for var in 0..k {
                let coef_idx = 1 + lag * k + var;
                a_lag[[eq, var]] = var_result.coefficients[eq][coef_idx];
            }
        }
        a_matrices.push(a_lag);
    }

    // Get sigma_u as Array2
    let sigma_u = Array2::from_shape_fn((k, k), |(i, j)| var_result.sigma_u[i][j]);

    // For orthogonalized IRF, compute Cholesky decomposition of sigma_u
    let impact_matrix = if orthogonalize {
        cholesky(&sigma_u.view()).map_err(|_| EconError::SingularMatrix {
            context: "sigma_u is not positive definite, so the orthogonalised impulse responses \
                      have no Cholesky impact matrix"
                .to_string(),
            suggestion: "Use orthogonalize = false, or drop a redundant series".to_string(),
        })?
    } else {
        // Identity matrix for non-orthogonalized IRF
        let mut identity = Array2::zeros((k, k));
        for i in 0..k {
            identity[[i, i]] = 1.0;
        }
        identity
    };

    // Compute IRF recursively
    // Φ_0 = P (impact matrix)
    // Φ_h = Σ_{j=1}^{min(h,p)} A_j Φ_{h-j}
    let mut phi_matrices: Vec<Array2<f64>> = resource::try_with_capacity(steps, "IRF horizons")?;
    phi_matrices.push(impact_matrix.clone());

    for h in 1..steps {
        resource::checkpoint()?;
        let mut phi_h = Array2::zeros((k, k));

        for j in 0..p.min(h) {
            let a_j = &a_matrices[j];
            let phi_hmj = &phi_matrices[h - j - 1];

            // phi_h += A_j * phi_{h-j-1}
            for i in 0..k {
                for l in 0..k {
                    for m in 0..k {
                        phi_h[[i, l]] += a_j[[i, m]] * phi_hmj[[m, l]];
                    }
                }
            }
        }

        phi_matrices.push(phi_h);
    }

    // Convert to nested Vec format
    let mut irf: Vec<Vec<Vec<f64>>> = resource::try_with_capacity(steps, "impulse responses")?;
    for phi in &phi_matrices {
        irf.push(
            (0..k)
                .map(|i| (0..k).map(|j| phi[[i, j]]).collect())
                .collect(),
        );
    }

    Ok(VarIrfResult {
        var_names: var_result.var_names.clone(),
        steps,
        lags: p,
        irf,
        orthogonalized: orthogonalize,
    })
}

/// Convenience function: Run VAR and compute IRF in one call.
pub fn run_var_irf(
    dataset: &Dataset,
    columns: &[&str],
    lags: usize,
    steps: usize,
) -> EconResult<VarIrfResult> {
    let var_result = run_var(dataset, columns, lags)?;
    compute_irf(&var_result, steps, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::econometrics::timeseries::common::test_fixtures::*;
    use crate::econometrics::timeseries::{granger_test, run_varma, run_vecm};
    use crate::regression::ols::na_fixtures::repo_dataset;
    use crate::resource::{Budget, CancelToken, with_budget};
    use polars::prelude::*;

    const CANADA: [&str; 4] = ["e", "prod", "rw", "U"];

    /// The dataset of case var_canada, for tests that compare no case value.
    fn canada_dataset() -> Dataset {
        repo_dataset("validation/datasets/generated/var_canada.csv")
    }

    /// ECON-79. VAR(3) of K = 3 series has m = 1 + K p = 10 coefficients per equation on
    /// T = n - p rows, and the K x K residual covariance needs T - m >= K, i.e.
    /// n >= (K + 1)(p + 1) = 16. R (vars 1.6.1): VAR() on the first 10 rows returns NA
    /// coefficients and NaN SEs and logLik() stops ("system is exactly singular"); at 13 rows
    /// (df 0) the residual SE is NaN. Data: round(matrix(rnorm(48), 16, 3), 4) after
    /// set.seed(79) (no value is pinned here).
    #[test]
    fn test_var_rejects_saturated() {
        let a = [
            1.1074, 0.5392, 0.2494, -0.8363, 1.0850, -2.6291, -1.1520, -1.0125, -0.9942, 0.0919,
            -2.1554, -0.3745, -0.5436, 0.4045, 1.5437, -0.7084,
        ];
        let b = [
            -1.4173, 0.7247, -0.3104, -0.4871, 1.2515, -0.7154, 1.5779, -0.1990, -1.3337, -1.0903,
            0.8764, -2.2822, -0.8152, -0.0305, 1.1114, -0.2963,
        ];
        let c = [
            1.7390, 0.2882, 1.3085, -0.5775, 0.0095, 1.1367, 0.6923, 0.5172, 1.0432, -0.7070,
            0.1435, -0.2663, -2.3517, -0.1631, 0.2768, -0.0246,
        ];
        let rows = |n: usize| {
            Dataset::new(df! { "a" => &a[..n], "b" => &b[..n], "c" => &c[..n] }.unwrap())
        };
        for n in [10usize, 13, 15] {
            match run_var(&rows(n), &["a", "b", "c"], 3) {
                Err(EconError::InsufficientData {
                    required,
                    provided,
                    context,
                }) => {
                    assert_eq!((required, provided), (16, n));
                    assert!(context.contains("1 + K p"), "{context}");
                }
                other => panic!("n = {n}: expected InsufficientData, got {other:?}"),
            }
        }
        let ok = run_var(&rows(16), &["a", "b", "c"], 3).unwrap();
        assert_eq!((ok.n_obs, ok.df_residual), (13, 3));
        assert!(
            ok.std_errors
                .iter()
                .flatten()
                .all(|s| s.is_finite() && *s > 0.0)
        );
        assert!(ok.log_likelihood.is_finite());
    }

    /// ECON-79. b = 2 a exactly: the lag b.l1 is aliased (R: coef(VAR(..)$varresult$a)["b.l1"]
    /// is NA, the later of two collinear columns, dqrdc2's rule). Before: the
    /// pseudo-inverse fit was returned as Ok.
    #[test]
    fn test_var_rejects_collinear_series() {
        let a: Vec<f64> = (0..40)
            .map(|i| ((i * 7919) % 97) as f64 / 97.0 - 0.5 + 0.02 * i as f64)
            .collect();
        let b: Vec<f64> = a.iter().map(|v| 2.0 * v).collect();
        let c: Vec<f64> = (0..40)
            .map(|i| ((i * 104_729) % 89) as f64 / 89.0 - 0.5)
            .collect();
        let ds = Dataset::new(df! { "a" => a, "b" => b, "c" => c }.unwrap());
        match run_var(&ds, &["a", "b", "c"], 1) {
            Err(EconError::SingularMatrix { context, .. }) => {
                assert!(context.contains("b.l1"), "{context}")
            }
            other => panic!("expected SingularMatrix naming b.l1, got {other:?}"),
        }
    }

    /// A VarResult built or deserialized with inconsistent shapes was indexed out of bounds.
    #[test]
    fn test_irf_rejects_inconsistent_var_result() {
        let var = run_var(&canada_dataset(), &CANADA, 2).unwrap();
        let mut short_row = var.clone();
        short_row.coefficients[1].pop();
        let mut short_sigma = var.clone();
        short_sigma.sigma_u.pop();
        let mut short_names = var.clone();
        short_names.var_names.pop();
        for bad in [short_row, short_sigma, short_names] {
            match compute_irf(&bad, 5, true) {
                Err(EconError::InvalidSpecification { message }) => {
                    assert!(
                        message.starts_with("VarResult is inconsistent"),
                        "{message}"
                    )
                }
                other => panic!("expected InvalidSpecification, got {other:?}"),
            }
        }
    }

    /// Column-major copy of a K x K matrix stored as rows (R's vec()).
    fn col_major(m: &[Vec<f64>]) -> Vec<f64> {
        (0..m.len())
            .flat_map(|j| m.iter().map(move |row| row[j]))
            .collect()
    }

    /// R's vec(irf(..)$irf[[impulse]]): responses in variable order, horizons 0..H-1 within
    /// each.
    fn impulse_column_major(r: &VarIrfResult, shock: usize) -> Vec<f64> {
        (0..r.var_names.len())
            .flat_map(|resp| r.irf.iter().map(move |h| h[resp][shock]))
            .collect()
    }

    /// Case `id` (var_canada; each caller names it) and VAR(2) on its dataset.
    fn canada_var(id: &str) -> (p2a_validation::Case, Dataset, VarResult) {
        let (case, ds) = crate::test_support::load_case(id);
        let var = run_var(&ds, &CANADA, 2).unwrap();
        (case, ds, var)
    }

    /// R: vars::VAR(Canada, p = 2, type = "const"); coef(summary(m$varresult[[eq]])) (lm per
    /// equation, residual df T - Kp - 1 = 73); crossprod(resid(m)) / 73
    /// (= summary(m)$covres). Case var_canada.
    #[test]
    fn test_validate_var_canada_coefficients_se() {
        use p2a_validation::{Check, Tol};
        let (case, _ds, var) = canada_var("var_canada");
        let n = var.n_obs; // 82
        let mut check = Check::new(&case);
        for (eq, name) in var.var_names.iter().enumerate() {
            check
                .named(
                    &format!("coef_{name}"),
                    &var.coef_names,
                    &var.coefficients[eq],
                    Tol::coef(n),
                )
                .named(
                    &format!("se_{name}"),
                    &var.coef_names,
                    &var.std_errors[eq],
                    Tol::se(n),
                )
                .named(
                    &format!("t_{name}"),
                    &var.coef_names,
                    &var.t_values[eq],
                    Tol::stat(n),
                )
                .named(
                    &format!("p_{name}"),
                    &var.coef_names,
                    &var.p_values[eq],
                    Tol::p_value(n),
                );
        }
        check
            .vector("sigma_df", &col_major(&var.sigma_u), Tol::stat(n))
            .scalar("n_obs", var.n_obs as f64, Tol::exact())
            .scalar("n_rows", (var.n_obs + var.lags) as f64, Tol::exact())
            .scalar("df_resid", var.df_residual as f64, Tol::exact())
            .finish();
    }

    /// R: logLik(m), AIC(m), BIC(m) (df = K (Kp + 1) = 36); log(det(crossprod(resid(m)) / T)).
    #[test]
    fn test_validate_var_canada_loglik_ic() {
        use p2a_validation::{Check, Tol};
        let (case, _ds, var) = canada_var("var_canada");
        let n = var.n_obs;
        Check::new(&case)
            .scalar("loglik", var.log_likelihood, Tol::stat(n))
            .scalar("aic_stats", var.aic, Tol::stat(n))
            .scalar("bic_stats", var.bic, Tol::stat(n))
            .scalar("sigma_ml_logdet", var.log_det_sigma_ml, Tol::stat(n))
            .vector("sigma_ml", &col_major(&var.sigma_u_ml), Tol::stat(n))
            .finish();
    }

    /// R: irf(m, n.ahead = 12, ortho = TRUE | FALSE, boot = FALSE)$irf[[impulse]]. p2a's
    /// `steps` counts the horizons including h = 0, so steps = n.ahead + 1 = 13.
    /// Psi_h = Phi_h P, P = t(chol(crossprod(resid) / 73)).
    #[test]
    fn test_validate_var_irf_canada() {
        use p2a_validation::{Check, Tol};
        let (case, ds, var) = canada_var("var_canada");
        let n = var.n_obs;
        let ortho = compute_irf(&var, 13, true).unwrap();
        let plain = compute_irf(&var, 13, false).unwrap();
        assert_eq!(run_var_irf(&ds, &CANADA, 2, 13).unwrap().irf, ortho.irf);
        let mut check = Check::new(&case);
        for (j, imp) in CANADA.iter().enumerate() {
            check
                .vector(
                    &format!("irf_ortho_{imp}"),
                    &impulse_column_major(&ortho, j),
                    Tol::stat(n),
                )
                .vector(
                    &format!("irf_plain_{imp}"),
                    &impulse_column_major(&plain, j),
                    Tol::stat(n),
                );
        }
        check.finish();
    }

    /// The coefficient table rows of "Equation <eq>:" (term, [estimate, SE, t]); p is
    /// printed but not parsed.
    fn equation_rows(text: &str, eq: &str) -> Vec<(String, Vec<f64>)> {
        let header = format!("Equation {eq}:");
        let lines = text.lines().skip_while(|l| l.trim() != header).skip(2); // header + titles
        let mut rows = Vec::new();
        for line in lines {
            let tok: Vec<&str> = line.split_whitespace().collect();
            if tok.is_empty() {
                break;
            }
            rows.push((
                tok[0].to_string(),
                tok[1..4]
                    .iter()
                    .map(|s| s.parse::<f64>().unwrap())
                    .collect(),
            ));
        }
        rows
    }

    /// Rows of "Impulse: <imp>": [h, response_1, .., response_K].
    fn impulse_rows(text: &str, imp: &str) -> Vec<Vec<f64>> {
        let header = format!("Impulse: {imp}");
        text.lines()
            .skip_while(|l| l.trim() != header)
            .skip(2)
            .take_while(|l| !l.trim().is_empty())
            .map(|l| {
                l.split_whitespace()
                    .map(|s| s.parse::<f64>().unwrap())
                    .collect()
            })
            .collect()
    }

    /// ECON-49: ts_var returns this text; it must contain the coefficients and SEs (R: vars
    /// summary per equation).
    #[test]
    fn test_validate_var_display_prints_coefficients() {
        use p2a_validation::{Check, Tol};
        let (case, _ds, var) = canada_var("var_canada");
        let text = var.to_string();
        for needle in [
            "Vector Autoregression VAR(2)",
            "No. Variables: 4",
            "Residual df per equation: 73",
            "Log-Likelihood: -175.8186",
            "AIC: 423.6371",
            "BIC: 510.2790",
            "Sigma_u",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in\n{text}");
        }
        let mut check = Check::new(&case);
        for eq in CANADA {
            let rows = equation_rows(&text, eq);
            assert_eq!(rows.len(), 9, "equation {eq}:\n{text}");
            let names: Vec<&str> = rows.iter().map(|(n, _)| n.as_str()).collect();
            let col = |c: usize| rows.iter().map(|(_, v)| v[c]).collect::<Vec<f64>>();
            check
                .named(&format!("coef_{eq}"), &names, &col(0), Tol::printed(6))
                .named(&format!("se_{eq}"), &names, &col(1), Tol::printed(6))
                .named(&format!("t_{eq}"), &names, &col(2), Tol::printed(3));
        }
        check.finish();
    }

    /// ECON-49: ts_var_irf(steps = 13) printed horizons 0-4 and "... (8 more steps)".
    #[test]
    fn test_validate_var_irf_display_prints_every_horizon() {
        use p2a_validation::{Check, Tol};
        let (case, _ds, var) = canada_var("var_canada");
        let text = compute_irf(&var, 13, true).unwrap().to_string();
        assert!(!text.contains("more steps"), "{text}");
        let mut check = Check::new(&case);
        for imp in CANADA {
            let rows = impulse_rows(&text, imp);
            assert_eq!(
                rows.iter().map(|r| r[0] as usize).collect::<Vec<_>>(),
                (0..13).collect::<Vec<_>>(),
                "{text}"
            );
            let printed: Vec<f64> = (0..4)
                .flat_map(|r| rows.iter().map(move |row| row[1 + r]))
                .collect();
            check.vector(&format!("irf_ortho_{imp}"), &printed, Tol::printed(6));
        }
        check.finish();
    }

    fn cancelled() -> Budget {
        let token = CancelToken::new();
        token.cancel();
        Budget::new(None, token)
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

    #[test]
    fn test_var_recovers_dgp() {
        // Test that VAR coefficients are reasonably close to true DGP values
        let dataset = create_var_validation_dataset();
        let result = run_var(&dataset, &["y1", "y2"], 1).unwrap();

        // True DGP: y1_t = 0.5*y1_{t-1} + 0.3*y2_{t-1}
        // Coefficients[0] = [intercept, y1_{t-1} coef, y2_{t-1} coef]
        let y1_coefs = &result.coefficients[0];

        // With noise, coefficients should be within reasonable range of true values
        // Using relaxed tolerance due to finite sample
        let y1_on_y1_lag = y1_coefs[1]; // Should be around 0.5
        let y1_on_y2_lag = y1_coefs[2]; // Should be around 0.3

        // Relaxed check: coefficients in plausible range
        assert!(
            y1_on_y1_lag > -0.5 && y1_on_y1_lag < 1.5,
            "y1 on y1_lag coefficient out of range: {}",
            y1_on_y1_lag
        );
        assert!(
            y1_on_y2_lag > -0.7 && y1_on_y2_lag < 1.3,
            "y1 on y2_lag coefficient out of range: {}",
            y1_on_y2_lag
        );
    }

    #[test]
    fn test_var_irf_shapes_and_decay() {
        // R reference:
        // irf(var_model, impulse = "y1", response = "y2", n.ahead = 5, ortho = TRUE)
        let dataset = create_var_validation_dataset();
        let result = run_var(&dataset, &["y1", "y2"], 1).unwrap();
        let irf = compute_irf(&result, 10, true).unwrap();

        // Verify IRF structure
        assert_eq!(irf.steps, 10);
        assert_eq!(irf.var_names.len(), 2);
        assert!(irf.orthogonalized);

        // IRF tensor: [step][response_var][shock_var]
        assert_eq!(irf.irf.len(), 10);
        assert_eq!(irf.irf[0].len(), 2); // 2 response variables
        assert_eq!(irf.irf[0][0].len(), 2); // 2 shock variables

        // At step 0, diagonal should be non-zero (own shock effect)
        assert!(
            irf.irf[0][0][0].abs() > 0.0,
            "Own shock effect at step 0 should be non-zero"
        );

        // IRF should decay towards zero for stationary VAR
        let final_step_sum: f64 = irf.irf[9].iter().flatten().map(|x| x.abs()).sum();
        let first_step_sum: f64 = irf.irf[0].iter().flatten().map(|x| x.abs()).sum();
        assert!(
            final_step_sum < first_step_sum * 2.0,
            "IRF should not explode for stationary VAR"
        );
    }

    #[test]
    fn test_irf_rejects_unallocatable_horizon() {
        let ds = create_var_validation_dataset();
        let var = run_var(&ds, &["y1", "y2"], 1).unwrap();
        expect_resource_limit(compute_irf(&var, usize::MAX / 4, true));
        expect_resource_limit(run_var_irf(&ds, &["y1", "y2"], 1, usize::MAX / 4));
    }

    #[test]
    fn test_lag_counts_cannot_overflow() {
        let ds = create_var_validation_dataset();
        assert!(matches!(
            run_var(&ds, &["y1", "y2"], usize::MAX),
            Err(EconError::InsufficientTimePeriods { .. })
        ));
        assert!(run_varma(&ds, &["y1", "y2"], usize::MAX, 0).is_err());
        assert!(run_varma(&ds, &["y1", "y2"], 1, usize::MAX).is_err());
        assert!(run_vecm(&ds, &["y1", "y2"], usize::MAX, 1).is_err());
        let g = create_granger_test_dataset();
        assert!(granger_test(&g, "y", "x", usize::MAX).is_err());
    }

    #[test]
    fn test_irf_stops_when_cancelled() {
        let ds = create_var_validation_dataset();
        let var = run_var(&ds, &["y1", "y2"], 1).unwrap();
        expect_cancelled(with_budget(cancelled(), || compute_irf(&var, 10, true)));
    }
}
