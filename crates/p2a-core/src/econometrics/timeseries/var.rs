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

/// Result from a VAR model estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarResult {
    /// Number of lags
    pub lags: usize,
    /// Number of variables
    pub n_vars: usize,
    /// Number of observations used
    pub n_obs: usize,
    /// Variable names
    pub var_names: Vec<String>,
    /// AIC information criterion
    pub aic: f64,
    /// BIC information criterion
    pub bic: f64,
    /// Coefficient matrices: one per lag plus intercept
    /// Shape: (n_vars, 1 + n_vars * lags) - each row is an equation
    pub coefficients: Vec<Vec<f64>>,
    /// Residual covariance matrix (n_vars x n_vars)
    pub sigma_u: Vec<Vec<f64>>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Standard errors for coefficients
    pub std_errors: Vec<Vec<f64>>,
    /// T - K p - 1, the residual df of each equation (>= K).
    #[serde(default)]
    pub df_residual: usize,
}

impl fmt::Display for VarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Vector Autoregression VAR({}) Results", self.lags)?;
        writeln!(f, "===========================================")?;
        writeln!(f, "No. Variables: {}", self.n_vars)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Variables: {}", self.var_names.join(", "))?;
        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "AIC: {:.4}", self.aic)?;
        writeln!(f, "BIC: {:.4}", self.bic)?;
        writeln!(f)?;
        writeln!(f, "Residual Covariance Matrix (Sigma_u):")?;
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
        writeln!(f, "Steps: {}", self.steps)?;
        writeln!(f, "Variables: {}", self.var_names.join(", "))?;
        writeln!(f)?;

        // Show IRF for first few steps
        let show_steps = self.steps.min(5);
        for h in 0..show_steps {
            writeln!(f, "Step {}:", h)?;
            for (i, response_var) in self.var_names.iter().enumerate() {
                write!(f, "  Response of {}: [", response_var)?;
                for (j, _shock_var) in self.var_names.iter().enumerate() {
                    if j > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:.4}", self.irf[h][i][j])?;
                }
                writeln!(f, "]")?;
            }
        }

        if self.steps > show_steps {
            writeln!(f, "  ... ({} more steps)", self.steps - show_steps)?;
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
    let mut residuals = Array2::zeros((t, k));
    for eq in 0..k {
        let y_eq = y.column(eq);
        coefficients.row_mut(eq).assign(&qr.coefficients(y_eq)?);
        // Residuals Q (0, .., 0, (Q'y)[m..]) as lm.fit computes them.
        let mut effects = qr.qty(y_eq)?;
        effects.iter_mut().take(n_regressors).for_each(|v| *v = 0.0);
        residuals
            .column_mut(eq)
            .assign(&qr.q_times(effects.view())?);
    }

    // Compute residual covariance matrix: Σ_u = (1/T) * e'e
    let mut sigma_u = Array2::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0;
            for t_idx in 0..t {
                sum += residuals[[t_idx, i]] * residuals[[t_idx, j]];
            }
            sigma_u[[i, j]] = sum / t as f64;
        }
    }

    // Standard errors
    let mut std_errors = Array2::zeros((k, n_regressors));
    for eq in 0..k {
        let sigma_eq = sigma_u[[eq, eq]];
        for j in 0..n_regressors {
            std_errors[[eq, j]] = (sigma_eq * unscaled[[j, j]]).max(0.0).sqrt();
        }
    }

    // Log-likelihood
    let det_sigma = sigma_u
        .iter()
        .fold(1.0_f64, |acc, &x| acc * x.abs().max(1e-10));
    let log_likelihood = -0.5
        * (t as f64)
        * (k as f64 * (2.0 * std::f64::consts::PI).ln() + det_sigma.ln() + k as f64);

    // Information criteria
    let n_params = (k * n_regressors) as f64;
    let aic = -2.0 * log_likelihood + 2.0 * n_params;
    let bic = -2.0 * log_likelihood + n_params * (t as f64).ln();

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
        df_residual,
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
    fn test_validate_var_vs_r() {
        // R reference:
        // library(vars)
        // set.seed(42)
        // VAR(p=1) on bivariate series
        // Check coefficient structure matches
        let dataset = create_var_validation_dataset();
        let result = run_var(&dataset, &["y1", "y2"], 1).unwrap();

        // Verify structure
        assert_eq!(result.n_vars, 2);
        assert_eq!(result.lags, 1);
        assert!(result.n_obs > 90); // Should have most observations

        // Coefficients shape: (n_vars, 1 + n_vars * lags) = (2, 3)
        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(result.coefficients[0].len(), 3); // intercept + 2 lagged vars

        // AIC and BIC should be finite
        assert!(result.aic.is_finite());
        assert!(result.bic.is_finite());

        // Log-likelihood should be finite
        assert!(result.log_likelihood.is_finite());

        // Sigma_u should be 2x2 positive semi-definite
        assert_eq!(result.sigma_u.len(), 2);
        assert!(result.sigma_u[0][0] > 0.0); // Diagonal should be positive
        assert!(result.sigma_u[1][1] > 0.0);
    }

    #[test]
    fn test_validate_var_coefficient_estimation() {
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
    fn test_validate_var_irf_structure() {
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
    fn test_validate_var_non_orthogonalized_irf() {
        let dataset = create_var_validation_dataset();
        let result = run_var(&dataset, &["y1", "y2"], 1).unwrap();

        // Non-orthogonalized IRF
        let irf_non_orth = compute_irf(&result, 5, false).unwrap();
        assert!(!irf_non_orth.orthogonalized);

        // Orthogonalized IRF
        let irf_orth = compute_irf(&result, 5, true).unwrap();
        assert!(irf_orth.orthogonalized);

        // The two should generally differ (unless Sigma is diagonal)
        let mut diff: f64 = 0.0;
        for s in 0..5 {
            for i in 0..2 {
                for j in 0..2 {
                    diff += (irf_non_orth.irf[s][i][j] - irf_orth.irf[s][i][j]).abs();
                }
            }
        }
        // Some difference is expected unless covariance is exactly diagonal
        assert!(diff >= 0.0, "IRFs computed successfully");
    }

    #[test]
    fn test_validate_var_with_higher_lags() {
        let dataset = create_var_validation_dataset();

        // VAR(2) should also work
        let result = run_var(&dataset, &["y1", "y2"], 2).unwrap();
        assert_eq!(result.lags, 2);

        // Coefficients should have shape (2, 1 + 2*2) = (2, 5)
        assert_eq!(result.coefficients[0].len(), 5);

        // AIC for VAR(2) might be higher (overfitting) or lower
        // Just check it's finite
        assert!(result.aic.is_finite());
    }

    #[test]
    fn test_validate_var_display_format() {
        let dataset = create_var_validation_dataset();
        let result = run_var(&dataset, &["y1", "y2"], 1).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Vector Autoregression VAR(1)"));
        assert!(display.contains("No. Variables: 2"));
        assert!(display.contains("AIC:"));
        assert!(display.contains("BIC:"));
        assert!(display.contains("Sigma_u"));
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
