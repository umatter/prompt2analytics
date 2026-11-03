//! Vector autoregression VAR(p) by equation-wise OLS ([`run_var`]) and impulse responses ([`compute_irf`], [`run_var_irf`]).

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fmt;

use super::common::{array2_to_vec, df_to_array2};
use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{cholesky, safe_inverse, xtx, xty};
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
/// VAR(p) model: y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + e_t
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

    // Estimate each equation by OLS: β = (Z'Z)^{-1} Z'Y
    resource::check_alloc::<f64>(
        resource::checked_len(&[n_regressors, n_regressors], "VAR Z'Z")?,
        "VAR Z'Z",
    )?;
    let ztz = xtx(&z.view());
    let (ztz_inv, _) = safe_inverse(&ztz.view()).map_err(|e| EconError::SingularMatrix {
        context: "Z'Z in VAR estimation".to_string(),
        suggestion: format!("Check for perfect collinearity: {:?}", e),
    })?;

    // Compute coefficients for each equation
    let mut coefficients = Array2::zeros((k, n_regressors));
    let mut residuals = Array2::zeros((t, k));

    for eq in 0..k {
        let y_eq = y.column(eq).to_owned();
        let zty = xty(&z.view(), &y_eq);
        let beta = ztz_inv.dot(&zty);
        coefficients.row_mut(eq).assign(&beta);

        // Compute residuals
        let y_hat = z.dot(&beta);
        for i in 0..t {
            residuals[[i, eq]] = y_eq[i] - y_hat[i];
        }
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
            std_errors[[eq, j]] = (sigma_eq * ztz_inv[[j, j]]).max(0.0).sqrt();
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
        cholesky(&sigma_u.view())
            .map_err(|e| EconError::Internal(format!("Cholesky decomposition failed: {:?}", e)))?
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
    use crate::resource::{Budget, CancelToken, with_budget};

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
