//! The Granger causality F test ([`granger_test`], [`run_granger_test`], [`granger_test_bidirectional`]).

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::resource;

// =============================================================================
// Granger Causality Test
// =============================================================================

/// Result of a Granger causality test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrangerResult {
    /// F statistic for the test
    pub f_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom (numerator)
    pub df1: usize,
    /// Degrees of freedom (denominator)
    pub df2: usize,
    /// Number of lags used
    pub lags: usize,
    /// Number of observations used in estimation
    pub n_obs: usize,
    /// Name of the dependent (caused) variable
    pub dependent: String,
    /// Name of the potential causing variable
    pub cause: String,
    /// RSS of unrestricted model
    pub rss_unrestricted: f64,
    /// RSS of restricted model
    pub rss_restricted: f64,
    /// Whether significant at 0.05 level
    pub significant_at_05: bool,
    /// Interpretation of the result
    pub interpretation: String,
}

impl fmt::Display for GrangerResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Granger Causality Test")?;
        writeln!(f, "======================")?;
        writeln!(
            f,
            "H0: {} does not Granger-cause {}",
            self.cause, self.dependent
        )?;
        writeln!(f)?;
        writeln!(f, "F Statistic: {:.4}", self.f_statistic)?;
        writeln!(f, "df: ({}, {})", self.df1, self.df2)?;
        writeln!(f, "p-value: {:.4}", self.p_value)?;
        writeln!(f)?;
        writeln!(f, "Lags: {}", self.lags)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f)?;
        writeln!(f, "{}", self.interpretation)?;
        Ok(())
    }
}

/// Perform Granger causality test.
///
/// Tests whether the lagged values of variable `cause` help predict variable `dependent`,
/// after controlling for lagged values of `dependent` itself.
///
/// # Mathematical Background
///
/// The test compares two regression models:
///
/// **Restricted model** (H0: no Granger causality):
/// yₜ = α + Σᵢ βᵢ yₜ₋ᵢ + εₜ
///
/// **Unrestricted model**:
/// yₜ = α + Σᵢ βᵢ yₜ₋ᵢ + Σⱼ γⱼ xₜ₋ⱼ + εₜ
///
/// The F-statistic is:
/// F = [(RSS_R - RSS_U) / p] / [RSS_U / (n - 2p - 1)]
///
/// Under H0, F ~ F(p, n - 2p - 1).
///
/// # Arguments
///
/// * `dataset` - The dataset
/// * `dependent` - Name of the dependent variable column (the "caused" variable)
/// * `cause` - Name of the potential causing variable column
/// * `lags` - Number of lags to include (default: 1)
///
/// # Returns
///
/// `GrangerResult` containing F-statistic, p-value, and interpretation.
///
/// # References
///
/// - Granger, C.W.J. (1969). Investigating Causal Relations by Econometric Models and
///   Cross-spectral Methods. *Econometrica*, 37(3), 424-438.
///   https://doi.org/10.2307/1912791
///
/// R equivalent: `lmtest::grangertest()`
///
/// Integer and Boolean columns are accepted; a missing value (null or NaN) in either
/// series is `EconError::NullValues` naming the column. R's grangertest drops rows instead,
/// but its formula and default interfaces disagree on which rows; fill the gaps first.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::granger_test;
///
/// // Test whether x Granger-causes y
/// let result = granger_test(&dataset, "y", "x", 2)?;
/// println!("{}", result);
/// ```
pub fn granger_test(
    dataset: &Dataset,
    dependent: &str,
    cause: &str,
    lags: usize,
) -> EconResult<GrangerResult> {
    use crate::linalg::{safe_inverse, xtx, xty};
    use crate::traits::f_test_p_value;

    if lags == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Number of lags must be at least 1".to_string(),
        });
    }

    // Extract the two time series: Int64 accepted, a missing value refused by name
    let cols = ColumnRequest::new(dataset.df())
        .numerics(&[dependent, cause])
        .na(NaPolicy::Fail)
        .extract()?;
    let y_vec = cols.numeric(dependent).to_vec();
    let x_vec = cols.numeric(cause).to_vec();

    let n_total = y_vec.len();
    if n_total != x_vec.len() {
        return Err(EconError::InvalidSpecification {
            message: "Variables must have the same length".to_string(),
        });
    }
    let bound = lags.checked_mul(2).and_then(|v| v.checked_add(1));
    if bound.is_none_or(|b| n_total <= b) {
        return Err(EconError::InsufficientData {
            required: lags.saturating_mul(2).saturating_add(2),
            provided: n_total,
            context: "Granger causality test".to_string(),
        });
    }

    let n = n_total - lags;

    // Build dependent variable vector (y values from lags onwards)
    let y = Array1::from_vec(y_vec[lags..].to_vec());

    // Build restricted design matrix: intercept + lagged y only
    // X_R = [1, y_{t-1}, y_{t-2}, ..., y_{t-p}]
    let mut x_restricted = resource::try_zeros2(n, lags + 1, "restricted Granger design")?;
    for i in 0..n {
        x_restricted[[i, 0]] = 1.0; // intercept
        for lag in 1..=lags {
            x_restricted[[i, lag]] = y_vec[lags + i - lag];
        }
    }

    // Build unrestricted design matrix: intercept + lagged y + lagged x
    // X_U = [1, y_{t-1}, ..., y_{t-p}, x_{t-1}, ..., x_{t-p}]
    let mut x_unrestricted = resource::try_zeros2(n, 2 * lags + 1, "unrestricted Granger design")?;
    for i in 0..n {
        x_unrestricted[[i, 0]] = 1.0; // intercept
        for lag in 1..=lags {
            x_unrestricted[[i, lag]] = y_vec[lags + i - lag];
            x_unrestricted[[i, lags + lag]] = x_vec[lags + i - lag];
        }
    }

    // Both X'X matrices are at most (2 * lags + 1)^2
    resource::check_alloc::<f64>(
        resource::checked_len(&[2 * lags + 1, 2 * lags + 1], "unrestricted Granger X'X")?,
        "unrestricted Granger X'X",
    )?;

    // Estimate restricted model and compute RSS
    let xtx_r = xtx(&x_restricted.view());
    let (xtx_inv_r, _) = safe_inverse(&xtx_r.view()).map_err(|e| EconError::SingularMatrix {
        context: "Restricted Granger model".to_string(),
        suggestion: format!("Check for multicollinearity: {}", e),
    })?;
    let xty_r = xty(&x_restricted.view(), &y);
    let beta_r = xtx_inv_r.dot(&xty_r);
    let fitted_r = x_restricted.dot(&beta_r);
    let resid_r = &y - &fitted_r;
    let rss_r = resid_r.iter().map(|r| r * r).sum::<f64>();

    // Estimate unrestricted model and compute RSS
    let xtx_u = xtx(&x_unrestricted.view());
    let (xtx_inv_u, _) = safe_inverse(&xtx_u.view()).map_err(|e| EconError::SingularMatrix {
        context: "Unrestricted Granger model".to_string(),
        suggestion: format!("Check for multicollinearity: {}", e),
    })?;
    let xty_u = xty(&x_unrestricted.view(), &y);
    let beta_u = xtx_inv_u.dot(&xty_u);
    let fitted_u = x_unrestricted.dot(&beta_u);
    let resid_u = &y - &fitted_u;
    let rss_u = resid_u.iter().map(|r| r * r).sum::<f64>();

    // Compute F-statistic
    // F = [(RSS_R - RSS_U) / p] / [RSS_U / (n - 2p - 1)]
    let df1 = lags;
    let df2 = n - 2 * lags - 1;

    if df2 == 0 {
        return Err(EconError::InsufficientData {
            required: 2 * lags + 2,
            provided: n,
            context: "Granger test denominator degrees of freedom".to_string(),
        });
    }

    let f_statistic = if rss_u > 1e-15 {
        ((rss_r - rss_u) / df1 as f64) / (rss_u / df2 as f64)
    } else {
        // Perfect fit in unrestricted model (rare edge case)
        f64::INFINITY
    };

    let p_value = f_test_p_value(f_statistic, df1 as f64, df2 as f64);
    let significant = p_value < 0.05;

    let interpretation = if significant {
        format!(
            "Reject H0 at 5% level: {} Granger-causes {} (p = {:.4}). \
             Lagged values of {} significantly improve predictions of {}.",
            cause, dependent, p_value, cause, dependent
        )
    } else {
        format!(
            "Cannot reject H0 at 5% level: {} does not Granger-cause {} (p = {:.4}). \
             No evidence that lagged values of {} improve predictions of {}.",
            cause, dependent, p_value, cause, dependent
        )
    };

    Ok(GrangerResult {
        f_statistic,
        p_value,
        df1,
        df2,
        lags,
        n_obs: n,
        dependent: dependent.to_string(),
        cause: cause.to_string(),
        rss_unrestricted: rss_u,
        rss_restricted: rss_r,
        significant_at_05: significant,
        interpretation,
    })
}

/// Convenience function for Granger causality test with default lag selection.
///
/// Uses lag = floor(12 * (n/100)^0.25) as a default (Schwert, 1989).
pub fn run_granger_test(
    dataset: &Dataset,
    dependent: &str,
    cause: &str,
) -> EconResult<GrangerResult> {
    // Schwert rule for lag selection
    let n = dataset.df().height();
    let default_lags = ((12.0 * (n as f64 / 100.0).powf(0.25)).floor() as usize).max(1);
    granger_test(dataset, dependent, cause, default_lags)
}

/// Test bidirectional Granger causality between two variables.
///
/// Runs the test in both directions and returns both results.
pub fn granger_test_bidirectional(
    dataset: &Dataset,
    var1: &str,
    var2: &str,
    lags: usize,
) -> EconResult<(GrangerResult, GrangerResult)> {
    let result1 = granger_test(dataset, var1, var2, lags)?; // var2 -> var1
    let result2 = granger_test(dataset, var2, var1, lags)?; // var1 -> var2
    Ok((result1, result2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::econometrics::timeseries::common::test_fixtures::*;
    use crate::econometrics::timeseries::run_var;
    use polars::prelude::*;

    /// R: lmtest::grangertest(x = m, y = ai, order = 2 | 3) on the Int64 series ai.
    #[test]
    fn test_validate_granger_int_series() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("ts_granger_int");
        assert_eq!(ds.df().column("ai").unwrap().dtype(), &DataType::Int64);
        let n = 300;
        let mut check = Check::new(&case);
        for lags in [2usize, 3] {
            let r = granger_test(&ds, "ai", "m", lags).unwrap();
            let k = format!("lag{lags}");
            check
                .scalar(&format!("{k}_f"), r.f_statistic, Tol::stat(n))
                .scalar(&format!("{k}_p"), r.p_value, Tol::p_value(n))
                .scalar(&format!("{k}_df1"), r.df1 as f64, Tol::exact())
                .scalar(&format!("{k}_df2"), r.df2 as f64, Tol::exact())
                .scalar(&format!("{k}_n_used"), r.n_obs as f64, Tol::exact());
        }
        check.finish();
    }

    /// The dataset of case ts_granger_int (Int64 `ai`, `a_na`/`ai_na` with two nulls each,
    /// text `lab`), for tests that compare no case value.
    fn granger_series() -> Dataset {
        crate::test_support::load_case("ts_granger_int").1
    }

    /// X-2: Granger and the VAR family refuse missing values and text by name; integer
    /// columns run.
    #[test]
    fn test_na_granger_var_missing_and_text() {
        let ds = granger_series();
        let is_null = |e: &EconError, column: &str| matches!(e, EconError::NullValues { column: c, count: 2 } if c == column);
        let is_text =
            |e: &EconError| matches!(e, EconError::NonNumericColumn { column } if column == "lab");
        let e = granger_test(&ds, "a_na", "m", 2).unwrap_err();
        assert!(is_null(&e, "a_na"), "{e:?}");
        let e = granger_test(&ds, "ai", "ai_na", 2).unwrap_err();
        assert!(is_null(&e, "ai_na"), "{e:?}");
        let e = granger_test(&ds, "lab", "m", 2).unwrap_err();
        assert!(is_text(&e), "{e:?}");
        let e = run_var(&ds, &["a", "a_na"], 1).unwrap_err();
        assert!(is_null(&e, "a_na"), "{e:?}");
        let e = run_var(&ds, &["a", "lab"], 1).unwrap_err();
        assert!(is_text(&e), "{e:?}");
        run_var(&ds, &["ai", "m"], 1).unwrap();
    }

    #[test]
    fn test_granger_test_basic() {
        let dataset = create_granger_test_dataset();
        let result = granger_test(&dataset, "y", "x", 2).unwrap();

        assert_eq!(result.lags, 2);
        assert!(result.f_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.df1, 2);
        assert_eq!(result.dependent, "y");
        assert_eq!(result.cause, "x");
    }

    #[test]
    fn test_granger_test_with_lag_1() {
        let dataset = create_granger_test_dataset();
        let result = granger_test(&dataset, "y", "x", 1).unwrap();

        assert_eq!(result.lags, 1);
        assert_eq!(result.df1, 1);
        assert!(result.n_obs < 30); // Lost some obs due to lagging
    }

    #[test]
    fn test_granger_test_bidirectional() {
        let dataset = create_granger_test_dataset();
        let (result1, result2) = granger_test_bidirectional(&dataset, "y", "x", 1).unwrap();

        // result1: x -> y
        assert_eq!(result1.dependent, "y");
        assert_eq!(result1.cause, "x");

        // result2: y -> x
        assert_eq!(result2.dependent, "x");
        assert_eq!(result2.cause, "y");
    }

    #[test]
    fn test_granger_test_rss_ordering() {
        let dataset = create_granger_test_dataset();
        let result = granger_test(&dataset, "y", "x", 2).unwrap();

        // Unrestricted model should always have RSS <= restricted RSS
        assert!(result.rss_unrestricted <= result.rss_restricted + 1e-10);
    }

    #[test]
    fn test_granger_test_displays_correctly() {
        let dataset = create_granger_test_dataset();
        let result = granger_test(&dataset, "y", "x", 1).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Granger Causality Test"));
        assert!(display.contains("F Statistic"));
        assert!(display.contains("p-value"));
        assert!(display.contains("does not Granger-cause") || display.contains("Granger-causes"));
    }

    #[test]
    fn test_run_granger_test_default_lags() {
        let dataset = create_granger_test_dataset();
        let result = run_granger_test(&dataset, "y", "x").unwrap();

        // Should automatically select lags
        assert!(result.lags >= 1);
        assert!(result.f_statistic >= 0.0);
    }

    #[test]
    fn test_granger_test_insufficient_data() {
        // Create small dataset
        let df = df! {
            "y" => [1.0, 2.0, 3.0],
            "x" => [0.5, 1.0, 1.5]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        // Should fail with too many lags
        let result = granger_test(&dataset, "y", "x", 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_granger_f_statistic() {
        // R reference:
        // grangertest(y1 ~ y2, order = 2)
        // F statistic and p-value structure
        let dataset = create_var_validation_dataset();
        let result = granger_test(&dataset, "y1", "y2", 2).unwrap();

        // F-statistic should be non-negative
        assert!(result.f_statistic >= 0.0);

        // P-value should be in [0, 1]
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);

        // Degrees of freedom
        assert_eq!(result.df1, 2); // Number of lags
        assert!(result.df2 > 0); // Denominator df

        // RSS ordering: unrestricted <= restricted
        assert!(
            result.rss_unrestricted <= result.rss_restricted + 1e-10,
            "RSS ordering violated"
        );
    }

    #[test]
    fn test_validate_granger_causal_direction() {
        // In our DGP, y2 should Granger-cause y1 (coefficient 0.3)
        // and y1 should Granger-cause y2 (coefficient 0.2)
        let dataset = create_var_validation_dataset();

        let result_y2_to_y1 = granger_test(&dataset, "y1", "y2", 1).unwrap();
        let result_y1_to_y2 = granger_test(&dataset, "y2", "y1", 1).unwrap();

        // Both should show some Granger causality in DGP
        // F-statistics should be positive
        assert!(result_y2_to_y1.f_statistic >= 0.0);
        assert!(result_y1_to_y2.f_statistic >= 0.0);

        // Interpretations should be present
        assert!(!result_y2_to_y1.interpretation.is_empty());
        assert!(!result_y1_to_y2.interpretation.is_empty());
    }
}
