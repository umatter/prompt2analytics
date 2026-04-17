//! Shared traits for econometric estimators.
//!
//! This module provides the `LinearEstimator` trait that unifies the interface
//! across OLS, panel data, IV, and other linear estimators.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Significance level indicators for p-values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignificanceLevel {
    /// p > 0.10
    NotSignificant,
    /// p < 0.10 (†)
    TenPercent,
    /// p < 0.05 (*)
    FivePercent,
    /// p < 0.01 (**)
    OnePercent,
    /// p < 0.001 (***)
    TenthPercent,
}

impl SignificanceLevel {
    /// Determine significance level from a p-value.
    pub fn from_p_value(p: f64) -> Self {
        if p < 0.001 {
            Self::TenthPercent
        } else if p < 0.01 {
            Self::OnePercent
        } else if p < 0.05 {
            Self::FivePercent
        } else if p < 0.10 {
            Self::TenPercent
        } else {
            Self::NotSignificant
        }
    }

    /// Get the star notation for this significance level.
    pub fn stars(&self) -> &'static str {
        match self {
            Self::NotSignificant => "",
            Self::TenPercent => "†",
            Self::FivePercent => "*",
            Self::OnePercent => "**",
            Self::TenthPercent => "***",
        }
    }

    /// Get a short code for this significance level (for CSV export).
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotSignificant => "",
            Self::TenPercent => "p<0.10",
            Self::FivePercent => "p<0.05",
            Self::OnePercent => "p<0.01",
            Self::TenthPercent => "p<0.001",
        }
    }
}

impl std::fmt::Display for SignificanceLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.stars())
    }
}

/// Core trait for linear regression-type estimators.
///
/// This trait provides a unified interface for accessing estimation results
/// and computing derived statistics. Implementors only need to provide the
/// core results; derived statistics are computed via default implementations.
pub trait LinearEstimator {
    /// Get the estimated coefficients.
    fn coefficients(&self) -> &Array1<f64>;

    /// Get the standard errors of the coefficients.
    fn std_errors(&self) -> &Array1<f64>;

    /// Get the residuals (y - X*beta).
    fn residuals(&self) -> &Array1<f64>;

    /// Get the variance-covariance matrix of the coefficients.
    fn vcov_matrix(&self) -> &Array2<f64>;

    /// Get the names of the variables (coefficients).
    fn variable_names(&self) -> &[String];

    /// Get the degrees of freedom for t-tests.
    fn degrees_of_freedom(&self) -> usize;

    /// Get the number of observations.
    fn n_obs(&self) -> usize;

    /// Compute t-statistics: beta / se(beta)
    fn t_stats(&self) -> Array1<f64> {
        let coefs = self.coefficients();
        let ses = self.std_errors();
        coefs / ses
    }

    /// Compute two-sided p-values from t-statistics.
    fn p_values(&self) -> Array1<f64> {
        let t_stats = self.t_stats();
        let df = self.degrees_of_freedom() as f64;

        if !df.is_finite() || df <= 0.0 {
            // Return NaN for invalid degrees of freedom (NaN, Inf, <= 0)
            return Array1::from_elem(t_stats.len(), f64::NAN);
        }

        let t_dist = match StudentsT::new(0.0, 1.0, df) {
            Ok(d) => d,
            Err(_) => return Array1::from_elem(t_stats.len(), f64::NAN),
        };
        t_stats.mapv(|t| 2.0 * (1.0 - t_dist.cdf(t.abs())))
    }

    /// Get significance levels for each coefficient.
    fn significance(&self) -> Vec<SignificanceLevel> {
        self.p_values()
            .iter()
            .map(|&p| SignificanceLevel::from_p_value(p))
            .collect()
    }

    /// Compute confidence intervals at the given level (e.g., 0.95 for 95%).
    fn confidence_intervals(&self, level: f64) -> Vec<(f64, f64)> {
        let coefs = self.coefficients();
        let ses = self.std_errors();
        let df = self.degrees_of_freedom() as f64;

        if !df.is_finite() || df <= 0.0 || !level.is_finite() || !(0.0 < level && level < 1.0) {
            return vec![(f64::NAN, f64::NAN); coefs.len()];
        }

        let alpha = 1.0 - level;
        let t_dist = match StudentsT::new(0.0, 1.0, df) {
            Ok(d) => d,
            Err(_) => return vec![(f64::NAN, f64::NAN); coefs.len()],
        };
        let t_crit = t_dist.inverse_cdf(1.0 - alpha / 2.0);

        coefs
            .iter()
            .zip(ses.iter())
            .map(|(&b, &se)| (b - t_crit * se, b + t_crit * se))
            .collect()
    }

    /// Compute 95% confidence intervals.
    fn confidence_intervals_95(&self) -> Vec<(f64, f64)> {
        self.confidence_intervals(0.95)
    }

    /// Compute R-squared (coefficient of determination).
    /// R² = 1 - SSR/SST where SSR = sum(residuals²), SST = sum((y - mean(y))²)
    /// Note: Default implementation returns NAN. Subclasses should override.
    fn r_squared(&self) -> f64 {
        // Default: return NAN since we need original y values to compute SST
        f64::NAN
    }

    /// Compute adjusted R-squared.
    /// Adj R² = 1 - (1 - R²) * (n - 1) / (n - k - 1)
    fn adj_r_squared(&self) -> f64 {
        let r2 = self.r_squared();
        let n = self.n_obs() as f64;
        let k = self.coefficients().len() as f64;

        if n - k - 1.0 <= 0.0 {
            return f64::NAN;
        }

        1.0 - (1.0 - r2) * (n - 1.0) / (n - k - 1.0)
    }

    /// Residual standard error: sqrt(SSR / (n - k))
    fn residual_std_error(&self) -> f64 {
        let residuals = self.residuals();
        let df = self.degrees_of_freedom() as f64;

        if df <= 0.0 {
            return f64::NAN;
        }

        let ssr: f64 = residuals.iter().map(|&r| r * r).sum();
        (ssr / df).sqrt()
    }

    /// Log-likelihood (assuming normally distributed errors).
    fn log_likelihood(&self) -> f64 {
        let residuals = self.residuals();
        let n = self.n_obs() as f64;
        let sigma2: f64 = residuals.iter().map(|&r| r * r).sum::<f64>() / n;

        if sigma2 <= 0.0 {
            return f64::NAN;
        }

        -n / 2.0 * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln())
    }

    /// Akaike Information Criterion: AIC = 2k - 2*ln(L)
    fn aic(&self) -> f64 {
        let k = self.coefficients().len() as f64;
        let ll = self.log_likelihood();
        2.0 * k - 2.0 * ll
    }

    /// Bayesian Information Criterion: BIC = k*ln(n) - 2*ln(L)
    fn bic(&self) -> f64 {
        let k = self.coefficients().len() as f64;
        let n = self.n_obs() as f64;
        let ll = self.log_likelihood();
        k * n.ln() - 2.0 * ll
    }
}

/// Compute p-value from t-statistic and degrees of freedom.
pub fn t_test_p_value(t_stat: f64, df: f64) -> f64 {
    // Handle invalid inputs (NaN, Inf, <= 0 df)
    if !df.is_finite() || df <= 0.0 || t_stat.is_nan() {
        return f64::NAN;
    }
    // For very large |t|, p-value is essentially 0
    if t_stat.is_infinite() || t_stat.abs() > 1e10 {
        return 0.0;
    }
    let t_dist = match StudentsT::new(0.0, 1.0, df) {
        Ok(d) => d,
        Err(_) => return f64::NAN,
    };
    2.0 * (1.0 - t_dist.cdf(t_stat.abs()))
}

/// Compute p-value from chi-squared statistic.
pub fn chi_squared_p_value(chi2: f64, df: f64) -> f64 {
    // Handle invalid inputs (NaN, Inf, <= 0 df, negative chi2)
    if !df.is_finite() || df <= 0.0 || chi2.is_nan() || chi2 < 0.0 {
        return f64::NAN;
    }
    // For very large chi2, p-value is essentially 0
    if chi2.is_infinite() || chi2 > 1e10 {
        return 0.0;
    }
    use statrs::distribution::ChiSquared;
    let chi2_dist = match ChiSquared::new(df) {
        Ok(d) => d,
        Err(_) => return f64::NAN,
    };
    1.0 - chi2_dist.cdf(chi2)
}

/// Compute p-value from F-statistic.
pub fn f_test_p_value(f_stat: f64, df1: f64, df2: f64) -> f64 {
    // Handle invalid inputs (NaN, Inf, <= 0 df, negative F)
    if !df1.is_finite()
        || df1 <= 0.0
        || !df2.is_finite()
        || df2 <= 0.0
        || f_stat.is_nan()
        || f_stat < 0.0
    {
        return f64::NAN;
    }
    // For very large F, p-value is essentially 0
    if f_stat.is_infinite() || f_stat > 1e10 {
        return 0.0;
    }
    use statrs::distribution::FisherSnedecor;
    let f_dist = match FisherSnedecor::new(df1, df2) {
        Ok(d) => d,
        Err(_) => return f64::NAN,
    };
    1.0 - f_dist.cdf(f_stat)
}

/// Critical value for t-distribution at given significance level.
pub fn t_critical(alpha: f64, df: f64) -> f64 {
    if !df.is_finite() || df <= 0.0 || !alpha.is_finite() || !(0.0 < alpha && alpha < 1.0) {
        return f64::NAN;
    }
    let t_dist = match StudentsT::new(0.0, 1.0, df) {
        Ok(d) => d,
        Err(_) => return f64::NAN,
    };
    t_dist.inverse_cdf(1.0 - alpha / 2.0)
}

/// Normal CDF (for Probit)
pub fn normal_cdf(x: f64) -> f64 {
    use statrs::distribution::Normal;
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.cdf(x)
}

/// Normal PDF (for Probit)
pub fn normal_pdf(x: f64) -> f64 {
    use statrs::distribution::Continuous;
    use statrs::distribution::Normal;
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.pdf(x)
}

/// Logistic CDF (for Logit)
pub fn logistic_cdf(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Logistic PDF (for Logit)
pub fn logistic_pdf(x: f64) -> f64 {
    let p = logistic_cdf(x);
    p * (1.0 - p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_significance_level() {
        assert_eq!(
            SignificanceLevel::from_p_value(0.0001),
            SignificanceLevel::TenthPercent
        );
        assert_eq!(
            SignificanceLevel::from_p_value(0.005),
            SignificanceLevel::OnePercent
        );
        assert_eq!(
            SignificanceLevel::from_p_value(0.03),
            SignificanceLevel::FivePercent
        );
        assert_eq!(
            SignificanceLevel::from_p_value(0.08),
            SignificanceLevel::TenPercent
        );
        assert_eq!(
            SignificanceLevel::from_p_value(0.15),
            SignificanceLevel::NotSignificant
        );
    }

    #[test]
    fn test_stars() {
        assert_eq!(SignificanceLevel::TenthPercent.stars(), "***");
        assert_eq!(SignificanceLevel::OnePercent.stars(), "**");
        assert_eq!(SignificanceLevel::FivePercent.stars(), "*");
        assert_eq!(SignificanceLevel::TenPercent.stars(), "†");
        assert_eq!(SignificanceLevel::NotSignificant.stars(), "");
    }

    #[test]
    fn test_t_test_p_value() {
        // t = 0 should give p = 1
        let p = t_test_p_value(0.0, 10.0);
        assert!((p - 1.0).abs() < 1e-10);

        // Large t should give small p
        let p = t_test_p_value(10.0, 100.0);
        assert!(p < 0.001);
    }

    #[test]
    fn test_p_value_helpers_survive_nonfinite_df() {
        // Regression guard: StudentsT::new used to panic here on NaN/Inf df.
        assert!(t_test_p_value(1.5, f64::NAN).is_nan());
        assert!(t_test_p_value(1.5, f64::INFINITY).is_nan());
        assert!(t_test_p_value(1.5, f64::NEG_INFINITY).is_nan());
        assert!(t_test_p_value(1.5, -1.0).is_nan());

        assert!(chi_squared_p_value(2.0, f64::NAN).is_nan());
        assert!(chi_squared_p_value(2.0, f64::INFINITY).is_nan());

        assert!(f_test_p_value(2.0, f64::NAN, 10.0).is_nan());
        assert!(f_test_p_value(2.0, 10.0, f64::INFINITY).is_nan());

        assert!(t_critical(0.05, f64::NAN).is_nan());
        assert!(t_critical(0.05, f64::INFINITY).is_nan());
        // invalid alpha
        assert!(t_critical(0.0, 10.0).is_nan());
        assert!(t_critical(1.0, 10.0).is_nan());
        assert!(t_critical(f64::NAN, 10.0).is_nan());
    }

    #[test]
    fn test_logistic_cdf() {
        assert!((logistic_cdf(0.0) - 0.5).abs() < 1e-10);
        assert!(logistic_cdf(10.0) > 0.99);
        assert!(logistic_cdf(-10.0) < 0.01);
    }

    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-10);
        assert!(normal_cdf(3.0) > 0.99);
        assert!(normal_cdf(-3.0) < 0.01);
    }
}
