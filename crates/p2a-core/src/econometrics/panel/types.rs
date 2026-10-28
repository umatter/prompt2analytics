//! Core types for panel data estimation.
//!
//! Contains `PanelResult` and `PanelMethod` used across panel estimators, and the
//! options of the linear panel estimators: [`PanelConfig`], [`PanelVcov`] and
//! [`ClusterAdjustment`].

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;
use std::str::FromStr;

use crate::errors::EconError;
use crate::traits::estimator::{LinearEstimator, SignificanceLevel, t_critical};

/// Small-sample factor of the entity-clustered (Arellano) covariance
/// `c · B M B`, with `n` observations, `k` coefficients and `G` entities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ClusterAdjustment {
    /// `c = 1` (plm `vcovHC(method = "arellano", type = "HC0")`).
    HC0,
    /// `c = n / (n - k)` (plm `type = "HC1"`).
    HC1,
    /// `c = G/(G - 1) · (n - 1)/(n - k)`, Stata's CR1 factor (plm `type = "sss"`).
    #[default]
    Stata,
}

/// Coefficient covariance of the linear panel estimators
/// ([`run_fixed_effects_with`](crate::econometrics::run_fixed_effects_with),
/// [`run_random_effects_with`](crate::econometrics::run_random_effects_with)).
///
/// Parsed from `classical`, `cluster` (= `Arellano(Stata)`), `cluster-hc0` and
/// `cluster-hc1`, ignoring case and surrounding white space:
///
/// ```
/// use p2a_core::econometrics::PanelVcov;
/// let v: PanelVcov = "cluster".parse().unwrap();
/// assert_eq!(v.to_string(), "clustered by entity (Arellano, CR1/sss)");
/// assert!("robust".parse::<PanelVcov>().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PanelVcov {
    /// The classical covariance plm reports by default: `s² (X'X)⁻¹` of the
    /// (within- or quasi-) demeaned model.
    #[default]
    Classical,
    /// Entity-clustered sandwich (Arellano 1987), as plm
    /// `vcovHC(method = "arellano")`.
    Arellano(ClusterAdjustment),
}

impl FromStr for PanelVcov {
    type Err = EconError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "classical" => Ok(PanelVcov::Classical),
            "cluster" => Ok(PanelVcov::Arellano(ClusterAdjustment::Stata)),
            "cluster-hc0" => Ok(PanelVcov::Arellano(ClusterAdjustment::HC0)),
            "cluster-hc1" => Ok(PanelVcov::Arellano(ClusterAdjustment::HC1)),
            _ => Err(EconError::InvalidSpecification {
                message: format!(
                    "unknown standard-error type '{s}'; expected one of: classical, cluster, \
                     cluster-hc0, cluster-hc1"
                ),
            }),
        }
    }
}

impl fmt::Display for PanelVcov {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PanelVcov::Classical => write!(f, "classical"),
            PanelVcov::Arellano(ClusterAdjustment::Stata) => {
                write!(f, "clustered by entity (Arellano, CR1/sss)")
            }
            PanelVcov::Arellano(ClusterAdjustment::HC0) => {
                write!(f, "clustered by entity (Arellano, HC0)")
            }
            PanelVcov::Arellano(ClusterAdjustment::HC1) => {
                write!(f, "clustered by entity (Arellano, HC1)")
            }
        }
    }
}

/// Reference distribution of a result's coefficient statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TestDistribution {
    /// Student t with `df` degrees of freedom (fixed effects, as plm).
    #[default]
    StudentT,
    /// Standard normal (random effects, as plm's `summary` for random models).
    Normal,
}

/// Options of the linear panel estimators.
#[derive(Debug, Clone, Default)]
pub struct PanelConfig {
    /// Coefficient covariance (default: classical, as plm).
    pub vcov: PanelVcov,
}

/// Result from a panel data estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelResult {
    /// Estimation method used
    pub method: PanelMethod,
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names (including intercept if present)
    pub variables: Vec<String>,
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors, from the covariance named by `vcov_type`
    pub std_errors: Vec<f64>,
    /// Coefficient / standard error: a t statistic for fixed effects, a z
    /// statistic for random effects (see `test_distribution`)
    pub t_stats: Vec<f64>,
    /// Two-sided p-values from `test_distribution`
    pub p_values: Vec<f64>,
    /// R-squared. Fixed effects: the within R², `1 - SSR/TSS` of the demeaned
    /// model. Random effects: `cor(y*, X*b)²` of the quasi-demeaned model, as
    /// plm's `r.squared`.
    pub r_squared: f64,
    /// Adjusted R-squared, plm `r.squared(dfcor = TRUE)`:
    /// `1 - (1 - R²)(n - 1)/df` (fixed effects: within).
    pub adj_r_squared: f64,
    /// F test of the slopes with the covariance of `vcov_type`, on the degrees
    /// of freedom [`PanelResult::f_df`] returns. Fixed effects: the classical
    /// within F test, or under the Arellano covariance the Wald F `W/K` (plm's
    /// `summary(fe, vcov = ...)`). Random effects: the F form `W/K` of the Wald
    /// test in `wald_chi2` (plm `pwaldtest(test = "F")`). Under the Arellano
    /// covariance the second degrees of freedom are `G - 1`, as plm's `df2adj`
    /// for a cluster vcov. NaN when the Wald statistic is not computed.
    pub f_stat: f64,
    /// p-value of `f_stat` on [`PanelResult::f_df`] degrees of freedom
    pub f_p_value: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of groups (entities)
    pub n_groups: usize,
    /// Residual degrees of freedom: `n - N - K` (fixed effects), `n - K - 1`
    /// (random effects)
    pub df: usize,
    /// Entity variable name
    pub entity_var: String,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,
    /// Standard deviation of the individual effects (random effects; Swamy-Arora)
    pub sigma_u: Option<f64>,
    /// Standard deviation of the idiosyncratic error (from the within regression)
    pub sigma_e: Option<f64>,
    /// Quasi-demeaning factor of random effects: θ on a balanced panel, the
    /// observation-weighted mean `Σ T_i θ_i / n` on an unbalanced one (plm's
    /// mean theta; the θ_i are in `theta_by_group`)
    pub theta: Option<f64>,
    /// Covariance the standard errors come from.
    #[serde(default)]
    pub vcov_type: PanelVcov,
    /// Distribution of `t_stats` and `p_values`: Student t (fixed effects) or
    /// standard normal (random effects).
    #[serde(default)]
    pub test_distribution: TestDistribution,
    /// Random effects: Wald chi-squared test of the slopes, `b_s' V_ss⁻¹ b_s`
    /// with the covariance of `vcov_type` (plm's `summary`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wald_chi2: Option<f64>,
    /// Degrees of freedom of `wald_chi2` (the number of slopes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wald_df: Option<usize>,
    /// p-value of `wald_chi2`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wald_p_value: Option<f64>,
    /// Random effects on an unbalanced panel: θ_i for each entity, in order of
    /// first appearance in the data. Empty for balanced panels and fixed effects.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub theta_by_group: Vec<f64>,
    /// Estimation warnings (for example, a negative variance component set to 0).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,

    // ── Trait-backing storage (LinearEstimator). Kept out of the public
    //    JSON surface to avoid duplicating the Vec-typed fields above; round-
    //    trip deserialization uses `default` to fall back to empty arrays.
    /// Coefficients as `Array1`, used by `LinearEstimator::coefficients`.
    #[serde(skip, default)]
    pub coef_arr: Array1<f64>,
    /// Standard errors as `Array1`, used by `LinearEstimator::std_errors`.
    #[serde(skip, default)]
    pub se_arr: Array1<f64>,
    /// Residuals (`Array1`), used by `LinearEstimator::residuals`: the within
    /// residuals for fixed effects, the residuals of the quasi-demeaned model
    /// `y* - X* b` for random effects (plm's `residuals`).
    #[serde(skip, default)]
    pub residuals: Array1<f64>,
    /// Coefficient variance-covariance matrix of `vcov_type`. Classical: fixed
    /// effects `s² (X̃'X̃)⁻¹` of the within model, random effects
    /// `s² (X*'X*)⁻¹` of the quasi-demeaned model with `s²` its residual
    /// variance (plm's `vcov`). Arellano: the entity-clustered sandwich.
    /// Used by `LinearEstimator::vcov_matrix`.
    #[serde(skip, default)]
    pub vcov: Array2<f64>,
}

impl PanelResult {
    /// Degrees of freedom `(df1, df2)` of `f_stat`: the number of slopes, and
    /// `G - 1` under the entity-clustered (Arellano) covariance (plm
    /// `pwaldtest`'s `df2adj` for a cluster vcov) or the residual degrees of
    /// freedom `df` under the classical one.
    pub fn f_df(&self) -> (usize, usize) {
        let df1 = match self.method {
            PanelMethod::FixedEffects => self.variables.len(),
            PanelMethod::RandomEffects => self.variables.len().saturating_sub(1),
        };
        let df2 = match self.vcov_type {
            PanelVcov::Classical => self.df,
            PanelVcov::Arellano(_) => self.n_groups.saturating_sub(1),
        };
        (df1, df2)
    }
}

impl LinearEstimator for PanelResult {
    fn coefficients(&self) -> &Array1<f64> {
        &self.coef_arr
    }
    fn std_errors(&self) -> &Array1<f64> {
        &self.se_arr
    }
    fn residuals(&self) -> &Array1<f64> {
        &self.residuals
    }
    fn vcov_matrix(&self) -> &Array2<f64> {
        &self.vcov
    }
    fn variable_names(&self) -> &[String] {
        &self.variables
    }
    fn degrees_of_freedom(&self) -> usize {
        self.df
    }
    fn n_obs(&self) -> usize {
        self.n_obs
    }
    /// The `p_values` field (Student t or normal, per `test_distribution`).
    fn p_values(&self) -> Array1<f64> {
        Array1::from_vec(self.p_values.clone())
    }
    /// `b ± q·se` from the `coefficients` and `std_errors` fields, with `q` the
    /// normal quantile when `test_distribution` is `Normal` and the t(df)
    /// quantile otherwise. NaN bounds when `level` is not in (0, 1) or, under
    /// Student t, when `df == 0`.
    fn confidence_intervals(&self, level: f64) -> Vec<(f64, f64)> {
        let nan = vec![(f64::NAN, f64::NAN); self.coefficients.len()];
        if !(level.is_finite() && 0.0 < level && level < 1.0) {
            return nan;
        }
        let q = match self.test_distribution {
            TestDistribution::Normal => match Normal::new(0.0, 1.0) {
                Ok(d) => d.inverse_cdf(1.0 - (1.0 - level) / 2.0),
                Err(_) => f64::NAN,
            },
            TestDistribution::StudentT if self.df == 0 => f64::NAN,
            TestDistribution::StudentT => t_critical(1.0 - level, self.df as f64),
        };
        if q.is_nan() {
            return nan;
        }
        self.coefficients
            .iter()
            .zip(&self.std_errors)
            .map(|(&b, &se)| (b - q * se, b + q * se))
            .collect()
    }
    /// The `r_squared` field.
    fn r_squared(&self) -> f64 {
        self.r_squared
    }
    /// The `adj_r_squared` field.
    fn adj_r_squared(&self) -> f64 {
        self.adj_r_squared
    }
}

/// Panel estimation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanelMethod {
    /// Fixed Effects (within) estimator
    FixedEffects,
    /// Random Effects (GLS) estimator
    RandomEffects,
}

impl fmt::Display for PanelMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PanelMethod::FixedEffects => write!(f, "Fixed Effects"),
            PanelMethod::RandomEffects => write!(f, "Random Effects"),
        }
    }
}

impl fmt::Display for PanelResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fe = self.method == PanelMethod::FixedEffects;
        writeln!(f, "{} Panel Regression Results", self.method)?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "Entity: {}", self.entity_var)?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "No. Groups: {}", self.n_groups)?;
        writeln!(f, "Standard errors: {}", self.vcov_type)?;
        let clustered = match self.vcov_type {
            PanelVcov::Classical => "",
            PanelVcov::Arellano(_) => " (clustered)",
        };
        if fe {
            writeln!(f, "R-squared (within): {:.4}", self.r_squared)?;
            writeln!(f, "Adj. R-squared (within): {:.4}", self.adj_r_squared)?;
            if !self.f_stat.is_nan() {
                let (df1, df2) = self.f_df();
                let label = match self.vcov_type {
                    PanelVcov::Classical => "F-statistic",
                    PanelVcov::Arellano(_) => "F-statistic (Wald, clustered)",
                };
                writeln!(
                    f,
                    "{label}: {:.4} on {df1} and {df2} DF (p-value: {:.4})",
                    self.f_stat, self.f_p_value
                )?;
            }
        } else {
            writeln!(f, "R-squared: {:.4}", self.r_squared)?;
            writeln!(f, "Adj. R-squared: {:.4}", self.adj_r_squared)?;
            if let (Some(w), Some(df), Some(p)) = (self.wald_chi2, self.wald_df, self.wald_p_value)
            {
                writeln!(
                    f,
                    "Wald chi-squared{clustered}: {w:.4} on {df} DF (p-value: {p:.4})"
                )?;
            }
        }

        if let Some(sigma_u) = self.sigma_u {
            writeln!(f, "sigma_u: {:.4}", sigma_u)?;
        }
        if let Some(sigma_e) = self.sigma_e {
            writeln!(f, "sigma_e: {:.4}", sigma_e)?;
        }
        if let Some(theta) = self.theta {
            if self.theta_by_group.is_empty() {
                writeln!(f, "theta: {:.4}", theta)?;
            } else {
                let lo = self
                    .theta_by_group
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                let hi = self
                    .theta_by_group
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                writeln!(f, "theta (mean): {theta:.4}, range {lo:.4}-{hi:.4}")?;
            }
        }

        let (stat, p) = match self.test_distribution {
            TestDistribution::StudentT => ("t", "P>|t|"),
            TestDistribution::Normal => ("z", "P>|z|"),
        };
        writeln!(f)?;
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef", "Std Err", stat, p
        )?;
        writeln!(f, "{}", "-".repeat(70))?;

        for i in 0..self.variables.len() {
            writeln!(
                f,
                "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.3}{}",
                self.variables[i],
                self.coefficients[i],
                self.std_errors[i],
                self.t_stats[i],
                self.p_values[i],
                self.significance[i].stars()
            )?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        if !self.warnings.is_empty() {
            writeln!(f, "Warnings:")?;
            for w in &self.warnings {
                writeln!(f, "  - {w}")?;
            }
        }

        Ok(())
    }
}
