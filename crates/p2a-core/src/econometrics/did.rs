//! Difference-in-Differences (DiD) estimation.
//!
//! Pure Rust implementation without external formula parsing.
//! Uses column-based API for simplicity.
//!
//! # Mathematical Background
//!
//! DiD estimates causal effects by comparing changes over time between
//! treatment and control groups:
//!
//! ## Basic 2×2 DiD
//!
//! For outcome Y, treatment D, and post-treatment period T:
//!
//! ATT = E[Y₁ - Y₀ | D=1] = (Ȳ_treated,post - Ȳ_treated,pre) - (Ȳ_control,post - Ȳ_control,pre)
//!
//! ## Regression Specification
//!
//! yᵢₜ = α + β₁Dᵢ + β₂Tₜ + **δ(Dᵢ × Tₜ)** + εᵢₜ
//!
//! The coefficient **δ** is the DiD estimator (ATT under parallel trends).
//!
//! The regression is solved as OLS is (normal equations on a full-rank,
//! well-conditioned design, `lm`'s pivoted QR (`dqrdc2`, tolerance 1e-7)
//! otherwise): an aliased control is `NA` (NaN) with R's "Coefficients not
//! defined because of singularities" warning, an aliased treatment × post interaction is an
//! error, and the residual degrees of freedom are n - rank. HC1 and CR1
//! standard errors use the identified columns (`sandwich::vcovHC(type = "HC1")`
//! and `vcovCL(type = "HC1")`); clustering needs at least two clusters.
//!
//! ## Key Assumptions
//!
//! 1. **Parallel trends**: Absent treatment, treated and control groups would
//!    have followed parallel outcome trajectories
//! 2. **No anticipation**: Treatment doesn't affect outcomes before implementation
//! 3. **SUTVA**: No spillovers between units
//!
//! # References
//!
//! - Ashenfelter, O., & Card, D. (1985). Using the longitudinal structure of
//!   earnings to estimate the effect of training programs. *Review of Economics
//!   and Statistics*, 67(4), 648-660. https://doi.org/10.2307/1924810
//!
//! - Card, D., & Krueger, A.B. (1994). Minimum wages and employment: A case study
//!   of the fast-food industry in New Jersey and Pennsylvania. *American Economic
//!   Review*, 84(4), 772-793. Classic DiD application.
//!
//! - Angrist, J.D., & Pischke, J.S. (2009). *Mostly Harmless Econometrics: An
//!   Empiricist's Companion*, Chapter 5. Princeton University Press.
//!
//! - Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University Press.
//!   https://mixtape.scunning.com/ Chapter on Difference-in-Differences.
//!
//! - Roth, J., Sant'Anna, P.H.C., Bilinski, A., & Poe, J. (2023). What's trending
//!   in difference-in-differences? A synthesis of the recent econometrics literature.
//!   *Journal of Econometrics*, 235(2), 2218-2244.
//!   https://doi.org/10.1016/j.jeconom.2023.03.008
//!
//! R equivalent: Manual implementation or packages like `did`, `fixest::feols()`

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::aliasing_message;
use crate::regression::ols::solve_ols;
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::{SignificanceLevel, t_test_p_value};

/// Result from a Difference-in-Differences estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiDResult {
    /// Dependent variable name
    pub dep_var: String,
    /// Treatment group variable
    pub treatment_var: String,
    /// Post-treatment period variable
    pub post_var: String,
    /// The DiD estimate (ATT - Average Treatment Effect on Treated)
    pub att: f64,
    /// Standard error of ATT estimate
    pub std_error: f64,
    /// t-statistic
    pub t_stat: f64,
    /// p-value for ATT estimate
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,
    /// R-squared
    pub r_squared: f64,
    /// Adjusted R-squared
    pub adj_r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Degrees of freedom
    pub df: usize,
    /// Control group pre-treatment mean
    pub control_pre_mean: f64,
    /// Control group post-treatment mean
    pub control_post_mean: f64,
    /// Treated group pre-treatment mean
    pub treated_pre_mean: f64,
    /// Treated group post-treatment mean
    pub treated_post_mean: f64,
    /// All coefficient estimates [intercept, treatment, post, treatment*post]
    pub coefficients: Vec<f64>,
    /// Standard errors for all coefficients
    pub std_errors: Vec<f64>,
    /// Variable names
    pub variables: Vec<String>,
    /// Control variables (if any)
    pub controls: Vec<String>,
    /// Cluster variable (if clustered SEs were used)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cluster_var: Option<String>,
    /// Number of clusters (if clustered SEs were used)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_clusters: Option<usize>,
    /// Rows dropped because of missing values (empty when none were).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
    /// Regressors not identified (aliased, as `lm`'s `NA` coefficients); their
    /// coefficients and standard errors are NaN.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliased: Vec<String>,
    /// Warnings (R's aliasing message).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl ResultNotes for DiDResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

/// `{v:>width.prec}`, or `NA` right-aligned for a NaN.
fn num_or_na(v: f64, width: usize, prec: usize) -> String {
    if v.is_nan() {
        format!("{:>width$}", "NA")
    } else {
        format!("{v:>width$.prec$}")
    }
}

impl fmt::Display for DiDResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Difference-in-Differences Estimation")?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(
            f,
            "Treatment: {}, Post: {}",
            self.treatment_var, self.post_var
        )?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        if let (Some(cvar), Some(nc)) = (&self.cluster_var, self.n_clusters) {
            writeln!(f, "Clustered SEs: {} ({} clusters)", cvar, nc)?;
        }
        writeln!(f, "R-squared: {:.4}", self.r_squared)?;
        writeln!(f, "Adj. R-squared: {:.4}", self.adj_r_squared)?;
        writeln!(f)?;

        writeln!(f, "DiD ESTIMATE (Average Treatment Effect on Treated):")?;
        writeln!(
            f,
            "  ATT = {:.4} (SE: {:.4}, t = {:.2}, p = {:.3}){}",
            self.att,
            self.std_error,
            self.t_stat,
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(f)?;

        writeln!(f, "Group Means:")?;
        writeln!(
            f,
            "  Control (Pre):  {:.4}    Control (Post): {:.4}",
            self.control_pre_mean, self.control_post_mean
        )?;
        writeln!(
            f,
            "  Treated (Pre):  {:.4}    Treated (Post): {:.4}",
            self.treated_pre_mean, self.treated_post_mean
        )?;
        writeln!(f)?;

        writeln!(f, "Full Regression Results:")?;
        writeln!(
            f,
            "{:<25} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef", "Std Err", "t", "P>|t|"
        )?;
        writeln!(f, "{}", "-".repeat(75))?;

        for i in 0..self.variables.len() {
            let (b, se) = (self.coefficients[i], self.std_errors[i]);
            let (t, p) = if b.is_nan() || se.is_nan() {
                (f64::NAN, f64::NAN)
            } else {
                let t = if se > 0.0 { b / se } else { 0.0 };
                (t, t_test_p_value(t, self.df as f64))
            };
            let stars = if p.is_nan() {
                ""
            } else {
                SignificanceLevel::from_p_value(p).stars()
            };
            writeln!(
                f,
                "{:<25} {} {} {} {}{}",
                self.variables[i],
                num_or_na(b, 12, 4),
                num_or_na(se, 12, 4),
                num_or_na(t, 10, 2),
                num_or_na(p, 10, 3),
                stars
            )?;
        }

        writeln!(f, "{}", "-".repeat(75))?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        let notes = notes_text(self);
        if !notes.is_empty() {
            writeln!(f)?;
            write!(f, "{notes}")?;
        }
        if let Some(note) = self.missing.note() {
            writeln!(f)?;
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

/// Run Difference-in-Differences estimation.
///
/// # Arguments
/// * `dataset` - The dataset
/// * `dep_var` - Dependent variable name
/// * `treatment_var` - Binary variable indicating treatment group (1 = treated, 0 = control)
/// * `post_var` - Binary variable indicating post-treatment period (1 = post, 0 = pre)
/// * `controls` - Optional control variables to include in the regression
/// * `cluster_var` - Optional variable to cluster standard errors on (e.g., unit/entity ID).
///   When provided, uses CR1 clustered SEs instead of HC1 robust SEs. Cluster ids may be
///   of any type (integers, fractional numbers, strings, categoricals, dates); every
///   distinct value is its own cluster.
///
/// # Missing values
/// The outcome, treatment, post indicator, controls and the cluster id are read through
/// [`crate::data::columns::ColumnRequest`]: a row with a missing value (null or NaN) in any
/// of them is dropped from all of them, and `DiDResult::missing` and the last Display line
/// report what was dropped. Text in a numeric role is `NonNumericColumn`; fewer than k + 1
/// complete rows (k regressors) is `InsufficientData`.
///
/// # Model
/// The model estimated is:
/// y = β₀ + β₁·treatment + β₂·post + β₃·(treatment × post) + controls + ε
///
/// The DiD estimate (ATT) is β₃.
pub fn run_did(
    dataset: &Dataset,
    dep_var: &str,
    treatment_var: &str,
    post_var: &str,
    controls: Option<&[&str]>,
    cluster_var: Option<&str>,
) -> EconResult<DiDResult> {
    let control_cols: &[&str] = controls.unwrap_or(&[]);
    let k_base = 4; // intercept, treatment, post, interaction
    let k = k_base + control_cols.len();

    // One request over every column the model uses: rows with a missing value in
    // any of them are dropped jointly (reported in `missing`).
    let mut request = ColumnRequest::new(dataset.df())
        .numeric(dep_var)
        .numeric(treatment_var)
        .numeric(post_var)
        .numerics(control_cols);
    if let Some(c) = cluster_var {
        request = request.factor(c);
    }
    let cols = request
        .na(NaPolicy::Omit)
        .min_rows(k + 1, "difference-in-differences")
        .extract()?;
    let y = cols.numeric_array(dep_var);
    let treatment = cols.numeric_array(treatment_var);
    let post = cols.numeric_array(post_var);
    let n = cols.n();

    // Create interaction term: treatment × post
    let interaction: Array1<f64> = treatment
        .iter()
        .zip(post.iter())
        .map(|(&t, &p)| t * p)
        .collect();

    // Compute group means for display
    let (control_pre_mean, control_post_mean, treated_pre_mean, treated_post_mean) =
        compute_group_means(&y, &treatment, &post);

    // Build design matrix: [intercept, treatment, post, interaction, controls...]
    let mut x = Array2::zeros((n, k));

    // Intercept
    x.column_mut(0).fill(1.0);
    // Treatment
    x.column_mut(1).assign(&treatment);
    // Post
    x.column_mut(2).assign(&post);
    // Interaction (DiD term)
    x.column_mut(3).assign(&interaction);

    // Variable names
    let mut var_names = vec![
        "const".to_string(),
        treatment_var.to_string(),
        post_var.to_string(),
        format!("{}*{}", treatment_var, post_var),
    ];

    // Add control variables
    let control_names: Vec<String> = control_cols.iter().map(|s| s.to_string()).collect();
    for (i, col_name) in control_cols.iter().enumerate() {
        for (cell, &v) in x
            .column_mut(k_base + i)
            .iter_mut()
            .zip(cols.numeric(col_name))
        {
            *cell = v;
        }
        var_names.push(col_name.to_string());
    }

    // OLS's solve: normal equations when well conditioned, lm's pivoted QR
    // otherwise; aliased regressors are NA
    let ls = solve_ols(x.view(), y.view())?;
    if ls.aliased.contains(&3) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the treatment × post interaction '{treatment_var}*{post_var}' is not \
                 identified (aliased): both groups must be observed before and after treatment"
            ),
        });
    }
    let aliased: Vec<String> = ls.aliased.iter().map(|&j| var_names[j].clone()).collect();
    let mut warnings = Vec::new();
    if let Some(w) = aliasing_message(&ls.aliased, &var_names) {
        warnings.push(w);
    }
    let residuals = &ls.residuals;
    let df =
        n.checked_sub(ls.rank)
            .filter(|d| *d > 0)
            .ok_or_else(|| EconError::InsufficientData {
                required: ls.rank + 1,
                provided: n,
                context: "difference-in-differences residual degrees of freedom".to_string(),
            })?;
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();

    // Sandwich on the identified columns: bread (X_id'X_id)^-1
    let id: Vec<usize> = (0..k).filter(|j| !ls.aliased.contains(j)).collect();
    let (x_id, bread) = if ls.aliased.is_empty() {
        (x, ls.unscaled)
    } else {
        (
            x.select(Axis(1), &id),
            ls.unscaled.select(Axis(0), &id).select(Axis(1), &id),
        )
    };
    let kid = id.len();
    let (vcov_id, n_clust) = if let Some(cvar) = cluster_var {
        // CR1, one cluster per factor level
        let factor = cols.factor(cvar);
        let g = factor.n_levels();
        if g < 2 {
            return Err(EconError::InsufficientData {
                required: 2,
                provided: g,
                context: format!(
                    "clustered standard errors need at least 2 clusters (cluster_var '{cvar}')"
                ),
            });
        }
        let mut scores = resource::try_zeros2(g, kid, "DiD cluster scores")?;
        for (i, &code) in factor.codes.iter().enumerate() {
            let e = residuals[i];
            let mut row = scores.row_mut(code as usize);
            row.scaled_add(e, &x_id.row(i));
        }
        let meat = scores.t().dot(&scores);
        let gf = g as f64;
        let correction = (gf / (gf - 1.0)) * ((n as f64 - 1.0) / df as f64);
        (bread.dot(&meat).dot(&bread) * correction, Some(g))
    } else {
        // HC1
        let mut xe = x_id;
        for (mut row, &e) in xe.rows_mut().into_iter().zip(residuals.iter()) {
            row *= e;
        }
        let meat = xe.t().dot(&xe);
        (bread.dot(&meat).dot(&bread) * (n as f64 / df as f64), None)
    };

    let mut std_errors = vec![f64::NAN; k];
    for (a, &j) in id.iter().enumerate() {
        std_errors[j] = vcov_id[[a, a]].max(0.0).sqrt();
    }
    let coefficients = ls.beta.to_vec();

    // The ATT is the coefficient on the interaction term (index 3)
    let att = coefficients[3];
    let att_se = std_errors[3];
    let t_stat = if att_se > 0.0 { att / att_se } else { 0.0 };
    let p_value = t_test_p_value(t_stat, df as f64);
    let significance = SignificanceLevel::from_p_value(p_value);

    // R-squared
    let y_mean = y.mean().unwrap_or(0.0);
    let sst: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let r_squared = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };
    let adj_r_squared = 1.0 - (1.0 - r_squared) * (n as f64 - 1.0) / (df as f64);

    Ok(DiDResult {
        dep_var: dep_var.to_string(),
        treatment_var: treatment_var.to_string(),
        post_var: post_var.to_string(),
        att,
        std_error: att_se,
        t_stat,
        p_value,
        significance,
        r_squared,
        adj_r_squared,
        n_obs: n,
        df,
        control_pre_mean,
        control_post_mean,
        treated_pre_mean,
        treated_post_mean,
        coefficients,
        std_errors,
        variables: var_names,
        controls: control_names,
        cluster_var: cluster_var.map(|s| s.to_string()),
        n_clusters: n_clust,
        missing: cols.missing().clone(),
        aliased,
        warnings,
    })
}

/// Compute group means for the four cells of the DiD design.
fn compute_group_means(
    y: &Array1<f64>,
    treatment: &Array1<f64>,
    post: &Array1<f64>,
) -> (f64, f64, f64, f64) {
    let mut control_pre = (0.0, 0);
    let mut control_post = (0.0, 0);
    let mut treated_pre = (0.0, 0);
    let mut treated_post = (0.0, 0);

    for i in 0..y.len() {
        let t = treatment[i];
        let p = post[i];
        let yi = y[i];

        if t < 0.5 && p < 0.5 {
            // Control, Pre
            control_pre.0 += yi;
            control_pre.1 += 1;
        } else if t < 0.5 && p >= 0.5 {
            // Control, Post
            control_post.0 += yi;
            control_post.1 += 1;
        } else if t >= 0.5 && p < 0.5 {
            // Treated, Pre
            treated_pre.0 += yi;
            treated_pre.1 += 1;
        } else {
            // Treated, Post
            treated_post.0 += yi;
            treated_post.1 += 1;
        }
    }

    let control_pre_mean = if control_pre.1 > 0 {
        control_pre.0 / control_pre.1 as f64
    } else {
        0.0
    };
    let control_post_mean = if control_post.1 > 0 {
        control_post.0 / control_post.1 as f64
    } else {
        0.0
    };
    let treated_pre_mean = if treated_pre.1 > 0 {
        treated_pre.0 / treated_pre.1 as f64
    } else {
        0.0
    };
    let treated_post_mean = if treated_post.1 > 0 {
        treated_post.0 / treated_post.1 as f64
    } else {
        0.0
    };

    (
        control_pre_mean,
        control_post_mean,
        treated_pre_mean,
        treated_post_mean,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_did_dataset() -> Dataset {
        // Classic DiD setup:
        // Control group: y goes from 10 to 12 (trend = +2)
        // Treatment group: y goes from 10 to 15 (trend + effect = +5)
        // True ATT = 15 - 10 - (12 - 10) = 3
        let df = df! {
            "y" => [10.0, 10.0, 12.0, 12.0, 10.0, 10.0, 15.0, 15.0],
            "treatment" => [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "post" => [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_group_means() {
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let treatment = Array1::from(vec![0.0, 0.0, 1.0, 1.0]);
        let post = Array1::from(vec![0.0, 1.0, 0.0, 1.0]);

        let (cp, cpo, tp, tpo) = compute_group_means(&y, &treatment, &post);

        assert!((cp - 1.0).abs() < 1e-10);
        assert!((cpo - 2.0).abs() < 1e-10);
        assert!((tp - 3.0).abs() < 1e-10);
        assert!((tpo - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_did_basic() {
        let dataset = create_did_dataset();
        let result = run_did(&dataset, "y", "treatment", "post", None, None).unwrap();

        // Check structure
        assert_eq!(result.n_obs, 8);

        // ATT should be 3.0 (treatment effect beyond parallel trend)
        assert!(
            (result.att - 3.0).abs() < 0.1,
            "ATT should be 3.0, got {}",
            result.att
        );

        // Check group means
        assert!((result.control_pre_mean - 10.0).abs() < 0.1);
        assert!((result.control_post_mean - 12.0).abs() < 0.1);
        assert!((result.treated_pre_mean - 10.0).abs() < 0.1);
        assert!((result.treated_post_mean - 15.0).abs() < 0.1);
    }

    #[test]
    fn test_did_with_noise() {
        // Add some noise to make it more realistic
        let df = df! {
            "y" => [9.8, 10.2, 11.9, 12.1, 9.9, 10.1, 14.8, 15.2],
            "treatment" => [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "post" => [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_did(&dataset, "y", "treatment", "post", None, None).unwrap();

        // ATT should still be close to 3.0
        assert!(
            (result.att - 3.0).abs() < 0.5,
            "ATT should be close to 3.0, got {}",
            result.att
        );
    }

    #[test]
    fn test_did_missing_column() {
        let dataset = create_did_dataset();
        let result = run_did(&dataset, "nonexistent", "treatment", "post", None, None);
        assert!(result.is_err());
    }

    // ════════════════════════════════════════════════════════════════════════════
    // VALIDATION TESTS - Comparing against R manual DiD
    // ════════════════════════════════════════════════════════════════════════════

    /// Create dataset for DiD validation with known ATT = 5
    /// Based on R code:
    /// ```r
    /// set.seed(42)
    /// data$y <- 10 + 2*treat + 3*post + 5*treat*post + noise
    /// ```
    fn create_validation_did_dataset() -> Dataset {
        // Classic 2x2 DiD with known parameters:
        // Baseline = 10
        // Treatment group level = +2
        // Time trend = +3
        // ATT = +5

        let mut y_vec = Vec::new();
        let mut treat_vec = Vec::new();
        let mut post_vec = Vec::new();

        let n_per_group = 50;

        // Control group, pre-treatment: y = 10 + noise
        for i in 0..n_per_group {
            let noise = (i as f64 * 0.789).sin() * 0.8;
            y_vec.push(10.0 + noise);
            treat_vec.push(0.0);
            post_vec.push(0.0);
        }

        // Control group, post-treatment: y = 10 + 3 + noise
        for i in 0..n_per_group {
            let noise = (i as f64 * 0.891).cos() * 0.8;
            y_vec.push(13.0 + noise);
            treat_vec.push(0.0);
            post_vec.push(1.0);
        }

        // Treatment group, pre-treatment: y = 10 + 2 + noise
        for i in 0..n_per_group {
            let noise = (i as f64 * 0.567).sin() * 0.8;
            y_vec.push(12.0 + noise);
            treat_vec.push(1.0);
            post_vec.push(0.0);
        }

        // Treatment group, post-treatment: y = 10 + 2 + 3 + 5 + noise = 20 + noise
        for i in 0..n_per_group {
            let noise = (i as f64 * 0.678).cos() * 0.8;
            y_vec.push(20.0 + noise);
            treat_vec.push(1.0);
            post_vec.push(1.0);
        }

        let df = df! {
            "y" => y_vec,
            "treat" => treat_vec,
            "post" => post_vec
        }
        .unwrap();
        Dataset::new(df)
    }

    /// Validation test: Classic 2x2 DiD with known ATT
    /// ATT should be close to 5.0
    ///
    /// R code:
    /// ```r
    /// did_fit <- lm(y ~ treat + post + treat:post, data = data)
    /// # ATT = coefficient on treat:post ≈ 5.0
    /// ```
    #[test]
    fn test_did_classic_2x2_recovers_dgp() {
        let dataset = create_validation_did_dataset();
        let result = run_did(&dataset, "y", "treat", "post", None, None).unwrap();

        // Structure checks
        assert_eq!(result.n_obs, 200); // 4 groups × 50 obs
        assert_eq!(result.treatment_var, "treat");
        assert_eq!(result.post_var, "post");

        // ATT should be close to true value (5.0)
        assert!(
            (result.att - 5.0).abs() < 0.5,
            "ATT should be close to 5.0, got {}",
            result.att
        );

        // Standard error should be positive and reasonable
        assert!(
            result.std_error > 0.0 && result.std_error < 2.0,
            "ATT SE should be positive and reasonable, got {}",
            result.std_error
        );

        // t-statistic should be large (significant effect)
        assert!(
            result.t_stat.abs() > 2.0,
            "t-stat should indicate significance, got {}",
            result.t_stat
        );

        // p-value should be small (significant)
        assert!(
            result.p_value < 0.05,
            "p-value should be < 0.05 for significant ATT, got {}",
            result.p_value
        );

        // Check group means
        // Control pre: ~10, Control post: ~13
        // Treated pre: ~12, Treated post: ~20
        assert!(
            (result.control_pre_mean - 10.0).abs() < 0.5,
            "Control pre mean should be ~10, got {}",
            result.control_pre_mean
        );
        assert!(
            (result.control_post_mean - 13.0).abs() < 0.5,
            "Control post mean should be ~13, got {}",
            result.control_post_mean
        );
        assert!(
            (result.treated_pre_mean - 12.0).abs() < 0.5,
            "Treated pre mean should be ~12, got {}",
            result.treated_pre_mean
        );
        assert!(
            (result.treated_post_mean - 20.0).abs() < 0.5,
            "Treated post mean should be ~20, got {}",
            result.treated_post_mean
        );

        // Manual DiD calculation should match regression
        let manual_did = (result.treated_post_mean - result.treated_pre_mean)
            - (result.control_post_mean - result.control_pre_mean);
        assert!(
            (result.att - manual_did).abs() < 0.01,
            "Regression ATT should match manual calculation"
        );
    }

    /// Validation test: DiD with covariates
    #[test]
    fn test_did_with_covariates_recovers_dgp() {
        // Add covariate to the dataset
        let mut y_vec = Vec::new();
        let mut treat_vec = Vec::new();
        let mut post_vec = Vec::new();
        let mut x_vec = Vec::new();

        let n_per_group = 50;

        for group in 0..4 {
            let (treat, post) = match group {
                0 => (0.0, 0.0), // Control, pre
                1 => (0.0, 1.0), // Control, post
                2 => (1.0, 0.0), // Treated, pre
                _ => (1.0, 1.0), // Treated, post
            };

            for i in 0..n_per_group {
                let x = (group * 50 + i) as f64 * 0.02;
                let noise = ((group * 50 + i) as f64 * 0.543).sin() * 0.5;

                // y = 10 + 2*treat + 3*post + 5*treat*post + 1.5*x + noise
                let y = 10.0 + 2.0 * treat + 3.0 * post + 5.0 * treat * post + 1.5 * x + noise;

                y_vec.push(y);
                treat_vec.push(treat);
                post_vec.push(post);
                x_vec.push(x);
            }
        }

        let df = df! {
            "y" => y_vec,
            "treat" => treat_vec,
            "post" => post_vec,
            "x" => x_vec
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_did(&dataset, "y", "treat", "post", Some(&["x"]), None).unwrap();

        // ATT should still be close to 5.0 (controlling for x)
        assert!(
            (result.att - 5.0).abs() < 0.5,
            "ATT with covariates should be close to 5.0, got {}",
            result.att
        );

        // Should have control variable in results
        assert!(
            result.controls.contains(&"x".to_string()),
            "Should record covariate x"
        );

        // Check coefficient on covariate (should be close to 1.5)
        // The covariate is the 5th variable: [const, treat, post, treat*post, x]
        let x_idx = result.variables.iter().position(|v| v == "x").unwrap();
        assert!(
            (result.coefficients[x_idx] - 1.5).abs() < 0.3,
            "Covariate coefficient should be close to 1.5, got {}",
            result.coefficients[x_idx]
        );
    }

    /// Validation test: Null case (no treatment effect)
    #[test]
    fn test_did_null_effect_recovers_dgp() {
        // Create data with NO treatment effect (ATT = 0)
        let mut y_vec = Vec::new();
        let mut treat_vec = Vec::new();
        let mut post_vec = Vec::new();

        let n_per_group = 100;

        for group in 0..4 {
            let (treat, post) = match group {
                0 => (0.0, 0.0),
                1 => (0.0, 1.0),
                2 => (1.0, 0.0),
                _ => (1.0, 1.0),
            };

            for i in 0..n_per_group {
                // y = 10 + 2*treat + 3*post + 0*treat*post + noise
                // NO interaction effect!
                let noise = ((group * 100 + i) as f64 * 0.432).sin() * 1.5;
                let y = 10.0 + 2.0 * treat + 3.0 * post + noise;

                y_vec.push(y);
                treat_vec.push(treat);
                post_vec.push(post);
            }
        }

        let df = df! {
            "y" => y_vec,
            "treat" => treat_vec,
            "post" => post_vec
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_did(&dataset, "y", "treat", "post", None, None).unwrap();

        // ATT should be close to 0
        assert!(
            result.att.abs() < 1.0,
            "ATT should be close to 0 when no effect, got {}",
            result.att
        );

        // Not significant. The data are deterministic: lm + vcovHC(type = "HC1")
        // (df 396) gives ATT -0.028054724238937286, SE 0.21315963502411217,
        // t -0.1316136811538629, p 0.89535670480011142 (quoted, not a registered case).
        assert!(result.p_value > 0.05, "p = {}", result.p_value);
        assert_eq!(result.df, 396);
        for (got, want) in [
            (result.att, -0.028054724238937286),
            (result.std_error, 0.21315963502411217),
            (result.t_stat, -0.1316136811538629),
            (result.p_value, 0.89535670480011142),
        ] {
            assert!((got - want).abs() <= 1e-8 * want.abs(), "{got} vs {want}");
        }
    }

    // ════════════════════════════════════════════════════════════════════════════
    // T3-WP21: lm's pivoted QR, checked df, clusters (CAUS-102); the R case
    // did_2x2_training (validation/r/cases/causal_did_cov.R)
    // ════════════════════════════════════════════════════════════════════════════

    /// CAUS-102: coefficients, HC1 and CR1 SEs, t, p, df, R2 and the group means
    /// against lm + sandwich on employee_training, including an aliased control
    /// (years_since_hire = tenure + 0.5), NA as in lm.
    #[test]
    fn test_validate_did_2x2_training() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("did_2x2_training");
        let rn = |ctrl: &[&str]| -> Vec<String> {
            [
                "(Intercept)",
                "training_group",
                "post_training",
                "training_group:post_training",
            ]
            .iter()
            .chain(ctrl.iter())
            .map(|s| s.to_string())
            .collect()
        };
        for (spec, ctrl) in [
            ("base", &[][..]),
            ("tenure", &["tenure"][..]),
            ("aliased", &["tenure", "years_since_hire"][..]),
        ] {
            let c = if ctrl.is_empty() { None } else { Some(ctrl) };
            let hc1 = run_did(
                &ds,
                "performance",
                "training_group",
                "post_training",
                c,
                None,
            )
            .unwrap();
            let cl = run_did(
                &ds,
                "performance",
                "training_group",
                "post_training",
                c,
                Some("employee_id"),
            )
            .unwrap();
            let (names, n) = (rn(ctrl), hc1.n_obs);
            let t = |r: &DiDResult| {
                r.coefficients
                    .iter()
                    .zip(&r.std_errors)
                    .map(|(b, s)| b / s)
                    .collect::<Vec<_>>()
            };
            let p = |r: &DiDResult| {
                t(r).iter()
                    .map(|&x| t_test_p_value(x, r.df as f64))
                    .collect::<Vec<_>>()
            };
            Check::new(&case)
                .named(
                    &format!("{spec}_coef"),
                    &names,
                    &hc1.coefficients,
                    Tol::coef(n),
                )
                .named(
                    &format!("{spec}_coef"),
                    &names,
                    &cl.coefficients,
                    Tol::coef(n),
                )
                .named(
                    &format!("{spec}_se_hc1"),
                    &names,
                    &hc1.std_errors,
                    Tol::se(n),
                )
                .named(&format!("{spec}_se_cl"), &names, &cl.std_errors, Tol::se(n))
                .named(&format!("{spec}_t_hc1"), &names, &t(&hc1), Tol::stat(n))
                .named(&format!("{spec}_t_cl"), &names, &t(&cl), Tol::stat(n))
                .named(&format!("{spec}_p_hc1"), &names, &p(&hc1), Tol::p_value(n))
                .named(&format!("{spec}_p_cl"), &names, &p(&cl), Tol::p_value(n))
                .scalar(&format!("{spec}_df_resid"), hc1.df as f64, Tol::exact())
                .scalar(&format!("{spec}_r2"), hc1.r_squared, Tol::stat(n))
                .scalar(&format!("{spec}_adj_r2"), hc1.adj_r_squared, Tol::stat(n))
                .finish();
            let ix = names
                .iter()
                .position(|s| s == "training_group:post_training")
                .unwrap();
            assert_eq!(
                (cl.att, cl.std_error, cl.t_stat),
                (cl.coefficients[ix], cl.std_errors[ix], t(&cl)[ix])
            );
            let want_p = case.named(&format!("{spec}_p_cl"), "training_group:post_training");
            assert!((cl.p_value - want_p).abs() <= 1e-12 + 1e-8 * cl.p_value);
            assert_eq!(cl.n_clusters, Some(case.scalar("n_clusters") as usize));
            if spec == "aliased" {
                assert_eq!(hc1.aliased, vec!["years_since_hire".to_string()]);
                assert!(
                    hc1.warnings.iter().any(|w| w
                        == "Coefficients not defined because of singularities: years_since_hire")
                );
                let shown = format!("{hc1}");
                assert!(
                    shown.contains("years_since_hire") && shown.contains("NA"),
                    "{shown}"
                );
                assert!(
                    shown.contains("Warning: Coefficients not defined"),
                    "{shown}"
                );
            } else {
                assert!(hc1.aliased.is_empty() && hc1.warnings.is_empty());
            }
        }
        let r = run_did(
            &ds,
            "performance",
            "training_group",
            "post_training",
            None,
            None,
        )
        .unwrap();
        Check::new(&case)
            .vector(
                "group_means",
                &[
                    r.control_pre_mean,
                    r.control_post_mean,
                    r.treated_pre_mean,
                    r.treated_post_mean,
                ],
                Tol::coef(r.n_obs),
            )
            .finish();
    }

    /// n <= k is refused (before T2-WP12's row check: a usize overflow panic, or
    /// se 0 and p NaN with 4 rows).
    #[test]
    fn test_run_did_rejects_n_le_k() {
        let df = df! {
            "y" => [1.0, 2.1, 2.9],
            "treat" => [0.0, 1.0, 1.0],
            "post" => [0.0, 0.0, 1.0],
            "cl" => [1i64, 2, 3],
        }
        .unwrap();
        let r = run_did(&Dataset::new(df), "y", "treat", "post", None, Some("cl"));
        assert!(
            matches!(r, Err(EconError::InsufficientData { .. })),
            "{r:?}"
        );
        let df = df! {
            "y" => [1.0, 2.1, 2.9, 4.2],
            "treat" => [0.0, 0.0, 1.0, 1.0],
            "post" => [0.0, 1.0, 0.0, 1.0],
        }
        .unwrap();
        let r = run_did(&Dataset::new(df), "y", "treat", "post", None, None);
        assert!(
            matches!(r, Err(EconError::InsufficientData { .. })),
            "{r:?}"
        );
    }

    /// One cluster: sandwich::vcovCL fails; before, the CR1 factor fell back to 1.
    /// employee_training, for the tests that are not R-pinned (not registered).
    fn training() -> Dataset {
        crate::test_support::load_case("did_2x2_training").1
    }

    #[test]
    fn test_run_did_rejects_single_cluster() {
        let ds = training();
        let mut df = ds.df().clone();
        let h = df.height();
        df.with_column(Column::new("one".into(), vec![1i64; h]))
            .unwrap();
        let r = run_did(
            &Dataset::new(df),
            "performance",
            "training_group",
            "post_training",
            None,
            Some("one"),
        );
        match r {
            Err(EconError::InsufficientData { context, .. }) => {
                assert!(context.contains("at least 2 clusters"), "{context}")
            }
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }

    /// Only treated units: the interaction is aliased with post, so the ATT is
    /// not identified (before: a generalized-inverse number).
    #[test]
    fn test_run_did_rejects_unidentified_att() {
        let ds = training();
        let mask = ds
            .df()
            .column("training_group")
            .unwrap()
            .as_materialized_series()
            .equal(1)
            .unwrap();
        let df = ds.df().filter(&mask).unwrap();
        let r = run_did(
            &Dataset::new(df),
            "performance",
            "training_group",
            "post_training",
            None,
            None,
        );
        match r {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(
                    message.contains("training_group*post_training"),
                    "{message}"
                );
                assert!(message.contains("not identified"), "{message}");
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    // ════════════════════════════════════════════════════════════════════════════
    // T2-WP12: columns through the shared extractor (CAUS-58)
    // ════════════════════════════════════════════════════════════════════════════

    /// `create_validation_did_dataset()` with three codings of 20 clusters: row i is in
    /// cluster i % 20, as an integer (`cl_int`), a string (`cl_str`, "c0".."c19") and a
    /// fraction (`cl_frac`, 0.0..1.9).
    fn did_with_clusters() -> Dataset {
        let ds = create_validation_did_dataset();
        let n = ds.nrows();
        let mut df = ds.df().clone();
        df.with_column(Column::new(
            "cl_int".into(),
            (0..n).map(|i| (i % 20) as i64).collect::<Vec<_>>(),
        ))
        .unwrap();
        df.with_column(Column::new(
            "cl_str".into(),
            (0..n).map(|i| format!("c{}", i % 20)).collect::<Vec<_>>(),
        ))
        .unwrap();
        df.with_column(Column::new(
            "cl_frac".into(),
            (0..n).map(|i| (i % 20) as f64 / 10.0).collect::<Vec<_>>(),
        ))
        .unwrap();
        Dataset::new(df)
    }

    fn same(a: f64, b: f64) -> bool {
        a == b || (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
    }

    fn assert_same_did(a: &DiDResult, b: &DiDResult) {
        assert_eq!(a.n_obs, b.n_obs);
        assert_eq!(a.n_clusters, b.n_clusters);
        for (x, y) in [
            (a.att, b.att),
            (a.std_error, b.std_error),
            (a.t_stat, b.t_stat),
            (a.p_value, b.p_value),
            (a.r_squared, b.r_squared),
            (a.adj_r_squared, b.adj_r_squared),
            (a.control_pre_mean, b.control_pre_mean),
            (a.control_post_mean, b.control_post_mean),
            (a.treated_pre_mean, b.treated_pre_mean),
            (a.treated_post_mean, b.treated_post_mean),
        ] {
            assert!(same(x, y), "{x} vs {y}");
        }
        for (x, y) in a
            .coefficients
            .iter()
            .chain(&a.std_errors)
            .zip(b.coefficients.iter().chain(&b.std_errors))
        {
            assert!(same(*x, *y), "{x} vs {y}");
        }
    }

    #[test]
    fn test_na_did_cluster_ids_are_factors() {
        let ds = did_with_clusters();
        let int = run_did(&ds, "y", "treat", "post", None, Some("cl_int")).unwrap();
        assert_eq!(int.n_clusters, Some(20));
        for c in ["cl_str", "cl_frac"] {
            let r = run_did(&ds, "y", "treat", "post", None, Some(c))
                .unwrap_or_else(|e| panic!("{c}: {e}"));
            assert_eq!(r.n_clusters, Some(20), "{c}");
            assert_same_did(&int, &r);
        }
    }

    #[test]
    fn test_na_did_omits_missing_rows() {
        let ds = did_with_clusters();
        let null_at = |ds: &Dataset, col: &str, row: usize| {
            let c = ds.df().column(col).unwrap();
            let values = match c.dtype() {
                DataType::Int64 => Column::new(
                    col.into(),
                    c.i64()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(i, v)| if i == row { None } else { v })
                        .collect::<Vec<_>>(),
                ),
                _ => Column::new(
                    col.into(),
                    c.f64()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(i, v)| if i == row { None } else { v })
                        .collect::<Vec<_>>(),
                ),
            };
            let mut df = ds.df().clone();
            df.with_column(values).unwrap();
            Dataset::new(df)
        };
        let na = null_at(&null_at(&ds, "y", 7), "cl_int", 150);
        let keep =
            BooleanChunked::from_iter_values("keep".into(), (0..200).map(|i| i != 7 && i != 150));
        let kept = Dataset::new(ds.df().filter(&keep).unwrap());
        let a = run_did(&na, "y", "treat", "post", None, Some("cl_int")).unwrap();
        let b = run_did(&kept, "y", "treat", "post", None, Some("cl_int")).unwrap();
        assert_same_did(&a, &b);
        assert_eq!(a.n_obs, 198);
        assert_eq!(a.missing.n_dropped, 2);
        assert_eq!(
            a.missing.by_column,
            vec![("y".to_string(), 1), ("cl_int".to_string(), 1)]
        );
        assert!(b.missing.is_empty());
        let shown = format!("{a}");
        assert!(
            shown.trim_end().ends_with(
                "Missing values: 2 of 200 rows dropped because of missing values (y: 1, cl_int: 1)"
            ),
            "{shown}"
        );
        assert!(!format!("{b}").contains("Missing values"));
    }

    #[test]
    fn test_na_did_errors_name_the_column() {
        let ds = did_with_clusters();
        let text: Vec<String> = ds
            .df()
            .column("y")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .map(|v| format!("{}", v.unwrap()))
            .collect();
        let mut df = ds.df().clone();
        df.with_column(Column::new("y".into(), text)).unwrap();
        match run_did(&Dataset::new(df), "y", "treat", "post", None, None) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "y"),
            other => panic!("expected NonNumericColumn, got {other:?}"),
        }
        match run_did(&ds, "nonexistent", "treat", "post", None, None) {
            Err(EconError::ColumnNotFound { column, .. }) => assert_eq!(column, "nonexistent"),
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
    }
}
