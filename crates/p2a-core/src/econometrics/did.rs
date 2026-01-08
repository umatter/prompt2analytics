//! Difference-in-Differences (DiD) estimation.
//!
//! Pure Rust implementation without external formula parsing.
//! Uses column-based API for simplicity.

use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconResult, EconError};
use crate::linalg::matrix_ops::{xtx, xty, safe_inverse};
use crate::linalg::design::DesignMatrix;
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
}

impl fmt::Display for DiDResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Difference-in-Differences Estimation")?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "Treatment: {}, Post: {}", self.treatment_var, self.post_var)?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "R-squared: {:.4}", self.r_squared)?;
        writeln!(f, "Adj. R-squared: {:.4}", self.adj_r_squared)?;
        writeln!(f)?;

        writeln!(f, "DiD ESTIMATE (Average Treatment Effect on Treated):")?;
        writeln!(f, "  ATT = {:.4} (SE: {:.4}, t = {:.2}, p = {:.3}){}",
                 self.att, self.std_error, self.t_stat, self.p_value, self.significance.stars())?;
        writeln!(f)?;

        writeln!(f, "Group Means:")?;
        writeln!(f, "  Control (Pre):  {:.4}    Control (Post): {:.4}",
                 self.control_pre_mean, self.control_post_mean)?;
        writeln!(f, "  Treated (Pre):  {:.4}    Treated (Post): {:.4}",
                 self.treated_pre_mean, self.treated_post_mean)?;
        writeln!(f)?;

        writeln!(f, "Full Regression Results:")?;
        writeln!(f, "{:<25} {:>12} {:>12} {:>10} {:>10}",
                 "Variable", "Coef", "Std Err", "t", "P>|t|")?;
        writeln!(f, "{}", "-".repeat(75))?;

        for i in 0..self.variables.len() {
            let sig = SignificanceLevel::from_p_value(
                t_test_p_value(self.coefficients[i] / self.std_errors[i], self.df as f64)
            );
            let t = if self.std_errors[i] > 0.0 {
                self.coefficients[i] / self.std_errors[i]
            } else {
                0.0
            };
            let p = t_test_p_value(t, self.df as f64);
            writeln!(f, "{:<25} {:>12.4} {:>12.4} {:>10.2} {:>10.3}{}",
                     self.variables[i],
                     self.coefficients[i],
                     self.std_errors[i],
                     t,
                     p,
                     sig.stars())?;
        }

        writeln!(f, "{}", "-".repeat(75))?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

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
) -> EconResult<DiDResult> {
    // Extract y
    let y = DesignMatrix::extract_column(dataset.df(), dep_var)
        .map_err(|e| EconError::ColumnNotFound {
            column: dep_var.to_string(),
            available: vec![format!("{:?}", e)],
        })?;

    // Extract treatment and post indicators
    let treatment = DesignMatrix::extract_column(dataset.df(), treatment_var)
        .map_err(|e| EconError::ColumnNotFound {
            column: treatment_var.to_string(),
            available: vec![format!("{:?}", e)],
        })?;

    let post = DesignMatrix::extract_column(dataset.df(), post_var)
        .map_err(|e| EconError::ColumnNotFound {
            column: post_var.to_string(),
            available: vec![format!("{:?}", e)],
        })?;

    let n = y.len();

    // Create interaction term: treatment × post
    let interaction: Array1<f64> = treatment.iter()
        .zip(post.iter())
        .map(|(&t, &p)| t * p)
        .collect();

    // Compute group means for display
    let (control_pre_mean, control_post_mean, treated_pre_mean, treated_post_mean) =
        compute_group_means(&y, &treatment, &post);

    // Build design matrix: [intercept, treatment, post, interaction, controls...]
    let k_base = 4; // intercept, treatment, post, interaction
    let k_controls = controls.map(|c| c.len()).unwrap_or(0);
    let k = k_base + k_controls;

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
    let control_names: Vec<String>;
    if let Some(ctrl) = controls {
        control_names = ctrl.iter().map(|s| s.to_string()).collect();
        for (i, col_name) in ctrl.iter().enumerate() {
            let col_data = DesignMatrix::extract_column(dataset.df(), col_name)
                .map_err(|e| EconError::ColumnNotFound {
                    column: col_name.to_string(),
                    available: vec![format!("{:?}", e)],
                })?;
            x.column_mut(k_base + i).assign(&col_data);
            var_names.push(col_name.to_string());
        }
    } else {
        control_names = vec![];
    }

    // OLS estimation: β = (X'X)^{-1} X'y
    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view())
        .map_err(|e| EconError::SingularMatrix {
            context: "X'X in DiD estimation".to_string(),
            suggestion: format!("Check for perfect collinearity: {:?}", e),
        })?;

    let xty_vec = xty(&x.view(), &y);
    let beta: Array1<f64> = xtx_inv.dot(&xty_vec);

    // Residuals
    let y_hat = x.dot(&beta);
    let residuals = &y - &y_hat;

    let df = n.saturating_sub(k);
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let _sigma2 = if df > 0 { ssr / df as f64 } else { ssr / n as f64 };

    // Robust standard errors (HC1)
    let scale = (n as f64) / (df as f64);
    let mut meat: Array2<f64> = Array2::zeros((k, k));
    for i in 0..n {
        let xi = x.row(i);
        let e2: f64 = residuals[i] * residuals[i];
        for j in 0..k {
            for l in 0..k {
                meat[[j, l]] += e2 * xi[j] * xi[l];
            }
        }
    }
    let meat: Array2<f64> = &meat * scale;

    // Sandwich estimator
    let mut vcov: Array2<f64> = Array2::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            for m in 0..k {
                for l in 0..k {
                    vcov[[i, j]] += xtx_inv[[i, m]] * meat[[m, l]] * xtx_inv[[l, j]];
                }
            }
        }
    }

    let std_errors: Vec<f64> = vcov.diag().mapv(|v: f64| v.max(0.0).sqrt()).to_vec();
    let coefficients = beta.to_vec();

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
    let adj_r_squared = 1.0 - (1.0 - r_squared) * ((n - 1) as f64) / (df as f64);

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

    let control_pre_mean = if control_pre.1 > 0 { control_pre.0 / control_pre.1 as f64 } else { 0.0 };
    let control_post_mean = if control_post.1 > 0 { control_post.0 / control_post.1 as f64 } else { 0.0 };
    let treated_pre_mean = if treated_pre.1 > 0 { treated_pre.0 / treated_pre.1 as f64 } else { 0.0 };
    let treated_post_mean = if treated_post.1 > 0 { treated_post.0 / treated_post.1 as f64 } else { 0.0 };

    (control_pre_mean, control_post_mean, treated_pre_mean, treated_post_mean)
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
        }.unwrap();
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
        let result = run_did(&dataset, "y", "treatment", "post", None).unwrap();

        // Check structure
        assert_eq!(result.n_obs, 8);

        // ATT should be 3.0 (treatment effect beyond parallel trend)
        assert!((result.att - 3.0).abs() < 0.1,
            "ATT should be 3.0, got {}", result.att);

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
        }.unwrap();
        let dataset = Dataset::new(df);

        let result = run_did(&dataset, "y", "treatment", "post", None).unwrap();

        // ATT should still be close to 3.0
        assert!((result.att - 3.0).abs() < 0.5,
            "ATT should be close to 3.0, got {}", result.att);
    }

    #[test]
    fn test_did_missing_column() {
        let dataset = create_did_dataset();
        let result = run_did(&dataset, "nonexistent", "treatment", "post", None);
        assert!(result.is_err());
    }
}
