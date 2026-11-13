//! Standard Errors for Contrasts in ANOVA Models.
//!
//! This module computes standard errors for linear contrasts of treatment means
//! in ANOVA models. Contrasts are linear combinations of group means that sum to zero.
//!
//! # Mathematical Background
//!
//! A contrast L is a linear combination of group means:
//!
//! L = Σᵢ cᵢμᵢ  where Σᵢ cᵢ = 0
//!
//! ## Standard Error Computation
//!
//! The standard error of a contrast is:
//!
//! SE(L) = √(MSE × Σᵢ cᵢ²/nᵢ)
//!
//! where MSE is the mean squared error from ANOVA and nᵢ is the sample size
//! for group i.
//!
//! ## Common Contrast Types
//!
//! - **Treatment vs Control**: Compare each treatment to a control group
//! - **Helmert**: Compare each level to the mean of subsequent levels
//! - **Polynomial**: Test for linear, quadratic, etc. trends: the k - 1 orthonormal columns of
//!   R's `contr.poly(k)` ([`contr_poly`])
//! - **Deviation**: Compare each level to the grand mean
//!
//! ## Confidence Intervals and Tests
//!
//! For a contrast with estimate L̂ and standard error SE(L):
//! - CI: L̂ ± t_{α/2,df} × SE(L)
//! - Test statistic: t = L̂ / SE(L) ~ t(df_MSE)
//!
//! # References
//!
//! - Scheffé, H. (1959). *The Analysis of Variance*. Wiley. ISBN: 978-0471345053.
//!   The classic treatment of contrasts and multiple comparisons.
//!
//! - Kirk, R.E. (2013). *Experimental Design: Procedures for the Behavioral
//!   Sciences* (4th ed.). SAGE. ISBN: 978-1412974455. Chapter 5 on contrasts.
//!
//! - Maxwell, S.E., Delaney, H.D., & Kelley, K. (2018). *Designing Experiments
//!   and Analyzing Data* (3rd ed.). Routledge. ISBN: 978-1138892286.
//!
//! - Rosenthal, R., Rosnow, R.L., & Rubin, D.B. (2000). *Contrasts and Effect
//!   Sizes in Behavioral Research*. Cambridge University Press.
//!   ISBN: 978-0521659802.
//!
//! R equivalent: `stats::se.contrast()`

use crate::errors::{EconError, EconResult};
use crate::stats::AnovaResult;
use serde::{Deserialize, Serialize};

/// Result of contrast standard error computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeContrastResult {
    /// Standard errors for each contrast
    pub se: Vec<f64>,
    /// The contrasts used (each row is a contrast)
    pub contrasts: Vec<Vec<f64>>,
    /// Group names
    pub group_names: Vec<String>,
    /// Mean squared error from ANOVA
    pub mse: f64,
    /// Degrees of freedom for MSE
    pub df_mse: f64,
}

/// Compute standard errors for contrasts in an ANOVA model.
///
/// A contrast is a linear combination of group means where the coefficients sum to zero.
/// The standard error of a contrast L = Σ c_i * μ_i is:
///   SE(L) = sqrt(MSE * Σ (c_i² / n_i))
///
/// # Arguments
/// * `anova` - The ANOVA result containing group statistics
/// * `contrasts` - Contrast coefficients (each inner vector is one contrast)
///
/// # Returns
/// A `SeContrastResult` containing standard errors for each contrast
///
/// # Example
/// ```
/// use p2a_core::stats::secontrast::se_contrast;
/// use p2a_core::stats::AnovaResult;
///
/// // Assuming you have an ANOVA result
/// // let result = se_contrast(&anova, &contrasts).unwrap();
/// ```
pub fn se_contrast(
    anova: &AnovaResult,
    contrasts: &[Vec<f64>],
) -> Result<SeContrastResult, String> {
    let k = anova.groups.len();

    if k == 0 {
        return Err("ANOVA must have at least one group".to_string());
    }

    let group_names: Vec<String> = anova.groups.iter().map(|g| g.group.clone()).collect();
    let group_ns: Vec<f64> = anova.groups.iter().map(|g| g.n as f64).collect();

    // Validate contrasts
    for (i, contrast) in contrasts.iter().enumerate() {
        if contrast.len() != k {
            return Err(format!(
                "Contrast {} has {} elements but there are {} groups",
                i + 1,
                contrast.len(),
                k
            ));
        }

        // Check that contrast sums to zero
        let sum: f64 = contrast.iter().sum();
        if sum.abs() > 1e-8 {
            return Err(format!(
                "Contrast {} coefficients sum to {} (should be 0)",
                i + 1,
                sum
            ));
        }
    }

    let mse = anova.ms_within;
    let df_mse = anova.df_within as f64;

    // Compute SE for each contrast
    let se: Vec<f64> = contrasts
        .iter()
        .map(|c| {
            // SE = sqrt(MSE * sum(c_i^2 / n_i))
            let var_factor: f64 = c
                .iter()
                .zip(group_ns.iter())
                .map(|(&ci, &ni)| ci * ci / ni)
                .sum();
            (mse * var_factor).sqrt()
        })
        .collect();

    Ok(SeContrastResult {
        se,
        contrasts: contrasts.to_vec(),
        group_names,
        mse,
        df_mse,
    })
}

/// Compute standard error for a single contrast.
pub fn se_contrast_single(anova: &AnovaResult, contrast: &[f64]) -> Result<f64, String> {
    let result = se_contrast(anova, &[contrast.to_vec()])?;
    Ok(result.se[0])
}

/// Estimate the contrast value from an ANOVA result.
///
/// Returns: Σ c_i * mean_i
pub fn estimate_contrast(anova: &AnovaResult, contrast: &[f64]) -> Result<f64, String> {
    let k = anova.groups.len();

    if contrast.len() != k {
        return Err(format!(
            "Contrast has {} elements but there are {} groups",
            contrast.len(),
            k
        ));
    }

    let estimate: f64 = contrast
        .iter()
        .zip(anova.groups.iter())
        .map(|(&c, g)| c * g.mean)
        .sum();

    Ok(estimate)
}

/// Compute t-statistic for a contrast.
///
/// t = estimate / SE
pub fn contrast_t_statistic(anova: &AnovaResult, contrast: &[f64]) -> Result<f64, String> {
    let estimate = estimate_contrast(anova, contrast)?;
    let se = se_contrast_single(anova, contrast)?;

    if se.abs() < 1e-10 {
        return Err("Standard error is zero, cannot compute t-statistic".to_string());
    }

    Ok(estimate / se)
}

/// Compute p-value for a contrast (two-tailed).
pub fn contrast_p_value(anova: &AnovaResult, contrast: &[f64]) -> Result<f64, String> {
    let t = contrast_t_statistic(anova, contrast)?;
    let df = anova.df_within as f64;

    // Two-tailed p-value using t-distribution
    let p = crate::traits::estimator::t_test_p_value(t, df);

    Ok(p)
}

/// Generate standard contrast sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContrastType {
    /// Compare each group to the first group (treatment vs control)
    Treatment,
    /// Helmert contrasts: each group vs mean of previous groups
    Helmert,
    /// Sum contrasts: each group vs grand mean
    Sum,
    /// Polynomial contrasts (for ordered factors)
    Poly,
}

crate::option_enum!(
    ContrastType,
    crate::options::Choices {
        option: "contrast type",
        values: &[
            (&["treatment", "contr.treatment"], ContrastType::Treatment),
            (&["helmert", "contr.helmert"], ContrastType::Helmert),
            (&["sum", "contr.sum"], ContrastType::Sum),
            (&["poly", "polynomial", "contr.poly"], ContrastType::Poly),
        ],
    }
);

/// Generate a standard contrast matrix for k groups.
///
/// k < 2 has no contrasts (R's contr.* stop with 'not enough degrees of freedom to define
/// contrasts'); an empty set is returned. [`ContrastType::Poly`] is [`contr_poly`], and also an
/// empty set where that is an error (more than 95 levels).
pub fn generate_contrasts(k: usize, contrast_type: ContrastType) -> Vec<Vec<f64>> {
    if k < 2 {
        return Vec::new();
    }
    match contrast_type {
        ContrastType::Treatment => {
            // Compare each group (2..k) to group 1
            (1..k)
                .map(|i| {
                    let mut c = vec![0.0; k];
                    c[0] = -1.0;
                    c[i] = 1.0;
                    c
                })
                .collect()
        }
        ContrastType::Helmert => {
            // Compare each group to mean of all previous groups
            (1..k)
                .map(|i| {
                    let mut c = vec![0.0; k];
                    let weight = -1.0 / i as f64;
                    for j in 0..i {
                        c[j] = weight;
                    }
                    c[i] = 1.0;
                    c
                })
                .collect()
        }
        ContrastType::Sum => {
            // Compare each group to grand mean (last group is reference)
            (0..k - 1)
                .map(|i| {
                    let mut c = vec![-1.0 / (k - 1) as f64; k];
                    c[i] = 1.0 - 1.0 / (k - 1) as f64;
                    // Adjust to sum to zero
                    let sum: f64 = c.iter().sum();
                    if sum.abs() > 1e-10 {
                        for val in &mut c {
                            *val -= sum / k as f64;
                        }
                    }
                    c
                })
                .collect()
        }
        ContrastType::Poly => contr_poly(k).unwrap_or_default(),
    }
}

/// Largest number of levels `contr_poly` accepts: R's `contr.poly` stops beyond 95 ("orthogonal
/// polynomials cannot be represented accurately enough for 95 degrees of freedom").
const CONTR_POLY_MAX_LEVELS: usize = 95;

/// R's `contr.poly(k)`: the k - 1 orthonormal polynomial contrasts on the scores 1..k, returned
/// as rows (row j - 1 is R's column j, the degree-j polynomial).
///
/// The centred scores' powers `y^0 .. y^(k-1)` are orthogonalised by Gram-Schmidt with
/// re-orthogonalisation (two passes), normalised, and the constant column dropped. Each column's
/// sign is R's: positive on its own power of the scores. From about degree 25 the raw-power basis
/// is too ill-conditioned for the high-degree columns to be the exact orthogonal polynomials,
/// here as in R (whose columns drift from them earlier), so neither matches them there and the
/// two agree with each other only for the lower degrees.
///
/// # Errors
///
/// `InvalidSpecification` for k < 2 (R: "contrasts not defined for 0 degrees of freedom") and
/// k > 95 (R: "orthogonal polynomials cannot be represented accurately enough for 95 degrees of
/// freedom").
pub fn contr_poly(k: usize) -> EconResult<Vec<Vec<f64>>> {
    if k < 2 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "polynomial contrasts need at least 2 levels, got {k} (R's contr.poly: \
                 contrasts not defined for 0 degrees of freedom)"
            ),
        });
    }
    if k > CONTR_POLY_MAX_LEVELS {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "polynomial contrasts are limited to {CONTR_POLY_MAX_LEVELS} levels, got {k} \
                 (R's contr.poly: orthogonal polynomials cannot be represented accurately \
                 enough for 95 degrees of freedom)"
            ),
        });
    }
    let centre = (k as f64 + 1.0) / 2.0;
    let y: Vec<f64> = (1..=k).map(|i| i as f64 - centre).collect();
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut power = vec![1.0; k];
    for degree in 0..k {
        if degree > 0 {
            for (p, &yi) in power.iter_mut().zip(&y) {
                *p *= yi;
            }
        }
        let mut v = power.clone();
        for _pass in 0..2 {
            for q in &basis {
                let d: f64 = q.iter().zip(&v).map(|(a, b)| a * b).sum();
                for (vi, qi) in v.iter_mut().zip(q) {
                    *vi -= d * qi;
                }
            }
        }
        let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        for vi in v.iter_mut() {
            *vi /= norm;
        }
        basis.push(v);
    }
    basis.remove(0);
    Ok(basis)
}

/// Run se.contrast (convenience wrapper).
pub fn run_se_contrast(
    anova: &AnovaResult,
    contrasts: &[Vec<f64>],
) -> Result<SeContrastResult, String> {
    se_contrast(anova, contrasts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use crate::stats::run_one_way_anova;
    use polars::prelude::*;

    fn create_test_dataset() -> Dataset {
        // Three groups with clear differences: A (mean ~5), B (mean ~7), C (mean ~9)
        let df = df! {
            "value" => [
                4.5, 5.0, 5.5, 4.8, 5.2, 5.0, 4.7, 5.3, 5.1, 4.9,  // Group A: mean ≈ 5
                6.5, 7.0, 7.5, 6.8, 7.2, 7.0, 6.7, 7.3, 7.1, 6.9,  // Group B: mean ≈ 7
                8.5, 9.0, 9.5, 8.8, 9.2, 9.0, 8.7, 9.3, 9.1, 8.9   // Group C: mean ≈ 9
            ],
            "group" => [
                "A", "A", "A", "A", "A", "A", "A", "A", "A", "A",
                "B", "B", "B", "B", "B", "B", "B", "B", "B", "B",
                "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"
            ]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_se_contrast_basic() {
        let dataset = create_test_dataset();
        let anova = run_one_way_anova(&dataset, "value", "group").unwrap();

        // Contrast: A vs B (first two groups)
        let k = anova.groups.len();
        let mut contrast = vec![0.0; k];
        contrast[0] = 1.0;
        contrast[1] = -1.0;
        let contrasts = vec![contrast];

        let result = se_contrast(&anova, &contrasts).unwrap();

        assert_eq!(result.se.len(), 1);
        // SE should be positive and reasonable
        assert!(result.se[0] > 0.0);
        assert!(result.se[0] < 1.0);
    }

    #[test]
    fn test_se_contrast_multiple() {
        let dataset = create_test_dataset();
        let anova = run_one_way_anova(&dataset, "value", "group").unwrap();

        let k = anova.groups.len();
        // Multiple contrasts
        let contrasts = vec![
            {
                let mut c = vec![0.0; k];
                c[0] = 1.0;
                c[1] = -1.0;
                c
            }, // A vs B
            {
                let mut c = vec![0.0; k];
                c[1] = 1.0;
                c[2] = -1.0;
                c
            }, // B vs C
            {
                let mut c = vec![0.0; k];
                c[0] = 0.5;
                c[1] = 0.5;
                c[2] = -1.0;
                c
            }, // (A+B)/2 vs C
        ];
        let result = se_contrast(&anova, &contrasts).unwrap();

        assert_eq!(result.se.len(), 3);
        for se in &result.se {
            assert!(*se > 0.0);
        }
    }

    #[test]
    fn test_estimate_contrast() {
        let dataset = create_test_dataset();
        let anova = run_one_way_anova(&dataset, "value", "group").unwrap();

        let k = anova.groups.len();
        // Contrast: first group vs second group
        let mut contrast = vec![0.0; k];
        contrast[0] = 1.0;
        contrast[1] = -1.0;
        let estimate = estimate_contrast(&anova, &contrast).unwrap();

        // The estimate should be negative (A mean < B mean)
        assert!(estimate < 0.0);
        // Should be close to -2 (5 - 7)
        assert!((estimate + 2.0).abs() < 0.5);
    }

    #[test]
    fn test_contrast_t_statistic() {
        let dataset = create_test_dataset();
        let anova = run_one_way_anova(&dataset, "value", "group").unwrap();

        let k = anova.groups.len();
        let mut contrast = vec![0.0; k];
        contrast[0] = 1.0;
        contrast[1] = -1.0;
        let t = contrast_t_statistic(&anova, &contrast).unwrap();

        // t should be negative (A < B) and significant
        assert!(t < 0.0);
        assert!(t.abs() > 2.0); // Should be clearly significant
    }

    #[test]
    fn test_generate_contrasts_treatment() {
        let contrasts = generate_contrasts(3, ContrastType::Treatment);

        assert_eq!(contrasts.len(), 2); // k-1 contrasts

        // First contrast: group 2 vs group 1
        assert!((contrasts[0][0] - (-1.0)).abs() < 1e-10);
        assert!((contrasts[0][1] - 1.0).abs() < 1e-10);
        assert!((contrasts[0][2] - 0.0).abs() < 1e-10);

        // Each should sum to zero
        for c in &contrasts {
            let sum: f64 = c.iter().sum();
            assert!(sum.abs() < 1e-10);
        }
    }

    #[test]
    fn test_generate_contrasts_helmert() {
        let contrasts = generate_contrasts(3, ContrastType::Helmert);

        assert_eq!(contrasts.len(), 2);

        // Helmert: each group vs mean of previous
        // c1: group 2 vs group 1 -> [-1, 1, 0]
        // c2: group 3 vs mean(1,2) -> [-0.5, -0.5, 1]
        assert!((contrasts[0][0] - (-1.0)).abs() < 1e-10);
        assert!((contrasts[0][1] - 1.0).abs() < 1e-10);

        assert!((contrasts[1][0] - (-0.5)).abs() < 1e-10);
        assert!((contrasts[1][1] - (-0.5)).abs() < 1e-10);
        assert!((contrasts[1][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_contrast_wrong_length() {
        let dataset = create_test_dataset();
        let anova = run_one_way_anova(&dataset, "value", "group").unwrap();

        // Wrong number of contrast coefficients
        let contrasts = vec![vec![1.0, -1.0]]; // Only 2 elements for 3 groups
        let result = se_contrast(&anova, &contrasts);

        assert!(result.is_err());
    }

    #[test]
    fn test_contrast_not_sum_zero() {
        let dataset = create_test_dataset();
        let anova = run_one_way_anova(&dataset, "value", "group").unwrap();

        // Contrast doesn't sum to zero
        let k = anova.groups.len();
        let contrasts = vec![vec![1.0; k]];
        let result = se_contrast(&anova, &contrasts);

        assert!(result.is_err());
    }

    /// STAT-84: k - 1 underflowed for Sum with k = 0; every type now returns no contrasts for k < 2.
    #[test]
    fn test_contrasts_k0() {
        for t in [
            ContrastType::Treatment,
            ContrastType::Helmert,
            ContrastType::Sum,
            ContrastType::Poly,
        ] {
            assert!(generate_contrasts(0, t).is_empty(), "{t:?}");
            assert!(generate_contrasts(1, t).is_empty(), "{t:?}");
        }
        assert_eq!(
            generate_contrasts(2, ContrastType::Sum),
            vec![vec![0.5, -0.5]]
        ); // unchanged for k >= 2
    }

    /// contr.poly(k) for k = 2, 3, 4, 6 (R's columns, column-major), and se.contrast and the
    /// estimates of the contr.poly(4) columns on four numeric dose levels in factor order.
    #[test]
    fn test_validate_se_contrast_poly_t4() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_secontrast_poly_t4");
        let mut check = Check::new(&case);
        for k in [2usize, 3, 4, 6] {
            let flat: Vec<f64> = contr_poly(k).unwrap().into_iter().flatten().collect();
            check.vector(&format!("poly{k}"), &flat, Tol::coef(24));
        }
        let anova = run_one_way_anova(&ds, "y", "dose").unwrap();
        let names: Vec<&str> = anova.groups.iter().map(|g| g.group.as_str()).collect();
        assert_eq!(names, ["5", "10", "20", "40"]);
        let cp = contr_poly(4).unwrap();
        let r = se_contrast(&anova, &cp).unwrap();
        let est: Vec<f64> = cp
            .iter()
            .map(|c| estimate_contrast(&anova, c).unwrap())
            .collect();
        check
            .vector("se4", &r.se, Tol::se(24))
            .vector("est4", &est, Tol::coef(24))
            .scalar("mse", r.mse, Tol::stat(24))
            .scalar("df_mse", r.df_mse, Tol::exact())
            .finish();
    }

    #[test]
    fn test_contr_poly_bounds() {
        for k in [0usize, 1, 96] {
            assert!(
                matches!(contr_poly(k), Err(EconError::InvalidSpecification { .. })),
                "k = {k}"
            );
        }
        assert!(generate_contrasts(96, ContrastType::Poly).is_empty());
        for k in 2..=20usize {
            let rows = contr_poly(k).unwrap();
            assert_eq!(rows.len(), k - 1);
            for (i, a) in rows.iter().enumerate() {
                assert_eq!(a.len(), k);
                assert!(a.iter().sum::<f64>().abs() < 1e-12, "k = {k}, row {i} sum");
                for (j, b) in rows.iter().enumerate() {
                    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((dot - want).abs() < 1e-12, "k = {k}, rows {i}, {j}: {dot}");
                }
            }
            // R's sign: the degree-j column rises with the j-th power, so the last cell is positive
            for (j, row) in rows.iter().enumerate() {
                assert!(row[k - 1] > 0.0, "k = {k}, degree {}", j + 1);
            }
        }
        assert_eq!(
            generate_contrasts(4, ContrastType::Poly),
            contr_poly(4).unwrap()
        );
    }

    #[test]
    fn test_contrast_type_parses_aliases_and_rejects_unknown() {
        assert_eq!(ContrastType::choices().validate(), Ok(()));
        assert_eq!(
            "contr.helmert".parse::<ContrastType>().unwrap(),
            ContrastType::Helmert
        );
        assert_eq!(
            "Polynomial".parse::<ContrastType>().unwrap(),
            ContrastType::Poly
        );
        assert_eq!("SUM".parse::<ContrastType>().unwrap(), ContrastType::Sum);
        assert!("deviation".parse::<ContrastType>().is_err());
    }
}
