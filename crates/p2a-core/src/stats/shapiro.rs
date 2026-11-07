//! Shapiro-Wilk Test for Normality.
//!
//! Tests the null hypothesis that a sample comes from a normally distributed population.
//!
//! # References
//!
//! - Shapiro, S. S. & Wilk, M. B. (1965). "An analysis of variance test for normality
//!   (complete samples)". *Biometrika*, 52(3-4), 591-611.
//! - Royston, J. P. (1982). "An extension of Shapiro and Wilk's W test for normality
//!   to large samples". *Journal of the Royal Statistical Society Series C*, 31(2), 115-124.
//! - Royston, P. (1995). "Remark AS R94: A remark on Algorithm AS 181: The W-test
//!   for normality". *Journal of the Royal Statistical Society Series C*, 44(4), 547-551.
//! - R Core Team. `stats::shapiro.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/shapiro.test.html>
//! - R Core Team. `src/library/stats/src/swilk.c`, the C implementation of AS R94 that
//!   `shapiro.test` calls; this module is a port of it.
//!
//! # Mathematical Background
//!
//! ## W statistic
//!
//! For the sorted sample x₍₁₎ ≤ … ≤ x₍ₙ₎,
//!
//! ```text
//! W = (Σ aᵢ x₍ᵢ₎)² / Σ (xᵢ - x̄)²
//! ```
//!
//! The exact coefficients a = m'V⁻¹ / ‖m'V⁻¹‖ need the covariance matrix V of the normal
//! order statistics, which neither R nor p2a computes. Both use Royston's approximation
//! (Royston 1992, 1995; AS R94):
//!
//! - mᵢ = Φ⁻¹((i - 3/8) / (n + 1/4)), the Blom scores;
//! - the largest coefficient aₙ, and for n > 5 also aₙ₋₁, is the normalised Blom score
//!   plus a polynomial in 1/√n; the other coefficients are the Blom scores rescaled so that
//!   Σ aᵢ² = 1. For n = 3 the coefficients are exact, (-√½, 0, √½);
//! - W is computed as the squared correlation of the data and the coefficients, through
//!   1 - W, which keeps its precision for W close to 1.
//!
//! ## P-value
//!
//! - n = 3: the exact distribution, p = (6/π)(asin √W - π/3).
//! - 4 ≤ n ≤ 11: with γ = -2.273 + 0.459 n, y = -ln(γ - ln(1 - W)) is approximately
//!   normal; its mean and log standard deviation are cubic polynomials in n, and p is the
//!   upper normal tail at y.
//! - 12 ≤ n ≤ 5000: ln(1 - W) is approximately normal; its mean is a cubic and its log
//!   standard deviation a quadratic polynomial in ln n, and p is the upper normal tail.
//!
//! As `shapiro.test` does, missing values (NaN) are dropped, a sample whose range is below
//! 1 is divided by its range, and a sample with a range below 1e-10 ("all values
//! identical"), with infinite values, or with fewer than 3 or more than 5000 values is
//! refused.

use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::data::Dataset;
use crate::data::columns::{MissingSummary, sample_column};
use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::traits::SignificanceLevel;
use crate::traits::estimator::normal_quantile;

// ═══════════════════════════════════════════════════════════════════════════════
// Constants of swilk.c (Royston 1995, AS R94)
// ═══════════════════════════════════════════════════════════════════════════════

/// Coefficients of gamma(n) for 4 <= n <= 11 (swilk.c `g`).
const G: [f64; 2] = [-2.273, 0.459];
/// Polynomial correction of a_n in 1/sqrt(n) (swilk.c `c1`).
const C1: [f64; 6] = [0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056];
/// Polynomial correction of a_(n-1) in 1/sqrt(n) (swilk.c `c2`).
const C2: [f64; 6] = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];
/// Mean of -ln(gamma - ln(1 - W)) in n, 4 <= n <= 11 (swilk.c `c3`).
const C3: [f64; 4] = [0.544, -0.39978, 0.025054, -6.714e-4];
/// Log standard deviation of the same in n (swilk.c `c4`).
const C4: [f64; 4] = [1.3822, -0.77857, 0.062767, -0.0020322];
/// Mean of ln(1 - W) in ln(n), n >= 12 (swilk.c `c5`).
const C5: [f64; 4] = [-1.5861, -0.31082, -0.083751, 0.0038915];
/// Log standard deviation of the same in ln(n) (swilk.c `c6`).
const C6: [f64; 3] = [-0.4803, -0.082676, 0.0030302];
/// swilk.c's literal for 6/pi (gives R's 0.99999999999999334 at W = 1 for n = 3).
const PI6: f64 = 1.909_859_317_102_74;
/// swilk.c's literal for asin(sqrt(3/4)) = pi/3.
const STQR: f64 = 1.047_197_551_196_60;
/// shapiro.test refuses a sample whose range is below this ("all 'x' values are identical").
const SMALL_RANGE: f64 = 1e-10;

/// Minimum and maximum sample sizes.
const MIN_N: usize = 3;
const MAX_N: usize = 5000;

// ═══════════════════════════════════════════════════════════════════════════════
// Result Struct
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of the Shapiro-Wilk normality test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapiroWilkResult {
    /// Test name
    pub test_name: String,

    /// W test statistic (0 < W ≤ 1)
    /// Values close to 1 indicate normality
    pub w_statistic: f64,

    /// P-value for the test
    /// Small p-values indicate evidence against normality
    pub p_value: f64,

    /// Significance level
    pub significance: SignificanceLevel,

    /// Sample size
    pub n: usize,

    /// Whether the null hypothesis (normality) is rejected at α = 0.05
    pub reject_normality: bool,

    /// Rows dropped because of missing values (dataset wrapper; nothing for a slice).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl std::fmt::Display for ShapiroWilkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;
        writeln!(
            f,
            "W = {:.6}, p-value = {:.6} {}",
            self.w_statistic,
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(f)?;
        writeln!(f, "Sample size: n = {}", self.n)?;
        writeln!(f)?;
        writeln!(f, "Null hypothesis: Data is normally distributed")?;
        writeln!(
            f,
            "Alternative hypothesis: Data is not normally distributed"
        )?;
        writeln!(f)?;
        if self.reject_normality {
            writeln!(
                f,
                "Conclusion: Reject H₀ at α = 0.05 (evidence against normality)"
            )?;
        } else {
            writeln!(
                f,
                "Conclusion: Fail to reject H₀ at α = 0.05 (no evidence against normality)"
            )?;
        }
        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Perform the Shapiro-Wilk test for normality.
///
/// Tests the null hypothesis that the data comes from a normal distribution.
///
/// # Arguments
/// * `x` - Sample data (must contain between 3 and 5000 non-missing values)
///
/// # Returns
/// * `ShapiroWilkResult` containing the W statistic and p-value
///
/// # Errors
/// As `shapiro.test`: fewer than 3 or more than 5000 non-missing values, or a range below
/// 1e-10 ("all values identical"). An infinite value is refused too, and so is a sample of
/// finite values whose range (max - min) overflows. NaN values are dropped; a sample whose
/// range is below 1 is divided by its range first.
///
/// # Example
/// ```ignore
/// let x = vec![0.1, 0.5, 0.2, 0.8, 0.4, 0.9, 0.3, 0.7, 0.6];
/// let result = shapiro_wilk_test(&x)?;
/// println!("W = {:.4}, p = {:.4}", result.w_statistic, result.p_value);
/// ```
///
/// # References
/// - R equivalent: `shapiro.test(x)`
pub fn shapiro_wilk_test(x: &[f64]) -> EconResult<ShapiroWilkResult> {
    // Missing values (NaN) are dropped; an infinite value is refused (R's shapiro.test would
    // compute on a range of Inf).
    let mut n = 0usize;
    for v in x.iter().filter(|v| !v.is_nan()) {
        if v.is_infinite() {
            return Err(EconError::InvalidSpecification {
                message: "Shapiro-Wilk test: x contains infinite values".to_string(),
            });
        }
        n += 1;
    }
    if n < MIN_N {
        return Err(EconError::InsufficientData {
            required: MIN_N,
            provided: n,
            context: "Shapiro-Wilk test requires at least 3 non-missing values".to_string(),
        });
    }
    if n > MAX_N {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Shapiro-Wilk test: sample size must be between {MIN_N} and {MAX_N} (got {n})"
            ),
        });
    }

    // Order statistics (n <= 5000 here).
    let mut sorted: Vec<f64> = resource::try_with_capacity(n, "Shapiro-Wilk sample")?;
    sorted.extend(x.iter().copied().filter(|v| !v.is_nan()));
    sorted.sort_by(f64::total_cmp);

    // shapiro.test: refuse a constant sample, rescale a small range (R PR#1768).
    let range = sorted[n - 1] - sorted[0];
    if !range.is_finite() {
        // Finite values whose difference overflows: every x / range would be 0 and W would
        // come out as 1 (R gives W = NaN). Refused, as an infinite value is.
        return Err(EconError::InvalidSpecification {
            message: "Shapiro-Wilk test: the range of x (max - min) overflows; rescale the \
                      values"
                .to_string(),
        });
    }
    if range < SMALL_RANGE {
        return Err(EconError::InvalidSpecification {
            message: "Shapiro-Wilk test: all 'x' values are identical".to_string(),
        });
    }
    if range < 1.0 {
        for v in &mut sorted {
            *v /= range;
        }
    }

    let (w, p_value) = swilk(&sorted)?;

    Ok(ShapiroWilkResult {
        test_name: "Shapiro-Wilk Normality Test".to_string(),
        w_statistic: w,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        n,
        reject_normality: p_value < 0.05,
        missing: MissingSummary::none(n),
    })
}

/// Perform Shapiro-Wilk test using a dataset column.
///
/// Missing values (null or NaN) are dropped, as `shapiro.test` does
/// (`x[complete.cases(x)]`), and counted in `missing`; integer, decimal and boolean columns
/// are accepted; infinite values are refused (`EconError::NonFiniteValues`), as is a text
/// column (`EconError::NonNumericColumn`).
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `column` - Name of the column to test
///
/// # Example
/// ```ignore
/// let result = run_shapiro_wilk(&dataset, "residuals")?;
/// if result.reject_normality {
///     println!("Warning: Data appears non-normal");
/// }
/// ```
pub fn run_shapiro_wilk(dataset: &Dataset, column: &str) -> EconResult<ShapiroWilkResult> {
    let (x, missing) = sample_column(dataset.df(), column, "Shapiro-Wilk")?;
    let mut result = shapiro_wilk_test(&x)?;
    result.missing = missing;
    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// swilk.c (AS R94)
// ═══════════════════════════════════════════════════════════════════════════════

/// `cc[0] + cc[1] x + ... + cc[k] x^k`, evaluated in the order of swilk.c's `poly`.
fn poly(cc: &[f64], x: f64) -> f64 {
    let last = cc.len() - 1;
    let mut ret = cc[0];
    if cc.len() > 1 {
        let mut p = x * cc[last];
        for &c in cc[1..last].iter().rev() {
            p = (p + c) * x;
        }
        ret += p;
    }
    ret
}

/// W and its p-value for a sorted sample with 3 <= n <= 5000 and a range of at least
/// 1e-10: R's `swilk.c`, line by line (its order of additions included).
fn swilk(x: &[f64]) -> EconResult<(f64, f64)> {
    let n = x.len();
    let nn2 = n / 2;
    let an = n as f64;

    // Coefficients a[0..nn2] (swilk.c's 1-based a[1..=nn2]), for the upper half.
    let mut a: Vec<f64> = resource::try_vec(nn2, 0.0, "Shapiro-Wilk coefficients")?;
    if n == 3 {
        a[0] = std::f64::consts::FRAC_1_SQRT_2;
    } else {
        let an25 = an + 0.25;
        let mut summ2 = 0.0;
        for (i, ai) in a.iter_mut().enumerate() {
            let m = normal_quantile(((i + 1) as f64 - 0.375) / an25)?;
            *ai = m;
            summ2 += m * m;
        }
        summ2 *= 2.0;
        let ssumm2 = summ2.sqrt();
        let rsn = 1.0 / an.sqrt();
        let a1 = poly(&C1, rsn) - a[0] / ssumm2;

        // Normalise the coefficients that are not replaced by a polynomial.
        let (i1, fac) = if n > 5 {
            let a2 = -a[1] / ssumm2 + poly(&C2, rsn);
            let fac = ((summ2 - 2.0 * (a[0] * a[0]) - 2.0 * (a[1] * a[1]))
                / (1.0 - 2.0 * (a1 * a1) - 2.0 * (a2 * a2)))
                .sqrt();
            a[1] = a2;
            (2, fac)
        } else {
            let fac = ((summ2 - 2.0 * (a[0] * a[0])) / (1.0 - 2.0 * (a1 * a1))).sqrt();
            (1, fac)
        };
        a[0] = a1;
        for ai in &mut a[i1..] {
            *ai /= -fac;
        }
    }

    // W as the squared correlation of the data and the coefficients (swilk.c's two loops;
    // i and j are its indices, sign(i - j) * a[min(i, j)] the coefficient of x[i]).
    let range = x[n - 1] - x[0];
    let mut sx = x[0] / range;
    let mut sa = -a[0];
    let mut i = 1usize;
    let mut j = n - 1;
    while i < n {
        sx += x[i] / range;
        i += 1;
        if i != j {
            let sign = if i > j { 1.0 } else { -1.0 };
            sa += sign * a[i.min(j) - 1];
        }
        j -= 1;
    }
    sa /= an;
    sx /= an;

    let (mut ssa, mut ssx, mut sax) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let j = n - 1 - i;
        let asa = if i != j {
            let sign = if i > j { 1.0 } else { -1.0 };
            sign * a[i.min(j)] - sa
        } else {
            -sa
        };
        let xsx = x[i] / range - sx;
        ssa += asa * asa;
        ssx += xsx * xsx;
        sax += asa * xsx;
    }

    // w1 = 1 - W, computed directly to keep its precision for W near 1. The guard against
    // a negative rounding result keeps ln and asin defined; it changes no regular sample.
    let ssassx = (ssa * ssx).sqrt();
    let w1 = ((ssassx - sax) * (ssassx + sax) / (ssa * ssx)).max(0.0);
    let w = 1.0 - w1;

    // Significance level.
    if n == 3 {
        let pw = PI6 * (w.min(1.0).sqrt().asin() - STQR);
        return Ok((w, pw.max(0.0)));
    }
    let normal = Normal::standard();
    let y = w1.ln();
    let xx = an.ln();
    let (y, m, s) = if n <= 11 {
        let gamma = poly(&G, an);
        if y >= gamma {
            // Unreachable for a valid sample (it needs W far below its minimum); kept as in
            // swilk.c.
            return Ok((w, 1e-99));
        }
        (-(gamma - y).ln(), poly(&C3, an), poly(&C4, an).exp())
    } else {
        (y, poly(&C5, xx), poly(&C6, xx).exp())
    };
    // Upper tail, as pnorm(y, m, s, lower.tail = FALSE).
    Ok((w, normal.sf((y - m) / s)))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_shapiro_wilk_normal_data() {
        // Data that should appear normal
        let x = vec![
            0.1, 0.5, 0.2, 0.8, 0.4, 0.9, 0.3, 0.7, 0.6, 0.45, 0.55, 0.35, 0.65, 0.25, 0.75, 0.15,
            0.85, 0.42, 0.58, 0.38,
        ];

        let result = shapiro_wilk_test(&x).unwrap();

        assert!(result.w_statistic > 0.0 && result.w_statistic <= 1.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.n, 20);
    }

    #[test]
    fn test_shapiro_wilk_non_normal_data() {
        // Highly skewed data (exponential-like)
        let x = vec![
            0.1, 0.2, 0.3, 0.5, 0.8, 1.3, 2.1, 3.4, 5.5, 8.9, 14.4, 23.3, 37.7, 61.0, 98.7,
        ];

        let result = shapiro_wilk_test(&x).unwrap();

        // Exponential data should have low W and small p-value
        assert!(result.w_statistic < 0.9);
        assert_eq!(result.n, 15);
    }

    #[test]
    fn test_shapiro_wilk_minimum_sample() {
        let x = vec![1.0, 2.0, 3.0];
        let result = shapiro_wilk_test(&x).unwrap();

        assert_eq!(result.n, 3);
        assert!(result.w_statistic > 0.0);
    }

    #[test]
    fn test_shapiro_wilk_insufficient_data() {
        let x = vec![1.0, 2.0];
        let result = shapiro_wilk_test(&x);
        assert!(matches!(result, Err(EconError::InsufficientData { .. })));
    }

    #[test]
    fn test_shapiro_wilk_zero_variance() {
        let x = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let result = shapiro_wilk_test(&x);
        assert!(matches!(
            result,
            Err(EconError::InvalidSpecification { .. })
        ));
    }

    #[test]
    fn test_shapiro_wilk_handles_nan() {
        let x = vec![1.0, 2.0, f64::NAN, 3.0, 4.0, 5.0];
        let result = shapiro_wilk_test(&x).unwrap();
        // NaN should be filtered out, leaving 5 observations
        assert_eq!(result.n, 5);
    }

    #[test]
    fn test_run_shapiro_wilk_from_dataset() {
        let df = df! {
            "values" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_shapiro_wilk(&dataset, "values").unwrap();
        assert_eq!(result.n, 10);
    }

    // ========================================================================
    // Validation against R (case stat_shapiro_wilk, validation/r/cases/stats_onesample.R)
    // ========================================================================

    const SHAPIRO_COLUMNS: [&str; 17] = [
        "n3_124",
        "n3_123",
        "n4",
        "n5",
        "n8_rnorm",
        "n10_unif",
        "n11_rnorm",
        "n12_rnorm",
        "n15_exp",
        "n20_rnorm",
        "n30_exp",
        "n50_rnorm",
        "n100_qnorm",
        "n200_exp",
        "n40_ties",
        "n25_int",
        "n5000_rnorm",
    ];

    /// STAT-1: W and p of the swilk.c port against shapiro.test, n = 3 to 5000 (ties,
    /// integer data).
    #[test]
    fn test_validate_shapiro_wilk_swilk() {
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_shapiro_wilk");
        let mut check = Check::new(&case);
        for col in SHAPIRO_COLUMNS {
            let (x, _) = numeric_column(ds.df(), col, NaPolicy::Omit).unwrap();
            let r = shapiro_wilk_test(&x).unwrap_or_else(|e| panic!("{col}: {e}"));
            check
                .scalar(&format!("{col}_w"), r.w_statistic, Tol::stat(r.n))
                .scalar(&format!("{col}_p"), r.p_value, Tol::p_value(r.n))
                .scalar(&format!("{col}_n"), r.n as f64, Tol::exact());
            assert_eq!(
                r.reject_normality,
                case.scalar(&format!("{col}_p")) < 0.05,
                "{col}"
            );
        }
        check.finish();
    }

    fn generated(name: &str) -> Dataset {
        let p = p2a_validation::repo_path(&format!("validation/datasets/generated/{name}.csv"));
        Dataset::new(crate::data::DataLoader::load_csv(p).unwrap())
    }

    /// X-2/X-8: NA-padded Float64 columns and Int64 columns, through run_shapiro_wilk.
    #[test]
    fn test_validate_shapiro_wilk_columns() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_shapiro_wilk");
        let rows = ds.df().height(); // 5000
        // Float64 columns first (pre-fix: the padding was read as 4995 zeros), Int64 columns
        // last (pre-fix: NonNumericColumn), so a pre-fix run reports the wrong numbers first.
        let (int_cols, float_cols): (Vec<&str>, Vec<&str>) = SHAPIRO_COLUMNS
            .into_iter()
            .partition(|c| ["n3_124", "n3_123", "n25_int"].contains(c));
        for group in [float_cols, int_cols] {
            let mut check = Check::new(&case);
            for col in group {
                let r = run_shapiro_wilk(&ds, col).unwrap_or_else(|e| panic!("{col}: {e}"));
                check
                    .scalar(&format!("{col}_w"), r.w_statistic, Tol::stat(r.n))
                    .scalar(&format!("{col}_p"), r.p_value, Tol::p_value(r.n))
                    .scalar(&format!("{col}_n"), r.n as f64, Tol::exact());
                assert_eq!(
                    (r.missing.n_total, r.missing.n_used, r.missing.n_dropped),
                    (rows, r.n, rows - r.n),
                    "{col}"
                );
                let want = if r.n == rows {
                    vec![]
                } else {
                    vec![(col.to_string(), rows - r.n)]
                };
                assert_eq!(r.missing.by_column, want, "{col}");
            }
            check.finish();
        }
    }

    #[test]
    fn test_validate_na_shapiro_wilk_columns() {
        let (case, ds) = crate::test_support::load_case("stat_shapiro_wilk");
        // a NaN cell counts as missing (the harness cannot write NaN: the one inline fixture)
        let nan = Dataset::new(df!("x" => [1.2, f64::NAN, 2.3, 3.1, 4.5, 5.2]).unwrap());
        let r = run_shapiro_wilk(&nan, "x").unwrap();
        assert_eq!(r.n, 5);
        assert_eq!(r.missing.by_column, vec![("x".to_string(), 1)]);
        assert!(p2a_validation::Tol::stat(5).allows(r.w_statistic, case.scalar("n5_w")));
        let text = r.to_string();
        assert_eq!(
            text.lines().last().unwrap(),
            format!("Missing values: {}", r.missing.note().unwrap())
        );
        // complete column: no note
        assert!(
            !run_shapiro_wilk(&ds, "n5000_rnorm")
                .unwrap()
                .to_string()
                .contains("Missing values")
        );
        // text column and absent column
        let ts = generated("two_sample_na");
        assert!(
            matches!(run_shapiro_wilk(&ts, "g"), Err(EconError::NonNumericColumn { column }) if column == "g")
        );
        assert!(
            matches!(run_shapiro_wilk(&ds, "nope"), Err(EconError::ColumnNotFound { column, .. }) if column == "nope")
        );
    }

    /// shapiro.test rescales a sample whose range is below 1 (PR#1768) and refuses only a
    /// range below 1e-10: W and p do not depend on the scale.
    #[test]
    fn test_validate_shapiro_wilk_rescales_small_range() {
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::Tol;
        let (case, ds) = crate::test_support::load_case("stat_shapiro_wilk");
        let (n5, _) = numeric_column(ds.df(), "n5", NaPolicy::Omit).unwrap();
        let tiny: Vec<f64> = n5.iter().map(|v| v * 1e-6).collect();
        let r = shapiro_wilk_test(&tiny).unwrap();
        assert!(
            Tol::stat(5).allows(r.w_statistic, case.scalar("n5_w")),
            "{}",
            r.w_statistic
        );
        assert!(
            Tol::p_value(5).allows(r.p_value, case.scalar("n5_p")),
            "{}",
            r.p_value
        );
        // Range 2e-10 (sum of squares 3e-20) is a valid sample for R: it is (0, 0, 0, 1)
        // rescaled, W 0.62977626491370753, p 0.0012407259319772808 for both (R 4.3.3).
        let a = shapiro_wilk_test(&[0.0, 0.0, 0.0, 2e-10]).unwrap();
        let b = shapiro_wilk_test(&[0.0, 0.0, 0.0, 1.0]).unwrap();
        assert!(
            (a.w_statistic - b.w_statistic).abs() <= 1e-12
                && (a.p_value - b.p_value).abs() <= 1e-12 * b.p_value,
            "{a:?} vs {b:?}"
        );
    }

    #[test]
    fn test_shapiro_wilk_input_errors() {
        // R: "all 'x' values are identical" (range < 1e-10), "sample size must be between 3
        // and 5000"
        let same = shapiro_wilk_test(&[5.0, 5.0 + 1e-12, 5.0, 5.0]).unwrap_err();
        assert!(
            matches!(&same, EconError::InvalidSpecification { message } if message.contains("identical")),
            "{same}"
        );
        assert!(matches!(
            shapiro_wilk_test(&[1.0, f64::NAN, 2.0]),
            Err(EconError::InsufficientData { provided: 2, .. })
        ));
        let big: Vec<f64> = (0..5001).map(|i| (i as f64 * 0.37).sin()).collect();
        let e = shapiro_wilk_test(&big).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("5000")),
            "{e}"
        );
        let inf = shapiro_wilk_test(&[1.0, 2.0, f64::INFINITY, 4.0]).unwrap_err();
        assert!(
            matches!(&inf, EconError::InvalidSpecification { message } if message.contains("infinite")),
            "{inf}"
        );
    }

    #[test]
    fn test_shapiro_wilk_refuses_overflowing_range() {
        // Finite values whose range (max - min) is not finite: every x / range is 0, W came out
        // as 1 and p as 1. R's shapiro.test(c(-1e308, 0, 1e308, 5)) gives W = NaN, p = NA.
        for x in [
            vec![-1e308, 0.0, 1e308, 5.0],
            vec![-f64::MAX, 1.0, 2.5, f64::MAX, -3.0],
        ] {
            let e = shapiro_wilk_test(&x).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.contains("range") && message.contains("overflows")),
                "{x:?}: {e}"
            );
        }
        // A wide range that is finite is computed.
        let wide = shapiro_wilk_test(&[-1e307, 0.0, 1e307, 5.0, 2e306]).unwrap();
        assert!(
            wide.w_statistic.is_finite() && wide.w_statistic < 1.0,
            "{wide:?}"
        );
    }

    #[test]
    fn test_shapiro_all_missing_column_is_named() {
        let ds =
            Dataset::new(df! { "x" => [1.2, 0.4, 2.2], "y" => [None::<f64>, None, None] }.unwrap());
        let e = run_shapiro_wilk(&ds, "y").unwrap_err().to_string();
        assert!(e.contains("'y'") && e.contains("Shapiro-Wilk"), "{e}");
    }
}
