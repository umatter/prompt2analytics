//! Tukey's resistant line fitting.
//!
//! This module implements Tukey's resistant line, a robust regression method that uses
//! medians instead of means, making it resistant to outliers. The method divides data
//! into three groups and uses group medians to determine the slope and intercept.
//!
//! # Mathematical Background
//!
//! The algorithm is R's `stats::line` (R >= 3.5, `tukeyline`):
//!
//! 1. **Thirds**: the left third is `x <= q(1/3)`, the right third `x >= q(2/3)`, with
//!    `q(p) = (z[floor((n - 1) p)] + z[ceil((n - 1) p)]) / 2` on the sorted x `z`
//! 2. **Slope**: from 0, each of `iter` steps adds
//!    `(median(r_R) - median(r_L)) / (median(x_R) - median(x_L))`, `r = y - slope x`
//! 3. **Intercept**: `median(y - slope x)`
//!
//! ## Resistance to Outliers
//!
//! Because the method uses medians rather than means, up to 1/3 of the data can
//! be outliers without affecting the fitted line. This gives a breakdown point
//! of approximately 33%.
//!
//! ## Comparison with OLS
//!
//! | Property | OLS | Resistant Line |
//! |----------|-----|----------------|
//! | Efficiency (normal data) | 100% | ~64% |
//! | Breakdown point | 0% | ~33% |
//! | Influence of outliers | High | None (for < 1/3 outliers) |
//!
//! # References
//!
//! - Tukey, J.W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
//!   ISBN: 978-0201076165. Chapter 10: Resistant Lines.
//!
//! - Velleman, P.F., & Hoaglin, D.C. (1981). *Applications, Basics, and Computing
//!   of Exploratory Data Analysis*. Duxbury Press. ISBN: 978-0871502537.
//!
//! - Emerson, J.D., & Hoaglin, D.C. (1983). Resistant lines for y versus x.
//!   In D.C. Hoaglin, F. Mosteller, & J.W. Tukey (Eds.), *Understanding Robust
//!   and Exploratory Data Analysis* (pp. 129-165). Wiley.
//!
//! - Mosteller, F., & Tukey, J.W. (1977). *Data Analysis and Regression*.
//!   Addison-Wesley. ISBN: 978-0201048544.
//!
//! Pairs with a missing (NaN) or infinite x or y are deleted first (R's `line()` uses the
//! complete pairs too) and reported in [`LineResult::missing`].
//!
//! The fit is R's `stats::line()` (R >= 3.5): case `line_tukey_t4` pins it.

use serde::{Deserialize, Serialize};

use crate::data::columns::MissingSummary;

/// Result of Tukey's resistant line fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineResult {
    /// Intercept coefficient
    pub intercept: f64,
    /// Slope coefficient
    pub slope: f64,
    /// Fitted values of the pairs used, in input order
    pub fitted: Vec<f64>,
    /// Residuals (y - fitted) of the pairs used, in input order
    pub residuals: Vec<f64>,
    /// Number of pairs used (pairs with a missing or infinite value are dropped)
    pub n: usize,
    /// Number of polishing iterations performed
    pub iter: usize,
    /// The x values used (the kept pairs, in input order)
    pub x: Vec<f64>,
    /// The y values used (the kept pairs, in input order)
    pub y: Vec<f64>,
    /// Pairs dropped because x or y was missing or infinite (`by_column` names `x`
    /// and `y`); absent from JSON when nothing was dropped.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl LineResult {
    /// Get coefficients as a tuple (intercept, slope)
    pub fn coef(&self) -> (f64, f64) {
        (self.intercept, self.slope)
    }

    /// Predict y values for new x values
    pub fn predict(&self, new_x: &[f64]) -> Vec<f64> {
        new_x
            .iter()
            .map(|&x| self.intercept + self.slope * x)
            .collect()
    }
}

/// Fewest complete (x, y) pairs [`line`] fits.
pub const MIN_PAIRS: usize = 3;

/// Fit Tukey's resistant line to data, as R's `stats::line` (R >= 3.5, `tukeyline`).
///
/// With `z` the sorted x and `q(p) = (z[floor((n - 1) p)] + z[ceil((n - 1) p)]) / 2`, the left
/// third is `x <= q(1/3)` and the right third `x >= q(2/3)`; `xL` and `xR` are the medians of x
/// in them. Starting from slope 0, each of the `iter` steps adds
/// `(median of the right third's residuals - median of the left third's) / (xR - xL)` to the
/// slope, the residuals being `y - slope x`. The intercept is the median of `y - slope x`.
///
/// # Arguments
/// * `x` - Predictor values
/// * `y` - Response values (must have same length as x)
/// * `iter` - Number of slope steps (default 1, at least 1)
///
/// Pairs with a NaN or infinite x or y are deleted before the fit; at least [`MIN_PAIRS`]
/// pairs must remain.
///
/// # Errors
///
/// A message for unequal lengths, `iter = 0`, fewer than [`MIN_PAIRS`] complete pairs, or
/// outer thirds whose x medians are equal (R divides by zero there and returns a non-finite
/// slope).
///
/// # Example
/// ```
/// use p2a_core::regression::line::line;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
/// let y = vec![2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.2, 17.9];
/// let result = line(&x, &y, Some(1)).unwrap();
/// println!("Intercept: {}, Slope: {}", result.intercept, result.slope);
/// ```
pub fn line(x: &[f64], y: &[f64], iter: Option<usize>) -> Result<LineResult, String> {
    if x.len() != y.len() {
        return Err("x and y must have the same length".to_string());
    }
    let iter_count = iter.unwrap_or(1);
    if iter_count == 0 {
        return Err("iter must be at least 1".to_string());
    }

    // Delete the pairs with a missing or infinite value
    let n_total = x.len();
    let bad_x = x.iter().filter(|v| !v.is_finite()).count();
    let bad_y = y.iter().filter(|v| !v.is_finite()).count();
    let (x, y): (Vec<f64>, Vec<f64>) = x
        .iter()
        .zip(y)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (*a, *b))
        .unzip();
    let (x, y) = (x.as_slice(), y.as_slice());
    let n = x.len();
    let missing = MissingSummary {
        n_total,
        n_used: n,
        n_dropped: n_total - n,
        by_column: [("x", bad_x), ("y", bad_y)]
            .into_iter()
            .filter(|(_, k)| *k > 0)
            .map(|(name, k)| (name.to_string(), k))
            .collect(),
    };
    if n < MIN_PAIRS {
        return Err(format!(
            "Need at least {MIN_PAIRS} complete (x, y) pairs for line fitting; {n} of {n_total} \
             pairs are complete (pairs with a missing or infinite value are dropped)"
        ));
    }

    let mut z = x.to_vec();
    z.sort_by(f64::total_cmp);
    let xb = r_quantile(&z, 1.0 / 3.0);
    let xt = r_quantile(&z, 2.0 / 3.0);
    let left: Vec<usize> = (0..n).filter(|&i| x[i] <= xb).collect();
    let right: Vec<usize> = (0..n).filter(|&i| x[i] >= xt).collect();
    let mut work = Vec::with_capacity(n);
    let mut median_of = |idx: &[usize], v: &[f64]| {
        work.clear();
        work.extend(idx.iter().map(|&i| v[i]));
        work.sort_by(f64::total_cmp);
        r_quantile(&work, 0.5)
    };
    let x_left = median_of(&left, x);
    let x_right = median_of(&right, x);
    if x_left == x_right {
        return Err(format!(
            "the medians of x in the outer thirds are equal ({x_left}); the resistant line has no \
             slope (too many tied x values)"
        ));
    }

    let mut slope = 0.0;
    let mut w = y.to_vec();
    for _ in 0..iter_count {
        let y_left = median_of(&left, &w);
        let y_right = median_of(&right, &w);
        slope += (y_right - y_left) / (x_right - x_left);
        for i in 0..n {
            w[i] = y[i] - slope * x[i];
        }
    }
    w.sort_by(f64::total_cmp);
    let intercept = r_quantile(&w, 0.5);

    // Final fitted values and residuals in input order
    let fitted: Vec<f64> = x.iter().map(|&xi| intercept + slope * xi).collect();
    let residuals: Vec<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(&yi, &fi)| yi - fi)
        .collect();

    Ok(LineResult {
        intercept,
        slope,
        fitted,
        residuals,
        n,
        iter: iter_count,
        x: x.to_vec(),
        y: y.to_vec(),
        missing,
    })
}

/// line.c's quantile of sorted `z`: `(z[floor((n - 1) p)] + z[ceil((n - 1) p)]) / 2`.
fn r_quantile(z: &[f64], p: f64) -> f64 {
    let h = (z.len() - 1) as f64 * p;
    0.5 * (z[h.floor() as usize] + z[h.ceil() as usize])
}

/// Compute the median of a slice.
#[cfg(test)]
fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(f64::total_cmp);
    r_quantile(&sorted, 0.5)
}

/// Run Tukey's line fitting (convenience wrapper).
pub fn run_line(x: &[f64], y: &[f64], iter: Option<usize>) -> Result<LineResult, String> {
    line(x, y, iter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_line_drops_incomplete_pairs_as_r() {
        // X-6: R's line() keeps complete.cases(x, y). NaN used to reach a partial_cmp sort (no
        // total order: a panic for n > 20 under Rust >= 1.81, or NaN quantiles and slope).
        let n = 40;
        let x: Vec<f64> = (0..n)
            .map(|i| i as f64 * 0.5 + 0.13 * ((i * 7 % 11) as f64))
            .collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| 1.5 + 0.8 * v + 0.3 * (((i * 5) % 7) as f64 - 3.0))
            .collect();
        let (mut xn, mut yn) = (x.clone(), y.clone());
        xn[5] = f64::NAN;
        yn[17] = f64::NAN;
        xn[30] = f64::NAN;
        let keep: Vec<usize> = (0..n).filter(|i| ![5, 17, 30].contains(i)).collect();
        let (xc, yc): (Vec<f64>, Vec<f64>) = keep.iter().map(|&i| (x[i], y[i])).unzip();
        let (a, b) = (
            line(&xn, &yn, Some(1)).unwrap(),
            line(&xc, &yc, Some(1)).unwrap(),
        );
        assert_eq!(
            (a.intercept, a.slope, &a.fitted),
            (b.intercept, b.slope, &b.fitted)
        );
        assert_eq!(
            (a.missing.n_dropped, a.missing.n_used, b.missing.n_dropped),
            (3, 37, 0)
        );
    }

    #[test]
    fn test_line_basic() {
        // Simple linear data
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 + 1.5 * xi).collect();

        let result = line(&x, &y, Some(1)).unwrap();

        // Should recover approximately slope=1.5, intercept=2.0
        assert_relative_eq!(result.slope, 1.5, epsilon = 0.1);
        assert_relative_eq!(result.intercept, 2.0, epsilon = 0.5);
    }

    #[test]
    fn test_line_with_outliers() {
        // Data with an outlier
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut y: Vec<f64> = x.iter().map(|&xi| 2.0 + 1.5 * xi).collect();
        y[4] = 100.0; // Add outlier at middle

        let result = line(&x, &y, Some(1)).unwrap();

        // Should still give reasonable estimates despite outlier
        // (median-based methods are resistant to outliers)
        assert!(result.slope > 0.0 && result.slope < 10.0);
    }

    #[test]
    fn test_line_residuals() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

        let result = line(&x, &y, Some(1)).unwrap();

        // Check that residuals = y - fitted
        for i in 0..x.len() {
            let expected_residual = y[i] - result.fitted[i];
            assert_relative_eq!(result.residuals[i], expected_residual, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_line_coef() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0];

        let result = line(&x, &y, Some(1)).unwrap();
        let (intercept, slope) = result.coef();

        assert_eq!(intercept, result.intercept);
        assert_eq!(slope, result.slope);
    }

    #[test]
    fn test_line_predict() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0];

        let result = line(&x, &y, Some(1)).unwrap();
        let predictions = result.predict(&[0.0, 10.0]);

        // Predictions should follow y = intercept + slope * x
        assert_relative_eq!(
            predictions[0],
            result.intercept + result.slope * 0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            predictions[1],
            result.intercept + result.slope * 10.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_line_length_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];

        let result = line(&x, &y, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_line_too_few_points() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0];

        let result = line(&x, &y, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_median() {
        // Odd number of elements
        assert_relative_eq!(median(&[1.0, 2.0, 3.0]), 2.0, epsilon = 1e-10);
        // Even number of elements
        assert_relative_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5, epsilon = 1e-10);
        // Unsorted input
        assert_relative_eq!(median(&[3.0, 1.0, 2.0]), 2.0, epsilon = 1e-10);
    }

    // ════════════════════════════════════════════════════════════════════════════
    // VALIDATION TESTS - Comparing against R reference implementations
    // ════════════════════════════════════════════════════════════════════════════

    /// Case `line_tukey_outliers` (three outliers in 30 points, y = 2 + 0.5 x + noise): R's
    /// `line` coefficients for iter = 1, 2, 3 (the case stores them printed to 3 decimals), and
    /// the resistant slope closer to 0.5 than the OLS slope R's `lm` gives on the same data
    /// (`ols_coef`, pinned here through `run_ols`).
    #[test]
    fn test_validate_line_resists_outliers() {
        use crate::regression::ols::{CovarianceType, run_ols};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("line_tukey_outliers");
        let n = case.scalar("n") as usize;
        let ols = run_ols(&ds, "y", &["x"], true, CovarianceType::Standard).unwrap();
        let df = ds.df();
        let num = |c: &str| {
            crate::data::columns::numeric_column(df, c, crate::data::columns::NaPolicy::Fail)
                .unwrap()
                .0
        };
        let (x, y) = (num("x"), num("y"));
        let coef = |it| {
            let r = line(&x, &y, Some(it)).unwrap();
            [r.intercept, r.slope]
        };
        Check::new(&case)
            .vector("iter1_coef", &coef(1), Tol::printed(3))
            .vector("iter2_coef", &coef(2), Tol::printed(3))
            .vector("iter3_coef", &coef(3), Tol::printed(3))
            .vector(
                "ols_coef",
                &ols.coefficients
                    .iter()
                    .map(|c| c.estimate)
                    .collect::<Vec<_>>(),
                Tol::coef(n),
            )
            .scalar("n", ols.n_obs as f64, Tol::exact())
            .finish();
        let result = line(&x, &y, Some(1)).unwrap();
        let ols_slope = case.vector("ols_coef")[1];
        assert!((result.slope - 0.5).abs() < 0.03, "slope {}", result.slope);
        assert!(
            (result.slope - 0.5).abs() < (ols_slope - 0.5).abs(),
            "resistant slope {} vs OLS {ols_slope}",
            result.slope
        );
    }

    /// Resistant line on clean data (no outliers) recovers y = 2 + 0.5 x.
    #[test]
    fn test_line_clean_data_recovers_dgp() {
        // Simple linear data without outliers
        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 2.0 + 0.5 * xi + 0.01 * (xi % 3.0 - 1.0))
            .collect();

        let result = line(&x, &y, Some(1)).unwrap();

        // Should recover approximately y = 2 + 0.5x
        assert!(
            (result.slope - 0.5).abs() < 0.1,
            "Clean data slope should be close to 0.5, got {}",
            result.slope
        );
        assert!(
            (result.intercept - 2.0).abs() < 0.5,
            "Clean data intercept should be close to 2.0, got {}",
            result.intercept
        );
    }

    /// The reg_smoothers_na case as polars reads it: `id` Int64, `x` and `y` Float64
    /// with 2 empty cells each, R's count of complete pairs, and no ties among the
    /// complete x (so supsmu's unique grid is the complete x).
    #[test]
    fn test_validate_reg_smoothers_na_fixture() {
        use p2a_validation::{Check, Tol};
        use polars::prelude::DataType;
        let (case, ds) = crate::test_support::load_case("reg_smoothers_na");
        let df = ds.df();
        assert_eq!(df.height(), 60);
        assert_eq!(df.column("id").unwrap().dtype(), &DataType::Int64);
        for c in ["x", "y"] {
            let col = df.column(c).unwrap();
            assert_eq!(col.dtype(), &DataType::Float64, "{c}");
            assert_eq!(col.null_count(), 2, "{c}");
        }
        let ok = &df.column("x").unwrap().is_not_null() & &df.column("y").unwrap().is_not_null();
        let n_complete = ok.sum().unwrap() as usize;
        Check::new(&case)
            .scalar("n_complete", n_complete as f64, Tol::exact())
            .finish();
        let x = crate::data::columns::numeric_options(df, "x").unwrap();
        let y = crate::data::columns::numeric_options(df, "y").unwrap();
        let mut xc: Vec<f64> = x
            .iter()
            .zip(&y)
            .filter_map(|(a, b)| a.zip(*b).map(|(a, _)| a))
            .collect();
        assert_eq!(xc.len(), n_complete);
        xc.sort_by(f64::total_cmp);
        assert!(
            xc.windows(2).all(|w| w[0] != w[1]),
            "tied complete x values"
        );
    }

    /// REG-13: pairs with a missing (NaN) or infinite x or y are deleted before the fit,
    /// as R's line uses the complete pairs; the fit on the NA data equals the fit on
    /// the complete pairs, and the sort cannot see a NaN.
    #[test]
    fn test_validate_line_na() {
        use crate::data::columns::MissingSummary;
        use crate::regression::ols::na_fixtures::nan_vec;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("reg_smoothers_na");
        let (x, y) = (nan_vec(&ds, "x"), nan_vec(&ds, "y"));
        let (xc, yc): (Vec<f64>, Vec<f64>) = x
            .iter()
            .zip(&y)
            .filter(|(a, b)| a.is_finite() && b.is_finite())
            .map(|(a, b)| (*a, *b))
            .unzip();
        let r = line(&x, &y, Some(1)).unwrap();
        let c = line(&xc, &yc, Some(1)).unwrap();
        assert_eq!(
            r.missing,
            MissingSummary {
                n_total: 60,
                n_used: 56,
                n_dropped: 4,
                by_column: vec![("x".into(), 2), ("y".into(), 2)],
            }
        );
        assert!(r.intercept.is_finite() && r.slope.is_finite());
        assert_eq!(
            (r.intercept, r.slope, &r.x, &r.y, &r.fitted, &r.residuals),
            (c.intercept, c.slope, &xc, &yc, &c.fitted, &c.residuals)
        );
        assert!(c.missing.is_empty());
        assert!(serde_json::to_value(&c).unwrap().get("missing").is_none());
        Check::new(&case)
            .scalar("n_complete", r.n as f64, Tol::exact())
            .vector("line_coef", &[r.intercept, r.slope], Tol::coef(56))
            .finish();
        // +Inf in a complete row is deleted too (the array API; datasets refuse Inf
        // earlier). Row 0 of the case is complete, so xc[1..] drops exactly that row.
        let mut xi = x.clone();
        xi[0] = f64::INFINITY;
        let ri = line(&xi, &y, Some(1)).unwrap();
        assert_eq!(
            (ri.n, ri.missing.by_column.clone()),
            (55, vec![("x".into(), 3), ("y".into(), 2)])
        );
        let d = line(&xc[1..], &yc[1..], Some(1)).unwrap();
        assert_eq!((ri.intercept, ri.slope), (d.intercept, d.slope));
        let e = line(&[1.0, f64::NAN, 3.0], &[1.0, 2.0, 3.0], None).unwrap_err();
        assert!(
            e.contains("at least 3 complete") && e.contains("2 of 3"),
            "{e}"
        );
    }

    fn t4_xy(file: &str) -> (Vec<f64>, Vec<f64>) {
        use crate::data::columns::{NaPolicy, numeric_column};
        let ds = crate::regression::ols::na_fixtures::repo_dataset(&format!(
            "validation/datasets/generated/{file}.csv"
        ));
        let c = |n: &str| numeric_column(ds.df(), n, NaPolicy::Fail).unwrap().0;
        (c("x"), c("y"))
    }

    /// Case `line_tukey_t4` (REG-12): stats::line (R >= 3.5 tukeyline) with iter = 1, 2, 3 on
    /// the eleven-point example, tied x (20 rows on 0..4) and n = 31.
    #[test]
    fn test_validate_line_tukey_t4() {
        use p2a_validation::{Check, Tol};
        let case = p2a_validation::Case::load("line_tukey_t4");
        let mut c = Check::new(&case);
        for (tag, file) in [
            ("eleven", "line_t4_eleven"),
            ("ties", "line_t4_ties"),
            ("n31", "line_t4_n31"),
        ] {
            let (x, y) = t4_xy(file);
            for it in 1..=3 {
                let r = line(&x, &y, Some(it)).unwrap();
                c.vector(
                    &format!("{tag}_iter{it}_coef"),
                    &[r.intercept, r.slope],
                    Tol::coef(x.len()),
                );
                if tag == "n31" && it == 1 {
                    c.vector("n31_iter1_fitted", &r.fitted, Tol::coef(31))
                        .vector("n31_iter1_residuals", &r.residuals, Tol::coef(31));
                }
            }
        }
        c.finish();
    }

    /// Outer thirds whose x medians are equal give the line no slope: an error naming them (R
    /// divides by zero and returns a non-finite slope).
    #[test]
    fn test_line_refuses_equal_third_medians() {
        let x = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0];
        let y = [0.3, 1.1, 0.7, 1.9, 0.2, 1.4, 0.8];
        let e = line(&x, &y, Some(1)).unwrap_err();
        assert!(e.contains("outer thirds"), "{e}");
        assert!(line(&[1.0, 2.0, 3.0], &[1.0, 2.5, 2.9], Some(0)).is_err());
    }
}
