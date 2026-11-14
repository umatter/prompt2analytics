//! Isotonic Regression (isoreg)
//!
//! Computes isotonic (monotonically increasing) least squares regression
//! which is piecewise constant.
//!
//! # References
//!
//! - Barlow, R. E., Bartholomew, D. J., Bremner, J. M., and Brunk, H. D. (1972).
//!   "Statistical Inference Under Order Restrictions: The Theory and Application
//!   of Isotonic Regression". John Wiley & Sons.
//! - R stats::isoreg documentation
//!   Source: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/isoreg.html

use serde::{Deserialize, Serialize};

/// Result of isotonic regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsoregResult {
    /// Original x values (sorted)
    pub x: Vec<f64>,
    /// Original y values (reordered to match sorted x)
    pub y: Vec<f64>,
    /// Fitted values (isotonic regression estimate)
    pub yf: Vec<f64>,
    /// Cumulative sums of the ordered y with a leading zero, `n + 1` values (R's
    /// `yc = cumsum(c(0, y))`)
    pub yc: Vec<f64>,
    /// 0-based index of the last element of each pooled block (R's `iKnots - 1`); the fitted
    /// value changes after each knot but the last
    pub i_knots: Vec<usize>,
    /// Whether the original x was already sorted (non-decreasing)
    pub is_ordered: bool,
    /// Permutation ordering the data by increasing x and, within tied x, decreasing y (R's
    /// `order(x, -y)`, 0-based), if x was not originally sorted
    pub ord: Option<Vec<usize>>,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for IsoregResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Isotonic Regression (isoreg) Results")?;
        writeln!(f, "====================================")?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Number of knots: {}", self.i_knots.len())?;
        writeln!(f, "Data was pre-sorted: {}", self.is_ordered)?;
        writeln!(f)?;

        // Show knot information
        writeln!(f, "Knots (last index of each pooled block):")?;
        writeln!(f, "  Index    X        Y       Fitted")?;
        for &k in &self.i_knots {
            writeln!(
                f,
                "  {:5}    {:8.4}  {:8.4}  {:8.4}",
                k, self.x[k], self.y[k], self.yf[k]
            )?;
        }
        writeln!(f)?;

        // Show first few fitted values
        let n_show = self.n.min(10);
        writeln!(f, "First {} observations:", n_show)?;
        writeln!(f, "  Index    X        Y       Fitted")?;
        for i in 0..n_show {
            writeln!(
                f,
                "  {:5}    {:8.4}  {:8.4}  {:8.4}",
                i, self.x[i], self.y[i], self.yf[i]
            )?;
        }
        if self.n > 10 {
            writeln!(f, "  ... ({} more observations)", self.n - 10)?;
        }

        Ok(())
    }
}

/// Compute isotonic regression using the Pool Adjacent Violators Algorithm (PAVA).
///
/// # Arguments
/// * `x` - Predictor values
/// * `y` - Response values (must have same length as x, all finite)
///
/// # Returns
/// * `IsoregResult` containing fitted values and knot information
///
/// # Algorithm
///
/// The Pool Adjacent Violators Algorithm (PAVA) works as follows:
/// 1. Sort data by x values
/// 2. Start with each y value as its own block
/// 3. Scan left to right, whenever adjacent blocks violate monotonicity
///    (left block > right block), pool them by taking weighted average
/// 4. Repeat until no violations exist
///
/// The result is a piecewise constant function that is monotonically increasing.
///
/// # References
///
/// - Barlow et al. (1972). "Statistical Inference Under Order Restrictions"
pub fn isoreg(x: &[f64], y: &[f64]) -> Result<IsoregResult, String> {
    let n = x.len();

    if n == 0 {
        return Err("x and y must have at least one element".to_string());
    }
    if y.len() != n {
        return Err(format!(
            "x and y must have the same length. x has {} elements, y has {}",
            n,
            y.len()
        ));
    }

    // Check for finite y values
    for (i, &yi) in y.iter().enumerate() {
        if !yi.is_finite() {
            return Err(format!("y[{}] = {} is not finite", i, yi));
        }
    }

    // R: "missing values not allowed"
    if let Some(i) = x.iter().position(|v| v.is_nan()) {
        return Err(format!("missing values not allowed: x[{i}] is NaN"));
    }

    // R: data already in non-decreasing x are used as given; otherwise ordered by
    // order(x, -y), increasing in x and decreasing in y within tied x, so that a tie
    // pools into one block.
    let is_ordered = x.windows(2).all(|w| w[0] <= w[1]);
    let indices: Vec<usize> = if is_ordered {
        (0..n).collect()
    } else {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(y[b].total_cmp(&y[a])));
        idx
    };

    // Create sorted x and y
    let x_sorted: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
    let y_sorted: Vec<f64> = indices.iter().map(|&i| y[i]).collect();

    // yc = cumsum(c(0, y)) of the ordered y
    let yc = cumsum_from_zero(&y_sorted);

    // Pool Adjacent Violators Algorithm (PAVA)
    let (yf, i_knots) = pava(&y_sorted, &yc);

    Ok(IsoregResult {
        x: x_sorted,
        y: y_sorted,
        yf,
        yc,
        i_knots,
        is_ordered,
        ord: if is_ordered { None } else { Some(indices) },
        n,
    })
}

/// Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.
///
/// Uses an O(n) single-pass stack-based approach: process each element
/// left to right, pushing onto a stack and merging with the previous
/// block while the previous block's mean is strictly greater (adjacent blocks
/// with equal means stay separate, as R's greatest-convex-minorant pass leaves
/// them). Each block's value is `(yc[end] - yc[start]) / (end - start)`, R's
/// arithmetic. Returns the fitted values and the 0-based last index of each
/// block.
fn pava(y: &[f64], yc: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let n = y.len();
    // Stack of blocks: (sum, count); a block's end is the start of the next.
    let mut stack_sum: Vec<f64> = Vec::with_capacity(n);
    let mut stack_cnt: Vec<usize> = Vec::with_capacity(n);

    for &yi in y {
        stack_sum.push(yi);
        stack_cnt.push(1);

        // Merge back while top-of-stack violates monotonicity with previous
        while stack_sum.len() >= 2 {
            let top = stack_sum.len() - 1;
            let prev = top - 1;
            let mean_prev = stack_sum[prev] / stack_cnt[prev] as f64;
            let mean_top = stack_sum[top] / stack_cnt[top] as f64;

            if mean_prev > mean_top {
                stack_sum[prev] += stack_sum[top];
                stack_cnt[prev] += stack_cnt[top];
                stack_sum.pop();
                stack_cnt.pop();
            } else {
                break;
            }
        }
    }

    // Reconstruct fitted values and block ends from the stack
    let mut yf = Vec::with_capacity(n);
    let mut knots = Vec::with_capacity(stack_cnt.len());
    let mut start = 0;
    for &cnt in &stack_cnt {
        let end = start + cnt;
        let mean = (yc[end] - yc[start]) / cnt as f64;
        yf.extend(std::iter::repeat_n(mean, cnt));
        knots.push(end - 1);
        start = end;
    }

    (yf, knots)
}

/// Cumulative sums with a leading zero: `n + 1` values (R's `cumsum(c(0, x))`).
fn cumsum_from_zero(x: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(x.len() + 1);
    let mut sum = 0.0;
    result.push(sum);
    for &xi in x {
        sum += xi;
        result.push(sum);
    }
    result
}

/// Isotonic regression from a single sorted vector.
///
/// This is a convenience function when x is just the index (1, 2, 3, ...).
pub fn isoreg_y(y: &[f64]) -> Result<IsoregResult, String> {
    let x: Vec<f64> = (0..y.len()).map(|i| i as f64 + 1.0).collect();
    isoreg(&x, y)
}

/// Predict fitted values at new x locations using linear interpolation.
///
/// # Arguments
/// * `result` - IsoregResult from a previous isoreg call
/// * `new_x` - New x values at which to predict
///
/// # Returns
/// * Vector of predicted y values
pub fn isoreg_predict(result: &IsoregResult, new_x: &[f64]) -> Vec<f64> {
    new_x
        .iter()
        .map(|&xi| {
            // Find where xi falls in the sorted x values
            if xi <= result.x[0] {
                result.yf[0]
            } else if xi >= result.x[result.n - 1] {
                result.yf[result.n - 1]
            } else {
                // Binary search for the interval
                let mut lo = 0;
                let mut hi = result.n - 1;
                while hi - lo > 1 {
                    let mid = (lo + hi) / 2;
                    if result.x[mid] <= xi {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                // For step function (isotonic regression), use the value at lo
                // since the function is constant within each interval
                result.yf[lo]
            }
        })
        .collect()
}

/// Convenience function to run isotonic regression.
pub fn run_isoreg(x: &[f64], y: &[f64]) -> Result<IsoregResult, String> {
    isoreg(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isoreg_basic() {
        // Example from R documentation
        let y = vec![1.0, 0.0, 4.0, 3.0, 3.0, 5.0, 4.0, 2.0, 0.0];
        let result = isoreg_y(&y).unwrap();

        assert_eq!(result.n, 9);
        assert!(result.is_ordered); // Index-based x is sorted

        // Fitted values should be monotonically non-decreasing
        for i in 1..result.n {
            assert!(
                result.yf[i] >= result.yf[i - 1] - 1e-10,
                "yf[{}]={} < yf[{}]={}",
                i,
                result.yf[i],
                i - 1,
                result.yf[i - 1]
            );
        }
    }

    #[test]
    fn test_isoreg_already_monotone() {
        // Already monotone data
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = isoreg(&x, &y).unwrap();

        // Fitted should equal original for monotone data
        for i in 0..5 {
            assert!(
                (result.yf[i] - result.y[i]).abs() < 1e-10,
                "yf[{}]={} != y[{}]={}",
                i,
                result.yf[i],
                i,
                result.y[i]
            );
        }
    }

    #[test]
    fn test_isoreg_constant() {
        // Constant data
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![5.0, 5.0, 5.0];
        let result = isoreg(&x, &y).unwrap();

        for &yi in &result.yf {
            assert!((yi - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_isoreg_decreasing() {
        // Strictly decreasing data should be pooled into one block
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let result = isoreg(&x, &y).unwrap();

        // All fitted values should be equal (pooled mean)
        let expected_mean = (5.0 + 4.0 + 3.0 + 2.0 + 1.0) / 5.0; // 3.0
        for &yi in &result.yf {
            assert!((yi - expected_mean).abs() < 1e-10);
        }
        assert_eq!(result.i_knots.len(), 1); // Only one knot
    }

    #[test]
    fn test_isoreg_unsorted_x() {
        // x values not sorted
        let x = vec![3.0, 1.0, 2.0];
        let y = vec![3.0, 1.0, 2.0];
        let result = isoreg(&x, &y).unwrap();

        assert!(!result.is_ordered);
        assert!(result.ord.is_some());

        // After sorting by x, we get x=[1,2,3], y=[1,2,3]
        assert_eq!(result.x, vec![1.0, 2.0, 3.0]);
        assert_eq!(result.y, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_isoreg_single_element() {
        let x = vec![1.0];
        let y = vec![5.0];
        let result = isoreg(&x, &y).unwrap();

        assert_eq!(result.n, 1);
        assert_eq!(result.yf[0], 5.0);
    }

    #[test]
    fn test_isoreg_r_example() {
        // Example from R: ir4 <- isoreg(1:10, c(5, 9, 1:2, 5:8, 3, 8))
        let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let y = vec![5.0, 9.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 3.0, 8.0];
        let result = isoreg(&x, &y).unwrap();

        // Fitted values should be monotonically non-decreasing
        for i in 1..result.n {
            assert!(
                result.yf[i] >= result.yf[i - 1] - 1e-10,
                "Monotonicity violated at i={}: yf[{}]={} < yf[{}]={}",
                i,
                i,
                result.yf[i],
                i - 1,
                result.yf[i - 1]
            );
        }
    }

    #[test]
    fn test_isoreg_predict() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 5.0, 2.0, 4.0, 6.0]; // Has violations
        let result = isoreg(&x, &y).unwrap();

        // Predict at original x values
        let pred = isoreg_predict(&result, &x);
        for i in 0..5 {
            assert!((pred[i] - result.yf[i]).abs() < 1e-10);
        }

        // Predict at interpolated values
        let new_x = vec![0.0, 1.5, 6.0];
        let pred = isoreg_predict(&result, &new_x);

        // Below range should get first value
        assert!((pred[0] - result.yf[0]).abs() < 1e-10);
        // Above range should get last value
        assert!((pred[2] - result.yf[4]).abs() < 1e-10);
    }

    #[test]
    fn test_isoreg_validation() {
        // Empty arrays should fail
        let result = isoreg(&[], &[]);
        assert!(result.is_err());

        // Mismatched lengths should fail
        let result = isoreg(&[1.0, 2.0], &[1.0]);
        assert!(result.is_err());

        // Non-finite y should fail
        let result = isoreg(&[1.0, 2.0], &[1.0, f64::INFINITY]);
        assert!(result.is_err());
    }

    #[test]
    fn test_isoreg_display() {
        let y = vec![1.0, 0.0, 4.0, 3.0, 3.0];
        let result = isoreg_y(&y).unwrap();
        let display = format!("{}", result);

        assert!(display.contains("Isotonic Regression"));
        assert!(display.contains("Number of observations: 5"));
        assert!(display.contains("Knots"));
    }

    #[test]
    fn test_cumsum() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let cs = cumsum_from_zero(&x);
        assert_eq!(cs, vec![0.0, 1.0, 3.0, 6.0, 10.0]);
    }

    // =========================================================================
    // Validation tests against R
    // =========================================================================

    #[test]
    fn test_isoreg_knots_are_block_ends() {
        // i_knots are R's iKnots - 1: the 0-based last index of each block. Adjacent blocks with
        // equal means stay separate, so monotone data has one block per point
        // (R: isoreg(c(1, 1, 3, 3, 5, 5))$iKnots is 1:6).
        let y = vec![1.0, 1.0, 3.0, 3.0, 5.0, 5.0];
        let result = isoreg_y(&y).unwrap();
        assert_eq!(result.yf, y);
        assert_eq!(result.i_knots, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(result.yc.len(), 7);
    }

    /// R's 1-based iKnots as p2a's 0-based block ends.
    fn knots_r(k: &[usize]) -> Vec<f64> {
        k.iter().map(|&i| (i + 1) as f64).collect()
    }

    #[test]
    fn test_validate_isoreg_examples() {
        use crate::stats::ranks::test_data::column_where;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_rank_isoreg");
        let col = |set: &str, c: &str| column_where(&ds, c, &[("set", set)]);
        let doc = isoreg_y(&col("doc", "y")).unwrap();
        let doc3 = isoreg_y(&col("doc3", "y")).unwrap();
        let ir4 = isoreg(&col("ir4", "x"), &col("ir4", "y")).unwrap();
        let mono = isoreg_y(&col("mono", "y")).unwrap();
        let decr = isoreg_y(&col("decr", "y")).unwrap();
        let un = isoreg(&col("unsorted", "x"), &col("unsorted", "y")).unwrap();
        let tied = isoreg(&col("tiedx", "x"), &col("tiedx", "y")).unwrap();
        assert!(!un.is_ordered);
        // R's ord is 1-based
        let ord: Vec<f64> = un
            .ord
            .as_ref()
            .unwrap()
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect();
        Check::new(&case)
            .vector("yf_doc", &doc.yf, Tol::coef(9))
            .vector("yf_doc3", &doc3.yf, Tol::coef(9))
            .vector("yf_ir4", &ir4.yf, Tol::coef(10))
            .vector("yf_mono", &mono.yf, Tol::coef(5))
            .vector("yf_decr", &decr.yf, Tol::coef(5))
            .scalar("n_knots_decr", decr.i_knots.len() as f64, Tol::exact())
            .vector("yf_unsorted", &un.yf, Tol::coef(15))
            .vector("x_sorted_unsorted", &un.x, Tol::exact())
            .vector("ord_unsorted", &ord, Tol::exact())
            .vector("iknots_doc", &knots_r(&doc.i_knots), Tol::exact())
            .vector("iknots_ir4", &knots_r(&ir4.i_knots), Tol::exact())
            .vector("yc_doc", &doc.yc, Tol::coef(10))
            .vector("yf_tiedx", &tied.yf, Tol::coef(5))
            .vector("iknots_tiedx", &knots_r(&tied.i_knots), Tol::exact())
            .finish();
    }

    /// Tied x pool through order(x, -y), whatever the row order within a tie; yc has a leading 0.
    #[test]
    fn test_validate_isoreg_tied_x() {
        use crate::stats::ranks::test_data::column_where;
        use p2a_validation::{Check, Tol};
        let (case, _) = crate::test_support::load_case("stat_isoreg_t4");
        let (_, ds) = crate::test_support::load_case("stat_rank_isoreg");
        let col = |set: &str, c: &str| column_where(&ds, c, &[("set", set)]);
        let (tx, ty) = (col("tiedx", "x"), col("tiedx", "y"));
        let tied = isoreg(&tx, &ty).unwrap();
        let ir4 = isoreg(&col("ir4", "x"), &col("ir4", "y")).unwrap();
        let doc = isoreg_y(&col("doc", "y")).unwrap();
        // R's 1-based permutation of the rows
        let perm: Vec<usize> = case
            .vector("perm")
            .iter()
            .map(|&p| p as usize - 1)
            .collect();
        let px: Vec<f64> = perm.iter().map(|&i| tx[i]).collect();
        let py: Vec<f64> = perm.iter().map(|&i| ty[i]).collect();
        let permuted = isoreg(&px, &py).unwrap();
        assert_eq!(permuted.yf, tied.yf);
        assert_eq!(permuted.i_knots, tied.i_knots);
        Check::new(&case)
            .vector("yc_tiedx", &tied.yc, Tol::coef(10))
            .vector("yc_ir4", &ir4.yc, Tol::coef(10))
            .vector("yc_doc", &doc.yc, Tol::coef(10))
            .vector("iknots_doc", &knots_r(&doc.i_knots), Tol::exact())
            .vector("yf_tiedx_perm", &permuted.yf, Tol::coef(10))
            .vector(
                "iknots_tiedx_perm",
                &knots_r(&permuted.i_knots),
                Tol::exact(),
            )
            .finish();
    }

    #[test]
    fn test_isoreg_rejects_nan_x() {
        // R: isoreg(c(1, NaN, 3), 1:3) -> "missing values not allowed". Before: Ok, with NaN
        // sorted by partial_cmp.
        let e = isoreg(&[1.0, f64::NAN, 3.0], &[1.0, 2.0, 3.0]).unwrap_err();
        assert!(
            e.contains("missing values not allowed") && e.contains("x[1]"),
            "{e}"
        );
    }
}
