//! McNemar's chi-squared test for count data
//!
//! Tests for symmetry in a 2x2 contingency table with paired/matched data.
//! Commonly used for comparing two classifiers on the same dataset or
//! for before/after studies.
//!
//! # References
//!
//! - McNemar, Q. (1947). "Note on the sampling error of the difference between
//!   correlated proportions or percentages". Psychometrika, 12(2), 153-157.
//! - Edwards, A. L. (1948). "Note on the 'correction for continuity' in testing
//!   the significance of the difference between correlated proportions".
//!   Psychometrika, 13(3), 185-187.
//! - R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mcnemar.test.html

use crate::errors::{EconError, EconResult};
use serde::{Deserialize, Serialize};

/// Result of McNemar's chi-squared test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McnemarResult {
    /// The chi-squared statistic
    pub statistic: f64,
    /// Degrees of freedom (always 1 for 2x2)
    pub df: usize,
    /// P-value from chi-squared distribution
    pub p_value: f64,
    /// Number of discordant pairs (b + c)
    pub n_discordant: u64,
    /// The b cell (off-diagonal)
    pub b: u64,
    /// The c cell (off-diagonal)
    pub c: u64,
    /// Whether continuity correction was applied
    pub continuity_correction: bool,
}

/// Perform McNemar's chi-squared test for symmetry in a 2x2 table.
///
/// # Arguments
///
/// * `b` - Upper-right cell of the 2x2 table (row 1, column 2)
/// * `c` - Lower-left cell of the 2x2 table (row 2, column 1)
/// * `correct` - Apply Yates' continuity correction (default: true)
///
/// # Returns
///
/// A `McnemarResult` containing the test statistic, degrees of freedom, and p-value.
///
/// # Example
///
/// ```
/// use p2a_core::stats::mcnemar::{mcnemar_test, McnemarResult};
///
/// // 2x2 contingency table:
/// //          Method 2+  Method 2-
/// // Method 1+    a=10     b=5
/// // Method 1-    c=15     d=20
/// //
/// // We only need b and c (discordant pairs)
///
/// let result = mcnemar_test(5, 15, true).unwrap();
/// println!("χ² = {}, p = {}", result.statistic, result.p_value);
/// ```
pub fn mcnemar_test(b: u64, c: u64, correct: bool) -> EconResult<McnemarResult> {
    // Need at least one discordant pair
    let n_discordant = b + c;
    if n_discordant == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "McNemar test requires at least one discordant pair (b + c > 0)".to_string(),
        });
    }

    // Compute test statistic
    let b_f = b as f64;
    let c_f = c as f64;
    let n_f = n_discordant as f64;

    let statistic = if correct {
        // With continuity correction (Edwards)
        // χ² = (|b - c| - 1)² / (b + c)
        let diff = (b_f - c_f).abs() - 1.0;
        if diff > 0.0 { diff * diff / n_f } else { 0.0 }
    } else {
        // Without continuity correction
        // χ² = (b - c)² / (b + c)
        let diff = b_f - c_f;
        diff * diff / n_f
    };

    // Degrees of freedom (always 1 for 2x2)
    let df = 1;

    // P-value from chi-squared distribution
    let p_value = chi_squared_p_value(statistic, df);

    Ok(McnemarResult {
        statistic,
        df,
        p_value,
        n_discordant,
        b,
        c,
        continuity_correction: correct,
    })
}

/// Perform McNemar's test from a 2x2 matrix.
///
/// # Arguments
///
/// * `table` - A 2x2 matrix [[a, b], [c, d]]
/// * `correct` - Apply continuity correction
///
/// # Returns
///
/// A `McnemarResult` containing the test results.
pub fn mcnemar_test_matrix(table: &[[u64; 2]; 2], correct: bool) -> EconResult<McnemarResult> {
    let b = table[0][1];
    let c = table[1][0];
    mcnemar_test(b, c, correct)
}

/// Compute p-value from chi-squared distribution
fn chi_squared_p_value(statistic: f64, df: usize) -> f64 {
    use statrs::distribution::{ChiSquared, ContinuousCDF};

    if df == 0 || statistic.is_nan() || statistic.is_infinite() {
        return f64::NAN;
    }

    let chi2 = ChiSquared::new(df as f64).unwrap();
    1.0 - chi2.cdf(statistic)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcnemar_basic() {
        // Basic test with clear asymmetry
        let result = mcnemar_test(5, 15, false).unwrap();

        assert_eq!(result.df, 1);
        assert_eq!(result.b, 5);
        assert_eq!(result.c, 15);
        assert_eq!(result.n_discordant, 20);
        assert!(!result.continuity_correction);

        // χ² = (5 - 15)² / 20 = 100 / 20 = 5.0
        assert!((result.statistic - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_mcnemar_with_correction() {
        // Same test with continuity correction
        let result = mcnemar_test(5, 15, true).unwrap();

        assert!(result.continuity_correction);

        // χ² = (|5 - 15| - 1)² / 20 = 81 / 20 = 4.05
        assert!((result.statistic - 4.05).abs() < 0.01);
    }

    #[test]
    fn test_mcnemar_symmetric() {
        // Symmetric case (b = c) should give low statistic
        let result = mcnemar_test(10, 10, false).unwrap();

        // χ² = (10 - 10)² / 20 = 0
        assert!((result.statistic - 0.0).abs() < 0.01);
        // p-value should be 1.0
        assert!((result.p_value - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_validate_mcnemar_against_r() {
        // R code:
        // Performance <- matrix(c(794, 86, 150, 570), nrow = 2,
        //                       dimnames = list("1st Survey" = c("Approve", "Disapprove"),
        //                                       "2nd Survey" = c("Approve", "Disapprove")))
        // mcnemar.test(Performance)
        //
        // R fills by column, so the matrix is:
        //                2nd Survey
        // 1st Survey    Approve  Disapprove
        //   Approve       794       150
        //   Disapprove     86       570
        //
        // b = 150 (row 1, col 2), c = 86 (row 2, col 1)
        //
        // Result: McNemar's chi-squared = 16.818, df = 1, p-value = 4.115e-05

        let result = mcnemar_test(150, 86, true).unwrap();

        println!(
            "With correction: χ² = {}, df = {}, p = {}",
            result.statistic, result.df, result.p_value
        );

        // R gives: χ² = 16.818, df = 1, p-value = 4.115e-05
        assert!(
            (result.statistic - 16.818).abs() < 0.01,
            "χ² mismatch: got {}",
            result.statistic
        );
        assert_eq!(result.df, 1);
        assert!(
            (result.p_value - 4.115e-05).abs() < 1e-06,
            "p-value mismatch: got {}",
            result.p_value
        );
    }

    #[test]
    fn test_validate_mcnemar_no_correction() {
        // R code:
        // mcnemar.test(Performance, correct = FALSE)
        //
        // Result: McNemar's chi-squared = 17.356, df = 1, p-value = 3.099e-05

        let result = mcnemar_test(150, 86, false).unwrap();

        println!(
            "No correction: χ² = {}, df = {}, p = {}",
            result.statistic, result.df, result.p_value
        );

        // χ² = (150 - 86)² / 236 = 4096 / 236 = 17.356
        assert!(
            (result.statistic - 17.356).abs() < 0.01,
            "χ² mismatch: got {}",
            result.statistic
        );
        assert!(
            (result.p_value - 3.099e-05).abs() < 1e-06,
            "p-value mismatch: got {}",
            result.p_value
        );
    }

    #[test]
    fn test_mcnemar_matrix() {
        // Matrix is [[a, b], [c, d]]
        // b = 150 (row 0, col 1), c = 86 (row 1, col 0)
        let table = [[794, 150], [86, 570]];
        let result = mcnemar_test_matrix(&table, true).unwrap();

        assert_eq!(result.b, 150);
        assert_eq!(result.c, 86);
    }

    #[test]
    fn test_mcnemar_validates_input() {
        // Should fail with zero discordant pairs
        let result = mcnemar_test(0, 0, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_mcnemar_small_correction() {
        // When |b - c| <= 1, corrected statistic should be 0
        let result = mcnemar_test(5, 6, true).unwrap();

        // |5 - 6| - 1 = 0, so χ² = 0
        assert!((result.statistic - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_mcnemar_large_asymmetry() {
        // Very asymmetric case
        let result = mcnemar_test(1, 100, false).unwrap();

        // χ² = (1 - 100)² / 101 = 9801 / 101 ≈ 97.04
        assert!((result.statistic - 97.04).abs() < 0.1);
        // Should definitely reject null
        assert!(result.p_value < 0.001);
    }
}
