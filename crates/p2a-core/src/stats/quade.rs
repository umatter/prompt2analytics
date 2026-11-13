//! Quade test for unreplicated blocked data.
//!
//! A non-parametric test for comparing treatments across blocks, similar to
//! the Friedman test but using a different weighting scheme that can be more
//! powerful when block effects vary considerably.
//!
//! # Mathematical Background
//!
//! The Quade test weights blocks by their range:
//!
//! 1. Compute the range within each block: Range_i = max(X_i) - min(X_i)
//! 2. Rank the ranges across blocks: Q_i
//! 3. Rank values within each block: R_ij
//! 4. Compute weighted scores: S_ij = Q_i × (R_ij - (k+1)/2)
//! 5. Compute treatment totals: S_j = Σ_i S_ij
//! 6. F statistic: T = (b-1)B / (A - B)
//!    where A = ΣΣ S_ij² and B = (1/b) Σ S_j²
//!
//! Ranks with `stats::ranks::midranks` (exact ties, as R).
//!
//! # References
//!
//! - Quade, D. (1979). "Using weighted rankings in the analysis of complete
//!   blocks with additive block effects". Journal of the American Statistical
//!   Association, 74(367), 680-683.
//! - Conover, W. J. (1999). Practical Nonparametric Statistics (3rd ed.).
//!   New York: Wiley. Pages 373-380.
//! - R Core Team. `stats::quade.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quade.test.html>

use crate::data::Dataset;
use crate::data::columns::MissingSummary;
use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::stats::friedman::unreplicated_blocks;
use crate::stats::ranks::{midranks, midranks_into};
use crate::traits::estimator::f_test_p_value;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;

/// Result of the Quade test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadeResult {
    /// The F statistic; NaN when A == B (identical rankings in every block, as R's
    /// `quade.test`), with p = (k!)^(1-b)
    pub statistic: f64,
    /// Numerator degrees of freedom (k - 1); kept when the statistic is NaN, where R reports
    /// NA
    pub df1: usize,
    /// Denominator degrees of freedom ((b - 1)(k - 1)); kept when the statistic is NaN, where
    /// R reports NA
    pub df2: usize,
    /// P-value from F distribution
    pub p_value: f64,
    /// Number of blocks (subjects/rows)
    pub n_blocks: usize,
    /// Number of treatments (groups/columns)
    pub n_treatments: usize,
    /// Treatment sums (S_j values)
    pub treatment_sums: Vec<f64>,
    /// Mean weighted rank per treatment
    pub mean_weighted_ranks: Vec<f64>,
    /// Treatment/group names
    pub treatment_names: Vec<String>,
    /// Block ranges
    pub block_ranges: Vec<f64>,
    /// A statistic (sum of squared S_ij)
    pub a_statistic: f64,
    /// B statistic (sum of squared S_j / b)
    pub b_statistic: f64,
    /// Rows of the blocks dropped because a response is missing (R drops such blocks);
    /// empty for the array-level function.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl std::fmt::Display for QuadeResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Quade test")?;
        writeln!(f, "==========")?;
        writeln!(f)?;
        writeln!(
            f,
            "F = {:.4}, df1 = {}, df2 = {}, p-value = {:.6}",
            self.statistic, self.df1, self.df2, self.p_value
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Blocks: {}, Treatments: {}",
            self.n_blocks, self.n_treatments
        )?;
        writeln!(f)?;
        writeln!(f, "Treatment weighted rank sums:")?;
        for (name, sum) in self.treatment_names.iter().zip(self.treatment_sums.iter()) {
            writeln!(f, "  {}: {:.4}", name, sum)?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "A = {:.4}, B = {:.4}",
            self.a_statistic, self.b_statistic
        )?;
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }
        Ok(())
    }
}

/// Perform Quade test on a matrix of observations.
///
/// # Arguments
///
/// * `data` - Matrix where rows are blocks and columns are treatments.
///            Each cell contains the measurement for that block-treatment combination.
/// * `treatment_names` - Names for each treatment (column)
///
/// # Returns
///
/// A `QuadeResult` containing the F statistic, degrees of freedom, and p-value.
///
/// # Example
///
/// ```
/// use p2a_core::stats::quade::{quade_test, QuadeResult};
///
/// // 5 blocks (subjects), 3 treatments
/// let data = vec![
///     vec![1.0, 2.0, 3.0],
///     vec![1.5, 2.5, 3.5],
///     vec![1.2, 2.2, 3.2],
///     vec![1.8, 2.8, 3.8],
///     vec![1.1, 2.1, 3.1],
/// ];
/// let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
///
/// let result = quade_test(&data, &names).unwrap();
/// println!("F = {}, df1 = {}, df2 = {}, p = {}",
///          result.statistic, result.df1, result.df2, result.p_value);
/// ```
///
/// # References
/// - R equivalent: `quade.test(y)`
pub fn quade_test(data: &[Vec<f64>], treatment_names: &[String]) -> EconResult<QuadeResult> {
    // Validate input
    if data.is_empty() {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: 0,
            context: "Quade test requires at least 2 blocks".to_string(),
        });
    }

    let b = data.len(); // number of blocks
    let k = data[0].len(); // number of treatments

    if b < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: b,
            context: "Quade test requires at least 2 blocks".to_string(),
        });
    }

    if k < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: k,
            context: "Quade test requires at least 2 treatments".to_string(),
        });
    }

    // Validate all rows have same length
    for (i, row) in data.iter().enumerate() {
        if row.len() != k {
            return Err(EconError::InvalidSpecification {
                message: format!("Block {} has {} treatments, expected {}", i, row.len(), k),
            });
        }
    }

    // Validate treatment names
    if treatment_names.len() != k {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Expected {} treatment names, got {}",
                k,
                treatment_names.len()
            ),
        });
    }

    // Non-finite cells cannot be ranked (run_quade_test drops incomplete blocks first)
    for (i, row) in data.iter().enumerate() {
        if row.iter().any(|v| !v.is_finite()) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Quade test: block {} contains a missing or infinite value; quade_test needs complete, finite blocks (run_quade_test drops incomplete blocks, as quade.test does)",
                    i + 1
                ),
            });
        }
    }

    // Step 1: Compute block ranges
    let mut block_ranges: Vec<f64> = resource::try_with_capacity(b, "Quade block ranges")?;
    for row in data {
        let min = row.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        block_ranges.push(max - min);
    }

    // Step 2: Rank the block ranges (Q_i), exact ties as R's rank()
    let (range_ranks, _) = midranks(&block_ranges)?;

    // Steps 3-6: rank within each block (R_ij, on reused scratch buffers), score
    // S_ij = Q_i * (R_ij - (k+1)/2), and accumulate the treatment totals S_j = Σ_i S_ij and
    // A = ΣΣ S_ij²
    let center = (k as f64 + 1.0) / 2.0;
    let mut pairs: Vec<(f64, usize)> = resource::try_with_capacity(k, "Quade ranks")?;
    let mut ranks: Vec<f64> = resource::try_with_capacity(k, "Quade ranks")?;
    let mut sizes: Vec<usize> = resource::try_with_capacity(k, "Quade ranks")?;
    let mut treatment_sums: Vec<f64> = vec![0.0; k];
    let mut a = 0.0;
    for (row, &q_i) in data.iter().zip(&range_ranks) {
        midranks_into(row, &mut pairs, &mut ranks, &mut sizes)?;
        for (sum, &r_ij) in treatment_sums.iter_mut().zip(&ranks) {
            let s_ij = q_i * (r_ij - center);
            *sum += s_ij;
            a += s_ij * s_ij;
        }
    }

    // B = (1/b) Σ S_j²
    let sum_sj_sq: f64 = treatment_sums.iter().map(|&s| s * s).sum();
    let b_stat = sum_sj_sq / b as f64;

    // Step 7: T = (b-1)B / (A - B). R: when A == B the statistic is undefined (NaN, df NA)
    // and p = (k!)^(1-b). A == B is decided as b A == Σ S_j² up to rounding (exact for
    // dyadic scores). A == 0 means every block is constant: no information, refused.
    let df1 = k - 1;
    let df2 = (b - 1) * (k - 1);
    let b_a = b as f64 * a;
    if a == 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "Quade test: every block is constant, so the treatments cannot be ranked \
                      (quade.test returns an uninformative p-value here)"
                .to_string(),
        });
    }
    let (f_stat, p_value) = if (b_a - sum_sj_sq).abs() <= 8.0 * f64::EPSILON * b_a {
        let p = ((1.0 - b as f64) * ln_gamma(k as f64 + 1.0)).exp();
        (f64::NAN, p)
    } else {
        let f = (b as f64 - 1.0) * b_stat / (a - b_stat);
        (f, f_test_p_value(f, df1 as f64, df2 as f64))
    };

    // Compute mean weighted ranks
    let mean_weighted_ranks: Vec<f64> = treatment_sums.iter().map(|&s| s / b as f64).collect();

    Ok(QuadeResult {
        statistic: f_stat,
        df1,
        df2,
        p_value,
        n_blocks: b,
        n_treatments: k,
        treatment_sums,
        mean_weighted_ranks,
        treatment_names: treatment_names.to_vec(),
        block_ranges,
        a_statistic: a,
        b_statistic: b_stat,
        missing: MissingSummary::none(b * k),
    })
}

/// Run Quade test on a Dataset with long-format data (R `quade.test(value, group, block)`).
///
/// The design must be an unreplicated complete block design, as R requires: a missing group
/// or block label is an error, and so is a (block, treatment) cell with more than one
/// observation or with none. A block whose response is missing is dropped and reported in
/// `missing`; the test then runs on the complete blocks (b = their number). R 4.3.3 keeps
/// b = the number of block levels after dropping (validation/stats/quade_test.md). Integer
/// responses are accepted; treatments follow R `factor()` order.
///
/// # Arguments
///
/// * `dataset` - The dataset containing the data
/// * `value_col` - Column name for the measured values
/// * `group_col` - Column name for the treatment/group variable
/// * `block_col` - Column name for the block/subject variable
///
/// # Returns
///
/// A `QuadeResult` containing the test results.
pub fn run_quade_test(
    dataset: &Dataset,
    value_col: &str,
    group_col: &str,
    block_col: &str,
) -> EconResult<QuadeResult> {
    let blocks = unreplicated_blocks(dataset.df(), value_col, group_col, block_col, "Quade test")?;
    if blocks.rows.len() < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: blocks.rows.len(),
            context: "Need at least 2 complete blocks for Quade test".to_string(),
        });
    }
    let mut result = quade_test(&blocks.rows, &blocks.treatment_names)?;
    result.missing = blocks.missing;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_quade_basic() {
        // Clear treatment differences across blocks
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.5, 2.5, 3.5],
            vec![1.2, 2.2, 3.2],
            vec![1.8, 2.8, 3.8],
            vec![1.1, 2.1, 3.1],
        ];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = quade_test(&data, &names).unwrap();

        assert_eq!(result.n_blocks, 5);
        assert_eq!(result.n_treatments, 3);
        assert_eq!(result.df1, 2);
        assert_eq!(result.df2, 8);
        // R gives F = 36, p = 0.0001 for this data
        assert!(
            (result.statistic - 36.0).abs() < 1.0,
            "F statistic mismatch: got {}, expected 36",
            result.statistic
        );
        // With perfect ordering, should reject H0
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_quade_no_difference() {
        // Similar values, rankings should vary
        let data = vec![
            vec![5.1, 5.0, 5.2],
            vec![4.9, 5.1, 5.0],
            vec![5.2, 5.0, 4.9],
            vec![5.0, 5.2, 5.1],
            vec![4.8, 5.0, 5.2],
        ];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = quade_test(&data, &names).unwrap();

        // With similar values, should not reject H0
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_quade_validates_input() {
        // Too few treatments
        let data = vec![vec![1.0], vec![2.0]];
        let names = vec!["A".to_string()];
        assert!(quade_test(&data, &names).is_err());

        // Empty data
        let data: Vec<Vec<f64>> = vec![];
        let names: Vec<String> = vec![];
        assert!(quade_test(&data, &names).is_err());

        // Single block
        let data = vec![vec![1.0, 2.0, 3.0]];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        assert!(quade_test(&data, &names).is_err());
    }

    #[test]
    fn test_run_quade_from_dataset() {
        // Long format data
        let df = df! {
            "block" => ["S1", "S1", "S1", "S2", "S2", "S2", "S3", "S3", "S3"],
            "treatment" => ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "value" => [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.2, 2.2, 3.2],
        }
        .unwrap();

        let dataset = Dataset::new(df);
        let result = run_quade_test(&dataset, "value", "treatment", "block").unwrap();

        assert_eq!(result.n_blocks, 3);
        assert_eq!(result.n_treatments, 3);
    }

    #[test]
    fn test_quade_block_ranges() {
        // Test that block ranges are computed correctly
        let data = vec![
            vec![1.0, 5.0, 3.0],  // range = 4
            vec![2.0, 10.0, 6.0], // range = 8
            vec![0.0, 2.0, 1.0],  // range = 2
        ];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = quade_test(&data, &names).unwrap();

        assert!((result.block_ranges[0] - 4.0).abs() < 1e-10);
        assert!((result.block_ranges[1] - 8.0).abs() < 1e-10);
        assert!((result.block_ranges[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_quade_display() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.5, 2.5, 3.5],
            vec![1.2, 2.2, 3.2],
        ];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = quade_test(&data, &names).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Quade"));
        assert!(display.contains("F ="));
        assert!(display.contains("p-value"));
    }

    // ========================================================================
    // R-pinned: complete blocks, missing responses and refused designs (T2-WP07)
    // ========================================================================

    use crate::stats::friedman::fixtures::*;
    use crate::test_support::load_case;
    use p2a_validation::{Check, Tol};

    #[test]
    fn test_validate_friedman_quade_quade() {
        let (case, ds) = load_case("stat_friedman_quade");
        let q = run_quade_test(&ds, "y", "treat", "block").unwrap();
        let qna = run_quade_test(&ds, "y_na", "treat", "block").unwrap();
        let (s, p, x) = (Tol::stat(24), Tol::p_value(24), Tol::exact());
        // na_quade_f / na_quade_df2 / na_quade_p are R's NA-path values, which keep
        // b = nlevels(blocks) = 8 after dropping block S2; they are deliberately not pinned
        // (validation/stats/quade_test.md). p2a computes Quade on the 7 complete blocks,
        // which equals quade.test on those blocks (na_quade_complete_*).
        Check::new(&case)
            .scalar("quade_f", q.statistic, s)
            .scalar("quade_df1", q.df1 as f64, x)
            .scalar("quade_df2", q.df2 as f64, x)
            .scalar("quade_p", q.p_value, p)
            .scalar("na_quade_complete_f", qna.statistic, s)
            .scalar("na_quade_complete_df1", qna.df1 as f64, x)
            .scalar("na_quade_complete_df2", qna.df2 as f64, x)
            .scalar("na_quade_complete_p", qna.p_value, p)
            .scalar("na_n_blocks", qna.n_blocks as f64, x)
            .finish();
        assert_eq!(q.treatment_names, ["A", "B", "C"]);
        assert!(q.missing.is_empty());
        assert_eq!(qna.missing.n_dropped, 3);
        assert!(
            qna.to_string()
                .contains("Missing values: 3 of 24 rows dropped"),
            "{qna}"
        );
    }

    fn quade_refusal(df: DataFrame) -> String {
        match run_quade_test(&Dataset::new(df), "y", "treat", "block") {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_quade_refuses_designs_r_refuses() {
        let (_case, ds) = load_case("stat_friedman_quade");
        let df = ds.df();
        let design = "not an unreplicated complete block design";
        let m = quade_refusal(with_duplicate_cell(df));
        assert!(
            m.contains(design) && m.contains("'S1'") && m.contains("Quade"),
            "{m}"
        );
        let m = quade_refusal(with_null_label(df, "block", 3));
        assert!(
            m.contains("NA's are not allowed in 'groups' or 'blocks'") && m.contains("'block'"),
            "{m}"
        );
        let m = quade_refusal(without_row(df, 4));
        assert!(m.contains(design), "{m}");
    }

    /// Column-type fixture: an Int64 response equals its Float64 cast; a text value column and
    /// an absent column are refused with the variant that names the problem.
    #[test]
    fn test_validate_na_quade_dtypes_and_errors() {
        let (_case, ds) = load_case("stat_friedman_quade");
        let int = run_quade_test(&ds, "y", "treat", "block").unwrap();
        let float =
            run_quade_test(&Dataset::new(as_float(ds.df(), "y")), "y", "treat", "block").unwrap();
        assert_eq!(int.statistic, float.statistic);
        assert!(matches!(
            run_quade_test(&ds, "treat", "treat", "block"),
            Err(EconError::NonNumericColumn { ref column }) if column == "treat"
        ));
        assert!(matches!(
            run_quade_test(&ds, "nope", "treat", "block"),
            Err(EconError::ColumnNotFound { ref column, .. }) if column == "nope"
        ));
    }

    #[test]
    fn test_validate_quade_small_scale() {
        let (case, ds) = load_case("stat_rank_small_blocks");
        let r = run_quade_test(&ds, "y", "treat", "block").unwrap();
        Check::new(&case)
            .scalar("quade_f", r.statistic, Tol::stat(32))
            .scalar("quade_df1", r.df1 as f64, Tol::exact())
            .scalar("quade_df2", r.df2 as f64, Tol::exact())
            .scalar("quade_p", r.p_value, Tol::p_value(32))
            .named(
                "quade_treatment_sums",
                &r.treatment_names,
                &r.treatment_sums,
                Tol::stat(32),
            )
            .scalar("quade_a", r.a_statistic, Tol::stat(32))
            .scalar("quade_b", r.b_statistic, Tol::stat(32))
            .finish();
    }

    #[test]
    fn test_validate_quade_store() {
        // R's own ?quade.test data (Conover 1999, p. 375); the old test used a modified row 6.
        let (case, ds) = load_case("stat_rank_quade_store");
        let r = run_quade_test(&ds, "sales", "brand", "store").unwrap();
        Check::new(&case)
            .scalar("f", r.statistic, Tol::stat(35))
            .scalar("df1", r.df1 as f64, Tol::exact())
            .scalar("df2", r.df2 as f64, Tol::exact())
            .scalar("p", r.p_value, Tol::p_value(35))
            .named(
                "treatment_sums",
                &r.treatment_names,
                &r.treatment_sums,
                Tol::stat(35),
            )
            .scalar("a", r.a_statistic, Tol::stat(35))
            .scalar("b", r.b_statistic, Tol::stat(35))
            .vector("block_ranges", &r.block_ranges, Tol::exact())
            .finish();
    }

    #[test]
    fn test_quade_rejects_non_finite() {
        let names: Vec<String> = ["A", "B", "C"].iter().map(|s| s.to_string()).collect();
        for bad in [f64::NAN, f64::INFINITY] {
            match quade_test(
                &[
                    vec![1.0, bad, 2.0],
                    vec![1.0, 2.0, 3.0],
                    vec![2.0, 3.0, 1.0],
                ],
                &names,
            ) {
                Err(EconError::InvalidSpecification { message }) => {
                    assert!(message.contains("block 1"), "{message}")
                }
                other => panic!("{bad}: expected InvalidSpecification, got {other:?}"),
            }
        }
    }

    /// R's A == B branch: identical rankings with tied ranges give a NaN statistic and
    /// p = (k!)^(1-b); distinct ranges give the F test; all-constant blocks are refused.
    /// Fails before: F = 0 and p = 1 when A == B.
    #[test]
    fn test_validate_quade_a_equals_b_t4() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_rank_quade_t4");
        let matrix = |set: &str| -> Vec<Vec<f64>> {
            let y = crate::stats::ranks::test_data::column_where(&ds, "y", &[("set", set)]);
            y.chunks(3).map(|c| c.to_vec()).collect()
        };
        let names = ["t1", "t2", "t3"].map(String::from);
        let conc = quade_test(&matrix("conc"), &names).unwrap();
        let dist = quade_test(&matrix("dist"), &names).unwrap();
        assert!(conc.statistic.is_nan());
        assert!(conc.to_string().contains("NaN"), "{conc}");
        // deviation STAT-64: const_p_r (R's p for all-constant blocks) is not compared
        match quade_test(&matrix("const"), &names) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("every block is constant"), "{message}")
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        Check::new(&case)
            .scalar("conc_p", conc.p_value, Tol::p_value(12))
            .scalar(
                "conc_stat_is_nan",
                f64::from(u8::from(conc.statistic.is_nan())),
                Tol::exact(),
            )
            .scalar("dist_f", dist.statistic, Tol::stat(12))
            .scalar("dist_p", dist.p_value, Tol::p_value(12))
            .finish();
    }
}
