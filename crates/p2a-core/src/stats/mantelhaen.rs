//! Cochran-Mantel-Haenszel Chi-Squared Test for Count Data.
//!
//! Tests the null hypothesis that two nominal variables are conditionally
//! independent in each stratum, assuming no three-way interaction.
//!
//! # Mathematical Background
//!
//! For 2×2×K tables, the CMH statistic is:
//!
//! ```text
//! CMH = (|Σ(a_k - E[a_k])| - 0.5)² / Σ Var(a_k)
//!
//! where:
//!   E[a_k] = n1_k × m1_k / n_k    (expected count for cell a in stratum k)
//!   Var(a_k) = n1_k × n2_k × m1_k × m2_k / (n_k² × (n_k - 1))
//! ```
//!
//! The Mantel-Haenszel common odds ratio estimator is:
//!
//! ```text
//! OR_MH = Σ(a_k × d_k / n_k) / Σ(b_k × c_k / n_k)
//! ```
//!
//! # References
//!
//! - Cochran, W. G. (1954). "Some Methods for Strengthening the Common χ² Tests".
//!   Biometrics, 10(4), 417-451.
//! - Mantel, N. & Haenszel, W. (1959). "Statistical Aspects of the Analysis of Data
//!   from Retrospective Studies of Disease". JNCI, 22(4), 719-748.
//! - R Core Team. `stats::mantelhaen.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mantelhaen.test.html>

use crate::data::Dataset;
use crate::data::columns::MissingSummary;
use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::stats::contingency::stratified_tables_2x2;
use crate::traits::chi_squared_p_value;
use serde::{Deserialize, Serialize};

/// Result of the Cochran-Mantel-Haenszel test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MantelHaenszelResult {
    /// Test method name
    pub test_name: String,
    /// CMH chi-squared statistic
    pub statistic: f64,
    /// Degrees of freedom (always 1 for 2×2×K)
    pub df: usize,
    /// P-value
    pub p_value: f64,
    /// Number of strata
    pub n_strata: usize,
    /// Total sample size across all strata
    pub total_n: usize,
    /// Mantel-Haenszel common odds ratio estimate
    pub common_odds_ratio: f64,
    /// 95% confidence interval for the common odds ratio
    pub odds_ratio_ci: (f64, f64),
    /// Whether continuity correction was applied
    pub continuity_correction: bool,
    /// Alternative hypothesis
    pub alternative: CmhAlternative,
    /// Per-stratum statistics
    pub stratum_stats: Vec<StratumStats>,
    /// Row level labels, common to every stratum (empty for array input).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub row_levels: Vec<String>,
    /// Column level labels, common to every stratum (empty for array input).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub col_levels: Vec<String>,
    /// Rows dropped because of a missing value (dataset input).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

/// Statistics for a single stratum (2×2 table).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumStats {
    /// Stratum identifier/name
    pub stratum: String,
    /// Cell counts [a, b, c, d] in row-major order
    pub counts: [f64; 4],
    /// Expected count for cell 'a' under conditional independence
    pub expected_a: f64,
    /// Variance of cell 'a' count
    pub variance_a: f64,
    /// Stratum total
    pub n: f64,
    /// Stratum-specific odds ratio
    pub odds_ratio: f64,
}

/// Alternative hypothesis for the CMH test.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CmhAlternative {
    /// Two-sided test (common odds ratio ≠ 1)
    TwoSided,
    /// One-sided test (common odds ratio > 1)
    Greater,
    /// One-sided test (common odds ratio < 1)
    Less,
}

impl std::fmt::Display for MantelHaenszelResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;
        writeln!(
            f,
            "CMH χ² = {:.4}, df = {}, p-value = {:.6}",
            self.statistic, self.df, self.p_value
        )?;
        writeln!(f)?;
        writeln!(f, "Common odds ratio: {:.4}", self.common_odds_ratio)?;
        writeln!(
            f,
            "95% CI: ({:.4}, {:.4})",
            self.odds_ratio_ci.0, self.odds_ratio_ci.1
        )?;
        writeln!(f)?;
        writeln!(f, "Strata: {}, Total N: {}", self.n_strata, self.total_n)?;
        if !self.row_levels.is_empty() || !self.col_levels.is_empty() {
            writeln!(
                f,
                "Rows: {}; columns: {}",
                self.row_levels.join(", "),
                self.col_levels.join(", ")
            )?;
        }
        if self.continuity_correction {
            writeln!(f, "(Continuity correction applied)")?;
        }
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }
        Ok(())
    }
}

/// A 2×2 table for one stratum.
///
/// Layout:
/// ```text
///            Col1  Col2
///    Row1     a     b   | n1 = a + b
///    Row2     c     d   | n2 = c + d
///           -----+-----
///            m1    m2   |  n = total
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Table2x2 {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl Table2x2 {
    /// Create a new 2×2 table.
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self { a, b, c, d }
    }

    /// Total count in this stratum.
    pub fn n(&self) -> f64 {
        self.a + self.b + self.c + self.d
    }

    /// Row 1 total (a + b).
    pub fn row1(&self) -> f64 {
        self.a + self.b
    }

    /// Row 2 total (c + d).
    pub fn row2(&self) -> f64 {
        self.c + self.d
    }

    /// Column 1 total (a + c).
    pub fn col1(&self) -> f64 {
        self.a + self.c
    }

    /// Column 2 total (b + d).
    pub fn col2(&self) -> f64 {
        self.b + self.d
    }

    /// Expected count for cell 'a' under conditional independence.
    pub fn expected_a(&self) -> f64 {
        let n = self.n();
        if n > 0.0 {
            self.row1() * self.col1() / n
        } else {
            0.0
        }
    }

    /// Variance of cell 'a' under conditional independence.
    pub fn variance_a(&self) -> f64 {
        let n = self.n();
        if n > 1.0 {
            self.row1() * self.row2() * self.col1() * self.col2() / (n * n * (n - 1.0))
        } else {
            0.0
        }
    }

    /// Odds ratio for this table.
    pub fn odds_ratio(&self) -> f64 {
        let bc = self.b * self.c;
        if bc > 0.0 {
            (self.a * self.d) / bc
        } else {
            f64::INFINITY
        }
    }
}

/// Perform the Cochran-Mantel-Haenszel test on a vector of 2×2 tables.
///
/// # Arguments
///
/// * `tables` - Vector of 2×2 tables, one per stratum
/// * `stratum_names` - Optional names for each stratum
/// * `correct` - Whether to apply Yates' continuity correction (default: true)
/// * `alternative` - Direction of alternative hypothesis
///
/// # Returns
///
/// A `MantelHaenszelResult` containing the test statistic, p-value, and odds ratio estimate.
///
/// # Example
///
/// ```
/// use p2a_core::stats::mantelhaen::{mantelhaen_test, Table2x2, CmhAlternative};
///
/// // Two strata with 2×2 tables
/// let tables = vec![
///     Table2x2::new(11.0, 43.0, 42.0, 169.0),  // Stratum 1
///     Table2x2::new(14.0, 104.0, 20.0, 138.0), // Stratum 2
/// ];
/// let names = vec!["Center1".to_string(), "Center2".to_string()];
///
/// let result = mantelhaen_test(&tables, Some(&names), true, CmhAlternative::TwoSided).unwrap();
/// println!("CMH χ² = {}, p = {}", result.statistic, result.p_value);
/// println!("Common OR = {}", result.common_odds_ratio);
/// ```
pub fn mantelhaen_test(
    tables: &[Table2x2],
    stratum_names: Option<&[String]>,
    correct: bool,
    alternative: CmhAlternative,
) -> EconResult<MantelHaenszelResult> {
    // Validate input
    if tables.is_empty() {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "CMH test requires at least 1 stratum".to_string(),
        });
    }

    let k = tables.len();

    // Compute DELTA = Σ(a_k - E[a_k])
    let mut delta = 0.0;
    let mut variance = 0.0;
    let mut or_numerator = 0.0; // Σ(a*d/n)
    let mut or_denominator = 0.0; // Σ(b*c/n)
    let mut total_n = 0.0;

    let mut stratum_stats = Vec::with_capacity(k);

    for (i, table) in tables.iter().enumerate() {
        let n = table.n();
        if n < 1.0 {
            continue; // Skip empty strata
        }

        let exp_a = table.expected_a();
        let var_a = table.variance_a();

        delta += table.a - exp_a;
        variance += var_a;
        total_n += n;

        // Mantel-Haenszel OR components
        or_numerator += table.a * table.d / n;
        or_denominator += table.b * table.c / n;

        let stratum_name = stratum_names
            .and_then(|names| names.get(i))
            .cloned()
            .unwrap_or_else(|| format!("Stratum{}", i + 1));

        stratum_stats.push(StratumStats {
            stratum: stratum_name,
            counts: [table.a, table.b, table.c, table.d],
            expected_a: exp_a,
            variance_a: var_a,
            n,
            odds_ratio: table.odds_ratio(),
        });
    }

    // Apply continuity correction
    let yates = if correct && delta.abs() >= 0.5 {
        0.5
    } else {
        0.0
    };

    // CMH statistic
    let statistic = if variance > 0.0 {
        let adj_delta = delta.abs() - yates;
        (adj_delta * adj_delta) / variance
    } else {
        0.0
    };

    // P-value from chi-squared distribution with df=1
    let p_value_two_sided = chi_squared_p_value(statistic, 1.0);

    let p_value = match alternative {
        CmhAlternative::TwoSided => p_value_two_sided,
        CmhAlternative::Greater => {
            // Test for OR > 1 (positive association)
            if delta > 0.0 {
                p_value_two_sided / 2.0
            } else {
                1.0 - p_value_two_sided / 2.0
            }
        }
        CmhAlternative::Less => {
            // Test for OR < 1 (negative association)
            if delta < 0.0 {
                p_value_two_sided / 2.0
            } else {
                1.0 - p_value_two_sided / 2.0
            }
        }
    };

    // Common odds ratio (Mantel-Haenszel estimator)
    let common_odds_ratio = if or_denominator > 0.0 {
        or_numerator / or_denominator
    } else {
        f64::INFINITY
    };

    // Confidence interval for OR using Robins et al. (1986) variance
    let (or_lower, or_upper) = compute_or_confidence_interval(tables, common_odds_ratio, 0.95)?;

    let test_name = if correct {
        "Cochran-Mantel-Haenszel chi-squared test (with continuity correction)"
    } else {
        "Cochran-Mantel-Haenszel chi-squared test"
    }
    .to_string();

    Ok(MantelHaenszelResult {
        test_name,
        statistic,
        df: 1,
        p_value,
        n_strata: k,
        total_n: total_n as usize,
        common_odds_ratio,
        odds_ratio_ci: (or_lower, or_upper),
        continuity_correction: correct && yates > 0.0,
        alternative,
        stratum_stats,
        row_levels: Vec::new(),
        col_levels: Vec::new(),
        missing: MissingSummary::none(total_n as usize),
    })
}

/// Compute confidence interval for the common odds ratio using Robins et al. (1986) formula.
fn compute_or_confidence_interval(
    tables: &[Table2x2],
    or_mh: f64,
    conf_level: f64,
) -> EconResult<(f64, f64)> {
    // Robins, Breslow, Greenland (1986) variance formula
    // Var(ln(OR_MH)) ≈ Σ{P_k R_k / (2R²)} + Σ{(P_k S_k + Q_k R_k) / (2RS)} + Σ{Q_k S_k / (2S²)}
    // where R = Σ(a*d/n), S = Σ(b*c/n), P = (a+d)/n, Q = (b+c)/n

    let mut r_sum = 0.0;
    let mut s_sum = 0.0;
    let mut pr_sum = 0.0;
    let mut ps_plus_qr_sum = 0.0;
    let mut qs_sum = 0.0;

    for table in tables {
        let n = table.n();
        if n < 1.0 {
            continue;
        }

        let r_k = table.a * table.d / n;
        let s_k = table.b * table.c / n;
        let p_k = (table.a + table.d) / n;
        let q_k = (table.b + table.c) / n;

        r_sum += r_k;
        s_sum += s_k;
        pr_sum += p_k * r_k;
        ps_plus_qr_sum += p_k * s_k + q_k * r_k;
        qs_sum += q_k * s_k;
    }

    // Avoid division by zero
    if r_sum <= 0.0 || s_sum <= 0.0 {
        return Ok((0.0, f64::INFINITY));
    }

    let var_ln_or = pr_sum / (2.0 * r_sum * r_sum)
        + ps_plus_qr_sum / (2.0 * r_sum * s_sum)
        + qs_sum / (2.0 * s_sum * s_sum);

    // z critical value for confidence level
    let z = crate::traits::estimator::z_critical(conf_level)?;

    let ln_or = or_mh.ln();
    let se_ln_or = var_ln_or.sqrt();

    let lower = (ln_or - z * se_ln_or).exp();
    let upper = (ln_or + z * se_ln_or).exp();

    Ok((lower, upper))
}

/// Run CMH test on a Dataset with factor columns.
///
/// Builds the 2 × 2 × K tables as [`stratified_tables`](crate::stats::stratified_tables)
/// does, as R's `mantelhaen.test(table(row, col, stratum))`: rows with a missing value
/// (null, or NaN in a float column) in any of the three columns are dropped and reported in
/// `missing`, and the row, column and stratum levels follow R `factor()` order (numbers
/// numerically, text in C-locale order), so the first level of each is the first row or
/// column of every stratum's table.
///
/// # Arguments
///
/// * `dataset` - The dataset containing the data
/// * `row_var` - Column name for the row variable (binary factor)
/// * `col_var` - Column name for the column variable (binary factor)
/// * `stratum_var` - Column name for the stratum/group variable
/// * `correct` - Whether to apply continuity correction
/// * `alternative` - Direction of alternative hypothesis
///
/// # Errors
///
/// [`stratified_tables`](crate::stats::stratified_tables)' errors (`ColumnNotFound`, a column
/// that cannot be categorical, no row left, `ResourceLimit`, `Cancelled`), and
/// `InvalidSpecification` naming the levels when the row or column variable does not have
/// exactly 2 levels, checked before any table is built.
///
/// # Returns
///
/// A `MantelHaenszelResult` containing the test results.
pub fn run_mantelhaen_test(
    dataset: &Dataset,
    row_var: &str,
    col_var: &str,
    stratum_var: &str,
    correct: bool,
    alternative: CmhAlternative,
) -> EconResult<MantelHaenszelResult> {
    let (strata, tables, missing) =
        stratified_tables_2x2(dataset.df(), row_var, col_var, stratum_var)?;
    let Some(first) = tables.first() else {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Mantel-Haenszel test: no stratum is left after removing rows with \
                      missing values"
                .to_string(),
        });
    };
    let (row_levels, col_levels) = (first.row_levels.clone(), first.col_levels.clone());
    let mut cells: Vec<Table2x2> = resource::try_with_capacity(tables.len(), "stratum tables")?;
    for t in &tables {
        // every stratum has the common 2 x 2 levels
        let [[a, b], [c, d]] = t
            .as_2x2()
            .ok_or_else(|| EconError::Computation("a stratum's table is not 2 x 2".to_string()))?;
        cells.push(Table2x2::new(a, b, c, d));
    }
    let mut result = mantelhaen_test(&cells, Some(&strata), correct, alternative)?;
    result.row_levels = row_levels;
    result.col_levels = col_levels;
    result.missing = missing;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_mantelhaen_basic() {
        // UCBAdmissions data example from R
        // Simplified to 2 strata
        let tables = vec![
            Table2x2::new(512.0, 313.0, 89.0, 19.0), // Dept A
            Table2x2::new(353.0, 207.0, 17.0, 8.0),  // Dept B
        ];
        let names = vec!["A".to_string(), "B".to_string()];

        let result =
            mantelhaen_test(&tables, Some(&names), true, CmhAlternative::TwoSided).unwrap();

        println!(
            "CMH = {}, df = {}, p = {}",
            result.statistic, result.df, result.p_value
        );
        println!("Common OR = {:.4}", result.common_odds_ratio);
        println!(
            "95% CI: ({:.4}, {:.4})",
            result.odds_ratio_ci.0, result.odds_ratio_ci.1
        );

        assert_eq!(result.n_strata, 2);
        assert_eq!(result.df, 1);
        assert!(result.statistic > 0.0);
    }

    #[test]
    fn test_table2x2_computations() {
        let table = Table2x2::new(10.0, 20.0, 30.0, 40.0);

        assert!((table.n() - 100.0).abs() < 1e-10);
        assert!((table.row1() - 30.0).abs() < 1e-10);
        assert!((table.row2() - 70.0).abs() < 1e-10);
        assert!((table.col1() - 40.0).abs() < 1e-10);
        assert!((table.col2() - 60.0).abs() < 1e-10);

        // Expected a = row1 * col1 / n = 30 * 40 / 100 = 12
        assert!((table.expected_a() - 12.0).abs() < 1e-10);

        // Variance = row1 * row2 * col1 * col2 / (n² * (n-1))
        // = 30 * 70 * 40 * 60 / (10000 * 99)
        let expected_var = 30.0 * 70.0 * 40.0 * 60.0 / (10000.0 * 99.0);
        assert!((table.variance_a() - expected_var).abs() < 1e-6);

        // OR = (a*d)/(b*c) = (10*40)/(20*30) = 400/600 = 2/3
        assert!((table.odds_ratio() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mantelhaen_single_stratum() {
        // Single 2×2 table - should be similar to chi-squared test
        let tables = vec![Table2x2::new(50.0, 30.0, 20.0, 100.0)];

        let result = mantelhaen_test(&tables, None, true, CmhAlternative::TwoSided).unwrap();

        assert_eq!(result.n_strata, 1);
        assert!(result.statistic > 0.0);
        // With clear association, p should be small
        println!(
            "Single stratum: CMH = {}, p = {}",
            result.statistic, result.p_value
        );
    }

    // ========================================================================
    // Validation Tests Against R
    // ========================================================================

    #[test]
    fn test_validate_mantelhaen_against_r() {
        // R code:
        // Rabbits <- array(c(
        //   0, 0, 6, 5,   # stratum 1: v[1:4] -> [1,1]=0, [2,1]=0, [1,2]=6, [2,2]=5
        //   3, 0, 3, 6,   # stratum 2: v[5:8] -> [1,1]=3, [2,1]=0, [1,2]=3, [2,2]=6
        //   6, 2, 0, 4,   # stratum 3: v[9:12] -> [1,1]=6, [2,1]=2, [1,2]=0, [2,2]=4
        //   5, 6, 1, 0,   # stratum 4: v[13:16] -> [1,1]=5, [2,1]=6, [1,2]=1, [2,2]=0
        //   2, 5, 0, 0    # stratum 5: v[17:20] -> [1,1]=2, [2,1]=5, [1,2]=0, [2,2]=0
        // ), dim = c(2, 2, 5),
        // dimnames = list(
        //   Delay = c("None", "1.5h"),
        //   Response = c("Cured", "Died"),
        //   Penicillin.Level = c("1/8", "1/4", "1/2", "1", "4")
        // ))
        // mantelhaen.test(Rabbits)
        //
        // R fills column-major: v[i*4+1], v[i*4+2], v[i*4+3], v[i*4+4] -> a=[1,1], c=[2,1], b=[1,2], d=[2,2]
        // Table2x2::new(a, b, c, d) where:
        //   a = v[1], b = v[3], c = v[2], d = v[4]
        //
        // Expected output:
        // Mantel-Haenszel chi-squared = 3.9286, df = 1, p-value = 0.04747
        // common odds ratio estimate: 7

        let tables = vec![
            Table2x2::new(0.0, 6.0, 0.0, 5.0), // 1/8: a=0, b=6, c=0, d=5
            Table2x2::new(3.0, 3.0, 0.0, 6.0), // 1/4: a=3, b=3, c=0, d=6
            Table2x2::new(6.0, 0.0, 2.0, 4.0), // 1/2: a=6, b=0, c=2, d=4
            Table2x2::new(5.0, 1.0, 6.0, 0.0), // 1:   a=5, b=1, c=6, d=0
            Table2x2::new(2.0, 0.0, 5.0, 0.0), // 4:   a=2, b=0, c=5, d=0
        ];
        let names = vec![
            "1/8".to_string(),
            "1/4".to_string(),
            "1/2".to_string(),
            "1".to_string(),
            "4".to_string(),
        ];

        let result =
            mantelhaen_test(&tables, Some(&names), true, CmhAlternative::TwoSided).unwrap();

        println!("R validation:");
        println!(
            "CMH = {:.4}, df = {}, p = {:.5}",
            result.statistic, result.df, result.p_value
        );
        println!("Common OR = {:.4}", result.common_odds_ratio);
        println!(
            "95% CI: ({:.4}, {:.4})",
            result.odds_ratio_ci.0, result.odds_ratio_ci.1
        );

        // R gives: chi-squared = 3.9286, p = 0.04747, OR = 7, 95% CI: (1.027, 47.73)
        assert!(
            (result.statistic - 3.9286).abs() < 0.1,
            "CMH statistic mismatch: got {:.4}, expected 3.9286",
            result.statistic
        );
        assert_eq!(result.df, 1);
        assert!(
            (result.p_value - 0.04747).abs() < 0.01,
            "p-value mismatch: got {:.5}, expected 0.04747",
            result.p_value
        );
        assert!(
            (result.common_odds_ratio - 7.0).abs() < 0.5,
            "OR mismatch: got {:.4}, expected 7.0",
            result.common_odds_ratio
        );
    }

    #[test]
    fn test_run_mantelhaen_from_dataset() {
        // Create test data
        let df = df! {
            "exposure" => ["Yes", "Yes", "Yes", "No", "No", "No", "Yes", "Yes", "Yes", "No", "No", "No"],
            "outcome" => ["Disease", "Disease", "Healthy", "Disease", "Healthy", "Healthy",
                          "Disease", "Healthy", "Healthy", "Disease", "Healthy", "Healthy"],
            "stratum" => ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
        }
        .unwrap();

        let dataset = Dataset::new(df);
        let result = run_mantelhaen_test(
            &dataset,
            "exposure",
            "outcome",
            "stratum",
            true,
            CmhAlternative::TwoSided,
        )
        .unwrap();

        assert_eq!(result.n_strata, 2);
        assert_eq!(result.df, 1);
        println!(
            "Dataset test: CMH = {}, p = {}",
            result.statistic, result.p_value
        );
    }

    #[test]
    fn test_mantelhaen_alternatives() {
        let tables = vec![
            Table2x2::new(20.0, 10.0, 5.0, 30.0),
            Table2x2::new(25.0, 15.0, 8.0, 35.0),
        ];

        let two_sided = mantelhaen_test(&tables, None, true, CmhAlternative::TwoSided).unwrap();
        let greater = mantelhaen_test(&tables, None, true, CmhAlternative::Greater).unwrap();
        let less = mantelhaen_test(&tables, None, true, CmhAlternative::Less).unwrap();

        // Same statistic for all
        assert!((two_sided.statistic - greater.statistic).abs() < 1e-10);
        assert!((two_sided.statistic - less.statistic).abs() < 1e-10);

        // One-sided p-values are half of two-sided (for the correct direction)
        println!(
            "Two-sided p = {:.4}, Greater p = {:.4}, Less p = {:.4}",
            two_sided.p_value, greater.p_value, less.p_value
        );
    }

    #[test]
    fn test_mantelhaen_display() {
        let tables = vec![Table2x2::new(50.0, 30.0, 20.0, 100.0)];
        let result = mantelhaen_test(&tables, None, true, CmhAlternative::TwoSided).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Cochran-Mantel-Haenszel"));
        assert!(display.contains("CMH"));
        assert!(display.contains("p-value"));
    }

    #[test]
    fn test_validate_mantelhaen_na() {
        // R: mantelhaen.test(table(treat, outcome, stratum)) drops the 3 NA strata: X2
        // 9.464008299682503, p 0.0020954268679205345; correct = FALSE 10.292389179376896,
        // p 0.0013358001984906988; common OR 2.397932948724719, CI 1.4029852155370948,
        // 4.098462594545848; 237 rows (case stat_mantelhaen_na). Fails before: AnyValue::Null
        // formats as "null", a 4th stratum: n_strata 4, total_n 240. (Its 3 rows are all "no",
        // so they add nothing to DELTA, the variance or the OR, and the statistic happens to
        // match R.)
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_mantelhaen_na");
        let r = run_mantelhaen_test(
            &ds,
            "treat",
            "outcome",
            "stratum",
            true,
            CmhAlternative::TwoSided,
        )
        .unwrap();
        let u = run_mantelhaen_test(
            &ds,
            "treat",
            "outcome",
            "stratum",
            false,
            CmhAlternative::TwoSided,
        )
        .unwrap();
        Check::new(&case)
            .scalar("statistic", r.statistic, Tol::stat(237))
            .scalar("p", r.p_value, Tol::p_value(237))
            .scalar("statistic_nocorrect", u.statistic, Tol::stat(237))
            .scalar("p_nocorrect", u.p_value, Tol::p_value(237))
            .scalar("common_or", r.common_odds_ratio, Tol::stat(237))
            .scalar("ci_lower", r.odds_ratio_ci.0, Tol::stat(237))
            .scalar("ci_upper", r.odds_ratio_ci.1, Tol::stat(237))
            .scalar("n_used", r.total_n as f64, Tol::exact())
            .scalar("n_used", r.missing.n_used as f64, Tol::exact())
            .finish();
        assert_eq!(r.n_strata, 3);
        assert_eq!(
            r.stratum_stats
                .iter()
                .map(|s| s.stratum.as_str())
                .collect::<Vec<_>>(),
            ["s1", "s2", "s3"]
        );
        assert_eq!(
            (r.row_levels.clone(), r.col_levels.clone()),
            (
                vec!["ctl".to_string(), "trt".into()],
                vec!["no".to_string(), "yes".into()]
            )
        );
        assert_eq!(r.missing.by_column, vec![("stratum".to_string(), 3)]);
        assert!(
            r.to_string()
                .contains("Missing values: 3 of 240 rows dropped")
        );
    }

    #[test]
    fn test_run_mantelhaen_refuses_tables_that_are_not_2x2() {
        // three treatment levels: the error names the levels after dropping missing rows
        let df = df! {
            "t" => ["a", "b", "c", "a", "b", "c"],
            "y" => ["n", "y", "n", "y", "n", "y"],
            "s" => ["1", "1", "1", "2", "2", "2"],
        }
        .unwrap();
        let e = run_mantelhaen_test(
            &Dataset::new(df),
            "t",
            "y",
            "s",
            true,
            CmhAlternative::TwoSided,
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("'t' has 3 level(s) [a, b, c]") && e.contains("2 x 2 x K"),
            "{e}"
        );
    }

    #[test]
    fn test_run_mantelhaen_refuses_the_shape_before_building_the_tables() {
        // Two 40-level columns in 2 strata are refused for their shape before the 2 x 40 x 40
        // counts are allocated; before, a small allocation ceiling reported a ResourceLimit for
        // tables that could only be refused.
        let a: Vec<i64> = (0..40).collect();
        let s: Vec<&str> = (0..40)
            .map(|i| if i % 3 == 0 { "s1" } else { "s2" })
            .collect();
        let ds = Dataset::new(df!("a" => a.clone(), "b" => a, "s" => s).unwrap());
        let r = resource::with_alloc_ceiling(Some(1024), || {
            run_mantelhaen_test(&ds, "a", "b", "s", true, CmhAlternative::TwoSided)
        });
        assert!(
            matches!(&r, Err(EconError::InvalidSpecification { message })
                if message.contains("2 x 2") && message.contains("'b' has 40 level(s)")),
            "{r:?}"
        );
    }
}
