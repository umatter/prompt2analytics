//! Fisher's Exact Test for count data.
//!
//! Provides Fisher's exact test for 2×2 contingency tables, which tests for
//! independence between two categorical variables using exact probability
//! calculations based on the hypergeometric distribution.
//!
//! # References
//!
//! - Fisher, R. A. (1935). "The logic of inductive inference".
//!   *Journal of the Royal Statistical Society*, 98(1), 39-82.
//! - Fisher, R. A. (1922). "On the interpretation of χ² from contingency tables,
//!   and the calculation of P". *Journal of the Royal Statistical Society*,
//!   85(1), 87-94.
//! - R Core Team. `stats::fisher.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html>
//! - SciPy. `scipy.stats.fisher_exact()` function.
//!   <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html>
//!
//! # Mathematical Background
//!
//! ## Hypergeometric Distribution
//!
//! For a 2×2 table:
//! ```text
//!              | Col 1 | Col 2 | Total
//!    ----------+-------+-------+-------
//!    Row 1     |   a   |   b   | a + b
//!    Row 2     |   c   |   d   | c + d
//!    ----------+-------+-------+-------
//!    Total     | a + c | b + d |   n
//! ```
//!
//! Given fixed marginals, the probability of observing exactly `a` in the
//! top-left cell follows the hypergeometric distribution:
//!
//! ```text
//! P(X = a) = C(a+b, a) × C(c+d, c) / C(n, a+c)
//!          = (a+b)! × (c+d)! × (a+c)! × (b+d)! / (a! × b! × c! × d! × n!)
//! ```
//!
//! ## Odds Ratio
//!
//! The sample odds ratio is: `OR = (a × d) / (b × c)`
//!
//! Under the null hypothesis of independence, the true odds ratio is 1.
//!
//! ## P-Value Calculation
//!
//! - **Two-sided**: Sum probabilities of all tables with probability ≤ observed
//! - **Greater**: P(X ≥ a) - tests for positive association
//! - **Less**: P(X ≤ a) - tests for negative association

use serde::{Deserialize, Serialize};
use statrs::distribution::{Discrete, DiscreteCDF, Hypergeometric};

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::traits::SignificanceLevel;

/// Alternative hypothesis for Fisher's exact test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FisherAlternative {
    /// Two-sided test: odds ratio ≠ 1
    #[default]
    TwoSided,
    /// One-sided test: odds ratio > 1 (positive association)
    Greater,
    /// One-sided test: odds ratio < 1 (negative association)
    Less,
}

impl std::fmt::Display for FisherAlternative {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FisherAlternative::TwoSided => write!(f, "two.sided"),
            FisherAlternative::Greater => write!(f, "greater"),
            FisherAlternative::Less => write!(f, "less"),
        }
    }
}

/// Result of Fisher's exact test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FisherExactResult {
    // ═══════════════════════════════════════════════════════════════════════
    // Test Identification
    // ═══════════════════════════════════════════════════════════════════════
    /// Description of the test performed
    pub test_name: String,

    // ═══════════════════════════════════════════════════════════════════════
    // Primary Results
    // ═══════════════════════════════════════════════════════════════════════
    /// P-value from the exact test
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,
    /// Alternative hypothesis tested
    pub alternative: FisherAlternative,

    // ═══════════════════════════════════════════════════════════════════════
    // Odds Ratio
    // ═══════════════════════════════════════════════════════════════════════
    /// Sample odds ratio: (a × d) / (b × c)
    pub odds_ratio: f64,
    /// Confidence interval for odds ratio (if computed)
    pub odds_ratio_ci: Option<(f64, f64)>,
    /// Confidence level used for CI
    pub conf_level: Option<f64>,

    // ═══════════════════════════════════════════════════════════════════════
    // Table Data
    // ═══════════════════════════════════════════════════════════════════════
    /// The 2×2 contingency table [a, b, c, d] in row-major order
    pub table: [f64; 4],
    /// Row marginals [a+b, c+d]
    pub row_totals: [f64; 2],
    /// Column marginals [a+c, b+d]
    pub col_totals: [f64; 2],
    /// Grand total
    pub n: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Probability Under Null
    // ═══════════════════════════════════════════════════════════════════════
    /// Probability of the observed table under null hypothesis
    pub prob_observed: f64,
}

impl std::fmt::Display for FisherExactResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;

        // Table display
        writeln!(f, "Observed:")?;
        writeln!(f, "         Col1    Col2    Total")?;
        writeln!(
            f,
            "  Row1 {:>6.0}  {:>6.0}  {:>6.0}",
            self.table[0], self.table[1], self.row_totals[0]
        )?;
        writeln!(
            f,
            "  Row2 {:>6.0}  {:>6.0}  {:>6.0}",
            self.table[2], self.table[3], self.row_totals[1]
        )?;
        writeln!(
            f,
            "  Total {:>5.0}  {:>6.0}  {:>6.0}",
            self.col_totals[0], self.col_totals[1], self.n
        )?;
        writeln!(f)?;

        // P-value
        writeln!(
            f,
            "p-value = {:.6} {}",
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(
            f,
            "alternative hypothesis: true odds ratio is not equal to 1"
        )?;
        writeln!(f)?;

        // Odds ratio
        if self.odds_ratio.is_finite() {
            writeln!(f, "sample odds ratio: {:.6}", self.odds_ratio)?;
            if let (Some((lo, hi)), Some(level)) = (self.odds_ratio_ci, self.conf_level) {
                writeln!(
                    f,
                    "{:.0}% confidence interval: ({:.4}, {:.4})",
                    level * 100.0,
                    lo,
                    hi
                )?;
            }
        } else {
            writeln!(f, "sample odds ratio: Inf (zero cell present)")?;
        }

        Ok(())
    }
}

/// Fisher's exact test for a 2×2 contingency table.
///
/// Tests the null hypothesis that the odds ratio equals 1 (independence)
/// using exact probability calculations based on the hypergeometric distribution.
///
/// # Arguments
///
/// * `table` - A 2×2 contingency table as `[[a, b], [c, d]]`
/// * `alternative` - Alternative hypothesis: TwoSided, Greater, or Less
/// * `conf_level` - Confidence level for odds ratio CI (e.g., 0.95)
///
/// # Returns
///
/// `FisherExactResult` containing p-value, odds ratio, and optionally CI.
///
/// # Mathematical Details
///
/// The test uses the hypergeometric distribution to compute exact p-values.
/// For a table:
/// ```text
///     | a | b | a+b
///     | c | d | c+d
///     a+c b+d   n
/// ```
///
/// The probability of observing exactly `a` in the top-left cell (given
/// fixed marginals) is:
/// ```text
/// P(X = a) = C(a+b, a) × C(c+d, c) / C(n, a+c)
/// ```
///
/// # Example
///
/// ```
/// use p2a_core::stats::fisher::{fisher_exact_test, FisherAlternative};
///
/// // Classic tea-tasting lady example
/// let table = [[3.0, 1.0], [1.0, 3.0]];
/// let result = fisher_exact_test(&table, FisherAlternative::TwoSided, Some(0.95)).unwrap();
/// println!("{}", result);
/// ```
pub fn fisher_exact_test(
    table: &[[f64; 2]; 2],
    alternative: FisherAlternative,
    conf_level: Option<f64>,
) -> EconResult<FisherExactResult> {
    let a = table[0][0];
    let b = table[0][1];
    let c = table[1][0];
    let d = table[1][1];

    // Validate inputs
    if a < 0.0 || b < 0.0 || c < 0.0 || d < 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "All table entries must be non-negative".to_string(),
        });
    }

    // Calculate marginals
    let row1 = a + b;
    let row2 = c + d;
    let col1 = a + c;
    let col2 = b + d;
    let n = a + b + c + d;

    if n == 0.0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Table total must be positive".to_string(),
        });
    }

    // Calculate sample odds ratio
    let odds_ratio = if b * c > 0.0 {
        (a * d) / (b * c)
    } else if a * d > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    };

    // Set up hypergeometric distribution
    // Hypergeometric(N = n, K = col1, n_draws = row1)
    // X ~ number of successes (col1 items) in row1 draws from population of n
    let n_u64 = n as u64;
    let col1_u64 = col1 as u64;
    let row1_u64 = row1 as u64;

    let hyper = Hypergeometric::new(n_u64, col1_u64, row1_u64).map_err(|e| {
        EconError::InvalidSpecification {
            message: format!("Failed to create hypergeometric distribution: {}", e),
        }
    })?;

    let a_u64 = a as u64;

    // Probability of observed table
    let prob_observed = hyper.pmf(a_u64);

    // Calculate p-value based on alternative hypothesis
    let p_value = match alternative {
        FisherAlternative::Greater => {
            // P(X >= a) = sf(a - 1) or 1 - cdf(a - 1)
            if a_u64 == 0 { 1.0 } else { hyper.sf(a_u64 - 1) }
        }
        FisherAlternative::Less => {
            // P(X <= a) = cdf(a)
            hyper.cdf(a_u64)
        }
        FisherAlternative::TwoSided => {
            // Sum probabilities of all tables as extreme or more extreme
            // (tables with probability <= prob_observed)
            compute_two_sided_pvalue(&hyper, a_u64, prob_observed, row1_u64, col1_u64, n_u64)
        }
    };

    // Compute confidence interval for odds ratio if requested
    let odds_ratio_ci = if let Some(level) = conf_level {
        if level <= 0.0 || level >= 1.0 {
            return Err(EconError::InvalidSpecification {
                message: "Confidence level must be between 0 and 1".to_string(),
            });
        }
        // Use the Cornfield method for exact CI
        Some(compute_odds_ratio_ci(
            a_u64, b as u64, c as u64, d as u64, level,
        ))
    } else {
        None
    };

    let test_name = format!(
        "Fisher's Exact Test for Count Data (alternative: {})",
        alternative
    );

    Ok(FisherExactResult {
        test_name,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        alternative,
        odds_ratio,
        odds_ratio_ci,
        conf_level,
        table: [a, b, c, d],
        row_totals: [row1, row2],
        col_totals: [col1, col2],
        n,
        prob_observed,
    })
}

/// Compute two-sided p-value for Fisher's exact test.
///
/// Computes all hypergeometric PMF values once into a Vec, then sums those
/// with probability <= observed probability. Includes early termination
/// when remaining probability mass is negligible (< 1e-15).
fn compute_two_sided_pvalue(
    hyper: &Hypergeometric,
    _observed: u64,
    prob_observed: f64,
    row1: u64,
    col1: u64,
    n: u64,
) -> f64 {
    // The support of X is [max(0, row1 + col1 - n), min(row1, col1)]
    let x_min = (row1 + col1).saturating_sub(n);
    let x_max = row1.min(col1);
    let support_size = (x_max - x_min + 1) as usize;

    // Optimization 1: Compute all PMF values once into a Vec
    let mut pmf_values: Vec<f64> = Vec::with_capacity(support_size);
    for x in x_min..=x_max {
        pmf_values.push(hyper.pmf(x));
    }

    // Optimization 2: Sort indices by descending PMF to enable early termination.
    // We accumulate p-value by summing qualifying PMFs, and track remaining mass.
    let threshold = prob_observed + 1e-10;
    let mut p_value = 0.0;
    let mut remaining_mass = 1.0;

    // Iterate through PMF values. We process from largest to smallest so that
    // once the remaining mass is negligible, we can stop early.
    // Since most distributions are unimodal, a simpler approach works well:
    // just scan all values and sum those that qualify, with early termination
    // when the cumulative remaining mass drops below the threshold.
    for &prob_x in &pmf_values {
        if prob_x <= threshold {
            p_value += prob_x;
        }
        remaining_mass -= prob_x;
        // Early termination: if all remaining mass is negligible, stop
        if remaining_mass < 1e-15 {
            break;
        }
    }

    // Clamp to [0, 1] to handle numerical issues
    p_value.min(1.0).max(0.0)
}

/// Pre-computed log-binomial coefficients for the non-central hypergeometric
/// distribution used in CI computation. Avoids recomputing ln_gamma calls
/// across binary search iterations.
struct PrecomputedLogBinom {
    /// log(C(col1, x)) for x in x_min..=x_max
    ln_choose_col1: Vec<f64>,
    /// log(C(col2, row1 - x)) for x in x_min..=x_max
    ln_choose_col2: Vec<f64>,
    /// Validity flags: true if the table entry is valid for this x
    valid: Vec<bool>,
    /// Minimum x in support
    x_min: u64,
    /// Maximum x in support
    x_max: u64,
}

impl PrecomputedLogBinom {
    /// Pre-compute all log-binomial coefficients for the support range.
    fn new(row1: u64, col1: u64, n: u64) -> Self {
        let x_min = (row1 + col1).saturating_sub(n);
        let x_max = row1.min(col1);
        let col2 = n - col1;
        let support_size = (x_max - x_min + 1) as usize;

        let mut ln_choose_col1 = Vec::with_capacity(support_size);
        let mut ln_choose_col2 = Vec::with_capacity(support_size);
        let mut valid = Vec::with_capacity(support_size);

        for x in x_min..=x_max {
            // Check validity: need x <= col1, row1 >= x, row1 - x <= col2
            if x > col1 || row1 < x || row1 - x > col2 {
                ln_choose_col1.push(0.0);
                ln_choose_col2.push(0.0);
                valid.push(false);
            } else {
                ln_choose_col1.push(log_binomial(col1, x));
                ln_choose_col2.push(log_binomial(col2, row1 - x));
                valid.push(true);
            }
        }

        Self {
            ln_choose_col1,
            ln_choose_col2,
            valid,
            x_min,
            x_max,
        }
    }
}

/// Compute exact confidence interval for odds ratio using iterative search.
///
/// Uses the Cornfield method: find odds ratios where Fisher's test
/// would give p-value = alpha/2 for one-sided alternatives.
///
/// Pre-computes log-binomial coefficients once and reuses them across
/// all binary search iterations for efficiency.
fn compute_odds_ratio_ci(a: u64, b: u64, c: u64, d: u64, conf_level: f64) -> (f64, f64) {
    let alpha = 1.0 - conf_level;

    // Handle edge cases
    if b == 0 || c == 0 {
        // Odds ratio is infinite - lower bound only
        let lower = if a == 0 || d == 0 { 0.0 } else { 0.0 };
        return (lower, f64::INFINITY);
    }
    if a == 0 || d == 0 {
        // Odds ratio is 0 - upper bound only
        let n = a + b + c + d;
        let row1 = a + b;
        let col1 = a + c;
        let precomp = PrecomputedLogBinom::new(row1, col1, n);
        return (
            0.0,
            compute_ci_upper_fast(a, b, c, d, alpha / 2.0, &precomp),
        );
    }

    let n = a + b + c + d;
    let row1 = a + b;
    let col1 = a + c;

    // Pre-compute log-binomial coefficients once for both lower and upper bounds
    let precomp = PrecomputedLogBinom::new(row1, col1, n);

    // Binary search for lower bound
    let lower = compute_ci_lower_fast(a, b, c, d, alpha / 2.0, &precomp);

    // Binary search for upper bound
    let upper = compute_ci_upper_fast(a, b, c, d, alpha / 2.0, &precomp);

    (lower, upper)
}

/// Find lower CI bound using binary search with pre-computed log-binomial coefficients.
fn compute_ci_lower_fast(
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    alpha: f64,
    precomp: &PrecomputedLogBinom,
) -> f64 {
    let n = a + b + c + d;
    let row1 = a + b;
    let col1 = a + c;

    // Lower bound search between 0 and sample OR
    let sample_or = (a as f64 * d as f64) / (b as f64 * c as f64);
    let mut lo = 0.0;
    let mut hi = sample_or.max(0.001);

    // Expand upper search bound if needed
    while compute_fisher_pvalue_given_or_fast(a, row1, col1, n, hi, FisherAlternative::Less, precomp) < alpha {
        hi *= 2.0;
        if hi > 1e6 {
            return 0.0;
        }
    }

    // Binary search
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        let p = compute_fisher_pvalue_given_or_fast(a, row1, col1, n, mid, FisherAlternative::Less, precomp);

        if p < alpha {
            lo = mid;
        } else {
            hi = mid;
        }

        if (hi - lo) < 1e-8 * lo.max(1e-10) {
            break;
        }
    }

    lo
}

/// Find upper CI bound using binary search with pre-computed log-binomial coefficients.
fn compute_ci_upper_fast(
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    alpha: f64,
    precomp: &PrecomputedLogBinom,
) -> f64 {
    let n = a + b + c + d;
    let row1 = a + b;
    let col1 = a + c;

    // Upper bound search starting from sample OR
    let sample_or = if b * c > 0 {
        (a as f64 * d as f64) / (b as f64 * c as f64)
    } else {
        1.0
    };
    let mut lo = sample_or.max(0.001);
    let mut hi = (sample_or * 10.0).max(10.0);

    // Expand upper search bound if needed
    while compute_fisher_pvalue_given_or_fast(a, row1, col1, n, hi, FisherAlternative::Greater, precomp) < alpha {
        hi *= 2.0;
        if hi > 1e10 {
            return f64::INFINITY;
        }
    }

    // Binary search
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        let p = compute_fisher_pvalue_given_or_fast(a, row1, col1, n, mid, FisherAlternative::Greater, precomp);

        if p < alpha {
            hi = mid;
        } else {
            lo = mid;
        }

        if (hi - lo) < 1e-8 * hi.max(1e-10) {
            break;
        }
    }

    hi
}

/// Compute Fisher's p-value under a specific null odds ratio, using pre-computed
/// log-binomial coefficients for efficiency.
///
/// This is the hot path for CI computation. By accepting pre-computed log-binomial
/// coefficients, we avoid redundant ln_gamma calls across binary search iterations.
/// Only the `x * ln(or)` term changes between iterations.
fn compute_fisher_pvalue_given_or_fast(
    observed: u64,
    row1: u64,
    col1: u64,
    _n: u64,
    or: f64,
    alternative: FisherAlternative,
    precomp: &PrecomputedLogBinom,
) -> f64 {
    let x_min = precomp.x_min;
    let x_max = precomp.x_max;
    let support_size = (x_max - x_min + 1) as usize;

    // Compute log-probabilities using pre-computed binomial coefficients
    // Only the OR^x term varies across binary search iterations
    let ln_or = or.ln();

    // First pass: compute log-probs and find maximum for numerical stability
    let mut log_probs: Vec<f64> = Vec::with_capacity(support_size);
    let mut max_log = f64::NEG_INFINITY;

    for (i, x) in (x_min..=x_max).enumerate() {
        if !precomp.valid[i] {
            log_probs.push(f64::NEG_INFINITY);
            continue;
        }
        // log P(X=x|OR) = log(C(col1, x)) + log(C(col2, row1-x)) + x*log(OR)
        let log_p =
            precomp.ln_choose_col1[i] + precomp.ln_choose_col2[i] + (x as f64) * ln_or;
        if log_p > max_log {
            max_log = log_p;
        }
        log_probs.push(log_p);
    }

    if max_log.is_infinite() && max_log < 0.0 {
        return 1.0; // All probabilities are 0
    }

    // Second pass: convert to normalized probabilities in a single allocation
    let mut probs: Vec<f64> = Vec::with_capacity(support_size);
    let mut total = 0.0;
    for &log_p in &log_probs {
        let p = (log_p - max_log).exp();
        total += p;
        probs.push(p);
    }

    if total == 0.0 {
        return 1.0;
    }

    let inv_total = 1.0 / total;
    for p in &mut probs {
        *p *= inv_total;
    }

    // Compute p-value
    let obs_idx = (observed - x_min) as usize;
    match alternative {
        FisherAlternative::Less => {
            let mut sum = 0.0;
            for i in 0..=obs_idx {
                sum += probs[i];
            }
            sum
        }
        FisherAlternative::Greater => {
            let mut sum = 0.0;
            for i in obs_idx..probs.len() {
                sum += probs[i];
            }
            sum
        }
        FisherAlternative::TwoSided => {
            // Sum probs of all values as extreme
            let obs_prob = probs[obs_idx];
            let threshold = obs_prob + 1e-10;
            let mut sum = 0.0;
            for &p in &probs {
                if p <= threshold {
                    sum += p;
                }
            }
            sum
        }
    }
}

/// Compute log of binomial coefficient C(n, k) = n! / (k! (n-k)!)
fn log_binomial(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    if k == 0 || k == n {
        return 0.0;
    }

    // Use log-gamma: log(C(n,k)) = log(n!) - log(k!) - log((n-k)!)
    use statrs::function::gamma::ln_gamma;

    let n_f = (n + 1) as f64;
    let k_f = (k + 1) as f64;
    let nk_f = (n - k + 1) as f64;

    ln_gamma(n_f) - ln_gamma(k_f) - ln_gamma(nk_f)
}

/// Fisher's exact test from integer table.
///
/// Convenience wrapper for integer inputs.
///
/// # Example
///
/// ```
/// use p2a_core::stats::fisher::{fisher_exact_test_int, FisherAlternative};
///
/// let table = [[3, 1], [1, 3]];
/// let result = fisher_exact_test_int(&table, FisherAlternative::TwoSided, Some(0.95)).unwrap();
/// assert!(result.p_value > 0.2);  // Not significant
/// ```
pub fn fisher_exact_test_int(
    table: &[[u64; 2]; 2],
    alternative: FisherAlternative,
    conf_level: Option<f64>,
) -> EconResult<FisherExactResult> {
    let table_f64 = [
        [table[0][0] as f64, table[0][1] as f64],
        [table[1][0] as f64, table[1][1] as f64],
    ];
    fisher_exact_test(&table_f64, alternative, conf_level)
}

/// Run Fisher's exact test on two categorical columns from a dataset.
///
/// Creates a 2×2 contingency table from two binary categorical columns.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `row_col` - Column name for rows (must have exactly 2 unique values)
/// * `col_col` - Column name for columns (must have exactly 2 unique values)
/// * `alternative` - Alternative hypothesis
/// * `conf_level` - Confidence level for odds ratio CI
///
/// # Example
///
/// ```ignore
/// let result = run_fisher_test(&dataset, "treatment", "outcome",
///     FisherAlternative::TwoSided, Some(0.95))?;
/// ```
pub fn run_fisher_test(
    dataset: &Dataset,
    row_col: &str,
    col_col: &str,
    alternative: FisherAlternative,
    conf_level: Option<f64>,
) -> EconResult<FisherExactResult> {
    use polars::prelude::*;

    let df = dataset.df();

    let available_cols: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Validate columns exist
    let row_series = df.column(row_col).map_err(|_| EconError::ColumnNotFound {
        column: row_col.to_string(),
        available: available_cols.clone(),
    })?;

    let col_series = df.column(col_col).map_err(|_| EconError::ColumnNotFound {
        column: col_col.to_string(),
        available: available_cols,
    })?;

    // Get unique values
    let row_unique = row_series
        .unique()
        .map_err(|e| EconError::InvalidSpecification {
            message: format!("Failed to get unique row values: {}", e),
        })?;

    let col_unique = col_series
        .unique()
        .map_err(|e| EconError::InvalidSpecification {
            message: format!("Failed to get unique column values: {}", e),
        })?;

    if row_unique.len() != 2 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Fisher's exact test requires exactly 2 unique values in row column, got {}",
                row_unique.len()
            ),
        });
    }

    if col_unique.len() != 2 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Fisher's exact test requires exactly 2 unique values in column column, got {}",
                col_unique.len()
            ),
        });
    }

    // Build 2×2 contingency table using group_by
    let grouped = df
        .clone()
        .lazy()
        .group_by([col(row_col), col(col_col)])
        .agg([len().alias("count")])
        .collect()
        .map_err(|e| EconError::InvalidSpecification {
            message: format!("Failed to group data: {}", e),
        })?;

    // Get sorted unique values for consistent ordering
    let mut row_values: Vec<String> = Vec::new();
    let mut col_values: Vec<String> = Vec::new();

    for i in 0..row_unique.len() {
        row_values.push(format!("{:?}", row_unique.get(i).unwrap()));
    }
    for i in 0..col_unique.len() {
        col_values.push(format!("{:?}", col_unique.get(i).unwrap()));
    }

    row_values.sort();
    col_values.sort();

    // Initialize 2×2 table
    let mut table = [[0.0; 2]; 2];

    // Fill in counts
    let row_data = grouped.column(row_col).unwrap();
    let col_data = grouped.column(col_col).unwrap();
    let count_data = grouped.column("count").unwrap();

    for i in 0..grouped.height() {
        let row_val = format!("{:?}", row_data.get(i).unwrap());
        let col_val = format!("{:?}", col_data.get(i).unwrap());
        let count = count_data.get(i).unwrap().try_extract::<u64>().unwrap_or(0) as f64;

        if let (Some(row_idx), Some(col_idx)) = (
            row_values.iter().position(|v| v == &row_val),
            col_values.iter().position(|v| v == &col_val),
        ) {
            table[row_idx][col_idx] = count;
        }
    }

    fisher_exact_test(&table, alternative, conf_level)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_fisher_exact() {
        // Classic lady tasting tea example
        let table = [[3.0, 1.0], [1.0, 3.0]];
        let result = fisher_exact_test(&table, FisherAlternative::TwoSided, Some(0.95)).unwrap();

        assert!(result.p_value > 0.0);
        assert!(result.p_value <= 1.0);
        assert_eq!(result.n, 8.0);
        assert_eq!(result.odds_ratio, 9.0); // (3*3)/(1*1) = 9
    }

    #[test]
    fn test_fisher_integer_wrapper() {
        let table = [[3, 1], [1, 3]];
        let result =
            fisher_exact_test_int(&table, FisherAlternative::TwoSided, Some(0.95)).unwrap();

        assert!(result.p_value > 0.0);
        assert_eq!(result.odds_ratio, 9.0);
    }

    #[test]
    fn test_one_sided_alternatives() {
        let table = [[10.0, 2.0], [3.0, 15.0]];

        let two_sided = fisher_exact_test(&table, FisherAlternative::TwoSided, None).unwrap();
        let greater = fisher_exact_test(&table, FisherAlternative::Greater, None).unwrap();
        let less = fisher_exact_test(&table, FisherAlternative::Less, None).unwrap();

        // Two-sided should be >= one-sided
        assert!(two_sided.p_value >= greater.p_value - 1e-10);
        // greater tests for odds ratio > 1 (positive association)
        assert!(greater.p_value < 0.05); // Should be significant
        // less tests for odds ratio < 1
        assert!(less.p_value > 0.5); // Should not be significant in this direction
    }

    #[test]
    fn test_zero_cells() {
        // Table with zero cell
        let table = [[0.0, 5.0], [5.0, 10.0]];
        let result = fisher_exact_test(&table, FisherAlternative::TwoSided, None).unwrap();

        assert!(result.p_value > 0.0);
        assert_eq!(result.odds_ratio, 0.0);
    }

    #[test]
    fn test_extreme_table() {
        // Very unbalanced table
        let table = [[50.0, 1.0], [1.0, 50.0]];
        let result = fisher_exact_test(&table, FisherAlternative::TwoSided, Some(0.95)).unwrap();

        assert!(result.p_value < 0.001); // Should be highly significant
        assert!(result.odds_ratio > 100.0); // Very large odds ratio
    }

    #[test]
    fn test_validate_against_r_basic() {
        // R code:
        // M <- matrix(c(1, 9, 11, 3), nrow = 2, byrow = TRUE)
        // # Note: R reads by column, so this is actually:
        // # [[1, 11], [9, 3]]
        // # We need to specify correctly:
        // M <- matrix(c(1, 11, 9, 3), nrow = 2, byrow = TRUE)
        // #      [,1] [,2]
        // # [1,]    1   11
        // # [2,]    9    3
        // fisher.test(M)
        //
        // p-value = 0.002759
        // odds ratio = 0.03717
        let table = [[1.0, 11.0], [9.0, 3.0]];
        let result = fisher_exact_test(&table, FisherAlternative::TwoSided, Some(0.95)).unwrap();

        // R gives p-value = 0.002759
        assert!(
            (result.p_value - 0.002759).abs() < 0.001,
            "Expected p-value ~0.002759, got {}",
            result.p_value
        );

        // Sample odds ratio = (1*3)/(11*9) = 3/99 ≈ 0.0303
        let expected_or = (1.0 * 3.0) / (11.0 * 9.0);
        assert!(
            (result.odds_ratio - expected_or).abs() < 0.001,
            "Expected OR ~{}, got {}",
            expected_or,
            result.odds_ratio
        );
    }

    #[test]
    fn test_validate_against_r_lady_tea() {
        // R code:
        // M <- matrix(c(3, 1, 1, 3), nrow = 2, byrow = TRUE)
        // fisher.test(M)
        //
        // p-value = 0.4857
        let table = [[3.0, 1.0], [1.0, 3.0]];
        let result = fisher_exact_test(&table, FisherAlternative::TwoSided, Some(0.95)).unwrap();

        // R gives p-value = 0.4857
        assert!(
            (result.p_value - 0.4857).abs() < 0.01,
            "Expected p-value ~0.4857, got {}",
            result.p_value
        );
    }

    #[test]
    fn test_validate_against_r_one_sided() {
        // R code:
        // M <- matrix(c(6, 2, 1, 7), nrow = 2, byrow = TRUE)
        // fisher.test(M, alternative = "greater")
        //
        // greater p-value: 0.0202797203
        // two.sided p-value: 0.0405594406
        // less p-value: 0.9993006993
        let table = [[6.0, 2.0], [1.0, 7.0]];

        // Test greater alternative
        let result_greater = fisher_exact_test(&table, FisherAlternative::Greater, None).unwrap();
        assert!(
            (result_greater.p_value - 0.02028).abs() < 0.005,
            "Expected greater p-value ~0.02028, got {}",
            result_greater.p_value
        );

        // Test less alternative
        let result_less = fisher_exact_test(&table, FisherAlternative::Less, None).unwrap();
        assert!(
            (result_less.p_value - 0.9993).abs() < 0.01,
            "Expected less p-value ~0.9993, got {}",
            result_less.p_value
        );

        // Test two-sided alternative
        let result_two = fisher_exact_test(&table, FisherAlternative::TwoSided, None).unwrap();
        assert!(
            (result_two.p_value - 0.04056).abs() < 0.005,
            "Expected two-sided p-value ~0.04056, got {}",
            result_two.p_value
        );
    }

    #[test]
    fn test_confidence_interval() {
        let table = [[10.0, 5.0], [5.0, 10.0]];
        let result = fisher_exact_test(&table, FisherAlternative::TwoSided, Some(0.95)).unwrap();

        // Check that CI is computed
        assert!(result.odds_ratio_ci.is_some());
        let (lo, hi) = result.odds_ratio_ci.unwrap();

        // CI should contain the sample OR
        assert!(lo < result.odds_ratio);
        assert!(hi > result.odds_ratio);

        // 95% CI should contain 1 when p > 0.05
        if result.p_value > 0.05 {
            assert!(
                lo < 1.0 && hi > 1.0,
                "95% CI should contain 1 when p > 0.05"
            );
        }
    }

    #[test]
    fn test_error_handling() {
        // Negative values
        let result = fisher_exact_test(
            &[[-1.0, 5.0], [5.0, 5.0]],
            FisherAlternative::TwoSided,
            None,
        );
        assert!(result.is_err());

        // Empty table
        let result =
            fisher_exact_test(&[[0.0, 0.0], [0.0, 0.0]], FisherAlternative::TwoSided, None);
        assert!(result.is_err());

        // Invalid confidence level
        let result = fisher_exact_test(
            &[[1.0, 1.0], [1.0, 1.0]],
            FisherAlternative::TwoSided,
            Some(1.5),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_display() {
        let table = [[3.0, 1.0], [1.0, 3.0]];
        let result = fisher_exact_test(&table, FisherAlternative::TwoSided, Some(0.95)).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Fisher's Exact Test"));
        assert!(display.contains("p-value"));
        assert!(display.contains("odds ratio"));
    }
}
