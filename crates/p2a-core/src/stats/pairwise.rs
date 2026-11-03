//! Pairwise comparison tests with p-value adjustment for multiple comparisons.
//!
//! Performs pairwise comparisons between group levels using t-tests or
//! Wilcoxon rank sum tests with corrections for multiple testing. These are
//! commonly used as post-hoc analyses after ANOVA/Kruskal-Wallis to identify
//! which specific groups differ.
//!
//! # Available Tests
//!
//! - **Pairwise t-test** (`pairwise_t_test`): Parametric comparisons
//! - **Pairwise Wilcoxon** (`pairwise_wilcox_test`): Non-parametric comparisons
//!
//! # References
//!
//! - R Core Team. `stats::pairwise.t.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/pairwise.t.test.html>
//! - R Core Team. `stats::pairwise.wilcox.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/pairwise.wilcox.test.html>
//! - Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure".
//!   *Scandinavian Journal of Statistics*, 6(2), 65-70.
//! - Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate".
//!   *Journal of the Royal Statistical Society Series B*, 57(1), 289-300.
//! - Benjamini, Y. & Yekutieli, D. (2001). "The control of the false discovery
//!   rate in multiple testing under dependency".
//!   *Annals of Statistics*, 29(4), 1165-1188.
//! - Hochberg, Y. (1988). "A Sharper Bonferroni Procedure for Multiple Tests
//!   of Significance". *Biometrika*, 75(4), 800-802.
//! - Hommel, G. (1988). "A Stagewise Rejective Multiple Test Procedure Based
//!   on a Modified Bonferroni Test". *Biometrika*, 75(2), 383-386.
//! - Mann, H. B. & Whitney, D. R. (1947). "On a Test of Whether one of Two
//!   Random Variables is Stochastically Larger than the Other".
//!   *Annals of Mathematical Statistics*, 18(1), 50-60.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::ttest::{Alternative, two_sample_t_test};
use super::wilcoxon::{WilcoxonConfig, wilcoxon_rank_sum};
use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, Columns, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::traits::SignificanceLevel;

/// Method for adjusting p-values in multiple comparisons.
///
/// # Family-Wise Error Rate (FWER) Methods
/// These control the probability of making any Type I error:
/// - Bonferroni, Holm, Hochberg, Hommel
///
/// # False Discovery Rate (FDR) Methods
/// These control the expected proportion of false discoveries:
/// - BH (Benjamini-Hochberg), BY (Benjamini-Yekutieli)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PValueAdjustMethod {
    /// Bonferroni correction: p_adj = min(1, m × p)
    /// Most conservative. Controls FWER.
    Bonferroni,

    /// Holm's step-down procedure (1979).
    /// Less conservative than Bonferroni, still controls FWER.
    #[default]
    Holm,

    /// Hochberg's step-up procedure (1988).
    /// More powerful than Holm, valid for independent or PRDS tests.
    Hochberg,

    /// Hommel's procedure (1988).
    /// Most powerful FWER method, valid under same conditions as Hochberg.
    Hommel,

    /// Benjamini-Hochberg procedure (1995).
    /// Controls FDR. More powerful than FWER methods.
    /// Also known as "fdr".
    BH,

    /// Benjamini-Yekutieli procedure (2001).
    /// Controls FDR under arbitrary dependency.
    BY,

    /// No adjustment - raw p-values.
    None,
}

impl PValueAdjustMethod {
    /// Parse from string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "bonferroni" | "bonf" => Some(Self::Bonferroni),
            "holm" => Some(Self::Holm),
            "hochberg" => Some(Self::Hochberg),
            "hommel" => Some(Self::Hommel),
            "bh" | "fdr" | "benjamini-hochberg" => Some(Self::BH),
            "by" | "benjamini-yekutieli" => Some(Self::BY),
            "none" => Some(Self::None),
            _ => None,
        }
    }

    /// Get the method name as displayed in R.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Bonferroni => "Bonferroni",
            Self::Holm => "Holm",
            Self::Hochberg => "Hochberg",
            Self::Hommel => "Hommel",
            Self::BH => "BH",
            Self::BY => "BY",
            Self::None => "none",
        }
    }
}

crate::option_enum!(
    PValueAdjustMethod,
    crate::options::Choices {
        option: "p_adjust_method",
        values: &[
            (&["holm"], PValueAdjustMethod::Holm),
            (&["hochberg"], PValueAdjustMethod::Hochberg),
            (&["hommel"], PValueAdjustMethod::Hommel),
            (&["bonferroni", "bonf"], PValueAdjustMethod::Bonferroni),
            (&["BH", "fdr", "benjamini-hochberg"], PValueAdjustMethod::BH),
            (&["BY", "benjamini-yekutieli"], PValueAdjustMethod::BY),
            (&["none"], PValueAdjustMethod::None),
        ],
    }
);

/// Result of a pairwise t-test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseTTestResult {
    /// Test description
    pub test_name: String,
    /// P-value adjustment method used
    pub p_adjust_method: PValueAdjustMethod,
    /// Whether pooled SD was used
    pub pool_sd: bool,
    /// Alternative hypothesis type
    pub alternative: Alternative,
    /// Group names in order
    pub group_names: Vec<String>,
    /// Raw p-values (lower triangular matrix stored as vector)
    /// Access via p_value(i, j) method
    pub p_values_raw: Vec<f64>,
    /// Adjusted p-values (lower triangular matrix stored as vector)
    pub p_values_adj: Vec<f64>,
    /// T-statistics for each comparison (lower triangular)
    pub t_statistics: Vec<f64>,
    /// Degrees of freedom for each comparison (lower triangular)
    pub df: Vec<f64>,
    /// Sample sizes per group
    pub group_sizes: Vec<usize>,
    /// Group means
    pub group_means: Vec<f64>,
    /// Pooled standard deviation (if pool_sd = true)
    pub pooled_sd: Option<f64>,
    /// Total number of comparisons
    pub n_comparisons: usize,
    /// Rows dropped because of a missing response or a missing label (dataset wrapper;
    /// nothing for slices).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl PairwiseTTestResult {
    /// Get the index in the lower triangular storage for pair (i, j) where i > j.
    pub fn index(&self, i: usize, j: usize) -> usize {
        // Lower triangular index: sum(1..i) + j = i*(i-1)/2 + j
        debug_assert!(i > j, "i must be greater than j for lower triangular");
        i * (i - 1) / 2 + j
    }

    /// Get the raw p-value for comparing groups i and j.
    pub fn p_value_raw(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return f64::NAN;
        }
        let (i, j) = if i > j { (i, j) } else { (j, i) };
        self.p_values_raw[self.index(i, j)]
    }

    /// Get the adjusted p-value for comparing groups i and j.
    pub fn p_value_adj(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return f64::NAN;
        }
        let (i, j) = if i > j { (i, j) } else { (j, i) };
        self.p_values_adj[self.index(i, j)]
    }

    /// Get the t-statistic for comparing groups i and j.
    pub fn t_statistic(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return f64::NAN;
        }
        let (i, j) = if i > j { (i, j) } else { (j, i) };
        self.t_statistics[self.index(i, j)]
    }

    /// Get the degrees of freedom for comparing groups i and j.
    pub fn degrees_of_freedom(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return f64::NAN;
        }
        let (i, j) = if i > j { (i, j) } else { (j, i) };
        self.df[self.index(i, j)]
    }

    /// Get a formatted p-value matrix (for display).
    pub fn p_value_matrix(&self) -> Vec<Vec<Option<f64>>> {
        let k = self.group_names.len();
        let mut matrix = vec![vec![None; k]; k];
        for i in 1..k {
            for j in 0..i {
                matrix[i][j] = Some(self.p_values_adj[self.index(i, j)]);
            }
        }
        matrix
    }
}

impl std::fmt::Display for PairwiseTTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;

        writeln!(
            f,
            "P value adjustment method: {}",
            self.p_adjust_method.name()
        )?;
        if self.pool_sd {
            if let Some(sd) = self.pooled_sd {
                writeln!(f, "Pooled SD: {:.6}", sd)?;
            }
        }
        writeln!(f)?;

        // Print matrix header
        let k = self.group_names.len();

        // Find max width for group names
        let max_width = self
            .group_names
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(6)
            .max(10);

        // Print column headers (all except last)
        write!(f, "{:width$}", "", width = max_width + 2)?;
        for j in 0..k - 1 {
            write!(f, "{:>width$} ", self.group_names[j], width = max_width)?;
        }
        writeln!(f)?;

        // Print rows
        for i in 1..k {
            write!(f, "{:width$}  ", self.group_names[i], width = max_width)?;
            for j in 0..i {
                let p = self.p_values_adj[self.index(i, j)];
                let sig = SignificanceLevel::from_p_value(p);
                if p < 0.0001 {
                    write!(f, "{:>width$.2e}{} ", p, sig.stars(), width = max_width - 3)?;
                } else {
                    write!(f, "{:>width$.4}{} ", p, sig.stars(), width = max_width - 3)?;
                }
            }
            // Fill remaining columns with blank
            for _ in i..k - 1 {
                write!(f, "{:>width$} ", "-", width = max_width)?;
            }
            writeln!(f)?;
        }
        writeln!(f)?;

        writeln!(f, "---")?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

/// Adjust p-values for multiple comparisons, as R's `p.adjust(p, method)`.
///
/// NaN p-values (R's `NA`) are kept in place and left out of the number of tests `n`. With
/// `n <= 1` the input is returned unchanged, and Hommel's method with `n = 2` is Hochberg's (R).
/// With `o` the order of the p-values (stable, by `f64::total_cmp`):
/// - Bonferroni: `min(1, n p)`;
/// - Holm: `min(1, cummax((n + 1 - i) p[o]))`, ascending;
/// - Hochberg: `min(1, cummin((n + 1 - i) p[o]))`, descending, i = n..1;
/// - Hommel: R's algorithm (Wright 1992) on the ascending p-values;
/// - BH: `min(1, cummin(n / i p[o]))`, descending;
/// - BY: as BH with the factor `sum_{k <= n} 1 / k`.
///
/// # Arguments
/// * `p_values` - Vector of raw p-values
/// * `method` - Adjustment method to use
///
/// # Returns
/// Vector of adjusted p-values in the same order as input
///
/// # References
/// - R `stats::p.adjust()` function
/// - Wright, S. P. (1992). Adjusted p-values for simultaneous inference. Biometrics 48, 1005-1013.
pub fn p_adjust(p_values: &[f64], method: PValueAdjustMethod) -> Vec<f64> {
    let mut out = p_values.to_vec();
    let keep: Vec<usize> = (0..p_values.len())
        .filter(|&i| !p_values[i].is_nan())
        .collect();
    let n = keep.len();
    if n <= 1 {
        return out;
    }
    let p: Vec<f64> = keep.iter().map(|&i| p_values[i]).collect();
    let method = if n == 2 && method == PValueAdjustMethod::Hommel {
        PValueAdjustMethod::Hochberg
    } else {
        method
    };
    let nf = n as f64;
    let ascending = || {
        let mut o: Vec<usize> = (0..n).collect();
        o.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
        o
    };
    let descending = || {
        let mut o: Vec<usize> = (0..n).collect();
        o.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
        o
    };
    // adjusted[k] belongs to p[o[k]]
    let (o, adjusted): (Vec<usize>, Vec<f64>) = match method {
        PValueAdjustMethod::None => return out,
        PValueAdjustMethod::Bonferroni => (
            (0..n).collect(),
            p.iter().map(|&v| (nf * v).min(1.0)).collect(),
        ),
        PValueAdjustMethod::Holm => {
            let o = ascending();
            let mut run = f64::NEG_INFINITY;
            let adj = o
                .iter()
                .enumerate()
                .map(|(k, &j)| {
                    run = run.max((nf - k as f64) * p[j]);
                    run.min(1.0)
                })
                .collect();
            (o, adj)
        }
        PValueAdjustMethod::Hochberg | PValueAdjustMethod::BH | PValueAdjustMethod::BY => {
            let o = descending();
            let q = if method == PValueAdjustMethod::BY {
                (1..=n).map(|k| 1.0 / k as f64).sum::<f64>()
            } else {
                1.0
            };
            let mut run = f64::INFINITY;
            let adj = o
                .iter()
                .enumerate()
                .map(|(k, &j)| {
                    // i runs n..1 along the descending order
                    let i = (n - k) as f64;
                    let v = match method {
                        PValueAdjustMethod::Hochberg => (nf + 1.0 - i) * p[j],
                        _ => q * nf / i * p[j],
                    };
                    run = run.min(v);
                    run.min(1.0)
                })
                .collect();
            (o, adj)
        }
        PValueAdjustMethod::Hommel => {
            let o = ascending();
            let ps: Vec<f64> = o.iter().map(|&j| p[j]).collect();
            let start = ps
                .iter()
                .enumerate()
                .map(|(k, &v)| nf * v / (k + 1) as f64)
                .fold(f64::INFINITY, f64::min);
            let mut q = vec![start; n];
            let mut pa = vec![start; n];
            for m in (2..n).rev() {
                let mf = m as f64;
                // i1 = 0..n-m+1, i2 = n-m+1..n
                let split = n - m + 1;
                let q1 = (split..n)
                    .enumerate()
                    .map(|(k, i)| mf * ps[i] / (k + 2) as f64)
                    .fold(f64::INFINITY, f64::min);
                for i in 0..split {
                    q[i] = (mf * ps[i]).min(q1);
                }
                let fill = q[n - m];
                for qi in q.iter_mut().skip(split) {
                    *qi = fill;
                }
                for (a, &b) in pa.iter_mut().zip(&q) {
                    *a = a.max(b);
                }
            }
            let adj = pa.iter().zip(&ps).map(|(&a, &b)| a.max(b)).collect();
            (o, adj)
        }
    };
    for (k, &j) in o.iter().enumerate() {
        out[keep[j]] = adjusted[k];
    }
    out
}

/// Perform pairwise t-tests between all groups.
///
/// # Arguments
/// * `values` - Vector of all observations
/// * `groups` - Vector of group labels (same length as values)
/// * `pool_sd` - If true, use pooled SD from all groups; if false, use Welch's t-test
/// * `alternative` - Direction of alternative hypothesis
/// * `p_adjust_method` - Method for p-value adjustment
///
/// # Returns
/// `PairwiseTTestResult` containing the p-value matrix and comparison details
///
/// # Example
/// ```ignore
/// let values = vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0];
/// let groups = vec!["A", "A", "A", "B", "B", "B", "C", "C", "C"];
/// let result = pairwise_t_test(&values, &groups, true, Alternative::TwoSided, PValueAdjustMethod::Holm)?;
/// ```
///
/// # References
/// - R equivalent: `pairwise.t.test(x, g, pool.sd = TRUE, p.adjust.method = "holm")`
pub fn pairwise_t_test<T: AsRef<str> + Clone + Eq + std::hash::Hash + ToString>(
    values: &[f64],
    groups: &[T],
    pool_sd: bool,
    alternative: Alternative,
    p_adjust_method: PValueAdjustMethod,
) -> EconResult<PairwiseTTestResult> {
    if values.len() != groups.len() {
        return Err(EconError::InvalidSpecification {
            message: "values and groups must have the same length".to_string(),
        });
    }

    if values.len() < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: values.len(),
            context: "pairwise t-test requires at least 2 observations".to_string(),
        });
    }

    // Group the data
    let mut group_data: HashMap<String, Vec<f64>> = HashMap::new();
    for (v, g) in values.iter().zip(groups.iter()) {
        group_data.entry(g.to_string()).or_default().push(*v);
    }

    // Get sorted group names for consistent ordering
    let mut group_names: Vec<String> = group_data.keys().cloned().collect();
    group_names.sort();

    let k = group_names.len();
    if k < 2 {
        return Err(EconError::InvalidSpecification {
            message: "pairwise t-test requires at least 2 groups".to_string(),
        });
    }

    // Compute group statistics
    let mut group_vecs: Vec<&Vec<f64>> = Vec::with_capacity(k);
    let mut group_sizes: Vec<usize> = Vec::with_capacity(k);
    let mut group_means: Vec<f64> = Vec::with_capacity(k);
    let mut group_vars: Vec<f64> = Vec::with_capacity(k);

    for name in &group_names {
        let data = group_data.get(name).unwrap();
        let n = data.len();
        if n < 2 {
            return Err(EconError::InsufficientData {
                required: 2,
                provided: n,
                context: format!("Group '{}' needs at least 2 observations", name),
            });
        }

        let mean = data.iter().sum::<f64>() / n as f64;
        let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

        group_vecs.push(data);
        group_sizes.push(n);
        group_means.push(mean);
        group_vars.push(var);
    }

    // Compute pooled SD if requested
    let pooled_sd = if pool_sd {
        // MSE from one-way ANOVA: pooled variance
        let total_df: usize = group_sizes.iter().map(|n| n - 1).sum();
        let ss_within: f64 = group_vars
            .iter()
            .zip(&group_sizes)
            .map(|(&var, &n)| var * (n - 1) as f64)
            .sum();
        let mse = ss_within / total_df as f64;
        Some(mse.sqrt())
    } else {
        None
    };

    // Number of comparisons: k*(k-1)/2
    let n_comparisons = comparison_count(k)?;
    let mut p_values_raw = resource::try_with_capacity(n_comparisons, "pairwise p-values")?;
    let mut t_statistics = resource::try_with_capacity(n_comparisons, "pairwise t statistics")?;
    let mut df_values = resource::try_with_capacity(n_comparisons, "pairwise degrees of freedom")?;

    // Perform pairwise t-tests (lower triangular)
    for i in 1..k {
        resource::checkpoint()?;
        for j in 0..i {
            let x = group_vecs[i];
            let y = group_vecs[j];

            let (t_stat, df, p_value) = if pool_sd {
                // Use pooled SD
                let sd = pooled_sd.unwrap();
                let n1 = group_sizes[i] as f64;
                let n2 = group_sizes[j] as f64;
                let se = sd * (1.0 / n1 + 1.0 / n2).sqrt();
                let t = (group_means[i] - group_means[j]) / se;
                let df = group_sizes.iter().map(|n| n - 1).sum::<usize>() as f64;
                let p = compute_p_value_t(t, df, alternative);
                (t, df, p)
            } else {
                // Use Welch's t-test
                // t and the Welch df from the two-sample test; the p-value from the
                // upper-tail helper, so small tails keep their relative precision (X-17)
                let result = two_sample_t_test(x, y, 0.0, alternative, false, 0.95)?;
                let p = compute_p_value_t(result.t_statistic, result.df, alternative);
                (result.t_statistic, result.df, p)
            };

            t_statistics.push(t_stat);
            df_values.push(df);
            p_values_raw.push(p_value);
        }
    }

    // Adjust p-values
    let p_values_adj = p_adjust(&p_values_raw, p_adjust_method);

    Ok(PairwiseTTestResult {
        test_name: "Pairwise comparisons using t tests".to_string(),
        p_adjust_method,
        pool_sd,
        alternative,
        group_names,
        p_values_raw,
        p_values_adj,
        t_statistics,
        df: df_values,
        group_sizes,
        group_means,
        pooled_sd,
        n_comparisons,
        missing: MissingSummary::none(values.len()),
    })
}

/// Perform pairwise t-tests using dataset columns.
///
/// The response and the factor are read in one request: a row with a missing response (null
/// or NaN) or a missing label is dropped from both, as R does, and the drop is reported in
/// `missing`. Integer, decimal and boolean responses are accepted and labels come from the
/// column as stored (a text label "NA" is a level like any other; the CSV loader reads the
/// token NA as missing). A text response is refused (`EconError::NonNumericColumn`), as are
/// infinite values (`EconError::NonFiniteValues`). The core function orders the groups by
/// label text.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `response` - Name of the response variable column
/// * `factor` - Name of the grouping factor column
/// * `pool_sd` - If true, use pooled SD; if false, use Welch's t-test
/// * `alternative` - Direction of alternative hypothesis
/// * `p_adjust_method` - Method for p-value adjustment
///
/// # Example
/// ```ignore
/// let result = run_pairwise_t_test(
///     &dataset, "yield", "treatment",
///     true, Alternative::TwoSided, PValueAdjustMethod::Holm
/// )?;
/// println!("{}", result);
/// ```
pub fn run_pairwise_t_test(
    dataset: &Dataset,
    response: &str,
    factor: &str,
    pool_sd: bool,
    alternative: Alternative,
    p_adjust_method: PValueAdjustMethod,
) -> EconResult<PairwiseTTestResult> {
    let cols = response_and_factor(dataset, response, factor)?;
    let labels = labels_of(&cols, factor)?;
    let mut result = pairwise_t_test(
        cols.numeric(response),
        &labels,
        pool_sd,
        alternative,
        p_adjust_method,
    )?;
    result.missing = cols.missing().clone();
    Ok(result)
}

/// The response and the factor of a pairwise test in one request: a row with a missing
/// response or label is dropped from both (`pairwise.t.test` and `pairwise.wilcox.test` do
/// the same). Nothing left is the wrappers' "No valid observations" error.
fn response_and_factor(dataset: &Dataset, response: &str, factor: &str) -> EconResult<Columns> {
    let cols = ColumnRequest::new(dataset.df())
        .numeric(response)
        .factor(factor)
        .na(NaPolicy::Omit)
        .min_rows(0, "pairwise test")
        .extract()?;
    if cols.n() == 0 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: 0,
            context: "No valid observations after removing missing values".to_string(),
        });
    }
    Ok(cols)
}

/// The label of each kept row, as stored in the factor column.
fn labels_of<'a>(cols: &'a Columns, factor: &str) -> EconResult<Vec<&'a str>> {
    let f = cols.factor(factor);
    let mut labels = resource::try_with_capacity(cols.n(), "group labels")?;
    labels.extend((0..cols.n()).map(|i| f.label(i)));
    Ok(labels)
}

/// `k(k-1)/2`, the number of pairwise comparisons among `k` groups (checked).
fn comparison_count(k: usize) -> EconResult<usize> {
    k.checked_mul(k.saturating_sub(1))
        .map(|v| v / 2)
        .ok_or_else(|| {
            EconError::ResourceLimit(format!(
                "pairwise comparisons among {k} groups are too many: the count overflows"
            ))
        })
}

/// Compute p-value for t-statistic (the upper-tail helper of `traits::estimator`, so small
/// tails keep their relative precision).
fn compute_p_value_t(t_stat: f64, df: f64, alternative: Alternative) -> f64 {
    use crate::traits::estimator::t_test_p_value;

    if df <= 0.0 || t_stat.is_nan() {
        return f64::NAN;
    }
    let two_sided = t_test_p_value(t_stat, df);
    let upper = |t: f64| {
        if t >= 0.0 {
            two_sided / 2.0
        } else {
            1.0 - two_sided / 2.0
        }
    };
    match alternative {
        Alternative::TwoSided => two_sided,
        Alternative::Greater => upper(t_stat),
        Alternative::Less => upper(-t_stat),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pairwise Wilcoxon Rank Sum Test
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a pairwise Wilcoxon rank sum test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseWilcoxResult {
    /// Test description
    pub test_name: String,
    /// P-value adjustment method used
    pub p_adjust_method: PValueAdjustMethod,
    /// Alternative hypothesis type
    pub alternative: Alternative,
    /// Group names in order
    pub group_names: Vec<String>,
    /// Raw p-values (lower triangular matrix stored as vector)
    pub p_values_raw: Vec<f64>,
    /// Adjusted p-values (lower triangular matrix stored as vector)
    pub p_values_adj: Vec<f64>,
    /// W statistics for each comparison (lower triangular)
    pub w_statistics: Vec<f64>,
    /// Sample sizes per group
    pub group_sizes: Vec<usize>,
    /// Group medians
    pub group_medians: Vec<f64>,
    /// Total number of comparisons
    pub n_comparisons: usize,
    /// Whether exact p-values were used (vs normal approximation)
    pub exact: bool,
    /// Warning message (if any, e.g., ties present)
    pub warning: Option<String>,
    /// Rows dropped because of a missing response or a missing label (dataset wrapper;
    /// nothing for slices).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl PairwiseWilcoxResult {
    /// Get the index in the lower triangular storage for pair (i, j) where i > j.
    pub fn index(&self, i: usize, j: usize) -> usize {
        debug_assert!(i > j, "i must be greater than j for lower triangular");
        i * (i - 1) / 2 + j
    }

    /// Get the raw p-value for comparing groups i and j.
    pub fn p_value_raw(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return f64::NAN;
        }
        let (i, j) = if i > j { (i, j) } else { (j, i) };
        self.p_values_raw[self.index(i, j)]
    }

    /// Get the adjusted p-value for comparing groups i and j.
    pub fn p_value_adj(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return f64::NAN;
        }
        let (i, j) = if i > j { (i, j) } else { (j, i) };
        self.p_values_adj[self.index(i, j)]
    }

    /// Get the W statistic for comparing groups i and j.
    pub fn w_statistic(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return f64::NAN;
        }
        let (i, j) = if i > j { (i, j) } else { (j, i) };
        self.w_statistics[self.index(i, j)]
    }

    /// Get a formatted p-value matrix (for display).
    pub fn p_value_matrix(&self) -> Vec<Vec<Option<f64>>> {
        let k = self.group_names.len();
        let mut matrix = vec![vec![None; k]; k];
        for i in 1..k {
            for j in 0..i {
                matrix[i][j] = Some(self.p_values_adj[self.index(i, j)]);
            }
        }
        matrix
    }
}

impl std::fmt::Display for PairwiseWilcoxResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;

        writeln!(
            f,
            "P value adjustment method: {}",
            self.p_adjust_method.name()
        )?;
        writeln!(f)?;

        // Print matrix header
        let k = self.group_names.len();

        // Find max width for group names
        let max_width = self
            .group_names
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(6)
            .max(10);

        // Print column headers (all except last)
        write!(f, "{:width$}", "", width = max_width + 2)?;
        for j in 0..k - 1 {
            write!(f, "{:>width$} ", self.group_names[j], width = max_width)?;
        }
        writeln!(f)?;

        // Print rows
        for i in 1..k {
            write!(f, "{:width$}  ", self.group_names[i], width = max_width)?;
            for j in 0..i {
                let p = self.p_values_adj[self.index(i, j)];
                let sig = SignificanceLevel::from_p_value(p);
                if p < 0.0001 {
                    write!(f, "{:>width$.2e}{} ", p, sig.stars(), width = max_width - 3)?;
                } else {
                    write!(f, "{:>width$.4}{} ", p, sig.stars(), width = max_width - 3)?;
                }
            }
            // Fill remaining columns with blank
            for _ in i..k - 1 {
                write!(f, "{:>width$} ", "-", width = max_width)?;
            }
            writeln!(f)?;
        }
        writeln!(f)?;

        if let Some(ref warn) = self.warning {
            writeln!(f, "Warning: {}", warn)?;
            writeln!(f)?;
        }

        writeln!(f, "---")?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

/// Perform pairwise Wilcoxon rank sum tests between all groups.
///
/// A non-parametric alternative to pairwise t-tests for comparing group
/// distributions without assuming normality.
///
/// # Arguments
/// * `values` - Vector of all observations
/// * `groups` - Vector of group labels (same length as values)
/// * `alternative` - Direction of alternative hypothesis
/// * `p_adjust_method` - Method for p-value adjustment
/// * `exact` - If Some(true), compute exact p-values; if Some(false), use normal
///             approximation; if None, auto-decide based on sample size
///
/// # Returns
/// `PairwiseWilcoxResult` containing the p-value matrix and comparison details
///
/// # Example
/// ```ignore
/// let values = vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0];
/// let groups = vec!["A", "A", "A", "B", "B", "B", "C", "C", "C"];
/// let result = pairwise_wilcox_test(&values, &groups, Alternative::TwoSided, PValueAdjustMethod::Holm, None)?;
/// ```
///
/// # References
/// - R equivalent: `pairwise.wilcox.test(x, g, p.adjust.method = "holm")`
pub fn pairwise_wilcox_test<T: AsRef<str> + Clone + Eq + std::hash::Hash + ToString>(
    values: &[f64],
    groups: &[T],
    alternative: Alternative,
    p_adjust_method: PValueAdjustMethod,
    exact: Option<bool>,
) -> EconResult<PairwiseWilcoxResult> {
    if values.len() != groups.len() {
        return Err(EconError::InvalidSpecification {
            message: "values and groups must have the same length".to_string(),
        });
    }

    if values.len() < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: values.len(),
            context: "pairwise Wilcoxon test requires at least 2 observations".to_string(),
        });
    }

    // Group the data
    let mut group_data: HashMap<String, Vec<f64>> = HashMap::new();
    for (v, g) in values.iter().zip(groups.iter()) {
        if v.is_finite() {
            group_data.entry(g.to_string()).or_default().push(*v);
        }
    }

    // Get sorted group names for consistent ordering
    let mut group_names: Vec<String> = group_data.keys().cloned().collect();
    group_names.sort();

    let k = group_names.len();
    if k < 2 {
        return Err(EconError::InvalidSpecification {
            message: "pairwise Wilcoxon test requires at least 2 groups".to_string(),
        });
    }

    // Compute group statistics
    let mut group_vecs: Vec<&Vec<f64>> = Vec::with_capacity(k);
    let mut group_sizes: Vec<usize> = Vec::with_capacity(k);
    let mut group_medians: Vec<f64> = Vec::with_capacity(k);

    for name in &group_names {
        let data = group_data.get(name).unwrap();
        let n = data.len();
        if n < 1 {
            return Err(EconError::InsufficientData {
                required: 1,
                provided: n,
                context: format!("Group '{}' needs at least 1 observation", name),
            });
        }

        // Compute median
        let mut sorted = data.clone();
        sorted.sort_by(f64::total_cmp);
        let median = if n.is_multiple_of(2) {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        group_vecs.push(data);
        group_sizes.push(n);
        group_medians.push(median);
    }

    // Number of comparisons: k*(k-1)/2
    let n_comparisons = comparison_count(k)?;
    let mut p_values_raw = resource::try_with_capacity(n_comparisons, "pairwise p-values")?;
    let mut w_statistics = resource::try_with_capacity(n_comparisons, "pairwise W statistics")?;
    let mut has_ties = false;
    let mut all_exact = true;

    // Configure Wilcoxon test
    let mut config = WilcoxonConfig::default();
    config.exact = exact;
    config.correct = true;
    config.conf_int = false;
    config.conf_level = 0.95;

    // Perform pairwise Wilcoxon rank sum tests (lower triangular)
    for i in 1..k {
        resource::checkpoint()?;
        for j in 0..i {
            let x = group_vecs[i];
            let y = group_vecs[j];

            let result = wilcoxon_rank_sum(x, y, 0.0, alternative, &config)?;

            w_statistics.push(result.statistic);
            p_values_raw.push(result.p_value);

            if result.n_ties > 0 {
                has_ties = true;
            }
            if !result.exact {
                all_exact = false;
            }
        }
    }

    // Adjust p-values
    let p_values_adj = p_adjust(&p_values_raw, p_adjust_method);

    // Generate warning if ties present
    let warning = if has_ties {
        Some("Cannot compute exact p-values with ties".to_string())
    } else {
        None
    };

    Ok(PairwiseWilcoxResult {
        test_name: "Pairwise comparisons using Wilcoxon rank sum test".to_string(),
        p_adjust_method,
        alternative,
        group_names,
        p_values_raw,
        p_values_adj,
        w_statistics,
        group_sizes,
        group_medians,
        n_comparisons,
        exact: all_exact,
        warning,
        missing: MissingSummary::none(values.len()),
    })
}

/// Perform pairwise Wilcoxon rank sum tests using dataset columns.
///
/// The response and the factor are read in one request: a row with a missing response (null
/// or NaN) or a missing label is dropped from both, as R does, and the drop is reported in
/// `missing`. Integer, decimal and boolean responses are accepted and labels come from the
/// column as stored (a text label "NA" is a level like any other; the CSV loader reads the
/// token NA as missing). A text response is refused (`EconError::NonNumericColumn`), as are
/// infinite values (`EconError::NonFiniteValues`). The core function orders the groups by
/// label text.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `response` - Name of the response variable column
/// * `factor` - Name of the grouping factor column
/// * `alternative` - Direction of alternative hypothesis
/// * `p_adjust_method` - Method for p-value adjustment
/// * `exact` - If Some(true), compute exact p-values; if None, auto-decide
///
/// # Example
/// ```ignore
/// let result = run_pairwise_wilcox_test(
///     &dataset, "yield", "treatment",
///     Alternative::TwoSided, PValueAdjustMethod::Holm, None
/// )?;
/// println!("{}", result);
/// ```
pub fn run_pairwise_wilcox_test(
    dataset: &Dataset,
    response: &str,
    factor: &str,
    alternative: Alternative,
    p_adjust_method: PValueAdjustMethod,
    exact: Option<bool>,
) -> EconResult<PairwiseWilcoxResult> {
    let cols = response_and_factor(dataset, response, factor)?;
    let labels = labels_of(&cols, factor)?;
    let mut result = pairwise_wilcox_test(
        cols.numeric(response),
        &labels,
        alternative,
        p_adjust_method,
        exact,
    )?;
    result.missing = cols.missing().clone();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_pairwise_tests_honour_cancellation() {
        use crate::resource::{self, Budget, CancelToken};
        let values = [5.1, 4.9, 5.3, 6.2, 6.0, 6.4, 7.1, 7.3, 6.9];
        let groups = ["a", "a", "a", "b", "b", "b", "c", "c", "c"];
        let t = CancelToken::new();
        t.cancel();
        resource::with_budget(Budget::new(None, t), || {
            assert!(matches!(
                pairwise_t_test(
                    &values,
                    &groups,
                    false,
                    Alternative::TwoSided,
                    PValueAdjustMethod::Holm
                ),
                Err(EconError::Cancelled(_))
            ));
            assert!(matches!(
                pairwise_wilcox_test(
                    &values,
                    &groups,
                    Alternative::TwoSided,
                    PValueAdjustMethod::Holm,
                    None
                ),
                Err(EconError::Cancelled(_))
            ));
        });
    }

    #[test]
    fn test_p_adjust_bonferroni() {
        let p = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let adj = p_adjust(&p, PValueAdjustMethod::Bonferroni);

        assert!((adj[0] - 0.05).abs() < 1e-10);
        assert!((adj[1] - 0.10).abs() < 1e-10);
        assert!((adj[2] - 0.15).abs() < 1e-10);
        assert!((adj[3] - 0.20).abs() < 1e-10);
        assert!((adj[4] - 0.25).abs() < 1e-10);
    }

    use p2a_validation::{Case, Check, Tol};
    use std::collections::BTreeMap;

    const METHODS: [(PValueAdjustMethod, &str); 7] = [
        (PValueAdjustMethod::Holm, "holm"),
        (PValueAdjustMethod::Hochberg, "hochberg"),
        (PValueAdjustMethod::Hommel, "hommel"),
        (PValueAdjustMethod::Bonferroni, "bonferroni"),
        (PValueAdjustMethod::BH, "bh"),
        (PValueAdjustMethod::BY, "by"),
        (PValueAdjustMethod::None, "none"),
    ];
    const VECTORS: [&str; 12] = [
        "legacy_holm",
        "legacy_validate",
        "legacy_five",
        "hommel_two",
        "hommel_six",
        "hommel_three",
        "hommel_five",
        "ties",
        "na_three",
        "na_four",
        "single",
        "ones",
    ];
    /// p.adjust is arithmetic on the p-values; NaN equals NaN.
    const ADJ: Tol = Tol::new(1e-12, 1e-15);

    fn adj_check(case: &Case, c: &mut Check, v: &str, m: PValueAdjustMethod, mn: &str) {
        c.vector(
            &format!("{v}_{mn}"),
            &p_adjust(&case.vector(&format!("{v}_p")), m),
            ADJ,
        );
    }

    /// set -> (x, g) from the case's long-form dataset (set, x, g).
    fn sets(ds: &crate::data::Dataset) -> BTreeMap<String, (Vec<f64>, Vec<String>)> {
        let df = ds.df();
        let text = |name: &str| -> Vec<String> {
            df.column(name)
                .unwrap()
                .str()
                .unwrap()
                .into_iter()
                .map(|v| v.unwrap().to_string())
                .collect()
        };
        let (s, g) = (text("set"), text("g"));
        let (x, _) =
            crate::data::columns::numeric_column(df, "x", crate::data::columns::NaPolicy::Fail)
                .unwrap();
        let mut out: BTreeMap<String, (Vec<f64>, Vec<String>)> = BTreeMap::new();
        for i in 0..x.len() {
            let e = out.entry(s[i].clone()).or_default();
            e.0.push(x[i]);
            e.1.push(g[i].clone());
        }
        out
    }

    /// R's lower triangle: (B, A), (C, A), (C, B)
    const PAIRS: [(usize, usize); 3] = [(1, 0), (2, 0), (2, 1)];

    #[test]
    fn test_validate_p_adjust_stat_p_adjust() {
        let case = Case::load("stat_p_adjust");
        let mut c = Check::new(&case);
        for v in VECTORS {
            for (m, mn) in METHODS {
                adj_check(&case, &mut c, v, m, mn);
            }
        }
        c.finish();
    }

    #[test]
    fn test_validate_p_adjust_hommel_against_r() {
        // STAT-31: a raw 0.5 came back as 0.006 (hommel_six)
        let case = Case::load("stat_p_adjust");
        let mut c = Check::new(&case);
        for v in [
            "hommel_two",
            "hommel_six",
            "hommel_three",
            "hommel_five",
            "legacy_five",
            "ties",
        ] {
            let raw = case.vector(&format!("{v}_p"));
            let adj = p_adjust(&raw, PValueAdjustMethod::Hommel);
            for (a, r) in adj.iter().zip(&raw) {
                assert!(a >= r, "{v}: adjusted {a} below raw {r}");
            }
            adj_check(&case, &mut c, v, PValueAdjustMethod::Hommel, "hommel");
        }
        c.finish();
    }

    #[test]
    fn test_validate_p_adjust_nan_kept_and_excluded_from_n() {
        // R: NA p-values stay NA and are not counted in n (STAT-30)
        let case = Case::load("stat_p_adjust");
        let mut c = Check::new(&case);
        for (m, mn) in METHODS {
            let adj = p_adjust(&case.vector("na_four_p"), m);
            assert!(adj[1].is_nan(), "{mn}: {adj:?}");
            adj_check(&case, &mut c, "na_four", m, mn);
        }
        c.finish();
        let mut p: Vec<f64> = (1..=25).map(|i| i as f64 / 30.0).collect();
        for i in [3, 11, 20] {
            p[i] = f64::NAN;
        }
        for (m, mn) in METHODS {
            let adj = p_adjust(&p, m);
            let nan: Vec<usize> = (0..25).filter(|&i| adj[i].is_nan()).collect();
            assert_eq!(nan, vec![3, 11, 20], "{mn}");
        }
    }

    #[test]
    fn test_p_adjust_method_parses_aliases_and_rejects_unknown() {
        PValueAdjustMethod::choices().validate().unwrap();
        for s in ["fdr", "BH", "benjamini-hochberg"] {
            assert_eq!(
                s.parse::<PValueAdjustMethod>().unwrap(),
                PValueAdjustMethod::BH,
                "{s}"
            );
        }
        assert_eq!(
            "bonf".parse::<PValueAdjustMethod>().unwrap(),
            PValueAdjustMethod::Bonferroni
        );
        assert_eq!(
            "Hommel".parse::<PValueAdjustMethod>().unwrap(),
            PValueAdjustMethod::Hommel
        );
        let e = "sidak"
            .parse::<PValueAdjustMethod>()
            .unwrap_err()
            .to_string();
        assert!(e.contains("holm"), "{e}");
        assert_eq!(PValueAdjustMethod::Hommel.to_string(), "hommel");
        // the inherent parser stays
        assert_eq!(
            PValueAdjustMethod::from_str("fdr"),
            Some(PValueAdjustMethod::BH)
        );
    }

    #[test]
    fn test_validate_p_adjust_holm() {
        // R: p.adjust(c(0.01, 0.04, 0.03, 0.02), "holm") is 0.04, 0.06, 0.06, 0.06
        let case = Case::load("stat_p_adjust");
        let mut c = Check::new(&case);
        adj_check(
            &case,
            &mut c,
            "legacy_holm",
            PValueAdjustMethod::Holm,
            "holm",
        );
        c.finish();
    }

    #[test]
    fn test_p_adjust_bh() {
        // R: p.adjust(c(0.01, 0.02, 0.03, 0.04, 0.05), method = "BH")
        // Expected: 0.05, 0.05, 0.05, 0.05, 0.05
        let p = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let adj = p_adjust(&p, PValueAdjustMethod::BH);

        // BH: multiply sorted p_i by n/rank (from largest), take cummin
        // All equal to 0.05 in this case
        for a in &adj {
            assert!((a - 0.05).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pairwise_t_test_basic() {
        let values = vec![
            1.0, 2.0, 3.0, 2.5, 1.5, // Group A: mean ≈ 2
            10.0, 11.0, 12.0, 10.5, 11.5, // Group B: mean ≈ 11
            20.0, 21.0, 22.0, 20.5, 21.5, // Group C: mean ≈ 21
        ];
        let groups = vec![
            "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C",
        ];

        let result = pairwise_t_test(
            &values,
            &groups,
            true,
            Alternative::TwoSided,
            PValueAdjustMethod::Holm,
        )
        .unwrap();

        assert_eq!(result.group_names, vec!["A", "B", "C"]);
        assert_eq!(result.n_comparisons, 3);
        assert_eq!(result.group_sizes, vec![5, 5, 5]);

        // All comparisons should be highly significant
        for p in &result.p_values_adj {
            assert!(*p < 0.001);
        }
    }

    #[test]
    fn test_pairwise_t_test_welch() {
        // Test with unequal variances (pool_sd = false)
        let values = vec![1.0, 2.0, 3.0, 2.5, 1.5, 10.0, 11.0, 12.0, 10.5, 11.5];
        let groups = vec!["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"];

        let result = pairwise_t_test(
            &values,
            &groups,
            false,
            Alternative::TwoSided,
            PValueAdjustMethod::None,
        )
        .unwrap();

        assert_eq!(result.n_comparisons, 1);
        assert!(!result.pool_sd);
        assert!(result.pooled_sd.is_none());
        assert!(result.p_values_raw[0] < 0.001);
    }

    #[test]
    fn test_pairwise_t_test_dataset() {
        let df = df! {
            "yield" => [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
            "treatment" => ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_pairwise_t_test(
            &dataset,
            "yield",
            "treatment",
            true,
            Alternative::TwoSided,
            PValueAdjustMethod::Bonferroni,
        )
        .unwrap();

        assert_eq!(result.group_names.len(), 3);
        assert_eq!(result.n_comparisons, 3);
    }

    // ========================================================================
    // Validation Tests Against R
    // ========================================================================

    #[test]
    fn test_validate_pairwise_t_test_pooled_against_r() {
        // R: pairwise.t.test(x, g, pool.sd = TRUE / FALSE, p.adjust.method = ...) on set a.
        // Tail p-values down to 7.1e-14: relative with no absolute floor (X-17).
        let (case, ds) = crate::test_support::load_case("stat_pairwise_legacy");
        let (x, g) = sets(&ds).remove("a").unwrap();
        let tol = Tol::new(1e-9, 0.0);
        let mut c = Check::new(&case);
        let runs = [
            (true, PValueAdjustMethod::None, "a_t_pooled_none"),
            (true, PValueAdjustMethod::Holm, "a_t_pooled_holm"),
            (true, PValueAdjustMethod::Hommel, "a_t_pooled_hommel"),
            (false, PValueAdjustMethod::None, "a_t_welch_none"),
        ];
        for (pool, m, key) in runs {
            let r = pairwise_t_test(&x, &g, pool, Alternative::TwoSided, m).unwrap();
            assert_eq!(r.group_names, vec!["A", "B", "C"]);
            let p: Vec<f64> = PAIRS
                .iter()
                .map(|&(i, j)| {
                    if m == PValueAdjustMethod::None {
                        r.p_value_raw(i, j)
                    } else {
                        r.p_value_adj(i, j)
                    }
                })
                .collect();
            c.vector(key, &p, tol);
        }
        c.finish();
    }

    #[test]
    fn test_validate_p_adjust_holm_against_r() {
        // R: p.adjust(c(0.001, 0.01, 0.05, 0.1), "holm") is 0.004, 0.03, 0.1, 0.1
        let case = Case::load("stat_p_adjust");
        let mut c = Check::new(&case);
        adj_check(
            &case,
            &mut c,
            "legacy_validate",
            PValueAdjustMethod::Holm,
            "holm",
        );
        c.finish();
    }

    #[test]
    fn test_validate_p_adjust_bh_against_r() {
        // R: p.adjust(c(0.001, 0.01, 0.05, 0.1), "BH") is 0.004, 0.02, 0.0667, 0.1
        let case = Case::load("stat_p_adjust");
        let mut c = Check::new(&case);
        adj_check(
            &case,
            &mut c,
            "legacy_validate",
            PValueAdjustMethod::BH,
            "bh",
        );
        c.finish();
    }

    #[test]
    fn test_pairwise_wilcox_basic() {
        let values = vec![
            1.0, 2.0, 3.0, 2.5, 1.5, // Group A
            10.0, 11.0, 12.0, 10.5, 11.5, // Group B
            20.0, 21.0, 22.0, 20.5, 21.5, // Group C
        ];
        let groups = vec![
            "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C",
        ];

        let result = pairwise_wilcox_test(
            &values,
            &groups,
            Alternative::TwoSided,
            PValueAdjustMethod::Holm,
            None,
        )
        .unwrap();

        assert_eq!(result.group_names, vec!["A", "B", "C"]);
        assert_eq!(result.n_comparisons, 3);
        assert_eq!(result.group_sizes, vec![5, 5, 5]);

        // All comparisons should be significant (medians very different)
        for p in &result.p_values_adj {
            assert!(*p < 0.05, "Expected p < 0.05, got {}", p);
        }
    }

    #[test]
    fn test_pairwise_wilcox_no_adjustment() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0];
        let groups = vec!["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"];

        let result = pairwise_wilcox_test(
            &values,
            &groups,
            Alternative::TwoSided,
            PValueAdjustMethod::None,
            None,
        )
        .unwrap();

        assert_eq!(result.n_comparisons, 1);
        assert!(result.p_values_raw[0] < 0.05);
        // With no adjustment, raw and adjusted should be equal
        assert!((result.p_values_raw[0] - result.p_values_adj[0]).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_wilcox_from_dataset() {
        let df = df! {
            "response" => [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
            "group" => ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_pairwise_wilcox_test(
            &dataset,
            "response",
            "group",
            Alternative::TwoSided,
            PValueAdjustMethod::Bonferroni,
            None,
        )
        .unwrap();

        assert_eq!(result.group_names.len(), 3);
        assert_eq!(result.n_comparisons, 3);
    }

    #[test]
    fn test_pairwise_wilcox_display() {
        let values = vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
        let groups = vec!["A", "A", "A", "B", "B", "B"];

        let result = pairwise_wilcox_test(
            &values,
            &groups,
            Alternative::TwoSided,
            PValueAdjustMethod::Holm,
            None,
        )
        .unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Wilcoxon"));
        assert!(display.contains("P value adjustment"));
        assert!(display.contains("Holm"));
    }

    // ========================================================================
    // Validation Tests Against R
    // ========================================================================

    #[test]
    fn test_validate_pairwise_wilcox_against_r() {
        // R: pairwise.wilcox.test(x, g, p.adjust.method = "none", exact = FALSE) on set a
        let (case, ds) = crate::test_support::load_case("stat_pairwise_legacy");
        let (x, g) = sets(&ds).remove("a").unwrap();
        let r = pairwise_wilcox_test(
            &x,
            &g,
            Alternative::TwoSided,
            PValueAdjustMethod::None,
            Some(false),
        )
        .unwrap();
        let p: Vec<f64> = PAIRS.iter().map(|&(i, j)| r.p_value_raw(i, j)).collect();
        Check::new(&case)
            .vector("a_w_normal_none", &p, Tol::p_value(15))
            .finish();
    }

    #[test]
    fn test_validate_pairwise_wilcox_exact_small_sample() {
        // R: pairwise.wilcox.test(x, g, exact = TRUE, p.adjust.method = "none") on set b
        let (case, ds) = crate::test_support::load_case("stat_pairwise_legacy");
        let (x, g) = sets(&ds).remove("b").unwrap();
        let r = pairwise_wilcox_test(
            &x,
            &g,
            Alternative::TwoSided,
            PValueAdjustMethod::None,
            Some(true),
        )
        .unwrap();
        assert!(r.exact);
        Check::new(&case)
            .vector("b_w_exact_none", &r.p_values_raw, Tol::p_value(6))
            .finish();
    }

    #[test]
    fn test_validate_pairwise_wilcox_with_holm_adjustment() {
        // R: pairwise.wilcox.test(x, g, p.adjust.method = "holm") on set c (exact: no ties)
        let (case, ds) = crate::test_support::load_case("stat_pairwise_legacy");
        let (x, g) = sets(&ds).remove("c").unwrap();
        let r = pairwise_wilcox_test(
            &x,
            &g,
            Alternative::TwoSided,
            PValueAdjustMethod::Holm,
            None,
        )
        .unwrap();
        let raw: Vec<f64> = PAIRS.iter().map(|&(i, j)| r.p_value_raw(i, j)).collect();
        let adj: Vec<f64> = PAIRS.iter().map(|&(i, j)| r.p_value_adj(i, j)).collect();
        Check::new(&case)
            .vector("c_w_holm", &adj, Tol::p_value(15))
            .vector("c_w_none", &raw, Tol::p_value(15))
            .finish();
    }

    #[test]
    fn test_pairwise_wilcox_medians() {
        let values = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, // Median = 3
            10.0, 11.0, 12.0, 13.0, 14.0, // Median = 12
        ];
        let groups = vec!["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"];

        let result = pairwise_wilcox_test(
            &values,
            &groups,
            Alternative::TwoSided,
            PValueAdjustMethod::None,
            None,
        )
        .unwrap();

        assert!((result.group_medians[0] - 3.0).abs() < 0.01);
        assert!((result.group_medians[1] - 12.0).abs() < 0.01);
    }

    // ========================================================================
    // Missing values and integer columns (case stat_pairwise_mahalanobis_na,
    // validation/r/cases/stats_onesample.R)
    // ========================================================================

    /// X-8: pairwise.t.test (pooled SD, Holm and unadjusted) on an Int64 response with
    /// missing values in rows 4 and 18.
    #[test]
    fn test_validate_pairwise_t_test_na() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_pairwise_mahalanobis_na");
        let r = run_pairwise_t_test(
            &ds,
            "y",
            "g",
            true,
            Alternative::TwoSided,
            PValueAdjustMethod::Holm,
        )
        .unwrap();
        let sizes: Vec<f64> = r.group_sizes.iter().map(|&n| n as f64).collect();
        let mut check = Check::new(&case);
        check
            .vector("pairwise_p", &r.p_values_adj, Tol::p_value(28))
            .vector("pairwise_p_raw", &r.p_values_raw, Tol::p_value(28))
            .vector("pairwise_t", &r.t_statistics, Tol::stat(28))
            .named("pairwise_group_n", &r.group_names, &sizes, Tol::exact())
            .named(
                "pairwise_group_mean",
                &r.group_names,
                &r.group_means,
                Tol::coef(28),
            )
            .scalar("pairwise_pooled_sd", r.pooled_sd.unwrap(), Tol::stat(28));
        for df in &r.df {
            check.scalar("pairwise_df", *df, Tol::exact());
        }
        check.finish();
        assert_eq!(r.missing.by_column, vec![("y".to_string(), 2)]);
    }

    #[test]
    fn test_validate_na_pairwise_tests_missing_rows_and_errors() {
        let (_case, ds) = crate::test_support::load_case("stat_pairwise_mahalanobis_na");
        // p-values not pinned: the Wilcoxon tie variance is STAT-42 (Tier 4)
        let w = run_pairwise_wilcox_test(
            &ds,
            "y",
            "g",
            Alternative::TwoSided,
            PValueAdjustMethod::Holm,
            None,
        )
        .unwrap();
        assert_eq!(w.group_sizes, [9, 9, 10]);
        assert_eq!(w.missing.n_dropped, 2);
        assert_eq!(
            w.to_string().lines().last().unwrap(),
            format!("Missing values: {}", w.missing.note().unwrap())
        );
        let t = run_pairwise_t_test(
            &ds,
            "y",
            "g",
            true,
            Alternative::TwoSided,
            PValueAdjustMethod::Holm,
        )
        .unwrap();
        assert_eq!(
            t.to_string().lines().last().unwrap(),
            format!("Missing values: {}", t.missing.note().unwrap())
        );
        let tt = |resp| {
            run_pairwise_t_test(
                &ds,
                resp,
                "g",
                true,
                Alternative::TwoSided,
                PValueAdjustMethod::Holm,
            )
        };
        let wt = |resp| {
            run_pairwise_wilcox_test(
                &ds,
                resp,
                "g",
                Alternative::TwoSided,
                PValueAdjustMethod::Holm,
                None,
            )
        };
        assert!(matches!(tt("g"), Err(EconError::NonNumericColumn { column }) if column == "g"));
        assert!(matches!(wt("g"), Err(EconError::NonNumericColumn { column }) if column == "g"));
        assert!(
            matches!(tt("nope"), Err(EconError::ColumnNotFound { column, .. }) if column == "nope")
        );
        assert!(
            matches!(wt("nope"), Err(EconError::ColumnNotFound { column, .. }) if column == "nope")
        );
    }
}
