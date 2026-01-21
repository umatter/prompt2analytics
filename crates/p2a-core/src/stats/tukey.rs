//! Tukey's Honest Significant Differences (HSD) test.
//!
//! Performs post-hoc pairwise comparisons of group means after a one-way ANOVA,
//! controlling for family-wise error rate using the Studentized range distribution.
//!
//! # References
//!
//! - Tukey, J. W. (1949). "Comparing Individual Means in the Analysis of Variance".
//!   *Biometrics*, 5(2), 99-114.
//! - Kramer, C. Y. (1956). "Extension of Multiple Range Tests to Group Means with
//!   Unequal Numbers of Replications". *Biometrics*, 12(3), 307-310.
//! - R Core Team. `stats::TukeyHSD()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/TukeyHSD.html>
//!
//! # Mathematical Background
//!
//! The Tukey HSD test compares all possible pairs of group means.
//!
//! ## Test Statistic (q)
//!
//! For each pair of groups i and j:
//! ```text
//! q = |ȳᵢ - ȳⱼ| / SE
//! ```
//!
//! where SE is the standard error of the difference.
//!
//! ## Standard Error (Tukey-Kramer method for unequal sample sizes)
//!
//! ```text
//! SE = √(MSW/2 × (1/nᵢ + 1/nⱼ))
//! ```
//!
//! where MSW is the mean square within groups from the ANOVA.
//!
//! ## Confidence Interval
//!
//! ```text
//! (ȳᵢ - ȳⱼ) ± q_{α,k,df} × SE
//! ```
//!
//! where q_{α,k,df} is the critical value from the Studentized range distribution
//! with k groups and df degrees of freedom.
//!
//! ## P-value
//!
//! P-value is computed using the upper tail of the Studentized range distribution:
//! ```text
//! p = 1 - P(Q ≤ |ȳᵢ - ȳⱼ| / SE)
//! ```

use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::stats::anova::{run_one_way_anova, AnovaResult};
use crate::traits::SignificanceLevel;

/// Result of a pairwise comparison in Tukey HSD test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseComparison {
    /// First group name
    pub group1: String,
    /// Second group name
    pub group2: String,
    /// Difference in means (group1 - group2)
    pub diff: f64,
    /// Lower bound of confidence interval
    pub ci_lower: f64,
    /// Upper bound of confidence interval
    pub ci_upper: f64,
    /// P-value (adjusted for multiple comparisons)
    pub p_adj: f64,
    /// Significance level based on p-value
    pub significance: SignificanceLevel,
}

impl std::fmt::Display for PairwiseComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}-{}: diff={:.4}, 95% CI=[{:.4}, {:.4}], p={:.4}{}",
            self.group1,
            self.group2,
            self.diff,
            self.ci_lower,
            self.ci_upper,
            self.p_adj,
            self.significance.stars()
        )
    }
}

/// Result of Tukey's HSD test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TukeyHsdResult {
    /// Response variable name
    pub response_var: String,
    /// Factor (grouping) variable name
    pub factor_var: String,
    /// Number of groups
    pub n_groups: usize,
    /// Degrees of freedom (error/within)
    pub df: usize,
    /// Mean Square Error (MSW from ANOVA)
    pub mse: f64,
    /// Confidence level used
    pub conf_level: f64,
    /// Pairwise comparisons
    pub comparisons: Vec<PairwiseComparison>,
}

impl std::fmt::Display for TukeyHsdResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Tukey HSD (Honestly Significant Differences)")?;
        writeln!(f, "=============================================")?;
        writeln!(
            f,
            "Response: {}  |  Factor: {}",
            self.response_var, self.factor_var
        )?;
        writeln!(
            f,
            "Groups = {}  |  df = {}  |  MSE = {:.4}",
            self.n_groups, self.df, self.mse
        )?;
        writeln!(f, "Confidence level: {:.0}%", self.conf_level * 100.0)?;
        writeln!(f)?;

        // Table header
        writeln!(
            f,
            "{:>20} {:>12} {:>12} {:>12} {:>10}",
            "Comparison", "diff", "lwr", "upr", "p adj"
        )?;
        writeln!(f, "{}", "-".repeat(70))?;

        for comp in &self.comparisons {
            writeln!(
                f,
                "{:>20} {:>12.4} {:>12.4} {:>12.4} {:>10.4} {}",
                format!("{}-{}", comp.group1, comp.group2),
                comp.diff,
                comp.ci_lower,
                comp.ci_upper,
                comp.p_adj,
                comp.significance.stars()
            )?;
        }

        writeln!(f)?;
        writeln!(f, "---")?;
        writeln!(f, "Signif. codes: '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")
    }
}

/// Compute the CDF of the Studentized range distribution.
///
/// Uses r_mathlib's implementation, which is a port of R's ptukey.
///
/// # Arguments
/// * `q` - The q statistic value
/// * `k` - Number of groups (nmeans)
/// * `df` - Degrees of freedom (error df)
///
/// # Returns
/// The probability P(Q <= q)
fn ptukey(q: f64, k: f64, df: f64) -> f64 {
    // r_mathlib::tukey_pdf is actually the CDF (named confusingly)
    // Parameters: q, rr (nranges), cc (nmeans), df, lower_tail, log_p
    r_mathlib::tukey_pdf(q, 1.0, k, df, true, false)
}

/// Compute the quantile of the Studentized range distribution.
///
/// Uses r_mathlib's implementation, which is a port of R's qtukey.
///
/// # Arguments
/// * `p` - The probability (1 - alpha for upper tail)
/// * `k` - Number of groups (nmeans)
/// * `df` - Degrees of freedom (error df)
///
/// # Returns
/// The q value such that P(Q <= q) = p
fn qtukey(p: f64, k: f64, df: f64) -> f64 {
    // Parameters: p, rr (nranges), cc (nmeans), df, lower_tail, log_p
    r_mathlib::tukey_quantile(p, 1.0, k, df, true, false)
}

/// Perform Tukey's HSD test on an ANOVA result.
///
/// This is the Tukey-Kramer method which handles unequal sample sizes.
///
/// # Arguments
/// * `anova` - Result from a one-way ANOVA
/// * `conf_level` - Confidence level for intervals (default: 0.95)
///
/// # Returns
/// `TukeyHsdResult` containing all pairwise comparisons with adjusted p-values
/// and confidence intervals.
///
/// # Example
///
/// ```ignore
/// use p2a_core::stats::anova::run_one_way_anova;
/// use p2a_core::stats::tukey::tukey_hsd;
///
/// let anova_result = run_one_way_anova(&dataset, "value", "group")?;
/// let tukey_result = tukey_hsd(&anova_result, 0.95)?;
/// println!("{}", tukey_result);
/// ```
///
/// # References
///
/// - Tukey, J. W. (1949). *Biometrics*, 5(2), 99-114.
/// - Kramer, C. Y. (1956). *Biometrics*, 12(3), 307-310.
pub fn tukey_hsd(anova: &AnovaResult, conf_level: f64) -> EconResult<TukeyHsdResult> {
    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(EconError::InvalidSpecification {
            message: "conf_level must be between 0 and 1".to_string(),
        });
    }

    let k = anova.n_groups;
    if k < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Tukey HSD requires at least 2 groups".to_string(),
        });
    }

    let df = anova.df_within;
    let mse = anova.ms_within;

    // Get the critical value from Studentized range distribution
    // q_crit = qtukey(conf_level, k, df)
    let q_crit = qtukey(conf_level, k as f64, df as f64);

    let mut comparisons = Vec::new();

    // Generate all pairwise comparisons (lower triangular)
    for i in 0..k {
        for j in 0..i {
            let group_i = &anova.groups[i];
            let group_j = &anova.groups[j];

            // Difference in means (following R convention: second - first in alphabetical order)
            let diff = group_i.mean - group_j.mean;

            // Standard error using Tukey-Kramer adjustment for unequal sample sizes
            // SE = sqrt((MSE/2) * (1/n_i + 1/n_j))
            let se = ((mse / 2.0) * (1.0 / group_i.n as f64 + 1.0 / group_j.n as f64)).sqrt();

            // Test statistic (q)
            let q_stat = diff.abs() / se;

            // P-value from upper tail of Studentized range distribution
            // p = 1 - P(Q <= q)
            let p_adj = 1.0 - ptukey(q_stat, k as f64, df as f64);

            // Confidence interval width
            let width = q_crit * se;

            comparisons.push(PairwiseComparison {
                group1: group_i.group.clone(),
                group2: group_j.group.clone(),
                diff,
                ci_lower: diff - width,
                ci_upper: diff + width,
                p_adj,
                significance: SignificanceLevel::from_p_value(p_adj),
            });
        }
    }

    Ok(TukeyHsdResult {
        response_var: anova.response_var.clone(),
        factor_var: anova.factor_var.clone(),
        n_groups: k,
        df,
        mse,
        conf_level,
        comparisons,
    })
}

/// Perform Tukey's HSD test directly from a dataset.
///
/// This is a convenience function that runs ANOVA first and then performs Tukey HSD.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `response_col` - Name of the response (dependent) variable column
/// * `factor_col` - Name of the factor (grouping) variable column
/// * `conf_level` - Confidence level for intervals (default: 0.95)
///
/// # Returns
/// A tuple of (AnovaResult, TukeyHsdResult)
///
/// # Example
///
/// ```ignore
/// use p2a_core::stats::tukey::run_tukey_hsd;
///
/// let (anova, tukey) = run_tukey_hsd(&dataset, "breaks", "tension", 0.95)?;
/// println!("ANOVA:\n{}", anova);
/// println!("\nTukey HSD:\n{}", tukey);
/// ```
pub fn run_tukey_hsd(
    dataset: &Dataset,
    response_col: &str,
    factor_col: &str,
    conf_level: f64,
) -> EconResult<(AnovaResult, TukeyHsdResult)> {
    let anova = run_one_way_anova(dataset, response_col, factor_col)?;
    let tukey = tukey_hsd(&anova, conf_level)?;
    Ok((anova, tukey))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_test_dataset() -> Dataset {
        // Create a dataset similar to R's warpbreaks
        // Three groups with different means
        let df = df! {
            "value" => [
                // Group A: mean ≈ 44
                44.0, 35.0, 31.0, 44.0, 29.0, 32.0, 44.0, 43.0, 36.0,
                // Group B: mean ≈ 28
                26.0, 30.0, 54.0, 25.0, 70.0, 52.0, 51.0, 26.0, 67.0,
                // Group C: mean ≈ 24
                18.0, 21.0, 29.0, 17.0, 12.0, 18.0, 35.0, 30.0, 36.0
            ],
            "group" => [
                "A", "A", "A", "A", "A", "A", "A", "A", "A",
                "B", "B", "B", "B", "B", "B", "B", "B", "B",
                "C", "C", "C", "C", "C", "C", "C", "C", "C"
            ]
        }
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_tukey_hsd_basic() {
        let dataset = create_test_dataset();
        let (anova, tukey) = run_tukey_hsd(&dataset, "value", "group", 0.95).unwrap();

        // Check basic properties
        assert_eq!(tukey.n_groups, 3);
        assert_eq!(tukey.df, anova.df_within);
        assert!((tukey.mse - anova.ms_within).abs() < 1e-10);
        assert!((tukey.conf_level - 0.95).abs() < 1e-10);

        // Should have k*(k-1)/2 = 3 comparisons
        assert_eq!(tukey.comparisons.len(), 3);

        // Check that all comparisons have valid values
        for comp in &tukey.comparisons {
            assert!(comp.ci_lower < comp.diff);
            assert!(comp.diff < comp.ci_upper);
            assert!(comp.p_adj >= 0.0 && comp.p_adj <= 1.0);
        }
    }

    #[test]
    fn test_tukey_hsd_significant_difference() {
        // Create data with clearly different group means
        let df = df! {
            "value" => [
                // Group A: mean = 10
                9.0, 10.0, 11.0, 10.0, 10.0,
                // Group B: mean = 20
                19.0, 20.0, 21.0, 20.0, 20.0,
                // Group C: mean = 30
                29.0, 30.0, 31.0, 30.0, 30.0
            ],
            "group" => [
                "A", "A", "A", "A", "A",
                "B", "B", "B", "B", "B",
                "C", "C", "C", "C", "C"
            ]
        }
        .unwrap();

        let dataset = Dataset::new(df);
        let (_, tukey) = run_tukey_hsd(&dataset, "value", "group", 0.95).unwrap();

        // All comparisons should be significant (p < 0.05)
        for comp in &tukey.comparisons {
            assert!(
                comp.p_adj < 0.05,
                "Comparison {}-{} should be significant, p = {}",
                comp.group1,
                comp.group2,
                comp.p_adj
            );
            // CI should not include 0 for significant differences
            assert!(
                comp.ci_lower > 0.0 || comp.ci_upper < 0.0,
                "CI for {}-{} should not include 0: [{}, {}]",
                comp.group1,
                comp.group2,
                comp.ci_lower,
                comp.ci_upper
            );
        }
    }

    #[test]
    fn test_tukey_hsd_no_difference() {
        // Create data with similar group means
        let df = df! {
            "value" => [
                // Group A: mean ≈ 10
                9.0, 10.0, 11.0, 10.0, 10.0, 9.5, 10.5, 9.8,
                // Group B: mean ≈ 10
                9.5, 10.5, 10.0, 9.8, 10.2, 10.0, 9.7, 10.3,
                // Group C: mean ≈ 10
                10.0, 10.0, 9.5, 10.5, 9.8, 10.2, 10.1, 9.9
            ],
            "group" => [
                "A", "A", "A", "A", "A", "A", "A", "A",
                "B", "B", "B", "B", "B", "B", "B", "B",
                "C", "C", "C", "C", "C", "C", "C", "C"
            ]
        }
        .unwrap();

        let dataset = Dataset::new(df);
        let (_, tukey) = run_tukey_hsd(&dataset, "value", "group", 0.95).unwrap();

        // All comparisons should be non-significant (p > 0.05)
        for comp in &tukey.comparisons {
            assert!(
                comp.p_adj > 0.05,
                "Comparison {}-{} should not be significant, p = {}",
                comp.group1,
                comp.group2,
                comp.p_adj
            );
            // CI should include 0 for non-significant differences
            assert!(
                comp.ci_lower < 0.0 && comp.ci_upper > 0.0,
                "CI for {}-{} should include 0: [{}, {}]",
                comp.group1,
                comp.group2,
                comp.ci_lower,
                comp.ci_upper
            );
        }
    }

    #[test]
    fn test_tukey_hsd_unequal_sizes() {
        // Test Tukey-Kramer with unequal sample sizes
        let df = df! {
            "value" => [
                // Group A: n=3
                10.0, 11.0, 12.0,
                // Group B: n=5
                20.0, 21.0, 19.0, 20.0, 21.0,
                // Group C: n=4
                30.0, 31.0, 29.0, 30.0
            ],
            "group" => [
                "A", "A", "A",
                "B", "B", "B", "B", "B",
                "C", "C", "C", "C"
            ]
        }
        .unwrap();

        let dataset = Dataset::new(df);
        let (_, tukey) = run_tukey_hsd(&dataset, "value", "group", 0.95).unwrap();

        assert_eq!(tukey.comparisons.len(), 3);

        // With clearly separated means, all should be significant
        for comp in &tukey.comparisons {
            assert!(comp.p_adj < 0.05);
        }
    }

    #[test]
    fn test_tukey_hsd_two_groups() {
        // Test with only 2 groups
        let df = df! {
            "value" => [
                10.0, 11.0, 12.0, 10.0, 11.0,
                20.0, 21.0, 19.0, 20.0, 21.0
            ],
            "group" => [
                "A", "A", "A", "A", "A",
                "B", "B", "B", "B", "B"
            ]
        }
        .unwrap();

        let dataset = Dataset::new(df);
        let (_, tukey) = run_tukey_hsd(&dataset, "value", "group", 0.95).unwrap();

        // Should have exactly 1 comparison
        assert_eq!(tukey.comparisons.len(), 1);
        assert_eq!(tukey.n_groups, 2);
    }

    #[test]
    fn test_invalid_conf_level() {
        let dataset = create_test_dataset();
        let anova = run_one_way_anova(&dataset, "value", "group").unwrap();

        // Test conf_level = 0
        let result = tukey_hsd(&anova, 0.0);
        assert!(result.is_err());

        // Test conf_level = 1
        let result = tukey_hsd(&anova, 1.0);
        assert!(result.is_err());

        // Test negative conf_level
        let result = tukey_hsd(&anova, -0.5);
        assert!(result.is_err());

        // Test conf_level > 1
        let result = tukey_hsd(&anova, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_tukey_against_r() {
        // Test case from R:
        // > y <- c(1, 2, 3, 4, 5, 6, 7, 8, 9)
        // > group <- factor(c("A", "A", "A", "B", "B", "B", "C", "C", "C"))
        // > fit <- aov(y ~ group)
        // > TukeyHSD(fit)
        //
        // Actual R output:
        //   Tukey multiple comparisons of means
        //     95% family-wise confidence level
        //
        // $group
        //          diff       lwr      upr     p adj
        // B-A    3 0.4947644 5.505236 0.0242291
        // C-A    6 3.4947644 8.505236 0.0007942
        // C-B    3 0.4947644 5.505236 0.0242291

        let df = df! {
            "y" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "group" => ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
        }
        .unwrap();

        let dataset = Dataset::new(df);
        let (_, tukey) = run_tukey_hsd(&dataset, "y", "group", 0.95).unwrap();

        // Find each comparison
        let find_comp = |g1: &str, g2: &str| -> &PairwiseComparison {
            tukey
                .comparisons
                .iter()
                .find(|c| {
                    (c.group1 == g1 && c.group2 == g2) || (c.group1 == g2 && c.group2 == g1)
                })
                .unwrap()
        };

        // B-A comparison
        let ba = find_comp("B", "A");
        assert!(
            (ba.diff.abs() - 3.0).abs() < 0.001,
            "B-A diff should be 3, got {}",
            ba.diff.abs()
        );
        // R: lwr=0.4947644, upr=5.505236
        assert!(
            (ba.ci_lower.abs() - 0.4948).abs() < 0.01,
            "B-A ci_lower should be ~0.4948, got {}",
            ba.ci_lower
        );
        assert!(
            (ba.ci_upper.abs() - 5.5052).abs() < 0.01,
            "B-A ci_upper should be ~5.5052, got {}",
            ba.ci_upper
        );
        // R: p adj = 0.0242291
        assert!(
            (ba.p_adj - 0.0242).abs() < 0.001,
            "B-A p-value should be ~0.0242, got {}",
            ba.p_adj
        );

        // C-A comparison
        let ca = find_comp("C", "A");
        assert!(
            (ca.diff.abs() - 6.0).abs() < 0.001,
            "C-A diff should be 6, got {}",
            ca.diff.abs()
        );
        // R: p adj = 0.0007942
        assert!(
            (ca.p_adj - 0.0008).abs() < 0.001,
            "C-A p-value should be ~0.0008, got {}",
            ca.p_adj
        );

        // C-B comparison
        let cb = find_comp("C", "B");
        assert!(
            (cb.diff.abs() - 3.0).abs() < 0.001,
            "C-B diff should be 3, got {}",
            cb.diff.abs()
        );
        // R: p adj = 0.0242291
        assert!(
            (cb.p_adj - 0.0242).abs() < 0.001,
            "C-B p-value should be ~0.0242, got {}",
            cb.p_adj
        );
    }

    #[test]
    fn test_tukey_display() {
        let dataset = create_test_dataset();
        let (_, tukey) = run_tukey_hsd(&dataset, "value", "group", 0.95).unwrap();

        let display = format!("{}", tukey);
        assert!(display.contains("Tukey HSD"));
        assert!(display.contains("Comparison"));
        assert!(display.contains("diff"));
        assert!(display.contains("lwr"));
        assert!(display.contains("upr"));
        assert!(display.contains("p adj"));
    }

    #[test]
    fn test_pairwise_comparison_display() {
        let comp = PairwiseComparison {
            group1: "A".to_string(),
            group2: "B".to_string(),
            diff: 5.0,
            ci_lower: 2.0,
            ci_upper: 8.0,
            p_adj: 0.001,
            significance: SignificanceLevel::TenthPercent,
        };

        let display = format!("{}", comp);
        assert!(display.contains("A-B"));
        assert!(display.contains("5.0"));
        assert!(display.contains("***")); // Highly significant
    }
}
