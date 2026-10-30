//! Analysis of Variance (ANOVA) for comparing group means.
//!
//! Provides one-way and two-way ANOVA implementations following the standard
//! sum of squares decomposition approach.
//!
//! # References
//!
//! - Fisher, R. A. (1925). *Statistical Methods for Research Workers*.
//!   Oliver & Boyd. (Original ANOVA development)
//! - R Core Team. `stats::aov()` and `stats::anova()` functions.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/anova.html>
//!
//! # Mathematical Background
//!
//! ANOVA partitions total variation into components:
//!
//! **Total Sum of Squares (SST):**
//! ```text
//! SST = Σᵢⱼ (yᵢⱼ - ȳ..)²
//! ```
//!
//! **Between-Groups Sum of Squares (SSB):**
//! ```text
//! SSB = Σᵢ nᵢ(ȳᵢ. - ȳ..)²
//! ```
//!
//! **Within-Groups Sum of Squares (SSW):**
//! ```text
//! SSW = Σᵢⱼ (yᵢⱼ - ȳᵢ.)²
//! ```
//!
//! The F-statistic tests H₀: all group means are equal:
//! ```text
//! F = (SSB / df_between) / (SSW / df_within) = MSB / MSW
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use statrs::distribution::{ContinuousCDF, FisherSnedecor};

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, Factor, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::traits::SignificanceLevel;

/// Statistics for a single group in ANOVA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupStats {
    /// Group identifier/name
    pub group: String,
    /// Number of observations in the group
    pub n: usize,
    /// Mean of the group
    pub mean: f64,
    /// Standard deviation of the group
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
}

/// Result of a one-way ANOVA analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnovaResult {
    // ═══════════════════════════════════════════════════════════════════════
    // Identification
    // ═══════════════════════════════════════════════════════════════════════
    /// Response variable name
    pub response_var: String,
    /// Factor (grouping) variable name
    pub factor_var: String,

    // ═══════════════════════════════════════════════════════════════════════
    // Sum of Squares Decomposition
    // ═══════════════════════════════════════════════════════════════════════
    /// Sum of Squares Between Groups (SSB or SSTreatment)
    pub ss_between: f64,
    /// Sum of Squares Within Groups (SSW or SSError)
    pub ss_within: f64,
    /// Total Sum of Squares (SST)
    pub ss_total: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Degrees of Freedom
    // ═══════════════════════════════════════════════════════════════════════
    /// Degrees of freedom between groups (k - 1)
    pub df_between: usize,
    /// Degrees of freedom within groups (n - k)
    pub df_within: usize,
    /// Total degrees of freedom (n - 1)
    pub df_total: usize,

    // ═══════════════════════════════════════════════════════════════════════
    // Mean Squares
    // ═══════════════════════════════════════════════════════════════════════
    /// Mean Square Between (MSB = SSB / df_between)
    pub ms_between: f64,
    /// Mean Square Within (MSW = SSW / df_within)
    pub ms_within: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Test Statistics
    // ═══════════════════════════════════════════════════════════════════════
    /// F-statistic (MSB / MSW)
    pub f_statistic: f64,
    /// P-value for F-test
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,

    // ═══════════════════════════════════════════════════════════════════════
    // Derived Statistics
    // ═══════════════════════════════════════════════════════════════════════
    /// Effect size: η² (eta-squared) = SSB / SST
    pub eta_squared: f64,
    /// Adjusted effect size: ω² (omega-squared)
    pub omega_squared: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Group Statistics
    // ═══════════════════════════════════════════════════════════════════════
    /// Number of groups (k)
    pub n_groups: usize,
    /// Total observations
    pub n_obs: usize,
    /// Grand mean (overall mean)
    pub grand_mean: f64,
    /// Statistics for each group
    pub groups: Vec<GroupStats>,
    /// Rows dropped because the response or the factor is missing (R aov's na.omit).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl std::fmt::Display for AnovaResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "One-Way ANOVA Results")?;
        writeln!(f, "=====================")?;
        writeln!(
            f,
            "Response: {}  |  Factor: {}",
            self.response_var, self.factor_var
        )?;
        writeln!(f, "N = {}  |  Groups = {}", self.n_obs, self.n_groups)?;
        writeln!(f)?;

        // ANOVA Table
        writeln!(f, "ANOVA Table")?;
        writeln!(
            f,
            "───────────────────────────────────────────────────────────────"
        )?;
        writeln!(
            f,
            "{:>12} {:>12} {:>8} {:>12} {:>10} {:>10}",
            "Source", "SS", "DF", "MS", "F", "Pr(>F)"
        )?;
        writeln!(
            f,
            "───────────────────────────────────────────────────────────────"
        )?;
        writeln!(
            f,
            "{:>12} {:>12.4} {:>8} {:>12.4} {:>10.4} {:>10.4} {}",
            "Between",
            self.ss_between,
            self.df_between,
            self.ms_between,
            self.f_statistic,
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(
            f,
            "{:>12} {:>12.4} {:>8} {:>12.4}",
            "Within", self.ss_within, self.df_within, self.ms_within
        )?;
        writeln!(
            f,
            "───────────────────────────────────────────────────────────────"
        )?;
        writeln!(
            f,
            "{:>12} {:>12.4} {:>8}",
            "Total", self.ss_total, self.df_total
        )?;
        writeln!(f)?;

        // Effect sizes
        writeln!(f, "Effect Sizes:")?;
        writeln!(f, "  η² (eta-squared):   {:.4}", self.eta_squared)?;
        writeln!(f, "  ω² (omega-squared): {:.4}", self.omega_squared)?;
        writeln!(f)?;

        // Group means
        writeln!(f, "Group Means:")?;
        writeln!(
            f,
            "{:>15} {:>8} {:>12} {:>12}",
            "Group", "N", "Mean", "Std.Dev"
        )?;
        writeln!(f, "───────────────────────────────────────────────────")?;
        for group in &self.groups {
            writeln!(
                f,
                "{:>15} {:>8} {:>12.4} {:>12.4}",
                truncate(&group.group, 15),
                group.n,
                group.mean,
                group.std_dev
            )?;
        }
        writeln!(f, "───────────────────────────────────────────────────")?;
        writeln!(
            f,
            "{:>15} {:>8} {:>12.4}",
            "Grand Mean", self.n_obs, self.grand_mean
        )?;
        writeln!(f)?;

        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

/// Truncate a string to a maximum length.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    } else {
        s.to_string()
    }
}

/// The values of each factor level, in level order (R `factor()` order), labelled by the
/// level's label. Every level of an extracted factor is observed, so no group is empty.
pub(crate) fn values_by_level(values: &[f64], factor: &Factor) -> Vec<(String, Vec<f64>)> {
    let counts = factor.counts();
    let mut groups: Vec<(String, Vec<f64>)> = factor
        .levels
        .iter()
        .zip(&counts)
        .map(|(label, &n)| (label.clone(), Vec::with_capacity(n)))
        .collect();
    for (&v, &code) in values.iter().zip(&factor.codes) {
        groups[code as usize].1.push(v);
    }
    groups
}

/// Perform one-way ANOVA on a dataset.
///
/// Tests whether the means of the response variable differ significantly
/// across levels of the factor variable.
///
/// Rows with a missing response or factor level are dropped, as `aov`'s `na.omit`, and
/// reported in `missing`; a missing label is never a group of its own. Integer responses
/// are accepted. Groups are listed in R `factor()` order (numeric labels numerically).
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `response` - Name of the response (dependent) variable column (numeric)
/// * `factor` - Name of the factor (grouping) variable column (categorical)
///
/// # Returns
/// An `AnovaResult` containing the ANOVA table and test statistics.
///
/// # Example
/// ```ignore
/// let result = run_one_way_anova(&dataset, "yield", "fertilizer")?;
/// println!("{}", result);
/// // If p < 0.05, at least one fertilizer type has a different mean yield
/// ```
///
/// # References
/// - R equivalent: `aov(yield ~ fertilizer, data = df)`
pub fn run_one_way_anova(
    dataset: &Dataset,
    response: &str,
    factor: &str,
) -> EconResult<AnovaResult> {
    let cols = ColumnRequest::new(dataset.df())
        .numeric(response)
        .factor(factor)
        .na(NaPolicy::Omit)
        .extract()?;
    let groups = values_by_level(cols.numeric(response), cols.factor(factor));
    let y_values = cols.numeric(response);
    let n_total = y_values.len();
    let n_groups = groups.len();

    if n_groups < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_groups,
            context: "ANOVA requires at least 2 groups".to_string(),
        });
    }

    // Compute grand mean
    let grand_mean: f64 = y_values.iter().sum::<f64>() / n_total as f64;

    // Compute group statistics and sum of squares
    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    let mut group_stats: Vec<GroupStats> = Vec::with_capacity(n_groups);

    for (group_name, values) in &groups {
        let n_i = values.len();
        let group_mean: f64 = values.iter().sum::<f64>() / n_i as f64;

        // SSB contribution: n_i * (group_mean - grand_mean)²
        ss_between += n_i as f64 * (group_mean - grand_mean).powi(2);

        // SSW contribution: Σ(y_ij - group_mean)²
        let group_ssw: f64 = values.iter().map(|&y| (y - group_mean).powi(2)).sum();
        ss_within += group_ssw;

        // Compute group stats
        let std_dev = if n_i > 1 {
            (group_ssw / (n_i - 1) as f64).sqrt()
        } else {
            0.0
        };

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        group_stats.push(GroupStats {
            group: group_name.clone(),
            n: n_i,
            mean: group_mean,
            std_dev,
            min,
            max,
        });
    }

    // Compute total sum of squares
    let ss_total = ss_between + ss_within;

    // Degrees of freedom
    let df_between = n_groups - 1;
    let df_within = n_total - n_groups;
    let df_total = n_total - 1;

    // Mean squares
    let ms_between = ss_between / df_between as f64;
    let ms_within = if df_within > 0 {
        ss_within / df_within as f64
    } else {
        f64::NAN
    };

    // F-statistic and p-value
    let f_statistic = if ms_within > 0.0 {
        ms_between / ms_within
    } else {
        f64::NAN
    };

    let p_value = f_upper_tail(f_statistic, df_between as f64, df_within as f64);

    // Effect sizes
    // η² (eta-squared) = SSB / SST - proportion of variance explained
    let eta_squared = if ss_total > 0.0 {
        ss_between / ss_total
    } else {
        0.0
    };

    // ω² (omega-squared) - less biased estimator
    // ω² = (SSB - (k-1)*MSW) / (SST + MSW)
    let omega_squared = if ss_total + ms_within > 0.0 {
        let numerator = ss_between - (df_between as f64) * ms_within;
        let denominator = ss_total + ms_within;
        (numerator / denominator).max(0.0) // Can be negative for small effects
    } else {
        0.0
    };

    Ok(AnovaResult {
        response_var: response.to_string(),
        factor_var: factor.to_string(),
        ss_between,
        ss_within,
        ss_total,
        df_between,
        df_within,
        df_total,
        ms_between,
        ms_within,
        f_statistic,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        eta_squared,
        omega_squared,
        n_groups,
        n_obs: n_total,
        grand_mean,
        groups: group_stats,
        missing: cols.missing().clone(),
    })
}

/// Result of a two-way ANOVA analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoWayAnovaResult {
    // ═══════════════════════════════════════════════════════════════════════
    // Identification
    // ═══════════════════════════════════════════════════════════════════════
    /// Response variable name
    pub response_var: String,
    /// First factor variable name
    pub factor_a: String,
    /// Second factor variable name
    pub factor_b: String,
    /// Whether interaction term is included
    pub with_interaction: bool,

    // ═══════════════════════════════════════════════════════════════════════
    // Sum of Squares
    // ═══════════════════════════════════════════════════════════════════════
    /// Sum of Squares for Factor A
    pub ss_a: f64,
    /// Sum of Squares for Factor B
    pub ss_b: f64,
    /// Sum of Squares for A×B Interaction (if included)
    pub ss_ab: Option<f64>,
    /// Sum of Squares Error (Within)
    pub ss_error: f64,
    /// Total Sum of Squares
    pub ss_total: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Degrees of Freedom
    // ═══════════════════════════════════════════════════════════════════════
    /// DF for Factor A (levels_a - 1)
    pub df_a: usize,
    /// DF for Factor B (levels_b - 1)
    pub df_b: usize,
    /// DF for Interaction (df_a * df_b)
    pub df_ab: Option<usize>,
    /// DF for Error
    pub df_error: usize,
    /// Total DF
    pub df_total: usize,

    // ═══════════════════════════════════════════════════════════════════════
    // Mean Squares
    // ═══════════════════════════════════════════════════════════════════════
    pub ms_a: f64,
    pub ms_b: f64,
    pub ms_ab: Option<f64>,
    pub ms_error: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // F-Statistics and P-Values
    // ═══════════════════════════════════════════════════════════════════════
    pub f_a: f64,
    pub p_a: f64,
    pub sig_a: SignificanceLevel,

    pub f_b: f64,
    pub p_b: f64,
    pub sig_b: SignificanceLevel,

    pub f_ab: Option<f64>,
    pub p_ab: Option<f64>,
    pub sig_ab: Option<SignificanceLevel>,

    // ═══════════════════════════════════════════════════════════════════════
    // Sample Info
    // ═══════════════════════════════════════════════════════════════════════
    pub n_obs: usize,
    pub levels_a: usize,
    pub levels_b: usize,
    pub grand_mean: f64,
    /// Sums of squares in this result (Type I = sequential, as R aov).
    #[serde(default)]
    pub ss_type: SsType,
    /// Rows dropped because the response or a factor is missing (R aov's na.omit).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl std::fmt::Display for TwoWayAnovaResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Two-Way ANOVA Results")?;
        writeln!(f, "=====================")?;
        writeln!(
            f,
            "Response: {}  |  Factors: {} × {}",
            self.response_var, self.factor_a, self.factor_b
        )?;
        writeln!(
            f,
            "N = {}  |  Levels: {} × {}",
            self.n_obs, self.levels_a, self.levels_b
        )?;
        if self.with_interaction {
            writeln!(
                f,
                "Model: {} + {} + {}:{}",
                self.factor_a, self.factor_b, self.factor_a, self.factor_b
            )?;
        } else {
            writeln!(f, "Model: {} + {} (additive)", self.factor_a, self.factor_b)?;
        }
        writeln!(
            f,
            "Sums of squares: {}",
            match self.ss_type {
                SsType::I => "Type I (sequential, as R aov)",
                SsType::II => "Type II (as car::Anova(type = 2))",
                SsType::III => "Type III (sum-to-zero contrasts, as car::Anova(type = 3))",
            }
        )?;
        writeln!(f)?;

        // ANOVA Table
        writeln!(f, "ANOVA Table")?;
        writeln!(
            f,
            "───────────────────────────────────────────────────────────────────"
        )?;
        writeln!(
            f,
            "{:>15} {:>12} {:>6} {:>12} {:>10} {:>10}",
            "Source", "SS", "DF", "MS", "F", "Pr(>F)"
        )?;
        writeln!(
            f,
            "───────────────────────────────────────────────────────────────────"
        )?;

        writeln!(
            f,
            "{:>15} {:>12.4} {:>6} {:>12.4} {:>10.4} {:>10.4} {}",
            self.factor_a,
            self.ss_a,
            self.df_a,
            self.ms_a,
            self.f_a,
            self.p_a,
            self.sig_a.stars()
        )?;

        writeln!(
            f,
            "{:>15} {:>12.4} {:>6} {:>12.4} {:>10.4} {:>10.4} {}",
            self.factor_b,
            self.ss_b,
            self.df_b,
            self.ms_b,
            self.f_b,
            self.p_b,
            self.sig_b.stars()
        )?;

        let label = format!("{}:{}", self.factor_a, self.factor_b);
        if let Some(df_ab) = self.df_ab {
            writeln!(
                f,
                "{:>15} {:>12.4} {:>6} {:>12.4} {:>10.4} {:>10.4} {}",
                truncate(&label, 15),
                self.ss_ab.unwrap_or(f64::NAN),
                df_ab,
                self.ms_ab.unwrap_or(f64::NAN),
                self.f_ab.unwrap_or(f64::NAN),
                self.p_ab.unwrap_or(f64::NAN),
                self.sig_ab.as_ref().map_or("", |s| s.stars())
            )?;
        } else if self.with_interaction {
            writeln!(
                f,
                "{:>15} not estimable (empty cells leave no interaction df)",
                truncate(&label, 15)
            )?;
        }

        writeln!(
            f,
            "{:>15} {:>12.4} {:>6} {:>12.4}",
            "Residuals", self.ss_error, self.df_error, self.ms_error
        )?;

        writeln!(
            f,
            "───────────────────────────────────────────────────────────────────"
        )?;
        writeln!(
            f,
            "{:>15} {:>12.4} {:>6}",
            "Total", self.ss_total, self.df_total
        )?;
        if self.ss_type != SsType::I {
            writeln!(
                f,
                "Note: Type II/III sums of squares need not add up to the total."
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

/// Sums of squares for a two-way ANOVA (R `anova(lm)` sequential Type I, or
/// `car::Anova` Type II / III).
///
/// Used by [`run_two_way_anova_with`], implemented by T2-WP07.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SsType {
    /// Sequential sums of squares (R `aov`, `anova(lm)`). Default.
    #[default]
    I,
    /// Each main effect adjusted for the other (R `car::Anova(type = 2)`).
    II,
    /// Each term adjusted for all others, sum-to-zero contrasts (R `car::Anova(type = 3)`).
    III,
}

impl std::fmt::Display for SsType {
    /// `"I"`, `"II"` or `"III"`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            SsType::I => "I",
            SsType::II => "II",
            SsType::III => "III",
        })
    }
}

impl std::str::FromStr for SsType {
    type Err = EconError;

    /// `"I"`/`"1"`, `"II"`/`"2"` or `"III"`/`"3"`, ignoring ASCII case, surrounding
    /// whitespace and an optional leading `"type"` (`"Type 3"`, `"type_II"`).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.trim().to_ascii_lowercase();
        let body = lower
            .strip_prefix("type")
            .unwrap_or(&lower)
            .trim_start_matches([' ', '_', '-']);
        match body {
            "i" | "1" => Ok(SsType::I),
            "ii" | "2" => Ok(SsType::II),
            "iii" | "3" => Ok(SsType::III),
            _ => {
                // Echo at most 40 characters of the input.
                let mut shown: String = s.chars().take(40).collect();
                if s.chars().nth(40).is_some() {
                    shown.push_str("...");
                }
                Err(EconError::InvalidSpecification {
                    message: format!(
                        "Unknown sum-of-squares type '{shown}'; use \"I\" (sequential, as R aov), \
                         \"II\" or \"III\" (as car::Anova)"
                    ),
                })
            }
        }
    }
}

/// Options of [`run_two_way_anova_with`] (implemented by T2-WP07).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TwoWayAnovaConfig {
    /// Include the A x B interaction (default true).
    pub interaction: bool,
    /// Sums of squares (default [`SsType::I`]).
    pub ss_type: SsType,
}

impl Default for TwoWayAnovaConfig {
    fn default() -> Self {
        TwoWayAnovaConfig {
            interaction: true,
            ss_type: SsType::I,
        }
    }
}

/// Two-way ANOVA with the options of [`TwoWayAnovaConfig`]: R `aov` (Type I) or
/// `car::Anova(type = 2 | 3)` for any layout, balanced or not, with or without empty cells.
///
/// The model is fitted by least squares on dummy-coded designs: treatment contrasts for
/// Types I and II, sum-to-zero contrasts (`contr.sum`) for Type III. Each term's sum of
/// squares is the reduction in the residual sum of squares when it enters a sequence of
/// nested models, and its df is the rank it adds (R's `lm` aliasing rule: a design column
/// whose residual on the earlier columns is below 1e-7 of its length is aliased):
///
/// - **Type I** (sequential, `aov`): A, then B given A, then A:B given both.
/// - **Type II** (`car::Anova(type = 2)`): A given B, B given A, A:B given both. With empty
///   cells the terms are still compared as nested models, as car does after its note that
///   the model has aliased coefficients.
/// - **Type III** (`car::Anova(type = 3)`, sum contrasts): every term given all others. It is
///   refused with [`EconError::InvalidSpecification`] when the model has aliased
///   coefficients (empty cells), as car refuses it.
///
/// Empty cells reduce the interaction df (a 3 x 2 layout with one empty cell has 1); when
/// they leave the interaction no df, its fields are `None`, as `aov` drops that row. The
/// residual df is `n - rank`. Rows with a missing response or factor level are dropped
/// (`aov`'s `na.omit`) and reported in `missing`; integer responses are accepted.
///
/// The fit works on the cell means weighted by the square roots of the cell sizes, so it
/// costs O(n) for the cells plus O(p c r) for c non-empty cells, p design columns and rank
/// r; each design column is a cancellation point.
pub fn run_two_way_anova_with(
    dataset: &Dataset,
    response: &str,
    factor_a: &str,
    factor_b: &str,
    config: &TwoWayAnovaConfig,
) -> EconResult<TwoWayAnovaResult> {
    if factor_a == factor_b {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Two-way ANOVA needs two different factors; '{factor_a}' was given twice"
            ),
        });
    }
    let cols = ColumnRequest::new(dataset.df())
        .numeric(response)
        .factor(factor_a)
        .factor(factor_b)
        .na(NaPolicy::Omit)
        .extract()?;
    let fa = cols.factor(factor_a);
    let fb = cols.factor(factor_b);
    let (levels_a, levels_b) = (fa.n_levels(), fb.n_levels());
    if levels_a < 2 || levels_b < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: levels_a.min(levels_b),
            context: "Two-way ANOVA requires at least 2 levels for each factor".to_string(),
        });
    }
    let y = cols.numeric(response);
    let n = y.len();
    let cells = CellLayout::new(y, fa, fb)?;
    let interaction = config.interaction;

    let main = |first: Term, second: Term| -> Vec<Term> {
        let mut t = vec![Term::Intercept, first, second];
        if interaction {
            t.push(Term::Ab);
        }
        t
    };
    // (SS, df) of A, B and A:B, and the full model's fit.
    let (term_a, term_b, term_ab, full) = match config.ss_type {
        SsType::I => {
            let full = fit_sequence(&cells, &main(Term::A, Term::B), Coding::Treatment)?;
            (
                full.terms[1],
                full.terms[2],
                full.terms.get(3).copied(),
                full,
            )
        }
        SsType::II => {
            let full = fit_sequence(&cells, &main(Term::A, Term::B), Coding::Treatment)?;
            let ba = fit_sequence(
                &cells,
                &[Term::Intercept, Term::B, Term::A],
                Coding::Treatment,
            )?;
            (ba.terms[2], full.terms[2], full.terms.get(3).copied(), full)
        }
        SsType::III => {
            let full_cells = levels_a.checked_mul(levels_b);
            let empty = full_cells.map_or(usize::MAX, |c| c.saturating_sub(cells.n_cells()));
            let refuse = || EconError::InvalidSpecification {
                message: format!(
                    "Type III sums of squares are not defined: the model has aliased \
                     coefficients ({empty} empty cell(s) in {levels_a} x {levels_b}), as \
                     car::Anova; use Type I or Type II"
                ),
            };
            if interaction && empty > 0 {
                return Err(refuse());
            }
            let full = fit_sequence(&cells, &main(Term::A, Term::B), Coding::Sum)?;
            if full.rank < full.n_cols {
                return Err(refuse());
            }
            let last = |others: &[Term], term: Term| -> EconResult<TermFit> {
                let mut t = vec![Term::Intercept];
                t.extend_from_slice(others);
                t.push(term);
                Ok(*fit_sequence(&cells, &t, Coding::Sum)?
                    .terms
                    .last()
                    .unwrap_or(&TermFit::default()))
            };
            let with_ab = |t: Term| -> Vec<Term> {
                if interaction {
                    vec![t, Term::Ab]
                } else {
                    vec![t]
                }
            };
            let a = last(&with_ab(Term::B), Term::A)?;
            let b = last(&with_ab(Term::A), Term::B)?;
            (a, b, full.terms.get(3).copied(), full)
        }
    };

    let df_error = n.saturating_sub(full.rank);
    let ss_error = full.rss;
    let ms_error = mean_square(ss_error, df_error);
    let f_of = |ms: f64| {
        if df_error > 0 && ms_error > 0.0 {
            ms / ms_error
        } else {
            f64::NAN
        }
    };
    let stats = |t: TermFit| {
        let ms = mean_square(t.ss, t.df);
        let f = f_of(ms);
        let p = f_upper_tail(f, t.df as f64, df_error as f64);
        (ms, f, p)
    };
    let (ms_a, f_a, p_a) = stats(term_a);
    let (ms_b, f_b, p_b) = stats(term_b);
    // An interaction that empty cells leave without df has no row (aov omits it).
    let term_ab = term_ab.filter(|t| t.df > 0);
    let ab = term_ab.map(stats);

    let grand_mean = y.iter().sum::<f64>() / n as f64;
    let ss_between_cells: f64 = cells
        .n
        .iter()
        .zip(&cells.mean)
        .map(|(&nc, &m)| nc as f64 * (m - grand_mean).powi(2))
        .sum();

    Ok(TwoWayAnovaResult {
        response_var: response.to_string(),
        factor_a: factor_a.to_string(),
        factor_b: factor_b.to_string(),
        with_interaction: interaction,
        ss_a: term_a.ss,
        ss_b: term_b.ss,
        ss_ab: term_ab.map(|t| t.ss),
        ss_error,
        ss_total: cells.ss_within + ss_between_cells,
        df_a: term_a.df,
        df_b: term_b.df,
        df_ab: term_ab.map(|t| t.df),
        df_error,
        df_total: n - 1,
        ms_a,
        ms_b,
        ms_ab: ab.map(|(ms, _, _)| ms),
        ms_error,
        f_a,
        p_a,
        sig_a: SignificanceLevel::from_p_value(p_a),
        f_b,
        p_b,
        sig_b: SignificanceLevel::from_p_value(p_b),
        f_ab: ab.map(|(_, f, _)| f),
        p_ab: ab.map(|(_, _, p)| p),
        sig_ab: ab.map(|(_, _, p)| SignificanceLevel::from_p_value(p)),
        n_obs: n,
        levels_a,
        levels_b,
        grand_mean,
        ss_type: config.ss_type,
        missing: cols.missing().clone(),
    })
}

/// Perform two-way ANOVA on a dataset: R `aov`'s sequential (Type I) table.
///
/// Tests the effects of two categorical factors on a response variable, optionally
/// including their interaction, for any layout: balanced, unbalanced, or with empty cells
/// (whose interaction df come from the rank, as in `aov`). Equivalent to
/// [`run_two_way_anova_with`] with [`SsType::I`]; Types II and III are available there.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `response` - Name of the response (dependent) variable column (numeric)
/// * `factor_a` - Name of the first factor variable column (categorical)
/// * `factor_b` - Name of the second factor variable column (categorical)
/// * `interaction` - Whether to include the interaction term A×B
///
/// # Returns
/// A `TwoWayAnovaResult` containing the ANOVA table and test statistics.
///
/// # Example
/// ```ignore
/// let result = run_two_way_anova(&dataset, "yield", "fertilizer", "irrigation", true)?;
/// println!("{}", result);
/// ```
///
/// # References
/// - R equivalent: `aov(yield ~ fertilizer * irrigation, data = df)` (with interaction)
/// - R equivalent: `aov(yield ~ fertilizer + irrigation, data = df)` (additive)
pub fn run_two_way_anova(
    dataset: &Dataset,
    response: &str,
    factor_a: &str,
    factor_b: &str,
    interaction: bool,
) -> EconResult<TwoWayAnovaResult> {
    run_two_way_anova_with(
        dataset,
        response,
        factor_a,
        factor_b,
        &TwoWayAnovaConfig {
            interaction,
            ss_type: SsType::I,
        },
    )
}

/// `ss / df`, NaN when `df == 0`.
fn mean_square(ss: f64, df: usize) -> f64 {
    if df == 0 { f64::NAN } else { ss / df as f64 }
}

/// Upper tail `P(F > f)` of the F distribution on (`df1`, `df2`) df: NaN for a NaN or
/// negative `f`, non-finite or non-positive df, or an invalid distribution; 0 for `+Inf`.
fn f_upper_tail(f: f64, df1: f64, df2: f64) -> f64 {
    if f.is_nan() || f < 0.0 || !df1.is_finite() || !df2.is_finite() || df1 <= 0.0 || df2 <= 0.0 {
        return f64::NAN;
    }
    if f == f64::INFINITY {
        return 0.0;
    }
    match FisherSnedecor::new(df1, df2) {
        Ok(dist) => dist.sf(f),
        Err(_) => f64::NAN,
    }
}

/// R's `lm` tolerance for aliased design columns (`dqrdc2`, `tol = 1e-7`).
const ALIAS_TOL: f64 = 1e-7;

/// The non-empty cells of a two-way layout: their levels and sufficient statistics.
struct CellLayout {
    /// Level code of factor A of each cell.
    a: Vec<u32>,
    /// Level code of factor B of each cell.
    b: Vec<u32>,
    /// Rows per cell.
    n: Vec<usize>,
    /// Response mean per cell.
    mean: Vec<f64>,
    /// `sqrt(n_c)`: the weight of a cell's row in the cell-level regression.
    w: Vec<f64>,
    /// Pooled within-cell sum of squares (the residual of the saturated model).
    ss_within: f64,
    levels_a: usize,
    levels_b: usize,
}

impl CellLayout {
    /// Two O(n) passes over the kept rows: cell counts and means, then within-cell squares.
    fn new(y: &[f64], fa: &Factor, fb: &Factor) -> EconResult<Self> {
        let n = y.len();
        let mut index: HashMap<(u32, u32), usize> = HashMap::new();
        let mut row_cell: Vec<usize> = resource::try_with_capacity(n, "two-way ANOVA cells")?;
        let (mut a, mut b, mut count, mut sum) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, &yi) in y.iter().enumerate() {
            let key = (fa.codes[i], fb.codes[i]);
            let c = *index.entry(key).or_insert_with(|| {
                a.push(key.0);
                b.push(key.1);
                count.push(0usize);
                sum.push(0.0);
                a.len() - 1
            });
            count[c] += 1;
            sum[c] += yi;
            row_cell.push(c);
        }
        let mean: Vec<f64> = sum
            .iter()
            .zip(&count)
            .map(|(&s, &k)| s / k as f64)
            .collect();
        let ss_within = y
            .iter()
            .zip(&row_cell)
            .map(|(&yi, &c)| (yi - mean[c]).powi(2))
            .sum();
        let w = count.iter().map(|&k| (k as f64).sqrt()).collect();
        Ok(CellLayout {
            a,
            b,
            n: count,
            mean,
            w,
            ss_within,
            levels_a: fa.n_levels(),
            levels_b: fb.n_levels(),
        })
    }

    fn n_cells(&self) -> usize {
        self.a.len()
    }
}

/// A term of the two-way model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Term {
    Intercept,
    A,
    B,
    Ab,
}

/// Contrasts of the factor columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Coding {
    /// R's default `contr.treatment`: the first level is the baseline.
    Treatment,
    /// `contr.sum`: the last level is minus the sum of the others.
    Sum,
}

/// One design column, as a function of a cell's levels.
#[derive(Debug, Clone, Copy)]
enum Column {
    One,
    /// Indicator of level `j` of A (treatment coding).
    A(u32),
    /// Indicator of level `j` of B (treatment coding).
    B(u32),
    /// Indicator of one cell (treatment-coded interaction; the column of an empty cell is
    /// zero, hence aliased, and never built).
    Cell(u32, u32),
    /// Sum code `j` of A.
    SumA(u32),
    /// Sum code `j` of B.
    SumB(u32),
    /// Product of sum code `i` of A and sum code `j` of B.
    SumAb(u32, u32),
}

/// `contr.sum` code `j` of `level` among `levels` levels.
fn sum_code(level: u32, j: u32, levels: usize) -> f64 {
    if level == j {
        1.0
    } else if level as usize + 1 == levels {
        -1.0
    } else {
        0.0
    }
}

impl Column {
    fn value(self, a: u32, b: u32, la: usize, lb: usize) -> f64 {
        let ind = |t: bool| if t { 1.0 } else { 0.0 };
        match self {
            Column::One => 1.0,
            Column::A(j) => ind(a == j),
            Column::B(j) => ind(b == j),
            Column::Cell(i, j) => ind(a == i && b == j),
            Column::SumA(j) => sum_code(a, j, la),
            Column::SumB(j) => sum_code(b, j, lb),
            Column::SumAb(i, j) => sum_code(a, i, la) * sum_code(b, j, lb),
        }
    }
}

/// The design columns of `term`.
fn term_columns(cells: &CellLayout, term: Term, coding: Coding) -> EconResult<Vec<Column>> {
    let (la, lb) = (cells.levels_a as u32, cells.levels_b as u32);
    let cols = match (term, coding) {
        (Term::Intercept, _) => vec![Column::One],
        (Term::A, Coding::Treatment) => (1..la).map(Column::A).collect(),
        (Term::B, Coding::Treatment) => (1..lb).map(Column::B).collect(),
        (Term::A, Coding::Sum) => (0..la - 1).map(Column::SumA).collect(),
        (Term::B, Coding::Sum) => (0..lb - 1).map(Column::SumB).collect(),
        (Term::Ab, Coding::Treatment) => cells
            .a
            .iter()
            .zip(&cells.b)
            .filter(|&(&i, &j)| i >= 1 && j >= 1)
            .map(|(&i, &j)| Column::Cell(i, j))
            .collect(),
        (Term::Ab, Coding::Sum) => {
            // Only built for a complete layout (Type III refuses empty cells first), so
            // (la - 1)(lb - 1) < la * lb = number of cells.
            let len = resource::checked_len(
                &[cells.levels_a - 1, cells.levels_b - 1],
                "two-way ANOVA interaction columns",
            )?;
            let mut v = resource::try_with_capacity(len, "two-way ANOVA interaction columns")?;
            for j in 0..lb - 1 {
                for i in 0..la - 1 {
                    v.push(Column::SumAb(i, j));
                }
            }
            v
        }
    };
    Ok(cols)
}

/// Sum of squares and df a term adds to a sequence of nested models.
#[derive(Debug, Clone, Copy, Default)]
struct TermFit {
    ss: f64,
    df: usize,
}

/// The least-squares fit of a sequence of terms.
struct SequenceFit {
    /// Per term, in sequence order.
    terms: Vec<TermFit>,
    /// Rank of the design.
    rank: usize,
    /// Number of design columns.
    n_cols: usize,
    /// Residual sum of squares of the full sequence (within-cell SS included).
    rss: f64,
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

/// Fit the terms in order on the weighted cell means: Gram-Schmidt with
/// re-orthogonalisation (two projection passes) over the design columns, R's aliasing
/// rule, and each term's SS as the squared projections of the response on its new
/// orthonormal directions. The weighted cell regression has the same normal equations as
/// the regression on the rows, so its SS and rank are those of `lm` on the data.
fn fit_sequence(cells: &CellLayout, terms: &[Term], coding: Coding) -> EconResult<SequenceFit> {
    let c = cells.n_cells();
    let mut columns = Vec::with_capacity(terms.len());
    let mut n_cols = 0usize;
    for &term in terms {
        let cols = term_columns(cells, term, coding)?;
        n_cols = n_cols
            .checked_add(cols.len())
            .ok_or_else(|| EconError::ResourceLimit("two-way ANOVA design".to_string()))?;
        columns.push(cols);
    }
    let max_rank = n_cols.min(c);
    let q_len = resource::checked_len(&[c, max_rank], "two-way ANOVA design")?;
    let mut q: Vec<f64> = resource::try_with_capacity(q_len, "two-way ANOVA design")?;
    let mut x: Vec<f64> = resource::try_vec(c, 0.0, "two-way ANOVA design column")?;
    // Residual of the weighted cell means on the columns accepted so far.
    let mut e: Vec<f64> = cells
        .mean
        .iter()
        .zip(&cells.w)
        .map(|(m, w)| m * w)
        .collect();
    let mut rank = 0usize;
    let mut fits = Vec::with_capacity(terms.len());
    for cols in &columns {
        let mut fit = TermFit::default();
        for &col in cols {
            resource::checkpoint()?;
            if rank == max_rank {
                continue; // the cell space is spanned: every further column is aliased
            }
            for (k, xk) in x.iter_mut().enumerate() {
                *xk =
                    cells.w[k] * col.value(cells.a[k], cells.b[k], cells.levels_a, cells.levels_b);
            }
            let norm_x = dot(&x, &x).sqrt();
            if norm_x == 0.0 {
                continue;
            }
            for _pass in 0..2 {
                for qj in q.chunks_exact(c) {
                    let d = dot(qj, &x);
                    for (xk, qk) in x.iter_mut().zip(qj) {
                        *xk -= d * qk;
                    }
                }
            }
            let norm_r = dot(&x, &x).sqrt();
            if !(norm_r >= ALIAS_TOL * norm_x) {
                continue; // aliased (or NaN)
            }
            for xk in x.iter_mut() {
                *xk /= norm_r;
            }
            let coef = dot(&x, &e);
            for (ek, xk) in e.iter_mut().zip(&x) {
                *ek -= coef * xk;
            }
            fit.ss += coef * coef;
            fit.df += 1;
            q.extend_from_slice(&x);
            rank += 1;
        }
        fits.push(fit);
    }
    Ok(SequenceFit {
        terms: fits,
        rank,
        n_cols,
        rss: cells.ss_within + dot(&e, &e),
    })
}

/// Generate ANOVA table from an OLS regression result.
///
/// This creates an ANOVA decomposition showing how much of the variance
/// is explained by the model vs. residual error. Equivalent to R's
/// `anova(lm(...))` for a fitted linear model.
///
/// # Arguments
/// * `ols_result` - A fitted OLS regression result from `run_ols`
///
/// # Returns
/// A simplified ANOVA table for the regression model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnovaResult {
    /// Model (explained) sum of squares
    pub ss_model: f64,
    /// Residual (error) sum of squares
    pub ss_residual: f64,
    /// Total sum of squares
    pub ss_total: f64,
    /// Model degrees of freedom
    pub df_model: usize,
    /// Residual degrees of freedom
    pub df_residual: usize,
    /// Total degrees of freedom
    pub df_total: usize,
    /// Mean square model
    pub ms_model: f64,
    /// Mean square residual
    pub ms_residual: f64,
    /// F-statistic
    pub f_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Significance
    pub significance: SignificanceLevel,
}

impl std::fmt::Display for RegressionAnovaResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Analysis of Variance Table")?;
        writeln!(
            f,
            "═══════════════════════════════════════════════════════════"
        )?;
        writeln!(
            f,
            "{:>12} {:>12} {:>6} {:>12} {:>10} {:>10}",
            "Source", "SS", "DF", "MS", "F", "Pr(>F)"
        )?;
        writeln!(
            f,
            "───────────────────────────────────────────────────────────"
        )?;
        writeln!(
            f,
            "{:>12} {:>12.4} {:>6} {:>12.4} {:>10.4} {:>10.4} {}",
            "Model",
            self.ss_model,
            self.df_model,
            self.ms_model,
            self.f_statistic,
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(
            f,
            "{:>12} {:>12.4} {:>6} {:>12.4}",
            "Residual", self.ss_residual, self.df_residual, self.ms_residual
        )?;
        writeln!(
            f,
            "───────────────────────────────────────────────────────────"
        )?;
        writeln!(
            f,
            "{:>12} {:>12.4} {:>6}",
            "Total", self.ss_total, self.df_total
        )?;
        Ok(())
    }
}

/// Create ANOVA table from OLS result.
///
/// Takes a fitted OLS model and produces an ANOVA decomposition.
///
/// # Example
/// ```ignore
/// let ols = run_ols(&dataset, "y", &["x1", "x2"], true, CovarianceType::Standard)?;
/// let anova = anova_from_ols(&ols);
/// println!("{}", anova);
/// ```
pub fn anova_from_ols(ols: &crate::regression::OlsResult) -> RegressionAnovaResult {
    // Use cached values from OLS result
    let ss_residual = ols.ssr;
    let ss_total = ols.sst;
    let ss_model = ss_total - ss_residual;

    let df_model = ols.df_model;
    let df_residual = ols.df_resid;
    let df_total = ols.n_obs - 1;

    let ms_model = if df_model > 0 {
        ss_model / df_model as f64
    } else {
        0.0
    };
    let ms_residual = if df_residual > 0 {
        ss_residual / df_residual as f64
    } else {
        f64::NAN
    };

    let f_statistic = ols.f_statistic;
    let p_value = ols.f_p_value;

    RegressionAnovaResult {
        ss_model,
        ss_residual,
        ss_total,
        df_model,
        df_residual,
        df_total,
        ms_model,
        ms_residual,
        f_statistic,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use polars::prelude::*;

    fn create_one_way_test_data() -> Dataset {
        // Classic fertilizer experiment data
        // Group A: mean ~10, Group B: mean ~15, Group C: mean ~20
        let df = df! {
            "yield" => [9.5, 10.2, 10.8, 9.8, 10.5,   // Group A
                        14.2, 15.5, 14.8, 15.2, 15.8,  // Group B
                        19.5, 20.2, 19.8, 20.5, 20.8], // Group C
            "fertilizer" => ["A", "A", "A", "A", "A",
                            "B", "B", "B", "B", "B",
                            "C", "C", "C", "C", "C"]
        }
        .unwrap();
        Dataset::new(df)
    }

    fn create_two_way_test_data() -> Dataset {
        // 2x2 factorial design
        let df = df! {
            "yield" => [10.0, 11.0, 12.0, 15.0, 16.0, 17.0,  // A-Low
                        20.0, 21.0, 22.0, 30.0, 31.0, 32.0], // A-High, B-Low, B-High
            "fertilizer" => ["A", "A", "A", "B", "B", "B",
                            "A", "A", "A", "B", "B", "B"],
            "water" => ["Low", "Low", "Low", "Low", "Low", "Low",
                       "High", "High", "High", "High", "High", "High"]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_one_way_anova_basic() {
        let dataset = create_one_way_test_data();
        let result = run_one_way_anova(&dataset, "yield", "fertilizer").unwrap();

        assert_eq!(result.n_obs, 15);
        assert_eq!(result.n_groups, 3);
        assert_eq!(result.df_between, 2);
        assert_eq!(result.df_within, 12);

        // Strong effect expected
        assert!(result.f_statistic > 100.0);
        assert!(result.p_value < 0.001);
        assert!(result.eta_squared > 0.9);
    }

    #[test]
    fn test_one_way_anova_ss_decomposition() {
        let dataset = create_one_way_test_data();
        let result = run_one_way_anova(&dataset, "yield", "fertilizer").unwrap();

        // SST = SSB + SSW
        let ss_diff = (result.ss_total - result.ss_between - result.ss_within).abs();
        assert!(
            ss_diff < 1e-10,
            "SS decomposition failed: diff = {}",
            ss_diff
        );
    }

    #[test]
    fn test_two_way_anova_additive() {
        let dataset = create_two_way_test_data();
        let result = run_two_way_anova(&dataset, "yield", "fertilizer", "water", false).unwrap();

        assert_eq!(result.levels_a, 2);
        assert_eq!(result.levels_b, 2);
        assert!(!result.with_interaction);
        assert!(result.ss_ab.is_none());
    }

    #[test]
    fn test_two_way_anova_with_interaction() {
        let dataset = create_two_way_test_data();
        let result = run_two_way_anova(&dataset, "yield", "fertilizer", "water", true).unwrap();

        assert!(result.with_interaction);
        assert!(result.ss_ab.is_some());
        assert!(result.df_ab.is_some());
    }

    #[test]
    fn test_anova_column_not_found() {
        let dataset = create_one_way_test_data();
        let result = run_one_way_anova(&dataset, "nonexistent", "fertilizer");
        assert!(matches!(result, Err(EconError::ColumnNotFound { .. })));
    }

    #[test]
    fn test_anova_insufficient_groups() {
        let df = df! {
            "yield" => [1.0, 2.0, 3.0],
            "group" => ["A", "A", "A"]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_one_way_anova(&dataset, "yield", "group");
        assert!(matches!(result, Err(EconError::InsufficientData { .. })));
    }

    // ========================================================================
    // Validation Tests Against R
    // ========================================================================
    //
    // These tests compare results with R's aov() function.
    // Expected values from: validation/scripts/anova_validation.R

    #[test]
    fn test_validate_one_way_anova_against_r() {
        // Test data matches R validation script exactly
        let dataset = create_one_way_test_data();
        let result = run_one_way_anova(&dataset, "yield", "fertilizer").unwrap();

        // R expected values from aov(yield ~ fertilizer, data = data1)
        let expected_ss_between = 250.012;
        let expected_ss_within = 3.744;
        let expected_df_between = 2;
        let expected_df_within = 12;
        let expected_ms_between = 125.006;
        let expected_ms_within = 0.312;
        let expected_f = 400.660256;
        // R gives p-value as 1.03161374845346e-11
        let expected_eta_sq = 0.985245669;

        // Validate sum of squares (tolerance 1e-3)
        assert!(
            (result.ss_between - expected_ss_between).abs() < 1e-3,
            "SS Between mismatch: Rust={}, R={}",
            result.ss_between,
            expected_ss_between
        );
        assert!(
            (result.ss_within - expected_ss_within).abs() < 1e-3,
            "SS Within mismatch: Rust={}, R={}",
            result.ss_within,
            expected_ss_within
        );

        // Validate degrees of freedom (exact match)
        assert_eq!(
            result.df_between, expected_df_between,
            "DF Between mismatch"
        );
        assert_eq!(result.df_within, expected_df_within, "DF Within mismatch");

        // Validate mean squares (tolerance 1e-3)
        assert!(
            (result.ms_between - expected_ms_between).abs() < 1e-3,
            "MS Between mismatch: Rust={}, R={}",
            result.ms_between,
            expected_ms_between
        );
        assert!(
            (result.ms_within - expected_ms_within).abs() < 1e-3,
            "MS Within mismatch: Rust={}, R={}",
            result.ms_within,
            expected_ms_within
        );

        // Validate F-statistic (tolerance 1e-3)
        assert!(
            (result.f_statistic - expected_f).abs() < 1e-3,
            "F-statistic mismatch: Rust={}, R={}",
            result.f_statistic,
            expected_f
        );

        // Validate p-value is very small (both should be < 1e-10)
        assert!(
            result.p_value < 1e-10,
            "P-value should be < 1e-10, got {}",
            result.p_value
        );

        // Validate eta-squared (tolerance 1e-4)
        assert!(
            (result.eta_squared - expected_eta_sq).abs() < 1e-4,
            "Eta-squared mismatch: Rust={}, R={}",
            result.eta_squared,
            expected_eta_sq
        );

        // Validate grand mean (R reports 15.14)
        assert!(
            (result.grand_mean - 15.14).abs() < 1e-6,
            "Grand mean mismatch: Rust={}, R=15.14",
            result.grand_mean
        );
    }

    #[test]
    fn test_validate_two_way_anova_against_r() {
        // Test data matches R validation script exactly
        let dataset = create_two_way_test_data();
        let result = run_two_way_anova(&dataset, "yield", "fertilizer", "water", true).unwrap();

        // R expected values from aov(yield ~ fertilizer * water, data = data2)
        // Note: R reports SS_A = 168.75, SS_B = 468.75, SS_AB = 18.75, SS_error = 8.0

        // Validate degrees of freedom (exact)
        assert_eq!(result.df_a, 1, "DF Factor A mismatch");
        assert_eq!(result.df_b, 1, "DF Factor B mismatch");
        assert_eq!(result.df_ab.unwrap(), 1, "DF Interaction mismatch");
        assert_eq!(result.df_error, 8, "DF Error mismatch");

        // Validate F-statistics
        // R: F_water = 468.75, F_interaction = 18.75
        assert!(
            (result.f_b - 468.75).abs() < 0.1,
            "F Factor B mismatch: Rust={}, R=468.75",
            result.f_b
        );
        assert!(
            (result.f_ab.unwrap() - 18.75).abs() < 0.1,
            "F Interaction mismatch: Rust={}, R=18.75",
            result.f_ab.unwrap()
        );

        // Validate p-values are significant
        assert!(result.p_a < 0.001, "Factor A should be significant");
        assert!(result.p_b < 0.001, "Factor B should be significant");
        assert!(
            result.p_ab.unwrap() < 0.01,
            "Interaction should be significant"
        );

        // Validate SS decomposition holds
        let ss_total_computed = result.ss_a + result.ss_b + result.ss_ab.unwrap() + result.ss_error;
        assert!(
            (result.ss_total - ss_total_computed).abs() < 1e-6,
            "SS decomposition failed"
        );
    }

    // ========================================================================
    // R-pinned two-way ANOVA on unbalanced and incomplete layouts (T2-WP07)
    // ========================================================================
    //
    // Case stat_anova_two_way_unbalanced (validation/r/cases/stats_anova.R): a 2 x 2 layout
    // with cells of 2/4/4/2 rows plus a row whose y and fa are missing, and a 3 x 2 layout
    // (y3 ~ fa3 * fb3) whose (C, L) cell is empty. R: aov (Type I), car::Anova 3.1.5 (Types II
    // and III, the latter with contr.sum).

    use crate::test_support::load_case;
    use p2a_validation::{Check, Tol};

    fn two_way_with(
        ds: &Dataset,
        y: &str,
        a: &str,
        b: &str,
        interaction: bool,
        ss_type: SsType,
    ) -> EconResult<TwoWayAnovaResult> {
        run_two_way_anova_with(
            ds,
            y,
            a,
            b,
            &TwoWayAnovaConfig {
                interaction,
                ss_type,
            },
        )
    }

    /// Check `<prefix>_ss`, `_df`, `_f`, `_p` of one term (df exact).
    fn check_term(
        check: &mut Check<'_>,
        prefix: &str,
        (ss, df, f, p): (f64, usize, f64, f64),
        n: usize,
    ) {
        check
            .scalar(&format!("{prefix}_ss"), ss, Tol::stat(n))
            .scalar(&format!("{prefix}_df"), df as f64, Tol::exact())
            .scalar(&format!("{prefix}_f"), f, Tol::stat(n))
            .scalar(&format!("{prefix}_p"), p, Tol::p_value(n));
    }

    fn term_a(r: &TwoWayAnovaResult) -> (f64, usize, f64, f64) {
        (r.ss_a, r.df_a, r.f_a, r.p_a)
    }

    fn term_b(r: &TwoWayAnovaResult) -> (f64, usize, f64, f64) {
        (r.ss_b, r.df_b, r.f_b, r.p_b)
    }

    fn term_ab(r: &TwoWayAnovaResult) -> (f64, usize, f64, f64) {
        (
            r.ss_ab.expect("interaction SS"),
            r.df_ab.expect("interaction df"),
            r.f_ab.expect("interaction F"),
            r.p_ab.expect("interaction p"),
        )
    }

    fn assert_type1_adds_up(r: &TwoWayAnovaResult) {
        let parts = r.ss_a + r.ss_b + r.ss_ab.unwrap_or(0.0) + r.ss_error;
        assert!(
            (parts - r.ss_total).abs() <= 1e-10 * r.ss_total,
            "Type I SS {parts} do not add up to the total {}",
            r.ss_total
        );
    }

    #[test]
    fn test_validate_two_way_anova_unbalanced_type1() {
        let (case, ds) = load_case("stat_anova_two_way_unbalanced");
        let int = run_two_way_anova(&ds, "y", "fa", "fb", true).unwrap();
        let add = run_two_way_anova(&ds, "y", "fa", "fb", false).unwrap();
        let n = int.n_obs;
        let mut check = Check::new(&case);
        check_term(&mut check, "int_fa", term_a(&int), n);
        check_term(&mut check, "int_fb", term_b(&int), n);
        check_term(&mut check, "int_fa_x_fb", term_ab(&int), n);
        check
            .scalar("int_resid_ss", int.ss_error, Tol::stat(n))
            .scalar("int_resid_df", int.df_error as f64, Tol::exact())
            .scalar("add_fa_ss", add.ss_a, Tol::stat(n))
            .scalar("add_fa_df", add.df_a as f64, Tol::exact())
            .scalar("add_fa_f", add.f_a, Tol::stat(n))
            .scalar("add_fa_p", add.p_a, Tol::p_value(n))
            .scalar("add_fb_ss", add.ss_b, Tol::stat(n))
            .scalar("add_fb_df", add.df_b as f64, Tol::exact())
            .scalar("add_fb_f", add.f_b, Tol::stat(n))
            .scalar("add_fb_p", add.p_b, Tol::p_value(n))
            .scalar("add_resid_ss", add.ss_error, Tol::stat(n))
            .scalar("add_resid_df", add.df_error as f64, Tol::exact())
            .scalar("n_used", int.n_obs as f64, Tol::exact())
            .scalar("n_dropped", int.missing.n_dropped as f64, Tol::exact())
            .finish();
        assert_eq!(int.ss_type, SsType::I);
        assert_eq!(
            int.missing.by_column,
            vec![("y".to_string(), 1), ("fa".to_string(), 1)]
        );
        assert_eq!(int.levels_a, 2);
        assert!(add.ss_ab.is_none() && add.df_ab.is_none());
        assert_type1_adds_up(&int);
        assert_type1_adds_up(&add);
    }

    #[test]
    fn test_validate_two_way_anova_unbalanced_type2() {
        let (case, ds) = load_case("stat_anova_two_way_unbalanced");
        let int = two_way_with(&ds, "y", "fa", "fb", true, SsType::II).unwrap();
        let add = two_way_with(&ds, "y", "fa", "fb", false, SsType::II).unwrap();
        let n = int.n_obs;
        let (s, p) = (Tol::stat(n), Tol::p_value(n));
        let (ss_ab, _, f_ab, p_ab) = term_ab(&int);
        Check::new(&case)
            .scalar("type2_fa_ss", int.ss_a, s)
            .scalar("type2_fa_f", int.f_a, s)
            .scalar("type2_fa_p", int.p_a, p)
            .scalar("type2_fb_ss", int.ss_b, s)
            .scalar("type2_fb_f", int.f_b, s)
            .scalar("type2_fb_p", int.p_b, p)
            .scalar("type2_int_ss", ss_ab, s)
            .scalar("type2_int_f", f_ab, s)
            .scalar("type2_int_p", p_ab, p)
            .scalar("type2_resid_ss", int.ss_error, s)
            .scalar("type2_resid_df", int.df_error as f64, Tol::exact())
            .scalar("add_type2_fa_ss", add.ss_a, s)
            .scalar("add_type2_fa_f", add.f_a, s)
            .scalar("add_type2_fa_p", add.p_a, p)
            .scalar("add_type2_fb_ss", add.ss_b, s)
            .scalar("add_type2_fb_f", add.f_b, s)
            .scalar("add_type2_fb_p", add.p_b, p)
            .finish();
        assert_eq!(int.ss_type, SsType::II);
        assert_eq!((int.df_a, int.df_b), (1, 1));
        assert_eq!((add.df_a, add.df_b), (1, 1));
    }

    #[test]
    fn test_validate_two_way_anova_unbalanced_type3() {
        let (case, ds) = load_case("stat_anova_two_way_unbalanced");
        let int = two_way_with(&ds, "y", "fa", "fb", true, SsType::III).unwrap();
        let add = two_way_with(&ds, "y", "fa", "fb", false, SsType::III).unwrap();
        let n = int.n_obs;
        let (s, p) = (Tol::stat(n), Tol::p_value(n));
        let (ss_ab, _, f_ab, p_ab) = term_ab(&int);
        Check::new(&case)
            .scalar("type3_fa_ss", int.ss_a, s)
            .scalar("type3_fa_f", int.f_a, s)
            .scalar("type3_fa_p", int.p_a, p)
            .scalar("type3_fb_ss", int.ss_b, s)
            .scalar("type3_fb_f", int.f_b, s)
            .scalar("type3_fb_p", int.p_b, p)
            .scalar("type3_int_ss", ss_ab, s)
            .scalar("type3_int_f", f_ab, s)
            .scalar("type3_int_p", p_ab, p)
            .scalar("add_type3_fa_ss", add.ss_a, s)
            .scalar("add_type3_fa_f", add.f_a, s)
            .scalar("add_type3_fb_ss", add.ss_b, s)
            .scalar("add_type3_fb_f", add.f_b, s)
            .finish();
        assert_eq!(int.ss_type, SsType::III);
        assert_eq!(add.ss_type, SsType::III);
    }

    #[test]
    fn test_validate_two_way_anova_unbalanced_empty_cell() {
        let (case, ds) = load_case("stat_anova_two_way_unbalanced");
        let (s, p, x) = (Tol::stat(13), Tol::p_value(13), Tol::exact());
        let mut check = Check::new(&case);

        // 3 x 2 with the (C, L) cell empty: the interaction keeps 1 df, the residual 8.
        let e1 = run_two_way_anova(&ds, "y3", "fa3", "fb3", true).unwrap();
        check_term(&mut check, "empty_fa3", term_a(&e1), 13);
        check_term(&mut check, "empty_fb3", term_b(&e1), 13);
        check_term(&mut check, "empty_fa3_x_fb3", term_ab(&e1), 13);
        check.scalar("empty_resid_ss", e1.ss_error, s).scalar(
            "empty_resid_df",
            e1.df_error as f64,
            x,
        );

        // Type II by model comparison, as car::Anova after "model has aliased coefficients".
        let e2 = two_way_with(&ds, "y3", "fa3", "fb3", true, SsType::II).unwrap();
        check
            .scalar("empty_type2_fa3_ss", e2.ss_a, s)
            .scalar("empty_type2_fa3_df", e2.df_a as f64, x)
            .scalar("empty_type2_fa3_f", e2.f_a, s)
            .scalar("empty_type2_fa3_p", e2.p_a, p)
            .scalar("empty_type2_fb3_ss", e2.ss_b, s)
            .scalar("empty_type2_int_ss", e2.ss_ab.expect("interaction SS"), s)
            .scalar(
                "empty_type2_int_df",
                e2.df_ab.expect("interaction df") as f64,
                x,
            );

        // Type III is not defined with aliased coefficients (car::Anova refuses too).
        match two_way_with(&ds, "y3", "fa3", "fb3", true, SsType::III) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("aliased"), "{message}")
            }
            other => panic!("Type III with an empty cell must be refused, got {other:?}"),
        }

        // The 2 x 2 layout without its (B, H) rows: the interaction has no df, and aov has no
        // fa:fb row. Row 13 (fa missing) stays in the frame and is dropped by the extractor.
        let df = ds.df();
        let fa = df.column("fa").unwrap().str().unwrap().clone();
        let fb = df.column("fb").unwrap().str().unwrap().clone();
        let mask: BooleanChunked = fa
            .iter()
            .zip(fb.iter())
            .map(|(a, b)| Some(!(a == Some("B") && b == Some("H"))))
            .collect();
        let nocell = Dataset::new(df.filter(&mask).unwrap());
        let nc = run_two_way_anova(&nocell, "y", "fa", "fb", true).unwrap();
        check
            .scalar("nocell_fa_ss", nc.ss_a, s)
            .scalar("nocell_fa_f", nc.f_a, s)
            .scalar("nocell_fa_p", nc.p_a, p)
            .scalar("nocell_fb_ss", nc.ss_b, s)
            .scalar("nocell_fb_f", nc.f_b, s)
            .scalar("nocell_fb_p", nc.p_b, p)
            .scalar("nocell_resid_ss", nc.ss_error, s)
            .scalar("nocell_resid_df", nc.df_error as f64, x)
            .scalar("nocell_n_used", nc.n_obs as f64, x)
            .finish();
        assert!(nc.with_interaction);
        assert_eq!(nc.df_ab, None);
        assert_eq!(nc.ss_ab, None);
        assert_eq!(nc.f_ab, None);
        assert_eq!(nc.p_ab, None);
        assert!(nc.ms_ab.is_none() && nc.sig_ab.is_none());
        assert!(nc.to_string().contains("not estimable"), "{nc}");
    }

    /// A 10 x 10 layout with 30 non-empty cells (2 rows each): the df come from the rank, as
    /// R aov gives them (a 9, b 9, a:b 11, residuals 30). Fails before with a usize underflow
    /// of n - levels_a * levels_b (X-7).
    #[test]
    fn test_two_way_anova_sparse_layout_df_from_rank() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut y = Vec::new();
        for i in 0..30i64 {
            for k in 0..2i64 {
                let (ai, bi) = (i % 10, i / 3);
                a.push(ai);
                b.push(bi);
                y.push(1.0 + 0.1 * ai as f64 + 0.05 * bi as f64 + ((i + 7 * k) as f64).sin());
            }
        }
        let ds = Dataset::new(df! { "y" => y, "a" => a, "b" => b }.unwrap());
        let r = run_two_way_anova(&ds, "y", "a", "b", true).unwrap();
        assert_eq!((r.df_a, r.df_b, r.df_ab, r.df_error), (9, 9, Some(11), 30));
        assert_eq!((r.levels_a, r.levels_b, r.n_obs), (10, 10, 60));
        assert_type1_adds_up(&r);
    }

    #[test]
    fn test_ss_type_parse_and_display() {
        let parsed: Vec<SsType> = ["I", " ii ", "III", "2", "Type 3"]
            .iter()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(
            parsed,
            [SsType::I, SsType::II, SsType::III, SsType::II, SsType::III]
        );
        match "IV".parse::<SsType>() {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(
                    message.contains("\"II\"") && message.contains("'IV'"),
                    "{message}"
                )
            }
            other => panic!("IV must be refused, got {other:?}"),
        }
        assert_eq!(SsType::III.to_string(), "III");
        assert_eq!(SsType::I.to_string(), "I");
    }

    // ========================================================================
    // R-pinned one-way ANOVA with missing responses and labels (T2-WP07)
    // ========================================================================
    //
    // Case stat_oneway_na_labels: y (Int64) is missing in 3 of 9 rows and g in 2, so aov
    // uses 5 rows; groups a, b, c in R factor() order, labels without quotes.

    #[test]
    fn test_validate_oneway_na_labels_aov() {
        let (case, ds) = load_case("stat_oneway_na_labels");
        let r = run_one_way_anova(&ds, "y", "g").unwrap();
        let names: Vec<&str> = r.groups.iter().map(|g| g.group.as_str()).collect();
        let sizes: Vec<f64> = r.groups.iter().map(|g| g.n as f64).collect();
        let means: Vec<f64> = r.groups.iter().map(|g| g.mean).collect();
        let (s, p, x) = (Tol::stat(5), Tol::p_value(5), Tol::exact());
        Check::new(&case)
            .scalar("aov_f", r.f_statistic, s)
            .scalar("aov_p", r.p_value, p)
            .scalar("aov_df_between", r.df_between as f64, x)
            .scalar("aov_df_within", r.df_within as f64, x)
            .scalar("aov_ss_between", r.ss_between, s)
            .scalar("aov_ss_within", r.ss_within, s)
            .scalar("aov_n_used", r.n_obs as f64, x)
            .scalar("aov_grand_mean", r.grand_mean, s)
            .named("aov_group_n", &names, &sizes, x)
            .named("aov_group_means", &names, &means, s)
            .finish();
        assert_eq!(names, ["a", "b", "c"]);
        assert_eq!(r.missing.n_dropped, 4);
        assert_eq!(
            r.missing.by_column,
            vec![("y".to_string(), 3), ("g".to_string(), 2)]
        );
        assert!(
            r.to_string()
                .contains("Missing values: 4 of 9 rows dropped because of missing values"),
            "{r}"
        );
    }

    /// Design 2.4 fixture: an Int64 response equals its Float64 cast; a text response and an
    /// absent factor are refused with the variant that names the problem.
    #[test]
    fn test_validate_na_one_way_anova_dtypes_and_errors() {
        let (_case, ds) = load_case("stat_oneway_na_labels");
        assert_eq!(ds.df().column("y").unwrap().dtype(), &DataType::Int64);
        let int = run_one_way_anova(&ds, "y", "g").unwrap();
        let mut df = ds.df().clone();
        let y = df.column("y").unwrap().cast(&DataType::Float64).unwrap();
        df.with_column(y).unwrap();
        let float = run_one_way_anova(&Dataset::new(df), "y", "g").unwrap();
        assert_eq!(int.f_statistic, float.f_statistic);
        assert_eq!(int.ss_between, float.ss_between);
        assert_eq!(int.ss_within, float.ss_within);
        assert!(matches!(
            run_one_way_anova(&ds, "g", "id"),
            Err(EconError::NonNumericColumn { ref column }) if column == "g"
        ));
        assert!(matches!(
            run_one_way_anova(&ds, "y", "nope"),
            Err(EconError::ColumnNotFound { ref column, .. }) if column == "nope"
        ));
    }

    /// Design 2.4 fixture of the two-way wrapper, on the Int64 case stat_friedman_quade:
    /// `y ~ treat + block` on the Int64 response equals the same model on its Float64 cast;
    /// a text response, an absent factor and a factor given twice are refused.
    #[test]
    fn test_validate_na_two_way_anova_dtypes_and_errors() {
        let (_case, ds) = load_case("stat_friedman_quade");
        assert_eq!(ds.df().column("y").unwrap().dtype(), &DataType::Int64);
        let int = run_two_way_anova(&ds, "y", "treat", "block", false).unwrap();
        let mut df = ds.df().clone();
        let y = df.column("y").unwrap().cast(&DataType::Float64).unwrap();
        df.with_column(y).unwrap();
        let float = run_two_way_anova(&Dataset::new(df), "y", "treat", "block", false).unwrap();
        assert_eq!(int.ss_a, float.ss_a);
        assert_eq!(int.ss_b, float.ss_b);
        assert_eq!(int.ss_error, float.ss_error);
        assert_eq!(int.f_a, float.f_a);
        assert_eq!(int.f_b, float.f_b);
        assert!(matches!(
            run_two_way_anova(&ds, "treat", "treat", "block", false),
            Err(EconError::NonNumericColumn { ref column }) if column == "treat"
        ));
        assert!(matches!(
            run_two_way_anova(&ds, "y", "treat", "nope", false),
            Err(EconError::ColumnNotFound { ref column, .. }) if column == "nope"
        ));
        assert!(matches!(
            run_two_way_anova(&ds, "y", "block", "block", true),
            Err(EconError::InvalidSpecification { .. })
        ));
    }
}
