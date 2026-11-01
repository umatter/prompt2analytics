//! Analysis of Variance (ANOVA) for comparing group means.
//!
//! One-way ANOVA by the sum-of-squares decomposition; two-way ANOVA by least
//! squares with Type I, II and III sums of squares ([`run_two_way_anova_with`]).
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
//! One-way ANOVA partitions total variation into components:
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
    /// Sum of Squares for Factor A (of the [`SsType`] in `ss_type`)
    pub ss_a: f64,
    /// Sum of Squares for Factor B (of the [`SsType`] in `ss_type`)
    pub ss_b: f64,
    /// Sum of Squares for the A×B interaction; `None` when the model has no interaction
    /// term or when empty cells leave it no df (aov omits the row)
    pub ss_ab: Option<f64>,
    /// Residual sum of squares of the fitted model
    pub ss_error: f64,
    /// Total Sum of Squares
    pub ss_total: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Degrees of Freedom
    // ═══════════════════════════════════════════════════════════════════════
    /// DF for factor A: the rank the term adds in the fitted sequence (levels - 1 unless
    /// aliased). 0 means the term is aliased with the other factor; `ms_a`, `f_a` and
    /// `p_a` are then NaN.
    pub df_a: usize,
    /// DF for factor B: the rank the term adds in the fitted sequence (levels - 1 unless
    /// aliased). 0 means the term is aliased with the other factor; `ms_b`, `f_b` and
    /// `p_b` are then NaN.
    pub df_b: usize,
    /// DF for the interaction: n_cells - rank(A + B); `None` without an interaction term
    /// or when empty cells leave it no df (aov omits the row)
    pub df_ab: Option<usize>,
    /// Residual DF: n - rank of the fitted model
    pub df_error: usize,
    /// Total DF
    pub df_total: usize,

    // ═══════════════════════════════════════════════════════════════════════
    // Mean Squares
    // ═══════════════════════════════════════════════════════════════════════
    /// Mean square of factor A (`ss_a / df_a`)
    pub ms_a: f64,
    /// Mean square of factor B (`ss_b / df_b`)
    pub ms_b: f64,
    /// Mean square of the interaction; `None` under the same condition as `df_ab`
    pub ms_ab: Option<f64>,
    /// Residual mean square (`ss_error / df_error`)
    pub ms_error: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // F-Statistics and P-Values
    // ═══════════════════════════════════════════════════════════════════════
    /// F statistic of factor A
    pub f_a: f64,
    /// P-value of factor A
    pub p_a: f64,
    /// Significance of factor A
    pub sig_a: SignificanceLevel,

    /// F statistic of factor B
    pub f_b: f64,
    /// P-value of factor B
    pub p_b: f64,
    /// Significance of factor B
    pub sig_b: SignificanceLevel,

    /// F statistic of the interaction; `None` under the same condition as `df_ab`
    pub f_ab: Option<f64>,
    /// P-value of the interaction; `None` under the same condition as `df_ab`
    pub p_ab: Option<f64>,
    /// Significance of the interaction; `None` under the same condition as `df_ab`
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

        // A main effect aliased with the other factor (df 0) has no row in aov either.
        let main_rows = [
            (
                &self.factor_a,
                self.ss_a,
                self.df_a,
                self.ms_a,
                self.f_a,
                self.p_a,
                self.sig_a,
            ),
            (
                &self.factor_b,
                self.ss_b,
                self.df_b,
                self.ms_b,
                self.f_b,
                self.p_b,
                self.sig_b,
            ),
        ];
        for (name, ss, df, ms, f_stat, p, sig) in main_rows {
            if df == 0 {
                writeln!(
                    f,
                    "{name:>15} not estimable (aliased with the other factor)"
                )?;
            } else {
                writeln!(
                    f,
                    "{:>15} {:>12.4} {:>6} {:>12.4} {:>10.4} {:>10.4} {}",
                    name,
                    ss,
                    df,
                    ms,
                    f_stat,
                    p,
                    sig.stars()
                )?;
            }
        }

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
/// Selects the table [`run_two_way_anova_with`] reports.
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

/// Options of [`run_two_way_anova_with`].
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
/// Each term's sum of squares is the reduction in the residual sum of squares when it
/// enters a sequence of nested least-squares models, and its df is the rank it adds (R's
/// `lm` aliasing rule: a design column whose residual on the earlier columns is below 1e-7
/// of its length is aliased):
///
/// - **Type I** (sequential, `aov`): A, then B given A, then A:B given both.
/// - **Type II** (`car::Anova(type = 2)`): A given B, B given A, A:B given both. With empty
///   cells the terms are still compared as nested models, as car does after its note that
///   the model has aliased coefficients.
/// - **Type III** (`car::Anova(type = 3)`, sum-to-zero contrasts): every term given all
///   others. It is refused with [`EconError::InvalidSpecification`] when the model has
///   aliased coefficients (empty cells, or an additive model whose factors are aliased), as
///   car refuses it.
///
/// The main effects are fitted on R's treatment-coded dummies. The interaction needs no
/// design columns: A + B + A:B spans the indicator of every non-empty cell, so the model
/// with the interaction is saturated (its fit is the cell means and its rank the number of
/// non-empty cells), and A:B given A and B is the residual the additive model leaves
/// between the cells. With the interaction, a Type III main effect is the Wald test of
/// equal unweighted marginal means, which is what its sum-to-zero coefficients test.
///
/// Empty cells reduce the interaction df (a 3 x 2 layout with one empty cell has 1); when
/// they leave the interaction no df, its fields are `None`, as `aov` drops that row. A main
/// effect aliased with the other factor (a disconnected layout, or one factor nested in the
/// other) has df 0. The residual df is `n - rank`. Rows with a missing response or factor
/// level are dropped (`aov`'s `na.omit`) and reported in `missing`; integer responses are
/// accepted.
///
/// The fit works on the cell means weighted by the square roots of the cell sizes, so it
/// costs O(n + c (La + Lb) r) for c non-empty cells, La and Lb levels and the additive
/// model's rank r; each design column is a cancellation point.
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
    let fit = two_way_terms(&cells, interaction, config.ss_type)?;
    let (term_a, term_b) = (fit.a, fit.b);

    let df_error = n.saturating_sub(fit.rank);
    let ss_error = fit.rss;
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
    let term_ab = fit.ab.filter(|t| t.df > 0);
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

/// A main-effect term of the two-way model (the interaction needs no design columns).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Term {
    Intercept,
    A,
    B,
}

/// One design column, as a function of a cell's levels (R's default `contr.treatment`: the
/// first level is the baseline).
#[derive(Debug, Clone, Copy)]
enum Column {
    One,
    /// Indicator of level `j` of A.
    A(u32),
    /// Indicator of level `j` of B.
    B(u32),
}

impl Column {
    fn value(self, a: u32, b: u32) -> f64 {
        let ind = |t: bool| if t { 1.0 } else { 0.0 };
        match self {
            Column::One => 1.0,
            Column::A(j) => ind(a == j),
            Column::B(j) => ind(b == j),
        }
    }
}

/// The design columns of `term`.
fn term_columns(cells: &CellLayout, term: Term) -> Vec<Column> {
    let (la, lb) = (cells.levels_a as u32, cells.levels_b as u32);
    match term {
        Term::Intercept => vec![Column::One],
        Term::A => (1..la).map(Column::A).collect(),
        Term::B => (1..lb).map(Column::B).collect(),
    }
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
    /// Residual sum of squares of the weighted cell means: what the fit leaves between the
    /// cells (the within-cell SS is not included).
    cell_rss: f64,
}

/// SS and df of A, B and A:B, and the rank and residual SS of the fitted model.
#[derive(Debug, Clone, Copy)]
struct TwoWayTerms {
    a: TermFit,
    b: TermFit,
    /// `None` without an interaction term; df 0 when empty cells leave it none.
    ab: Option<TermFit>,
    rank: usize,
    rss: f64,
}

/// The terms of the table `ss_type` selects (see [`run_two_way_anova_with`]).
///
/// Only the additive model is fitted, as (1, A, B) and, for A adjusted for B, (1, B, A).
/// With the interaction the model is saturated: rank = number of non-empty cells c,
/// residual = within-cell SS, and A:B adds the additive fit's between-cell residual on
/// `c - rank(1, A, B)` df.
fn two_way_terms(
    cells: &CellLayout,
    interaction: bool,
    ss_type: SsType,
) -> EconResult<TwoWayTerms> {
    let (levels_a, levels_b, c) = (cells.levels_a, cells.levels_b, cells.n_cells());
    let a_then_b = fit_sequence(cells, &[Term::Intercept, Term::A, Term::B])?;
    let b_then_a = || fit_sequence(cells, &[Term::Intercept, Term::B, Term::A]);
    let empty = levels_a
        .checked_mul(levels_b)
        .map_or(usize::MAX, |k| k.saturating_sub(c));
    let refuse = || EconError::InvalidSpecification {
        message: format!(
            "Type III sums of squares are not defined: the model has aliased \
             coefficients ({empty} empty cell(s) in {levels_a} x {levels_b}), as \
             car::Anova; use Type I or Type II"
        ),
    };
    let (a, b) = match ss_type {
        SsType::I => (a_then_b.terms[1], a_then_b.terms[2]),
        SsType::II => (b_then_a()?.terms[2], a_then_b.terms[2]),
        SsType::III if interaction => {
            // Only a complete layout has no aliased coefficients with the interaction.
            if empty > 0 {
                return Err(refuse());
            }
            (
                marginal_means_ss(&cells.a, levels_a, levels_b, cells)?,
                marginal_means_ss(&cells.b, levels_b, levels_a, cells)?,
            )
        }
        SsType::III => {
            // Additive: each main effect last is the Type II comparison; the contrasts do
            // not change the column space of (1, A, B).
            if a_then_b.rank < a_then_b.n_cols {
                return Err(refuse());
            }
            (b_then_a()?.terms[2], a_then_b.terms[2])
        }
    };
    let fit = if interaction {
        TwoWayTerms {
            a,
            b,
            ab: Some(TermFit {
                ss: a_then_b.cell_rss,
                df: c.saturating_sub(a_then_b.rank),
            }),
            rank: c,
            rss: cells.ss_within,
        }
    } else {
        TwoWayTerms {
            a,
            b,
            ab: None,
            rank: a_then_b.rank,
            rss: cells.ss_within + a_then_b.cell_rss,
        }
    };
    Ok(fit)
}

/// Type III SS of a main effect (level `codes` per cell, `levels` levels, `other` levels of
/// the other factor) in the model with the interaction on a complete layout.
///
/// It is the Wald form `(L mu)' (L D^-1 L')^-1 (L mu)` for equal unweighted marginal means,
/// with `mu` the cell means, `D = diag(n_c)` and `L` the contrasts of the marginal means
/// `m_l = sum_k mu_lk / other`: the hypothesis the factor's sum-to-zero coefficients test.
/// The `m_l` are independent with variances `s_l sigma^2`,
/// `s_l = sum_k 1 / (other^2 n_lk)`, so the form equals `sum_l (m_l - m_w)^2 / s_l` with `m_w`
/// the `1 / s_l`-weighted mean of the `m_l`: O(c + levels) for c cells.
fn marginal_means_ss(
    codes: &[u32],
    levels: usize,
    other: usize,
    cells: &CellLayout,
) -> EconResult<TermFit> {
    let mut mean_sum: Vec<f64> = resource::try_vec(levels, 0.0, "two-way ANOVA marginal means")?;
    let mut inv_n_sum: Vec<f64> = resource::try_vec(levels, 0.0, "two-way ANOVA marginal means")?;
    for ((&l, &m), &k) in codes.iter().zip(&cells.mean).zip(&cells.n) {
        mean_sum[l as usize] += m;
        inv_n_sum[l as usize] += 1.0 / k as f64;
    }
    let other = other as f64;
    // (m_l, 1 / s_l) per level.
    let marginal = |l: usize| (mean_sum[l] / other, other * other / inv_n_sum[l]);
    let (mut sum_w, mut sum_wm) = (0.0, 0.0);
    for l in 0..levels {
        let (m, w) = marginal(l);
        sum_w += w;
        sum_wm += w * m;
    }
    let m_w = sum_wm / sum_w;
    let ss = (0..levels)
        .map(|l| {
            let (m, w) = marginal(l);
            w * (m - m_w).powi(2)
        })
        .sum();
    Ok(TermFit { ss, df: levels - 1 })
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

/// Fit the main-effect terms in order (treatment coding).
fn fit_sequence(cells: &CellLayout, terms: &[Term]) -> EconResult<SequenceFit> {
    let columns: Vec<Vec<Column>> = terms.iter().map(|&t| term_columns(cells, t)).collect();
    fit_columns(cells, &columns, Column::value)
}

/// Fit the design columns of each term in order on the weighted cell means: Gram-Schmidt
/// with re-orthogonalisation (two projection passes), R's aliasing rule, and each term's SS
/// as the squared projections of the response on its new orthonormal directions. The
/// weighted cell regression has the same normal equations as the regression on the rows,
/// so its SS and rank are those of `lm` on the data. `value(column, a, b)` is a column's
/// entry for the cell with levels `a` and `b`.
fn fit_columns<C: Copy>(
    cells: &CellLayout,
    columns: &[Vec<C>],
    value: impl Fn(C, u32, u32) -> f64,
) -> EconResult<SequenceFit> {
    let c = cells.n_cells();
    let mut n_cols = 0usize;
    for cols in columns {
        n_cols = n_cols
            .checked_add(cols.len())
            .ok_or_else(|| EconError::ResourceLimit("two-way ANOVA design".to_string()))?;
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
    let mut fits = Vec::with_capacity(columns.len());
    for cols in columns {
        let mut fit = TermFit::default();
        for &col in cols {
            resource::checkpoint()?;
            if rank == max_rank {
                continue; // the cell space is spanned: every further column is aliased
            }
            for (k, xk) in x.iter_mut().enumerate() {
                *xk = cells.w[k] * value(col, cells.a[k], cells.b[k]);
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
            if norm_r.is_nan() || norm_r < ALIAS_TOL * norm_x {
                continue; // aliased (or not a number)
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
        cell_rss: dot(&e, &e),
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
    // Reference: the fit on explicit interaction columns
    // ========================================================================
    //
    // The interaction used to be fitted on design columns: one indicator per non-empty cell
    // (treatment coding, Types I and II), or the (La - 1)(Lb - 1) products of the sum codes
    // (Type III, every term sum-coded). That fit is kept here, on the same Gram-Schmidt
    // core, as the reference for the saturated-model and marginal-means shortcuts.

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum RefTerm {
        Intercept,
        A,
        B,
        Ab,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum RefCoding {
        Treatment,
        Sum,
    }

    #[derive(Debug, Clone, Copy)]
    enum RefColumn {
        One,
        A(u32),
        B(u32),
        Cell(u32, u32),
        SumA(u32),
        SumB(u32),
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

    fn ref_columns(cells: &CellLayout, term: RefTerm, coding: RefCoding) -> Vec<RefColumn> {
        let (la, lb) = (cells.levels_a as u32, cells.levels_b as u32);
        match (term, coding) {
            (RefTerm::Intercept, _) => vec![RefColumn::One],
            (RefTerm::A, RefCoding::Treatment) => (1..la).map(RefColumn::A).collect(),
            (RefTerm::B, RefCoding::Treatment) => (1..lb).map(RefColumn::B).collect(),
            (RefTerm::A, RefCoding::Sum) => (0..la - 1).map(RefColumn::SumA).collect(),
            (RefTerm::B, RefCoding::Sum) => (0..lb - 1).map(RefColumn::SumB).collect(),
            // Empty cells have zero columns (aliased), so they are not built.
            (RefTerm::Ab, RefCoding::Treatment) => cells
                .a
                .iter()
                .zip(&cells.b)
                .filter(|&(&i, &j)| i >= 1 && j >= 1)
                .map(|(&i, &j)| RefColumn::Cell(i, j))
                .collect(),
            (RefTerm::Ab, RefCoding::Sum) => (0..lb - 1)
                .flat_map(|j| (0..la - 1).map(move |i| RefColumn::SumAb(i, j)))
                .collect(),
        }
    }

    fn ref_fit(cells: &CellLayout, terms: &[RefTerm], coding: RefCoding) -> SequenceFit {
        let columns: Vec<Vec<RefColumn>> = terms
            .iter()
            .map(|&t| ref_columns(cells, t, coding))
            .collect();
        let (la, lb) = (cells.levels_a, cells.levels_b);
        let ind = |t: bool| if t { 1.0 } else { 0.0 };
        fit_columns(cells, &columns, |col, a, b| match col {
            RefColumn::One => 1.0,
            RefColumn::A(j) => ind(a == j),
            RefColumn::B(j) => ind(b == j),
            RefColumn::Cell(i, j) => ind(a == i && b == j),
            RefColumn::SumA(j) => sum_code(a, j, la),
            RefColumn::SumB(j) => sum_code(b, j, lb),
            RefColumn::SumAb(i, j) => sum_code(a, i, la) * sum_code(b, j, lb),
        })
        .unwrap()
    }

    /// The terms as the explicit-column fit gives them; `None` where Type III is refused.
    fn reference_terms(
        cells: &CellLayout,
        interaction: bool,
        ss_type: SsType,
    ) -> Option<TwoWayTerms> {
        use RefCoding::{Sum, Treatment};
        use RefTerm::{A, Ab, B, Intercept};
        let main = |first: RefTerm, second: RefTerm| {
            let mut t = vec![Intercept, first, second];
            if interaction {
                t.push(Ab);
            }
            t
        };
        let (a, b, full) = match ss_type {
            SsType::I => {
                let full = ref_fit(cells, &main(A, B), Treatment);
                (full.terms[1], full.terms[2], full)
            }
            SsType::II => {
                let full = ref_fit(cells, &main(A, B), Treatment);
                let ba = ref_fit(cells, &[Intercept, B, A], Treatment);
                (ba.terms[2], full.terms[2], full)
            }
            SsType::III => {
                let empty = cells.levels_a * cells.levels_b - cells.n_cells();
                if interaction && empty > 0 {
                    return None;
                }
                let full = ref_fit(cells, &main(A, B), Sum);
                if full.rank < full.n_cols {
                    return None;
                }
                let last = |others: &[RefTerm], term: RefTerm| {
                    let mut t = vec![Intercept];
                    t.extend_from_slice(others);
                    t.push(term);
                    *ref_fit(cells, &t, Sum).terms.last().unwrap()
                };
                let with_ab = |t: RefTerm| if interaction { vec![t, Ab] } else { vec![t] };
                (last(&with_ab(B), A), last(&with_ab(A), B), full)
            }
        };
        Some(TwoWayTerms {
            a,
            b,
            ab: full.terms.get(3).copied(),
            rank: full.rank,
            rss: cells.ss_within + full.cell_rss,
        })
    }

    fn layout(ds: &Dataset) -> CellLayout {
        let cols = ColumnRequest::new(ds.df())
            .numeric("y")
            .factor("a")
            .factor("b")
            .na(NaPolicy::Omit)
            .extract()
            .unwrap();
        CellLayout::new(cols.numeric("y"), cols.factor("a"), cols.factor("b")).unwrap()
    }

    /// A noisy `la` x `lb` layout with `count(a, b)` rows in cell (a, b) (none when 0).
    fn noisy_layout(la: i64, lb: i64, count: impl Fn(i64, i64) -> usize) -> Dataset {
        let (mut a, mut b, mut y) = (Vec::new(), Vec::new(), Vec::new());
        let mut i = 0.0f64;
        for ai in 0..la {
            for bi in 0..lb {
                for _ in 0..count(ai, bi) {
                    a.push(format!("a{ai:02}"));
                    b.push(format!("b{bi:02}"));
                    let (x, z) = (ai as f64, bi as f64);
                    y.push(2.0 + 0.4 * x - 0.3 * z + 0.2 * x * z + (1.7 * i + 0.3).sin());
                    i += 1.0;
                }
            }
        }
        Dataset::new(df! { "y" => y, "a" => a, "b" => b }.unwrap())
    }

    /// The shortcuts against the explicit-column fit, at 1e-10 relative: SS of terms with
    /// df (an SS without df is the rounding of zero, held to 1e-10 of the total), df, rank
    /// and residual SS, for Types I, II and III, with and without the interaction.
    #[test]
    fn test_two_way_anova_shortcuts_match_explicit_interaction_columns() {
        let layouts = [
            // Complete and unbalanced (1 to 4 rows per cell).
            (
                "complete 3 x 4",
                noisy_layout(3, 4, |a, b| 1 + ((3 * a + 2 * b) % 4) as usize),
            ),
            // Two empty cells, still connected.
            (
                "empty cells 4 x 3",
                noisy_layout(4, 3, |a, b| match (a, b) {
                    (0, 2) | (3, 0) => 0,
                    _ => 1 + ((a + 2 * b) % 3) as usize,
                }),
            ),
            // B nested in A: A given B has no df.
            (
                "nested 2 x 4",
                noisy_layout(2, 4, |a, b| usize::from(b / 2 == a) * 3),
            ),
            // Two diagonal cells: B given A has no df.
            (
                "diagonal 2 x 2",
                noisy_layout(2, 2, |a, b| usize::from(a == b) * 3),
            ),
        ];
        let mut compared_type3_interaction = 0;
        for (name, ds) in &layouts {
            let cells = layout(ds);
            let total = run_two_way_anova(ds, "y", "a", "b", true).unwrap().ss_total;
            for interaction in [true, false] {
                for ss_type in [SsType::I, SsType::II, SsType::III] {
                    let what = format!("{name}, interaction {interaction}, Type {ss_type}");
                    let got = two_way_terms(&cells, interaction, ss_type);
                    let want = reference_terms(&cells, interaction, ss_type);
                    let (got, want) = match (got, want) {
                        (Ok(g), Some(w)) => (g, w),
                        (Err(EconError::InvalidSpecification { .. }), None) => continue,
                        (g, w) => panic!("{what}: {g:?} vs reference {w:?}"),
                    };
                    let close = |x: f64, y: f64, df: usize, term: &str| {
                        let ok = if df == 0 {
                            x.abs() <= 1e-10 * total && y.abs() <= 1e-10 * total
                        } else {
                            (x - y).abs() <= 1e-10 * x.abs().max(y.abs())
                        };
                        assert!(ok, "{what}: {term} {x} vs reference {y}");
                    };
                    assert_eq!(got.a.df, want.a.df, "{what}: df(A)");
                    assert_eq!(got.b.df, want.b.df, "{what}: df(B)");
                    close(got.a.ss, want.a.ss, got.a.df, "SS(A)");
                    close(got.b.ss, want.b.ss, got.b.df, "SS(B)");
                    assert_eq!(
                        got.ab.map(|t| t.df),
                        want.ab.map(|t| t.df),
                        "{what}: df(A:B)"
                    );
                    if let (Some(g), Some(w)) = (got.ab, want.ab) {
                        close(g.ss, w.ss, g.df, "SS(A:B)");
                    }
                    assert_eq!(got.rank, want.rank, "{what}: rank");
                    close(got.rss, want.rss, 1, "RSS");
                    if interaction && ss_type == SsType::III {
                        compared_type3_interaction += 1;
                    }
                }
            }
        }
        // Type III with the interaction exists only on the complete layout.
        assert_eq!(compared_type3_interaction, 1);
    }

    /// A main effect aliased with the other factor has df 0 and no numbers; the table says
    /// it is not estimable, as aov drops the row. Two cells, (a1, b1) and (a2, b2): B given A
    /// is aliased (Type I), and so is A given B (Type II). Fails before: the row was printed
    /// with df 0 and NaN.
    #[test]
    fn test_two_way_anova_aliased_main_effect_not_estimable() {
        let ds = Dataset::new(
            df! {
                "y" => [1.2, 0.8, 1.1, 3.1, 2.7, 3.4],
                "a" => ["a1", "a1", "a1", "a2", "a2", "a2"],
                "b" => ["b1", "b1", "b1", "b2", "b2", "b2"],
            }
            .unwrap(),
        );
        let not_estimable = |r: &TwoWayAnovaResult, name: &str| {
            let line = format!("{name} not estimable (aliased with the other factor)");
            r.to_string().lines().any(|l| l.trim_start() == line)
        };
        let t1 = run_two_way_anova(&ds, "y", "a", "b", true).unwrap();
        assert_eq!((t1.df_a, t1.df_b, t1.df_ab, t1.df_error), (1, 0, None, 4));
        assert!(t1.ms_b.is_nan() && t1.f_b.is_nan() && t1.p_b.is_nan());
        assert!(not_estimable(&t1, "b"), "{t1}");
        assert!(!not_estimable(&t1, "a"), "{t1}");
        assert!(t1.to_string().contains("a:b not estimable"), "{t1}");
        let t2 = two_way_with(&ds, "y", "a", "b", false, SsType::II).unwrap();
        assert_eq!((t2.df_a, t2.df_b), (0, 0));
        assert!(not_estimable(&t2, "a") && not_estimable(&t2, "b"), "{t2}");
    }

    /// The model with the interaction is fitted without one design column per cell, so its
    /// largest buffer holds c x (La + Lb - 1) values, not c x c. Fails before: with the
    /// interaction columns the c x c buffer exceeded this ceiling and the call was refused.
    #[test]
    fn test_two_way_anova_interaction_needs_no_cell_columns() {
        let ds = noisy_layout(40, 40, |_, _| 2);
        resource::with_alloc_ceiling(Some(4 << 20), || {
            for ss_type in [SsType::I, SsType::II, SsType::III] {
                let r = two_way_with(&ds, "y", "a", "b", true, ss_type).unwrap();
                assert_eq!(
                    (r.df_a, r.df_b, r.df_ab, r.df_error),
                    (39, 39, Some(39 * 39), 1600),
                    "Type {ss_type}"
                );
                assert!(r.f_ab.unwrap().is_finite(), "Type {ss_type}");
            }
        });
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

    /// Column-type fixture: an Int64 response equals its Float64 cast; a text response and an
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

    /// Column-type fixture of the two-way wrapper, on the Int64 case stat_friedman_quade:
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
