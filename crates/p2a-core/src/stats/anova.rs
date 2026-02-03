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

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::traits::{SignificanceLevel, f_test_p_value};

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

/// Perform one-way ANOVA on a dataset.
///
/// Tests whether the means of the response variable differ significantly
/// across levels of the factor variable.
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
    let df = dataset.df();

    // Validate columns exist
    let response_col = df.column(response).map_err(|_| EconError::ColumnNotFound {
        column: response.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    let factor_col = df.column(factor).map_err(|_| EconError::ColumnNotFound {
        column: factor.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    // Extract response values as f64
    let y_values: Vec<f64> = response_col
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: response.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    if y_values.is_empty() {
        return Err(EconError::EmptyDataset);
    }

    let n_total = y_values.len();

    // Group data by factor
    let mut groups: HashMap<String, Vec<f64>> = HashMap::new();

    for i in 0..factor_col.len() {
        let group_key = format!(
            "{:?}",
            factor_col.get(i).map_err(|e| {
                EconError::Internal(format!("Failed to get factor value: {}", e))
            })?
        );

        // Get corresponding y value
        if let Some(&y_val) = y_values.get(i) {
            groups.entry(group_key).or_default().push(y_val);
        }
    }

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

        // Clean up the group name (remove quotes from AnyValue format)
        let clean_name = group_name
            .trim_matches('"')
            .trim_start_matches("String(\"")
            .trim_end_matches("\")")
            .trim_start_matches("Int64(")
            .trim_start_matches("Int32(")
            .trim_start_matches("Float64(")
            .trim_end_matches(')')
            .to_string();

        group_stats.push(GroupStats {
            group: clean_name,
            n: n_i,
            mean: group_mean,
            std_dev,
            min,
            max,
        });
    }

    // Sort groups by name for consistent output
    group_stats.sort_by(|a, b| a.group.cmp(&b.group));

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

    let p_value = f_test_p_value(f_statistic, df_between as f64, df_within as f64);

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

        if self.with_interaction {
            let label = format!("{}:{}", self.factor_a, self.factor_b);
            writeln!(
                f,
                "{:>15} {:>12.4} {:>6} {:>12.4} {:>10.4} {:>10.4} {}",
                truncate(&label, 15),
                self.ss_ab.unwrap_or(0.0),
                self.df_ab.unwrap_or(0),
                self.ms_ab.unwrap_or(0.0),
                self.f_ab.unwrap_or(0.0),
                self.p_ab.unwrap_or(1.0),
                self.sig_ab.as_ref().map_or("", |s| s.stars())
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
        writeln!(f)?;

        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        Ok(())
    }
}

/// Perform two-way ANOVA on a dataset.
///
/// Tests the effects of two categorical factors on a response variable,
/// optionally including their interaction.
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
    let df = dataset.df();

    // Validate columns exist
    let response_col = df.column(response).map_err(|_| EconError::ColumnNotFound {
        column: response.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    let factor_a_col = df.column(factor_a).map_err(|_| EconError::ColumnNotFound {
        column: factor_a.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    let factor_b_col = df.column(factor_b).map_err(|_| EconError::ColumnNotFound {
        column: factor_b.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    // Extract response values
    let y_values: Vec<f64> = response_col
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: response.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    if y_values.is_empty() {
        return Err(EconError::EmptyDataset);
    }

    let n_total = y_values.len();

    // Group data by both factors
    // Key: (factor_a_level, factor_b_level) -> values
    let mut cells: HashMap<(String, String), Vec<f64>> = HashMap::new();
    let mut level_a_values: HashMap<String, Vec<f64>> = HashMap::new();
    let mut level_b_values: HashMap<String, Vec<f64>> = HashMap::new();

    for i in 0..n_total {
        let a_key = format!(
            "{:?}",
            factor_a_col.get(i).map_err(|e| {
                EconError::Internal(format!("Failed to get factor A value: {}", e))
            })?
        );
        let b_key = format!(
            "{:?}",
            factor_b_col.get(i).map_err(|e| {
                EconError::Internal(format!("Failed to get factor B value: {}", e))
            })?
        );

        let y = y_values[i];

        cells
            .entry((a_key.clone(), b_key.clone()))
            .or_default()
            .push(y);
        level_a_values.entry(a_key).or_default().push(y);
        level_b_values.entry(b_key).or_default().push(y);
    }

    let levels_a = level_a_values.len();
    let levels_b = level_b_values.len();

    if levels_a < 2 || levels_b < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: levels_a.min(levels_b),
            context: "Two-way ANOVA requires at least 2 levels for each factor".to_string(),
        });
    }

    // Compute means
    let grand_mean: f64 = y_values.iter().sum::<f64>() / n_total as f64;

    // Marginal means for factor A
    let means_a: HashMap<String, f64> = level_a_values
        .iter()
        .map(|(k, v)| (k.clone(), v.iter().sum::<f64>() / v.len() as f64))
        .collect();

    // Marginal means for factor B
    let means_b: HashMap<String, f64> = level_b_values
        .iter()
        .map(|(k, v)| (k.clone(), v.iter().sum::<f64>() / v.len() as f64))
        .collect();

    // Cell means
    let cell_means: HashMap<(String, String), f64> = cells
        .iter()
        .map(|(k, v)| (k.clone(), v.iter().sum::<f64>() / v.len() as f64))
        .collect();

    // Compute sum of squares
    // SST = Σ(y_ijk - grand_mean)²
    let ss_total: f64 = y_values.iter().map(|&y| (y - grand_mean).powi(2)).sum();

    // SS_A = Σ n_i. * (mean_i. - grand_mean)²
    let ss_a: f64 = means_a
        .iter()
        .map(|(k, &mean)| {
            let n = level_a_values.get(k).map_or(0, |v| v.len());
            n as f64 * (mean - grand_mean).powi(2)
        })
        .sum();

    // SS_B = Σ n_.j * (mean_.j - grand_mean)²
    let ss_b: f64 = means_b
        .iter()
        .map(|(k, &mean)| {
            let n = level_b_values.get(k).map_or(0, |v| v.len());
            n as f64 * (mean - grand_mean).powi(2)
        })
        .sum();

    // SS_AB (interaction) and SS_Error
    let (ss_ab, ss_error, df_ab) = if interaction {
        // SS_AB = Σ n_ij * (cell_mean - mean_i. - mean_.j + grand_mean)²
        let ss_ab: f64 = cells
            .iter()
            .map(|((a, b), values)| {
                let cell_mean = cell_means.get(&(a.clone(), b.clone())).unwrap_or(&0.0);
                let mean_a = means_a.get(a).unwrap_or(&0.0);
                let mean_b = means_b.get(b).unwrap_or(&0.0);
                let interaction_effect = cell_mean - mean_a - mean_b + grand_mean;
                values.len() as f64 * interaction_effect.powi(2)
            })
            .sum();

        // SS_Error = Σ(y_ijk - cell_mean_ij)²
        let ss_error: f64 = cells
            .iter()
            .map(|(key, values)| {
                let cell_mean = cell_means.get(key).unwrap_or(&0.0);
                values.iter().map(|&y| (y - cell_mean).powi(2)).sum::<f64>()
            })
            .sum();

        let df_ab = (levels_a - 1) * (levels_b - 1);

        (Some(ss_ab), ss_error, Some(df_ab))
    } else {
        // Additive model: no interaction term
        // SS_Error = SST - SS_A - SS_B
        let ss_error = ss_total - ss_a - ss_b;
        (None, ss_error.max(0.0), None)
    };

    // Degrees of freedom
    let df_a = levels_a - 1;
    let df_b = levels_b - 1;
    let df_error = if interaction {
        n_total - levels_a * levels_b
    } else {
        n_total - levels_a - levels_b + 1
    };
    let df_total = n_total - 1;

    // Mean squares
    let ms_a = ss_a / df_a as f64;
    let ms_b = ss_b / df_b as f64;
    let ms_ab = df_ab.map(|df| ss_ab.unwrap_or(0.0) / df as f64);
    let ms_error = if df_error > 0 {
        ss_error / df_error as f64
    } else {
        f64::NAN
    };

    // F-statistics
    let f_a = if ms_error > 0.0 {
        ms_a / ms_error
    } else {
        f64::NAN
    };
    let f_b = if ms_error > 0.0 {
        ms_b / ms_error
    } else {
        f64::NAN
    };
    let f_ab = ms_ab.map(|ms| {
        if ms_error > 0.0 {
            ms / ms_error
        } else {
            f64::NAN
        }
    });

    // P-values
    let p_a = f_test_p_value(f_a, df_a as f64, df_error as f64);
    let p_b = f_test_p_value(f_b, df_b as f64, df_error as f64);
    let p_ab = df_ab.map(|df| f_test_p_value(f_ab.unwrap_or(0.0), df as f64, df_error as f64));

    Ok(TwoWayAnovaResult {
        response_var: response.to_string(),
        factor_a: factor_a.to_string(),
        factor_b: factor_b.to_string(),
        with_interaction: interaction,
        ss_a,
        ss_b,
        ss_ab,
        ss_error,
        ss_total,
        df_a,
        df_b,
        df_ab,
        df_error,
        df_total,
        ms_a,
        ms_b,
        ms_ab,
        ms_error,
        f_a,
        p_a,
        sig_a: SignificanceLevel::from_p_value(p_a),
        f_b,
        p_b,
        sig_b: SignificanceLevel::from_p_value(p_b),
        f_ab,
        p_ab,
        sig_ab: p_ab.map(SignificanceLevel::from_p_value),
        n_obs: n_total,
        levels_a,
        levels_b,
        grand_mean,
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
}
