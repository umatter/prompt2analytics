//! Model Tables for ANOVA Models.
//!
//! This module provides functionality to compute tables of means, effects,
//! and standard errors from ANOVA model results, similar to R's `model.tables()`.
//!
//! # Mathematical Background
//!
//! ## Table of Means
//!
//! For each factor level i, the estimated marginal mean is:
//!
//! μ̂ᵢ = ȳᵢ. (sample mean for level i)
//!
//! ## Table of Effects
//!
//! The effect for level i is the deviation from the grand mean:
//!
//! αᵢ = μ̂ᵢ - μ̂.. where Σᵢ nᵢαᵢ = 0
//!
//! ## Two-Way ANOVA
//!
//! The tables are R's projections of the sequential fit `y ~ a + b + a:b` (the term order of
//! `aov`), each averaged over the term's cells:
//!
//! - effect of a: fit(a) - grand mean, where fit(a) is the a level mean;
//! - effect of b: fit(a + b) - fit(a), with fit(a + b) the least-squares additive fit;
//! - effect of a:b: fit(a * b) - fit(a + b), fit(a * b) being the cell mean.
//!
//! A term's means are the grand mean plus the effects of the term and of its margins, so the a
//! and a:b means are the raw level and cell means, while on an unbalanced layout the b means
//! are not the raw b margins. On a balanced layout every table is the raw margin.
//!
//! ## Standard Errors
//!
//! As R, one number per term and only for a balanced design, with r the replication of the
//! term (n / levels):
//! - means: the standard error of a difference of two means, √(2 MSE / r);
//! - effects: the standard error of an effect, √(MSE / r).
//!
//! An unbalanced design gets no standard error, and the note R prints instead
//! ([`UNBALANCED_SE_NOTE`]).
//!
//! # References
//!
//! - Fisher, R.A. (1925). *Statistical Methods for Research Workers*. Oliver & Boyd.
//!   The foundational work on ANOVA methodology.
//!
//! - Yates, F. (1934). The analysis of multiple classifications with unequal numbers
//!   in the different classes. *Journal of the American Statistical Association*,
//!   29(185), 51-66. https://doi.org/10.1080/01621459.1934.10502686
//!
//! - Searle, S.R. (1987). *Linear Models for Unbalanced Data*. Wiley.
//!   ISBN: 978-0471848806. Treatment of unbalanced designs.
//!
//! - Milliken, G.A., & Johnson, D.E. (2009). *Analysis of Messy Data Volume 1:
//!   Designed Experiments* (2nd ed.). CRC Press. ISBN: 978-1584883340.
//!
//! R equivalent: `stats::model.tables()`

use crate::stats::AnovaResult;
use serde::{Deserialize, Serialize};

/// What R's `model.tables` prints instead of standard errors for an unbalanced design.
pub const UNBALANCED_SE_NOTE: &str = "Design is unbalanced - use se.contrast() for se's";

/// The kind of standard error of a means table (R: "Standard errors for differences of means").
pub const SE_KIND_MEANS: &str = "differences of means";
/// The kind of standard error of an effects table (R: "Standard errors of effects").
pub const SE_KIND_EFFECTS: &str = "effects";

/// Type of table to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TableType {
    /// Table of means
    #[default]
    Means,
    /// Table of effects (deviations from grand mean)
    Effects,
}

crate::option_enum!(
    TableType,
    crate::options::Choices {
        option: "table type",
        values: &[
            (&["means", "mean"], TableType::Means),
            (&["effects", "effect"], TableType::Effects),
        ],
    }
);

/// Result of model.tables computation for one-way ANOVA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTablesResult {
    /// Type of table computed
    pub table_type: String,
    /// Grand mean
    pub grand_mean: f64,
    /// Group names
    pub group_names: Vec<String>,
    /// Group values (means or effects)
    pub values: Vec<f64>,
    /// Group sample sizes
    pub n: Vec<usize>,
    /// The term's standard error, one value (R reports one number per term): for a means
    /// table the standard error of a difference of two means, for an effects table that of an
    /// effect. `None` when not asked for or when the design is unbalanced (see `se_note`).
    pub se: Option<Vec<f64>>,
    /// What `se` is: [`SE_KIND_MEANS`] or [`SE_KIND_EFFECTS`]
    #[serde(default)]
    pub se_kind: Option<String>,
    /// Why there is no standard error although one was asked for ([`UNBALANCED_SE_NOTE`])
    #[serde(default)]
    pub se_note: Option<String>,
    /// Replications (sample sizes)
    pub replications: Vec<usize>,
    /// Mean squared error from ANOVA
    pub mse: f64,
    /// Degrees of freedom for MSE
    pub df_mse: f64,
}

/// Result of model.tables computation for two-way ANOVA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoWayModelTablesResult {
    /// Type of table computed
    pub table_type: String,
    /// Grand mean
    pub grand_mean: f64,
    /// Factor A names
    pub factor_a_names: Vec<String>,
    /// Factor A values (means or effects)
    pub factor_a_values: Vec<f64>,
    /// Factor A sample sizes
    pub factor_a_n: Vec<usize>,
    /// Factor B names
    pub factor_b_names: Vec<String>,
    /// Factor B values (means or effects). On an unbalanced layout these are R's projections
    /// (fit(a + b) - fit(a) averaged over each b level), not the raw b margins.
    pub factor_b_values: Vec<f64>,
    /// Factor B sample sizes
    pub factor_b_n: Vec<usize>,
    /// The a:b table (factor_a x factor_b): the cell means, or for effects R's a:b effects
    /// (the interaction, cell mean - fit(a + b)). An empty cell is NaN.
    pub cell_values: Option<Vec<Vec<f64>>>,
    /// Cell sample sizes
    pub cell_n: Option<Vec<Vec<usize>>>,
    /// Interaction effects (effects tables only; the same values as `cell_values`)
    pub interaction: Option<Vec<Vec<f64>>>,
    /// Standard errors, one value per term (balanced designs only)
    pub se: Option<ModelTablesSE>,
    /// What `se` is: [`SE_KIND_MEANS`] or [`SE_KIND_EFFECTS`]
    #[serde(default)]
    pub se_kind: Option<String>,
    /// Why there is no standard error although one was asked for
    #[serde(default)]
    pub se_note: Option<String>,
    /// Mean squared error from ANOVA
    pub mse: f64,
    /// Degrees of freedom for MSE
    pub df_mse: f64,
}

/// Standard errors for model tables: one value per term, as R reports them (each vector has
/// length 1, `cells` is 1 x 1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTablesSE {
    /// SE for factor A means/effects
    pub factor_a: Option<Vec<f64>>,
    /// SE for factor B means/effects
    pub factor_b: Option<Vec<f64>>,
    /// SE for the a:b means/effects
    pub cells: Option<Vec<Vec<f64>>>,
}

/// The term's standard error: √(2 MSE / r) for a means table (a difference of two means),
/// √(MSE / r) for an effects table.
fn term_se(table_type: TableType, mse: f64, r: f64) -> f64 {
    match table_type {
        TableType::Means => (2.0 * mse / r).sqrt(),
        TableType::Effects => (mse / r).sqrt(),
    }
}

fn se_kind_of(table_type: TableType) -> String {
    match table_type {
        TableType::Means => SE_KIND_MEANS.to_string(),
        TableType::Effects => SE_KIND_EFFECTS.to_string(),
    }
}

/// Compute model tables from one-way ANOVA result.
///
/// Returns tables of means or effects (deviations from grand mean)
/// along with R's standard error of the term (balanced designs only).
///
/// # Arguments
/// * `anova` - The ANOVA result
/// * `table_type` - Type of table to compute (Means or Effects)
/// * `se` - Whether to compute standard errors
///
/// # Returns
/// A `ModelTablesResult` containing the requested table
///
/// # Example
/// ```
/// use p2a_core::stats::modeltables::{model_tables, TableType};
/// use p2a_core::stats::AnovaResult;
///
/// // Assuming you have an ANOVA result
/// // let result = model_tables(&anova, TableType::Means, true).unwrap();
/// ```
pub fn model_tables(
    anova: &AnovaResult,
    table_type: TableType,
    compute_se: bool,
) -> Result<ModelTablesResult, String> {
    if anova.groups.is_empty() {
        return Err("ANOVA must have at least one group".to_string());
    }

    let grand_mean = anova.grand_mean;
    let mse = anova.ms_within;
    let df_mse = anova.df_within as f64;

    let group_names: Vec<String> = anova.groups.iter().map(|g| g.group.clone()).collect();
    let group_ns: Vec<usize> = anova.groups.iter().map(|g| g.n).collect();
    let group_means: Vec<f64> = anova.groups.iter().map(|g| g.mean).collect();

    let values = match table_type {
        TableType::Means => group_means.clone(),
        TableType::Effects => {
            // Effects are deviations from grand mean
            group_means.iter().map(|m| m - grand_mean).collect()
        }
    };

    // R: one standard error for the term, and none for an unbalanced design.
    let balanced = group_ns[0] > 0 && group_ns.iter().all(|&n| n == group_ns[0]);
    let (se, se_kind, se_note) = if !compute_se {
        (None, None, None)
    } else if !balanced {
        (None, None, Some(UNBALANCED_SE_NOTE.to_string()))
    } else {
        let r = group_ns[0] as f64;
        (
            Some(vec![term_se(table_type, mse, r)]),
            Some(se_kind_of(table_type)),
            None,
        )
    };

    Ok(ModelTablesResult {
        table_type: match table_type {
            TableType::Means => "means".to_string(),
            TableType::Effects => "effects".to_string(),
        },
        grand_mean,
        group_names,
        values,
        n: group_ns.clone(),
        se,
        se_kind,
        se_note,
        replications: group_ns,
        mse,
        df_mse,
    })
}

/// Compute model tables with default settings (means table with SE).
pub fn model_tables_means(anova: &AnovaResult) -> Result<ModelTablesResult, String> {
    model_tables(anova, TableType::Means, true)
}

/// Compute effects table from one-way ANOVA.
pub fn model_tables_effects(anova: &AnovaResult) -> Result<ModelTablesResult, String> {
    model_tables(anova, TableType::Effects, true)
}

/// Compute model tables from two-way ANOVA data.
///
/// This function takes raw data organized by two factors and computes
/// means or effects tables.
///
/// # Arguments
/// * `data` - Data organized as `data[factor_a][factor_b]` containing vectors of observations
/// * `factor_a_names` - Names for factor A levels
/// * `factor_b_names` - Names for factor B levels
/// * `table_type` - Type of table to compute
/// * `compute_se` - Whether to compute standard errors
///
/// # Returns
/// A `TwoWayModelTablesResult` containing the requested tables. An empty cell, and a factor level
/// with no observations, have a NaN mean (R: NA); their effects and interaction terms are NaN too.
/// The layout must be rectangular: a row whose length differs from `data[0]` is an error.
pub fn model_tables_two_way(
    data: &[Vec<Vec<f64>>],
    factor_a_names: &[String],
    factor_b_names: &[String],
    table_type: TableType,
    compute_se: bool,
) -> Result<TwoWayModelTablesResult, String> {
    let a_levels = data.len();
    let b_levels = if a_levels > 0 { data[0].len() } else { 0 };

    if a_levels == 0 || b_levels == 0 {
        return Err("Data must have at least one level for each factor".to_string());
    }

    if factor_a_names.len() != a_levels {
        return Err(format!(
            "Factor A names length ({}) doesn't match data ({} levels)",
            factor_a_names.len(),
            a_levels
        ));
    }

    if factor_b_names.len() != b_levels {
        return Err(format!(
            "Factor B names length ({}) doesn't match data ({} levels)",
            factor_b_names.len(),
            b_levels
        ));
    }

    for (i, row) in data.iter().enumerate() {
        if row.len() != b_levels {
            return Err(format!(
                "data[{i}] has {} factor B cells, data[0] has {b_levels}; the layout must be \
                 rectangular (one cell per factor A x factor B level, empty cells as empty vectors)",
                row.len()
            ));
        }
    }

    // Compute cell statistics (a cell with no observations stays NaN, R's NA)
    let mut cell_means = vec![vec![f64::NAN; b_levels]; a_levels];
    let mut cell_n = vec![vec![0usize; b_levels]; a_levels];
    let mut total_sum = 0.0;
    let mut total_n = 0;
    let mut ss_within = 0.0;

    for (i, a_data) in data.iter().enumerate() {
        for (j, cell_data) in a_data.iter().enumerate() {
            let n = cell_data.len();
            if n > 0 {
                let sum: f64 = cell_data.iter().sum();
                let mean = sum / n as f64;
                cell_means[i][j] = mean;
                cell_n[i][j] = n;
                total_sum += sum;
                total_n += n;

                // SS within
                for &x in cell_data {
                    ss_within += (x - mean).powi(2);
                }
            }
        }
    }

    if total_n == 0 {
        return Err("No observations in data".to_string());
    }

    let grand_mean = total_sum / total_n as f64;
    // The residual df of aov(y ~ a * b): n minus the number of non-empty cells (the rank of
    // the full model). Each non-empty cell holds at least one row, so this cannot underflow.
    let n_nonempty = cell_n.iter().flatten().filter(|&&n| n > 0).count();
    let df_within = total_n - n_nonempty;
    let mse = if df_within > 0 {
        ss_within / df_within as f64
    } else {
        0.0
    };

    // Raw margins of factor A (fit(a), R's a means) and the level counts of both factors
    let mut factor_a_means = vec![f64::NAN; a_levels];
    let mut factor_a_n = vec![0usize; a_levels];
    let mut factor_b_n = vec![0usize; b_levels];
    for i in 0..a_levels {
        let mut sum = 0.0;
        for j in 0..b_levels {
            let n = cell_n[i][j];
            if n > 0 {
                sum += cell_means[i][j] * n as f64;
                factor_a_n[i] += n;
                factor_b_n[j] += n;
            }
        }
        if factor_a_n[i] > 0 {
            factor_a_means[i] = sum / factor_a_n[i] as f64;
        }
    }

    // Balanced: every cell holds the same number r > 0 of rows. Then the additive fit is
    // the sum of the raw margins and every table is a raw margin.
    let r_cell = cell_n[0][0];
    let balanced = r_cell > 0 && cell_n.iter().flatten().all(|&n| n == r_cell);

    // fit(a + b) at each non-empty cell (constant within a cell)
    let fit_ab: Vec<Vec<f64>> = if balanced {
        let mut factor_b_means = vec![0.0; b_levels];
        for (j, m) in factor_b_means.iter_mut().enumerate() {
            *m = (0..a_levels).map(|i| cell_means[i][j]).sum::<f64>() / a_levels as f64;
        }
        (0..a_levels)
            .map(|i| {
                (0..b_levels)
                    .map(|j| factor_a_means[i] + factor_b_means[j] - grand_mean)
                    .collect()
            })
            .collect()
    } else {
        additive_cell_fit(&cell_means, &cell_n)?
    };

    // Effects: a = fit(a) - grand; b = fit(a + b) - fit(a) averaged over the b level;
    // a:b = fit(a * b) - fit(a + b), the cell mean minus the additive fit.
    let a_effects: Vec<f64> = factor_a_means.iter().map(|m| m - grand_mean).collect();
    let mut b_effects = vec![f64::NAN; b_levels];
    for (j, eff) in b_effects.iter_mut().enumerate() {
        if factor_b_n[j] > 0 {
            let s: f64 = (0..a_levels)
                .filter(|&i| cell_n[i][j] > 0)
                .map(|i| cell_n[i][j] as f64 * (fit_ab[i][j] - factor_a_means[i]))
                .sum();
            *eff = s / factor_b_n[j] as f64;
        }
    }
    let ab_effects: Vec<Vec<f64>> = (0..a_levels)
        .map(|i| {
            (0..b_levels)
                .map(|j| {
                    if cell_n[i][j] > 0 {
                        cell_means[i][j] - fit_ab[i][j]
                    } else {
                        f64::NAN
                    }
                })
                .collect()
        })
        .collect();

    // A term's means: the grand mean plus the effects of the term and its margins (for a:b
    // that is the cell mean).
    let (factor_a_values, factor_b_values, cell_values, interaction) = match table_type {
        TableType::Means => (
            factor_a_means,
            b_effects.iter().map(|e| grand_mean + e).collect(),
            Some(cell_means),
            None,
        ),
        TableType::Effects => (
            a_effects,
            b_effects,
            Some(ab_effects.clone()),
            Some(ab_effects),
        ),
    };

    // Standard errors: one per term, balanced designs with residual df only. The term's
    // replication is n / (its number of cells): r * B for a, r * A for b, r for a:b.
    let (se, se_kind, se_note) = if !compute_se {
        (None, None, None)
    } else if !balanced {
        (None, None, Some(UNBALANCED_SE_NOTE.to_string()))
    } else if df_within == 0 {
        (
            None,
            None,
            Some("no residual degrees of freedom: no standard errors".to_string()),
        )
    } else {
        let r = r_cell as f64;
        let se = ModelTablesSE {
            factor_a: Some(vec![term_se(table_type, mse, r * b_levels as f64)]),
            factor_b: Some(vec![term_se(table_type, mse, r * a_levels as f64)]),
            cells: Some(vec![vec![term_se(table_type, mse, r)]]),
        };
        (Some(se), Some(se_kind_of(table_type)), None)
    };

    Ok(TwoWayModelTablesResult {
        table_type: match table_type {
            TableType::Means => "means".to_string(),
            TableType::Effects => "effects".to_string(),
        },
        grand_mean,
        factor_a_names: factor_a_names.to_vec(),
        factor_a_values,
        factor_a_n,
        factor_b_names: factor_b_names.to_vec(),
        factor_b_values,
        factor_b_n,
        cell_values,
        cell_n: Some(cell_n),
        interaction,
        se,
        se_kind,
        se_note,
        mse,
        df_mse: df_within as f64,
    })
}

/// The least-squares additive fit fit(a + b) of `y ~ a + b`, at each non-empty cell: the
/// cell means regressed on an intercept and treatment dummies of both factors, weighted by
/// the cell counts (the same fitted values as the fit on the rows). Dummies of levels with no
/// rows are aliased. Empty cells get NaN.
fn additive_cell_fit(
    cell_means: &[Vec<f64>],
    cell_n: &[Vec<usize>],
) -> Result<Vec<Vec<f64>>, String> {
    use ndarray::Array1;
    let (a_levels, b_levels) = (cell_n.len(), cell_n[0].len());
    let cells: Vec<(usize, usize)> = (0..a_levels)
        .flat_map(|i| (0..b_levels).map(move |j| (i, j)))
        .filter(|&(i, j)| cell_n[i][j] > 0)
        .collect();
    let p = a_levels + b_levels - 1;
    let mut x = crate::resource::try_zeros2(cells.len(), p, "additive fit of the model tables")
        .map_err(|e| e.to_string())?;
    let mut y = Array1::zeros(cells.len());
    let mut w = Array1::zeros(cells.len());
    for (k, &(i, j)) in cells.iter().enumerate() {
        x[[k, 0]] = 1.0;
        if i > 0 {
            x[[k, i]] = 1.0;
        }
        if j > 0 {
            x[[k, a_levels - 1 + j]] = 1.0;
        }
        y[k] = cell_means[i][j];
        w[k] = cell_n[i][j] as f64;
    }
    let fit = crate::linalg::solve::weighted_least_squares(x.view(), y.view(), w.view())
        .map_err(|e| format!("additive fit of the model tables failed: {e}"))?;
    let mut out = vec![vec![f64::NAN; b_levels]; a_levels];
    for (k, &(i, j)) in cells.iter().enumerate() {
        out[i][j] = fit.fitted[k];
    }
    Ok(out)
}

/// Convenience wrapper for model.tables.
pub fn run_model_tables(
    anova: &AnovaResult,
    table_type: TableType,
    compute_se: bool,
) -> Result<ModelTablesResult, String> {
    model_tables(anova, table_type, compute_se)
}

/// Format model tables result as a string table, followed by R's standard-error line (or the
/// note R prints for an unbalanced design).
pub fn format_model_tables(result: &ModelTablesResult) -> String {
    let mut output = String::new();

    output.push_str(&format!("Tables of {} \n\n", result.table_type));
    output.push_str(&format!("Grand mean: {:.4}\n\n", result.grand_mean));

    output.push_str(" Group     |     Value     |      n      \n");
    output.push_str("-----------+---------------+-------------\n");
    for (i, name) in result.group_names.iter().enumerate() {
        output.push_str(&format!(
            " {:<9} | {:>13.4} | {:>11} \n",
            name, result.values[i], result.n[i]
        ));
    }

    if let (Some(se), Some(kind)) = (&result.se, &result.se_kind) {
        let title = if kind == SE_KIND_MEANS {
            "Standard errors for differences of means"
        } else {
            "Standard errors of effects"
        };
        output.push_str(&format!("\n{title}\n"));
        output.push_str(&format!("        {:.4}\n", se[0]));
        output.push_str(&format!(
            "replic. {}\n",
            result.n.first().copied().unwrap_or(0)
        ));
    } else if let Some(note) = &result.se_note {
        output.push_str(&format!("\n{note}\n"));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use crate::stats::run_one_way_anova;
    use polars::prelude::*;

    fn create_test_dataset() -> Dataset {
        // Three groups with clear differences: A (mean ~5), B (mean ~7), C (mean ~9)
        let df = df! {
            "value" => [
                4.5, 5.0, 5.5, 4.8, 5.2, 5.0, 4.7, 5.3, 5.1, 4.9,  // Group A: mean ≈ 5
                6.5, 7.0, 7.5, 6.8, 7.2, 7.0, 6.7, 7.3, 7.1, 6.9,  // Group B: mean ≈ 7
                8.5, 9.0, 9.5, 8.8, 9.2, 9.0, 8.7, 9.3, 9.1, 8.9   // Group C: mean ≈ 9
            ],
            "group" => [
                "A", "A", "A", "A", "A", "A", "A", "A", "A", "A",
                "B", "B", "B", "B", "B", "B", "B", "B", "B", "B",
                "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"
            ]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_model_tables_means() {
        let dataset = create_test_dataset();
        let anova = run_one_way_anova(&dataset, "value", "group").unwrap();
        let result = model_tables(&anova, TableType::Means, true).unwrap();

        assert_eq!(result.table_type, "means");
        // Grand mean should be around 7 (average of 5, 7, 9)
        assert!((result.grand_mean - 7.0).abs() < 0.1);
        assert_eq!(result.group_names.len(), 3);

        // Values should be group means
        assert_eq!(result.values.len(), 3);
        for val in &result.values {
            assert!(*val > 0.0);
        }

        // SE should exist and be positive
        let se = result.se.unwrap();
        for s in &se {
            assert!(*s > 0.0);
        }
    }

    #[test]
    fn test_model_tables_effects() {
        let dataset = create_test_dataset();
        let anova = run_one_way_anova(&dataset, "value", "group").unwrap();
        let result = model_tables(&anova, TableType::Effects, true).unwrap();

        assert_eq!(result.table_type, "effects");

        // Effects should sum to approximately zero
        let sum: f64 = result.values.iter().sum();
        assert!(sum.abs() < 0.5);
    }

    #[test]
    fn test_model_tables_two_way() {
        // 2x2 factorial design
        let data = vec![
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], // Factor A level 1
            vec![vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0]], // Factor A level 2
        ];
        let factor_a_names = vec!["A1".to_string(), "A2".to_string()];
        let factor_b_names = vec!["B1".to_string(), "B2".to_string()];

        let result = model_tables_two_way(
            &data,
            &factor_a_names,
            &factor_b_names,
            TableType::Means,
            true,
        )
        .unwrap();

        assert_eq!(result.table_type, "means");
        assert_eq!(result.factor_a_names.len(), 2);
        assert_eq!(result.factor_b_names.len(), 2);

        // Grand mean = (1+2+3+4+5+6+7+8+9+10+11+12) / 12 = 78/12 = 6.5
        assert!((result.grand_mean - 6.5).abs() < 1e-10);

        // Cell means
        let cell_values = result.cell_values.unwrap();
        assert!((cell_values[0][0] - 2.0).abs() < 1e-10); // mean(1,2,3)
        assert!((cell_values[0][1] - 5.0).abs() < 1e-10); // mean(4,5,6)
        assert!((cell_values[1][0] - 8.0).abs() < 1e-10); // mean(7,8,9)
        assert!((cell_values[1][1] - 11.0).abs() < 1e-10); // mean(10,11,12)
    }

    #[test]
    fn test_model_tables_two_way_effects() {
        let data = vec![
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            vec![vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0]],
        ];
        let factor_a_names = vec!["A1".to_string(), "A2".to_string()];
        let factor_b_names = vec!["B1".to_string(), "B2".to_string()];

        let result = model_tables_two_way(
            &data,
            &factor_a_names,
            &factor_b_names,
            TableType::Effects,
            true,
        )
        .unwrap();

        assert_eq!(result.table_type, "effects");

        // Grand mean = 6.5
        // Factor A means: A1 = 3.5, A2 = 9.5
        // Factor A effects: A1 = 3.5 - 6.5 = -3, A2 = 9.5 - 6.5 = 3
        assert!((result.factor_a_values[0] - (-3.0)).abs() < 1e-10);
        assert!((result.factor_a_values[1] - 3.0).abs() < 1e-10);

        // Factor B means: B1 = 5, B2 = 8
        // Factor B effects: B1 = 5 - 6.5 = -1.5, B2 = 8 - 6.5 = 1.5
        assert!((result.factor_b_values[0] - (-1.5)).abs() < 1e-10);
        assert!((result.factor_b_values[1] - 1.5).abs() < 1e-10);

        // Interaction effects should exist
        assert!(result.interaction.is_some());
    }

    #[test]
    fn test_format_model_tables() {
        let dataset = create_test_dataset();
        let anova = run_one_way_anova(&dataset, "value", "group").unwrap();
        let result = model_tables(&anova, TableType::Means, true).unwrap();
        let formatted = format_model_tables(&result);

        assert!(formatted.contains("Tables of means"));
        assert!(formatted.contains("Grand mean"));
        assert!(formatted.contains("Group"));
    }

    #[test]
    fn test_two_way_empty_data() {
        // Test with empty data
        let data: Vec<Vec<Vec<f64>>> = vec![];
        let factor_a_names: Vec<String> = vec![];
        let factor_b_names: Vec<String> = vec![];

        let result = model_tables_two_way(
            &data,
            &factor_a_names,
            &factor_b_names,
            TableType::Means,
            true,
        );

        assert!(result.is_err());
    }
    // ========================================================================
    // R-pinned: residual df of aov and the one-way case (T2-WP07)
    // ========================================================================

    use crate::data::columns::{ColumnRequest, NaPolicy};
    use crate::test_support::load_case;
    use p2a_validation::{Check, Tol};

    /// The 3 x 2 layout y3 ~ fa3 * fb3 of stat_anova_two_way_unbalanced, whose (C, L) cell is
    /// empty: the residual df is n minus the 5 non-empty cells (aov), not n - 3 * 2.
    #[test]
    fn test_validate_model_tables_two_way_empty_cell() {
        let (case, ds) = load_case("stat_anova_two_way_unbalanced");
        let cols = ColumnRequest::new(ds.df())
            .numeric("y3")
            .factor("fa3")
            .factor("fb3")
            .na(NaPolicy::Omit)
            .extract()
            .unwrap();
        let (fa, fb) = (cols.factor("fa3"), cols.factor("fb3"));
        let mut data = vec![vec![Vec::new(); fb.n_levels()]; fa.n_levels()];
        for (i, &y) in cols.numeric("y3").iter().enumerate() {
            data[fa.codes[i] as usize][fb.codes[i] as usize].push(y);
        }
        assert!(data[2][1].is_empty(), "the (C, L) cell is empty");
        let r =
            model_tables_two_way(&data, &fa.levels, &fb.levels, TableType::Means, true).unwrap();
        // Unbalanced: R gives no standard errors, only its note.
        assert!(r.se.is_none() && r.se_kind.is_none());
        assert_eq!(r.se_note.as_deref(), Some(UNBALANCED_SE_NOTE));
        Check::new(&case)
            .scalar("empty_resid_df", r.df_mse, Tol::exact())
            .scalar("empty_resid_ss", r.mse * r.df_mse, Tol::stat(13))
            .finish();
    }

    /// Model tables of the one-way ANOVA on stat_oneway_na_labels (5 of 9 rows used).
    #[test]
    fn test_validate_model_tables_one_way_na_labels() {
        let (case, ds) = load_case("stat_oneway_na_labels");
        let anova = run_one_way_anova(&ds, "y", "g").unwrap();
        let r = model_tables(&anova, TableType::Means, true).unwrap();
        let sizes: Vec<f64> = r.n.iter().map(|&n| n as f64).collect();
        // Groups of 2, 2 and 1 rows: unbalanced, so no standard error, only R's note.
        assert!(r.se.is_none());
        assert_eq!(r.se_note.as_deref(), Some(UNBALANCED_SE_NOTE));
        Check::new(&case)
            .named("aov_group_means", &r.group_names, &r.values, Tol::stat(5))
            .named("aov_group_n", &r.group_names, &sizes, Tol::exact())
            .scalar("aov_grand_mean", r.grand_mean, Tol::stat(5))
            .scalar("aov_ss_within", r.mse * r.df_mse, Tol::stat(5))
            .scalar("aov_df_within", r.df_mse, Tol::exact())
            .finish();
        assert_eq!(r.group_names, ["a", "b", "c"]);
    }

    /// STAT-83: a shorter or longer row is refused.
    #[test]
    fn test_modeltables_ragged_is_error() {
        let a = vec!["A1".to_string(), "A2".to_string()];
        let b = vec!["B1".to_string(), "B2".to_string()];
        let short = vec![vec![vec![1.0, 2.0], vec![3.0, 4.1]], vec![vec![5.0, 6.2]]];
        let long = vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.1]],
            vec![vec![5.0], vec![6.2], vec![7.4]],
        ];
        for data in [short, long] {
            let e = model_tables_two_way(&data, &a, &b, TableType::Means, true).unwrap_err();
            assert!(e.contains("data[1]") && e.contains("rectangular"), "{e}");
        }
    }

    /// STAT-83: an empty cell is NaN (R: NA), not 0; its effects and interaction are NaN; the rest is R's.
    #[test]
    fn test_validate_modeltables_t3_empty_cell() {
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_modeltables_t3");
        let df = ds.df();
        let y = numeric_column(df, "y", NaPolicy::Fail).unwrap().0;
        let fa = df.column("fa").unwrap().str().unwrap();
        let fb = df.column("fb").unwrap().str().unwrap();
        let (al, bl) = (["A1", "A2", "A3"], ["B1", "B2", "B3"]);
        let mut data = vec![vec![Vec::new(); 3]; 3];
        for (i, &yi) in y.iter().enumerate() {
            let a = al.iter().position(|l| fa.get(i) == Some(*l)).unwrap();
            let b = bl.iter().position(|l| fb.get(i) == Some(*l)).unwrap();
            data[a][b].push(yi);
        }
        let names = |l: [&str; 3]| l.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let m =
            model_tables_two_way(&data, &names(al), &names(bl), TableType::Means, true).unwrap();
        let cells: Vec<f64> = m
            .cell_values
            .as_ref()
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .collect();
        let ns: Vec<f64> = m
            .cell_n
            .as_ref()
            .unwrap()
            .iter()
            .flatten()
            .map(|&n| n as f64)
            .collect();
        let a_n: Vec<f64> = m.factor_a_n.iter().map(|&n| n as f64).collect();
        Check::new(&case)
            .vector("cell_means", &cells, Tol::coef(23))
            .vector("cell_n", &ns, Tol::exact())
            .named(
                "a_means",
                &m.factor_a_names,
                &m.factor_a_values,
                Tol::coef(23),
            )
            .named("a_n", &m.factor_a_names, &a_n, Tol::exact())
            .named(
                "b_means_model_tables",
                &m.factor_b_names,
                &m.factor_b_values,
                Tol::coef(23),
            )
            .scalar("grand_mean", m.grand_mean, Tol::coef(23))
            .scalar("resid_df", m.df_mse, Tol::exact())
            .scalar("resid_ss", m.mse * m.df_mse, Tol::stat(23))
            .finish();
        assert!(cells[7].is_nan()); // A3:B2
        // unbalanced: no standard errors, R's note instead
        assert!(m.se.is_none());
        assert_eq!(m.se_note.as_deref(), Some(UNBALANCED_SE_NOTE));
        let e =
            model_tables_two_way(&data, &names(al), &names(bl), TableType::Effects, false).unwrap();
        assert!(e.cell_values.unwrap()[2][1].is_nan());
        let inter = e.interaction.unwrap();
        assert!(inter[2][1].is_nan());
        assert!(inter.iter().flatten().filter(|v| v.is_nan()).count() == 1);
        // a factor level with no observations at all: NaN margin, not 0
        let mut gap = data.clone();
        gap[2] = vec![Vec::new(); 3];
        let g =
            model_tables_two_way(&gap, &names(al), &names(bl), TableType::Means, false).unwrap();
        assert!(g.factor_a_values[2].is_nan() && g.factor_a_n[2] == 0);
    }

    // ========================================================================
    // R-pinned: model.tables standard errors and two-way projections (stats_mcp_t4)
    // ========================================================================

    fn mcp_t4_dataset(name: &str) -> Dataset {
        let path = p2a_validation::repo_path(&format!("validation/datasets/generated/{name}.csv"));
        Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap())
    }

    /// A two-way layout data[a][b] of y ~ a * b from a stats_mcp_t4 dataset.
    fn two_way_layout(name: &str) -> (Vec<Vec<Vec<f64>>>, Vec<String>, Vec<String>, usize) {
        let ds = mcp_t4_dataset(name);
        let cols = ColumnRequest::new(ds.df())
            .numeric("y")
            .factor("a")
            .factor("b")
            .na(NaPolicy::Omit)
            .extract()
            .unwrap();
        let (fa, fb) = (cols.factor("a"), cols.factor("b"));
        let mut data = vec![vec![Vec::new(); fb.n_levels()]; fa.n_levels()];
        let y = cols.numeric("y");
        for (i, &v) in y.iter().enumerate() {
            data[fa.codes[i] as usize][fb.codes[i] as usize].push(v);
        }
        (data, fa.levels.clone(), fb.levels.clone(), y.len())
    }

    /// STAT-75: a balanced one-way table has R's one standard error per term: of a difference
    /// of means for the means table, of an effect for the effects table.
    #[test]
    fn test_validate_model_tables_one_way_se_stats_mcp_t4() {
        let case = p2a_validation::Case::load("stats_mcp_t4");
        let ds = mcp_t4_dataset("modeltables_t4_oneway");
        let anova = run_one_way_anova(&ds, "y", "g").unwrap();
        let n = 12;
        for (tt, key, se_key, kind) in [
            (TableType::Means, "ow_means_g", "ow_se_means", SE_KIND_MEANS),
            (
                TableType::Effects,
                "ow_effects_g",
                "ow_se_effects",
                SE_KIND_EFFECTS,
            ),
        ] {
            let r = model_tables(&anova, tt, true).unwrap();
            let se = r.se.as_ref().unwrap();
            assert_eq!(se.len(), 1, "one standard error per term");
            assert_eq!(r.se_kind.as_deref(), Some(kind));
            assert!(r.se_note.is_none());
            let sizes: Vec<f64> = r.n.iter().map(|&n| n as f64).collect();
            Check::new(&case)
                .vector(key, &r.values, Tol::coef(n))
                .scalar("ow_grand", r.grand_mean, Tol::coef(n))
                .scalar("ow_mse", r.mse, Tol::stat(n))
                .scalar("ow_df_resid", r.df_mse, Tol::exact())
                .scalar(se_key, se[0], Tol::se(n))
                .scalar("ow_n", sizes[0], Tol::exact())
                .finish();
            let text = format_model_tables(&r);
            let title = if tt == TableType::Means {
                "Standard errors for differences of means"
            } else {
                "Standard errors of effects"
            };
            assert!(text.contains(title) && text.contains("replic. 4"), "{text}");
        }
    }

    /// STAT-75: an unbalanced one-way design gets no standard error, only R's note.
    #[test]
    fn test_validate_model_tables_one_way_unbalanced_stats_mcp_t4() {
        let case = p2a_validation::Case::load("stats_mcp_t4");
        let ds = mcp_t4_dataset("modeltables_t4_oneway");
        let anova = run_one_way_anova(&ds, "yu", "gu").unwrap();
        for (tt, key) in [
            (TableType::Means, "owu_means_g"),
            (TableType::Effects, "owu_effects_g"),
        ] {
            let r = model_tables(&anova, tt, true).unwrap();
            assert!(r.se.is_none() && r.se_kind.is_none());
            assert_eq!(r.se_note.as_deref(), Some(UNBALANCED_SE_NOTE));
            let sizes: Vec<f64> = r.n.iter().map(|&n| n as f64).collect();
            Check::new(&case)
                .vector(key, &r.values, Tol::coef(12))
                .scalar("owu_grand", r.grand_mean, Tol::coef(12))
                .scalar("owu_mse", r.mse, Tol::stat(12))
                .vector("owu_n", &sizes, Tol::exact())
                .finish();
            assert!(format_model_tables(&r).contains(UNBALANCED_SE_NOTE));
        }
    }

    /// TOOL-22, STAT-75: two-way tables are R's projections of the sequential fit on balanced,
    /// unbalanced and empty-cell layouts; standard errors only for the balanced one.
    #[test]
    fn test_validate_model_tables_two_way_stats_mcp_t4() {
        let case = p2a_validation::Case::load("stats_mcp_t4");
        for (prefix, name) in [
            ("bal_", "modeltables_t4_bal"),
            ("unbal_", "modeltables_t4_unbal"),
            ("empty_", "modeltables_t4_empty"),
        ] {
            let (data, la, lb, n) = two_way_layout(name);
            for (tt, kind) in [(TableType::Means, "means"), (TableType::Effects, "effects")] {
                let r = model_tables_two_way(&data, &la, &lb, tt, true).unwrap();
                let ab: Vec<f64> = r
                    .cell_values
                    .as_ref()
                    .unwrap()
                    .iter()
                    .flatten()
                    .copied()
                    .collect();
                let cell_n: Vec<f64> = r
                    .cell_n
                    .as_ref()
                    .unwrap()
                    .iter()
                    .flatten()
                    .map(|&c| c as f64)
                    .collect();
                let k = |t: &str| format!("{prefix}{kind}_{t}");
                // the empty cell's a1:b3 effect is -2.8e-17 in R: an absolute floor
                let ab_tol = if prefix == "empty_" && tt == TableType::Effects {
                    Tol::new(1e-8, 1e-12)
                } else {
                    Tol::coef(n)
                };
                Check::new(&case)
                    .vector(&k("a"), &r.factor_a_values, Tol::coef(n))
                    .vector(&k("b"), &r.factor_b_values, Tol::coef(n))
                    .vector(&k("ab"), &ab, ab_tol)
                    .scalar(&format!("{prefix}grand"), r.grand_mean, Tol::coef(n))
                    .scalar(&format!("{prefix}mse"), r.mse, Tol::stat(n))
                    .scalar(&format!("{prefix}df_resid"), r.df_mse, Tol::exact())
                    .vector(&format!("{prefix}cell_n"), &cell_n, Tol::exact())
                    .finish();
                if tt == TableType::Effects {
                    // equal cell by cell, an empty cell NaN in both
                    let bits = |m: &Vec<Vec<f64>>| -> Vec<u64> {
                        m.iter().flatten().map(|v| v.to_bits()).collect()
                    };
                    assert_eq!(
                        bits(r.interaction.as_ref().unwrap()),
                        bits(r.cell_values.as_ref().unwrap())
                    );
                }
                if prefix == "bal_" {
                    let se = r.se.as_ref().unwrap();
                    let got = [
                        se.factor_a.as_ref().unwrap()[0],
                        se.factor_b.as_ref().unwrap()[0],
                        se.cells.as_ref().unwrap()[0][0],
                    ];
                    let key = if tt == TableType::Means {
                        "bal_se_means"
                    } else {
                        "bal_se_effects"
                    };
                    Check::new(&case).vector(key, &got, Tol::se(n)).finish();
                    assert_eq!(r.se_kind.as_deref(), Some(kind_of(tt)));
                } else {
                    assert!(r.se.is_none());
                    assert_eq!(r.se_note.as_deref(), Some(UNBALANCED_SE_NOTE));
                }
            }
        }
        // the unbalanced b means are the projections, not the raw margins
        let (data, la, lb, _) = two_way_layout("modeltables_t4_unbal");
        let m = model_tables_two_way(&data, &la, &lb, TableType::Means, false).unwrap();
        let raw = case.vector("unbal_b_raw_means");
        assert!((m.factor_b_values[0] - raw[0]).abs() > 0.1);
        assert!(m.se.is_none() && m.se_note.is_none(), "se not asked for");
    }

    fn kind_of(tt: TableType) -> &'static str {
        match tt {
            TableType::Means => SE_KIND_MEANS,
            TableType::Effects => SE_KIND_EFFECTS,
        }
    }

    #[test]
    fn test_table_type_parses_aliases_and_rejects_unknown() {
        assert_eq!(TableType::choices().validate(), Ok(()));
        assert_eq!("Effects".parse::<TableType>().unwrap(), TableType::Effects);
        assert_eq!("mean".parse::<TableType>().unwrap(), TableType::Means);
        assert!("medians".parse::<TableType>().is_err());
    }
}
