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
//! Standard error: SE(μ̂ᵢ) = √(MSE/nᵢ)
//!
//! ## Table of Effects
//!
//! The effect for level i is the deviation from the grand mean:
//!
//! αᵢ = μ̂ᵢ - μ̂.. where Σᵢ nᵢαᵢ = 0
//!
//! For balanced designs: αᵢ = ȳᵢ. - ȳ..
//!
//! ## Two-Way ANOVA
//!
//! For factors A (levels i) and B (levels j):
//!
//! - Main effect A: αᵢ = μ̂ᵢ. - μ̂..
//! - Main effect B: βⱼ = μ̂.ⱼ - μ̂..
//! - Interaction: (αβ)ᵢⱼ = μ̂ᵢⱼ - μ̂ᵢ. - μ̂.ⱼ + μ̂..
//!
//! ## Standard Errors
//!
//! Using pooled MSE from ANOVA:
//! - For means: SE = √(MSE/n)
//! - For effects: SE = √(MSE × (1/nᵢ + 1/n..))  approximately
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

/// Type of table to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TableType {
    /// Table of means
    #[default]
    Means,
    /// Table of effects (deviations from grand mean)
    Effects,
}

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
    /// Standard errors (if applicable)
    pub se: Option<Vec<f64>>,
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
    /// Factor B values (means or effects)
    pub factor_b_values: Vec<f64>,
    /// Factor B sample sizes
    pub factor_b_n: Vec<usize>,
    /// Cell means or effects (factor_a x factor_b)
    pub cell_values: Option<Vec<Vec<f64>>>,
    /// Cell sample sizes
    pub cell_n: Option<Vec<Vec<usize>>>,
    /// Interaction effects (if applicable)
    pub interaction: Option<Vec<Vec<f64>>>,
    /// Standard errors
    pub se: Option<ModelTablesSE>,
    /// Mean squared error from ANOVA
    pub mse: f64,
    /// Degrees of freedom for MSE
    pub df_mse: f64,
}

/// Standard errors for model tables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTablesSE {
    /// SE for factor A means/effects
    pub factor_a: Option<Vec<f64>>,
    /// SE for factor B means/effects
    pub factor_b: Option<Vec<f64>>,
    /// SE for cell means/effects
    pub cells: Option<Vec<Vec<f64>>>,
}

/// Compute model tables from one-way ANOVA result.
///
/// Returns tables of means or effects (deviations from grand mean)
/// along with standard errors.
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

    let se = if compute_se {
        // SE for group means: sqrt(MSE / n_i)
        // SE for effects: same as for means in one-way ANOVA
        Some(group_ns.iter().map(|&n| (mse / n as f64).sqrt()).collect())
    } else {
        None
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
/// A `TwoWayModelTablesResult` containing the requested tables
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

    // Compute cell statistics
    let mut cell_means = vec![vec![0.0; b_levels]; a_levels];
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
    let df_within = total_n.saturating_sub(a_levels * b_levels);
    let mse = if df_within > 0 {
        ss_within / df_within as f64
    } else {
        0.0
    };

    // Compute marginal means for factor A
    let mut factor_a_means = vec![0.0; a_levels];
    let mut factor_a_n = vec![0usize; a_levels];
    for i in 0..a_levels {
        let mut sum = 0.0;
        let mut n = 0;
        for j in 0..b_levels {
            for &x in &data[i][j] {
                sum += x;
                n += 1;
            }
        }
        if n > 0 {
            factor_a_means[i] = sum / n as f64;
            factor_a_n[i] = n;
        }
    }

    // Compute marginal means for factor B
    let mut factor_b_means = vec![0.0; b_levels];
    let mut factor_b_n = vec![0usize; b_levels];
    for j in 0..b_levels {
        let mut sum = 0.0;
        let mut n = 0;
        for i in 0..a_levels {
            for &x in &data[i][j] {
                sum += x;
                n += 1;
            }
        }
        if n > 0 {
            factor_b_means[j] = sum / n as f64;
            factor_b_n[j] = n;
        }
    }

    // Compute values based on table type
    let (factor_a_values, factor_b_values, cell_values, interaction) = match table_type {
        TableType::Means => (
            factor_a_means.clone(),
            factor_b_means.clone(),
            Some(cell_means.clone()),
            None,
        ),
        TableType::Effects => {
            // Main effects: deviation from grand mean
            let a_effects: Vec<f64> = factor_a_means.iter().map(|m| m - grand_mean).collect();
            let b_effects: Vec<f64> = factor_b_means.iter().map(|m| m - grand_mean).collect();

            // Cell effects: cell_mean - grand_mean
            let cell_effects: Vec<Vec<f64>> = cell_means
                .iter()
                .map(|row| row.iter().map(|m| m - grand_mean).collect())
                .collect();

            // Interaction effects: cell_effect - a_effect - b_effect
            let interaction_effects: Vec<Vec<f64>> = (0..a_levels)
                .map(|i| {
                    (0..b_levels)
                        .map(|j| cell_effects[i][j] - a_effects[i] - b_effects[j])
                        .collect()
                })
                .collect();

            (
                a_effects,
                b_effects,
                Some(cell_effects),
                Some(interaction_effects),
            )
        }
    };

    // Compute standard errors
    let se = if compute_se && mse > 0.0 {
        let factor_a_se: Vec<f64> = factor_a_n
            .iter()
            .map(|&n| {
                if n > 0 {
                    (mse / n as f64).sqrt()
                } else {
                    f64::NAN
                }
            })
            .collect();

        let factor_b_se: Vec<f64> = factor_b_n
            .iter()
            .map(|&n| {
                if n > 0 {
                    (mse / n as f64).sqrt()
                } else {
                    f64::NAN
                }
            })
            .collect();

        let cell_se: Vec<Vec<f64>> = cell_n
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&n| {
                        if n > 0 {
                            (mse / n as f64).sqrt()
                        } else {
                            f64::NAN
                        }
                    })
                    .collect()
            })
            .collect();

        Some(ModelTablesSE {
            factor_a: Some(factor_a_se),
            factor_b: Some(factor_b_se),
            cells: Some(cell_se),
        })
    } else {
        None
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
        mse,
        df_mse: df_within as f64,
    })
}

/// Convenience wrapper for model.tables.
pub fn run_model_tables(
    anova: &AnovaResult,
    table_type: TableType,
    compute_se: bool,
) -> Result<ModelTablesResult, String> {
    model_tables(anova, table_type, compute_se)
}

/// Format model tables result as a string table.
pub fn format_model_tables(result: &ModelTablesResult) -> String {
    let mut output = String::new();

    output.push_str(&format!("Tables of {} \n\n", result.table_type));
    output.push_str(&format!("Grand mean: {:.4}\n\n", result.grand_mean));

    // Header
    output.push_str(" Group     |     Value     |      n      ");
    if result.se.is_some() {
        output.push_str("|      SE      ");
    }
    output.push('\n');
    output.push_str("-----------+---------------+-------------");
    if result.se.is_some() {
        output.push_str("+--------------");
    }
    output.push('\n');

    // Data rows
    for (i, name) in result.group_names.iter().enumerate() {
        output.push_str(&format!(
            " {:<9} | {:>13.4} | {:>11} ",
            name, result.values[i], result.n[i]
        ));
        if let Some(ref se) = result.se {
            output.push_str(&format!("| {:>12.4} ", se[i]));
        }
        output.push('\n');
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
}
