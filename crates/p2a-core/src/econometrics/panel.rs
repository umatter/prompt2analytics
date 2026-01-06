//! Panel data estimators: Fixed Effects (FE) and Random Effects (RE).

use anyhow::{anyhow, Result};
use greeners::{Formula, FixedEffects, RandomEffects};
use ndarray::Array1;
use std::collections::HashMap;
use std::fmt;

use crate::data::Dataset;
use super::convert::polars_to_greeners;

/// Result from a panel data estimation.
#[derive(Debug, Clone)]
pub struct PanelResult {
    /// Estimation method used
    pub method: String,
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names
    pub variables: Vec<String>,
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// t-statistics
    pub t_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// R-squared
    pub r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of groups (entities)
    pub n_groups: usize,
    /// Entity variable name
    pub entity_var: String,
}

impl fmt::Display for PanelResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} Panel Regression Results", self.method)?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "Entity: {}", self.entity_var)?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "No. Groups: {}", self.n_groups)?;
        writeln!(f, "R-squared: {:.4}", self.r_squared)?;
        writeln!(f)?;
        writeln!(f, "{:<20} {:>12} {:>12} {:>10} {:>10}",
                 "Variable", "Coef", "Std Err", "t", "P>|t|")?;
        writeln!(f, "{}", "-".repeat(70))?;

        for i in 0..self.variables.len() {
            let sig = if self.p_values[i] < 0.001 {
                "***"
            } else if self.p_values[i] < 0.01 {
                "**"
            } else if self.p_values[i] < 0.05 {
                "*"
            } else if self.p_values[i] < 0.1 {
                "."
            } else {
                ""
            };

            writeln!(f, "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.3}{}",
                     self.variables[i],
                     self.coefficients[i],
                     self.std_errors[i],
                     self.t_stats[i],
                     self.p_values[i],
                     sig)?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")?;

        Ok(())
    }
}

/// Extract entity IDs from a DataFrame column and return as Vec<usize>.
fn extract_entity_ids(dataset: &Dataset, entity_var: &str) -> Result<Vec<usize>> {
    let df = dataset.df();
    let col = df.column(entity_var)
        .map_err(|e| anyhow!("Entity column '{}' not found: {}", entity_var, e))?;

    // Create a mapping from unique values to integer IDs
    let mut id_map: HashMap<String, usize> = HashMap::new();
    let mut next_id = 0usize;

    let ids: Vec<usize> = if let Ok(int_col) = col.i64() {
        int_col.into_iter()
            .map(|v| {
                let key = v.unwrap_or(0).to_string();
                *id_map.entry(key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect()
    } else if let Ok(str_col) = col.str() {
        str_col.into_iter()
            .map(|v| {
                let key = v.unwrap_or("").to_string();
                *id_map.entry(key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect()
    } else {
        // Try to cast to string
        let casted = col.cast(&polars::prelude::DataType::String)
            .map_err(|e| anyhow!("Cannot convert entity column to IDs: {}", e))?;
        let str_col = casted.str()
            .map_err(|e| anyhow!("Cannot read entity column as string: {}", e))?;
        str_col.into_iter()
            .map(|v| {
                let key = v.unwrap_or("").to_string();
                *id_map.entry(key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect()
    };

    Ok(ids)
}

/// Run Fixed Effects (within) panel estimation.
///
/// # Arguments
/// * `dataset` - The dataset containing the panel data
/// * `formula` - R-style formula (e.g., "y ~ x1 + x2")
/// * `entity_var` - Column name for entity/individual identifier
pub fn run_fixed_effects(
    dataset: &Dataset,
    formula: &str,
    entity_var: &str,
) -> Result<PanelResult> {
    // Convert Polars DataFrame to greeners DataFrame
    let gdf = polars_to_greeners(dataset.df())?;

    // Parse the formula
    let parsed_formula = Formula::parse(formula)
        .map_err(|e| anyhow!("Failed to parse formula '{}': {}", formula, e))?;

    // Extract entity IDs
    let entity_ids = extract_entity_ids(dataset, entity_var)?;
    let n_groups = entity_ids.iter().max().map(|m| m + 1).unwrap_or(0);

    // Run Fixed Effects estimation
    let result = FixedEffects::from_formula(&parsed_formula, &gdf, &entity_ids)
        .map_err(|e| anyhow!("Fixed Effects estimation failed: {}", e))?;

    // Extract results from struct fields
    let coefficients = result.params.to_vec();
    let std_errors = result.std_errors.to_vec();
    let t_stats = result.t_values.to_vec();
    let p_values = result.p_values.to_vec();
    let variables = result.variable_names
        .unwrap_or_else(|| (0..coefficients.len()).map(|i| format!("x{}", i)).collect());

    // Extract dependent variable from formula
    let dep_var = formula.split('~').next().unwrap_or("y").trim().to_string();

    Ok(PanelResult {
        method: "Fixed Effects".to_string(),
        dep_var,
        variables,
        coefficients,
        std_errors,
        t_stats,
        p_values,
        r_squared: result.r_squared,
        n_obs: result.n_obs,
        n_groups,
        entity_var: entity_var.to_string(),
    })
}

/// Run Random Effects (GLS) panel estimation.
///
/// # Arguments
/// * `dataset` - The dataset containing the panel data
/// * `formula` - R-style formula (e.g., "y ~ x1 + x2")
/// * `entity_var` - Column name for entity/individual identifier
pub fn run_random_effects(
    dataset: &Dataset,
    formula: &str,
    entity_var: &str,
) -> Result<PanelResult> {
    // Convert Polars DataFrame to greeners DataFrame
    let gdf = polars_to_greeners(dataset.df())?;

    // Parse the formula
    let parsed_formula = Formula::parse(formula)
        .map_err(|e| anyhow!("Failed to parse formula '{}': {}", formula, e))?;

    // Extract entity IDs as i64 for Random Effects (it requires Array1<i64>)
    let entity_ids_vec = extract_entity_ids(dataset, entity_var)?;
    let n_obs = entity_ids_vec.len();
    let n_groups = entity_ids_vec.iter().max().map(|m| m + 1).unwrap_or(0);
    let entity_ids_i64: Vec<i64> = entity_ids_vec.iter().map(|&x| x as i64).collect();
    let entity_ids = Array1::from(entity_ids_i64);

    // Run Random Effects estimation
    let result = RandomEffects::from_formula(&parsed_formula, &gdf, &entity_ids)
        .map_err(|e| anyhow!("Random Effects estimation failed: {}", e))?;

    // Extract results from struct fields
    let coefficients = result.params.to_vec();
    let std_errors = result.std_errors.to_vec();
    let t_stats = result.t_values.to_vec();
    let p_values = result.p_values.to_vec();

    // RandomEffectsResult doesn't have variable_names, generate from formula
    let variables: Vec<String> = parsed_formula.independents.iter()
        .map(|s| s.to_string())
        .collect();

    // Extract dependent variable from formula
    let dep_var = formula.split('~').next().unwrap_or("y").trim().to_string();

    Ok(PanelResult {
        method: "Random Effects".to_string(),
        dep_var,
        variables,
        coefficients,
        std_errors,
        t_stats,
        p_values,
        r_squared: result.r_squared_overall,
        n_obs,
        n_groups,
        entity_var: entity_var.to_string(),
    })
}
