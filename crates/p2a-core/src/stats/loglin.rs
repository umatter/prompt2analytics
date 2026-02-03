//! Log-linear Models for Contingency Tables (loglin)
//!
//! Fits log-linear models to multidimensional contingency tables using
//! Iterative Proportional Fitting (IPF).
//!
//! # References
//!
//! - Haberman, S. J. (1972). "Log-Linear Fit for Contingency Tables—Algorithm AS 51".
//!   Journal of the Royal Statistical Society. Series C (Applied Statistics), 21(2), 218-225.
//! - Agresti, A. (2002). Categorical Data Analysis (2nd ed.). Wiley-Interscience.
//! - R stats::loglin documentation
//!   Source: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/loglin.html

use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Result of log-linear model fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoglinResult {
    /// Likelihood Ratio Test statistic (G²)
    pub lrt: f64,
    /// Pearson chi-squared statistic (X²)
    pub pearson: f64,
    /// Degrees of freedom
    pub df: usize,
    /// P-value for the LRT
    pub p_value_lrt: f64,
    /// P-value for Pearson chi-squared
    pub p_value_pearson: f64,
    /// Fitted values (flattened array)
    pub fit: Vec<f64>,
    /// Original table dimensions
    pub dimensions: Vec<usize>,
    /// Number of iterations until convergence
    pub n_iter: usize,
    /// Whether the model converged
    pub converged: bool,
    /// Margins used in the model
    pub margins: Vec<Vec<usize>>,
    /// Total count in the table
    pub total: f64,
    /// Number of cells
    pub n_cells: usize,
}

impl std::fmt::Display for LoglinResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Log-linear Model Results")?;
        writeln!(f, "========================")?;
        writeln!(f)?;
        writeln!(f, "Model specification:")?;
        writeln!(f, "  Margins: {:?}", self.margins)?;
        writeln!(f, "  Table dimensions: {:?}", self.dimensions)?;
        writeln!(f, "  Total cells: {}", self.n_cells)?;
        writeln!(f, "  Total count: {:.0}", self.total)?;
        writeln!(f)?;
        writeln!(f, "Goodness of Fit:")?;
        writeln!(f, "  Likelihood Ratio Test (G²): {:.4}", self.lrt)?;
        writeln!(f, "  Pearson Chi-squared (X²):   {:.4}", self.pearson)?;
        writeln!(f, "  Degrees of freedom:         {}", self.df)?;
        writeln!(f, "  P-value (LRT):              {:.4}", self.p_value_lrt)?;
        writeln!(
            f,
            "  P-value (Pearson):          {:.4}",
            self.p_value_pearson
        )?;
        writeln!(f)?;
        writeln!(f, "Convergence:")?;
        writeln!(f, "  Converged: {}", self.converged)?;
        writeln!(f, "  Iterations: {}", self.n_iter)?;
        writeln!(f)?;

        // Show interpretation
        let sig = if self.p_value_lrt < 0.05 {
            "significant"
        } else {
            "not significant"
        };
        writeln!(f, "Interpretation:")?;
        writeln!(
            f,
            "  The model fit is {} at α=0.05 (p={:.4}).",
            sig, self.p_value_lrt
        )?;
        if self.p_value_lrt >= 0.05 {
            writeln!(f, "  This suggests the model adequately fits the data.")?;
        } else {
            writeln!(
                f,
                "  This suggests the model does not adequately fit the data."
            )?;
        }

        Ok(())
    }
}

/// Fit a log-linear model to a contingency table using Iterative Proportional Fitting.
///
/// # Arguments
/// * `table` - Contingency table as a flattened vector (row-major order)
/// * `dimensions` - Shape of the table (e.g., [2, 3, 4] for a 2×3×4 table)
/// * `margins` - List of margins to fit. Each margin is a vector of dimension indices (0-based).
///   For example, `[[0, 1], [1, 2]]` fits the (0,1) and (1,2) two-way margins.
/// * `eps` - Convergence tolerance (default: 0.1)
/// * `max_iter` - Maximum iterations (default: 20)
///
/// # Returns
/// * `LoglinResult` containing fitted values and test statistics
///
/// # Algorithm
///
/// Iterative Proportional Fitting (IPF):
/// 1. Initialize fitted values to uniform (total / n_cells)
/// 2. For each margin in the model:
///    a. Compute observed margin
///    b. Compute fitted margin
///    c. Adjust fitted values so fitted margin equals observed margin
/// 3. Repeat until convergence (max deviation < eps)
///
/// # References
///
/// - Haberman (1972). Algorithm AS 51.
pub fn loglin(
    table: &[f64],
    dimensions: &[usize],
    margins: &[Vec<usize>],
    eps: Option<f64>,
    max_iter: Option<usize>,
) -> Result<LoglinResult, String> {
    let eps = eps.unwrap_or(0.1);
    let max_iter = max_iter.unwrap_or(20);

    // Validate inputs
    let n_cells: usize = dimensions.iter().product();
    if table.len() != n_cells {
        return Err(format!(
            "Table length ({}) doesn't match dimensions product ({})",
            table.len(),
            n_cells
        ));
    }

    if margins.is_empty() {
        return Err("At least one margin must be specified".to_string());
    }

    let n_dims = dimensions.len();
    for (i, margin) in margins.iter().enumerate() {
        for &dim in margin {
            if dim >= n_dims {
                return Err(format!(
                    "Margin {} contains dimension {} but table only has {} dimensions",
                    i, dim, n_dims
                ));
            }
        }
    }

    // Total count
    let total: f64 = table.iter().sum();
    if total <= 0.0 {
        return Err("Table must have positive total".to_string());
    }

    // Initialize fitted values uniformly
    let mut fit: Vec<f64> = vec![total / n_cells as f64; n_cells];

    // Iterative Proportional Fitting
    let mut n_iter = 0;
    let mut converged = false;

    for iter in 0..max_iter {
        n_iter = iter + 1;
        let mut max_dev: f64 = 0.0;

        // Apply each margin
        for margin in margins {
            // Compute observed margin
            let obs_margin = compute_margin(table, dimensions, margin);

            // Compute fitted margin
            let fit_margin = compute_margin(&fit, dimensions, margin);

            // Compute adjustment factors
            let margin_size = margin_shape(dimensions, margin);
            let mut adjustments = vec![0.0; obs_margin.len()];
            for i in 0..obs_margin.len() {
                if fit_margin[i] > 1e-10 {
                    adjustments[i] = obs_margin[i] / fit_margin[i];
                } else {
                    adjustments[i] = 1.0;
                }
            }

            // Apply adjustments to fitted values
            for cell_idx in 0..n_cells {
                let coords = index_to_coords(cell_idx, dimensions);
                let margin_idx = coords_to_margin_index(&coords, margin, &margin_size);
                fit[cell_idx] *= adjustments[margin_idx];
            }

            // Check convergence for this margin
            let new_fit_margin = compute_margin(&fit, dimensions, margin);
            for i in 0..obs_margin.len() {
                let dev = (new_fit_margin[i] - obs_margin[i]).abs();
                max_dev = max_dev.max(dev);
            }
        }

        if max_dev < eps {
            converged = true;
            break;
        }
    }

    // Compute test statistics
    let (lrt, pearson) = compute_test_statistics(table, &fit);

    // Compute degrees of freedom
    // df = n_cells - 1 - (number of independent parameters)
    // For log-linear models: df = n_cells - number of free parameters
    let df = compute_degrees_of_freedom(dimensions, margins, n_cells);

    // Compute p-values
    let (p_value_lrt, p_value_pearson) = if df > 0 {
        let chi_sq = ChiSquared::new(df as f64).unwrap();
        (1.0 - chi_sq.cdf(lrt), 1.0 - chi_sq.cdf(pearson))
    } else {
        (1.0, 1.0)
    };

    Ok(LoglinResult {
        lrt,
        pearson,
        df,
        p_value_lrt,
        p_value_pearson,
        fit,
        dimensions: dimensions.to_vec(),
        n_iter,
        converged,
        margins: margins.to_vec(),
        total,
        n_cells,
    })
}

/// Compute a margin (sum over specified dimensions) of a table.
fn compute_margin(table: &[f64], dimensions: &[usize], margin: &[usize]) -> Vec<f64> {
    let margin_shape = margin_shape(dimensions, margin);
    let margin_size: usize = margin_shape.iter().product();
    let mut result = vec![0.0; margin_size];

    for (cell_idx, &value) in table.iter().enumerate() {
        let coords = index_to_coords(cell_idx, dimensions);
        let margin_idx = coords_to_margin_index(&coords, margin, &margin_shape);
        result[margin_idx] += value;
    }

    result
}

/// Get the shape of a margin.
fn margin_shape(dimensions: &[usize], margin: &[usize]) -> Vec<usize> {
    margin.iter().map(|&d| dimensions[d]).collect()
}

/// Convert a linear index to multidimensional coordinates.
fn index_to_coords(index: usize, dimensions: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; dimensions.len()];
    let mut remaining = index;
    for i in (0..dimensions.len()).rev() {
        coords[i] = remaining % dimensions[i];
        remaining /= dimensions[i];
    }
    coords
}

/// Convert multidimensional coordinates to a linear index for a margin.
fn coords_to_margin_index(coords: &[usize], margin: &[usize], margin_shape: &[usize]) -> usize {
    let mut idx = 0;
    let mut multiplier = 1;
    for i in (0..margin.len()).rev() {
        idx += coords[margin[i]] * multiplier;
        multiplier *= margin_shape[i];
    }
    idx
}

/// Compute test statistics: Likelihood Ratio (G²) and Pearson Chi-squared (X²).
fn compute_test_statistics(observed: &[f64], fitted: &[f64]) -> (f64, f64) {
    let mut lrt = 0.0;
    let mut pearson = 0.0;

    for (o, f) in observed.iter().zip(fitted.iter()) {
        if *o > 0.0 && *f > 0.0 {
            lrt += 2.0 * o * (o / f).ln();
        }
        if *f > 0.0 {
            let diff = o - f;
            pearson += diff * diff / f;
        }
    }

    (lrt, pearson)
}

/// Compute degrees of freedom for the log-linear model.
fn compute_degrees_of_freedom(
    dimensions: &[usize],
    margins: &[Vec<usize>],
    n_cells: usize,
) -> usize {
    // For a saturated model: df = 0
    // For an independence model: df = n_cells - 1 - sum(dim_i - 1)

    // Count free parameters for each margin
    // This is a simplification; exact calculation depends on the model structure
    let mut n_params = 1; // grand mean

    // Collect unique effects
    let mut counted_effects: std::collections::HashSet<Vec<usize>> =
        std::collections::HashSet::new();

    for margin in margins {
        // Add parameters for this margin and all its sub-margins
        add_margin_params(&mut counted_effects, margin, dimensions);
    }

    for effect in &counted_effects {
        let effect_size: usize = effect.iter().map(|&d| dimensions[d] - 1).product();
        n_params += effect_size;
    }

    // df = n_cells - 1 - (n_params - 1) = n_cells - n_params
    n_cells.saturating_sub(n_params)
}

/// Add a margin and all its sub-margins to the set of effects.
fn add_margin_params(
    effects: &mut std::collections::HashSet<Vec<usize>>,
    margin: &[usize],
    _dimensions: &[usize],
) {
    // Add all subsets of the margin (including the margin itself)
    let n = margin.len();
    for mask in 1..(1 << n) {
        let mut subset: Vec<usize> = Vec::new();
        for i in 0..n {
            if (mask >> i) & 1 == 1 {
                subset.push(margin[i]);
            }
        }
        subset.sort();
        effects.insert(subset);
    }
}

/// Fit a simple independence model for a two-way table.
///
/// This is a convenience function for the common case of testing
/// independence in a 2D contingency table.
pub fn loglin_independence(
    table: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Result<LoglinResult, String> {
    let dimensions = vec![n_rows, n_cols];
    // Independence model: only main effects (no interaction)
    let margins = vec![vec![0], vec![1]];
    loglin(table, &dimensions, &margins, None, None)
}

/// Fit a saturated model (includes all interactions).
///
/// The saturated model fits the data perfectly (df=0).
pub fn loglin_saturated(table: &[f64], dimensions: &[usize]) -> Result<LoglinResult, String> {
    // Saturated model includes all dimensions
    let all_dims: Vec<usize> = (0..dimensions.len()).collect();
    loglin(table, dimensions, &[all_dims], None, None)
}

/// Convenience function to run loglin.
pub fn run_loglin(
    table: &[f64],
    dimensions: &[usize],
    margins: &[Vec<usize>],
    eps: Option<f64>,
    max_iter: Option<usize>,
) -> Result<LoglinResult, String> {
    loglin(table, dimensions, margins, eps, max_iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loglin_2x2_independence() {
        // 2x2 table: testing independence
        // Expected: 10, 20, 30, 40 but observed: 15, 15, 25, 45
        let observed = vec![15.0, 15.0, 25.0, 45.0];
        let result = loglin_independence(&observed, 2, 2).unwrap();

        assert_eq!(result.dimensions, vec![2, 2]);
        assert!(result.converged);
        assert!(result.lrt >= 0.0);
        assert!(result.pearson >= 0.0);
        assert_eq!(result.df, 1); // df = (r-1)(c-1) = 1 for 2x2 independence
    }

    #[test]
    fn test_loglin_perfect_independence() {
        // Perfectly independent 2x2 table
        // Row sums: 40, 60 -> proportions 0.4, 0.6
        // Col sums: 50, 50 -> proportions 0.5, 0.5
        // Expected: 20, 20, 30, 30
        let observed = vec![20.0, 20.0, 30.0, 30.0];
        let result = loglin_independence(&observed, 2, 2).unwrap();

        // G² and X² should be near 0 for perfect independence
        assert!(result.lrt < 1e-10, "LRT should be ~0, got {}", result.lrt);
        assert!(
            result.pearson < 1e-10,
            "Pearson should be ~0, got {}",
            result.pearson
        );
        assert!(result.p_value_lrt > 0.99);
    }

    #[test]
    fn test_loglin_3way() {
        // 2x2x2 table
        let observed = vec![
            10.0, 20.0, // [0,0,*]
            30.0, 40.0, // [0,1,*]
            15.0, 25.0, // [1,0,*]
            35.0, 45.0, // [1,1,*]
        ];
        let dimensions = vec![2, 2, 2];

        // Model: (0,1) and (1,2) interactions (no 3-way)
        let margins = vec![vec![0, 1], vec![1, 2]];
        let result = loglin(&observed, &dimensions, &margins, None, None).unwrap();

        assert_eq!(result.dimensions, vec![2, 2, 2]);
        assert!(result.converged);
        assert_eq!(result.n_cells, 8);
    }

    #[test]
    fn test_loglin_saturated() {
        // Saturated model should have df=0 and fit perfectly
        let observed = vec![10.0, 20.0, 30.0, 40.0];
        let result = loglin_saturated(&observed, &[2, 2]).unwrap();

        assert_eq!(result.df, 0);
        // Fitted should equal observed (approximately)
        for (o, f) in observed.iter().zip(result.fit.iter()) {
            assert!((o - f).abs() < 1e-6, "o={}, f={}", o, f);
        }
    }

    #[test]
    fn test_index_to_coords() {
        let dims = vec![2, 3, 4];

        let coords = index_to_coords(0, &dims);
        assert_eq!(coords, vec![0, 0, 0]);

        let coords = index_to_coords(1, &dims);
        assert_eq!(coords, vec![0, 0, 1]);

        let coords = index_to_coords(4, &dims);
        assert_eq!(coords, vec![0, 1, 0]);

        let coords = index_to_coords(23, &dims); // Last cell: 2*3*4 - 1
        assert_eq!(coords, vec![1, 2, 3]);
    }

    #[test]
    fn test_compute_margin() {
        // 2x3 table
        let table = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
        ];
        let dims = vec![2, 3];

        // Row margin (sum over columns)
        let row_margin = compute_margin(&table, &dims, &[0]);
        assert_eq!(row_margin, vec![6.0, 15.0]); // 1+2+3, 4+5+6

        // Column margin (sum over rows)
        let col_margin = compute_margin(&table, &dims, &[1]);
        assert_eq!(col_margin, vec![5.0, 7.0, 9.0]); // 1+4, 2+5, 3+6
    }

    #[test]
    fn test_loglin_validation() {
        // Wrong table size
        let result = loglin(&[1.0, 2.0], &[2, 2], &[vec![0]], None, None);
        assert!(result.is_err());

        // Empty margins
        let result = loglin(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &[], None, None);
        assert!(result.is_err());

        // Invalid dimension in margin
        let result = loglin(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &[vec![5]], None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_loglin_display() {
        let observed = vec![20.0, 20.0, 30.0, 30.0];
        let result = loglin_independence(&observed, 2, 2).unwrap();
        let display = format!("{}", result);

        assert!(display.contains("Log-linear Model Results"));
        assert!(display.contains("Likelihood Ratio Test"));
        assert!(display.contains("Degrees of freedom"));
    }

    #[test]
    fn test_loglin_convergence() {
        // Test that IPF converges reasonably fast
        let observed = vec![10.0, 20.0, 30.0, 40.0, 15.0, 25.0];
        let result = loglin(&observed, &[2, 3], &[vec![0], vec![1]], None, None).unwrap();

        assert!(result.converged);
        assert!(result.n_iter <= 20);
    }
}
