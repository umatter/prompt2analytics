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

use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};
use crate::traits::estimator::chi_squared_p_value;

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
    /// Number of IPF cycles run (R's `nlast`)
    pub n_iter: usize,
    /// Whether the largest margin deviation of the last cycle fell below `eps`
    pub converged: bool,
    /// Largest absolute deviation between a fitted and an observed margin in the last cycle,
    /// each measured before that margin's adjustment (AS 51, R)
    #[serde(default)]
    pub max_deviation: f64,
    /// The convergence record of the IPF iteration
    #[serde(default)]
    pub convergence: Convergence,
    /// Warnings (R's "algorithm did not converge")
    #[serde(default)]
    pub warnings: Vec<String>,
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

        write!(f, "{}", notes_text(self))
    }
}

impl ResultNotes for LoglinResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        Some(self.convergence.clone())
    }
}

/// Fit a log-linear model to a contingency table using Iterative Proportional Fitting.
///
/// # Arguments
/// * `table` - Contingency table as a flattened vector (row-major order)
/// * `dimensions` - Shape of the table (e.g., [2, 3, 4] for a 2×3×4 table)
/// * `margins` - List of margins to fit. Each margin is a vector of dimension indices (0-based).
///   For example, `[[0, 1], [1, 2]]` fits the (0,1) and (1,2) two-way margins.
/// * `eps` - Convergence tolerance (default: 0.1); must be finite and greater than 0
/// * `max_iter` - Maximum iterations (default: 20); must be at least 1
///
/// # Returns
/// * `LoglinResult` containing fitted values and test statistics
///
/// # Errors
///
/// - `eps` that is not finite and greater than 0, or `max_iter` of 0.
/// - A dimension with no levels, a table whose cell count overflows `usize`, or a table whose
///   length differs from the product of `dimensions`.
/// - No margins, a margin naming a dimension the table does not have, or a margin that names the
///   same dimension more than once.
/// - A cell that is not finite or is negative, or a table whose total is not positive.
/// - A fitted table or margin buffer too large to allocate (the per-allocation ceiling
///   included), and cancellation through [`resource::checkpoint`], which is checked before every
///   margin step of every IPF iteration.
///
/// # Algorithm
///
/// Iterative Proportional Fitting (IPF), as AS 51 and R's `loglin`:
/// 1. The fitted table starts at 1 in every cell; the observed margins are computed once.
/// 2. One cycle visits each margin in order: the fitted margin is computed, its largest absolute
///    deviation from the observed margin is recorded, and the fit is rescaled to the observed
///    margin (a cell whose fitted margin is 0 becomes 0).
/// 3. After a cycle whose largest deviation is below `eps` the fit has converged; otherwise the
///    next cycle starts, up to `max_iter` cycles. A fit that stops at `max_iter` is returned with
///    `converged = false` and a warning (R: "algorithm did not converge").
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

    // Validate the parameters
    if !(eps.is_finite() && eps > 0.0) {
        return Err(format!(
            "eps must be a finite number greater than 0, got {eps}"
        ));
    }
    if max_iter < 1 {
        return Err("max_iter must be at least 1".to_string());
    }

    // Validate the table shape; the cell count uses checked arithmetic
    if let Some(i) = dimensions.iter().position(|&d| d == 0) {
        return Err(format!("dimension {i} has no levels"));
    }
    let n_cells =
        resource::checked_len(dimensions, "contingency table").map_err(|e| e.to_string())?;
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

    // A margin may name each dimension at most once
    let mut listed = vec![false; n_dims];
    for (i, margin) in margins.iter().enumerate() {
        let mut repeated = None;
        for &dim in margin {
            if listed[dim] {
                repeated = Some(dim);
                break;
            }
            listed[dim] = true;
        }
        for &dim in margin {
            listed[dim] = false;
        }
        if let Some(dim) = repeated {
            return Err(format!("margin {i} lists dimension {dim} more than once"));
        }
    }

    // Every cell must be a finite, non-negative count
    if let Some((idx, v)) = table
        .iter()
        .enumerate()
        .find(|&(_, &v)| !(v.is_finite() && v >= 0.0))
    {
        return Err(format!(
            "table cells must be finite and non-negative (cell {idx} is {v})"
        ));
    }

    // Total count
    let total: f64 = table.iter().sum();
    if total <= 0.0 {
        return Err("Table must have positive total".to_string());
    }

    // R's start: 1 in every cell
    let mut fit: Vec<f64> =
        resource::try_vec(n_cells, 1.0, "fitted table").map_err(|e| e.to_string())?;

    // The observed margins, once
    let mut observed = Vec::new();
    observed
        .try_reserve_exact(margins.len())
        .map_err(|_| "observed margins: allocation failed".to_string())?;
    for margin in margins {
        observed.push(compute_margin(table, dimensions, margin)?);
    }

    // Iterative Proportional Fitting
    let mut n_iter = 0;
    let mut converged = false;
    let mut max_dev: f64 = f64::NAN;

    for iter in 0..max_iter {
        resource::checkpoint().map_err(|e| e.to_string())?;
        n_iter = iter + 1;
        max_dev = 0.0;

        // Measure each margin, then adjust it
        for (margin, obs) in margins.iter().zip(&observed) {
            max_dev = max_dev.max(fit_one_margin(obs, &mut fit, dimensions, margin)?);
        }

        if max_dev < eps {
            converged = true;
            break;
        }
    }

    let convergence = Convergence::new(
        converged,
        n_iter,
        max_iter,
        "maximum absolute margin deviation",
        max_dev,
        eps,
    );
    let warnings: Vec<String> = convergence.note("loglin").into_iter().collect();

    // Compute test statistics
    let (lrt, pearson) = compute_test_statistics(table, &fit);

    // Compute degrees of freedom
    // df = n_cells - 1 - (number of independent parameters)
    // For log-linear models: df = n_cells - number of free parameters
    let df = n_cells
        .saturating_sub(count_free_parameters(dimensions, margins).map_err(|e| e.to_string())?);

    // Upper-tail p-values; a statistic at or below 0 (a perfect fit, or rounding below it) has
    // p = 1, as has a saturated model
    let p_value = |stat: f64| {
        if df > 0 && stat > 0.0 {
            chi_squared_p_value(stat, df as f64)
        } else {
            1.0
        }
    };
    let (p_value_lrt, p_value_pearson) = (p_value(lrt), p_value(pearson));

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
        max_deviation: max_dev,
        convergence,
        warnings,
        margins: margins.to_vec(),
        total,
        n_cells,
    })
}

/// One IPF step: return the largest absolute deviation of `fit`'s `margin` from the observed
/// margin `obs_margin`, measured before the adjustment (AS 51, R), then scale `fit` to it: each cell
/// by `obs / fitted` of its margin cell, or to 0 where the fitted margin is 0 (R's `adjust`).
///
/// A step walks the whole table twice (the fitted margin and the scaling) and a model may list
/// hundreds of margins, so the budget is checked before each step, not only once per IPF cycle.
fn fit_one_margin(
    obs_margin: &[f64],
    fit: &mut [f64],
    dimensions: &[usize],
    margin: &[usize],
) -> Result<f64, String> {
    resource::checkpoint().map_err(|e| e.to_string())?;

    let fit_margin = compute_margin(fit, dimensions, margin)?;
    let mut max_dev: f64 = 0.0;
    let mut adjustments = resource::try_vec(obs_margin.len(), 0.0, "margin adjustments")
        .map_err(|e| e.to_string())?;
    for i in 0..obs_margin.len() {
        max_dev = max_dev.max((fit_margin[i] - obs_margin[i]).abs());
        if fit_margin[i] > 0.0 {
            adjustments[i] = obs_margin[i] / fit_margin[i];
        }
    }

    for_each_margin_index(dimensions, margin, |cell_idx, margin_idx| {
        fit[cell_idx] *= adjustments[margin_idx];
    });
    Ok(max_dev)
}

/// Compute a margin (sum over specified dimensions) of a table.
///
/// The margin buffer is allocated fallibly. `margin` must not repeat a dimension.
fn compute_margin(
    table: &[f64],
    dimensions: &[usize],
    margin: &[usize],
) -> Result<Vec<f64>, String> {
    let margin_shape = margin_shape(dimensions, margin);
    let margin_size =
        resource::checked_len(&margin_shape, "margin table").map_err(|e| e.to_string())?;
    let mut result =
        resource::try_vec(margin_size, 0.0, "margin table").map_err(|e| e.to_string())?;

    for_each_margin_index(dimensions, margin, |cell_idx, margin_idx| {
        result[margin_idx] += table[cell_idx];
    });

    Ok(result)
}

/// Get the shape of a margin.
fn margin_shape(dimensions: &[usize], margin: &[usize]) -> Vec<usize> {
    margin.iter().map(|&d| dimensions[d]).collect()
}

/// Call `f(cell, margin_index)` for every cell of the table, in row-major cell order.
///
/// `margin_index` is the row-major index of the cell's coordinates in the margin table whose
/// axes are `margin` in the listed order. Coordinates are walked with an odometer (last dimension
/// fastest) and the margin index is updated with per-dimension strides, so no coordinate vector is
/// built per cell. Dimensions with one level never change coordinate and are skipped.
///
/// The caller guarantees that the product of `dimensions` fits in `usize` and that `margin` lists
/// valid dimensions, each at most once.
fn for_each_margin_index(dimensions: &[usize], margin: &[usize], mut f: impl FnMut(usize, usize)) {
    // Margin stride of each table dimension (0 when the dimension is not in the margin)
    let mut margin_stride = vec![0usize; dimensions.len()];
    let mut stride = 1usize;
    for &d in margin.iter().rev() {
        margin_stride[d] = stride;
        stride *= dimensions[d];
    }

    // Dimensions that actually vary, in table order: (levels, margin stride)
    let active: Vec<(usize, usize)> = dimensions
        .iter()
        .zip(&margin_stride)
        .filter(|&(&levels, _)| levels > 1)
        .map(|(&levels, &s)| (levels, s))
        .collect();
    let n_cells: usize = active.iter().map(|&(levels, _)| levels).product();

    let mut coords = vec![0usize; active.len()];
    let mut margin_idx = 0usize;
    for cell in 0..n_cells {
        f(cell, margin_idx);
        for (k, &(levels, s)) in active.iter().enumerate().rev() {
            coords[k] += 1;
            if coords[k] < levels {
                margin_idx += s;
                break;
            }
            coords[k] = 0;
            margin_idx -= s * (levels - 1);
        }
    }
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

/// Count the free parameters of a hierarchical log-linear model: the grand mean plus, for every
/// distinct effect generated by the margins (each margin and all of its non-empty subsets), the
/// product of `levels - 1` over the effect's dimensions. `df = n_cells - free parameters`.
///
/// A dimension with one level makes every effect that contains it contribute 0, so only the
/// dimensions with two or more levels are mapped to bits, and each margin's effects are the
/// non-empty submasks of its mask. There are at most 63 such dimensions because `2^k <= n_cells`,
/// and a margin visits at most `n_cells` submasks. The effects already counted are a bitset over
/// all `2^k` masks, allocated fallibly: `2^k / 8 <= n_cells / 8` bytes, rounded up to whole
/// 8-byte words.
fn count_free_parameters(dimensions: &[usize], margins: &[Vec<usize>]) -> EconResult<usize> {
    let mut bit_of: Vec<Option<u32>> = vec![None; dimensions.len()];
    let mut free_levels: Vec<usize> = Vec::new();
    for (d, &levels) in dimensions.iter().enumerate() {
        if levels >= 2 {
            if free_levels.len() >= 63 {
                return Err(EconError::ResourceLimit(
                    "contingency table has too many dimensions with two or more levels".to_string(),
                ));
            }
            bit_of[d] = Some(free_levels.len() as u32);
            free_levels.push(levels - 1);
        }
    }

    // One bit per effect mask; 2^k <= n_cells fits in usize, so the shift cannot overflow
    let n_masks = 1usize
        .checked_shl(free_levels.len() as u32)
        .ok_or_else(|| {
            EconError::ResourceLimit(
                "contingency table has too many dimensions with two or more levels".to_string(),
            )
        })?;
    let mut counted: Vec<u64> = resource::try_vec(n_masks.div_ceil(64), 0, "model effects")?;
    let mut n_params = 1usize; // grand mean
    let mut visited = 0u64;
    for margin in margins {
        let mask = margin
            .iter()
            .filter_map(|&d| bit_of[d])
            .fold(0u64, |m, b| m | (1u64 << b));
        let mut sub = mask;
        while sub != 0 {
            visited += 1;
            if visited.is_multiple_of(65_536) {
                resource::checkpoint()?;
            }
            let (word, bit) = ((sub / 64) as usize, 1u64 << (sub % 64));
            if counted[word] & bit == 0 {
                counted[word] |= bit;
                let mut bits = sub;
                let mut effect_size = 1usize;
                while bits != 0 {
                    effect_size =
                        effect_size.saturating_mul(free_levels[bits.trailing_zeros() as usize]);
                    bits &= bits - 1;
                }
                n_params = n_params.saturating_add(effect_size);
            }
            sub = (sub - 1) & mask;
        }
    }
    Ok(n_params)
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
    use crate::resource::{self, Budget, CancelToken};
    use p2a_validation::{Case, Check, Tol};

    /// Reference: convert a linear index to multidimensional coordinates.
    fn index_to_coords(index: usize, dimensions: &[usize]) -> Vec<usize> {
        let mut coords = vec![0; dimensions.len()];
        let mut remaining = index;
        for i in (0..dimensions.len()).rev() {
            coords[i] = remaining % dimensions[i];
            remaining /= dimensions[i];
        }
        coords
    }

    /// Reference: convert multidimensional coordinates to a linear index for a margin.
    fn coords_to_margin_index(coords: &[usize], margin: &[usize], margin_shape: &[usize]) -> usize {
        let mut idx = 0;
        let mut multiplier = 1;
        for i in (0..margin.len()).rev() {
            idx += coords[margin[i]] * multiplier;
            multiplier *= margin_shape[i];
        }
        idx
    }

    #[test]
    fn test_loglin_rejects_overflowing_dimensions() {
        // pre-fix: `dimensions.iter().product()` overflows (debug panic, release wraps)
        let e = loglin(&[1.0; 4], &[2, usize::MAX], &[vec![0], vec![1]], None, None).unwrap_err();
        assert!(e.contains("too large"), "{e}");
    }

    #[test]
    fn test_loglin_rejects_bad_parameters() {
        let t = [15.0, 15.0, 25.0, 45.0];
        let m = [vec![0], vec![1]];
        for eps in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let e = loglin(&t, &[2, 2], &m, Some(eps), None).unwrap_err();
            assert!(e.contains("eps"), "eps={eps}: {e}");
        }
        assert!(
            loglin(&t, &[2, 2], &m, None, Some(0))
                .unwrap_err()
                .contains("max_iter")
        );
        for bad in [-2.0, f64::NAN, f64::INFINITY] {
            let e = loglin(&[1.0, bad, 3.0, 4.0], &[2, 2], &m, None, None).unwrap_err();
            assert!(e.contains("non-negative"), "{e}");
        }
        assert!(
            loglin(&t, &[2, 2], &[vec![0, 0]], None, None)
                .unwrap_err()
                .contains("more than once")
        );
        assert!(
            loglin(&[], &[2, 0], &m, None, None)
                .unwrap_err()
                .contains("no levels")
        );
    }

    #[test]
    fn test_loglin_df_with_many_single_level_dimensions() {
        // pre-fix: add_margin_params computes `1 << 42` on i32 and panics (shift overflow)
        let mut dims = vec![2, 2];
        dims.extend(std::iter::repeat_n(1, 40));
        let all: Vec<usize> = (0..dims.len()).collect();
        assert_eq!(
            loglin(&[10.0, 20.0, 30.0, 40.0], &dims, &[all], None, None)
                .unwrap()
                .df,
            0
        );
        let mains: Vec<Vec<usize>> = (0..dims.len()).map(|d| vec![d]).collect();
        assert_eq!(
            loglin(&[15.0, 15.0, 25.0, 45.0], &dims, &mains, None, None)
                .unwrap()
                .df,
            1
        );
    }

    #[test]
    fn test_margin_index_matches_coordinate_reference() {
        let dims = [2usize, 3, 4];
        for margin in [
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![1, 2],
            vec![0, 2],
            vec![2, 0],
            vec![0, 1, 2],
            vec![2, 1, 0],
        ] {
            let shape: Vec<usize> = margin.iter().map(|&d| dims[d]).collect();
            let expected: Vec<usize> = (0..24)
                .map(|c| coords_to_margin_index(&index_to_coords(c, &dims), &margin, &shape))
                .collect();
            let mut got = Vec::new();
            for_each_margin_index(&dims, &margin, |cell, m| {
                assert_eq!(cell, got.len());
                got.push(m);
            });
            assert_eq!(got, expected, "margin {margin:?}");
        }
    }

    // R 4.3.3: obs <- c(12,7,9,15, 8,11,14,6, 10,13,5,9, 6,14,11,8, 13,9,7,12, 9,6,15,10)
    //   A <- array(0, c(2,3,4)); for (i in 0:1) for (j in 0:2) for (k in 0:3) A[i+1,j+1,k+1] <- obs[i*12+j*4+k+1]
    //   r <- loglin(A, list(c(1,2), c(2,3)), fit = TRUE); r$fit flattened back to row-major
    const LOGLIN_OBS: [f64; 24] = [
        12., 7., 9., 15., 8., 11., 14., 6., 10., 13., 5., 9., 6., 14., 11., 8., 13., 9., 7., 12.,
        9., 6., 15., 10.,
    ];
    const R_FIT: [f64; 24] = [
        9.439024390243903,
        11.012195121951219,
        10.487804878048781,
        12.060975609756097,
        10.2375,
        9.75,
        10.2375,
        8.775,
        9.12987012987013,
        9.12987012987013,
        9.61038961038961,
        9.12987012987013,
        8.560975609756097,
        9.987804878048781,
        9.512195121951219,
        10.939024390243903,
        10.7625,
        10.25,
        10.7625,
        9.225,
        9.87012987012987,
        9.87012987012987,
        10.38961038961039,
        9.87012987012987,
    ];

    #[test]
    fn test_validate_loglin_decomposable_against_r() {
        let r = loglin(
            &LOGLIN_OBS,
            &[2, 3, 4],
            &[vec![0, 1], vec![1, 2]],
            None,
            None,
        )
        .unwrap();
        assert!((r.lrt - 20.21839799570819).abs() < 1e-10, "{}", r.lrt);
        assert!(
            (r.pearson - 19.740264919955614).abs() < 1e-10,
            "{}",
            r.pearson
        );
        assert!((r.p_value_lrt - 0.016611401640345802).abs() < 1e-10);
        assert_eq!(r.df, 9);
        for (f, e) in r.fit.iter().zip(R_FIT) {
            assert!((f - e).abs() < 1e-10, "{f} vs {e}");
        }
    }

    #[test]
    fn test_validate_loglin_df_against_r() {
        // R: loglin(A, m, print = FALSE)$df on any 2x3x4 table with positive cells
        let cases: Vec<(Vec<Vec<usize>>, usize)> = vec![
            (vec![vec![0, 1], vec![1, 2]], 9),
            (vec![vec![0, 1], vec![0, 2], vec![1, 2]], 6),
            (vec![vec![0], vec![1, 2]], 11),
            (vec![vec![0], vec![1], vec![2]], 17),
            (vec![vec![0, 1, 2]], 0),
        ];
        for (margins, df) in cases {
            assert_eq!(
                loglin(&LOGLIN_OBS, &[2, 3, 4], &margins, None, None)
                    .unwrap()
                    .df,
                df,
                "{margins:?}"
            );
        }
    }

    const M3: [&[usize]; 3] = [&[0, 1], &[0, 2], &[1, 2]];

    fn m3() -> Vec<Vec<usize>> {
        M3.iter().map(|m| m.to_vec()).collect()
    }

    /// The case's long-form table (Var1 A..B, Var2 A..C, Var3 A..D, Freq), row-major in dims
    /// [2, 3, 4] (last index fastest).
    fn case_table(ds: &crate::data::Dataset) -> Vec<f64> {
        let df = ds.df();
        let level = |name: &str| -> Vec<usize> {
            df.column(name)
                .unwrap()
                .str()
                .unwrap()
                .into_iter()
                .map(|s| (s.unwrap().as_bytes()[0] - b'A') as usize)
                .collect()
        };
        let (a, b, c) = (level("Var1"), level("Var2"), level("Var3"));
        let (freq, _) =
            crate::data::columns::numeric_column(df, "Freq", crate::data::columns::NaPolicy::Fail)
                .unwrap();
        let mut t = vec![0.0; 24];
        for i in 0..freq.len() {
            t[a[i] * 12 + b[i] * 4 + c[i]] = freq[i];
        }
        t
    }

    #[test]
    fn test_validate_loglin_nodecomp() {
        // R: loglin(A, list(c(1,2), c(1,3), c(2,3)), eps = 1e-12, iter = 1000, fit = TRUE) on
        // loglin_2x3x4 (no three-way interaction: not decomposable, so IPF iterates), and the
        // decomposable list(c(1,2), c(1,3)). Pinned at the solution; convergence at R's defaults
        // (eps 0.1, iter 20) and non-convergence after one cycle hold for any correct IPF.
        let (case, ds) = crate::test_support::load_case("stat_loglin_nodecomp");
        let table = case_table(&ds);
        let n = table.iter().sum::<f64>() as usize; // 652
        let r = loglin(&table, &[2, 3, 4], &m3(), Some(1e-12), Some(1000)).unwrap();
        let d = loglin(&table, &[2, 3, 4], &m3(), None, None).unwrap();
        let one = loglin(&table, &[2, 3, 4], &m3(), None, Some(1)).unwrap();
        let dec = loglin(
            &table,
            &[2, 3, 4],
            &[vec![0, 1], vec![0, 2]],
            Some(1e-12),
            Some(1000),
        )
        .unwrap();
        assert!(r.converged && d.converged && !one.converged && dec.converged);
        Check::new(&case)
            .vector("counts_rowmajor", &table, Tol::exact())
            .scalar("lrt", r.lrt, Tol::stat(n))
            .scalar("pearson", r.pearson, Tol::stat(n))
            .scalar("df", r.df as f64, Tol::exact())
            .scalar("p_lrt", r.p_value_lrt, Tol::p_value(n))
            .scalar("p_pearson", r.p_value_pearson, Tol::p_value(n))
            .vector("fit_rowmajor", &r.fit, Tol::stat(n))
            .scalar("decomposable_lrt", dec.lrt, Tol::stat(n))
            .scalar("decomposable_df", dec.df as f64, Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_loglin_nodecomp_2x2x2() {
        // STAT-28's repro, laid out row-major in dims (2, 2, 2). R: G2 8.1479 at eps 1e-10
        // (24 cycles; 8.148 after 5 at R's default eps 0.1). One cycle gives 10.0185, the value
        // the pre-fix code reported as converged.
        let case = Case::load("stat_loglin_nodecomp");
        let t = case.vector("t222_counts");
        let d = loglin(&t, &[2, 2, 2], &m3(), None, None).unwrap();
        let e = loglin(&t, &[2, 2, 2], &m3(), Some(1e-10), Some(1000)).unwrap();
        let one = loglin(&t, &[2, 2, 2], &m3(), None, Some(1)).unwrap();
        assert!(d.converged && e.converged);
        assert!(
            !one.converged && one.warnings.iter().any(|w| w.contains("did not converge")),
            "{:?}",
            one.warnings
        );
        Check::new(&case)
            .scalar("t222_eps10_lrt", e.lrt, Tol::stat(180))
            .scalar("t222_eps10_pearson", e.pearson, Tol::stat(180))
            .scalar("t222_eps10_p_lrt", e.p_value_lrt, Tol::p_value(180))
            .scalar("t222_eps10_p_pearson", e.p_value_pearson, Tol::p_value(180))
            .vector("t222_eps10_fit_rowmajor", &e.fit, Tol::stat(180))
            .scalar("t222_df", e.df as f64, Tol::exact())
            .finish();
    }

    #[test]
    fn test_loglin_reports_nonconvergence() {
        // R: loglin(.., eps = 1e-12, iter = 3) warns "algorithm did not converge" and returns the
        // 3-cycle fit. Not R-pinned: the case's table, no R value compared.
        let path = p2a_validation::repo_path("validation/datasets/generated/loglin_2x3x4.csv");
        let ds = crate::data::Dataset::new(crate::data::DataLoader::load_csv(path).unwrap());
        let table = case_table(&ds);
        let r = loglin(&table, &[2, 3, 4], &m3(), Some(1e-12), Some(3)).unwrap();
        assert!(
            !r.converged
                && r.n_iter == 3
                && !r.convergence.converged
                && r.convergence.iterations == 3
        );
        assert!(r.max_deviation > 1e-12 && r.max_deviation.is_finite());
        assert!(
            r.warnings
                .iter()
                .any(|w| w.starts_with("loglin did not converge after 3 iterations")),
            "{:?}",
            r.warnings
        );
        let text = format!("{r}");
        assert!(text.contains("Converged: no (3 iterations"), "{text}");
    }

    #[test]
    fn test_loglin_honours_cancellation() {
        let t = CancelToken::new();
        t.cancel();
        let e = resource::with_budget(Budget::new(None, t), || {
            loglin(
                &[15.0, 15.0, 25.0, 45.0],
                &[2, 2],
                &[vec![0], vec![1]],
                None,
                None,
            )
        })
        .unwrap_err();
        assert!(e.contains("stopped"), "{e}");
    }

    #[test]
    fn test_ipf_margin_step_checks_the_budget() {
        // One IPF iteration can walk a 1M-cell table twice for each of 500 margins, so each
        // margin step checks the budget itself, not only the iteration around it
        let t = CancelToken::new();
        t.cancel();
        let mut fit = vec![25.0; 4];
        let e = resource::with_budget(Budget::new(None, t), || {
            fit_one_margin(&[30.0, 70.0], &mut fit, &[2, 2], &[0])
        })
        .unwrap_err();
        assert!(e.contains("stopped"), "{e}");
        assert_eq!(
            fit,
            vec![25.0; 4],
            "a refused step leaves the fit untouched"
        );

        // Without a budget the step reports the deviation of the fitted margin (50, 50) from the
        // observed one (30, 70), measured before the adjustment, and scales the fit to it
        let observed = compute_margin(&[15.0, 15.0, 25.0, 45.0], &[2, 2], &[0]).unwrap();
        let dev = fit_one_margin(&observed, &mut fit, &[2, 2], &[0]).unwrap();
        assert_eq!(fit, vec![15.0, 15.0, 35.0, 35.0]);
        assert_eq!(dev, 20.0);
    }

    #[test]
    fn test_count_free_parameters_allocates_effects_fallibly() {
        // 20 two-level dimensions and the saturated margin: 2^20 effect masks, a 128 KiB bitset.
        // Pre-fix the counted effects were an infallible HashSet of about 1M entries.
        let dims = vec![2usize; 20];
        let all: Vec<usize> = (0..20).collect();
        let e = resource::with_alloc_ceiling(Some(64 * 1024), || {
            count_free_parameters(&dims, std::slice::from_ref(&all))
        })
        .unwrap_err();
        assert!(
            matches!(&e, EconError::ResourceLimit(m) if m.contains("model effects")),
            "{e}"
        );
        // Saturated: every one of the 2^20 cells is a parameter
        assert_eq!(count_free_parameters(&dims, &[all]).unwrap(), 1 << 20);
    }

    #[test]
    fn test_loglin_respects_allocation_ceiling() {
        // 24 cells x 8 bytes = 192 bytes > 64
        let e = resource::with_alloc_ceiling(Some(64), || {
            loglin(&LOGLIN_OBS, &[2, 3, 4], &[vec![0, 1]], None, None)
        })
        .unwrap_err();
        assert!(e.contains("limit"), "{e}");
    }

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
        let row_margin = compute_margin(&table, &dims, &[0]).unwrap();
        assert_eq!(row_margin, vec![6.0, 15.0]); // 1+2+3, 4+5+6

        // Column margin (sum over rows)
        let col_margin = compute_margin(&table, &dims, &[1]).unwrap();
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
