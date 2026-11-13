//! High-Dimensional Fixed Effects (HDFE) estimator.
//!
//! Implements the Method of Alternating Projections (MAP) for efficient estimation
//! of linear models with multiple high-dimensional fixed effects. This approach
//! avoids creating dummy variables by iteratively demeaning the data until convergence.
//!
//! # References
//!
//! - Gaure, S. (2013). "lfe: Linear Group Fixed Effects". *The R Journal*, 5(2), 104-117.
//!   <https://journal.r-project.org/articles/RJ-2013-031/>
//! - Guimarães, P. & Portugal, P. (2010). "A Simple Feasible Procedure to Fit Models
//!   with High-Dimensional Fixed Effects". *Stata Journal*, 10(4), 628-649.
//! - Correia, S. (2017). "Linear Models with Multi-Way Fixed Effects: An Efficient
//!   and Feasible Estimator". Working Paper. <http://scorreia.com/research/hdfe.pdf>
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::{Dataset, run_hdfe, HdfeConfig, CovarianceType};
//!
//! let result = run_hdfe(
//!     &dataset,
//!     "outcome",                    // dependent variable
//!     &["treatment", "control"],    // regressors
//!     &["firm_id", "year"],         // fixed effects to absorb
//!     None,                         // use default config
//!     CovarianceType::HC1,          // robust standard errors
//! )?;
//!
//! println!("Coefficient: {}", result.coefficients[0]);
//! println!("Converged in {} iterations", result.iterations);
//! ```

use ndarray::{Array1, Array2, ArrayView2, Axis, CowArray, Ix2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, Factor, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::{LM_QR_TOL, Qr, SingularPolicy, sym_inverse};
use crate::regression::CovarianceType;
use crate::regression::ols::{NormalInverse, normal_equations_inverse};
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};
use crate::traits::estimator::{SignificanceLevel, f_test_p_value, t_test_p_value};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for HDFE estimation.
///
/// Controls the convergence behavior of the Method of Alternating Projections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdfeConfig {
    /// Relative tolerance of the alternating projections: a sweep stops when
    /// ||change|| <= tolerance * max(||demeaned column||, 1e-6 ||column||).
    /// Must be finite and greater than 0.
    /// Default: 1e-8
    pub tolerance: f64,

    /// Maximum number of iterations for the demeaning algorithm.
    /// Must be at least 1.
    /// Default: 10000
    pub max_iterations: usize,

    /// Whether to use Gearhart-Koshy acceleration for faster convergence.
    /// Default: true
    pub accelerate: bool,
}

impl Default for HdfeConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iterations: 10000,
            accelerate: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Factor Information
// ═══════════════════════════════════════════════════════════════════════════════

/// Information about a single fixed effect factor (dimension).
#[derive(Debug, Clone)]
pub struct FactorInfo {
    /// Column name in the dataset
    pub name: String,
    /// Number of unique levels (groups)
    pub n_levels: usize,
    /// Mapping from observation index to level ID (0 to n_levels-1)
    pub ids: Vec<usize>,
    /// Pre-computed group counts (number of observations per level)
    pub counts: Vec<usize>,
}

/// `FactorInfo` of an absorbed fixed effect from its extracted factor (codes in R
/// `factor()` order of the ids); fewer than 2 levels is `InsufficientData`.
pub(crate) fn factor_info(name: &str, factor: &Factor) -> EconResult<FactorInfo> {
    let n_levels = factor.n_levels();
    if n_levels < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_levels,
            context: format!(
                "Fixed effect column '{}' has only {} unique level(s)",
                name, n_levels
            ),
        });
    }
    Ok(FactorInfo {
        name: name.to_string(),
        n_levels,
        ids: factor.codes.iter().map(|&c| c as usize).collect(),
        counts: factor.counts(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Method of Alternating Projections
// ═══════════════════════════════════════════════════════════════════════════════

/// Demean a slice in-place by a single factor using pre-computed counts.
///
/// Uses a scratch buffer for group sums to avoid allocation.
fn demean_by_factor_inplace(data: &mut [f64], factor: &FactorInfo, sums_buf: &mut [f64]) {
    // Zero the scratch buffer
    for s in sums_buf.iter_mut() {
        *s = 0.0;
    }

    // Accumulate sums
    for (i, &val) in data.iter().enumerate() {
        sums_buf[factor.ids[i]] += val;
    }

    // Convert sums to means using pre-computed counts
    for (s, &c) in sums_buf.iter_mut().zip(factor.counts.iter()) {
        if c > 0 {
            *s /= c as f64;
        }
    }

    // Subtract group means in-place
    for (i, val) in data.iter_mut().enumerate() {
        *val -= sums_buf[factor.ids[i]];
    }
}

/// Perform one round of alternating projections in-place.
fn map_step_inplace(data: &mut [f64], factors: &[FactorInfo], sums_bufs: &mut [Vec<f64>]) {
    for (factor, sums_buf) in factors.iter().zip(sums_bufs.iter_mut()) {
        demean_by_factor_inplace(data, factor, sums_buf);
    }
}

/// Demean a vector using the Method of Alternating Projections.
///
/// Iteratively projects out group means for each factor until convergence.
///
/// # Arguments
/// * `data` - The input vector to demean
/// * `factors` - Slice of factor information for each fixed effect dimension
/// * `tolerance` - Relative convergence threshold: a sweep stops when
///   `||change|| <= tolerance * max(||current||, MAP_SCALE_FLOOR * ||data||)`
/// * `max_iter` - Maximum number of iterations
/// * `accelerate` - Whether to use Gearhart-Koshy acceleration
///
/// # Returns
/// A tuple of (demeaned_data, iterations, final_change, converged), where
/// `final_change` is the last relative change `||change|| / scale`, or
/// `EconError::Cancelled` when the current budget runs out between sweeps.
fn demean_map(
    data: &Array1<f64>,
    factors: &[FactorInfo],
    tolerance: f64,
    max_iter: usize,
    accelerate: bool,
) -> EconResult<(Array1<f64>, usize, f64, bool)> {
    if factors.is_empty() {
        return Ok((data.clone(), 0, 0.0, true));
    }

    let n = data.len();

    // Pre-allocate scratch buffers for group sums (one per factor)
    let mut sums_bufs: Vec<Vec<f64>> = factors.iter().map(|f| vec![0.0; f.n_levels]).collect();

    // Single factor: one pass is sufficient
    if factors.len() == 1 {
        let mut result = data.to_vec();
        demean_by_factor_inplace(&mut result, &factors[0], &mut sums_bufs[0]);
        return Ok((Array1::from_vec(result), 1, 0.0, true));
    }

    // Working buffers — avoid repeated allocations
    let mut current = data.to_vec();
    let mut prev = vec![0.0; n];
    let mut prev_prev = vec![0.0; n];
    let mut converged = false;
    let mut final_change = f64::MAX;
    // The scale of the stopping rule never falls below 1e-6 of the column's own norm,
    // so a column the effects absorb converges once its residue is rounding noise.
    let floor = MAP_SCALE_FLOOR * data.iter().map(|v| v * v).sum::<f64>().sqrt();

    for iter in 1..=max_iter {
        resource::checkpoint()?;

        // Rotate buffers: prev_prev <- prev, prev <- current (memcpy, no allocation)
        if iter > 1 {
            prev_prev.copy_from_slice(&prev);
        }
        prev.copy_from_slice(&current);

        // One round of alternating projections (in-place)
        map_step_inplace(&mut current, factors, &mut sums_bufs);

        // Change and current norm (L2) in one pass
        let mut delta_sq: f64 = 0.0;
        let mut norm_sq: f64 = 0.0;
        for i in 0..n {
            let d = current[i] - prev[i];
            delta_sq += d * d;
            norm_sq += current[i] * current[i];
        }
        let delta = delta_sq.sqrt();
        let current_norm = norm_sq.sqrt();
        let scale = current_norm.max(floor).max(f64::MIN_POSITIVE);
        final_change = delta / scale;

        // Relative stopping rule: invariant to rescaling the column
        if delta <= tolerance * scale {
            converged = true;
            return Ok((Array1::from_vec(current), iter, final_change, converged));
        }

        // Gearhart-Koshy acceleration (after at least 2 iterations)
        if accelerate && iter > 2 {
            // Compute dot products inline without temporary arrays
            let mut numerator: f64 = 0.0;
            let mut denominator: f64 = 0.0;
            for i in 0..n {
                let d = current[i] - prev[i];
                let dp = prev[i] - prev_prev[i];
                numerator += d * dp;
                denominator += dp * dp;
            }

            // Scale-free guard: skip only a step that is rounding noise
            if denominator > (f64::EPSILON * current_norm).powi(2) {
                let alpha = (numerator / denominator).clamp(0.0, 1.0);
                if alpha > 0.0 && alpha < 1.0 {
                    // Apply acceleration in-place
                    for i in 0..n {
                        let d = current[i] - prev[i];
                        current[i] = prev[i] + d * (1.0 + alpha);
                    }
                }
            }
        }
    }

    Ok((Array1::from_vec(current), max_iter, final_change, converged))
}

/// Rows from which [`demean_matrix_map`] demeans the columns of a design in parallel;
/// below it they run on the calling thread.
const PARALLEL_DEMEAN_MIN_ROWS: usize = 2_000;

/// Demean a matrix column-wise using the Method of Alternating Projections.
///
/// Columns are independent under the alternating-projections algorithm, so
/// from [`PARALLEL_DEMEAN_MIN_ROWS`] rows and two or more columns we process
/// them in parallel via `rayon` and reduce their per-column
/// convergence statistics at the end. The caller's budget is installed on each
/// worker, so every sweep honours it.
pub(crate) fn demean_matrix_map(
    x: &Array2<f64>,
    factors: &[FactorInfo],
    tolerance: f64,
    max_iter: usize,
    accelerate: bool,
) -> EconResult<(Array2<f64>, usize, f64, bool)> {
    use rayon::prelude::*;

    let (n, k) = x.dim();

    let demean_column = |j: usize| {
        demean_map(
            &x.column(j).to_owned(),
            factors,
            tolerance,
            max_iter,
            accelerate,
        )
    };
    // Small problems run on the calling thread: handing columns to the pool costs a
    // wake-up per call, more than the sweeps over short columns. Columns are
    // independent, so the result is the same either way.
    let per_column: Vec<(Array1<f64>, usize, f64, bool)> = if k <= 1 || n < PARALLEL_DEMEAN_MIN_ROWS
    {
        (0..k).map(demean_column).collect::<EconResult<Vec<_>>>()?
    } else {
        // Rayon workers do not see the caller's thread-local budget: capture it
        // here and install it inside each closure. Each call to `demean_map`
        // allocates its own scratch buffers, so workers share no mutable state.
        let budget = resource::current_budget();
        (0..k)
            .into_par_iter()
            .map(|j| match &budget {
                Some(b) => resource::with_budget(b.clone(), || demean_column(j)),
                None => demean_column(j),
            })
            .collect::<EconResult<Vec<_>>>()?
    };

    let mut x_demeaned = Array2::<f64>::zeros((n, k));
    let mut max_iterations: usize = 0;
    let mut max_change: f64 = 0.0;
    let mut all_converged = true;
    for (j, (col_demeaned, iters, change, converged)) in per_column.into_iter().enumerate() {
        x_demeaned.column_mut(j).assign(&col_demeaned);
        max_iterations = max_iterations.max(iters);
        max_change = max_change.max(change);
        all_converged = all_converged && converged;
    }

    Ok((x_demeaned, max_iterations, max_change, all_converged))
}

/// Check the MAP controls: a tolerance of zero or less can never be met, and
/// zero iterations cannot demean anything.
fn validate_config(config: &HdfeConfig) -> EconResult<()> {
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "tolerance must be a finite number greater than 0 (got {})",
                config.tolerance
            ),
        });
    }
    if config.max_iterations == 0 {
        return Err(EconError::InvalidSpecification {
            message: "max_iterations must be at least 1 (got 0)".to_string(),
        });
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regressors absorbed by the fixed effects or collinear
// ═══════════════════════════════════════════════════════════════════════════════

/// A regressor whose demeaned sum of squares is at most this fraction of its centred raw
/// sum of squares has no variation left after the fixed effects and is dropped. Absorbed
/// columns leave rounding residue (ratios near 1e-30 on the reference designs); genuine
/// regressors keep ratios of order 0.1-1.
pub(crate) const WITHIN_VARIATION_TOL: f64 = 1e-12;

/// The MAP stopping rule's scale never falls below this fraction of the column's norm
/// (`sqrt(WITHIN_VARIATION_TOL)`), so an absorbed column converges once its residue is
/// below `tolerance * 1e-6 * ||column||`.
pub(crate) const MAP_SCALE_FLOOR: f64 = 1e-6;

/// How a [`RegressorScreen`] solves: OLS's decision (`regression::ols`), the normal
/// equations on a full-rank, well-conditioned design and `lm.fit`'s pivoted QR otherwise.
#[derive(Debug, Clone)]
enum ScreenSolve {
    /// `(X'X)^{-1}` of the screened columns; none is aliased (the reciprocal-condition
    /// threshold keeps every column far above the QR's `LM_QR_TOL`).
    Normal(NormalInverse),
    /// Pivoted QR (`LM_QR_TOL`) of the screened columns in their original order; its
    /// `aliased()` columns are the collinear ones. `kept_pos` are the positions of the kept
    /// columns among the screened ones.
    Qr { qr: Qr, kept_pos: Vec<usize> },
}

/// The outcome of [`screen_regressors`]: which columns are estimated, which were dropped
/// and why, the kept demeaned columns and the factorization the caller solves from.
#[derive(Debug, Clone)]
pub(crate) struct RegressorScreen {
    /// Original indices of the estimated columns, ascending.
    pub kept: Vec<usize>,
    /// Names of the dropped columns: the absorbed ones, then the collinear ones.
    pub dropped: Vec<String>,
    /// One warning per kind of drop (absorbed, collinear).
    pub warnings: Vec<String>,
    /// The kept demeaned columns (n x kept), in `kept` order.
    pub x: Array2<f64>,
    solve: ScreenSolve,
}

impl RegressorScreen {
    /// Least-squares coefficients of `y` (demeaned like the columns) on the kept
    /// columns, in `kept` order.
    pub fn coefficients(&self, y: &Array1<f64>) -> EconResult<Array1<f64>> {
        match &self.solve {
            ScreenSolve::Normal(normal) => Ok(normal.coefficients(self.x.view(), y)),
            ScreenSolve::Qr { qr, kept_pos } => {
                let all = qr.coefficients(y.view())?;
                Ok(kept_pos.iter().map(|&p| all[p]).collect())
            }
        }
    }

    /// `(X_k' X_k)^{-1}` of the kept demeaned columns, in `kept` order.
    pub fn unscaled_covariance(&self) -> EconResult<Array2<f64>> {
        match &self.solve {
            ScreenSolve::Normal(normal) => Ok(normal.inverse.clone()),
            ScreenSolve::Qr { qr, kept_pos } => {
                let all = qr.unscaled_covariance()?;
                let k = kept_pos.len();
                let mut out = resource::try_zeros2(k, k, "unscaled covariance")?;
                for (a, &pa) in kept_pos.iter().enumerate() {
                    for (b, &pb) in kept_pos.iter().enumerate() {
                        out[[a, b]] = all[[pa, pb]];
                    }
                }
                Ok(out)
            }
        }
    }

    /// Residual sum of squares of `y` (demeaned like the columns) on the kept columns.
    pub fn rss(&self, y: &Array1<f64>) -> EconResult<f64> {
        match &self.solve {
            ScreenSolve::Normal(normal) => {
                let beta = normal.coefficients(self.x.view(), y);
                let residuals = y - &self.x.dot(&beta);
                Ok(residuals.iter().map(|v| v * v).sum::<f64>())
            }
            ScreenSolve::Qr { qr, .. } => {
                let effects = qr.qty(y.view())?;
                Ok(effects.iter().skip(qr.rank()).map(|v| v * v).sum::<f64>())
            }
        }
    }

    /// Whether the fit solves by the normal equations (`false`: the pivoted QR).
    pub fn uses_normal_equations(&self) -> bool {
        matches!(self.solve, ScreenSolve::Normal(_))
    }
}

/// `'id'` or `'firm' + 'year'`: the fixed effects as the screen's warnings name them.
pub(crate) fn fe_label(fe_cols: &[&str]) -> String {
    fe_cols
        .iter()
        .map(|c| format!("'{c}'"))
        .collect::<Vec<_>>()
        .join(" + ")
}

fn quoted(names: &[String]) -> String {
    names
        .iter()
        .map(|n| format!("'{n}'"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Drop the regressors the fixed effects absorb, then the later of collinear ones, as
/// plm (time-invariant regressors, `aliased`) and fixest (`collin.var`) do.
///
/// 1. Column `j` is absorbed when its centred raw sum of squares is 0 or its demeaned sum
///    of squares is at most [`WITHIN_VARIATION_TOL`] times the centred one.
/// 2. The remaining demeaned columns take OLS's solve decision: the normal equations
///    when they are of full rank and well conditioned (no column can be aliased then),
///    otherwise pivoted QR at `LM_QR_TOL` (`lm.fit`), whose aliased columns, the later of
///    each collinear set, are dropped.
///
/// `raw` and `demeaned` are n x k in the same column order as `names`. With no column
/// left the call is `InvalidSpecification` ("no regressor left to estimate: ...").
pub(crate) fn screen_regressors<'a>(
    raw: ArrayView2<f64>,
    demeaned: impl Into<CowArray<'a, f64, Ix2>>,
    names: &[String],
    fe_label: &str,
) -> EconResult<RegressorScreen> {
    let demeaned: CowArray<'a, f64, Ix2> = demeaned.into();
    let k = raw.ncols();
    let mut absorbed = Vec::new();
    let mut screened = Vec::new();
    for j in 0..k {
        let col = raw.column(j);
        let n = col.len().max(1) as f64;
        let mean = col.sum() / n;
        let ss_c: f64 = col.iter().map(|v| (v - mean) * (v - mean)).sum();
        let ss_w: f64 = demeaned.column(j).iter().map(|v| v * v).sum();
        if ss_c == 0.0 || ss_w <= WITHIN_VARIATION_TOL * ss_c {
            absorbed.push(j);
        } else {
            screened.push(j);
        }
    }
    let name = |j: usize| {
        names
            .get(j)
            .cloned()
            .unwrap_or_else(|| format!("x{}", j + 1))
    };
    let mut warnings = Vec::new();
    let mut dropped: Vec<String> = absorbed.iter().map(|&j| name(j)).collect();
    if !dropped.is_empty() {
        warnings.push(format!(
            "{} dropped: no variation left after removing the {fe_label} fixed effects \
             (not estimated, as in plm and fixest)",
            quoted(&dropped)
        ));
    }
    if screened.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: format!("no regressor left to estimate: {}", warnings.join("; ")),
        });
    }
    // The screened demeaned columns, without a copy when none was absorbed
    let sub: Array2<f64> = if screened.len() == k {
        demeaned.into_owned()
    } else {
        demeaned.select(Axis(1), &screened)
    };
    // OLS's solve decision: a design the normal equations accept has no aliased column
    // (see `ols::certainly_full_rank`), so the normal equations solve it.
    if let Some(normal) = normal_equations_inverse(sub.view())? {
        return Ok(RegressorScreen {
            kept: screened,
            dropped,
            warnings,
            x: sub,
            solve: ScreenSolve::Normal(normal),
        });
    }
    let qr = Qr::decompose(sub.view(), LM_QR_TOL)?;
    let aliased = qr.aliased();
    if !aliased.is_empty() {
        let collinear: Vec<String> = aliased.iter().map(|&p| name(screened[p])).collect();
        warnings.push(format!(
            "{} dropped: collinear with the other regressors after removing the {fe_label} \
             fixed effects (the later of the collinear regressors is not estimated, as in fixest)",
            quoted(&collinear)
        ));
        dropped.extend(collinear);
    }
    let kept_pos: Vec<usize> = (0..screened.len())
        .filter(|p| !aliased.contains(p))
        .collect();
    let kept: Vec<usize> = kept_pos.iter().map(|&p| screened[p]).collect();
    let x = if kept_pos.len() == screened.len() {
        sub
    } else {
        sub.select(Axis(1), &kept_pos)
    };
    Ok(RegressorScreen {
        kept,
        dropped,
        warnings,
        x,
        solve: ScreenSolve::Qr { qr, kept_pos },
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Degrees of Freedom
// ═══════════════════════════════════════════════════════════════════════════════

/// Union-find `find` with path compression over a flat parent array.
fn uf_find(parent: &mut [usize], x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
        root = parent[root];
    }
    // Path compression.
    let mut cur = x;
    while parent[cur] != root {
        let next = parent[cur];
        parent[cur] = root;
        cur = next;
    }
    root
}

/// Union-find `union` by rank.
fn uf_union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra == rb {
        return;
    }
    match rank[ra].cmp(&rank[rb]) {
        std::cmp::Ordering::Less => parent[ra] = rb,
        std::cmp::Ordering::Greater => parent[rb] = ra,
        std::cmp::Ordering::Equal => {
            parent[rb] = ra;
            rank[ra] += 1;
        }
    }
}

/// Count connected components ("mobility groups") of the bipartite graph whose
/// vertices are the levels of `f0` and `f1` and whose edges are the observed
/// (level_0, level_1) co-occurrences. Levels of `f1` are offset by `f0.n_levels`
/// so the two factors occupy disjoint vertex ranges.
fn count_connected_components(f0: &FactorInfo, f1: &FactorInfo) -> usize {
    let n0 = f0.n_levels;
    let total = n0 + f1.n_levels;
    let mut parent: Vec<usize> = (0..total).collect();
    let mut rank = vec![0u8; total];

    let n_obs = f0.ids.len().min(f1.ids.len());
    for i in 0..n_obs {
        uf_union(&mut parent, &mut rank, f0.ids[i], n0 + f1.ids[i]);
    }

    let mut roots = std::collections::HashSet::new();
    for x in 0..total {
        roots.insert(uf_find(&mut parent, x));
    }
    roots.len()
}

/// Compute degrees of freedom absorbed by fixed effects.
///
/// For a single FE the absorbed DF is simply its number of levels.
///
/// For two-way FE the redundancy equals the number of connected components
/// ("mobility groups") of the bipartite graph linking the two factors' levels
/// (Abowd–Creecy–Kramarz; Correia 2017): `df = n_0 + n_1 - C`. When the graph
/// is a single connected component this reduces to the familiar `N + T - 1`,
/// but for C > 1 components the naive `N + T - 1` over-counts the absorbed DF —
/// which would understate the residual DF and inflate the standard errors by
/// `sqrt(df / df_naive)` — so the component count must be computed explicitly.
///
/// For three or more FE dimensions there is no simple closed form; we use the
/// connected-components count for the first two dimensions and add
/// `n_levels - 1` for each additional dimension (the `reghdfe` default, exact
/// in the fully-connected case).
fn compute_absorbed_df(factors: &[FactorInfo]) -> usize {
    match factors.len() {
        0 => 0,
        1 => factors[0].n_levels,
        _ => {
            let components = count_connected_components(&factors[0], &factors[1]);
            let mut df = (factors[0].n_levels + factors[1].n_levels).saturating_sub(components);
            for f in &factors[2..] {
                df += f.n_levels.saturating_sub(1);
            }
            df
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HDFE Result
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from High-Dimensional Fixed Effects estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdfeResult {
    // ═══════════════════════════════════════════════════════════════════════════
    // Identification
    // ═══════════════════════════════════════════════════════════════════════════
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names (regressors only, no intercept since FE absorbs it)
    pub variables: Vec<String>,

    // ═══════════════════════════════════════════════════════════════════════════
    // Fixed Effects Info
    // ═══════════════════════════════════════════════════════════════════════════
    /// Names of absorbed fixed effect dimensions
    pub fe_dimensions: Vec<String>,
    /// Number of unique levels per FE dimension
    pub fe_counts: Vec<usize>,

    // ═══════════════════════════════════════════════════════════════════════════
    // Core Results
    // ═══════════════════════════════════════════════════════════════════════════
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// t-statistics
    pub t_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,

    // ═══════════════════════════════════════════════════════════════════════════
    // Fit Statistics
    // ═══════════════════════════════════════════════════════════════════════════
    /// Within R-squared (computed on demeaned data)
    pub r_squared_within: f64,
    /// Adjusted within R-squared
    pub adj_r_squared_within: f64,
    /// F-statistic for joint significance
    pub f_stat: f64,
    /// F-statistic p-value
    pub f_p_value: f64,
    /// Residual standard error
    pub residual_std_error: f64,

    // ═══════════════════════════════════════════════════════════════════════════
    // Dimensions
    // ═══════════════════════════════════════════════════════════════════════════
    /// Number of observations
    pub n_obs: usize,
    /// Residual degrees of freedom (n - k - absorbed)
    pub df_resid: usize,
    /// Total absorbed degrees of freedom
    pub df_absorbed: usize,

    // ═══════════════════════════════════════════════════════════════════════════
    // Convergence Info
    // ═══════════════════════════════════════════════════════════════════════════
    /// Number of iterations to convergence
    pub iterations: usize,
    /// Final convergence metric: the largest relative change of a demeaned column,
    /// `||change|| / max(||demeaned column||, 1e-6 ||column||)`
    pub convergence: f64,
    /// Whether convergence was achieved
    pub converged: bool,
    /// The iteration limit of the alternating projections
    #[serde(default)]
    pub max_iterations: usize,
    /// The relative tolerance of the alternating projections
    #[serde(default)]
    pub tolerance: f64,

    // ═══════════════════════════════════════════════════════════════════════════
    // Covariance Type
    // ═══════════════════════════════════════════════════════════════════════════
    /// Type of standard errors used
    pub cov_type: CovarianceType,

    // ═══════════════════════════════════════════════════════════════════════════
    // Internal caches (not serialized, kept for potential future extensions)
    // ═══════════════════════════════════════════════════════════════════════════
    #[serde(skip)]
    #[allow(dead_code)]
    pub(crate) beta: Array1<f64>,
    #[serde(skip)]
    #[allow(dead_code)]
    pub(crate) se: Array1<f64>,
    #[serde(skip)]
    #[allow(dead_code)]
    pub(crate) resid: Array1<f64>,
    #[serde(skip)]
    #[allow(dead_code)]
    pub(crate) vcov: Array2<f64>,
    #[serde(skip)]
    #[allow(dead_code)]
    pub(crate) xtx_inv: Array2<f64>,

    /// Rows dropped because of a missing value in y, a regressor or a fixed-effect id
    /// (fixest drops them too).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,

    /// Regressors not estimated: absorbed by the fixed effects or collinear with the
    /// others after removing them (plm and fixest drop them too)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dropped: Vec<String>,
    /// Warnings: dropped regressors, non-convergence of the alternating projections
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl ResultNotes for HdfeResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        Some(Convergence::new(
            self.converged,
            self.iterations,
            self.max_iterations,
            "relative change of the demeaned columns",
            self.convergence,
            self.tolerance,
        ))
    }
}

impl fmt::Display for HdfeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "High-Dimensional Fixed Effects Regression Results")?;
        writeln!(f, "==================================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "Df Residuals: {}", self.df_resid)?;
        writeln!(f, "Df Absorbed: {}", self.df_absorbed)?;
        writeln!(f)?;

        writeln!(f, "Fixed Effects:")?;
        for (name, count) in self.fe_dimensions.iter().zip(self.fe_counts.iter()) {
            writeln!(f, "  {}: {} levels", name, count)?;
        }
        writeln!(f)?;

        writeln!(f, "R-squared (within): {:.4}", self.r_squared_within)?;
        writeln!(f, "Adj. R-squared: {:.4}", self.adj_r_squared_within)?;
        writeln!(
            f,
            "F-statistic: {:.4} (p-value: {:.4})",
            self.f_stat, self.f_p_value
        )?;
        writeln!(f, "Covariance Type: {:?}", self.cov_type)?;
        writeln!(f)?;

        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef", "Std Err", "t", "P>|t|"
        )?;
        writeln!(f, "{}", "-".repeat(70))?;

        for i in 0..self.variables.len() {
            writeln!(
                f,
                "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.3}{}",
                self.variables[i],
                self.coefficients[i],
                self.std_errors[i],
                self.t_stats[i],
                self.p_values[i],
                self.significance[i].stars()
            )?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        // Notes before the missing-value note, which stays the last line
        write!(f, "{}", notes_text(self))?;
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Estimation Function
// ═══════════════════════════════════════════════════════════════════════════════

/// Run High-Dimensional Fixed Effects estimation.
///
/// Uses the Method of Alternating Projections to efficiently absorb multiple
/// high-dimensional fixed effects without creating dummy variables.
///
/// # Arguments
///
/// * `dataset` - The dataset containing the panel data
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the independent variable columns (regressors)
/// * `fe_cols` - Names of the fixed effect columns to absorb
/// * `config` - Optional configuration (uses defaults if None)
/// * `cov_type` - Type of standard errors (Standard, HC0-HC3)
///
/// # Returns
///
/// An `HdfeResult` containing coefficient estimates, standard errors,
/// and diagnostic information.
///
/// # Missing values
///
/// Rows with a missing value (null or NaN) in `y`, a regressor or a fixed-effect
/// column are dropped and reported in `missing` (fixest drops them too); an infinite
/// value is refused (`NonFiniteValues`). Fixed-effect levels follow R factor order.
///
/// # Example
///
/// ```ignore
/// let result = run_hdfe(
///     &dataset,
///     "wage",
///     &["experience", "education"],
///     &["person_id", "year"],
///     None,
///     CovarianceType::HC1,
/// )?;
/// ```
///
/// # References
///
/// - Gaure, S. (2013). "lfe: Linear Group Fixed Effects". The R Journal.
/// - Guimarães, P. & Portugal, P. (2010). Stata Journal.
pub fn run_hdfe(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    fe_cols: &[&str],
    config: Option<HdfeConfig>,
    cov_type: CovarianceType,
) -> EconResult<HdfeResult> {
    let config = config.unwrap_or_default();
    validate_config(&config)?;

    // Validate inputs
    if fe_cols.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "At least one fixed effect column must be specified".to_string(),
        });
    }

    if x_cols.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "At least one regressor (X) column must be specified".to_string(),
        });
    }

    // y, X and every fixed-effect column on the rows complete in all of them
    // (fixest drops incomplete rows too)
    let cols = ColumnRequest::new(dataset.df())
        .numeric(y_col)
        .numerics(x_cols)
        .factors(fe_cols)
        .na(NaPolicy::Omit)
        .extract()?;
    let y = cols.numeric_array(y_col);

    // Design matrix WITHOUT intercept (FE absorbs the constant)
    let design = cols.matrix(x_cols, false)?;
    let x = design.data;
    let all_names = design.column_names;
    let n = y.len();

    // Factor information for each FE dimension (levels in R factor order)
    let factors: Vec<FactorInfo> = fe_cols
        .iter()
        .map(|col| factor_info(col, cols.factor(col)))
        .collect::<EconResult<Vec<_>>>()?;

    // HC2/HC3 need the fixed-effect leverage; a design it is not available for is
    // refused before any demeaning
    let h_fe = if matches!(cov_type, CovarianceType::HC2 | CovarianceType::HC3) {
        Some(fe_leverage(&factors)?)
    } else {
        None
    };

    // Compute absorbed degrees of freedom
    let df_absorbed = compute_absorbed_df(&factors);

    // Demean y using MAP
    let (y_demeaned, y_iters, y_change, y_converged) = demean_map(
        &y,
        &factors,
        config.tolerance,
        config.max_iterations,
        config.accelerate,
    )?;

    // Demean X using MAP
    let (x_demeaned_all, x_iters, x_change, x_converged) = demean_matrix_map(
        &x,
        &factors,
        config.tolerance,
        config.max_iterations,
        config.accelerate,
    )?;

    let iterations = y_iters.max(x_iters);
    let convergence = y_change.max(x_change);
    let converged = y_converged && x_converged;

    // Regressors absorbed by the effects or collinear after them are not estimated
    let screen = screen_regressors(x.view(), x_demeaned_all, &all_names, &fe_label(fe_cols))?;
    let mut warnings = screen.warnings.clone();
    if !converged {
        // fixest warns and returns the last iterate ("Absence of convergence")
        let record = Convergence::new(
            false,
            iterations,
            config.max_iterations,
            "relative change of the demeaned columns",
            convergence,
            config.tolerance,
        );
        warnings.extend(record.note("HDFE alternating projections"));
    }
    let x_demeaned = &screen.x;
    let var_names: Vec<String> = screen.kept.iter().map(|&j| all_names[j].clone()).collect();
    let k = screen.kept.len();

    // Residual degrees of freedom
    let df_resid = n
        .checked_sub(k + df_absorbed)
        .filter(|&d| d > 0)
        .ok_or_else(|| EconError::InsufficientData {
            required: k + df_absorbed + 1,
            provided: n,
            context: "Not enough observations for HDFE estimation".to_string(),
        })?;

    // OLS on demeaned data from the screen's factorization
    let beta = screen.coefficients(&y_demeaned)?;
    let xtx_inv = screen.unscaled_covariance()?;

    // Residuals from demeaned model
    let y_hat: Array1<f64> = x_demeaned.dot(&beta);
    let residuals = &y_demeaned - &y_hat;

    // Sum of squared residuals
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma2 = ssr / df_resid as f64;

    // Compute variance-covariance matrix based on cov_type
    let hc = |hc_type| {
        compute_hc_vcov(
            &x_demeaned.view(),
            &residuals,
            &xtx_inv,
            hc_type,
            df_resid,
            h_fe.as_ref(),
        )
    };
    let vcov = match cov_type {
        CovarianceType::Standard => &xtx_inv * sigma2,
        CovarianceType::HC0 => hc(0)?.0,
        CovarianceType::HC1 => hc(1)?.0,
        CovarianceType::HC2 | CovarianceType::HC3 => {
            let (v, n_unit) = hc(if cov_type == CovarianceType::HC2 {
                2
            } else {
                3
            })?;
            match n_unit {
                0 => {}
                1 => warnings.push(
                    "1 observation has leverage 1 (it is fitted exactly, e.g. a singleton \
                     fixed-effect level) and contributes nothing to the HC2/HC3 meat"
                        .to_string(),
                ),
                m => warnings.push(format!(
                    "{m} observations have leverage 1 (they are fitted exactly, e.g. singleton \
                     fixed-effect levels) and contribute nothing to the HC2/HC3 meat"
                )),
            }
            v
        }
    };

    // Standard errors
    let se: Array1<f64> = vcov.diag().mapv(|v| v.max(0.0).sqrt());

    // R-squared (within)
    let y_mean_demeaned = y_demeaned.mean().unwrap_or(0.0);
    let sst: f64 = y_demeaned
        .iter()
        .map(|y| (y - y_mean_demeaned).powi(2))
        .sum();
    let r_squared_within = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    // Adjusted R-squared
    let adj_r_squared_within = if df_resid > 0 && n > k + df_absorbed {
        1.0 - (1.0 - r_squared_within) * ((n - 1) as f64) / (df_resid as f64)
    } else {
        r_squared_within
    };

    // F-statistic
    let f_stat = if k > 0 && ssr > 0.0 && df_resid > 0 {
        (sst - ssr) / (k as f64) / (ssr / df_resid as f64)
    } else {
        0.0
    };
    let f_p_value = f_test_p_value(f_stat, k as f64, df_resid as f64);

    // t-statistics and p-values
    let t_stats: Vec<f64> = beta
        .iter()
        .zip(se.iter())
        .map(|(&b, &s)| if s > 0.0 { b / s } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| t_test_p_value(t, df_resid as f64))
        .collect();
    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    // Residual standard error
    let residual_std_error = sigma2.sqrt();

    Ok(HdfeResult {
        dep_var: y_col.to_string(),
        variables: var_names,
        fe_dimensions: factors.iter().map(|f| f.name.clone()).collect(),
        fe_counts: factors.iter().map(|f| f.n_levels).collect(),
        coefficients: beta.to_vec(),
        std_errors: se.to_vec(),
        t_stats,
        p_values,
        significance,
        r_squared_within,
        adj_r_squared_within,
        f_stat,
        f_p_value,
        residual_std_error,
        n_obs: n,
        df_resid,
        df_absorbed,
        iterations,
        convergence,
        converged,
        max_iterations: config.max_iterations,
        tolerance: config.tolerance,
        cov_type,
        beta,
        se,
        resid: residuals,
        vcov,
        xtx_inv,
        missing: cols.missing().clone(),
        dropped: screen.dropped,
        warnings,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Heteroskedasticity-Consistent Covariance
// ═══════════════════════════════════════════════════════════════════════════════

/// Largest number of levels of the smaller of two fixed effects for which the HC2/HC3
/// leverage is computed (the Schur complement is a dense square matrix of this order).
const HDFE_LEVERAGE_MAX_LEVELS: usize = 1_000;

/// Leverage at or above `1 - LEVERAGE_ONE_TOL` counts as 1.
const LEVERAGE_ONE_TOL: f64 = 1e-10;

/// The diagonal of the fixed-effect projection (the leverage of the dummies), for
/// HC2/HC3.
///
/// - One fixed effect: `1 / T_g`, the size of the observation's level.
/// - Two: with `A` the dummies of the factor with more levels and `B` the other,
///   `h_i = 1 / c_a + (e_b - u_a)' S^+ (e_b - u_a)`, where `S = B' M_A B =
///   diag(c_b) - sum_a C_a C_a' / c_a` is the Schur complement on `B` (`C_a` the cell
///   counts of level `a`, `u_a = C_a / c_a`). `e_b - u_a` lies in the range of `S`, so
///   any generalized inverse gives the same value, also for a design with several
///   connected components. Refused when the smaller factor has more than
///   [`HDFE_LEVERAGE_MAX_LEVELS`] levels.
/// - Three or more: refused (as fixest and reghdfe, which offer no HC2/HC3 there).
pub(crate) fn fe_leverage(factors: &[FactorInfo]) -> EconResult<Array1<f64>> {
    match factors {
        [] => Err(EconError::InvalidSpecification {
            message: "HC2/HC3 need at least one fixed effect".to_string(),
        }),
        [f] => {
            let n = f.ids.len();
            let mut h = resource::try_zeros1(n, "HDFE fixed-effect leverage")?;
            for (hi, &g) in h.iter_mut().zip(&f.ids) {
                *hi = 1.0 / f.counts[g] as f64;
            }
            Ok(h)
        }
        [f1, f2] => {
            let (fa, fb) = if f1.n_levels >= f2.n_levels {
                (f1, f2)
            } else {
                (f2, f1)
            };
            let n1 = fb.n_levels;
            if n1 > HDFE_LEVERAGE_MAX_LEVELS {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "HC2/HC3 need the fixed-effect leverage, computed for two fixed \
                         effects when the smaller one has at most {HDFE_LEVERAGE_MAX_LEVELS} \
                         levels ('{}' has {n1}); use HC1",
                        fb.name
                    ),
                });
            }
            two_way_leverage(fa, fb)
        }
        _ => Err(EconError::InvalidSpecification {
            message: format!(
                "HC2/HC3 with three or more fixed effects are not available, as in fixest and \
                 reghdfe ({} given); use HC1",
                factors.len()
            ),
        }),
    }
}

/// [`fe_leverage`] for two factors, `fb` the one with fewer levels.
fn two_way_leverage(fa: &FactorInfo, fb: &FactorInfo) -> EconResult<Array1<f64>> {
    let what = "HDFE two-way fixed-effect leverage";
    let n = fa.ids.len();
    let (na, n1) = (fa.n_levels, fb.n_levels);

    // The rows of each a-level (counting sort)
    let mut start = resource::try_vec(na + 1, 0usize, what)?;
    for &a in &fa.ids {
        start[a + 1] += 1;
    }
    for a in 0..na {
        start[a + 1] += start[a];
    }
    let mut order = resource::try_vec(n, 0usize, what)?;
    let mut next = start.clone();
    for (i, &a) in fa.ids.iter().enumerate() {
        order[next[a]] = i;
        next[a] += 1;
    }
    drop(next);

    // Cells (b, count) of each a-level, gathered through a dense scratch row
    let mut cell_start = resource::try_vec(na + 1, 0usize, what)?;
    let mut cells: Vec<(usize, f64)> = Vec::new();
    let mut scratch = resource::try_vec(n1, 0usize, what)?;
    let mut touched: Vec<usize> = Vec::new();
    // S = diag(c_b) - sum_a C_a C_a' / c_a, row-major n1 x n1
    let mut s = resource::try_vec(resource::checked_len(&[n1, n1], what)?, 0.0, what)?;
    for (b, &c) in fb.counts.iter().enumerate() {
        s[b * n1 + b] = c as f64;
    }
    for a in 0..na {
        if a % 256 == 0 {
            resource::checkpoint()?;
        }
        for &i in &order[start[a]..start[a + 1]] {
            let b = fb.ids[i];
            if scratch[b] == 0 {
                touched.push(b);
            }
            scratch[b] += 1;
        }
        let inv_ca = 1.0 / fa.counts[a] as f64;
        let first = cells.len();
        for &b in &touched {
            cells.push((b, scratch[b] as f64));
            scratch[b] = 0;
        }
        touched.clear();
        let block = &cells[first..];
        for &(b, cb) in block {
            let w = cb * inv_ca;
            let row = &mut s[b * n1..(b + 1) * n1];
            for &(d, cd) in block {
                row[d] -= w * cd;
            }
        }
        cell_start[a + 1] = cells.len();
    }
    drop(scratch);

    // S has rank n1 less the connected components (its null space is the indicator of
    // each component's b-levels). Dropping one b-level per component leaves a
    // positive-definite block whose inverse, padded with zeros, is a generalized
    // inverse of S; a Cholesky of the block is much cheaper than an eigendecomposition
    // of S. `pos[b]` is the b-level's index in the block, or None when dropped.
    let pos = schur_block_positions(fa.n_levels, fb.n_levels, &cells, &cell_start)?;
    let r = pos.iter().flatten().count();
    let mut block = resource::try_zeros2(r, r, what)?;
    for (b, pb) in pos.iter().enumerate() {
        let Some(i) = *pb else { continue };
        for (d, pd) in pos.iter().enumerate() {
            if let Some(j) = *pd {
                block[[i, j]] = s[b * n1 + d];
            }
        }
    }
    drop(s);
    let sp = if r == 0 {
        block
    } else {
        sym_inverse(
            block.view(),
            SingularPolicy::Generalized,
            "the two-way fixed-effect leverage matrix",
        )?
        .inverse
    };

    // h_i = 1 / c_a + q_a - 2 v_a[b] + Sp[b, b], v_a = Sp u_a, q_a = u_a' v_a (a dropped
    // b-level has a zero row and column in Sp); v_a gathers the rows of Sp (symmetric)
    let sp = sp.as_standard_layout();
    let sp_flat = sp.as_slice().expect("standard layout");
    let mut h = resource::try_zeros1(n, what)?;
    let mut v = resource::try_vec(r, 0.0, what)?;
    for a in 0..na {
        if a % 256 == 0 {
            resource::checkpoint()?;
        }
        let inv_ca = 1.0 / fa.counts[a] as f64;
        let block = &cells[cell_start[a]..cell_start[a + 1]];
        v.fill(0.0);
        for &(b, cb) in block {
            if let Some(j) = pos[b] {
                let w = cb * inv_ca;
                for (vi, &x) in v.iter_mut().zip(&sp_flat[j * r..(j + 1) * r]) {
                    *vi += w * x;
                }
            }
        }
        let q: f64 = block
            .iter()
            .filter_map(|&(b, cb)| pos[b].map(|j| cb * v[j]))
            .sum::<f64>()
            * inv_ca;
        for &i in &order[start[a]..start[a + 1]] {
            h[i] = inv_ca
                + q
                + match pos[fb.ids[i]] {
                    Some(j) => sp_flat[j * r + j] - 2.0 * v[j],
                    None => 0.0,
                };
        }
    }
    Ok(h)
}

/// For [`two_way_leverage`]: the index of each b-level in the Schur block with one
/// b-level per connected component of the a-b graph left out (the highest-numbered
/// b-level of each component), or None for the left-out levels. `cells` holds each
/// a-level's `(b, count)` cells from `cell_start[a]`.
fn schur_block_positions(
    na: usize,
    n1: usize,
    cells: &[(usize, f64)],
    cell_start: &[usize],
) -> EconResult<Vec<Option<usize>>> {
    let what = "HDFE two-way fixed-effect leverage";
    // Union-find over the b-levels: the b-levels of one a-level are connected
    let mut parent: Vec<usize> = resource::try_with_capacity(n1, what)?;
    parent.extend(0..n1);
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for a in 0..na {
        let block = &cells[cell_start[a]..cell_start[a + 1]];
        if let Some(&(b0, _)) = block.first() {
            let r0 = find(&mut parent, b0);
            for &(b, _) in &block[1..] {
                let rb = find(&mut parent, b);
                if rb != r0 {
                    parent[rb] = r0;
                }
            }
        }
    }
    // Every b-level occurs in some cell (its count is positive), so each component's
    // b-levels are found; leave out the last one of each component
    let mut last_of_root: Vec<usize> = resource::try_vec(n1, usize::MAX, what)?;
    for b in 0..n1 {
        let root = find(&mut parent, b);
        last_of_root[root] = b;
    }
    let mut drop_b: Vec<bool> = resource::try_vec(n1, false, what)?;
    for &b in last_of_root.iter().filter(|&&b| b != usize::MAX) {
        drop_b[b] = true;
    }
    let mut pos = resource::try_vec(n1, None, what)?;
    let mut next = 0;
    for b in 0..n1 {
        if !drop_b[b] {
            pos[b] = Some(next);
            next += 1;
        }
    }
    Ok(pos)
}

/// Compute heteroskedasticity-consistent variance-covariance matrix.
///
/// Implements HC0-HC3 robust standard errors. For HC2/HC3 the leverage of an
/// observation is that of the demeaned regressors plus `fe_leverage`, the diagonal of
/// the fixed-effect projection; an observation with leverage 1 gets weight 0 (it has a
/// zero residual), and their number is returned with the matrix.
fn compute_hc_vcov(
    x: &ndarray::ArrayView2<f64>,
    residuals: &Array1<f64>,
    xtx_inv: &Array2<f64>,
    hc_type: usize,
    df_resid: usize,
    fe_leverage: Option<&Array1<f64>>,
) -> EconResult<(Array2<f64>, usize)> {
    let (n, k) = x.dim();

    // Leverage (hat matrix diagonal) for HC2/HC3: rowwise (x A) . x plus the
    // fixed-effect part
    let leverage: Option<Array1<f64>> = if hc_type >= 2 {
        let xa = x.dot(xtx_inv);
        let mut h = resource::try_zeros1(n, "HDFE leverage")?;
        for (i, hi) in h.iter_mut().enumerate() {
            *hi = xa.row(i).dot(&x.row(i)) + fe_leverage.map_or(0.0, |f| f[i]);
        }
        Some(h)
    } else {
        None
    };

    // Compute meat of sandwich: X' * diag(u^2 * weight) * X
    let mut meat = Array2::zeros((k, k));
    let mut n_unit = 0usize;

    for i in 0..n {
        let e_i = residuals[i];
        let xi = x.row(i);

        // Compute weight based on HC type
        let weight = match (hc_type, &leverage) {
            (0, _) => 1.0,
            (1, _) => n as f64 / df_resid as f64,
            (_, Some(h)) if h[i] >= 1.0 - LEVERAGE_ONE_TOL => {
                n_unit += 1;
                0.0
            }
            (2, Some(h)) => 1.0 / (1.0 - h[i]),
            (3, Some(h)) => 1.0 / (1.0 - h[i]).powi(2),
            _ => 1.0,
        };

        let e_weighted = e_i * e_i * weight;
        if e_weighted == 0.0 {
            continue;
        }

        for j in 0..k {
            for l in 0..k {
                meat[[j, l]] += xi[j] * xi[l] * e_weighted;
            }
        }
    }

    // Sandwich: (X'X)^{-1} * meat * (X'X)^{-1}
    let temp = xtx_inv.dot(&meat);
    Ok((temp.dot(xtx_inv), n_unit))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{Budget, CancelToken, with_budget};
    use polars::df;

    fn cancelled() -> Budget {
        let token = CancelToken::new();
        token.cancel();
        Budget::new(None, token)
    }

    fn expect_invalid<T>(res: EconResult<T>, needle: &str) {
        match res {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(
                    message.contains(needle),
                    "message {message:?} should name `{needle}`"
                )
            }
            Err(e) => panic!("expected InvalidSpecification naming `{needle}`, got: {e}"),
            Ok(_) => panic!("expected InvalidSpecification naming `{needle}`, got Ok"),
        }
    }

    fn expect_cancelled<T>(res: EconResult<T>) {
        match res {
            Err(EconError::Cancelled(_)) => {}
            Err(e) => panic!("expected Cancelled, got: {e}"),
            Ok(_) => panic!("expected Cancelled, got Ok"),
        }
    }

    fn create_balanced_panel() -> Dataset {
        // 3 firms, 4 years each = 12 observations
        // True model: y = 2*x + firm_effect + time_effect + noise
        // Firm effects: A=0, B=5, C=10
        // Time effects: 2020=0, 2021=1, 2022=2, 2023=3
        //
        // IMPORTANT: x must have variation that is NOT explained by firm or year FE.
        // We need the demeaned x (after removing firm and year means) to still have variation.
        // This requires x to have "interaction-like" variation across firm-year cells.
        let df = df! {
            "firm" => ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            "year" => [2020i64, 2021, 2022, 2023, 2020, 2021, 2022, 2023, 2020, 2021, 2022, 2023],
            // x with idiosyncratic variation that won't be absorbed by FE
            // Each firm has different patterns over time
            "x" => [1.0, 3.0, 2.0, 5.0,   // Firm A: irregular pattern
                    2.0, 1.0, 4.0, 3.0,   // Firm B: different pattern
                    3.0, 4.0, 1.0, 2.0],  // Firm C: yet another pattern
            "y" => [
                // Firm A (effect=0): 2*x + 0 + time + noise
                2.0*1.0 + 0.0 + 0.0 + 0.1,   // year 2020
                2.0*3.0 + 0.0 + 1.0 + 0.2,   // year 2021
                2.0*2.0 + 0.0 + 2.0 + 0.1,   // year 2022
                2.0*5.0 + 0.0 + 3.0 + 0.0,   // year 2023
                // Firm B (effect=5): 2*x + 5 + time + noise
                2.0*2.0 + 5.0 + 0.0 + 0.0,
                2.0*1.0 + 5.0 + 1.0 + 0.1,
                2.0*4.0 + 5.0 + 2.0 - 0.1,
                2.0*3.0 + 5.0 + 3.0 + 0.2,
                // Firm C (effect=10): 2*x + 10 + time + noise
                2.0*3.0 + 10.0 + 0.0 + 0.2,
                2.0*4.0 + 10.0 + 1.0 - 0.1,
                2.0*1.0 + 10.0 + 2.0 + 0.0,
                2.0*2.0 + 10.0 + 3.0 - 0.2
            ],
        }
        .unwrap();
        Dataset::new(df)
    }

    fn create_unbalanced_panel() -> Dataset {
        // Unbalanced: A has 3 obs, B has 2 obs, C has 4 obs. `x` is year - 2019 in
        // every row (absorbed by the year effects); `x_mixed` varies within years.
        let df = df! {
            "firm" => ["A", "A", "A", "B", "B", "C", "C", "C", "C"],
            "year" => [2020i64, 2021, 2022, 2021, 2022, 2020, 2021, 2022, 2023],
            "x" => [1.0, 2.0, 3.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0],
            "x_mixed" => [1.0, 2.5, 3.0, 2.0, 1.5, 1.0, 2.2, 2.9, 4.6],
            "y" => [2.1, 4.3, 6.0, 9.2, 11.0, 12.1, 14.0, 15.9, 18.2],
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_extract_factor_info() {
        let dataset = create_balanced_panel();
        let cols = crate::data::columns::ColumnRequest::new(dataset.df())
            .factor("firm")
            .extract()
            .unwrap();
        let factor = factor_info("firm", cols.factor("firm")).unwrap();

        assert_eq!(factor.name, "firm");
        assert_eq!(factor.n_levels, 3);
        assert_eq!(factor.ids.len(), 12);
        assert_eq!(factor.counts, [4, 4, 4]);

        // Check that each firm maps to a unique ID
        let unique_ids: std::collections::HashSet<_> = factor.ids.iter().collect();
        assert_eq!(unique_ids.len(), 3);
    }

    #[test]
    fn test_demean_by_factor() {
        let mut data = vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
        let factor = FactorInfo {
            name: "test".to_string(),
            n_levels: 2,
            ids: vec![0, 0, 0, 1, 1, 1],
            counts: vec![3, 3],
        };
        let mut sums = vec![0.0; 2];

        demean_by_factor_inplace(&mut data, &factor, &mut sums);

        // Group 0 mean: (1+2+3)/3 = 2
        // Group 1 mean: (10+11+12)/3 = 11
        assert!((data[0] - (-1.0)).abs() < 1e-10); // 1 - 2 = -1
        assert!((data[1] - 0.0).abs() < 1e-10); // 2 - 2 = 0
        assert!((data[2] - 1.0).abs() < 1e-10); // 3 - 2 = 1
        assert!((data[3] - (-1.0)).abs() < 1e-10); // 10 - 11 = -1
        assert!((data[4] - 0.0).abs() < 1e-10); // 11 - 11 = 0
        assert!((data[5] - 1.0).abs() < 1e-10); // 12 - 11 = 1
    }

    #[test]
    fn test_demean_map_single_factor() {
        let data = Array1::from(vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
        let factor = FactorInfo {
            name: "test".to_string(),
            n_levels: 2,
            ids: vec![0, 0, 0, 1, 1, 1],
            counts: vec![3, 3],
        };

        let (demeaned, iters, _change, converged) =
            demean_map(&data, &[factor], 1e-8, 1000, false).unwrap();

        assert!(converged);
        assert_eq!(iters, 1); // Single factor converges in one pass
        assert!((demeaned.sum()).abs() < 1e-10); // Demeaned data sums to ~0
    }

    #[test]
    fn test_demean_map_two_factors() {
        let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let factor1 = FactorInfo {
            name: "entity".to_string(),
            n_levels: 2,
            ids: vec![0, 0, 0, 1, 1, 1],
            counts: vec![3, 3],
        };
        let factor2 = FactorInfo {
            name: "time".to_string(),
            n_levels: 3,
            ids: vec![0, 1, 2, 0, 1, 2],
            counts: vec![2, 2, 2],
        };

        let (_demeaned, iters, change, converged) =
            demean_map(&data, &[factor1, factor2], 1e-8, 1000, false).unwrap();

        assert!(converged);
        assert!(iters < 100); // Should converge reasonably fast
        assert!(change < 1e-8);
    }

    #[test]
    fn test_compute_absorbed_df_connected() {
        // Fully connected 2-way FE: 2 entities x 3 times, every pair observed.
        let entity = FactorInfo {
            name: "entity".to_string(),
            n_levels: 2,
            ids: vec![0, 0, 0, 1, 1, 1],
            counts: vec![3, 3],
        };
        let time = FactorInfo {
            name: "time".to_string(),
            n_levels: 3,
            ids: vec![0, 1, 2, 0, 1, 2],
            counts: vec![2, 2, 2],
        };
        // Single connected component => N + T - 1 = 2 + 3 - 1 = 4.
        assert_eq!(compute_absorbed_df(&[entity, time]), 4);
    }

    #[test]
    fn test_compute_absorbed_df_disconnected() {
        // Two firm/period clusters that never co-occur => 2 components.
        // Firms {0,1} only appear in periods {0,1}; firms {2,3} only in {2,3}.
        let firm = FactorInfo {
            name: "firm".to_string(),
            n_levels: 4,
            ids: vec![0, 0, 1, 1, 2, 2, 3, 3],
            counts: vec![2, 2, 2, 2],
        };
        let period = FactorInfo {
            name: "period".to_string(),
            n_levels: 4,
            ids: vec![0, 1, 0, 1, 2, 3, 2, 3],
            counts: vec![2, 2, 2, 2],
        };
        // Correct: n0 + n1 - C = 4 + 4 - 2 = 6 (the naive N + T - 1 would give 7,
        // over-counting the absorbed DF and inflating the standard errors).
        assert_eq!(compute_absorbed_df(&[firm, period]), 6);
    }

    #[test]
    fn test_hdfe_two_way_balanced() {
        let dataset = create_balanced_panel();

        let result = run_hdfe(
            &dataset,
            "y",
            &["x"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        assert!(result.converged);
        assert!(result.iterations < 100);
        assert_eq!(result.fe_dimensions.len(), 2);
        assert_eq!(result.fe_counts.len(), 2);
        assert_eq!(result.fe_counts[0], 3); // 3 firms
        assert_eq!(result.fe_counts[1], 4); // 4 years

        // Check we have coefficients
        assert!(
            !result.coefficients.is_empty(),
            "Coefficients should not be empty"
        );

        // The true coefficient is 2.0
        assert!(
            (result.coefficients[0] - 2.0).abs() < 0.5,
            "Coefficient should be close to 2.0, got {}",
            result.coefficients[0]
        );

        // R-squared should be high
        assert!(
            result.r_squared_within > 0.9,
            "R² should be high, got {}",
            result.r_squared_within
        );
    }

    #[test]
    fn test_hdfe_single_fe() {
        // Single FE should match panel.rs results
        let dataset = create_balanced_panel();

        let result = run_hdfe(
            &dataset,
            "y",
            &["x"],
            &["firm"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        assert!(result.converged);
        assert_eq!(result.iterations, 1); // Single factor: one pass
        assert_eq!(result.fe_dimensions.len(), 1);
    }

    #[test]
    fn test_hdfe_unbalanced_panel() {
        let dataset = create_unbalanced_panel();

        let result = run_hdfe(
            &dataset,
            "y",
            &["x_mixed"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        assert!(result.converged);
        // Unbalanced panels may need more iterations
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_hdfe_robust_se() {
        let dataset = create_balanced_panel();

        let result_std = run_hdfe(
            &dataset,
            "y",
            &["x"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        let result_hc1 = run_hdfe(
            &dataset,
            "y",
            &["x"],
            &["firm", "year"],
            None,
            CovarianceType::HC1,
        )
        .unwrap();

        // Coefficients should be identical
        assert!(
            (result_std.coefficients[0] - result_hc1.coefficients[0]).abs() < 1e-10,
            "Coefficients should match"
        );

        // Standard errors will differ (robust SEs are typically larger)
        // Just check they're both positive
        assert!(result_std.std_errors[0] > 0.0);
        assert!(result_hc1.std_errors[0] > 0.0);
    }

    #[test]
    fn test_hdfe_convergence_config() {
        let dataset = create_balanced_panel();

        let config = HdfeConfig {
            tolerance: 1e-12,
            max_iterations: 50000,
            accelerate: true,
        };

        let result = run_hdfe(
            &dataset,
            "y",
            &["x"],
            &["firm", "year"],
            Some(config),
            CovarianceType::Standard,
        )
        .unwrap();

        assert!(result.converged);
        assert!(result.convergence < 1e-12);
    }

    #[test]
    fn test_hdfe_display() {
        let dataset = create_balanced_panel();
        let result = run_hdfe(
            &dataset,
            "y",
            &["x"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        let display = format!("{}", result);
        assert!(display.contains("High-Dimensional Fixed Effects"));
        assert!(display.contains("firm"));
        assert!(display.contains("year"));
        assert!(display.contains("Converged: yes"));
        assert!(!display.contains("Convergence:"));
    }

    #[test]
    fn test_hdfe_missing_column() {
        let dataset = create_balanced_panel();
        let result = run_hdfe(
            &dataset,
            "y",
            &["nonexistent"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_hdfe_no_fe_columns() {
        let dataset = create_balanced_panel();
        let result = run_hdfe(&dataset, "y", &["x"], &[], None, CovarianceType::Standard);
        assert!(result.is_err());
    }

    // =========================================================================
    // DGP-recovery tests on lfe::felm()-style examples
    // =========================================================================
    //
    // The tests below check that the MAP algorithm recovers known coefficients
    // and degrees of freedom on small simulated panels; they do not compare with
    // R output. The R-pinned HDFE tests are `test_validate_hdfe_grunfeld*`
    // (case `panel_hdfe_grunfeld`, fixest), further down.
    //
    // Reference: Gaure, S. (2013). "lfe: Linear Group Fixed Effects".
    //            The R Journal, 5(2), 104-117.
    // =========================================================================

    /// A small two-way panel shaped like the felm() documentation example (DGP
    /// recovery only; no R value is compared).
    ///
    /// R code that generates the same data:
    /// ```r
    /// library(lfe)
    /// set.seed(42)
    ///
    /// # Small reproducible example (n=20 for exact comparison)
    /// n <- 20
    /// d <- data.frame(
    ///   x1 = c(0.37, -0.56, 0.36, 0.63, 0.40, -0.11, 1.51, -0.09, 2.02, -0.06,
    ///          1.30, 2.29, -1.39, -0.28, -0.13, 0.64, -0.28, -2.66, 2.40, -0.13),
    ///   x2 = c(-0.31, -1.78, -0.17, 0.98, -1.07, -0.14, -0.43, -0.62, 1.04, -0.66,
    ///          -0.68, 0.18, -0.32, 1.10, -1.25, -0.57, 0.82, 0.69, 0.55, -0.06),
    ///   id = factor(c(1, 2, 1, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2)),
    ///   firm = factor(c(1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1))
    /// )
    ///
    /// # True coefficients: beta1 = 1.0, beta2 = 0.5
    /// id.eff <- c(0.5, -0.3, 0.2)  # Fixed effects for id 1,2,3
    /// firm.eff <- c(1.0, 0.0, -0.5)  # Fixed effects for firm 1,2,3
    ///
    /// d$y <- 1.0 * d$x1 + 0.5 * d$x2 + id.eff[d$id] + firm.eff[d$firm] +
    ///        c(0.1, -0.05, 0.08, -0.12, 0.03, 0.07, -0.04, 0.11, -0.06, 0.02,
    ///          -0.09, 0.05, 0.13, -0.08, 0.04, -0.07, 0.06, 0.01, -0.03, 0.09)
    ///
    /// est <- felm(y ~ x1 + x2 | id + firm, data = d)
    /// ```
    ///
    /// The tests on this data check DGP recovery (beta1 = 1.0, beta2 = 0.5) and
    /// the degrees of freedom only. No R output is pinned: the figures once quoted
    /// here (x1 0.99766, se 0.02834) are not what R gives on this data (fixest:
    /// x1 0.9802154032, se 0.01506135460; x2 0.4826382350, se 0.02363228724;
    /// 13 residual df). An R-pinned version needs its own reference case.
    fn create_recovery_panel() -> Dataset {
        let df = df! {
            "x1" => [0.37, -0.56, 0.36, 0.63, 0.40, -0.11, 1.51, -0.09, 2.02, -0.06,
                     1.30, 2.29, -1.39, -0.28, -0.13, 0.64, -0.28, -2.66, 2.40, -0.13],
            "x2" => [-0.31, -1.78, -0.17, 0.98, -1.07, -0.14, -0.43, -0.62, 1.04, -0.66,
                     -0.68, 0.18, -0.32, 1.10, -1.25, -0.57, 0.82, 0.69, 0.55, -0.06],
            "id" => [1i64, 2, 1, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2],
            "firm" => [1i64, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1],
            // y = 1.0*x1 + 0.5*x2 + id_effect + firm_effect + noise
            // id effects: [0.5, -0.3, 0.2], firm effects: [1.0, 0.0, -0.5]
            "y" => [
                1.0*0.37 + 0.5*(-0.31) + 0.5 + 1.0 + 0.1,    // id=1, firm=1
                1.0*(-0.56) + 0.5*(-1.78) + (-0.3) + 1.0 + (-0.05),
                1.0*0.36 + 0.5*(-0.17) + 0.5 + 0.0 + 0.08,   // id=1, firm=2
                1.0*0.63 + 0.5*0.98 + 0.2 + 0.0 + (-0.12),
                1.0*0.40 + 0.5*(-1.07) + (-0.3) + (-0.5) + 0.03,
                1.0*(-0.11) + 0.5*(-0.14) + 0.2 + (-0.5) + 0.07,
                1.0*1.51 + 0.5*(-0.43) + 0.5 + 1.0 + (-0.04),
                1.0*(-0.09) + 0.5*(-0.62) + (-0.3) + 1.0 + 0.11,
                1.0*2.02 + 0.5*1.04 + 0.2 + 0.0 + (-0.06),
                1.0*(-0.06) + 0.5*(-0.66) + 0.5 + 0.0 + 0.02,
                1.0*1.30 + 0.5*(-0.68) + (-0.3) + (-0.5) + (-0.09),
                1.0*2.29 + 0.5*0.18 + 0.2 + (-0.5) + 0.05,
                1.0*(-1.39) + 0.5*(-0.32) + 0.5 + 1.0 + 0.13,
                1.0*(-0.28) + 0.5*1.10 + (-0.3) + 1.0 + (-0.08),
                1.0*(-0.13) + 0.5*(-1.25) + 0.2 + 0.0 + 0.04,
                1.0*0.64 + 0.5*(-0.57) + 0.5 + 0.0 + (-0.07),
                1.0*(-0.28) + 0.5*0.82 + (-0.3) + (-0.5) + 0.06,
                1.0*(-2.66) + 0.5*0.69 + 0.2 + (-0.5) + 0.01,
                1.0*2.40 + 0.5*0.55 + 0.5 + 1.0 + (-0.03),
                1.0*(-0.13) + 0.5*(-0.06) + (-0.3) + 1.0 + 0.09
            ],
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_hdfe_coefficients_recovers_dgp() {
        // DGP recovery on create_recovery_panel() (no R output is compared).
        //
        // The key validation is that we recover coefficients close to
        // the true DGP values (beta1=1.0, beta2=0.5), which demonstrates
        // that the MAP algorithm correctly removes the fixed effects.
        let dataset = create_recovery_panel();

        let result = run_hdfe(
            &dataset,
            "y",
            &["x1", "x2"],
            &["id", "firm"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        // True coefficients used in data generation
        let true_beta1 = 1.0;
        let true_beta2 = 0.5;

        // Check that we're recovering coefficients close to true values
        // (within 0.05 for a small sample of n=20 with noise)
        assert!(
            (result.coefficients[0] - true_beta1).abs() < 0.05,
            "x1 should be close to true value {}, got {}",
            true_beta1,
            result.coefficients[0]
        );
        assert!(
            (result.coefficients[1] - true_beta2).abs() < 0.05,
            "x2 should be close to true value {}, got {}",
            true_beta2,
            result.coefficients[1]
        );

        // The computed values, for reference (not compared with R)
        println!(
            "HDFE Coefficients: x1={:.6}, x2={:.6}",
            result.coefficients[0], result.coefficients[1]
        );
        println!("True values: x1={}, x2={}", true_beta1, true_beta2);
    }

    #[test]
    fn test_hdfe_standard_errors_properties() {
        // This test validates SE computation and degrees of freedom.
        //
        // For a model with n=20, k=2 regressors, and two FE dimensions
        // (3 id levels + 3 firm levels), the degrees of freedom are:
        //   df_absorbed = 3 + 3 - 1 = 5 (subtract 1 for redundant grand mean)
        //   df_resid = 20 - 2 - 5 = 13
        let dataset = create_recovery_panel();

        let result = run_hdfe(
            &dataset,
            "y",
            &["x1", "x2"],
            &["id", "firm"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        // Check degrees of freedom calculation
        // n=20, k=2, df_absorbed=5 (3 id + 3 firm - 1 redundant)
        assert_eq!(result.n_obs, 20, "Should have 20 observations");
        assert_eq!(
            result.df_resid, 13,
            "df_resid should be 13 (20 - 2 - 5), got {}",
            result.df_resid
        );

        // Standard errors should be positive and reasonable
        assert!(
            result.std_errors[0] > 0.0 && result.std_errors[0] < 0.1,
            "x1 SE should be positive and small, got {}",
            result.std_errors[0]
        );
        assert!(
            result.std_errors[1] > 0.0 && result.std_errors[1] < 0.1,
            "x2 SE should be positive and small, got {}",
            result.std_errors[1]
        );

        // Residual SE should be close to noise level (~0.08 based on noise in DGP)
        assert!(
            result.residual_std_error > 0.0 && result.residual_std_error < 0.2,
            "Residual SE should be small, got {}",
            result.residual_std_error
        );

        // t-stats should be significant (large) given true coefficients
        assert!(
            result.t_stats[0].abs() > 10.0,
            "x1 t-stat should be large, got {}",
            result.t_stats[0]
        );
        assert!(
            result.t_stats[1].abs() > 5.0,
            "x2 t-stat should be large, got {}",
            result.t_stats[1]
        );

        // The computed values, for reference (not compared with R)
        println!(
            "HDFE Standard Errors: x1={:.6}, x2={:.6}",
            result.std_errors[0], result.std_errors[1]
        );
        println!(
            "Residual SE: {:.6}, df_resid: {}",
            result.residual_std_error, result.df_resid
        );
    }

    #[test]
    fn test_hdfe_r_squared_properties() {
        // DGP check: with coefficients 1.0 and 0.5 and small noise, the within
        // R² (after projecting out the fixed effects) is high. No R value is
        // compared.
        let dataset = create_recovery_panel();

        let result = run_hdfe(
            &dataset,
            "y",
            &["x1", "x2"],
            &["id", "firm"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        // Within R² should be very high given the data was generated with
        // true coefficients 1.0 and 0.5 plus small noise
        assert!(
            result.r_squared_within > 0.95,
            "Within R² should be > 0.95, got {}",
            result.r_squared_within
        );
    }

    /// Validation with single fixed effect to match standard within-estimator.
    ///
    /// R code to verify:
    /// ```r
    /// library(lfe)
    /// library(plm)
    ///
    /// # Create panel data with known DGP: y = 2*x + id_effect + noise
    /// d <- data.frame(
    ///   id = factor(rep(1:3, each=4)),
    ///   t = rep(1:4, 3),
    ///   x = c(1.0, 2.0, 3.0, 4.0,   # id=1
    ///         1.5, 2.5, 3.5, 4.5,   # id=2
    ///         2.0, 3.0, 4.0, 5.0)   # id=3
    /// )
    /// # id effects: 1=0, 2=5, 3=10
    /// # true beta = 2.0
    /// noise <- c(0.1, -0.1, 0.05, -0.05, 0.08, -0.08, 0.03, -0.03, 0.06, -0.06, 0.02, -0.02)
    /// d$y <- 2.0 * d$x + c(rep(0, 4), rep(5, 4), rep(10, 4)) + noise
    ///
    /// # Estimate with felm
    /// est <- felm(y ~ x | id, data = d)
    /// coef(est)  # Should be close to 2.0
    ///
    /// # Compare with plm within estimator
    /// pdata <- pdata.frame(d, index=c("id", "t"))
    /// fe_plm <- plm(y ~ x, data=pdata, model="within")
    /// coef(fe_plm)  # Should match
    /// ```
    #[test]
    fn test_single_fe_against_within_estimator_recovers_dgp() {
        // Data with known DGP: y = 2*x + id_effect + noise
        // id effects: 1=0, 2=5, 3=10
        let df = df! {
            "id" => [1i64, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "t" => [1i64, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            "x" => [1.0, 2.0, 3.0, 4.0,    // id=1
                    1.5, 2.5, 3.5, 4.5,    // id=2 (different x pattern)
                    2.0, 3.0, 4.0, 5.0],   // id=3 (different x pattern)
            // y = 2*x + id_effect + noise
            "y" => [
                2.0*1.0 + 0.0 + 0.1,   2.0*2.0 + 0.0 - 0.1,
                2.0*3.0 + 0.0 + 0.05,  2.0*4.0 + 0.0 - 0.05,
                2.0*1.5 + 5.0 + 0.08,  2.0*2.5 + 5.0 - 0.08,
                2.0*3.5 + 5.0 + 0.03,  2.0*4.5 + 5.0 - 0.03,
                2.0*2.0 + 10.0 + 0.06, 2.0*3.0 + 10.0 - 0.06,
                2.0*4.0 + 10.0 + 0.02, 2.0*5.0 + 10.0 - 0.02
            ],
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_hdfe(
            &dataset,
            "y",
            &["x"],
            &["id"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        // With single FE, should converge in 1 iteration
        assert_eq!(result.iterations, 1, "Single FE should converge in 1 pass");

        // Coefficient should be close to 2.0 (the true value used to generate data)
        assert!(
            (result.coefficients[0] - 2.0).abs() < 0.05,
            "Coefficient should be close to 2.0, got {}",
            result.coefficients[0]
        );

        // Check FE info
        assert_eq!(result.fe_dimensions.len(), 1);
        assert_eq!(result.fe_counts[0], 3); // 3 entities

        // Document for comparison with R
        println!(
            "Single FE coefficient: {:.6} (true value: 2.0)",
            result.coefficients[0]
        );
    }

    /// Large-scale validation following felm() documentation example structure.
    ///
    /// R code (reference for generating expected patterns):
    /// ```r
    /// library(lfe)
    /// set.seed(42)
    /// n <- 1000
    ///
    /// d <- data.frame(
    ///   x1 = rnorm(n),
    ///   x2 = rnorm(n),
    ///   id = factor(sample(20, n, replace = TRUE)),
    ///   firm = factor(sample(13, n, replace = TRUE))
    /// )
    ///
    /// id.eff <- rnorm(20)
    /// firm.eff <- rnorm(13)
    ///
    /// # True: beta1 = 1.0, beta2 = 0.5
    /// d$y <- 1.0 * d$x1 + 0.5 * d$x2 + id.eff[d$id] + firm.eff[d$firm] + rnorm(n)
    ///
    /// est <- felm(y ~ x1 + x2 | id + firm, data = d)
    /// summary(est)
    ///
    /// # With n=1000 and seed=42, coefficients should be within 0.05 of true values
    /// # with high probability due to large sample size
    /// ```
    ///
    /// Note: This test uses a deterministic approximation of the R random data
    /// to ensure reproducibility without requiring R.
    #[test]
    fn test_large_panel_coefficient_recovery_recovers_dgp() {
        // Create a larger panel with known structure
        // 50 observations, 5 entities, 10 time periods
        let n = 50;
        let n_entities = 5;
        let n_time = 10;

        // Generate pseudo-random but deterministic data
        let mut x1 = Vec::with_capacity(n);
        let mut x2 = Vec::with_capacity(n);
        let mut id = Vec::with_capacity(n);
        let mut firm = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);

        // Entity effects (fixed)
        let id_eff = [0.5, -0.3, 0.8, -0.2, 0.1];
        // Time effects (fixed)
        let time_eff = [0.0, 0.2, -0.1, 0.3, -0.2, 0.4, -0.3, 0.1, 0.0, -0.1];

        // True coefficients
        let beta1 = 1.5;
        let beta2 = -0.8;

        for i in 0..n_entities {
            for t in 0..n_time {
                let idx = i * n_time + t;
                // Pseudo-random x values based on position
                let x1_val = ((idx * 17 + 3) % 100) as f64 / 50.0 - 1.0;
                let x2_val = ((idx * 23 + 7) % 100) as f64 / 50.0 - 1.0;
                // Add idiosyncratic variation
                let noise = ((idx * 31 + 11) % 100) as f64 / 500.0 - 0.1;

                x1.push(x1_val);
                x2.push(x2_val);
                id.push((i + 1) as i64);
                firm.push((t + 1) as i64);
                y.push(beta1 * x1_val + beta2 * x2_val + id_eff[i] + time_eff[t] + noise);
            }
        }

        let df = df! {
            "x1" => x1,
            "x2" => x2,
            "id" => id,
            "firm" => firm,
            "y" => y,
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_hdfe(
            &dataset,
            "y",
            &["x1", "x2"],
            &["id", "firm"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();

        assert!(result.converged, "Should converge");

        // Coefficients should recover true values closely
        // (tolerance allows for noise and numerical precision)
        assert!(
            (result.coefficients[0] - beta1).abs() < 0.1,
            "x1 coefficient should be close to {}, got {}",
            beta1,
            result.coefficients[0]
        );
        assert!(
            (result.coefficients[1] - beta2).abs() < 0.1,
            "x2 coefficient should be close to {}, got {}",
            beta2,
            result.coefficients[1]
        );

        // Should have correct FE structure
        assert_eq!(result.fe_counts[0], n_entities);
        assert_eq!(result.fe_counts[1], n_time);
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // R-pinned validation: Grunfeld (1958), case `panel_hdfe_grunfeld`
    // (validation/r/cases/panel.R; fixest 0.14.2, plm 2.6.7), registered in
    // validation/methods/panel.json
    //
    // On a balanced firm x year panel the two demeaning projections commute, so
    // the MAP is exact after one sweep and the README tolerances apply without
    // an iterative allowance.
    //
    // Not compared: `se_cluster_firm` (run_hdfe has no cluster option),
    // `adj_r2_within_fixest` (fixest's (n - N)/df convention; p2a reports plm's
    // adjusted within R-squared) and `r2_full` (not reported).
    // ═══════════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_validate_hdfe_grunfeld() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("panel_hdfe_grunfeld");
        let r = run_hdfe(
            &ds,
            "inv",
            &["value", "capital"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();
        let n = r.n_obs;
        Check::new(&case)
            .named("coef", &r.variables, &r.coefficients, Tol::coef(n))
            .named("se_iid", &r.variables, &r.std_errors, Tol::se(n))
            .scalar("r2_within", r.r_squared_within, Tol::stat(n))
            .scalar("adj_r2_within_plm", r.adj_r_squared_within, Tol::stat(n))
            .scalar("within_f", r.f_stat, Tol::stat(n))
            .scalar("within_f_df1", r.variables.len() as f64, Tol::exact())
            .scalar("within_f_df2", r.df_resid as f64, Tol::exact())
            .scalar("within_f_p", r.f_p_value, Tol::p_value(n))
            .scalar("df_resid", r.df_resid as f64, Tol::exact())
            .scalar("n_obs", n as f64, Tol::exact())
            .finish();
        assert_eq!(r.fe_counts, vec![10, 20]);
    }

    /// HC1 (n / df_resid) standard errors, t and p against fixest's "hetero".
    #[test]
    fn test_validate_hdfe_grunfeld_hc1() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("panel_hdfe_grunfeld");
        let r = run_hdfe(
            &ds,
            "inv",
            &["value", "capital"],
            &["firm", "year"],
            None,
            CovarianceType::HC1,
        )
        .unwrap();
        let n = r.n_obs;
        Check::new(&case)
            .named("coef", &r.variables, &r.coefficients, Tol::coef(n))
            .named("se_hetero", &r.variables, &r.std_errors, Tol::se(n))
            .named("t_hetero", &r.variables, &r.t_stats, Tol::stat(n))
            .named("p_hetero", &r.variables, &r.p_values, Tol::p_value(n))
            .finish();
    }

    /// Firm effects only: the one-way within estimator (fixest, equal to plm's).
    #[test]
    fn test_validate_hdfe_grunfeld_firm_only() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("panel_hdfe_grunfeld");
        let r = run_hdfe(
            &ds,
            "inv",
            &["value", "capital"],
            &["firm"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();
        let n = r.n_obs;
        Check::new(&case)
            .named(
                "coef_firm_only",
                &r.variables,
                &r.coefficients,
                Tol::coef(n),
            )
            .named("se_firm_only_iid", &r.variables, &r.std_errors, Tol::se(n))
            .finish();
    }

    #[test]
    fn test_hdfe_rejects_invalid_iteration_controls() {
        let ds = create_balanced_panel();
        let run = |config: HdfeConfig| {
            run_hdfe(
                &ds,
                "y",
                &["x"],
                &["firm", "year"],
                Some(config),
                CovarianceType::HC1,
            )
        };
        for tolerance in [0.0, -1e-8, f64::NAN, f64::INFINITY] {
            expect_invalid(
                run(HdfeConfig {
                    tolerance,
                    ..Default::default()
                }),
                "tolerance",
            );
        }
        expect_invalid(
            run(HdfeConfig {
                max_iterations: 0,
                ..Default::default()
            }),
            "max_iterations",
        );
    }

    /// `create_balanced_panel`'s columns with `y[0] = y0`, keeping row 0 or not.
    fn balanced_panel_with_y0(y0: f64, keep_row0: bool) -> Dataset {
        let firm = ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"];
        let year = [
            2020i64, 2021, 2022, 2023, 2020, 2021, 2022, 2023, 2020, 2021, 2022, 2023,
        ];
        let x = [1.0, 3.0, 2.0, 5.0, 2.0, 1.0, 4.0, 3.0, 3.0, 4.0, 1.0, 2.0];
        let y = [
            y0,
            2.0 * 3.0 + 0.0 + 1.0 + 0.2,
            2.0 * 2.0 + 0.0 + 2.0 + 0.1,
            2.0 * 5.0 + 0.0 + 3.0 + 0.0,
            2.0 * 2.0 + 5.0 + 0.0 + 0.0,
            2.0 * 1.0 + 5.0 + 1.0 + 0.1,
            2.0 * 4.0 + 5.0 + 2.0 - 0.1,
            2.0 * 3.0 + 5.0 + 3.0 + 0.2,
            2.0 * 3.0 + 10.0 + 0.0 + 0.2,
            2.0 * 4.0 + 10.0 + 1.0 - 0.1,
            2.0 * 1.0 + 10.0 + 2.0 + 0.0,
            2.0 * 2.0 + 10.0 + 3.0 - 0.2,
        ];
        let from = usize::from(!keep_row0);
        Dataset::new(
            df! {
                "firm" => &firm[from..],
                "year" => &year[from..],
                "x" => &x[from..],
                "y" => &y[from..],
            }
            .unwrap(),
        )
    }

    /// A NaN in y is a missing cell (fixest drops it): the fit equals the
    /// fit without that row and reports it; an infinite value is refused by name.
    #[test]
    fn test_hdfe_non_finite_data() {
        let fit = |ds: &Dataset| {
            run_hdfe(
                ds,
                "y",
                &["x"],
                &["firm", "year"],
                None,
                CovarianceType::HC1,
            )
        };
        let r = fit(&balanced_panel_with_y0(f64::NAN, true)).unwrap();
        let reference = fit(&balanced_panel_with_y0(0.0, false)).unwrap();
        let tol = p2a_validation::Tol::new(1e-12, 1e-12);
        for (a, b) in r
            .coefficients
            .iter()
            .chain(&r.std_errors)
            .zip(reference.coefficients.iter().chain(&reference.std_errors))
        {
            assert!(tol.allows(*a, *b), "{a} vs {b}");
        }
        assert_eq!(r.missing.n_dropped, 1);
        assert_eq!(r.n_obs, 11);
        match fit(&balanced_panel_with_y0(f64::INFINITY, true)) {
            Err(EconError::NonFiniteValues { column, count }) => {
                assert_eq!((column.as_str(), count), ("y", 1))
            }
            other => panic!("expected NonFiniteValues for y, got {other:?}"),
        }
    }

    /// Two-way HDFE on Grunfeld with missing cells, missing ids and missing years,
    /// pinned to `fixest::feols(| firm_name + year, vcov = "iid")` on the 193 complete
    /// rows; the numeric and the text firm id give the same partition. ECON-37: with y
    /// and x complete, missing ids and years used to become level 0 (merged into firm 0
    /// and a real year); they now drop their rows.
    #[test]
    fn test_validate_hdfe_grunfeld_na() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("panel_grunfeld_na");
        let subset = [
            "inv".to_string(),
            "value".to_string(),
            "capital".to_string(),
        ];
        let xy = Dataset::new(ds.df().drop_nulls(Some(&subset)).unwrap());
        let check = |r: &HdfeResult| {
            let n = r.n_obs;
            Check::new(&case)
                .named("hdfe_coef", &r.variables, &r.coefficients, Tol::coef(n))
                .named("hdfe_se_iid", &r.variables, &r.std_errors, Tol::se(n))
                .scalar("hdfe_nobs", n as f64, Tol::exact())
                .scalar("n_groups", r.fe_counts[0] as f64, Tol::exact())
                .scalar("n_years", r.fe_counts[1] as f64, Tol::exact())
                .finish();
        };
        for fe in [["firm_name", "year"], ["firm_id", "year"]] {
            let r = run_hdfe(
                &ds,
                "inv",
                &["value", "capital"],
                &fe,
                None,
                CovarianceType::Standard,
            )
            .unwrap();
            check(&r);
            Check::new(&case)
                .scalar("hdfe_n_removed", r.missing.n_dropped as f64, Tol::exact())
                .finish();
            let note = format!(
                "7 of 200 rows dropped because of missing values (inv: 2, value: 1, {}: 2, year: 2)",
                fe[0]
            );
            assert_eq!(r.missing.note().unwrap(), note);
            let text = r.to_string();
            assert!(
                text.trim_end()
                    .ends_with(&format!("Missing values: {note}")),
                "{text}"
            );
        }
        let r = run_hdfe(
            &xy,
            "inv",
            &["value", "capital"],
            &["firm_id", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();
        check(&r);
        assert_eq!(r.missing.n_dropped, 4);
    }

    #[test]
    fn test_hdfe_stops_when_cancelled() {
        let ds = create_balanced_panel();
        expect_cancelled(with_budget(cancelled(), || {
            run_hdfe(
                &ds,
                "y",
                &["x"],
                &["firm", "year"],
                None,
                CovarianceType::HC1,
            )
        }));
    }

    #[test]
    fn test_demean_matrix_map_carries_budget_into_rayon_workers() {
        // factor1/factor2 exactly as in test_demean_map_two_factors
        let factor1 = FactorInfo {
            name: "entity".to_string(),
            n_levels: 2,
            ids: vec![0, 0, 0, 1, 1, 1],
            counts: vec![3, 3],
        };
        let factor2 = FactorInfo {
            name: "time".to_string(),
            n_levels: 3,
            ids: vec![0, 1, 2, 0, 1, 2],
            counts: vec![2, 2, 2],
        };
        let x = Array2::from_shape_fn((6, 2), |(i, j)| (i * (j + 2)) as f64 % 5.0 + 0.1 * j as f64);
        expect_cancelled(with_budget(cancelled(), || {
            demean_matrix_map(&x, &[factor1, factor2], 1e-8, 1000, true)
        }));
    }

    /// One factor from PARALLEL_DEMEAN_MIN_ROWS rows: the columns, demeaned in
    /// parallel, equal the column-by-column result bit for bit.
    #[test]
    fn test_demean_matrix_map_one_factor_parallel_equals_serial() {
        let (n, k) = (5_000, 4);
        let levels = 437;
        let ids: Vec<usize> = (0..n).map(|i| (i * 7919 + i / 13) % levels).collect();
        let mut counts = vec![0; levels];
        for &g in &ids {
            counts[g] += 1;
        }
        let factor = FactorInfo {
            name: "g".to_string(),
            n_levels: levels,
            ids,
            counts,
        };
        let x = Array2::from_shape_fn((n, k), |(i, j)| {
            ((i * (j + 3) * 104_729) % 1_009) as f64 / 97.0 + 0.01 * (i % (j + 5)) as f64
        });
        let factors = [factor];
        let (all, _, _, converged) = demean_matrix_map(&x, &factors, 1e-8, 1000, true).unwrap();
        assert!(converged);
        for j in 0..k {
            let (col, _, _, _) =
                demean_map(&x.column(j).to_owned(), &factors, 1e-8, 1000, true).unwrap();
            for i in 0..n {
                assert_eq!(all[[i, j]].to_bits(), col[i].to_bits(), "({i}, {j})");
            }
        }
    }

    // =========================================================================
    // T3-WP17: relative MAP rule (ECON-69), absorbed regressors (ECON-72),
    // disconnected two-way effects (X-16)
    // =========================================================================

    #[test]
    fn test_validate_hdfe_converges_at_scale_1e9() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("hdfe_map_scale");
        let r = run_hdfe(
            &ds,
            "y_big",
            &["x"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();
        assert!(r.converged, "{:?}", r.warnings);
        // MAP to relative 1e-8; R replica 6.7e-16 from fixest
        Check::new(&case)
            .named(
                "coef_big",
                &r.variables,
                &r.coefficients,
                Tol::iterative(1e-8),
            )
            .named(
                "se_iid_big",
                &r.variables,
                &r.std_errors,
                Tol::iterative(1e-6),
            )
            .scalar("df_resid", r.df_resid as f64, Tol::exact())
            .finish();
        let r1 = run_hdfe(
            &ds,
            "y",
            &["x"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();
        Check::new(&case)
            .named(
                "coef",
                &r1.variables,
                &r1.coefficients,
                Tol::iterative(1e-8),
            )
            .named(
                "se_iid",
                &r1.variables,
                &r1.std_errors,
                Tol::iterative(1e-6),
            )
            .scalar("r2_within", r1.r_squared_within, Tol::iterative(1e-6))
            .scalar("n_obs", r1.n_obs as f64, Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_hdfe_map_scale_invariant_iterations() {
        // y * 1e-6, y, y * 1e6: the stopping rule must not depend on units. R replica of
        // the relative rule: 10/10/10 sweeps (the absolute rule: 11/11/18).
        let (_case, ds) = crate::test_support::load_case("hdfe_map_scale");
        use crate::regression::ols::na_fixtures::with_derived;
        let ds = with_derived(
            &with_derived(&ds, "y_small", "y", |v| v * 1e-6),
            "y_large",
            "y",
            |v| v * 1e6,
        );
        let fit = |col: &str| {
            run_hdfe(
                &ds,
                col,
                &["x"],
                &["firm", "year"],
                None,
                CovarianceType::Standard,
            )
            .unwrap()
        };
        let (s, m, l) = (fit("y_small"), fit("y"), fit("y_large"));
        for r in [&s, &l] {
            assert!(
                r.iterations.abs_diff(m.iterations) <= 1,
                "{} vs {}",
                r.iterations,
                m.iterations
            );
        }
        assert!((s.coefficients[0] / 1e-6 / m.coefficients[0] - 1.0).abs() < 1e-10);
        assert!((l.coefficients[0] / 1e6 / m.coefficients[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_validate_hdfe_map_iteration_limit_is_reported() {
        let (_case, ds) = crate::test_support::load_case("hdfe_map_scale");
        let config = HdfeConfig {
            max_iterations: 2,
            ..HdfeConfig::default()
        };
        let r = run_hdfe(
            &ds,
            "y",
            &["x"],
            &["firm", "year"],
            Some(config),
            CovarianceType::Standard,
        )
        .unwrap();
        assert!(!r.converged);
        assert!(
            r.warnings.iter().any(|w| w
                .starts_with("HDFE alternating projections did not converge after 2 iterations")),
            "{:?}",
            r.warnings
        );
        assert!(format!("{r}").contains("Converged: no (2 iterations"));
    }

    #[test]
    fn test_validate_hdfe_absorbed_regressors() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("panel_absorbed_regressors");
        let n = 180;
        // MAP exact after 2 sweeps on this balanced design: README tolerances
        let two = run_hdfe(
            &ds,
            "y",
            &["x1", "x2", "z", "w"],
            &["id", "t"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();
        assert_eq!(two.dropped, vec!["z".to_string(), "w".to_string()]);
        assert!(
            two.warnings.iter().any(|w| w
                == "'z', 'w' dropped: no variation left after removing the 'id' + 't' fixed \
                    effects (not estimated, as in plm and fixest)"),
            "{:?}",
            two.warnings
        );
        Check::new(&case)
            .named(
                "hdfe2_coef",
                &two.variables,
                &two.coefficients,
                Tol::coef(n),
            )
            .named("hdfe2_se_iid", &two.variables, &two.std_errors, Tol::se(n))
            .scalar("hdfe2_df_resid", two.df_resid as f64, Tol::exact())
            .finish();
        let one = run_hdfe(
            &ds,
            "y",
            &["x1", "x2", "z", "w"],
            &["id"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();
        assert_eq!(one.dropped, vec!["z".to_string()]);
        Check::new(&case)
            .named(
                "hdfe1_coef",
                &one.variables,
                &one.coefficients,
                Tol::coef(n),
            )
            .named("hdfe1_se_iid", &one.variables, &one.std_errors, Tol::se(n))
            .finish();
        let x3 = run_hdfe(
            &ds,
            "y",
            &["x1", "x2", "x3"],
            &["id", "t"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();
        assert_eq!(x3.dropped, vec!["x3".to_string()]);
        assert!(x3.warnings.iter().any(|w| w.starts_with(
            "'x3' dropped: collinear with the other regressors after removing the 'id' + 't' \
             fixed effects"
        )));
        Check::new(&case)
            .named(
                "hdfe2_x3_coef",
                &x3.variables,
                &x3.coefficients,
                Tol::coef(n),
            )
            .finish();
    }

    #[test]
    fn test_hdfe_all_regressors_absorbed_is_an_error() {
        // create_unbalanced_panel's x is year - 2019 in every row: absorbed by the year effects.
        let e = run_hdfe(
            &create_unbalanced_panel(),
            "y",
            &["x"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message.starts_with("no regressor left to estimate: 'x' dropped")),
            "{e:?}"
        );
    }

    #[test]
    fn test_screen_regressors_thresholds() {
        // 3 firms x 4 years, two-way effects
        let firm = FactorInfo {
            name: "firm".to_string(),
            n_levels: 3,
            ids: (0..12).map(|i| i / 4).collect(),
            counts: vec![4, 4, 4],
        };
        let year = FactorInfo {
            name: "year".to_string(),
            n_levels: 4,
            ids: (0..12).map(|i| i % 4).collect(),
            counts: vec![3, 3, 3, 3],
        };
        let x1 = [
            0.3, -1.2, 0.8, 2.1, -0.4, 1.7, -0.9, 0.2, 1.1, -0.6, 0.5, -1.8,
        ];
        let x2 = [
            1.4, 0.2, -0.7, 0.9, -1.5, 0.6, 1.9, -0.3, 0.1, -1.1, 0.75, 0.4,
        ];
        let x3 = [
            -0.5, 0.9, 1.3, -1.6, 0.2, -0.8, 0.45, 1.2, -1.9, 0.3, 0.65, -0.1,
        ];
        let (a, b) = ([1.0, -2.0, 0.5], [0.3, 1.1, -0.7, 2.0]);
        let ab: Vec<f64> = (0..12).map(|i| a[i / 4] + b[i % 4]).collect();
        let x12: Vec<f64> = (0..12).map(|i| x1[i] + x2[i]).collect();
        let r: Vec<f64> = (0..12).map(|i| ab[i] + 0.01 * x3[i]).collect();
        let raw = Array2::from_shape_fn((12, 6), |(i, j)| match j {
            0 => x1[i],
            1 => x2[i],
            2 => ab[i],
            3 => x12[i],
            4 => r[i],
            _ => 7.0,
        });
        let factors = [firm, year];
        let (mut dm, _, _, converged) =
            demean_matrix_map(&raw, &factors, 1e-10, 1000, true).unwrap();
        assert!(converged);
        // Column 4: its demeaned sum of squares set to 1e-10 of its centred one (above
        // the 1e-12 threshold), along the demeaned x3 (not collinear with x1, x2)
        let (x3_dm, _, _, _) =
            demean_map(&Array1::from(x3.to_vec()), &factors, 1e-10, 1000, true).unwrap();
        let mean = r.iter().sum::<f64>() / 12.0;
        let ss_c: f64 = r.iter().map(|v| (v - mean).powi(2)).sum();
        let ss_x3: f64 = x3_dm.iter().map(|v| v * v).sum();
        dm.column_mut(4)
            .assign(&(&x3_dm * (1e-10 * ss_c / ss_x3).sqrt()));
        let names: Vec<String> = ["x1", "x2", "ab", "x12", "r", "c"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let screen = screen_regressors(raw.view(), dm.view(), &names, "'firm' + 'year'").unwrap();
        assert_eq!(screen.kept, vec![0, 1, 4]);
        assert_eq!(screen.dropped, vec!["ab", "c", "x12"]);
        assert_eq!(
            screen.warnings,
            vec![
                "'ab', 'c' dropped: no variation left after removing the 'firm' + 'year' fixed \
                 effects (not estimated, as in plm and fixest)"
                    .to_string(),
                "'x12' dropped: collinear with the other regressors after removing the 'firm' + \
                 'year' fixed effects (the later of the collinear regressors is not estimated, \
                 as in fixest)"
                    .to_string(),
            ]
        );
        assert_eq!(fe_label(&["id"]), "'id'");
        assert_eq!(fe_label(&["id", "t"]), "'id' + 't'");
    }

    #[test]
    fn test_validate_hdfe_disconnected_components() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("hdfe_disconnected");
        let n = 80;
        // MAP exact after 2 sweeps on this design (balanced blocks): README tolerances
        let r = run_hdfe(
            &ds,
            "y",
            &["x", "z"],
            &["firm", "year"],
            None,
            CovarianceType::Standard,
        )
        .unwrap();
        Check::new(&case)
            .named("coef", &r.variables, &r.coefficients, Tol::coef(n))
            .named("se", &r.variables, &r.std_errors, Tol::se(n))
            .scalar("df_resid", r.df_resid as f64, Tol::exact())
            .scalar("df_absorbed", r.df_absorbed as f64, Tol::exact())
            .scalar("n_obs", r.n_obs as f64, Tol::exact())
            .finish();
        // the pre-5f86b49 rule (n0 + n1 - 1 = 25 absorbed, df 53) would give se_naive_df
        assert!((r.std_errors[0] / case.named("se_naive_df", "x") - 1.0).abs() > 5e-3);
    }

    // ── T3-PERF1: normal equations first, pivoted QR fallback

    /// A noisy two-way design: `k` regressors with a common component (correlated, not
    /// collinear), demeaned by firm and year; the last column is scaled by `scale`.
    fn noisy_two_way_design(
        n_firm: usize,
        n_year: usize,
        k: usize,
        scale: f64,
        seed: u64,
    ) -> (Array2<f64>, Array2<f64>, Array1<f64>, Vec<FactorInfo>) {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let nd = Normal::new(0.0, 1.0).unwrap();
        let n = n_firm * n_year;
        let firm = FactorInfo {
            name: "firm".to_string(),
            n_levels: n_firm,
            ids: (0..n).map(|i| i / n_year).collect(),
            counts: vec![n_year; n_firm],
        };
        let year = FactorInfo {
            name: "year".to_string(),
            n_levels: n_year,
            ids: (0..n).map(|i| i % n_year).collect(),
            counts: vec![n_firm; n_year],
        };
        let common: Vec<f64> = (0..n).map(|_| nd.sample(&mut rng)).collect();
        let mut raw = Array2::from_shape_fn((n, k), |(i, _)| 0.6 * common[i]);
        for v in raw.iter_mut() {
            *v += nd.sample(&mut rng);
        }
        raw.column_mut(k - 1).mapv_inplace(|v| v * scale);
        let factors = vec![firm, year];
        let (dm, _, _, converged) = demean_matrix_map(&raw, &factors, 1e-12, 10_000, true).unwrap();
        assert!(converged);
        let y_raw =
            Array1::from_shape_fn(n, |i| raw.row(i).sum() + nd.sample(&mut rng) + common[i]);
        let (y, _, _, _) = demean_map(&y_raw, &factors, 1e-12, 10_000, true).unwrap();
        (raw, dm, y, factors)
    }

    fn max_rel(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / y.abs().max(f64::MIN_POSITIVE))
            .fold(0.0, f64::max)
    }

    /// The normal-equations solve of a well-conditioned screen agrees with the pivoted
    /// QR it replaces to 1e-10 relative: coefficients, (X'X)^{-1} and the RSS, on noisy
    /// designs of several shapes and scales (one with a regressor of scale 1e4, refined).
    #[test]
    fn test_screen_normal_equations_match_qr() {
        for (nf, ny, k, scale, seed) in [
            (12, 8, 2, 1.0, 1),
            (30, 20, 4, 1.0, 2),
            (50, 10, 3, 1e4, 3),
            (7, 5, 5, 1e-3, 4),
        ] {
            let (raw, dm, y, _) = noisy_two_way_design(nf, ny, k, scale, seed);
            let names: Vec<String> = (0..k).map(|j| format!("x{}", j + 1)).collect();
            let screen = screen_regressors(raw.view(), dm.view(), &names, "'firm'").unwrap();
            assert!(screen.uses_normal_equations(), "shape {nf}x{ny}x{k}");
            assert_eq!(screen.kept, (0..k).collect::<Vec<_>>());
            assert!(screen.dropped.is_empty() && screen.warnings.is_empty());
            assert_eq!(screen.x, dm);
            let qr = Qr::decompose(dm.view(), LM_QR_TOL).unwrap();
            assert_eq!(qr.rank(), k);
            let b = screen.coefficients(&y).unwrap();
            let b_qr = qr.coefficients(y.view()).unwrap();
            assert!(max_rel(b.as_slice().unwrap(), b_qr.as_slice().unwrap()) < 1e-10);
            let u = screen.unscaled_covariance().unwrap();
            let u_qr = qr.unscaled_covariance().unwrap();
            assert!(max_rel(u.as_slice().unwrap(), u_qr.as_slice().unwrap()) < 1e-10);
            let rss_qr: f64 = qr
                .qty(y.view())
                .unwrap()
                .iter()
                .skip(k)
                .map(|v| v * v)
                .sum();
            let rss = screen.rss(&y).unwrap();
            assert!((rss / rss_qr - 1.0).abs() < 1e-10, "{rss} vs {rss_qr}");
        }
    }

    /// Nearly collinear demeaned columns (x3 = x1 + x2 + 1e-6 noise) are declined by the
    /// normal equations and decided by the pivoted QR, which keeps all three (their
    /// relative distance is far above LM_QR_TOL) and solves as before; exactly collinear
    /// ones still reach the QR and are dropped.
    #[test]
    fn test_screen_near_collinear_reaches_qr() {
        let (raw, mut dm, y, factors) = noisy_two_way_design(20, 10, 3, 1.0, 5);
        let (x1, x2) = (dm.column(0).to_owned(), dm.column(1).to_owned());
        let wiggle = dm.column(2).to_owned();
        let near = &x1 + &x2 + &(&wiggle * 1e-6);
        dm.column_mut(2).assign(&near);
        let names: Vec<String> = ["x1", "x2", "x3"].iter().map(|s| s.to_string()).collect();
        let screen = screen_regressors(raw.view(), dm.view(), &names, "'firm'").unwrap();
        assert!(!screen.uses_normal_equations());
        assert_eq!(screen.kept, vec![0, 1, 2]);
        let qr = Qr::decompose(dm.view(), LM_QR_TOL).unwrap();
        assert_eq!(
            screen.coefficients(&y).unwrap(),
            qr.coefficients(y.view()).unwrap()
        );
        // exactly collinear after the effects: the QR drops the later column
        let mut raw2 = raw.clone();
        let sum = &raw.column(0) + &raw.column(1);
        raw2.column_mut(2).assign(&sum);
        let (dm2, _, _, _) = demean_matrix_map(&raw2, &factors, 1e-12, 10_000, true).unwrap();
        let screen2 = screen_regressors(raw2.view(), dm2, &names, "'firm'").unwrap();
        assert!(!screen2.uses_normal_equations());
        assert_eq!(screen2.kept, vec![0, 1]);
        assert_eq!(screen2.dropped, vec!["x3".to_string()]);
        assert_eq!(screen2.x.ncols(), 2);
    }

    // ── HC2/HC3 with the fixed-effect leverage (hdfe_hc_leverage: sandwich::vcovHC on
    // lm with factor dummies)

    /// HC0-HC3 SEs, coefficients and df of one design against `<prefix>_*` of the case.
    fn check_hc_design(
        case: &p2a_validation::Case,
        prefix: &str,
        ds: &Dataset,
        y: &str,
        x: &[&str],
        fe: &[&str],
        coef_tol: p2a_validation::Tol,
        se_tol: p2a_validation::Tol,
    ) {
        use p2a_validation::{Check, Tol};
        for (cov, key) in [
            (CovarianceType::HC0, "hc0"),
            (CovarianceType::HC1, "hc1"),
            (CovarianceType::HC2, "hc2"),
            (CovarianceType::HC3, "hc3"),
        ] {
            let r = run_hdfe(ds, y, x, fe, None, cov).unwrap_or_else(|e| panic!("{key}: {e}"));
            Check::new(case)
                .named(
                    &format!("{prefix}_coef"),
                    &r.variables,
                    &r.coefficients,
                    coef_tol,
                )
                .named(
                    &format!("{prefix}_se_{key}"),
                    &r.variables,
                    &r.std_errors,
                    se_tol,
                )
                .scalar(
                    &format!("{prefix}_df_resid"),
                    r.df_resid as f64,
                    Tol::exact(),
                )
                .finish();
            assert!(
                !r.warnings.iter().any(|w| w.contains("leverage 1")),
                "{prefix} {key}: {:?}",
                r.warnings
            );
        }
    }

    /// Grunfeld, firm and firm + year (MAP exact on the balanced panel).
    #[test]
    fn test_validate_hdfe_hc_leverage_grunfeld() {
        use p2a_validation::Tol;
        let (case, _) = crate::test_support::load_case("hdfe_hc_leverage");
        let (_, ds) = crate::test_support::load_case("panel_hdfe_grunfeld");
        let x = ["value", "capital"];
        for (prefix, fe) in [
            ("grunfeld_firm", &["firm"][..]),
            ("grunfeld_twoway", &["firm", "year"][..]),
        ] {
            check_hc_design(
                &case,
                prefix,
                &ds,
                "inv",
                &x,
                fe,
                Tol::coef(200),
                Tol::se(200),
            );
        }
    }

    /// A short unbalanced panel (377 rows, 100 ids with 2 to 4 periods), one- and
    /// two-way.
    #[test]
    fn test_validate_hdfe_hc_leverage_short_panel() {
        use p2a_validation::Tol;
        let (case, ds) = crate::test_support::load_case("hdfe_hc_leverage");
        assert_eq!(ds.df().height(), 377);
        let x = ["x1", "x2"];
        check_hc_design(
            &case,
            "short_oneway",
            &ds,
            "y",
            &x,
            &["id"],
            Tol::coef(377),
            Tol::se(377),
        );
        check_hc_design(
            &case,
            "short_twoway",
            &ds,
            "y",
            &x,
            &["id", "t"],
            Tol::coef(377),
            Tol::se(377),
        );
    }

    /// Two connected components: the leverage matrix is singular and its generalized
    /// inverse gives the same leverage.
    #[test]
    fn test_validate_hdfe_hc_leverage_disconnected() {
        use p2a_validation::Tol;
        let (case, _) = crate::test_support::load_case("hdfe_hc_leverage");
        let (_, ds) = crate::test_support::load_case("hdfe_disconnected");
        check_hc_design(
            &case,
            "disconnected",
            &ds,
            "y",
            &["x", "z"],
            &["firm", "year"],
            Tol::coef(80),
            Tol::se(80),
        );
    }

    /// A noisy panel with three fixed effects (24 rows).
    fn three_way_panel() -> Dataset {
        let n = 24;
        let a: Vec<i64> = (0..n).map(|i| (i % 4) as i64).collect();
        let b: Vec<i64> = (0..n).map(|i| (i % 3) as i64).collect();
        let c: Vec<i64> = (0..n).map(|i| (i / 12) as i64).collect();
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 37) % 11) as f64 * 0.3 + 0.1 * i as f64)
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 0.7 * x[i] + ((i * 7919) % 101) as f64 / 101.0 + a[i] as f64)
            .collect();
        Dataset::new(df! {"a" => a, "b" => b, "c" => c, "x" => x, "y" => y}.unwrap())
    }

    #[test]
    fn test_hdfe_hc2_refuses_three_fixed_effects() {
        let ds = three_way_panel();
        for cov in [CovarianceType::HC2, CovarianceType::HC3] {
            expect_invalid(
                run_hdfe(&ds, "y", &["x"], &["a", "b", "c"], None, cov),
                "three or more fixed effects",
            );
        }
        let r = run_hdfe(
            &ds,
            "y",
            &["x"],
            &["a", "b", "c"],
            None,
            CovarianceType::HC1,
        )
        .unwrap();
        assert!(r.std_errors[0].is_finite());
    }

    #[test]
    fn test_hdfe_hc3_refuses_too_many_levels() {
        // 1,001 x 1,001 levels, connected: cells (i, i) and (i, i + 1 mod 1001), twice each
        let m = 1001usize;
        let mut fa = Vec::with_capacity(4 * m);
        let mut fb = Vec::with_capacity(4 * m);
        for _ in 0..2 {
            for i in 0..m {
                fa.push(i as i64);
                fb.push(i as i64);
                fa.push(i as i64);
                fb.push(((i + 1) % m) as i64);
            }
        }
        let n = fa.len();
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 7919) % 1009) as f64 / 1009.0)
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 0.5 * x[i] + ((i * 104_729) % 997) as f64 / 997.0)
            .collect();
        let ds = Dataset::new(df! {"fa" => fa, "fb" => fb, "x" => x, "y" => y}.unwrap());
        for cov in [CovarianceType::HC2, CovarianceType::HC3] {
            match run_hdfe(&ds, "y", &["x"], &["fa", "fb"], None, cov) {
                Err(EconError::InvalidSpecification { message }) => assert!(
                    message.contains("at most 1000 levels") && message.contains("use HC1"),
                    "{message}"
                ),
                other => panic!("expected InvalidSpecification, got {other:?}"),
            }
        }
        let r = run_hdfe(&ds, "y", &["x"], &["fa", "fb"], None, CovarianceType::HC1).unwrap();
        // connected: 2m - 1 absorbed, one slope
        assert_eq!(r.df_resid, n - (2 * m - 1) - 1);
        assert!(r.std_errors[0].is_finite());
    }

    /// The fixed-effect leverage equals the diagonal of `D (D'D)^+ D'` with `D` the
    /// dummies of both factors, on a noisy unbalanced two-way design.
    #[test]
    fn test_hdfe_fe_leverage_matches_dense_projection() {
        let n = 30;
        let a: Vec<usize> = (0..n).map(|i| (i * 11 + i / 3) % 6).collect();
        let b: Vec<usize> = (0..n).map(|i| (i * 3 + i / 4) % 4).collect();
        let info = |name: &str, ids: &[usize], levels: usize| {
            let mut counts = vec![0; levels];
            for &g in ids {
                counts[g] += 1;
            }
            FactorInfo {
                name: name.to_string(),
                n_levels: levels,
                ids: ids.to_vec(),
                counts,
            }
        };
        let factors = [info("a", &a, 6), info("b", &b, 4)];
        assert!(factors.iter().all(|f| f.counts.iter().all(|&c| c > 0)));
        let h = fe_leverage(&factors).unwrap();
        let mut d = Array2::<f64>::zeros((n, 10));
        for i in 0..n {
            d[[i, a[i]]] = 1.0;
            d[[i, 6 + b[i]]] = 1.0;
        }
        let dtd = d.t().dot(&d);
        let g = sym_inverse(dtd.view(), SingularPolicy::Generalized, "D'D")
            .unwrap()
            .inverse;
        let p = d.dot(&g).dot(&d.t());
        for i in 0..n {
            assert!(
                (h[i] - p[[i, i]]).abs() <= 1e-12,
                "{i}: {} vs {}",
                h[i],
                p[[i, i]]
            );
        }
        // one factor: 1 / T_g
        let h1 = fe_leverage(&factors[1..]).unwrap();
        for i in 0..n {
            assert_eq!(h1[i], 1.0 / factors[1].counts[b[i]] as f64);
        }
    }

    /// A singleton fixed-effect level has leverage 1: it contributes nothing to the
    /// HC2/HC3 meat, the SEs stay finite and a warning says so.
    #[test]
    fn test_hdfe_hc3_singleton_group_contributes_nothing() {
        let firm: Vec<i64> = vec![1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4];
        let x = vec![0.3, 1.1, 2.4, 0.7, 1.9, 2.2, 3.5, 1.0, 0.2, 2.8, 1.6];
        let y: Vec<f64> = x
            .iter()
            .zip(&firm)
            .enumerate()
            .map(|(i, (&x, &f))| 0.8 * x + f as f64 + ((i * 7919) % 101) as f64 / 150.0)
            .collect();
        let ds = Dataset::new(df! {"firm" => firm, "x" => x, "y" => y}.unwrap());
        for cov in [CovarianceType::HC2, CovarianceType::HC3] {
            let r = run_hdfe(&ds, "y", &["x"], &["firm"], None, cov).unwrap();
            assert!(
                r.std_errors[0].is_finite() && r.std_errors[0] > 0.0,
                "{cov:?}"
            );
            assert!(
                r.warnings.iter().any(|w| w
                    == "1 observation has leverage 1 (it is fitted exactly, e.g. a singleton \
                        fixed-effect level) and contributes nothing to the HC2/HC3 meat"),
                "{cov:?}: {:?}",
                r.warnings
            );
        }
    }
}
