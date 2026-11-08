//! Changepoint detection for time series.
//!
//! Implements the PELT (Pruned Exact Linear Time) algorithm for detecting
//! multiple changepoints in time series data.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::data::columns::{NaPolicy, numeric_column};

/// Result of changepoint detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangepointResult {
    /// Indices of detected changepoints (0-indexed positions where changes occur)
    pub changepoints: Vec<usize>,
    /// Number of changepoints detected
    pub n_changepoints: usize,
    /// Segment statistics (mean and variance for each segment)
    pub segments: Vec<SegmentStats>,
    /// Sum of the segment costs, without the penalties: changepoint's cost functions, including
    /// their ln(segment length) term under the MBIC default
    pub total_cost: f64,
    /// Penalty value used
    pub penalty: f64,
    /// Method used for detection
    pub method: String,
}

/// Statistics for a segment between changepoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentStats {
    /// Start index (inclusive)
    pub start: usize,
    /// End index (exclusive)
    pub end: usize,
    /// Number of points in segment
    pub n_points: usize,
    /// Mean of the segment
    pub mean: f64,
    /// Variance of the segment
    pub variance: f64,
}

/// Cost function type for changepoint detection, as the normal costs of the changepoint package
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CostFunction {
    /// Changes in mean under unit variance (`cpt.mean`): scale the series first, as with
    /// `cpt.mean`
    #[default]
    MeanChange,
    /// Changes in variance about the mean of the whole series (`cpt.var` with mu = mean(data))
    VarianceChange,
    /// Changes in both mean and variance (`cpt.meanvar`)
    MeanAndVariance,
}

/// Detect changepoints in a time series using the PELT algorithm.
///
/// # Arguments
/// * `data` - Time series data
/// * `penalty` - Penalty for adding a changepoint. `None` is changepoint's MBIC: (k + 2) ln(n), with
///   k = 1 for mean or variance and 2 for both, and a cost term of ln(segment length) per
///   segment. `Some(p)` is its "Manual": penalty p and the plain costs.
/// * `min_segment_length` - Minimum length of a segment (default: 2)
/// * `cost_function` - Type of change to detect
///
/// # Returns
/// * `ChangepointResult` with detected changepoints and segment statistics
///
/// The costs run on data centred by the mean of the series, so a large level does not cancel.
/// Pruning uses the cost without the ln(segment length) term (see `pelt`), and the pruned search
/// returns what optimal partitioning returns while the costs are sub-additive. A `MeanAndVariance`
/// segment whose variance takes the floor of `segment_cost` can break that, and under a small
/// manual penalty the search can then stop above the optimum (no mismatch was found with MBIC).
/// On a series at a large level `cpt.meanvar` and `cpt.mean`, which do not centre, can differ.
///
/// # Errors
/// Fails on a series too short for the minimum segment length, on a non-finite value or penalty
/// (a penalty must be finite and non-negative), on values so large that the sum of squares
/// overflows, and when the budget installed with [`crate::resource::with_budget`] is cancelled or
/// past its deadline.
pub fn detect_changepoints(
    data: &[f64],
    penalty: Option<f64>,
    min_segment_length: Option<usize>,
    cost_function: CostFunction,
) -> Result<ChangepointResult, String> {
    let n = data.len();

    if n < 2 {
        return Err("Data must have at least 2 points".to_string());
    }

    let min_seg = min_segment_length.unwrap_or(2).max(2);

    // Saturating: a caller's minimum near usize::MAX must be refused, not wrap.
    if n < min_seg.saturating_mul(2) {
        return Err(format!(
            "Data length ({}) must be at least 2x minimum segment length ({})",
            n, min_seg
        ));
    }

    if let Some(p) = penalty
        && !(p.is_finite() && p >= 0.0)
    {
        return Err(format!("penalty must be a finite number >= 0, got {p}"));
    }

    // changepoint's MBIC when no penalty is given: (k + 2) ln(n), k parameters per segment.
    let mbic = penalty.is_none();
    let pen = penalty.unwrap_or_else(|| {
        let log_n = (n as f64).ln();
        match cost_function {
            CostFunction::MeanChange => 3.0 * log_n,
            CostFunction::VarianceChange => 3.0 * log_n,
            CostFunction::MeanAndVariance => 4.0 * log_n,
        }
    });

    // Precompute cumulative sums for efficient cost calculation
    let (cum_sum, cum_sum_sq) = checked_cumulative_sums(data)?;

    // PELT algorithm
    let changepoints = pelt(n, &cum_sum, &cum_sum_sq, pen, min_seg, cost_function, mbic)?;

    // Compute segment statistics
    let segments = compute_segment_stats(data, &changepoints);

    // Compute total cost
    let total_cost =
        compute_total_cost(n, &cum_sum, &cum_sum_sq, &changepoints, cost_function, mbic);

    let kind = match cost_function {
        CostFunction::MeanChange => "mean change",
        CostFunction::VarianceChange => "variance change",
        CostFunction::MeanAndVariance => "mean and variance change",
    };
    let method = format!(
        "PELT ({kind}; {} penalty)",
        if mbic { "MBIC" } else { "manual" }
    );

    Ok(ChangepointResult {
        n_changepoints: changepoints.len(),
        changepoints,
        segments,
        total_cost,
        penalty: pen,
        method,
    })
}

/// Compensated (Neumaier) running sum.
#[derive(Default)]
struct Neumaier {
    sum: f64,
    c: f64,
}

impl Neumaier {
    fn add(&mut self, v: f64) {
        let t = self.sum + v;
        if self.sum.abs() >= v.abs() {
            self.c += (self.sum - t) + v;
        } else {
            self.c += (v - t) + self.sum;
        }
        self.sum = t;
    }

    fn value(&self) -> f64 {
        self.sum + self.c
    }
}

/// changepoint's sumstat on data centred by the global mean (`cpt.var`'s mu): prefix sums of z and
/// z^2, each a running compensated value. Centring makes every cost shift-invariant, so a level
/// of 1e7 no longer cancels.
fn compute_cumulative_sums(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    let mut total = Neumaier::default();
    for &v in data {
        total.add(v);
    }
    let mu = total.value() / n as f64;

    let mut cum_sum = vec![0.0; n + 1];
    let mut cum_sum_sq = vec![0.0; n + 1];
    let mut s1 = Neumaier::default();
    let mut s2 = Neumaier::default();
    for i in 0..n {
        let z = data[i] - mu;
        s1.add(z);
        s2.add(z * z);
        cum_sum[i + 1] = s1.value();
        cum_sum_sq[i + 1] = s2.value();
    }

    (cum_sum, cum_sum_sq)
}

/// The prefix sums of a series, or an error when no cost could be computed from them: a value that
/// is not finite, or values so large that the sums overflow. The costs would be NaN, no segment
/// would compare as best, and the search would report no changepoint.
fn checked_cumulative_sums(data: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
    if let Some((i, v)) = data.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(format!(
            "changepoint detection needs finite data: {v} at position {i} (R: NA is not allowed in the data)"
        ));
    }
    let (cum_sum, cum_sum_sq) = compute_cumulative_sums(data);
    // a^2 reaches m q (Cauchy-Schwarz), so a finite n Q keeps every intermediate of every cost
    // finite. A sum or square that overflowed is Inf or NaN here too.
    if !(cum_sum_sq[data.len()] * data.len() as f64).is_finite() {
        return Err("values too large: the sum of squares overflows".to_string());
    }
    Ok((cum_sum, cum_sum_sq))
}

/// The cost of a segment [start, end) on the centred prefix sums: changepoint's `mll_var`,
/// `mll_meanvar` and `mll_mean`, without the ln(segment length) term of their MBIC variants.
///
/// `mll_meanvar` floors a variance of zero at 1e-11. After centring by a mean that is not
/// representable, a segment of tied values has a variance of rounding noise, not zero, so the floor
/// applies at the rounding level of the prefix sums, tau = 4 eps Q, with Q the sum of squares of the
/// whole series. A variance at or under tau cannot be told from zero here, and takes tau, or 1e-11
/// when tau is smaller. Noise that escaped the floor would make the cost of the segment depend on
/// the rounding, and would break the sub-additivity that `pelt`'s pruning relies on. The floor
/// itself can break it too, for segments whose variance sits near tau: then, under a small manual
/// penalty, `pelt` can stop above the optimum.
///
/// One value far from the rest raises Q, and tau with it, above the variance of ordinary segments.
/// A floor below tau would reward each of them for that, so they take tau, the variance nearest to
/// theirs that the sums can still tell from zero. Limit: from an outlier near 1e9 times the scale of
/// the series, centring by the mean it contaminates loses the other regimes to rounding, so a
/// change among them can go unreported where R, on raw sums, finds it.
#[inline]
fn segment_cost(
    cum_sum: &[f64],
    cum_sum_sq: &[f64],
    start: usize,
    end: usize,
    cost_function: CostFunction,
) -> f64 {
    let m = (end - start) as f64;
    if m <= 0.0 {
        return f64::INFINITY;
    }

    let a = cum_sum[end] - cum_sum[start];
    let q = cum_sum_sq[end] - cum_sum_sq[start];
    let ln_2pi = (2.0 * std::f64::consts::PI).ln();

    match cost_function {
        CostFunction::MeanChange => q - a * a / m,
        CostFunction::VarianceChange => {
            let x3 = if q <= 0.0 { 1e-11 } else { q };
            m * (ln_2pi + (x3 / m).ln() + 1.0)
        }
        CostFunction::MeanAndVariance => {
            let mut sigma2 = (q - a * a / m) / m;
            let tau = 4.0 * f64::EPSILON * cum_sum_sq[cum_sum_sq.len() - 1];
            if sigma2 <= tau {
                sigma2 = tau.max(1e-11);
            }
            m * (ln_2pi + sigma2.ln() + 1.0)
        }
    }
}

/// PELT (Pruned Exact Linear Time) algorithm for multiple changepoint detection.
///
/// Under MBIC the objective adds ln(segment length) per segment, looked up in a table. Pruning
/// uses the cost without that term: the base costs are sub-additive and ln(T - s) >= ln(T - t), so
/// F[s] + C(s, t) > F[t] rules s out for good, while the test with the term (changepoint's) is
/// not a valid PELT bound: two halves of a segment of length L add 2 ln(L / 2) > ln(L).
///
/// A candidate that fails the test at t stays eligible until t + min_seg: the argument for
/// dropping it compares with the segmentation that ends at t, which a later T cannot extend
/// before T >= t + min_seg.
///
/// The bound needs sub-additive costs. A `MeanAndVariance` segment at the variance floor of
/// `segment_cost` can violate that, and the result can then sit above the optimum under a small
/// manual penalty.
///
/// When pruning removes no candidate (as with a large penalty), the work is
/// quadratic in the length of the series, so the budget installed by the
/// caller is checked at every step.
fn pelt(
    n: usize,
    cum_sum: &[f64],
    cum_sum_sq: &[f64],
    penalty: f64,
    min_seg: usize,
    cost_function: CostFunction,
    mbic: bool,
) -> Result<Vec<usize>, String> {
    let ln_len: Vec<f64> = if mbic {
        (0..=n)
            .map(|m| if m == 0 { 0.0 } else { (m as f64).ln() })
            .collect()
    } else {
        Vec::new()
    };

    // F[t] = minimum cost of segmenting data[0..t]
    let mut f = vec![f64::INFINITY; n + 1];
    f[0] = -penalty; // So that the first segment doesn't get penalized twice

    // cp[t] = last changepoint before t in the optimal segmentation
    let mut cp: Vec<Option<usize>> = vec![None; n + 1];

    // Candidate changepoints: position, base cost at this t, and the t from which a pruned
    // candidate is dropped (usize::MAX while it has passed the test).
    let mut cands: Vec<(usize, f64, usize)> = vec![(0, 0.0, usize::MAX)];

    for t in min_seg..=n {
        // One step costs O(|R|), so a check every step is negligible.
        crate::resource::checkpoint().map_err(|e| e.to_string())?;

        // Find the best last changepoint for position t
        let mut best_cost = f64::INFINITY;
        let mut best_cp = None;

        for (s, base, _) in cands.iter_mut() {
            if t - *s >= min_seg {
                *base = segment_cost(cum_sum, cum_sum_sq, *s, t, cost_function);
                let ln_term = if mbic { ln_len[t - *s] } else { 0.0 };
                let cost = f[*s] + *base + ln_term + penalty;
                if cost < best_cost {
                    best_cost = cost;
                    best_cp = Some(*s);
                }
            }
        }

        f[t] = best_cost;
        cp[t] = best_cp;

        // Pruning step: Keep s while F[s] + C(y_{s+1:t}) <= F[t]
        cands.retain_mut(|(s, base, drop_at)| {
            if *s + min_seg <= t && *drop_at == usize::MAX && f[*s] + *base > best_cost {
                *drop_at = t + min_seg;
            }
            t + 1 < *drop_at
        });
        cands.push((t, 0.0, usize::MAX));
    }

    // Backtrack to find changepoints
    let mut changepoints = Vec::new();
    let mut t = n;

    while let Some(s) = cp[t] {
        if s > 0 {
            changepoints.push(s);
        }
        t = s;
    }

    changepoints.reverse();
    Ok(changepoints)
}

/// Compute statistics for each segment
fn compute_segment_stats(data: &[f64], changepoints: &[usize]) -> Vec<SegmentStats> {
    let n = data.len();
    let mut segments = Vec::new();

    // Create segment boundaries
    let mut boundaries: Vec<usize> = vec![0];
    boundaries.extend(changepoints.iter().copied());
    boundaries.push(n);

    for i in 0..boundaries.len() - 1 {
        let start = boundaries[i];
        let end = boundaries[i + 1];
        let segment_data = &data[start..end];

        let n_points = segment_data.len();
        let mean = segment_data.iter().sum::<f64>() / n_points as f64;
        let variance =
            segment_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_points as f64;

        segments.push(SegmentStats {
            start,
            end,
            n_points,
            mean,
            variance,
        });
    }

    segments
}

/// Compute total cost of the segmentation
fn compute_total_cost(
    n: usize,
    cum_sum: &[f64],
    cum_sum_sq: &[f64],
    changepoints: &[usize],
    cost_function: CostFunction,
    mbic: bool,
) -> f64 {
    let mut boundaries: Vec<usize> = vec![0];
    boundaries.extend(changepoints.iter().copied());
    boundaries.push(n);

    let mut total = 0.0;
    for i in 0..boundaries.len() - 1 {
        let (start, end) = (boundaries[i], boundaries[i + 1]);
        total += segment_cost(cum_sum, cum_sum_sq, start, end, cost_function);
        if mbic {
            total += ((end - start) as f64).ln();
        }
    }

    total
}

/// Binary segmentation algorithm (simpler alternative to PELT)
/// Recursively finds the single best changepoint in each segment
///
/// # Errors
/// Fails on a series of fewer than 2 points, on a series too short for two
/// segments of the minimum segment length, on a NaN threshold, on a non-finite
/// value, on values so large that the sum of squares overflows, and when the
/// budget installed with [`crate::resource::with_budget`] is cancelled or past
/// its deadline.
pub fn binary_segmentation(
    data: &[f64],
    max_changepoints: Option<usize>,
    min_segment_length: Option<usize>,
    significance_threshold: Option<f64>,
) -> Result<ChangepointResult, String> {
    let n = data.len();

    if n < 2 {
        return Err("Data must have at least 2 points".to_string());
    }

    let min_seg = min_segment_length.unwrap_or(2).max(2);

    // As in detect_changepoints: a series that cannot hold two segments is
    // refused, not reported as having no changepoint. Saturating, so that a
    // minimum near usize::MAX is refused instead of wrapping.
    if n < min_seg.saturating_mul(2) {
        return Err(format!(
            "Data length ({}) must be at least 2x minimum segment length ({})",
            n, min_seg
        ));
    }

    let max_cp = max_changepoints.unwrap_or(10);
    let threshold = significance_threshold.unwrap_or(0.0);
    // No gain compares as above NaN: the search would report no changepoint.
    if threshold.is_nan() {
        return Err(format!(
            "significance_threshold must be a number, got {threshold}"
        ));
    }

    let (cum_sum, cum_sum_sq) = checked_cumulative_sums(data)?;

    let mut changepoints = Vec::new();
    let mut segments_to_check: Vec<(usize, usize)> = vec![(0, n)];

    while !segments_to_check.is_empty() && changepoints.len() < max_cp {
        // Each pass scans the whole series once.
        crate::resource::checkpoint().map_err(|e| e.to_string())?;

        // Find segment with best potential changepoint
        let mut best_gain = threshold;
        let mut best_cp = None;
        let mut best_segment_idx = 0;

        for (idx, &(start, end)) in segments_to_check.iter().enumerate() {
            if end - start < min_seg.saturating_mul(2) {
                continue;
            }

            let base_cost =
                segment_cost(&cum_sum, &cum_sum_sq, start, end, CostFunction::MeanChange);

            for cp in (start + min_seg)..(end - min_seg + 1) {
                let left_cost =
                    segment_cost(&cum_sum, &cum_sum_sq, start, cp, CostFunction::MeanChange);
                let right_cost =
                    segment_cost(&cum_sum, &cum_sum_sq, cp, end, CostFunction::MeanChange);
                let gain = base_cost - left_cost - right_cost;

                if gain > best_gain {
                    best_gain = gain;
                    best_cp = Some(cp);
                    best_segment_idx = idx;
                }
            }
        }

        if let Some(cp) = best_cp {
            let (start, end) = segments_to_check.remove(best_segment_idx);
            changepoints.push(cp);
            segments_to_check.push((start, cp));
            segments_to_check.push((cp, end));
        } else {
            break;
        }
    }

    changepoints.sort();

    let segments = compute_segment_stats(data, &changepoints);
    let total_cost = compute_total_cost(
        n,
        &cum_sum,
        &cum_sum_sq,
        &changepoints,
        CostFunction::MeanChange,
        false,
    );

    // Use BIC as penalty estimate
    let penalty = (n as f64).ln();

    Ok(ChangepointResult {
        n_changepoints: changepoints.len(),
        changepoints,
        segments,
        total_cost,
        penalty,
        method: "Binary Segmentation".to_string(),
    })
}

/// Run changepoint detection on a dataset column.
///
/// This is a convenience wrapper that extracts the column data from a Dataset
/// and runs the changepoint detection algorithm.
///
/// # Errors
/// Fails when `column` is missing, not numeric (integer, float or Boolean), or has a
/// missing value (`EconError::NullValues` inside the anyhow error), on fewer than
/// 4 observations, and with every error of [`detect_changepoints`], including
/// a cancelled or expired budget installed with
/// [`crate::resource::with_budget`].
pub fn run_changepoint(
    dataset: &Dataset,
    column: &str,
    penalty: Option<f64>,
    min_segment_length: Option<usize>,
    cost_function: CostFunction,
) -> Result<ChangepointResult> {
    // Extract time series data (an EconError inside the anyhow error)
    let (values, _) = numeric_column(dataset.df(), column, NaPolicy::Fail)?;

    if values.len() < 4 {
        return Err(anyhow!(
            "Time series must have at least 4 observations, found {}",
            values.len()
        ));
    }

    detect_changepoints(&values, penalty, min_segment_length, cost_function).map_err(|e| anyhow!(e))
}

/// Run binary segmentation changepoint detection on a dataset column.
///
/// # Errors
/// Fails when `column` is missing, not numeric (integer, float or Boolean), or has a
/// missing value (`EconError::NullValues` inside the anyhow error), on fewer than
/// 4 observations, and with every error of [`binary_segmentation`], including
/// a cancelled or expired budget installed with
/// [`crate::resource::with_budget`].
pub fn run_binary_segmentation(
    dataset: &Dataset,
    column: &str,
    max_changepoints: Option<usize>,
    min_segment_length: Option<usize>,
    significance_threshold: Option<f64>,
) -> Result<ChangepointResult> {
    // Extract time series data (an EconError inside the anyhow error)
    let (values, _) = numeric_column(dataset.df(), column, NaPolicy::Fail)?;

    if values.len() < 4 {
        return Err(anyhow!(
            "Time series must have at least 4 observations, found {}",
            values.len()
        ));
    }

    binary_segmentation(
        &values,
        max_changepoints,
        min_segment_length,
        significance_threshold,
    )
    .map_err(|e| anyhow!(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    use p2a_validation::{Check, Tol};

    fn col(ds: &crate::data::Dataset, name: &str) -> Vec<f64> {
        crate::data::columns::numeric_column(ds.df(), name, crate::data::columns::NaPolicy::Omit)
            .unwrap()
            .0
    }

    fn cps(r: &ChangepointResult) -> Vec<f64> {
        r.changepoints.iter().map(|&c| c as f64).collect()
    }

    /// Case ts_filter_ar_cpt, lev: N(0, 1) x 80, N(0, 3) x 70, N(0, 1) x 60, + 1e7, 6 decimals.
    /// changepoint::cpt.var(lev, method = "PELT", penalty = "MBIC"): 84, 151; pen 3 ln 210; the
    /// sum of the mbic var.norm costs = logLik(cv)[1] + sum(log(seg.len)). cpt.var on the
    /// centred series gives the same changepoints.
    #[test]
    fn test_validate_cpt_var_large_level() {
        let (case, ds) = crate::test_support::load_case("ts_filter_ar_cpt");
        let lev = col(&ds, "lev");
        let r = detect_changepoints(&lev, None, None, CostFunction::VarianceChange).unwrap();
        let mean = lev.iter().sum::<f64>() / lev.len() as f64;
        let centred: Vec<f64> = lev.iter().map(|v| v - mean).collect();
        let r0 = detect_changepoints(&centred, None, None, CostFunction::VarianceChange).unwrap();
        Check::new(&case)
            .vector("cpt_var_cpts", &cps(&r), Tol::exact())
            .vector("cpt_var_centred_cpts", &cps(&r0), Tol::exact())
            .scalar("cpt_var_pen", r.penalty, Tol::stat(210))
            .scalar("cpt_var_cost", r.total_cost, Tol::stat(210))
            .finish();
        assert!(r.method.contains("MBIC"), "{}", r.method);
    }

    /// cpt.var(lev, penalty = "Manual", pen.value = 10): 84, 151, cost logLik(cv)[1] (no ln m
    /// terms).
    #[test]
    fn test_validate_cpt_var_manual_penalty() {
        let (case, ds) = crate::test_support::load_case("ts_filter_ar_cpt");
        let lev = col(&ds, "lev");
        let r = detect_changepoints(&lev, Some(10.0), None, CostFunction::VarianceChange).unwrap();
        Check::new(&case)
            .vector("cpt_var_manual10_cpts", &cps(&r), Tol::exact())
            .scalar("cpt_var_manual10_cost", r.total_cost, Tol::stat(210))
            .finish();
        assert!(r.method.contains("manual"), "{}", r.method);
    }

    /// cpt.meanvar(lev - mean(lev), MBIC): 2, 84, 152, pen 4 ln 210. (R on the raw series gives
    /// 54 changepoints: changepoint's own raw-sum cancellation, key cpt_meanvar_cpts, recorded
    /// only; p2a centres, so the raw and centred series agree.)
    #[test]
    fn test_validate_cpt_meanvar_large_level() {
        let (case, ds) = crate::test_support::load_case("ts_filter_ar_cpt");
        let lev = col(&ds, "lev");
        let r = detect_changepoints(&lev, None, None, CostFunction::MeanAndVariance).unwrap();
        Check::new(&case)
            .vector("cpt_meanvar_centred_cpts", &cps(&r), Tol::exact())
            .scalar("cpt_meanvar_pen", r.penalty, Tol::stat(210))
            .finish();
    }

    /// cpt.mean(lev - mean(lev), MBIC, minseglen = 2): 107, 121, 135, 148, 150 (unit-variance
    /// cost, as R; p2a's minimum segment is 2, R's cpt.mean default is 1).
    #[test]
    fn test_validate_cpt_mean_large_level() {
        let (case, ds) = crate::test_support::load_case("ts_filter_ar_cpt");
        let lev = col(&ds, "lev");
        let r = detect_changepoints(&lev, None, None, CostFunction::MeanChange).unwrap();
        Check::new(&case)
            .vector("cpt_mean_centred_cpts", &cps(&r), Tol::exact())
            .scalar("cpt_mean_pen", r.penalty, Tol::stat(210))
            .finish();
    }

    /// R checkData: "Missing value: NA is not allowed in the data"; a manual penalty must be a
    /// number >= 0.
    #[test]
    fn test_changepoint_rejects_bad_inputs() {
        let data: Vec<f64> = (0..40)
            .map(|i| ((i * 37) % 11) as f64 * 0.3 + if i < 20 { 0.0 } else { 4.0 })
            .collect();
        let mut with_nan = data.clone();
        with_nan[7] = f64::NAN;
        let e = detect_changepoints(&with_nan, None, None, CostFunction::MeanChange).unwrap_err();
        assert!(e.contains("finite") && e.contains("position 7"), "{e}");
        for pen in [-1.0, f64::NAN, f64::INFINITY] {
            let e =
                detect_changepoints(&data, Some(pen), None, CostFunction::MeanChange).unwrap_err();
            assert!(e.contains("penalty"), "{e}");
        }
        let e = detect_changepoints(&data, None, Some(usize::MAX), CostFunction::MeanChange)
            .unwrap_err();
        assert!(e.contains("minimum segment length"), "{e}");
    }

    /// `gain > NaN` is never true, so a NaN threshold would report no changepoint.
    #[test]
    fn test_binary_segmentation_refuses_a_nan_threshold() {
        let data = create_mean_shift_data(30, 30, 0.0, 4.0, 1.0);
        let e = binary_segmentation(&data, Some(3), None, Some(f64::NAN)).unwrap_err();
        assert!(
            e.contains("significance_threshold") && e.contains("NaN"),
            "{e}"
        );
        let r = binary_segmentation(&data, Some(3), None, Some(1.0)).unwrap();
        assert!(r.n_changepoints >= 1);
    }

    /// Optimal partitioning: the exact minimiser that PELT's pruning must not change. O(n^2).
    /// Ties go to the smallest last changepoint, as in `pelt`.
    fn optimal_partitioning(
        data: &[f64],
        penalty: f64,
        min_seg: usize,
        cost: CostFunction,
        mbic: bool,
    ) -> (Vec<usize>, f64) {
        let n = data.len();
        let (s1, s2) = compute_cumulative_sums(data);
        let mut f = vec![f64::INFINITY; n + 1];
        f[0] = -penalty;
        let mut last = vec![0usize; n + 1];
        for t in min_seg..=n {
            for s in 0..=(t - min_seg) {
                if !f[s].is_finite() {
                    continue;
                }
                let m = (t - s) as f64;
                let c = segment_cost(&s1, &s2, s, t, cost) + if mbic { m.ln() } else { 0.0 };
                let v = f[s] + c + penalty;
                if v < f[t] {
                    f[t] = v;
                    last[t] = s;
                }
            }
        }
        let mut cps = Vec::new();
        let mut t = n;
        while t > 0 {
            let s = last[t];
            if s > 0 {
                cps.push(s);
            }
            t = s;
        }
        cps.reverse();
        (cps, f[n])
    }

    /// On seeded noisy series with a mean shift, a variance shift and both, PELT returns the
    /// optimum for each cost, under MBIC (pruned on the base cost: the ln(segment length) term
    /// breaks the PELT condition) and manual penalties, for minimum segments of 2 and 7. No segment
    /// of these series reaches the variance floor, so the costs stay sub-additive.
    #[test]
    fn test_pelt_matches_optimal_partitioning() {
        let mut seed: u64 = 20261107;
        let mut noise = |mean: f64, sd: f64| mean + sd * box_muller_cp(&mut seed);
        let n = 200;
        let mean_shift: Vec<f64> = (0..n)
            .map(|i| noise(if (70..130).contains(&i) { 2.0 } else { 0.0 }, 1.0))
            .collect();
        let var_shift: Vec<f64> = (0..n)
            .map(|i| noise(1.0, if i < 110 { 1.0 } else { 3.0 }))
            .collect();
        let both: Vec<f64> = (0..n)
            .map(|i| {
                if i < 90 {
                    noise(0.0, 0.5)
                } else {
                    noise(3.0, 2.0)
                }
            })
            .collect();
        let mut compared = 0;
        for series in [&mean_shift, &var_shift, &both] {
            for cost in [
                CostFunction::MeanChange,
                CostFunction::VarianceChange,
                CostFunction::MeanAndVariance,
            ] {
                let k = if cost == CostFunction::MeanAndVariance {
                    4.0
                } else {
                    3.0
                };
                for min_seg in [2usize, 7] {
                    for manual in [None, Some(1.5), Some(8.0)] {
                        let r = detect_changepoints(series, manual, Some(min_seg), cost).unwrap();
                        let pen = manual.unwrap_or(k * (n as f64).ln());
                        let (cps, f_n) =
                            optimal_partitioning(series, pen, min_seg, cost, manual.is_none());
                        assert_eq!(r.changepoints, cps, "{cost:?} {min_seg} {manual:?}");
                        let penalised = r.total_cost + pen * r.n_changepoints as f64;
                        assert!(
                            (penalised - f_n).abs() <= 1e-9 * f_n.abs().max(1.0),
                            "{cost:?} {min_seg} {manual:?}: {penalised} vs {f_n}"
                        );
                        compared += 1;
                    }
                }
            }
        }
        assert_eq!(compared, 54);
    }

    /// Seeded Poisson counts (Knuth's product of uniforms): whole numbers, with many tied values
    /// at a low rate. The rate changes at the middle of the series.
    fn poisson_counts(n: usize, rate: f64, seed: u64) -> Vec<f64> {
        let mut seed = seed;
        (0..n)
            .map(|i| {
                let lambda = if i < n / 2 { rate } else { 2.0 * rate };
                let limit = (-lambda).exp();
                let mut count = 0.0;
                let mut product = lcg_rand_cp(&mut seed);
                while product > limit {
                    count += 1.0;
                    product *= lcg_rand_cp(&mut seed);
                }
                count
            })
            .collect()
    }

    /// On whole numbers a tied segment has no variance, so it takes the 1e-11 floor. Centring by
    /// a mean that is not representable leaves rounding noise in q - a^2 / m, and noise that
    /// escapes the floor breaks sub-additivity, which is what makes pruning exact. Several
    /// segmentations of tied values can reach one objective, so the objective is what is compared.
    #[test]
    fn test_pelt_matches_optimal_partitioning_on_tied_counts() {
        let n = 200;
        let cost = CostFunction::MeanAndVariance;
        let pen_mbic = 4.0 * (n as f64).ln();
        let mut compared = 0;
        for rate in [0.5, 1.0, 3.0, 8.0] {
            for seed in 1..=15u64 {
                let series = poisson_counts(n, rate, seed * 7919);
                for manual in [None, Some(1.0), Some(8.0)] {
                    let r = detect_changepoints(&series, manual, Some(2), cost).unwrap();
                    let pen = manual.unwrap_or(pen_mbic);
                    let (_, f_n) = optimal_partitioning(&series, pen, 2, cost, manual.is_none());
                    let label = format!("rate {rate} seed {seed} {manual:?}");
                    let penalised = r.total_cost + pen * r.n_changepoints as f64;
                    assert!(
                        (penalised - f_n).abs() <= 1e-9 * f_n.abs().max(1.0),
                        "{label}: {penalised} vs {f_n}"
                    );
                    compared += 1;
                }
            }
        }
        assert_eq!(compared, 180);
    }

    /// The prefix sums of the raw series, which on small whole numbers are exact: the path that
    /// centring must agree with.
    fn raw_prefix_sums(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut s1 = vec![0.0; data.len() + 1];
        let mut s2 = vec![0.0; data.len() + 1];
        for (i, v) in data.iter().enumerate() {
            s1[i + 1] = s1[i] + v;
            s2[i + 1] = s2[i] + v * v;
        }
        (s1, s2)
    }

    /// Centring must not change any cost of a series of small whole numbers, tied segments
    /// included, nor the segmentation found from those costs.
    #[test]
    fn test_centred_and_raw_costs_agree_on_counts() {
        let n = 200;
        for (rate, seed) in [(0.5, 11u64), (1.0, 23), (3.0, 37), (4.0, 41)] {
            let series = poisson_counts(n, rate, seed);
            let (c1, c2) = compute_cumulative_sums(&series);
            let (r1, r2) = raw_prefix_sums(&series);
            // Both paths floor a tied segment at 1e-11 only while their rounding level is below it.
            assert!(4.0 * f64::EPSILON * r2[n] < 1e-11, "rate {rate}: {}", r2[n]);
            for cost in [CostFunction::MeanChange, CostFunction::MeanAndVariance] {
                for start in 0..n - 1 {
                    for end in start + 2..=n {
                        let centred = segment_cost(&c1, &c2, start, end, cost);
                        let raw = segment_cost(&r1, &r2, start, end, cost);
                        assert!(
                            (centred - raw).abs() <= 1e-9 * raw.abs().max(1.0),
                            "{cost:?} rate {rate} [{start}, {end}): {centred} vs {raw}"
                        );
                    }
                }
                let pen = 4.0 * (n as f64).ln();
                // Tied counts have several optima, so the two paths may stop at different ones of
                // equal objective: each segmentation is scored on the same (raw) sums.
                let objective = |cps: &[usize]| {
                    compute_total_cost(n, &r1, &r2, cps, cost, true) + pen * cps.len() as f64
                };
                let centred = objective(&pelt(n, &c1, &c2, pen, 2, cost, true).unwrap());
                let raw = objective(&pelt(n, &r1, &r2, pen, 2, cost, true).unwrap());
                assert!(
                    (centred - raw).abs() <= 1e-9 * raw.abs().max(1.0),
                    "{cost:?} rate {rate}: {centred} vs {raw}"
                );
            }
        }
    }

    /// N(0, 1) x 300, then N(0, 20) x 300.
    fn two_variance_regimes(seed: u64) -> Vec<f64> {
        let mut seed = seed;
        (0..600)
            .map(|i| (if i < 300 { 1.0 } else { 20.0 }) * box_muller_cp(&mut seed))
            .collect()
    }

    /// The rounding level follows the sum of squares of the whole series, so one large value
    /// raises it far above the variance of an ordinary segment. Such a segment takes the level
    /// itself, not a floor below it: a short segment under the level is not rewarded for it.
    #[test]
    fn test_meanvar_one_large_value_adds_only_its_own_boundary() {
        let cost = CostFunction::MeanAndVariance;
        for seed in [4u64, 5, 6] {
            let clean = two_variance_regimes(seed * 7919);
            let r = detect_changepoints(&clean, None, None, cost).unwrap();
            assert_eq!(r.n_changepoints, 1, "seed {seed}: {:?}", r.changepoints);
            assert!(r.changepoints[0].abs_diff(300) <= 10, "seed {seed}");

            for outlier in [1e6, 1e7] {
                let mut series = clean.clone();
                series[599] = outlier;
                let r = detect_changepoints(&series, None, None, cost).unwrap();
                let label = format!("seed {seed} outlier {outlier}: {:?}", r.changepoints);
                assert_eq!(r.n_changepoints, 2, "{label}");
                assert!(r.changepoints[0].abs_diff(300) <= 10, "{label}");
                assert_eq!(r.changepoints[1], 598, "{label}");
            }
        }
    }

    /// With the rounding level raised by one large value, ordinary segments of whole numbers sit
    /// at or near it, and under the MBIC default the pruned search still reaches the reference's
    /// objective. This covers the MBIC default only: under a small manual penalty the floor can
    /// break sub-additivity there, and the search can stop above the optimum.
    #[test]
    fn test_pelt_matches_optimal_partitioning_with_one_large_value() {
        let n = 200;
        let cost = CostFunction::MeanAndVariance;
        let pen = 4.0 * (n as f64).ln();
        for rate in [0.5, 3.0, 8.0] {
            for seed in 1..=5u64 {
                for outlier in [1e6, 1e7, 1e8] {
                    let mut series = poisson_counts(n, rate, seed * 104729);
                    series[n - 1] = outlier;
                    let r = detect_changepoints(&series, None, Some(2), cost).unwrap();
                    let (_, f_n) = optimal_partitioning(&series, pen, 2, cost, true);
                    let penalised = r.total_cost + pen * r.n_changepoints as f64;
                    assert!(
                        (penalised - f_n).abs() <= 1e-9 * f_n.abs().max(1.0),
                        "rate {rate} seed {seed} outlier {outlier}: {penalised} vs {f_n}"
                    );
                }
            }
        }
    }

    /// Values whose squares or sum overflow would make every cost NaN, and PELT then finds no
    /// best segment and reports no changepoint.
    #[test]
    fn test_overflowing_values_are_refused() {
        let noisy = |scale: f64| -> Vec<f64> {
            (0..40)
                .map(|i| scale * (1.0 + ((i * 37) % 11) as f64 * 0.03))
                .collect()
        };
        for scale in [1e200, 1e308, 1e155] {
            let data = noisy(scale);
            for cost in [
                CostFunction::MeanChange,
                CostFunction::VarianceChange,
                CostFunction::MeanAndVariance,
            ] {
                let e = detect_changepoints(&data, None, None, cost).unwrap_err();
                assert!(e.contains("too large"), "{scale} {cost:?}: {e}");
            }
            let e = binary_segmentation(&data, Some(3), None, None).unwrap_err();
            assert!(e.contains("too large"), "{scale}: {e}");
        }
        // Large values are fine as long as the sums hold them.
        let mut data = noisy(1e140);
        for v in &mut data[20..] {
            *v *= 3.0;
        }
        let r = detect_changepoints(&data, Some(1.0), None, CostFunction::MeanAndVariance).unwrap();
        assert!(r.total_cost.is_finite());
    }

    /// `detect_changepoints` refuses a value that is not finite; a direct call of
    /// `binary_segmentation` got NaN in every prefix sum and answered "no changepoints".
    #[test]
    fn test_binary_segmentation_refuses_non_finite_values() {
        let clean = create_mean_shift_data(20, 20, 0.0, 5.0, 1.0);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut data = clean.clone();
            data[7] = bad;
            let e = binary_segmentation(&data, Some(3), None, None).unwrap_err();
            assert!(e.contains("finite") && e.contains("position 7"), "{e}");
        }
    }

    /// X-2: an Int64 column gives the changepoints of its Float64 copy; missing values,
    /// text and absent columns are refused by name (they were read as 0).
    #[test]
    fn test_na_changepoint_int64_and_missing() {
        use crate::forecasting::test_fixtures::*;
        let ds = with_float_copy(&arima_series(), "ai");
        let pelt = |col: &str| run_changepoint(&ds, col, None, None, CostFunction::MeanChange);
        let binseg = |col: &str| run_binary_segmentation(&ds, col, Some(5), None, None);
        for run in [&pelt as &dyn Fn(&str) -> Result<ChangepointResult>, &binseg] {
            let int = run("ai").unwrap();
            let flt = run("ai_f64").unwrap();
            assert_eq!(int.changepoints, flt.changepoints);
            assert_eq!(int.total_cost.to_bits(), flt.total_cost.to_bits());
            assert_null_values(econ(&run("a_na").unwrap_err()), "a_na", 2);
            assert_null_values(econ(&run("ai_na").unwrap_err()), "ai_na", 2);
            assert_non_numeric(econ(&run("lab").unwrap_err()), "lab");
            assert_not_found(econ(&run("nope").unwrap_err()), "nope");
        }
    }

    #[test]
    fn test_no_changepoint() {
        // Constant data - no changepoints expected
        let data: Vec<f64> = vec![1.0; 50];
        let result =
            detect_changepoints(&data, Some(10.0), None, CostFunction::MeanChange).unwrap();
        assert_eq!(result.n_changepoints, 0);
        assert_eq!(result.segments.len(), 1);
    }

    #[test]
    fn test_single_changepoint() {
        // Clear mean shift at position 25
        let mut data: Vec<f64> = vec![0.0; 25];
        data.extend(vec![10.0; 25]);

        let result = detect_changepoints(&data, Some(5.0), None, CostFunction::MeanChange).unwrap();

        // Should detect changepoint around position 25
        assert!(
            result.n_changepoints >= 1,
            "Should detect at least one changepoint"
        );
        if result.n_changepoints > 0 {
            let cp = result.changepoints[0];
            assert!(
                (cp as i32 - 25).abs() <= 2,
                "Changepoint should be near position 25, got {}",
                cp
            );
        }
    }

    #[test]
    fn test_multiple_changepoints() {
        // Three segments with different means
        let mut data: Vec<f64> = vec![0.0; 30];
        data.extend(vec![5.0; 30]);
        data.extend(vec![0.0; 30]);

        let result =
            detect_changepoints(&data, Some(10.0), None, CostFunction::MeanChange).unwrap();

        // Should detect 2 changepoints
        assert_eq!(result.n_changepoints, 2, "Should detect 2 changepoints");
        assert_eq!(result.segments.len(), 3, "Should have 3 segments");
    }

    #[test]
    fn test_binary_segmentation() {
        let mut data: Vec<f64> = vec![0.0; 25];
        data.extend(vec![10.0; 25]);

        let result = binary_segmentation(&data, Some(5), None, Some(10.0)).unwrap();

        assert!(result.n_changepoints >= 1);
    }

    #[test]
    fn test_segment_stats() {
        let mut data: Vec<f64> = vec![1.0; 20];
        data.extend(vec![5.0; 20]);

        let changepoints = vec![20];
        let segments = compute_segment_stats(&data, &changepoints);

        assert_eq!(segments.len(), 2);
        assert!((segments[0].mean - 1.0).abs() < 0.01);
        assert!((segments[1].mean - 5.0).abs() < 0.01);
    }

    // ========================================================================
    // Property tests on simulated data
    // ========================================================================

    /// LCG for deterministic random numbers
    fn lcg_rand_cp(seed: &mut u64) -> f64 {
        let a: u64 = 1103515245;
        let c: u64 = 12345;
        let m: u64 = 2_u64.pow(31);
        *seed = (a.wrapping_mul(*seed).wrapping_add(c)) % m;
        (*seed as f64) / (m as f64)
    }

    fn box_muller_cp(seed: &mut u64) -> f64 {
        let u1 = lcg_rand_cp(seed).max(1e-10);
        let u2 = lcg_rand_cp(seed);
        ((-2.0_f64 * u1.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn create_mean_shift_data(n1: usize, n2: usize, mean1: f64, mean2: f64, sd: f64) -> Vec<f64> {
        let mut seed: u64 = 42;
        let mut data = Vec::with_capacity(n1 + n2);

        for _ in 0..n1 {
            data.push(mean1 + sd * box_muller_cp(&mut seed));
        }
        for _ in 0..n2 {
            data.push(mean2 + sd * box_muller_cp(&mut seed));
        }
        data
    }

    #[test]
    fn test_pelt_single_changepoint_recovers_dgp() {
        // Single mean change at position 50
        let data = create_mean_shift_data(50, 50, 0.0, 5.0, 1.0);
        let result = detect_changepoints(&data, Some(5.0), None, CostFunction::MeanChange).unwrap();

        // Should detect 1 changepoint
        assert!(
            result.n_changepoints >= 1,
            "Should detect at least 1 changepoint, found {}",
            result.n_changepoints
        );

        // Changepoint should be near position 50 (within 15 positions)
        if result.n_changepoints > 0 {
            let cp = result.changepoints[0];
            let diff = (cp as i32 - 50).abs();
            assert!(
                diff <= 15,
                "Changepoint at {} should be near 50 (diff={})",
                cp,
                diff
            );
        }
    }

    #[test]
    fn test_pelt_two_changepoints_recovers_dgp() {
        // Three segments: mean 0, mean 5, mean 0
        let mut data = create_mean_shift_data(30, 40, 0.0, 5.0, 1.0);
        let third_segment = create_mean_shift_data(0, 30, 5.0, 0.0, 1.0);
        data.extend(&third_segment[..30]);

        let result =
            detect_changepoints(&data, Some(10.0), None, CostFunction::MeanChange).unwrap();

        // Should detect 2 changepoints
        assert!(
            result.n_changepoints >= 1,
            "Should detect at least 1 changepoint for three-segment data"
        );

        // Segments should exist
        assert!(
            result.segments.len() >= 2,
            "Should have at least 2 segments"
        );
    }

    #[test]
    fn test_pelt_segment_means_recover_dgp() {
        // Clear mean shift: 0 -> 10
        let data = create_mean_shift_data(50, 50, 0.0, 10.0, 0.5);
        let result = detect_changepoints(&data, Some(5.0), None, CostFunction::MeanChange).unwrap();

        // Verify segment statistics
        assert_eq!(result.segments.len(), 2);
        // First segment should have mean near 0
        assert!(
            result.segments[0].mean.abs() < 2.0,
            "First segment mean {} should be near 0",
            result.segments[0].mean
        );

        // Second segment should have mean near 10
        assert!(
            (result.segments[1].mean - 10.0).abs() < 2.0,
            "Second segment mean {} should be near 10",
            result.segments[1].mean
        );
    }

    #[test]
    fn test_pelt_variance_change_finds_the_change() {
        // Variance change: low variance -> high variance
        let mut seed: u64 = 42;
        let mut data = Vec::with_capacity(100);

        // First 50: low variance
        for _ in 0..50 {
            data.push(box_muller_cp(&mut seed) * 0.5);
        }
        // Last 50: high variance
        for _ in 0..50 {
            data.push(box_muller_cp(&mut seed) * 3.0);
        }

        let result =
            detect_changepoints(&data, Some(5.0), None, CostFunction::VarianceChange).unwrap();

        // The variance changes at 50
        assert!(
            result.changepoints.iter().any(|&c| (45..=55).contains(&c)),
            "no changepoint near 50: {:?}",
            result.changepoints
        );
        assert!(result.total_cost.is_finite());
    }

    #[test]
    fn test_binary_segmentation_and_pelt_agree_on_dgp() {
        // Both methods should find similar changepoints
        let data = create_mean_shift_data(50, 50, 0.0, 5.0, 1.0);

        let pelt_result =
            detect_changepoints(&data, Some(5.0), None, CostFunction::MeanChange).unwrap();
        let bs_result = binary_segmentation(&data, Some(3), None, Some(10.0)).unwrap();

        // Both should detect at least one changepoint
        assert!(
            pelt_result.n_changepoints >= 1,
            "PELT should find changepoint"
        );
        assert!(
            bs_result.n_changepoints >= 1,
            "Binary segmentation should find changepoint"
        );

        // Changepoints should be similar (within 10 positions)
        if pelt_result.n_changepoints > 0 && bs_result.n_changepoints > 0 {
            let pelt_cp = pelt_result.changepoints[0] as i32;
            let bs_cp = bs_result.changepoints[0] as i32;
            let diff = (pelt_cp - bs_cp).abs();
            assert!(
                diff <= 15,
                "PELT ({}) and BS ({}) should find similar changepoints",
                pelt_cp,
                bs_cp
            );
        }
    }

    #[test]
    fn test_pelt_min_segment_length() {
        let data = create_mean_shift_data(50, 50, 0.0, 5.0, 1.0);

        // With large minimum segment length, may find fewer changepoints
        let result_short =
            detect_changepoints(&data, Some(5.0), Some(5), CostFunction::MeanChange).unwrap();
        let result_long =
            detect_changepoints(&data, Some(5.0), Some(30), CostFunction::MeanChange).unwrap();

        // Both should complete without error
        assert!(!result_short.segments.is_empty());
        assert!(!result_long.segments.is_empty());
    }

    #[test]
    fn test_pelt_penalty_effect() {
        let data = create_mean_shift_data(50, 50, 0.0, 5.0, 1.0);

        // Low penalty: more changepoints
        let result_low =
            detect_changepoints(&data, Some(1.0), None, CostFunction::MeanChange).unwrap();

        // High penalty: fewer changepoints
        let result_high =
            detect_changepoints(&data, Some(100.0), None, CostFunction::MeanChange).unwrap();

        // High penalty should find <= changepoints than low penalty
        assert!(
            result_high.n_changepoints <= result_low.n_changepoints + 1,
            "High penalty ({}) should find <= changepoints than low ({})",
            result_high.n_changepoints,
            result_low.n_changepoints
        );
    }

    #[test]
    fn test_pelt_constant_data_has_no_changepoint() {
        // Constant data should have no changepoints
        let data: Vec<f64> = vec![5.0; 100];
        let result =
            detect_changepoints(&data, Some(10.0), None, CostFunction::MeanChange).unwrap();

        assert_eq!(
            result.n_changepoints, 0,
            "Constant data should have 0 changepoints"
        );
        assert_eq!(result.segments.len(), 1, "Should have 1 segment");
        assert!((result.segments[0].mean - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_pelt_total_cost_with_manual_penalty() {
        let data = create_mean_shift_data(50, 50, 0.0, 5.0, 1.0);
        let result = detect_changepoints(&data, Some(5.0), None, CostFunction::MeanChange).unwrap();

        // Total cost should be positive and finite
        assert!(result.total_cost.is_finite());
        assert!(result.total_cost >= 0.0);

        // Penalty should match what was passed
        assert!((result.penalty - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_pelt_segment_boundaries() {
        let data = create_mean_shift_data(50, 50, 0.0, 5.0, 1.0);
        let result = detect_changepoints(&data, Some(5.0), None, CostFunction::MeanChange).unwrap();

        // Verify segment boundaries are consistent
        let mut expected_start = 0;
        for segment in &result.segments {
            assert_eq!(segment.start, expected_start, "Segment start mismatch");
            assert!(segment.end > segment.start, "Segment end should be > start");
            assert_eq!(segment.n_points, segment.end - segment.start);
            expected_start = segment.end;
        }

        // Last segment should end at data length
        assert_eq!(
            result.segments.last().unwrap().end,
            data.len(),
            "Last segment should end at data length"
        );
    }

    #[test]
    fn test_pelt_mean_and_variance_runs_with_mbic() {
        let data = create_mean_shift_data(50, 50, 0.0, 5.0, 1.0);
        let result = detect_changepoints(&data, None, None, CostFunction::MeanAndVariance).unwrap();

        // Should run successfully
        assert!(result.method.contains("mean and variance"));
        assert!(result.total_cost.is_finite());
    }

    fn cancelled<R>(f: impl FnOnce() -> R) -> R {
        let token = crate::resource::CancelToken::new();
        token.cancel();
        crate::resource::with_budget(crate::resource::Budget::new(None, token), f)
    }

    #[test]
    fn pelt_stops_when_the_budget_is_cancelled() {
        // Before the fix pelt() never looked at the budget, so a call past its
        // deadline ran to the end of its quadratic scan and returned Ok.
        let data = create_mean_shift_data(2_000, 2_000, 0.0, 0.0, 1.0);
        for cost in [
            CostFunction::MeanChange,
            CostFunction::VarianceChange,
            CostFunction::MeanAndVariance,
        ] {
            assert!(detect_changepoints(&data, None, None, cost).is_ok());
            let err = cancelled(|| detect_changepoints(&data, None, None, cost))
                .expect_err("a cancelled budget must stop PELT");
            assert!(err.contains("cancelled"), "{cost:?}: {err}");
        }
    }

    #[test]
    fn binary_segmentation_stops_when_the_budget_is_cancelled() {
        let data = create_mean_shift_data(2_000, 2_000, 0.0, 0.0, 1.0);
        assert!(binary_segmentation(&data, Some(5), None, None).is_ok());
        let err = cancelled(|| binary_segmentation(&data, Some(5), None, None))
            .expect_err("a cancelled budget must stop binary segmentation");
        assert!(err.contains("cancelled"), "{err}");
    }

    #[test]
    fn a_huge_min_segment_length_is_refused_not_overflowed() {
        // `min_seg * 2` overflowed: a debug build panicked, and a release
        // build wrapped (to 0 for 2^63, so PELT ran and reported no change).
        let data = create_mean_shift_data(50, 50, 0.0, 5.0, 1.0);
        for min_seg in [usize::MAX, usize::MAX / 2 + 1] {
            let err =
                detect_changepoints(&data, Some(5.0), Some(min_seg), CostFunction::MeanChange)
                    .expect_err("a minimum segment longer than the series must be refused");
            assert!(err.contains("minimum segment length"), "{err}");
            let err = binary_segmentation(&data, Some(5), Some(min_seg), None)
                .expect_err("a minimum segment longer than the series must be refused");
            assert!(err.contains("minimum segment length"), "{err}");
        }
    }

    #[test]
    fn binary_segmentation_refuses_a_series_shorter_than_two_minimum_segments() {
        // Binary segmentation skipped every segment shorter than two minimum
        // segments, so it answered "no changepoints" for a series it never
        // examined, where PELT refuses the same input.
        let data = create_mean_shift_data(50, 50, 0.0, 5.0, 1.0);
        let err = binary_segmentation(&data, Some(5), Some(51), None)
            .expect_err("100 points cannot hold two segments of 51");
        assert!(
            err.contains("Data length (100) must be at least 2x minimum segment length (51)"),
            "{err}"
        );
        let pelt_err = detect_changepoints(&data, Some(5.0), Some(51), CostFunction::MeanChange)
            .expect_err("PELT refuses it too");
        assert_eq!(err, pelt_err);
        // Two segments of 50 fit exactly; the shift sits at their boundary.
        let r = binary_segmentation(&data, Some(5), Some(50), None).unwrap();
        assert_eq!(r.changepoints, vec![50]);
    }
}
