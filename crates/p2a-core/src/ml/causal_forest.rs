//! Causal forests for heterogeneous treatment effects, as `grf::causal_forest` (grf 2.x).
//!
//! 1. **Local centering.** `Y.hat = E[Y | X]` and `W.hat = P(W = 1 | X)` are out-of-bag
//!    predictions of two honest regression forests (`max(50, n_trees / 4)` trees each, min.node.size
//!    5, honesty fraction 0.5), unless supplied ([`causal_forest_arrays_with_nuisance`]).
//! 2. **Causal trees on the residuals** `Y - Y.hat`, `W - W.hat` (the R-learner moment), grown in
//!    groups of `ci_group_size` on shared half-samples, honest, with grf's gradient split:
//!    pseudo-outcomes `rho_i = (W~_i - mean W~)((Y~_i - mean Y~) - tau_node (W~_i - mean W~))`,
//!    criterion `sum_l^2 / n_l + sum_r^2 / n_r`, stabilised children (`min_node_size` units with
//!    W~ below and at or above the node mean; at least 5% of the node's variation in W~). Empty
//!    honest leaves are pruned.
//! 3. **Prediction** pools the leaves a unit falls into (grf's forest weights):
//!    `tau(x) = (avg(YW) - avg(Y) avg(W)) / (avg(WW) - avg(W)^2)` over leaf means of the
//!    residuals; training units are predicted out of bag. No per-leaf effect is formed, so a leaf
//!    without treated or control units cannot contribute tau = 0.
//! 4. **Variance** of each prediction by little bags with objective-Bayes debiasing (grf
//!    `variance.estimates`).
//! 5. **ATE** from AIPW scores `Gamma_i = tau_i + (W_i - W.hat_i) / (W.hat_i (1 - W.hat_i)) (Y_i -
//!    Y.hat_i - tau_i (W_i - W.hat_i))`: `mean(Gamma)` with SE `sd(Gamma) / sqrt(n)` (grf
//!    `average_treatment_effect`).
//!
//! R equivalent: `grf::causal_forest(X, Y, W)`, `predict(cf)`, `average_treatment_effect(cf)`.
//!
//! Differences from grf: split thresholds are midpoints between consecutive distinct values;
//! `honesty_fraction` is the estimation share (1 - grf's `honesty.fraction`); `max_depth` caps the
//! causal trees; the default `mtry` is `ceil(sqrt(p))` as the mean of grf's Poisson candidate
//! count (grf: `min(ceil(sqrt(p) + 20), p)`); no sample weights, clusters, tuning or target
//! samples other than "all"; W must be binary; undefined estimates (a unit without an out-of-bag
//! tree, a propensity of 0 or 1, no treatment variation around a unit) are errors instead of NaN.
//!
//! # References
//!
//! - Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous
//!   Treatment Effects using Random Forests. *Journal of the American Statistical
//!   Association*, 113(523), 1228-1242. <https://doi.org/10.1080/01621459.2017.1319839>
//!
//! - Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests.
//!   *Annals of Statistics*, 47(2), 1148-1178. <https://doi.org/10.1214/18-AOS1709>
//!
//! - Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects.
//!   *Biometrika*, 108(2), 299-319. <https://doi.org/10.1093/biomet/asaa076>
//!
//! - Sexton, J., & Laake, P. (2009). Standard errors for bagged and random forest estimators.
//!   *Computational Statistics & Data Analysis*, 53(3), 801-811.
//!
//! - Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with limited
//!   overlap in estimation of average treatment effects. *Biometrika*, 96(1), 187-199.
//!
//! - R package `grf`: Tibshirani, J., Athey, S., & Wager, S. (2024).
//!   *grf: Generalized Random Forests*. <https://grf-labs.github.io/grf/>

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::fmt;

use super::{default_mtry, lcg_random, mul, partition_exact, pow2};
use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::resource::{check_alloc, checked_len, checkpoint, try_vec, try_with_capacity};
use crate::stats::robust::{QuantileType, quantile_sorted};
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::{
    SignificanceLevel, normal_p_value_two_sided, validate_level, z_critical,
};
use statrs::function::erf::erfc;

/// Configuration for causal forest estimation (`grf::causal_forest` arguments).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalForestConfig {
    /// Trees in the causal forest (default 2000), grown in groups of `ci_group_size` on shared
    /// half-samples; rounded down to a multiple of `ci_group_size`.
    pub n_trees: usize,
    /// grf `min.node.size` (default 5, at least 1): a node with at most this many growing units
    /// is not split, and a causal split keeps this many units with `W - W.hat` below and this
    /// many at or above the node's mean on each side.
    pub min_node_size: usize,
    /// The mean of the Poisson number of candidate covariates per node (clamped to [1, p]), as
    /// grf's `mtry`; at most p. Default `ceil(sqrt(p))` (grf's default is
    /// `min(ceil(sqrt(p) + 20), p)`). The fitted result stores the value used.
    pub mtry: Option<usize>,
    /// Honest trees (default true): one part of each tree's subsample grows it, the other
    /// estimates its leaves.
    pub honesty: bool,
    /// Share of each tree's subsample that estimates the leaves (default 0.5; grf's
    /// `honesty.fraction` is the share that grows the tree, 1 minus this).
    pub honesty_fraction: f64,
    /// Depth cap of the causal trees (default 10; grf has none; the nuisance forests are not
    /// capped).
    pub max_depth: usize,
    /// Seed of the one random stream (nuisance forests, then causal trees); None = 42.
    pub seed: Option<u64>,
    /// Compute the split-frequency variable importance (default true).
    pub compute_importance: bool,
    /// Subsample per tree as a share of n (default 0.5; at most 0.5 when `ci_group_size > 1`).
    pub sample_fraction: f64,
    /// grf `ci.group.size` (default 2): trees per half-sample group for the little-bags variance;
    /// 1 = no variance estimates.
    #[serde(default = "default_ci_group_size")]
    pub ci_group_size: usize,
    /// Trees in each nuisance forest (Y.hat, W.hat); None = `max(50, n_trees / 4)`, as grf.
    #[serde(default)]
    pub nuisance_trees: Option<usize>,
}

fn default_ci_group_size() -> usize {
    2
}

impl Default for CausalForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 2000,
            min_node_size: 5,
            mtry: None,
            honesty: true,
            honesty_fraction: 0.5,
            max_depth: 10,
            seed: None,
            compute_importance: true,
            sample_fraction: 0.5,
            ci_group_size: 2,
            nuisance_trees: None,
        }
    }
}

impl CausalForestConfig {
    /// An upper bound, in bytes, of what [`causal_forest()`] holds at once for a dataset of
    /// `n_obs` rows and `n_covariates` covariates under this configuration. The allocator's own
    /// overhead and the dataset itself come on top. Saturates at `u64::MAX`; monotone in
    /// `n_trees`, `nuisance_trees`, `max_depth` and the rows, and falling in `min_node_size`.
    ///
    /// The bound holds whatever the data are, because each term counts the most the code can
    /// allocate:
    ///
    /// - **Causal trees**, kept in the result: `n_trees` rounded down to a multiple of
    ///   `ci_group_size` (at least one group), each with its in-bag bitset of `ceil(n / 64)` words. A tree has at most
    ///   `2 L - 1` nodes for `L` leaves; `L` is at most `2^max_depth`, and at most the growing
    ///   part of its subsample divided by `2 min_node_size` (each child of a causal split keeps
    ///   `min_node_size` units on each side of the node's mean of `W - W.hat`).
    /// - **Nuisance trees**: the two regression forests are grown one tree at a time and each
    ///   tree is dropped after its out-of-bag predictions are added, so one tree is counted,
    ///   with its leaves bounded by the growing part of its subsample (no depth cap; a split may
    ///   leave a single unit in a child), and the out-of-bag sums, counts and flags.
    /// - **Data and results.** The outcome, treatment and covariates [`causal_forest()`]
    ///   extracts, and the column it is extracting; `Y.hat`, `W.hat`, the residuals, the AIPW
    ///   scores, the predictions, the variances and the sorted copy the result's summary makes;
    ///   the leaf list of one unit's out-of-bag prediction; the importances.
    /// - **One tree's working copies**: the permutation the subsamples are drawn from, the
    ///   group's half-sample, the subsample and its two parts, the growth stack (at most twice
    ///   the growing part), the relabelled responses and the sort order, and the node vector
    ///   while it grows (up to twice its length), with the repopulation sums.
    pub fn memory_bound(&self, n_obs: usize, n_covariates: usize) -> u64 {
        let n = n_obs as u128;
        let p = n_covariates as u128;
        let node = size_of::<Node>() as u128;
        // node vector with growth slack, its exact-length copy, repopulation sums and redirects,
        // and growth-stack entries (two per split, with slack)
        let node_work = 3 * node + (size_of::<(usize, f64, f64, f64, f64)>() as u128) + 8 + 2 * 40;
        let words = n_obs.div_ceil(64) as u128;
        // at least one group: fewer trees are refused, and the fewest the fit accepts must not
        // have a larger bound than a refused count
        let grown = self.grown_trees().max(self.group_size()) as u128;
        let causal_tree = mul(self.max_nodes_per_tree(n_obs), node)
            .saturating_add(size_of::<HonestTree>() as u128 + 24)
            .saturating_add(mul(8, words));
        let causal = mul(grown, causal_tree)
            .saturating_add(mul(grown, size_of::<(usize, LeafStats)>() as u128));
        // y, w, x and the column being extracted (2); Y.hat, W.hat, the two residuals, scores,
        // predictions, variances and the summary's sorted copy (8); importances (3 per covariate)
        let data = mul(mul(8, n), p.saturating_add(4 + 8)).saturating_add(mul(24, p));
        // nuisance out-of-bag sums, counts and flags
        let nuisance =
            mul(17, n).saturating_add(mul(self.max_nodes_per_nuisance_tree(n_obs), node_work));
        let (s_c, g_c) = self.causal_parts(n_obs);
        let (s_n, g_n) = self.nuisance_parts(n_obs);
        let (s, g) = (s_c.max(s_n) as u128, g_c.max(g_n) as u128);
        // permutation and half-sample; subsample, its copy and its two parts; growth stack and
        // children (2), responses, sort order; the drawn covariates
        let work = mul(8, n.saturating_add(n.div_ceil(2)))
            .saturating_add(mul(24, s))
            .saturating_add(mul(32, g))
            .saturating_add(mul(8, p))
            .saturating_add(mul(self.max_nodes_per_tree(n_obs), node_work));
        u64::try_from(
            causal
                .saturating_add(data)
                .saturating_add(nuisance)
                .saturating_add(work),
        )
        .unwrap_or(u64::MAX)
    }

    /// `ci_group_size`, at least 1 (as the bound reads an unvalidated configuration).
    fn group_size(&self) -> usize {
        self.ci_group_size.max(1)
    }

    /// The causal trees the fit grows: `n_trees` rounded down to a multiple of `ci_group_size`.
    fn grown_trees(&self) -> usize {
        (self.n_trees / self.group_size()) * self.group_size()
    }

    /// Trees in each nuisance forest: `nuisance_trees`, or `max(50, n_trees / 4)`.
    fn nuisance_tree_count(&self) -> usize {
        self.nuisance_trees.unwrap_or((self.n_trees / 4).max(50))
    }

    /// The sizes `(subsample, growing part)` of a causal tree for `n_obs` rows, as
    /// [`draw_subsample`] and [`honest_split`] draw them.
    fn causal_parts(&self, n_obs: usize) -> (usize, usize) {
        let s = subsample_size(n_obs, self.sample_fraction, self.group_size());
        let g = if self.honesty {
            growing_size(s, self.honesty_fraction)
        } else {
            s
        };
        (s, g)
    }

    /// The sizes `(subsample, growing part)` of a nuisance tree for `n_obs` rows.
    fn nuisance_parts(&self, n_obs: usize) -> (usize, usize) {
        let s = subsample_size(n_obs, self.sample_fraction, 1);
        (s, growing_size(s, NUISANCE_HONESTY_FRACTION))
    }

    /// The most nodes one causal tree can have for `n_obs` rows ([`Self::memory_bound`]).
    fn max_nodes_per_tree(&self, n_obs: usize) -> u128 {
        let (_, g) = self.causal_parts(n_obs);
        let per_leaf = self.min_node_size.max(1).saturating_mul(2) as u128;
        let leaves = (g as u128 / per_leaf).min(pow2(self.max_depth)).max(1);
        2 * leaves - 1
    }

    /// The most nodes one nuisance tree can have for `n_obs` rows ([`Self::memory_bound`]).
    fn max_nodes_per_nuisance_tree(&self, n_obs: usize) -> u128 {
        let (_, g) = self.nuisance_parts(n_obs);
        2 * (g as u128).max(1) - 1
    }
}

/// The subsample one tree draws from `n` units: `ceil(n * fraction)` of them (at least 1), or,
/// in groups of `group_size > 1`, `ceil(h * 2 * fraction)` of a half-sample of `h = ceil(n / 2)`.
fn subsample_size(n: usize, fraction: f64, group_size: usize) -> usize {
    let (pool, share) = if group_size > 1 {
        (n.div_ceil(2), 2.0 * fraction)
    } else {
        (n, fraction)
    };
    let k = ((pool as f64) * share).ceil();
    let k = if k.is_finite() && k > 0.0 {
        k as usize
    } else {
        1
    };
    k.max(1).min(pool)
}

/// The growing part of an honest subsample of `m` units: `ceil(m (1 - honesty_fraction))`,
/// clamped so that both parts are non-empty when `m >= 2`.
fn growing_size(m: usize, honesty_fraction: f64) -> usize {
    let g = ((m as f64) * (1.0 - honesty_fraction)).ceil();
    let g = if g.is_finite() && g > 0.0 {
        g as usize
    } else {
        0
    };
    if m >= 2 { g.clamp(1, m - 1) } else { m }
}

/// Result from causal forest estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalForestResult {
    /// Out-of-bag CATE estimates of the training units (grf `predict(cf)$predictions`).
    pub predictions: Vec<f64>,
    /// Little-bags variances of `predictions` (grf `variance.estimates`); NaN when
    /// `ci_group_size = 1` or a unit has no complete out-of-bag tree group.
    pub variance_estimates: Vec<f64>,
    /// AIPW average treatment effect: the mean of `aipw_scores` (grf
    /// `average_treatment_effect`).
    pub ate: f64,
    /// `sqrt(sum((Gamma - ate)^2) / (n (n - 1)))`.
    pub ate_se: f64,
    /// z = ate / ate_se (normal reference).
    pub ate_t_stat: f64,
    /// 2 Phi(-|z|).
    pub ate_p_value: f64,
    /// ate - qnorm(0.975) ate_se.
    pub ate_ci_lower: f64,
    /// ate + qnorm(0.975) ate_se.
    pub ate_ci_upper: f64,
    /// Significance level of the ATE.
    pub ate_significance: SignificanceLevel,
    /// Split-frequency importance of the causal trees (each split adds 2^-depth, normalised per
    /// tree, then over the forest), sorted in decreasing order.
    pub variable_importance: Vec<(String, f64)>,
    /// Trees grown in the causal forest (`n_trees` rounded down to a multiple of
    /// `ci_group_size`).
    pub n_trees: usize,
    /// Number of observations.
    pub n_obs: usize,
    /// Mean out-of-bag R-loss (Nie & Wager 2021; grf's debiased.error + excess.error).
    pub oob_error: f64,
    /// Covariate names.
    pub covariate_names: Vec<String>,
    /// Configuration used (`mtry` is the value used).
    pub config: CausalForestConfig,
    /// Out-of-bag E[Y | X] (grf `Y.hat`), or the supplied values.
    #[serde(default)]
    pub y_hat: Vec<f64>,
    /// Out-of-bag P(W = 1 | X) (grf `W.hat`), or the supplied values.
    #[serde(default)]
    pub w_hat: Vec<f64>,
    /// Doubly robust scores Gamma_i (grf `get_scores`).
    #[serde(default)]
    pub aipw_scores: Vec<f64>,
    /// Trees of the Y.hat forest (0: supplied).
    #[serde(default)]
    pub y_hat_trees: usize,
    /// Trees of the W.hat forest (0: supplied).
    #[serde(default)]
    pub w_hat_trees: usize,
    /// Overlap, tree-count and variance notes.
    #[serde(default)]
    pub warnings: Vec<String>,
    /// The causal trees, for prediction (not serialized).
    #[serde(skip)]
    trees: Vec<HonestTree>,
}

impl ResultNotes for CausalForestResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl fmt::Display for CausalForestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Causal Forest Results (grf: local centering, honest trees, AIPW average treatment \
             effect)"
        )?;
        writeln!(f, "===========================================")?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(
            f,
            "No. Trees: {} (groups of {})",
            self.n_trees, self.config.ci_group_size
        )?;
        for (name, trees) in [("Y.hat", self.y_hat_trees), ("W.hat", self.w_hat_trees)] {
            if trees == 0 {
                writeln!(f, "{name}: supplied")?;
            } else {
                writeln!(f, "{name}: out-of-bag regression forest ({trees} trees)")?;
            }
        }
        if !self.w_hat.is_empty() {
            let lo = self.w_hat.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = self.w_hat.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            writeln!(f, "W.hat range: [{lo:.4}, {hi:.4}]")?;
        }
        writeln!(f, "Out-of-bag R-loss: {:.4}", self.oob_error)?;
        writeln!(f)?;

        writeln!(f, "AVERAGE TREATMENT EFFECT (AIPW, all units):")?;
        writeln!(
            f,
            "  ATE = {:.4} (SE: {:.4}, z = {:.2}, p = {:.3}){}",
            self.ate,
            self.ate_se,
            self.ate_t_stat,
            self.ate_p_value,
            self.ate_significance.stars()
        )?;
        writeln!(
            f,
            "  95% CI: [{:.4}, {:.4}]",
            self.ate_ci_lower, self.ate_ci_upper
        )?;
        writeln!(f)?;

        writeln!(f, "CATE DISTRIBUTION (out of bag):")?;
        if !self.predictions.is_empty() {
            let mut sorted = self.predictions.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let q = |p: f64| quantile_sorted(&sorted, p, QuantileType::T7);
            writeln!(f, "  Min:    {:.4}", sorted[0])?;
            writeln!(f, "  10%:    {:.4}", q(0.1))?;
            writeln!(f, "  25%:    {:.4}", q(0.25))?;
            writeln!(f, "  Median: {:.4}", q(0.5))?;
            writeln!(f, "  75%:    {:.4}", q(0.75))?;
            writeln!(f, "  90%:    {:.4}", q(0.9))?;
            writeln!(f, "  Max:    {:.4}", sorted[sorted.len() - 1])?;
        }
        writeln!(f)?;

        writeln!(f, "VARIABLE IMPORTANCE (split frequency):")?;
        for (i, (name, importance)) in self.variable_importance.iter().enumerate() {
            if i >= 10 {
                writeln!(
                    f,
                    "  ... ({} more variables)",
                    self.variable_importance.len() - 10
                )?;
                break;
            }
            writeln!(f, "  {:<20} {:.4}", name, importance)?;
        }
        writeln!(f)?;

        writeln!(f, "Configuration:")?;
        if self.config.honesty {
            writeln!(
                f,
                "  Honest splitting: true (estimation share {})",
                self.config.honesty_fraction
            )?;
        } else {
            writeln!(f, "  Honest splitting: false")?;
        }
        writeln!(f, "  Min node size: {}", self.config.min_node_size)?;
        writeln!(f, "  Max depth (causal trees): {}", self.config.max_depth)?;
        match self.config.mtry {
            Some(m) => writeln!(f, "  mtry: {m}")?,
            None => writeln!(f, "  mtry: default")?,
        }
        write!(f, "{}", notes_text(self))
    }
}

/// Run causal forest estimation (`grf::causal_forest` on the dataset's columns).
///
/// Outcome, treatment and covariates are read with [`NaPolicy::Fail`]: a missing value is an
/// error naming its column, as forests refuse missing values. The treatment must be 0/1.
///
/// # Example
/// ```ignore
/// use p2a_core::ml::{causal_forest, CausalForestConfig};
///
/// let mut config = CausalForestConfig::default();
/// config.n_trees = 1000;
///
/// let result = causal_forest(
///     &dataset,
///     "outcome",
///     "treatment",
///     &["age", "income", "education"],
///     config,
/// )?;
///
/// println!("ATE: {:.4} (SE: {:.4})", result.ate, result.ate_se);
/// ```
///
/// # References
///
/// - Athey, Tibshirani & Wager (2019), Annals of Statistics; R `grf::causal_forest()`.
pub fn causal_forest(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: CausalForestConfig,
) -> EconResult<CausalForestResult> {
    if covariate_cols.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "At least one covariate is required".to_string(),
        });
    }
    let cols = ColumnRequest::new(dataset.df())
        .numeric(outcome_col)
        .numeric(treatment_col)
        .numerics(covariate_cols)
        .na(NaPolicy::Fail)
        .extract()?;
    let y = cols.numeric_array(outcome_col);
    let w = cols.numeric_array(treatment_col);
    if let Some(&val) = w.iter().find(|&&v| v != 0.0 && v != 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment variable '{}' must be binary (0/1), found value: {}",
                treatment_col, val
            ),
        });
    }
    let x = cols.matrix(covariate_cols, false)?.data;
    causal_forest_arrays(
        x.view(),
        y.view(),
        w.view(),
        covariate_cols.iter().map(|s| s.to_string()).collect(),
        config,
    )
}

/// Check the counts and fractions of a causal-forest configuration.
///
/// `n_trees >= 1`; `sample_fraction` finite and in (0, 1]; with honesty, `honesty_fraction` in
/// the open interval (0, 1), as `grf` requires; `ci_group_size >= 1`, and with
/// `ci_group_size > 1` `sample_fraction <= 0.5` and `n_trees >= ci_group_size`;
/// `min_node_size >= 1`; `nuisance_trees` and `mtry` at least 1 when given.
fn validate_causal_forest_config(config: &CausalForestConfig) -> EconResult<()> {
    let invalid = |message: String| Err(EconError::InvalidSpecification { message });
    if config.n_trees == 0 {
        return invalid("n_trees must be at least 1 (got 0)".to_string());
    }
    let sf = config.sample_fraction;
    if !(sf.is_finite() && sf > 0.0 && sf <= 1.0) {
        return invalid(format!("sample_fraction must be in (0, 1] (got {sf})"));
    }
    let hf = config.honesty_fraction;
    if config.honesty && !(hf.is_finite() && hf > 0.0 && hf < 1.0) {
        return invalid(format!(
            "honesty_fraction must be in (0, 1) when honesty is on (got {hf})"
        ));
    }
    let g = config.ci_group_size;
    if g == 0 {
        return invalid("ci_group_size must be at least 1 (got 0)".to_string());
    }
    if g > 1 && sf > 0.5 {
        return invalid(format!(
            "sample_fraction must be at most 0.5 when ci_group_size > 1 (got {sf}); grf: \"When \
             confidence intervals are enabled, the sampling fraction must be less than 0.5\". Set \
             ci_group_size = 1 to fit without variance estimates"
        ));
    }
    if config.n_trees < g {
        return invalid(format!(
            "n_trees must be at least ci_group_size ({g}) to grow one group of trees (got {})",
            config.n_trees
        ));
    }
    if config.min_node_size == 0 {
        return invalid("min_node_size must be at least 1 (got 0)".to_string());
    }
    if config.nuisance_trees == Some(0) {
        return invalid("nuisance_trees must be at least 1 (got 0)".to_string());
    }
    if config.mtry == Some(0) {
        return invalid("mtry must be at least 1 (got 0)".to_string());
    }
    Ok(())
}

/// `InvalidSpecification` naming the first non-finite value of `values`.
fn check_finite_1d(values: &ArrayView1<f64>, name: &str) -> EconResult<()> {
    match values.iter().position(|v| !v.is_finite()) {
        Some(i) => Err(EconError::InvalidSpecification {
            message: format!(
                "{name} must contain only finite values (row {i} is {})",
                values[i]
            ),
        }),
        None => Ok(()),
    }
}

/// AIPW (doubly robust) scores of a binary treatment, as grf's `get_scores.causal_forest`:
/// `Gamma_i = tau_i + (W_i - W.hat_i) / (W.hat_i (1 - W.hat_i)) * (Y_i - Y.hat_i - tau_i (W_i -
/// W.hat_i))`.
///
/// All five arrays must have the same length and finite values, `w` must be 0/1 and every
/// `w_hat` strictly between 0 and 1 (the scores divide by `W.hat (1 - W.hat)`).
pub fn aipw_scores(
    y: ArrayView1<f64>,
    w: ArrayView1<f64>,
    y_hat: ArrayView1<f64>,
    w_hat: ArrayView1<f64>,
    tau_hat: ArrayView1<f64>,
) -> EconResult<Vec<f64>> {
    let n = y.len();
    for (name, len) in [
        ("w", w.len()),
        ("y_hat", y_hat.len()),
        ("w_hat", w_hat.len()),
        ("tau_hat", tau_hat.len()),
    ] {
        if len != n {
            return Err(EconError::InvalidSpecification {
                message: format!("{name} has {len} values for {n} observations"),
            });
        }
    }
    check_finite_1d(&y, "y")?;
    check_finite_1d(&w, "w")?;
    check_finite_1d(&y_hat, "y_hat")?;
    check_finite_1d(&w_hat, "w_hat")?;
    check_finite_1d(&tau_hat, "tau_hat")?;
    if let Some(i) = w.iter().position(|&v| v != 0.0 && v != 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("w must be binary (0/1); row {i} is {}", w[i]),
        });
    }
    if let Some(i) = w_hat.iter().position(|&e| !(e > 0.0 && e < 1.0)) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "w_hat must be strictly between 0 and 1 for the AIPW scores (overlap); row {i} is {}",
                w_hat[i]
            ),
        });
    }
    Ok((0..n)
        .map(|i| {
            let (e, d) = (w_hat[i], w[i] - w_hat[i]);
            tau_hat[i] + d / (e * (1.0 - e)) * (y[i] - y_hat[i] - tau_hat[i] * d)
        })
        .collect())
}

/// grf `average_treatment_effect` (AIPW, `target.sample = "all"`, no clusters) from the scores:
/// their mean and `sqrt(sum((s - mean)^2) / (m (m - 1)))` (`.sigma2.hat` with its `m / (m - 1)`
/// correction). At least two scores, all finite.
pub fn aipw_ate(scores: &[f64]) -> EconResult<(f64, f64)> {
    let m = scores.len();
    if m < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: m,
            context: "an AIPW standard error needs at least 2 scores".to_string(),
        });
    }
    if let Some(i) = scores.iter().position(|s| !s.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!("AIPW score {i} is not finite ({})", scores[i]),
        });
    }
    let mf = m as f64;
    let mean = scores.iter().sum::<f64>() / mf;
    let ss: f64 = scores.iter().map(|s| (s - mean) * (s - mean)).sum();
    Ok((mean, (ss / (mf * (mf - 1.0))).sqrt()))
}

/// Core causal forest algorithm on arrays: [`causal_forest_arrays_with_nuisance`] with both
/// nuisance functions estimated by out-of-bag regression forests.
pub fn causal_forest_arrays(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    w: ArrayView1<f64>,
    covariate_names: Vec<String>,
    config: CausalForestConfig,
) -> EconResult<CausalForestResult> {
    causal_forest_arrays_with_nuisance(x, y, w, None, None, covariate_names, config)
}

/// The causal forest of `grf::causal_forest(X, Y, W, Y.hat = , W.hat = )`: either nuisance may be
/// supplied (finite, length n; `W.hat` strictly in (0, 1)), the other is estimated out of bag by
/// an honest regression forest (`nuisance_trees` trees, min.node.size 5, honesty fraction 0.5,
/// the forest's `sample_fraction` and `mtry`, no depth cap).
///
/// `w` must be 0/1 with at least 5 treated and 5 control units; `x` needs at least one column
/// and as many names; every value must be finite. A unit that is in the subsample of every tree
/// of a forest, an estimated propensity of 0 or 1, and a unit without treatment variation around
/// it are errors (grf returns NaN there).
pub fn causal_forest_arrays_with_nuisance(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    w: ArrayView1<f64>,
    y_hat: Option<ArrayView1<f64>>,
    w_hat: Option<ArrayView1<f64>>,
    covariate_names: Vec<String>,
    mut config: CausalForestConfig,
) -> EconResult<CausalForestResult> {
    validate_causal_forest_config(&config)?;
    let invalid = |message: String| EconError::InvalidSpecification { message };

    let n = x.nrows();
    let p = x.ncols();
    if n < 10 {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: n,
            context: "Causal forest requires at least 10 observations".to_string(),
        });
    }
    if y.len() != n || w.len() != n {
        return Err(invalid(
            "Outcome, treatment, and covariates must have same number of observations".to_string(),
        ));
    }
    if p == 0 {
        return Err(invalid(
            "x must have at least one covariate column".to_string(),
        ));
    }
    if covariate_names.len() != p {
        return Err(invalid(format!(
            "covariate_names has {} names for {p} columns of x",
            covariate_names.len()
        )));
    }
    if let Some(((i, j), v)) = x.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(invalid(format!(
            "x must contain only finite values (row {i}, column {j} is {v})"
        )));
    }
    check_finite_1d(&y, "y")?;
    check_finite_1d(&w, "w")?;
    if let Some(i) = w.iter().position(|&v| v != 0.0 && v != 1.0) {
        return Err(invalid(format!(
            "w must be binary (0/1); row {i} is {}",
            w[i]
        )));
    }
    let n_treated = w.iter().filter(|&&v| v == 1.0).count();
    let n_control = n - n_treated;
    if n_treated < 5 || n_control < 5 {
        return Err(EconError::InsufficientData {
            required: 5,
            provided: n_treated.min(n_control),
            context: "Need at least 5 treated and 5 control units".to_string(),
        });
    }
    let mtry = config.mtry.unwrap_or_else(|| default_mtry(p));
    if mtry > p {
        return Err(invalid(format!(
            "mtry must be at most the number of covariates ({p}), got {mtry}"
        )));
    }
    let g = config.ci_group_size;
    let grown = config.grown_trees();
    let nt = config.nuisance_tree_count();
    check_alloc::<HonestTree>(grown, "causal forest trees")?;
    check_alloc::<HonestTree>(nt, "nuisance forest trees")?;
    check_alloc::<f64>(
        checked_len(&[n, 8], "causal forest vectors")?,
        "causal forest vectors",
    )?;

    // One random stream: the Y.hat forest, the W.hat forest, then the causal trees.
    let mut rng = config.seed.unwrap_or(42);
    let nuisance_spec = TreeSpec {
        min_node_size: NUISANCE_MIN_NODE_SIZE,
        max_depth: usize::MAX,
        mtry,
        honesty: true,
        honesty_fraction: NUISANCE_HONESTY_FRACTION,
    };
    let y_vec = y.to_vec();
    let w_vec = w.to_vec();
    let fit_nuisance =
        |name: &str, supplied: Option<ArrayView1<f64>>, target: &[f64], rng: &mut u64| {
            nuisance_estimate(
                x,
                name,
                supplied,
                target,
                nt,
                config.sample_fraction,
                &nuisance_spec,
                rng,
            )
        };
    let (y_hat, y_hat_trees) = fit_nuisance("Y.hat", y_hat, &y_vec, &mut rng)?;
    let w_supplied = w_hat.is_some();
    let (w_hat, w_hat_trees) = fit_nuisance("W.hat", w_hat, &w_vec, &mut rng)?;
    if w_supplied {
        if let Some(i) = w_hat.iter().position(|&e| !(e > 0.0 && e < 1.0)) {
            return Err(invalid(format!(
                "w_hat must be strictly between 0 and 1 (overlap); row {i} is {}",
                w_hat[i]
            )));
        }
    } else {
        let outside: Vec<usize> = (0..n)
            .filter(|&i| !(w_hat[i] > 0.0 && w_hat[i] < 1.0))
            .collect();
        if let Some(&i) = outside.first() {
            return Err(invalid(format!(
                "the estimated propensity W.hat is 0 or 1 for {} observation(s) (first: row {i}); \
                 the AIPW average treatment effect needs 0 < W.hat < 1 (overlap). Drop the units \
                 without overlap or coarsen the covariates",
                outside.len()
            )));
        }
    }

    let mut warnings = Vec::new();
    let lo = w_hat.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = w_hat.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if lo <= 0.05 && hi >= 0.95 {
        warnings.push(format!(
            "Estimated treatment propensities take values between {lo:.3} and {hi:.3} and in \
             particular get very close to 0 and 1; the AIPW estimate may be unstable. Filtering \
             the data as in Crump, Hotz, Imbens and Mitnik (Biometrika, 2009) may help"
        ));
    } else if lo <= 0.05 {
        warnings.push(format!(
            "Estimated treatment propensities go as low as {lo:.3}, so treatment effects for \
             some controls may not be well identified"
        ));
    } else if hi >= 0.95 {
        warnings.push(format!(
            "Estimated treatment propensities go as high as {hi:.3}, so treatment effects for \
             some treated units may not be well identified"
        ));
    }
    if grown != config.n_trees {
        warnings.push(format!(
            "n_trees = {} was rounded down to {grown}, a multiple of ci_group_size = {g}",
            config.n_trees
        ));
    }

    // The R-learner residuals and the causal trees grown on them.
    let yt: Vec<f64> = (0..n).map(|i| y[i] - y_hat[i]).collect();
    let zt: Vec<f64> = (0..n).map(|i| w[i] - w_hat[i]).collect();
    let causal_spec = TreeSpec {
        min_node_size: config.min_node_size,
        max_depth: config.max_depth,
        mtry,
        honesty: config.honesty,
        honesty_fraction: config.honesty_fraction,
    };
    let data = ForestData {
        x: x.view(),
        y: &yt,
        z: Some(&zt),
    };
    let forest = fit_honest_forest(
        &data,
        config.n_trees,
        g,
        config.sample_fraction,
        &causal_spec,
        &mut rng,
        "causal forest trees",
    )?;

    // Out-of-bag CATEs, their little-bags variances and the R-loss.
    let mut leaves = try_with_capacity::<(usize, LeafStats)>(grown, "causal forest leaves")?;
    let mut predictions = try_with_capacity::<f64>(n, "causal forest predictions")?;
    let mut variance_estimates = try_with_capacity::<f64>(n, "causal forest variances")?;
    let mut rloss = 0.0;
    let mut no_group = 0usize;
    for i in 0..n {
        checkpoint()?;
        leaves.clear();
        for (t, tree) in forest.trees.iter().enumerate() {
            if in_bag(&forest.drawn[t], i) {
                continue;
            }
            let l = tree.leaf(x.row(i));
            if l.n > 0 {
                leaves.push((t, *l));
            }
        }
        if leaves.is_empty() {
            return Err(invalid(format!(
                "observation {i} is in the subsample of every causal tree; use more trees \
                 (n_trees = {grown})"
            )));
        }
        let pooled = pooled_causal(&leaves, g);
        if !pooled.tau.is_finite() {
            return Err(invalid(format!(
                "the CATE of observation {i} is undefined: W - W.hat does not vary among the \
                 units sharing its leaves"
            )));
        }
        if g >= 2 && pooled.variance.is_nan() {
            no_group += 1;
        }
        let r = yt[i] - pooled.mean_y - pooled.tau * (zt[i] - pooled.mean_z);
        rloss += r * r;
        predictions.push(pooled.tau);
        variance_estimates.push(pooled.variance);
    }
    drop(leaves);
    let HonestForest { trees, drawn } = forest;
    drop(drawn);
    let oob_error = rloss / n as f64;
    if g == 1 {
        warnings.push(
            "variance_estimates are NaN: CATE variance estimates need ci_group_size >= 2 (little \
             bags)"
                .to_string(),
        );
    } else if no_group > 0 {
        warnings.push(format!(
            "{no_group} observation(s) have no complete out-of-bag tree group; their variance \
             estimates are NaN"
        ));
    }

    // The AIPW average treatment effect.
    let aipw = aipw_scores(
        y,
        w,
        ArrayView1::from(&y_hat),
        ArrayView1::from(&w_hat),
        ArrayView1::from(&predictions),
    )?;
    let (ate, ate_se) = aipw_ate(&aipw)?;
    let ate_t_stat = ate / ate_se;
    let ate_p_value = normal_p_value_two_sided(ate_t_stat);
    let crit = z_critical(0.95)?;

    let mut importance = Array1::<f64>::zeros(p);
    if config.compute_importance {
        for tree in &trees {
            importance += &tree.split_importance(p);
        }
        let total = importance.sum();
        if total > 0.0 {
            importance /= total;
        }
    }
    let mut variable_importance: Vec<(String, f64)> = covariate_names
        .iter()
        .cloned()
        .zip(importance.iter().copied())
        .collect();
    variable_importance.sort_by(|a, b| b.1.total_cmp(&a.1));

    config.mtry = Some(mtry);
    Ok(CausalForestResult {
        predictions,
        variance_estimates,
        ate,
        ate_se,
        ate_t_stat,
        ate_p_value,
        ate_ci_lower: ate - crit * ate_se,
        ate_ci_upper: ate + crit * ate_se,
        ate_significance: SignificanceLevel::from_p_value(ate_p_value),
        variable_importance,
        n_trees: grown,
        n_obs: n,
        oob_error,
        covariate_names,
        config,
        y_hat,
        w_hat,
        aipw_scores: aipw,
        y_hat_trees,
        w_hat_trees,
        warnings,
        trees,
    })
}

/// A nuisance function (`name` "Y.hat" or "W.hat"): the supplied values (length n, finite), or
/// the out-of-bag predictions of an honest regression forest of `nt` trees on `target`, with
/// the number of trees grown (0 when supplied).
#[allow(clippy::too_many_arguments)]
fn nuisance_estimate(
    x: ArrayView2<f64>,
    name: &str,
    supplied: Option<ArrayView1<f64>>,
    target: &[f64],
    nt: usize,
    sample_fraction: f64,
    spec: &TreeSpec,
    rng: &mut u64,
) -> EconResult<(Vec<f64>, usize)> {
    let n = x.nrows();
    let field = if name == "Y.hat" { "y_hat" } else { "w_hat" };
    match supplied {
        Some(v) => {
            if v.len() != n {
                return Err(EconError::InvalidSpecification {
                    message: format!("{field} has {} values for {n} observations", v.len()),
                });
            }
            check_finite_1d(&v, field)?;
            Ok((v.to_vec(), 0))
        }
        None => {
            let data = ForestData {
                x: x.view(),
                y: target,
                z: None,
            };
            let est = oob_regression_forest(&data, nt, sample_fraction, spec, rng)?;
            if let Some(i) = est.iter().position(|v| v.is_nan()) {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "observation {i} is in the subsample of every one of the {nt} trees of \
                         the {name} forest, so it has no out-of-bag estimate; use more trees \
                         (nuisance_trees or n_trees)"
                    ),
                });
            }
            Ok((est, nt))
        }
    }
}

/// Predict CATE for new observations from the dataset's covariate columns (read with
/// [`NaPolicy::Fail`]), in the order the forest was fitted on.
pub fn causal_forest_predict(
    forest: &CausalForestResult,
    new_data: &Dataset,
    covariate_cols: &[&str],
) -> EconResult<Vec<f64>> {
    if covariate_cols.len() != forest.covariate_names.len() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Number of covariates ({}) doesn't match training ({})",
                covariate_cols.len(),
                forest.covariate_names.len()
            ),
        });
    }
    let cols = ColumnRequest::new(new_data.df())
        .numerics(covariate_cols)
        .na(NaPolicy::Fail)
        .extract()?;
    let x = cols.matrix(covariate_cols, false)?.data;
    causal_forest_predict_arrays(forest, x.view())
}

/// Predict CATE for the rows of `x` (as many columns as the forest's covariates, finite): grf's
/// forest-weighted estimate, pooling the leaves of every causal tree.
pub fn causal_forest_predict_arrays(
    forest: &CausalForestResult,
    x: ArrayView2<f64>,
) -> EconResult<Vec<f64>> {
    if forest.trees.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "Forest has no trees (may have been deserialized without tree structures)"
                .to_string(),
        });
    }
    let p = forest.covariate_names.len();
    if x.ncols() != p {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "x has {} columns; the forest was fitted on {p} covariates",
                x.ncols()
            ),
        });
    }
    if let Some(((i, j), v)) = x.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!("x must contain only finite values (row {i}, column {j} is {v})"),
        });
    }
    let n = x.nrows();
    let mut leaves =
        try_with_capacity::<(usize, LeafStats)>(forest.trees.len(), "causal forest leaves")?;
    let mut predictions = try_with_capacity::<f64>(n, "causal forest predictions")?;
    for i in 0..n {
        checkpoint()?;
        leaves.clear();
        for (t, tree) in forest.trees.iter().enumerate() {
            let l = tree.leaf(x.row(i));
            if l.n > 0 {
                leaves.push((t, *l));
            }
        }
        // only tau is needed: no variance (group size 1)
        let tau = pooled_causal(&leaves, 1).tau;
        if !tau.is_finite() {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "the CATE of row {i} is undefined: W - W.hat does not vary among the units \
                     sharing its leaves"
                ),
            });
        }
        predictions.push(tau);
    }
    Ok(predictions)
}

/// An average treatment effect with its standard error and two-sided normal interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AteEstimate {
    /// The mean of the AIPW scores.
    pub estimate: f64,
    /// grf's standard error: `sqrt(sum((s - mean)^2) / (m (m - 1)))` over the `m` scores.
    pub std_err: f64,
    /// `estimate - z * std_err`, with `z` the normal quantile of `level`.
    pub ci_lower: f64,
    /// `estimate + z * std_err`.
    pub ci_upper: f64,
    /// Confidence level of the interval.
    pub level: f64,
    /// Units averaged.
    pub n: usize,
}

/// AIPW average treatment effect over `subset` (all units when None), grf
/// `average_treatment_effect(forest, target.sample = "all", method = "AIPW", subset = )`: the
/// mean of the stored AIPW scores with SE `sqrt(sum((s - mean)^2) / (m (m - 1)))` and a
/// two-sided normal interval at `level`.
///
/// Subset indices must be below `n_obs` and distinct, and at least two units are needed; a
/// result without stored scores (deserialized from a version before the AIPW estimator) is an
/// error.
pub fn average_treatment_effect_with(
    forest: &CausalForestResult,
    subset: Option<&[usize]>,
    level: f64,
) -> EconResult<AteEstimate> {
    validate_level(level, "level")?;
    let scores = &forest.aipw_scores;
    if scores.len() != forest.n_obs {
        return Err(EconError::InvalidSpecification {
            message: "the causal forest result has no AIPW scores (deserialized from a version \
                      before the AIPW estimator); refit it"
                .to_string(),
        });
    }
    let picked: Vec<f64> = match subset {
        None => scores.clone(),
        Some(idx) => {
            let mut seen = try_vec(forest.n_obs, false, "average treatment effect subset")?;
            let mut v = try_with_capacity::<f64>(idx.len(), "average treatment effect subset")?;
            for &i in idx {
                if i >= forest.n_obs {
                    return Err(EconError::InvalidSpecification {
                        message: format!(
                            "subset index {i} is out of range for {} observations",
                            forest.n_obs
                        ),
                    });
                }
                if std::mem::replace(&mut seen[i], true) {
                    return Err(EconError::InvalidSpecification {
                        message: format!("subset lists observation {i} more than once"),
                    });
                }
                v.push(scores[i]);
            }
            v
        }
    };
    let (estimate, std_err) = aipw_ate(&picked)?;
    let crit = z_critical(level)?;
    Ok(AteEstimate {
        estimate,
        std_err,
        ci_lower: estimate - crit * std_err,
        ci_upper: estimate + crit * std_err,
        level,
        n: picked.len(),
    })
}

/// AIPW average treatment effect over `subset` (all units when None): `(estimate, 95% lower,
/// 95% upper)` of [`average_treatment_effect_with`] at level 0.95.
pub fn average_treatment_effect(
    forest: &CausalForestResult,
    subset_indices: Option<&[usize]>,
) -> EconResult<(f64, f64, f64)> {
    let a = average_treatment_effect_with(forest, subset_indices, 0.95)?;
    Ok((a.estimate, a.ci_lower, a.ci_upper))
}

/// Run causal forest from dataset with string arguments (for MCP integration).
///
/// This is a convenience wrapper that handles string-to-slice conversion.
pub fn run_causal_forest(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: Vec<String>,
    n_trees: Option<usize>,
    min_node_size: Option<usize>,
    honesty: Option<bool>,
    max_depth: Option<usize>,
    seed: Option<u64>,
) -> EconResult<CausalForestResult> {
    let cov_refs: Vec<&str> = covariate_cols.iter().map(|s| s.as_str()).collect();
    let config = run_causal_forest_config(n_trees, min_node_size, honesty, max_depth, seed);
    causal_forest(dataset, outcome_col, treatment_col, &cov_refs, config)
}

/// The configuration [`run_causal_forest`] fits with for these arguments
/// (the defaults of [`CausalForestConfig`] where an argument is `None`), for
/// example to check its [`CausalForestConfig::memory_bound`] first.
pub fn run_causal_forest_config(
    n_trees: Option<usize>,
    min_node_size: Option<usize>,
    honesty: Option<bool>,
    max_depth: Option<usize>,
    seed: Option<u64>,
) -> CausalForestConfig {
    let defaults = CausalForestConfig::default();
    CausalForestConfig {
        n_trees: n_trees.unwrap_or(defaults.n_trees),
        min_node_size: min_node_size.unwrap_or(defaults.min_node_size),
        honesty: honesty.unwrap_or(defaults.honesty),
        max_depth: max_depth.unwrap_or(defaults.max_depth),
        seed,
        ..defaults
    }
}

// ---- grf-style honest forests (private) ----

/// grf `alpha` (default 0.05): minimum child share. Regression: of the node's count; causal: of the
/// node's sum of squared centred treatment residuals (InstrumentalSplittingRule `size`).
const GRF_ALPHA: f64 = 0.05;
/// grf `equal_doubles(denominator, 0, 1e-10)` in InstrumentalRelabelingStrategy::relabel, and the
/// same guard on a unit's pooled first stage. W - W.hat lies in [-1, 1], so the absolute threshold
/// is on a fixed scale.
const RELABEL_EPS: f64 = 1e-10;
/// causal_forest's `args.orthog`: min.node.size = 5 and honesty.fraction = 0.5 for the nuisance
/// forests.
const NUISANCE_MIN_NODE_SIZE: usize = 5;
const NUISANCE_HONESTY_FRACTION: f64 = 0.5;

/// Uniform in [0, 1) from the module LCG: 32 bits, identical on 32- and 64-bit targets.
fn unit_uniform(rng: &mut u64) -> f64 {
    ((lcg_random(rng) as u64) & 0xFFFF_FFFF) as f64 / 4_294_967_296.0
}

/// Fisher-Yates shuffle with the module LCG.
fn shuffle(v: &mut [usize], rng: &mut u64) {
    for i in (1..v.len()).rev() {
        let j = lcg_random(rng) % (i + 1);
        v.swap(i, j);
    }
}

/// grf create_split_variable_subset: Poisson(lambda) candidate variables, clamped to [1, cap].
/// Counts unit-rate exponential arrivals in [0, lambda], so it cannot underflow for large lambda.
fn poisson_count(lambda: f64, cap: usize, rng: &mut u64) -> usize {
    let (mut k, mut t) = (0usize, 0.0f64);
    while k < cap {
        t -= (1.0 - unit_uniform(rng)).ln();
        if t > lambda {
            break;
        }
        k += 1;
    }
    k.clamp(1, cap.max(1))
}

/// Means over a leaf's estimation units (grf's precomputed prediction values, unit weights): `y`
/// the response, `z` the centred treatment (= instrument), `yz`, `zz` the means of the products.
/// Regression leaves use `n` and `y`. `n == 0` is an empty leaf.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct LeafStats {
    n: usize,
    y: f64,
    z: f64,
    yz: f64,
    zz: f64,
}

/// A node of an [`HonestTree`]: children are later in the node vector than their parent.
#[derive(Debug, Clone)]
enum Node {
    Split {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
    Leaf(LeafStats),
}

/// One honest tree after repopulation and pruning; `root` may differ from 0 after pruning.
#[derive(Debug, Clone)]
struct HonestTree {
    nodes: Vec<Node>,
    root: usize,
}

impl HonestTree {
    /// The leaf `x` falls into (`x[feature] <= threshold` goes left).
    fn leaf(&self, x: ArrayView1<f64>) -> &LeafStats {
        let mut v = self.root;
        loop {
            match &self.nodes[v] {
                Node::Leaf(s) => return s,
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => {
                    v = if x[*feature] <= *threshold {
                        *left
                    } else {
                        *right
                    };
                }
            }
        }
    }

    /// Each split reachable from the root adds 2^-depth (root depth 0) to its feature; normalised
    /// to sum 1 (zeros without splits).
    fn split_importance(&self, p: usize) -> Array1<f64> {
        let mut imp = Array1::zeros(p);
        let mut stack = vec![(self.root, 0i32)];
        while let Some((v, d)) = stack.pop() {
            if let Node::Split {
                feature,
                left,
                right,
                ..
            } = &self.nodes[v]
            {
                imp[*feature] += 0.5f64.powi(d);
                stack.push((*left, d + 1));
                stack.push((*right, d + 1));
            }
        }
        let s = imp.sum();
        if s > 0.0 {
            imp /= s;
        }
        imp
    }
}

/// What a forest is grown on. `y`: Y (Y.hat forest), W (W.hat forest) or Y - Y.hat (causal
/// forest); `z`: W - W.hat for a causal forest (grf: treatment = instrument), None for a
/// regression forest.
#[derive(Clone, Copy)]
struct ForestData<'a> {
    x: ArrayView2<'a, f64>,
    y: &'a [f64],
    z: Option<&'a [f64]>,
}

/// How the trees of one forest are grown.
#[derive(Debug, Clone, Copy)]
struct TreeSpec {
    min_node_size: usize,
    max_depth: usize,
    mtry: usize,
    honesty: bool,
    honesty_fraction: f64,
}

/// The split [`best_split`] chooses.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SplitChoice {
    feature: usize,
    threshold: f64,
    decrease: f64,
}

/// grf InstrumentalRelabelingStrategy::relabel with instrument = treatment and unit weights:
/// returns the node's tau = sum (z - zbar)(y - ybar) / sum (z - zbar)^2 and writes
/// rho_i = (z_i - zbar)((y_i - ybar) - tau (z_i - zbar)) to `out` in `samples` order. None (the
/// node becomes a leaf) when sum (z - zbar)^2 < RELABEL_EPS.
fn relabel(samples: &[usize], y: &[f64], z: &[f64], out: &mut Vec<f64>) -> Option<f64> {
    let m = samples.len() as f64;
    let (mut sy, mut sz) = (0.0, 0.0);
    for &i in samples {
        sy += y[i];
        sz += z[i];
    }
    let (yb, zb) = (sy / m, sz / m);
    let (mut num, mut den) = (0.0, 0.0);
    for &i in samples {
        let dz = z[i] - zb;
        num += dz * (y[i] - yb);
        den += dz * dz;
    }
    if den.is_nan() || den.abs() < RELABEL_EPS {
        return None;
    }
    let tau = num / den;
    out.clear();
    out.extend(samples.iter().map(|&i| {
        let dz = z[i] - zb;
        dz * ((y[i] - yb) - tau * dz)
    }));
    Some(tau)
}

/// grf's split search on one node. `responses[k]` belongs to `samples[k]` (rho for a causal node,
/// the response for a regression node). Per feature (in the given order) the node's units are
/// sorted stably by x (`total_cmp`) and every midpoint between consecutive distinct values is a
/// candidate, scanned in ascending order (left: x <= threshold).
/// decrease = sum_l^2 / n_l + sum_r^2 / n_r; the first strictly largest positive decrease wins.
///
/// Regression (RegressionSplittingRule): each child needs max(ceil(alpha m), 1) units; a short
/// left child skips the candidate, a short right child ends the feature. Causal
/// (InstrumentalSplittingRule, stabilize.splits): each child needs `min_node_size` units with z
/// below the node mean of z and `min_node_size` with z at or above it (left short: skip; right
/// short: end the feature), and each child's sum z^2 - (sum z)^2 / n must be at least alpha times
/// the node's (else skip).
fn best_split(
    data: &ForestData,
    samples: &[usize],
    responses: &[f64],
    features: &[usize],
    min_node_size: usize,
    order: &mut Vec<usize>,
) -> Option<SplitChoice> {
    let m = samples.len();
    if m < 2 {
        return None;
    }
    let total: f64 = responses.iter().sum();
    // causal node constants: (z, mean z, sum z, sum z^2, alpha * size, #(z < mean))
    let node = data.z.map(|z| {
        let (mut s, mut ss) = (0.0, 0.0);
        for &i in samples {
            s += z[i];
            ss += z[i] * z[i];
        }
        let mean = s / m as f64;
        let n_small = samples.iter().filter(|&&i| z[i] < mean).count();
        (z, mean, s, ss, (ss - s * s / m as f64) * GRF_ALPHA, n_small)
    });
    let min_child = ((m as f64 * GRF_ALPHA).ceil() as usize).max(1);
    let (mut best, mut best_dec) = (None, 0.0);
    for &f in features {
        order.clear();
        order.extend(0..m);
        order.sort_by(|&a, &b| data.x[[samples[a], f]].total_cmp(&data.x[[samples[b], f]]));
        let (mut nl, mut sl, mut small_l, mut zl, mut zzl) = (0usize, 0.0, 0usize, 0.0, 0.0);
        for k in 0..m - 1 {
            let pos = order[k];
            let i = samples[pos];
            nl += 1;
            sl += responses[pos];
            if let Some((z, mean, ..)) = node {
                if z[i] < mean {
                    small_l += 1;
                }
                zl += z[i];
                zzl += z[i] * z[i];
            }
            let (xv, xn) = (data.x[[i, f]], data.x[[samples[order[k + 1]], f]]);
            if xv == xn {
                continue;
            }
            let nr = m - nl;
            match node {
                Some((_, _, s, ss, min_size, n_small)) => {
                    if small_l < min_node_size || nl - small_l < min_node_size {
                        continue;
                    }
                    let small_r = n_small - small_l;
                    if small_r < min_node_size || nr - small_r < min_node_size {
                        break;
                    }
                    let size_l = zzl - zl * zl / nl as f64;
                    let size_r = (ss - zzl) - (s - zl) * (s - zl) / nr as f64;
                    if size_l < min_size || size_r < min_size {
                        continue;
                    }
                }
                None => {
                    if nl < min_child {
                        continue;
                    }
                    if nr < min_child {
                        break;
                    }
                }
            }
            let sr = total - sl;
            let dec = sl * sl / nl as f64 + sr * sr / nr as f64;
            if dec > best_dec {
                best_dec = dec;
                best = Some(SplitChoice {
                    feature: f,
                    threshold: (xv + xn) / 2.0,
                    decrease: dec,
                });
            }
        }
    }
    best
}

/// grf TreeTrainer: grow on `growing` with an explicit stack (no recursion), children after their
/// parent in the node vector (pruning relies on it). A node is a leaf when it has at most
/// `min_node_size` units, reaches `max_depth`, its relabeling stops, or no candidate is
/// admissible.
fn grow_tree(data: &ForestData, growing: &[usize], spec: &TreeSpec, rng: &mut u64) -> Vec<Node> {
    let p = data.x.ncols();
    let mut nodes = vec![Node::Leaf(LeafStats::default())];
    let mut stack: Vec<(usize, Vec<usize>, usize)> = vec![(0, growing.to_vec(), 0)];
    let (mut resp, mut order) = (Vec::new(), Vec::new());
    let mut features: Vec<usize> = (0..p).collect();
    while let Some((v, samples, depth)) = stack.pop() {
        if samples.len() <= spec.min_node_size || depth >= spec.max_depth {
            continue;
        }
        match data.z {
            Some(z) => {
                if relabel(&samples, data.y, z, &mut resp).is_none() {
                    continue;
                }
            }
            None => {
                resp.clear();
                resp.extend(samples.iter().map(|&i| data.y[i]));
            }
        }
        let k = poisson_count(spec.mtry as f64, p, rng);
        // partial Fisher-Yates: features[..k] is a uniform k-subset
        for j in 0..k {
            let r = j + lcg_random(rng) % (p - j);
            features.swap(j, r);
        }
        let Some(choice) = best_split(
            data,
            &samples,
            &resp,
            &features[..k],
            spec.min_node_size,
            &mut order,
        ) else {
            continue;
        };
        let (l, r) = partition_exact(samples.iter().copied(), |i| {
            data.x[[i, choice.feature]] <= choice.threshold
        });
        drop(samples);
        let (li, ri) = (nodes.len(), nodes.len() + 1);
        nodes.push(Node::Leaf(LeafStats::default()));
        nodes.push(Node::Leaf(LeafStats::default()));
        nodes[v] = Node::Split {
            feature: choice.feature,
            threshold: choice.threshold,
            left: li,
            right: ri,
        };
        stack.push((ri, r, depth + 1));
        stack.push((li, l, depth + 1));
    }
    nodes
}

/// grf repopulate_leaf_nodes + Tree::honesty_prune_leaves: fill the leaves with the means over
/// `estimation`, then, bottom-up (children have larger indices than their parent), replace a
/// split that has an empty-leaf child by its other child, or by an empty leaf when both are
/// empty.
fn populate(mut nodes: Vec<Node>, data: &ForestData, estimation: &[usize]) -> HonestTree {
    let mut acc = vec![(0usize, 0.0f64, 0.0f64, 0.0f64, 0.0f64); nodes.len()];
    for &i in estimation {
        let mut v = 0;
        while let Node::Split {
            feature,
            threshold,
            left,
            right,
        } = &nodes[v]
        {
            v = if data.x[[i, *feature]] <= *threshold {
                *left
            } else {
                *right
            };
        }
        let z = data.z.map_or(0.0, |z| z[i]);
        let y = data.y[i];
        let a = &mut acc[v];
        a.0 += 1;
        a.1 += y;
        a.2 += z;
        a.3 += y * z;
        a.4 += z * z;
    }
    for (v, a) in acc.iter().enumerate() {
        if let Node::Leaf(s) = &mut nodes[v] {
            if a.0 > 0 {
                let n = a.0 as f64;
                *s = LeafStats {
                    n: a.0,
                    y: a.1 / n,
                    z: a.2 / n,
                    yz: a.3 / n,
                    zz: a.4 / n,
                };
            }
        }
    }
    let empty = |n: &Node| matches!(n, Node::Leaf(s) if s.n == 0);
    let mut redirect: Vec<usize> = (0..nodes.len()).collect();
    for v in (0..nodes.len()).rev() {
        if let Node::Split { left, right, .. } = nodes[v] {
            let (l, r) = (redirect[left], redirect[right]);
            if let Node::Split {
                left: lv,
                right: rv,
                ..
            } = &mut nodes[v]
            {
                *lv = l;
                *rv = r;
            }
            match (empty(&nodes[l]), empty(&nodes[r])) {
                (true, true) => nodes[v] = Node::Leaf(LeafStats::default()),
                (true, false) => redirect[v] = r,
                (false, true) => redirect[v] = l,
                (false, false) => {}
            }
        }
    }
    nodes.shrink_to_fit();
    HonestTree {
        root: redirect[0],
        nodes,
    }
}

/// Honest halves of a tree's subsample: shuffled, the first ceil(m (1 - honesty_fraction)) units
/// grow the tree and the rest estimate the leaves; both parts are non-empty when m >= 2.
fn honest_split(
    indices: &[usize],
    honesty_fraction: f64,
    rng: &mut u64,
) -> (Vec<usize>, Vec<usize>) {
    let mut s = indices.to_vec();
    shuffle(&mut s, rng);
    let g = growing_size(s.len(), honesty_fraction);
    let est = s.split_off(g);
    s.shrink_to_fit();
    (s, est)
}

/// A fitted honest forest with each tree's in-bag set (bitset over the n units) for out-of-bag
/// prediction.
struct HonestForest {
    trees: Vec<HonestTree>,
    drawn: Vec<Vec<u64>>,
}

fn in_bag(bits: &[u64], i: usize) -> bool {
    (bits[i / 64] >> (i % 64)) & 1 == 1
}

/// One tree's subsample of `k` units, at its exact length: from the group's half-sample `half`
/// when it is not empty (shuffled copy, truncated), else from the permutation `all` of every unit
/// (shuffled in place, its first `k`).
fn draw_subsample(all: &mut [usize], half: &[usize], k: usize, rng: &mut u64) -> Vec<usize> {
    if half.is_empty() {
        shuffle(all, rng);
        all[..k].to_vec()
    } else {
        let mut h = half.to_vec();
        shuffle(&mut h, rng);
        h.truncate(k);
        h.shrink_to_fit();
        h
    }
}

/// grf ForestTrainer: `n_trees / group_size` groups. group_size > 1 (train_ci_group): a
/// half-sample of ceil(n / 2) units per group, each tree subsamples ceil(|half| * 2 *
/// sample_fraction) of it. group_size = 1 (train_tree): each tree draws ceil(n * sample_fraction)
/// units. Honest halves by [`honest_split`]; without honesty the subsample both grows and
/// estimates. `on_tree(tree, subsample)` receives each tree as it is grown, in order. A
/// checkpoint per tree.
fn grow_forest(
    data: &ForestData,
    n_trees: usize,
    group_size: usize,
    sample_fraction: f64,
    spec: &TreeSpec,
    rng: &mut u64,
    mut on_tree: impl FnMut(HonestTree, &[usize]) -> EconResult<()>,
) -> EconResult<()> {
    let n = data.x.nrows();
    let k = subsample_size(n, sample_fraction, group_size);
    let mut all: Vec<usize> = (0..n).collect();
    for _ in 0..n_trees / group_size {
        let half: Vec<usize> = if group_size > 1 {
            shuffle(&mut all, rng);
            all[..n.div_ceil(2)].to_vec()
        } else {
            Vec::new()
        };
        for _ in 0..group_size {
            checkpoint()?;
            let sample = draw_subsample(&mut all, &half, k, rng);
            let tree = if spec.honesty {
                let (growing, estimation) = honest_split(&sample, spec.honesty_fraction, rng);
                let nodes = grow_tree(data, &growing, spec, rng);
                drop(growing);
                populate(nodes, data, &estimation)
            } else {
                let nodes = grow_tree(data, &sample, spec, rng);
                populate(nodes, data, &sample)
            };
            on_tree(tree, &sample)?;
        }
    }
    Ok(())
}

/// [`grow_forest`], keeping every tree and its in-bag bitset (`trees x ceil(n / 64)` words,
/// checked before any work).
fn fit_honest_forest(
    data: &ForestData,
    n_trees: usize,
    group_size: usize,
    sample_fraction: f64,
    spec: &TreeSpec,
    rng: &mut u64,
    what: &str,
) -> EconResult<HonestForest> {
    let n = data.x.nrows();
    let grown = (n_trees / group_size) * group_size;
    let words = n.div_ceil(64);
    check_alloc::<u64>(checked_len(&[grown, words], what)?, what)?;
    let mut trees = try_with_capacity::<HonestTree>(grown, what)?;
    let mut drawn = try_with_capacity::<Vec<u64>>(grown, what)?;
    grow_forest(
        data,
        n_trees,
        group_size,
        sample_fraction,
        spec,
        rng,
        |tree, sample| {
            let mut bits = vec![0u64; words];
            for &i in sample {
                bits[i / 64] |= 1 << (i % 64);
            }
            trees.push(tree);
            drawn.push(bits);
            Ok(())
        },
    )?;
    Ok(HonestForest { trees, drawn })
}

/// grf RegressionPredictionStrategy, out of bag, without keeping the forest: the mean, over the
/// trees not drawing unit i whose leaf is not empty, of the leaf means (summed in tree order as
/// each tree is grown, then the tree is dropped). NaN when no such tree exists.
fn oob_regression_forest(
    data: &ForestData,
    n_trees: usize,
    sample_fraction: f64,
    spec: &TreeSpec,
    rng: &mut u64,
) -> EconResult<Vec<f64>> {
    let n = data.x.nrows();
    let mut sum = try_vec(n, 0.0f64, "nuisance forest out-of-bag sums")?;
    let mut count = try_vec(n, 0usize, "nuisance forest out-of-bag counts")?;
    let mut drawn = try_vec(n, false, "nuisance forest subsample flags")?;
    grow_forest(
        data,
        n_trees,
        1,
        sample_fraction,
        spec,
        rng,
        |tree, sample| {
            for &i in sample {
                drawn[i] = true;
            }
            for i in 0..n {
                if drawn[i] {
                    continue;
                }
                let l = tree.leaf(data.x.row(i));
                if l.n > 0 {
                    sum[i] += l.y;
                    count[i] += 1;
                }
            }
            for &i in sample {
                drawn[i] = false;
            }
            Ok(())
        },
    )?;
    Ok(sum
        .iter()
        .zip(&count)
        .map(|(&s, &k)| if k > 0 { s / k as f64 } else { f64::NAN })
        .collect())
}

/// One unit's causal estimate from its leaves (grf InstrumentalPredictionStrategy).
#[derive(Debug, Clone, Copy)]
struct Pooled {
    tau: f64,
    mean_y: f64,
    mean_z: f64,
    first_stage: f64,
    var_between: f64,
    var_total: f64,
    group_noise: f64,
    variance: f64,
}

/// `leaves[k] = (tree index, stats)` of every tree that may predict the unit and whose leaf is
/// not empty, in tree order. tau = (avg(yz) - avg(y) avg(z)) / first, first = avg(zz) - avg(z)^2
/// (NaN when there is no leaf or |first| < RELABEL_EPS).
///
/// Variance (group_size >= 2; grf compute_variance): over the groups (tree / group_size) whose
/// every tree is present, rho_b = (psi1_b - avg(z) psi2_b) / first with psi1 = yz - zz tau - z mu,
/// psi2 = y - z tau - mu, mu = avg(y) - avg(z) tau; var_between = sum_g mean_g(rho)^2 / G,
/// var_total = sum rho^2 / (G g), group_noise = (var_total - var_between) / (g - 1),
/// variance = bayes_debias(var_between, group_noise, G). NaN when no group is complete or
/// group_size = 1.
fn pooled_causal(leaves: &[(usize, LeafStats)], group_size: usize) -> Pooled {
    let nan = Pooled {
        tau: f64::NAN,
        mean_y: f64::NAN,
        mean_z: f64::NAN,
        first_stage: f64::NAN,
        var_between: f64::NAN,
        var_total: f64::NAN,
        group_noise: f64::NAN,
        variance: f64::NAN,
    };
    let k = leaves.len();
    if k == 0 {
        return nan;
    }
    let kf = k as f64;
    let (mut ay, mut az, mut ayz, mut azz) = (0.0, 0.0, 0.0, 0.0);
    for (_, l) in leaves {
        ay += l.y;
        az += l.z;
        ayz += l.yz;
        azz += l.zz;
    }
    let (ay, az, ayz, azz) = (ay / kf, az / kf, ayz / kf, azz / kf);
    let first = azz - az * az;
    let mut out = Pooled {
        mean_y: ay,
        mean_z: az,
        first_stage: first,
        ..nan
    };
    if first.is_nan() || first.abs() < RELABEL_EPS {
        return out;
    }
    let tau = (ayz - ay * az) / first;
    out.tau = tau;
    if group_size < 2 {
        return out;
    }
    let mu = ay - az * tau;
    let (mut sum_group_sq, mut sum_sq, mut good) = (0.0, 0.0, 0usize);
    let mut j = 0;
    while j < k {
        let g = leaves[j].0 / group_size;
        let mut e = j;
        while e < k && leaves[e].0 / group_size == g {
            e += 1;
        }
        if e - j == group_size {
            let mut gs = 0.0;
            for (_, l) in &leaves[j..e] {
                let psi1 = l.yz - l.zz * tau - l.z * mu;
                let psi2 = l.y - l.z * tau - mu;
                let rho = (psi1 - az * psi2) / first;
                gs += rho;
                sum_sq += rho * rho;
            }
            let gm = gs / group_size as f64;
            sum_group_sq += gm * gm;
            good += 1;
        }
        j = e;
    }
    if good == 0 {
        return out;
    }
    let gf = good as f64;
    out.var_between = sum_group_sq / gf;
    out.var_total = sum_sq / (gf * group_size as f64);
    out.group_noise = (out.var_total - out.var_between) / (group_size as f64 - 1.0);
    out.variance = bayes_debias(out.var_between, out.group_noise, gf);
    out
}

/// grf ObjectiveBayesDebiaser::debias: e + s phi(e/s) / Phi(e/s) with e = var_between -
/// group_noise and s = max(var_between, group_noise) sqrt(2 / groups) (the posterior mean of a
/// non-negative variance). 0 when s is not positive (grf: |s| < 1e-10, which is not scale-free).
/// Below a ratio of -30, phi/Phi is the Mills continued fraction x + 1/(x + 2/(x + 3/(...))) at
/// x = -ratio (40 terms; exp and erfc underflow near -37).
fn bayes_debias(var_between: f64, group_noise: f64, groups: f64) -> f64 {
    let e = var_between - group_noise;
    let s = var_between.max(group_noise) * (2.0 / groups).sqrt();
    if !(s > 0.0 && s.is_finite()) {
        return 0.0;
    }
    let r = e / s;
    let ratio = if r >= -30.0 {
        let phi = (-0.5 * r * r).exp() / (2.0 * std::f64::consts::PI).sqrt();
        phi / (0.5 * erfc(-r / std::f64::consts::SQRT_2))
    } else {
        let x = -r;
        let mut t = x;
        for k in (1..=40).rev() {
            t = x + k as f64 / t;
        }
        t
    };
    e + s * ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{Budget, CancelToken, with_alloc_ceiling, with_budget};
    use ndarray::array;

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    fn names3() -> Vec<String> {
        vec!["x0".into(), "x1".into(), "x2".into()]
    }

    fn small_config() -> CausalForestConfig {
        CausalForestConfig {
            n_trees: 10,
            max_depth: 4,
            seed: Some(3),
            ..Default::default()
        }
    }

    /// Distinct covariate values, alternating treatment and a noisy outcome with an effect that
    /// varies with x0: every threshold the stabilisation allows is a candidate, so the trees grow
    /// as large as the size rules let them.
    fn splittable_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let mut rng = seed;
        let mut x = Array2::zeros((n, 3));
        let mut y = Array1::zeros(n);
        let mut w = Array1::zeros(n);
        for i in 0..n {
            for j in 0..3 {
                x[[i, j]] = (lcg_random(&mut rng) % 1_000_003) as f64 + i as f64 * 1e-7;
            }
            w[i] = (i % 2) as f64;
            let noise = (lcg_random(&mut rng) % 1000) as f64 / 1000.0;
            y[i] = x[[i, 0]] * 1e-6 * (1.0 + w[i]) + noise;
        }
        (x, y, w)
    }

    /// The node bounds behind `memory_bound` hold for every tree of the causal forest and of a
    /// nuisance forest grown as the fit grows them, with and without honesty, at node sizes 1 to
    /// 5 and depths 3 to 100, and are not loose by more than a few times.
    #[test]
    fn memory_bound_counts_at_least_the_nodes_every_tree_grows() {
        for (n, honesty, min_node_size, max_depth) in [
            (400, true, 5, 10),
            (400, true, 1, 100),
            (400, false, 1, 100),
            (400, false, 2, 100),
            (400, false, 5, 100),
            (250, false, 4, 3),
            (61, true, 5, 10),
            (1_000, false, 1, 100),
        ] {
            let cfg = CausalForestConfig {
                n_trees: 40,
                honesty,
                min_node_size,
                max_depth,
                seed: Some(11),
                ..Default::default()
            };
            let bound = cfg.max_nodes_per_tree(n);
            let (x, y, w) = splittable_data(n, 7 + n as u64);
            let fit =
                causal_forest_arrays(x.view(), y.view(), w.view(), names3(), cfg.clone()).unwrap();
            let most = fit
                .trees
                .iter()
                .map(|t| t.nodes.len() as u128)
                .max()
                .unwrap();
            assert!(
                most <= bound,
                "{cfg:?} on {n} rows: a tree has {most} nodes, bound {bound}"
            );
            assert!(4 * most >= bound, "{cfg:?} on {n} rows: {most} of {bound}");
            // a nuisance forest, as the fit grows it (the W.hat forest's target)
            let nuisance_bound = cfg.max_nodes_per_nuisance_tree(n);
            let spec = TreeSpec {
                min_node_size: NUISANCE_MIN_NODE_SIZE,
                max_depth: usize::MAX,
                mtry: 3,
                honesty: true,
                honesty_fraction: NUISANCE_HONESTY_FRACTION,
            };
            let w_vec = w.to_vec();
            let data = ForestData {
                x: x.view(),
                y: &w_vec,
                z: None,
            };
            let mut nuisance_most = 0u128;
            let mut rng = 5u64;
            grow_forest(&data, 6, 1, cfg.sample_fraction, &spec, &mut rng, |t, _| {
                nuisance_most = nuisance_most.max(t.nodes.len() as u128);
                Ok(())
            })
            .unwrap();
            assert!(
                nuisance_most <= nuisance_bound,
                "nuisance tree on {n} rows: {nuisance_most} nodes, bound {nuisance_bound}"
            );
            assert!(
                8 * nuisance_most >= nuisance_bound,
                "nuisance tree on {n} rows: {nuisance_most} of {nuisance_bound}"
            );
        }
    }

    /// The subsamples, their honest parts and a split's children are allocated at their exact
    /// lengths, as `memory_bound` counts them: `Iterator::partition` and `filter().collect()`
    /// grow a vector by pushing and can leave close to twice its length allocated.
    #[test]
    fn index_vectors_are_allocated_at_their_lengths() {
        let mut rng = 5;
        for (n, fraction, group) in [(1_000, 0.5, 2), (777, 0.632, 1), (61, 1.0, 1), (10, 0.1, 2)] {
            let k = subsample_size(n, fraction, group);
            let mut all: Vec<usize> = (0..n).collect();
            let half: Vec<usize> = if group > 1 {
                all[..n.div_ceil(2)].to_vec()
            } else {
                Vec::new()
            };
            let sample = draw_subsample(&mut all, &half, k, &mut rng);
            assert_eq!(sample.len(), k);
            assert_eq!(sample.capacity(), k, "{n} rows, fraction {fraction}");
            let (growing, estimation) = honest_split(&sample, 0.5, &mut rng);
            assert_eq!(growing.len() + estimation.len(), k);
            assert_eq!(growing.capacity(), growing.len(), "{n} rows");
            assert_eq!(estimation.capacity(), estimation.len(), "{n} rows");
        }
        let x = Array2::from_shape_fn((1_000, 1), |(i, _)| i as f64 + 0.25 * (i % 3) as f64);
        let indices: Vec<usize> = (0..1_000).rev().collect();
        for threshold in [0.1, 3.0, 500.0, 996.9, 2_000.0] {
            let (left, right) =
                partition_exact(indices.iter().copied(), |i| x[[i, 0]] <= threshold);
            assert_eq!(left.len() + right.len(), 1_000);
            assert_eq!(left.capacity(), left.len(), "threshold {threshold}");
            assert_eq!(right.capacity(), right.len(), "threshold {threshold}");
        }
    }

    #[test]
    fn memory_bound_grows_with_its_arguments_and_saturates() {
        let d = CausalForestConfig::default();
        let at = |cfg: &CausalForestConfig, n: usize, p: usize| cfg.memory_bound(n, p);
        let base = at(&d, 10_000, 4);
        let with = |f: &dyn Fn(&mut CausalForestConfig)| {
            let mut c = d.clone();
            f(&mut c);
            c
        };
        assert!(at(&with(&|c| c.n_trees = 3_000), 10_000, 4) > base);
        assert!(at(&d, 20_000, 4) > base);
        assert!(at(&d, 10_000, 8) > base);
        assert!(at(&with(&|c| c.honesty = false), 10_000, 4) > base);
        assert!(at(&with(&|c| c.min_node_size = 1), 10_000, 4) > base);
        assert!(at(&with(&|c| c.max_depth = 20), 1_000_000, 4) > at(&d, 1_000_000, 4));
        // the nuisance trees are grown and dropped one at a time: their number does not count
        assert_eq!(
            at(&with(&|c| c.nuisance_trees = Some(5_000)), 10_000, 4),
            base
        );
        // groups of 2 round 2001 trees down to 2000; groups of 1 keep all of them
        let odd = with(&|c| c.n_trees = 2_001);
        let ungrouped = CausalForestConfig {
            ci_group_size: 1,
            ..odd.clone()
        };
        assert_eq!(at(&odd, 10_000, 4), base);
        assert!(at(&ungrouped, 10_000, 4) > at(&odd, 10_000, 4));
        // At the default depth a causal tree has at most 2^11 - 1 nodes however many rows.
        let node = size_of::<Node>() as u128;
        assert_eq!(d.max_nodes_per_tree(10_000_000), 2_047);
        assert!(u128::from(at(&d, 1_000_000, 4)) > 2_000 * 2_047 * node);
        let huge = CausalForestConfig {
            n_trees: usize::MAX,
            max_depth: usize::MAX,
            min_node_size: 0,
            ..d.clone()
        };
        assert_eq!(at(&huge, usize::MAX, usize::MAX), u64::MAX);
        assert_eq!(at(&d, usize::MAX, 4), u64::MAX);
        assert!(at(&d, 0, 0) > 0);
    }

    /// A `min_node_size` above `usize::MAX / 2` cannot split any node, with or without honesty,
    /// and neither panics nor wraps (in the split rule or in the memory bound): every causal tree
    /// is a single leaf, exactly as at a `min_node_size` of all the rows.
    #[test]
    fn a_min_node_size_above_half_of_usize_neither_panics_nor_wraps() {
        let (x, y, w, _) = overlap_test_data(60, 5);
        for honesty in [true, false] {
            let fit = |min_node_size| {
                let cfg = CausalForestConfig {
                    honesty,
                    min_node_size,
                    n_trees: 40,
                    ..small_config()
                };
                causal_forest_arrays(x.view(), y.view(), w.view(), names3(), cfg).unwrap()
            };
            let bits = |v: &[f64]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            let want = fit(60);
            for min_node_size in [usize::MAX / 2 + 1, usize::MAX - 1, usize::MAX] {
                let got = fit(min_node_size);
                let case = format!("honesty {honesty}, min_node_size {min_node_size}");
                assert!(
                    got.trees
                        .iter()
                        .all(|t| matches!(t.nodes[t.root], Node::Leaf(_))),
                    "{case}"
                );
                assert_eq!(bits(&got.predictions), bits(&want.predictions), "{case}");
                assert_eq!(
                    bits(&[got.ate, got.ate_se, got.oob_error]),
                    bits(&[want.ate, want.ate_se, want.oob_error]),
                    "{case}"
                );
                let cfg = CausalForestConfig {
                    honesty,
                    min_node_size,
                    ..small_config()
                };
                assert_eq!(cfg.max_nodes_per_tree(1_000_000), 1, "{case}");
            }
        }
    }

    /// `run_causal_forest_config` gives the configuration `run_causal_forest` used before it
    /// existed: the arguments, and the defaults where they are `None` (also of the fields it
    /// does not take).
    #[test]
    fn run_config_takes_the_arguments_and_the_defaults() {
        let c = run_causal_forest_config(None, None, None, None, None);
        let d = CausalForestConfig::default();
        assert_eq!(
            (c.n_trees, c.min_node_size, c.honesty, c.max_depth, c.seed),
            (2000, 5, true, 10, None)
        );
        assert_eq!(
            (c.sample_fraction, c.honesty_fraction, c.mtry),
            (d.sample_fraction, d.honesty_fraction, d.mtry)
        );
        assert_eq!((c.ci_group_size, c.nuisance_trees), (2, None));
        let c = run_causal_forest_config(Some(7), Some(2), Some(false), Some(3), Some(9));
        assert_eq!(
            (c.n_trees, c.min_node_size, c.honesty, c.max_depth, c.seed),
            (7, 2, false, 3, Some(9))
        );
    }

    /// WP47-9: the default mtry stays p2a's `ceil(sqrt(p))`, stored in the fitted result.
    #[test]
    fn test_causal_forest_default_mtry() {
        for (p, want) in [(3usize, 2usize), (10, 4), (100, 10)] {
            let (x3, y, w, _) = overlap_test_data(120, 21);
            let x = Array2::from_shape_fn((120, p), |(i, j)| x3[[i, j % 3]] + j as f64 * 1e-3);
            let mut c = CausalForestConfig::default();
            c.n_trees = 20;
            c.seed = Some(21);
            let r = causal_forest_arrays(x.view(), y.view(), w.view(), names(p), c).unwrap();
            assert_eq!(r.config.mtry, Some(want), "p = {p}");
            assert_eq!(default_mtry(p), want);
        }
    }

    #[test]
    fn causal_forest_huge_n_trees_is_an_error_not_an_abort() {
        let (x, y, w) = generate_test_data(60, 5);
        let cfg = CausalForestConfig {
            n_trees: 1usize << 44,
            ..small_config()
        };
        let r = with_alloc_ceiling(Some(1 << 30), || {
            causal_forest_arrays(x.view(), y.view(), w.view(), names3(), cfg)
        });
        assert!(
            matches!(r, Err(EconError::ResourceLimit(_))),
            "{:?}",
            r.err()
        );
    }

    #[test]
    fn causal_forest_validates_counts_and_fractions() {
        let ok = small_config();
        assert!(validate_causal_forest_config(&ok).is_ok());
        for (cfg, param) in [
            (
                CausalForestConfig {
                    n_trees: 0,
                    ..ok.clone()
                },
                "n_trees",
            ),
            (
                CausalForestConfig {
                    sample_fraction: 1.5,
                    ..ok.clone()
                },
                "sample_fraction",
            ), // looped forever
            (
                CausalForestConfig {
                    sample_fraction: 0.0,
                    ..ok.clone()
                },
                "sample_fraction",
            ),
            (
                CausalForestConfig {
                    sample_fraction: f64::NAN,
                    ..ok.clone()
                },
                "sample_fraction",
            ),
            (
                CausalForestConfig {
                    honesty_fraction: 1.0,
                    ..ok.clone()
                },
                "honesty_fraction",
            ),
            (
                CausalForestConfig {
                    honesty_fraction: 0.0,
                    ..ok.clone()
                },
                "honesty_fraction",
            ),
            (
                CausalForestConfig {
                    ci_group_size: 0,
                    ..ok.clone()
                },
                "ci_group_size",
            ),
            (
                CausalForestConfig {
                    sample_fraction: 0.75,
                    ..ok.clone()
                },
                "sample_fraction",
            ), // grf: < 0.5 with CIs
            (
                CausalForestConfig {
                    n_trees: 1,
                    ..ok.clone()
                },
                "n_trees",
            ), // fewer than one group
            (
                CausalForestConfig {
                    min_node_size: 0,
                    ..ok.clone()
                },
                "min_node_size",
            ),
            (
                CausalForestConfig {
                    nuisance_trees: Some(0),
                    ..ok.clone()
                },
                "nuisance_trees",
            ),
            (
                CausalForestConfig {
                    mtry: Some(0),
                    ..ok.clone()
                },
                "mtry",
            ),
        ] {
            let e = validate_causal_forest_config(&cfg).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.starts_with(param)),
                "{e}"
            );
        }
        // honesty_fraction is irrelevant without honesty
        let no_honesty = CausalForestConfig {
            honesty: false,
            honesty_fraction: 1.0,
            ..ok.clone()
        };
        assert!(validate_causal_forest_config(&no_honesty).is_ok());
        // without little bags a larger subsample is allowed
        assert!(
            validate_causal_forest_config(&CausalForestConfig {
                ci_group_size: 1,
                sample_fraction: 0.75,
                ..ok.clone()
            })
            .is_ok()
        );
        // full path: n_trees = 0 used to underflow `preds.len() - 1`
        let (x, y, w) = generate_test_data(60, 5);
        let zero = CausalForestConfig { n_trees: 0, ..ok };
        assert!(causal_forest_arrays(x.view(), y.view(), w.view(), names3(), zero).is_err());
    }

    #[test]
    fn causal_forest_honours_cancellation_and_rejects_non_finite_inputs() {
        let (mut x, y, w) = generate_test_data(60, 5);
        let r = with_budget(cancelled(), || {
            causal_forest_arrays(x.view(), y.view(), w.view(), names3(), small_config())
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "{:?}", r.err());
        x[[3, 1]] = f64::NAN;
        let r = causal_forest_arrays(x.view(), y.view(), w.view(), names3(), small_config());
        assert!(
            matches!(&r, Err(EconError::InvalidSpecification { message }) if message.contains("finite")),
            "{:?}",
            r.err()
        );
    }

    /// The decrease of the causal split at `threshold` on feature 0, recomputed from the explicit
    /// partition of the relabelled node.
    fn causal_decrease(x: &Array2<f64>, rho: &[f64], idx: &[usize], threshold: f64) -> f64 {
        let (mut sl, mut nl, mut sr, mut nr) = (0.0, 0.0, 0.0, 0.0);
        for (k, &i) in idx.iter().enumerate() {
            if x[[i, 0]] <= threshold {
                sl += rho[k];
                nl += 1.0;
            } else {
                sr += rho[k];
                nr += 1.0;
            }
        }
        sl * sl / nl + sr * sr / nr
    }

    /// Mirror-image data: the groups x = 0 and x = 2 hold the same (y, w) pairs, at least two
    /// treated and two control rows each, and x = 1 a few other rows, all shuffled; y on a grid
    /// of 1 / `denom`. The two thresholds then have equal criteria (up to rounding).
    fn mirrored_data(s: &mut u64, denom: f64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let k = 4 + lcg_random(s) % 3;
        let mut group: Vec<(f64, f64)> = Vec::with_capacity(k);
        for i in 0..k {
            let w = match i {
                0 | 1 => 1.0,
                2 | 3 => 0.0,
                _ => (lcg_random(s) % 2) as f64,
            };
            group.push(((lcg_random(s) % 21) as f64 / denom, w));
        }
        let mut rows: Vec<(f64, f64, f64)> = group.iter().map(|&(y, w)| (0.0, y, w)).collect();
        for _ in 0..1 + lcg_random(s) % 3 {
            let y = (lcg_random(s) % 21) as f64 / denom;
            rows.push((1.0, y, (lcg_random(s) % 2) as f64));
        }
        rows.extend(group.iter().map(|&(y, w)| (2.0, y, w)));
        for i in (1..rows.len()).rev() {
            let j = lcg_random(s) % (i + 1);
            rows.swap(i, j);
        }
        let n = rows.len();
        (
            Array2::from_shape_fn((n, 1), |(i, _)| rows[i].0),
            Array1::from_shape_fn(n, |i| rows[i].1),
            Array1::from_shape_fn(n, |i| rows[i].2),
        )
    }

    /// The centred treatment of a node: w minus its mean.
    fn centred(w: &Array1<f64>) -> Vec<f64> {
        let m = w.mean().unwrap();
        w.iter().map(|v| v - m).collect()
    }

    /// Mirror-image groups (x = 0 and x = 2 hold the same (y, w) pairs) give x <= 0.5 and
    /// x <= 1.5 the same decrease: the groups' pseudo-outcomes have the same sum. Rounding
    /// decides which one a search sees as larger, so either is a correct answer; the sweep and
    /// the naive search must each return one.
    #[test]
    fn causal_split_tie_takes_one_of_the_tied_splits() {
        let x = Array2::from_shape_vec((9, 1), vec![2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0])
            .unwrap();
        let y = vec![0.5, 0.5, 0.5, 0.3, 0.9, 0.3, 0.5, 0.4, 0.4];
        let w = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
        let z = centred(&w);
        let idx: Vec<usize> = (0..9).collect();
        let mut rho = Vec::new();
        relabel(&idx, &y, &z, &mut rho).unwrap();
        let data = ForestData {
            x: x.view(),
            y: &y,
            z: Some(&z),
        };
        let (a, b) = (
            causal_decrease(&x, &rho, &idx, 0.5),
            causal_decrease(&x, &rho, &idx, 1.5),
        );
        assert!((a - b).abs() <= 1e-12 * a.max(1.0), "{a} vs {b}");
        let sweep = best_split(&data, &idx, &rho, &[0], 1, &mut Vec::new())
            .map(|c| (c.feature, c.threshold));
        let reference = naive_best_split(&x.view(), &y, Some(&z), &idx, &[0], 1);
        for (who, split) in [("sweep", sweep), ("reference", reference)] {
            assert!(
                matches!(split, Some((0, t)) if t == 0.5 || t == 1.5),
                "{who}: {split:?}"
            );
        }
    }

    /// Reference guard on tie-prone (mirror-image) data: the sweep chooses the naive search's
    /// split, or a different one with the same decrease.
    #[test]
    fn causal_split_sweep_on_tied_data_matches_or_ties_the_reference() {
        let mut s = 77u64;
        for denom in [1.0, 10.0] {
            for rep in 0..1000 {
                let (x, y, w) = mirrored_data(&mut s, denom);
                let y = y.to_vec();
                let z = centred(&w);
                let idx: Vec<usize> = (0..y.len()).collect();
                let min_node_size = [1usize, 2][rep % 2];
                let mut rho = Vec::new();
                if relabel(&idx, &y, &z, &mut rho).is_none() {
                    continue;
                }
                let data = ForestData {
                    x: x.view(),
                    y: &y,
                    z: Some(&z),
                };
                let got = best_split(&data, &idx, &rho, &[0], min_node_size, &mut Vec::new())
                    .map(|c| (c.feature, c.threshold));
                let want = naive_best_split(&x.view(), &y, Some(&z), &idx, &[0], min_node_size);
                let ctx = format!("denom={denom} rep={rep} x={x} y={y:?} w={w}");
                if got != want {
                    // A split with no decrease beyond rounding (pseudo-outcomes summing to zero
                    // on both sides) may be found by one search and not by the other.
                    let scale: f64 = rho
                        .iter()
                        .map(|r| r * r)
                        .sum::<f64>()
                        .max(f64::MIN_POSITIVE);
                    let dec = |s: Option<(usize, f64)>| {
                        s.map_or(0.0, |(_, t)| causal_decrease(&x, &rho, &idx, t))
                    };
                    let (a, b) = (dec(got), dec(want));
                    assert!(
                        (a - b).abs() <= 1e-12 * scale,
                        "{ctx}: {got:?} decrease {a} vs {want:?} decrease {b} (scale {scale})"
                    );
                }
            }
        }
    }

    // ---- Task 1: grf machinery (formula pins against cases written out in R) ----

    use crate::data::columns::{NaPolicy, numeric_column};
    use p2a_validation::{Check, Tol};

    /// Columns of a case dataset read through the T2 extractor, truncated to `rows`.
    fn case_columns(ds: &Dataset, names: &[&str], rows: usize) -> Vec<Vec<f64>> {
        names
            .iter()
            .map(|c| {
                let mut v = numeric_column(ds.df(), c, NaPolicy::Fail).unwrap().0;
                v.truncate(rows);
                v
            })
            .collect()
    }

    fn matrix(cols: &[Vec<f64>]) -> Array2<f64> {
        Array2::from_shape_fn((cols[0].len(), cols.len()), |(i, j)| cols[j][i])
    }

    /// grf InstrumentalRelabelingStrategy, InstrumentalSplittingRule (stabilised) and
    /// RegressionSplittingRule on one node, written out in R (cf_split_node).
    #[test]
    fn test_validate_cf_split_node_causal() {
        let (case, ds) = crate::test_support::load_case("cf_split_node");
        let c = case_columns(&ds, &["x", "x2", "y", "z"], 30);
        let x = matrix(&c[..2]);
        let (y, z) = (c[2].as_slice(), c[3].as_slice());
        let samples: Vec<usize> = (0..30).collect();
        let data = ForestData {
            x: x.view(),
            y,
            z: Some(z),
        };
        let mut rho = Vec::new();
        let tau = relabel(&samples, y, z, &mut rho).unwrap();
        let mut order = Vec::new();
        let s3 = best_split(&data, &samples, &rho, &[0, 1], 3, &mut order).unwrap();
        let s5 = best_split(&data, &samples, &rho, &[0, 1], 5, &mut order).unwrap();
        let s3x2 = best_split(&data, &samples, &rho, &[1], 3, &mut order).unwrap();
        Check::new(&case)
            .scalar("node_tau", tau, Tol::coef(30))
            .vector("rho", &rho, Tol::coef(30))
            .scalar("causal3_feature", s3.feature as f64, Tol::exact())
            .scalar("causal3_threshold", s3.threshold, Tol::exact())
            .scalar("causal3_decrease", s3.decrease, Tol::stat(30))
            .scalar("causal5_feature", s5.feature as f64, Tol::exact())
            // 0.485: the stabilisation moved the split
            .scalar("causal5_threshold", s5.threshold, Tol::exact())
            .scalar("causal5_decrease", s5.decrease, Tol::stat(30))
            .scalar("causal3_x2_threshold", s3x2.threshold, Tol::exact())
            .scalar("causal3_x2_decrease", s3x2.decrease, Tol::stat(30))
            .finish();
    }

    #[test]
    fn test_validate_cf_split_node_regression() {
        let (case, ds) = crate::test_support::load_case("cf_split_node");
        let c = case_columns(&ds, &["x", "x2", "r"], 30);
        let x = matrix(&c[..2]);
        let samples: Vec<usize> = (0..30).collect();
        let data = ForestData {
            x: x.view(),
            y: &c[2],
            z: None,
        };
        let mut order = Vec::new();
        let s = best_split(&data, &samples, &c[2], &[0, 1], 5, &mut order).unwrap();
        Check::new(&case)
            .scalar("reg_feature", s.feature as f64, Tol::exact())
            .scalar("reg_threshold", s.threshold, Tol::exact())
            .scalar("reg_decrease", s.decrease, Tol::stat(30))
            .finish();
    }

    /// grf InstrumentalPredictionStrategy::predict / compute_variance and ObjectiveBayesDebiaser
    /// (cf_little_bags): the 12 leaves are trees 0..12 in groups of 2; the "part" values drop
    /// tree 1, leaving group 0 incomplete.
    #[test]
    fn test_validate_cf_little_bags() {
        let (case, ds) = crate::test_support::load_case("cf_little_bags");
        let c = case_columns(&ds, &["n", "y", "w", "yw", "ww"], 12);
        let leaves: Vec<(usize, LeafStats)> = (0..12)
            .map(|t| {
                (
                    t,
                    LeafStats {
                        n: c[0][t] as usize,
                        y: c[1][t],
                        z: c[2][t],
                        yz: c[3][t],
                        zz: c[4][t],
                    },
                )
            })
            .collect();
        let full = pooled_causal(&leaves, 2);
        let part_leaves: Vec<(usize, LeafStats)> =
            leaves.iter().copied().filter(|(t, _)| *t != 1).collect();
        let part = pooled_causal(&part_leaves, 2);
        let mu = full.mean_y - full.mean_z * full.tau;
        Check::new(&case)
            .scalar("tau", full.tau, Tol::coef(12))
            .scalar("mu", mu, Tol::coef(12))
            .scalar("first_stage", full.first_stage, Tol::coef(12))
            .scalar("var_between", full.var_between, Tol::stat(12))
            .scalar("var_total", full.var_total, Tol::stat(12))
            .scalar("group_noise", full.group_noise, Tol::stat(12))
            .scalar("variance", full.variance, Tol::stat(12))
            .scalar("part_tau", part.tau, Tol::coef(12))
            .scalar("part_first_stage", part.first_stage, Tol::coef(12))
            .scalar("part_var_between", part.var_between, Tol::stat(12))
            .scalar("part_group_noise", part.group_noise, Tol::stat(12))
            .scalar("part_variance", part.variance, Tol::stat(12))
            .finish();
        assert!(pooled_causal(&leaves, 1).variance.is_nan()); // ci_group_size 1: no variance
        assert!(pooled_causal(&[], 2).tau.is_nan());
    }

    #[test]
    fn test_validate_cf_bayes_debiaser() {
        let (case, _) = crate::test_support::load_case("cf_little_bags");
        let (vb, gn, g) = (
            case.vector("debias_var_between"),
            case.vector("debias_group_noise"),
            case.vector("debias_groups"),
        );
        let out: Vec<f64> = (0..vb.len())
            .map(|k| bayes_debias(vb[k], gn[k], g[k]))
            .collect();
        // out[2] has ratio -38.69: exp/erfc underflow there; the continued fraction must be used
        Check::new(&case)
            .vector("debias_out", &out, Tol::stat(12))
            .finish();
        assert!(out.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    /// Guard for the sorted sweep: same choice as recomputing every quantity per candidate
    /// threshold from an explicit partition (grf's rules, written naively).
    fn naive_best_split(
        x: &ArrayView2<f64>,
        y: &[f64],
        z: Option<&[f64]>,
        samples: &[usize],
        features: &[usize],
        min_node_size: usize,
    ) -> Option<(usize, f64)> {
        let m = samples.len();
        let responses: Vec<f64> = match z {
            Some(z) => {
                let mut r = Vec::new();
                relabel(samples, y, z, &mut r)?;
                r
            }
            None => samples.iter().map(|&i| y[i]).collect(),
        };
        let pos: std::collections::HashMap<usize, usize> =
            samples.iter().enumerate().map(|(k, &i)| (i, k)).collect();
        let zmean = z.map(|z| samples.iter().map(|&i| z[i]).sum::<f64>() / m as f64);
        let size = |idx: &[usize], z: &[f64]| {
            let s: f64 = idx.iter().map(|&i| z[i]).sum();
            idx.iter().map(|&i| z[i] * z[i]).sum::<f64>() - s * s / idx.len() as f64
        };
        let (mut best, mut best_dec) = (None, 0.0);
        for &f in features {
            let mut vals: Vec<f64> = samples.iter().map(|&i| x[[i, f]]).collect();
            vals.sort_by(|a, b| a.total_cmp(b));
            vals.dedup();
            'thresholds: for win in vals.windows(2) {
                let t = (win[0] + win[1]) / 2.0;
                let (l, r): (Vec<usize>, Vec<usize>) =
                    samples.iter().partition(|&&i| x[[i, f]] <= t);
                match (z, zmean) {
                    (Some(z), Some(zm)) => {
                        let small = |v: &[usize]| v.iter().filter(|&&i| z[i] < zm).count();
                        let (sl, sr) = (small(&l), small(&r));
                        if sl < min_node_size || l.len() - sl < min_node_size {
                            continue 'thresholds;
                        }
                        if sr < min_node_size || r.len() - sr < min_node_size {
                            break 'thresholds;
                        }
                        let min_size = size(samples, z) * GRF_ALPHA;
                        if size(&l, z) < min_size || size(&r, z) < min_size {
                            continue 'thresholds;
                        }
                    }
                    _ => {
                        let mcc = ((m as f64 * GRF_ALPHA).ceil() as usize).max(1);
                        if l.len() < mcc {
                            continue 'thresholds;
                        }
                        if r.len() < mcc {
                            break 'thresholds;
                        }
                    }
                }
                let sum = |v: &[usize]| v.iter().map(|i| responses[pos[i]]).sum::<f64>();
                let dec = sum(&l).powi(2) / l.len() as f64 + sum(&r).powi(2) / r.len() as f64;
                if dec > best_dec {
                    best_dec = dec;
                    best = Some((f, t));
                }
            }
        }
        best
    }

    /// The sorted sweep chooses the split the naive search chooses, causal and regression, on
    /// `generate_test_data` (x on a 0.01 grid: many ties).
    #[test]
    fn causal_split_sweep_matches_naive_reference() {
        let (x, y, w) = generate_test_data(400, 77);
        let z: Vec<f64> = (0..400).map(|i| w[i] - (0.3 + 0.4 * x[[i, 0]])).collect();
        let y: Vec<f64> = y.to_vec();
        let mut s = 1234u64;
        let mut order = Vec::new();
        let mut checked = 0;
        for min_node_size in [1usize, 3, 5] {
            for causal in [true, false] {
                for rep in 0..30 {
                    let mut idx: Vec<usize> = (0..400).collect();
                    shuffle(&mut idx, &mut s);
                    idx.truncate(20 + lcg_random(&mut s) % 380);
                    let data = ForestData {
                        x: x.view(),
                        y: &y,
                        z: causal.then_some(z.as_slice()),
                    };
                    let mut resp = Vec::new();
                    if causal {
                        if relabel(&idx, &y, &z, &mut resp).is_none() {
                            continue;
                        }
                    } else {
                        resp = idx.iter().map(|&i| y[i]).collect();
                    }
                    let got = best_split(&data, &idx, &resp, &[0, 1, 2], min_node_size, &mut order)
                        .map(|c| (c.feature, c.threshold));
                    let want =
                        naive_best_split(&x.view(), &y, data.z, &idx, &[0, 1, 2], min_node_size);
                    assert_eq!(
                        got, want,
                        "min_node_size={min_node_size} causal={causal} rep={rep}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked >= 170, "{checked}");
    }

    /// grf Tree::honesty_prune_leaves: a split with an empty child is replaced by the other child.
    #[test]
    fn test_honest_tree_prune_promotes_the_non_empty_sibling() {
        // root: x0 <= 0.5 ? leaf 1 : (x1 <= 0.5 ? leaf 3 : leaf 4)
        let nodes = vec![
            Node::Split {
                feature: 0,
                threshold: 0.5,
                left: 1,
                right: 2,
            },
            Node::Leaf(LeafStats::default()),
            Node::Split {
                feature: 1,
                threshold: 0.5,
                left: 3,
                right: 4,
            },
            Node::Leaf(LeafStats::default()),
            Node::Leaf(LeafStats::default()),
        ];
        let x = array![[0.1, 0.9], [0.2, 0.1], [0.8, 0.2], [0.9, 0.3]];
        let y = [1.0, 2.0, 3.0, 5.0];
        let data = ForestData {
            x: x.view(),
            y: &y,
            z: None,
        };
        // leaf 4 (x0 > 0.5, x1 > 0.5) stays empty
        let tree = populate(nodes.clone(), &data, &[0, 1, 2, 3]);
        let at = |a: f64, b: f64| *tree.leaf(array![a, b].view());
        assert_eq!(
            at(0.9, 0.9).n,
            2,
            "the point of the empty leaf now reaches leaf 3"
        );
        assert!((at(0.9, 0.9).y - 4.0).abs() < 1e-15);
        assert_eq!(at(0.1, 0.1).n, 2);
        assert!((at(0.1, 0.1).y - 1.5).abs() < 1e-15);
        let empty = populate(nodes, &data, &[]);
        assert_eq!(empty.leaf(array![0.1, 0.1].view()).n, 0);
    }

    /// grf get_scores.causal_forest and average_treatment_effect (AIPW, .sigma2.hat), written out
    /// (cf_aipw_formula).
    #[test]
    fn test_validate_cf_aipw_formula() {
        let (case, ds) = crate::test_support::load_case("cf_aipw_formula");
        let c = case_columns(&ds, &["y", "w", "y_hat", "w_hat", "tau_hat"], 40);
        let v = |k: usize| Array1::from(c[k].clone());
        let scores = aipw_scores(
            v(0).view(),
            v(1).view(),
            v(2).view(),
            v(3).view(),
            v(4).view(),
        )
        .unwrap();
        let (est, se) = aipw_ate(&scores).unwrap();
        let (sub_est, sub_se) = aipw_ate(&scores[..15]).unwrap();
        Check::new(&case)
            .vector("scores", &scores, Tol::coef(40))
            .scalar("estimate", est, Tol::coef(40))
            .scalar("std_err", se, Tol::se(40))
            .scalar("subset_estimate", sub_est, Tol::coef(40))
            .scalar("subset_std_err", sub_se, Tol::se(40))
            .finish();
    }

    #[test]
    fn test_aipw_scores_refuse_invalid_inputs() {
        let a = |v: &[f64]| Array1::from(v.to_vec());
        let (y, w, yh, th) = (
            a(&[1.0, 2.0, 3.0]),
            a(&[0.0, 1.0, 1.0]),
            a(&[1.5, 2.0, 2.5]),
            a(&[1.0, 1.0, 1.0]),
        );
        let err = |wh: &[f64]| {
            aipw_scores(y.view(), w.view(), yh.view(), a(wh).view(), th.view())
                .unwrap_err()
                .to_string()
        };
        assert!(
            err(&[0.5, 1.0, 0.5]).contains("w_hat"),
            "W.hat = 1 divides by zero"
        );
        assert!(err(&[0.0, 0.5, 0.5]).contains("w_hat"));
        assert!(err(&[0.5, f64::NAN, 0.5]).contains("finite"));
        assert!(
            aipw_scores(
                y.view(),
                a(&[0.0, 0.5, 1.0]).view(),
                yh.view(),
                a(&[0.5; 3]).view(),
                th.view()
            )
            .unwrap_err()
            .to_string()
            .contains("binary")
        );
        // length
        assert!(
            aipw_scores(
                y.view(),
                w.view(),
                yh.view(),
                a(&[0.5; 2]).view(),
                th.view()
            )
            .is_err()
        );
        assert!(aipw_ate(&[1.0]).is_err()); // one score: no SE
        assert!(aipw_ate(&[1.0, f64::INFINITY]).is_err());
    }

    // ---- Task 3: the orthogonalised forest (property tests on R-generated designs) ----

    fn names(p: usize) -> Vec<String> {
        (0..p).map(|j| format!("x{j}")).collect()
    }

    /// The test configuration: `n_trees`, `seed` and mtry = p (the thresholds were calibrated
    /// with every covariate a candidate, whatever the default).
    fn cfg(n_trees: usize, seed: u64, p: usize) -> CausalForestConfig {
        let mut c = CausalForestConfig::default();
        c.n_trees = n_trees;
        c.seed = Some(seed);
        c.mtry = Some(p);
        c
    }

    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len() as f64
    }

    fn quantile7(v: &[f64], p: f64) -> f64 {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        crate::stats::robust::quantile_sorted(&s, p, crate::stats::robust::QuantileType::T7)
    }

    /// cf_confounded: X = (x0, x1, x2), the outcome column `y`, W. Callers name the case id.
    fn confounded(
        case_id: &str,
        y: &str,
    ) -> (p2a_validation::Case, Array2<f64>, Array1<f64>, Array1<f64>) {
        let (case, ds) = crate::test_support::load_case(case_id);
        let c = case_columns(&ds, &["x0", "x1", "x2", "w", y], 1000);
        (
            case,
            matrix(&c[..3]),
            Array1::from(c[4].clone()),
            Array1::from(c[3].clone()),
        )
    }

    /// ML-20: confounded by x0, no effect. Before: about the naive difference 2.54 (no
    /// centering; splits on raw Y). Property test (grf not installed); emulation, 24 seeds: ATE
    /// in [-0.039, 0.009], SE in [0.0710, 0.0755].
    #[test]
    fn test_validate_cf_confounded_zero_effect_covers_zero() {
        let (case, x, y, w) = confounded("cf_confounded", "y0");
        let r = causal_forest_arrays(x.view(), y.view(), w.view(), names(3), cfg(400, 4701, 3))
            .unwrap();
        let naive = case.scalar("naive_y0"); // 2.536097700229202
        let oracle_se = case.scalar("oracle_y0_se"); // 0.0738093632543377
        assert!(r.ate.abs() < 0.15, "ate {} (naive {naive})", r.ate);
        assert!(
            naive - r.ate > 20.0 * r.ate_se,
            "ate {} se {}",
            r.ate,
            r.ate_se
        );
        assert!(
            r.ate.abs() <= 1.9599639845400536 * r.ate_se,
            "95% CI excludes 0: ate {} se {}",
            r.ate,
            r.ate_se
        );
        assert!(r.ate_p_value > 0.05);
        let ratio = r.ate_se / oracle_se;
        assert!(
            (0.8..=1.25).contains(&ratio),
            "se {} oracle {oracle_se}",
            r.ate_se
        );
        assert!(mean(&r.predictions).abs() < 0.1);
    }

    /// ML-20: tau = 1, same confounding. Before: about the naive 3.54. Emulation: ATE
    /// [0.961, 0.997], mean CATE [0.970, 1.008], q90 - q10 of the CATEs 0.31-0.33.
    #[test]
    fn test_validate_cf_confounded_constant_effect_recovered() {
        let (case, x, y, w) = confounded("cf_confounded", "y1");
        let r = causal_forest_arrays(x.view(), y.view(), w.view(), names(3), cfg(400, 4702, 3))
            .unwrap();
        assert!((r.ate - 1.0).abs() <= 0.15, "ate {}", r.ate);
        assert!(
            (r.ate - 1.0).abs() <= 2.5758293035489004 * r.ate_se,
            "ate {} se {}",
            r.ate,
            r.ate_se
        );
        assert!(case.scalar("naive_y1") - r.ate > 20.0 * r.ate_se);
        assert!((mean(&r.predictions) - 1.0).abs() <= 0.15);
        assert!(
            quantile7(&r.predictions, 0.9) - quantile7(&r.predictions, 0.1) < 0.6,
            "spurious heterogeneity"
        );
    }

    /// ML-20 (CATEs): tau = 1 + 2 x1, confounded by x0. Emulation: ATE [1.937, 2.005] (SATE
    /// 1.98548), cor(CATE, tau) >= 0.94, x1 the most important; new points tau(0.5, 0.1, 0.5) =
    /// 1.2 -> [1.127, 1.215], tau(0.5, 0.9, 0.5) = 2.8 -> [2.672, 2.830].
    #[test]
    fn test_validate_cf_confounded_heterogeneous_effect_tracks_tau() {
        let (case, x, y, w) = confounded("cf_confounded", "yh");
        let r = causal_forest_arrays(x.view(), y.view(), w.view(), names(3), cfg(400, 4703, 3))
            .unwrap();
        assert!(
            (r.ate - case.scalar("sate_yh")).abs() <= 0.15,
            "ate {}",
            r.ate
        );
        let tau: Vec<f64> = x.column(1).iter().map(|v| 1.0 + 2.0 * v).collect();
        let (mp, mt) = (mean(&r.predictions), mean(&tau));
        let cov: f64 = r
            .predictions
            .iter()
            .zip(&tau)
            .map(|(p, t)| (p - mp) * (t - mt))
            .sum();
        let sp = r
            .predictions
            .iter()
            .map(|p| (p - mp).powi(2))
            .sum::<f64>()
            .sqrt();
        let st = tau.iter().map(|t| (t - mt).powi(2)).sum::<f64>().sqrt();
        assert!(cov / (sp * st) >= 0.8, "cor {}", cov / (sp * st));
        assert_eq!(
            r.variable_importance[0].0, "x1",
            "{:?}",
            r.variable_importance
        );
        let new = array![[0.5, 0.1, 0.5], [0.5, 0.9, 0.5]];
        let p = causal_forest_predict_arrays(&r, new.view()).unwrap();
        assert!(
            (p[0] - 1.2).abs() <= 0.4 && (p[1] - 2.8).abs() <= 0.4,
            "{p:?}"
        );
        assert!(p[1] - p[0] >= 1.0, "{p:?}");
    }

    /// ML-21: randomised design. Before: SE 0.51 (n = 500), 0.45 (n = 2000), i.e. 5.7 and 10
    /// times the Welch SE. Emulation: SE / Welch SE 1.02-1.06 (500) and 1.02-1.04 (2000);
    /// SE(2000) / SE(500) in [0.488, 0.502].
    #[test]
    fn test_validate_cf_rct_ate_se_shrinks_with_n() {
        let (case, ds) = crate::test_support::load_case("cf_rct");
        let mut se = Vec::new();
        for (rows, welch, oracle_est, oracle_se) in [
            (500, "dim_se_500", "oracle_est_500", "oracle_se_500"),
            (2000, "dim_se_2000", "oracle_est_2000", "oracle_se_2000"),
        ] {
            let c = case_columns(&ds, &["x0", "x1", "w", "y"], rows);
            let x = matrix(&c[..2]);
            let (w, y) = (Array1::from(c[2].clone()), Array1::from(c[3].clone()));
            let r = causal_forest_arrays(x.view(), y.view(), w.view(), names(2), cfg(200, 4705, 2))
                .unwrap();
            let ratio = r.ate_se / case.scalar(welch);
            assert!(
                (0.85..=1.3).contains(&ratio),
                "n {rows}: se {} welch {}",
                r.ate_se,
                case.scalar(welch)
            );
            assert!(
                (r.ate - case.scalar(oracle_est)).abs() <= 0.75 * case.scalar(oracle_se),
                "n {rows}: ate {}",
                r.ate
            );
            se.push(r.ate_se);
        }
        assert!((0.4..=0.6).contains(&(se[1] / se[0])), "se {se:?}"); // sqrt(500/2000) = 0.5
    }

    /// ML-74: 10 of 200 treated, tau = 5. Before: per-tree leaf contrasts of 0 for estimation
    /// halves without a treated unit shrank every CATE (expected 5 (1 - 0.75^10) = 4.718; repro
    /// 4.83). Emulation: mean CATE [4.973, 5.059], min CATE >= 4.918, ATE [4.911, 5.024], W.hat
    /// min [0.019, 0.033].
    #[test]
    fn test_validate_cf_rare_treatment_is_not_shrunk() {
        let (case, ds) = crate::test_support::load_case("cf_rare");
        let c = case_columns(&ds, &["x0", "x1", "w", "y"], 200);
        let x = matrix(&c[..2]);
        let (w, y) = (Array1::from(c[2].clone()), Array1::from(c[3].clone()));
        let r = causal_forest_arrays(x.view(), y.view(), w.view(), names(2), cfg(400, 4706, 2))
            .unwrap();
        let m = mean(&r.predictions);
        assert!((m - 5.0).abs() <= 0.1, "mean CATE {m}");
        assert!(
            m - case.scalar("shrunk_by_empty_arm") > 0.15,
            "mean CATE {m}"
        );
        let min = r.predictions.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(min >= 4.85, "min CATE {min}");
        assert!(
            (r.ate - 5.0).abs() <= 0.15 && (r.ate - 5.0).abs() <= 2.5758293035489004 * r.ate_se,
            "ate {} se {}",
            r.ate,
            r.ate_se
        );
        assert!(
            r.warnings
                .iter()
                .any(|t| t.starts_with("Estimated treatment propensities go as low as")),
            "{:?}",
            r.warnings
        );
        assert!(
            r.to_string()
                .contains("Warning: Estimated treatment propensities go as low as")
        );
    }

    /// ML-76 (side effect): oob_error is the out-of-bag R-loss. Before, the across-tree variance
    /// of OOB predictions was smallest for a root-only forest. Emulation (paired, 200 trees):
    /// R-loss 1.118-1.143 (depth 0) against 1.043-1.075 (depth 10), difference >= 0.055. The
    /// nuisance forests do not depend on max_depth.
    #[test]
    fn test_validate_cf_confounded_oob_error_is_the_r_loss() {
        let (_, x, y, w) = confounded("cf_confounded", "yh");
        let deep = causal_forest_arrays(x.view(), y.view(), w.view(), names(3), cfg(200, 4707, 3))
            .unwrap();
        let mut c0 = cfg(200, 4707, 3);
        c0.max_depth = 0;
        let shallow = causal_forest_arrays(x.view(), y.view(), w.view(), names(3), c0).unwrap();
        assert_eq!(deep.y_hat, shallow.y_hat);
        assert!(
            shallow.oob_error - deep.oob_error > 0.02,
            "{} vs {}",
            shallow.oob_error,
            deep.oob_error
        );
        assert!((0.9..=1.25).contains(&deep.oob_error), "{}", deep.oob_error);
    }

    /// grf sample_poisson(mtry) clamped to [1, p] (create_split_variable_subset).
    #[test]
    fn test_poisson_count_matches_clamped_poisson_mean() {
        let mut rng = 99u64;
        let draws = |lambda: f64, cap: usize, k: usize, rng: &mut u64| -> Vec<usize> {
            (0..k).map(|_| poisson_count(lambda, cap, rng)).collect()
        };
        for (lambda, cap, k, mean, sd) in [
            (3.0, 10, 20_000, 3.04940297348399, 1.65608812440467),
            (1.0, 3, 20_000, 1.34454251472851, 0.621639577581618),
            // a product-of-uniforms sampler underflows here
            (800.0, 1000, 2_000, 800.0, 28.2842712473095),
        ] {
            let d = draws(lambda, cap, k, &mut rng);
            assert!(d.iter().all(|&v| (1..=cap).contains(&v)));
            let m = d.iter().sum::<usize>() as f64 / k as f64;
            assert!(
                (m - mean).abs() <= 5.0 * sd / (k as f64).sqrt(),
                "lambda {lambda}: mean {m}, expected {mean}"
            );
        }
        assert!(draws(23.0, 3, 100, &mut rng).iter().all(|&v| v == 3));
    }

    fn generate_test_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let mut rng = seed;

        let mut x = Array2::zeros((n, 3));
        let mut y = Array1::zeros(n);
        let mut w = Array1::zeros(n);

        for i in 0..n {
            // Generate covariates
            x[[i, 0]] = (lcg_random(&mut rng) % 100) as f64 / 100.0;
            x[[i, 1]] = (lcg_random(&mut rng) % 100) as f64 / 100.0;
            x[[i, 2]] = (lcg_random(&mut rng) % 100) as f64 / 100.0;

            // Treatment depends on x0
            w[i] = if x[[i, 0]] + (lcg_random(&mut rng) % 100) as f64 / 200.0 > 0.5 {
                1.0
            } else {
                0.0
            };

            // True CATE: tau(x) = 1 + 2*x0 (heterogeneous treatment effect)
            let tau = 1.0 + 2.0 * x[[i, 0]];

            // Outcome with treatment effect and noise
            let noise = ((lcg_random(&mut rng) % 100) as f64 - 50.0) / 100.0;
            y[i] = 5.0 + x[[i, 0]] + 0.5 * x[[i, 1]] + tau * w[i] + noise;
        }

        (x, y, w)
    }

    /// grf causal_forest(Y.hat = , W.hat = ): with the true nuisances the AIPW ATE is the
    /// oracle's up to the CATE noise. Emulation: ATE [0.9937, 0.9979] (oracle 1.00098), SE
    /// [0.0742, 0.0745] (oracle 0.07381).
    #[test]
    fn test_validate_cf_confounded_supplied_nuisance_matches_oracle_aipw() {
        let (case, x, y, w) = confounded("cf_confounded", "y1");
        let e: Array1<f64> = x.column(0).mapv(|v| 0.1 + 0.8 * v);
        let m: Array1<f64> = x.column(0).mapv(|v| 10.0 * v) + &e;
        let r = causal_forest_arrays_with_nuisance(
            x.view(),
            y.view(),
            w.view(),
            Some(m.view()),
            Some(e.view()),
            names(3),
            cfg(400, 4704, 3),
        )
        .unwrap();
        assert_eq!(r.y_hat, m.to_vec());
        assert_eq!(r.w_hat, e.to_vec());
        assert_eq!((r.y_hat_trees, r.w_hat_trees), (0, 0));
        let (est, se) = (case.scalar("oracle_y1_est"), case.scalar("oracle_y1_se"));
        assert!(
            (r.ate - est).abs() <= 0.25 * se,
            "ate {} oracle {est}",
            r.ate
        );
        assert!(
            (r.ate_se / se - 1.0).abs() <= 0.05,
            "se {} oracle {se}",
            r.ate_se
        );
        assert_eq!(aipw_ate(&r.aipw_scores).unwrap(), (r.ate, r.ate_se));
        assert!(r.to_string().contains("Y.hat: supplied"));
    }

    /// Overlap DGP for the unit tests: x ~ U(0,1)^3 (module LCG), e(x) = 0.2 + 0.6 x0,
    /// W ~ Bernoulli(e), tau(x) = 1 + 2 x0, Y = 5 + x0 + 0.5 x1 + tau W + 0.5 N(0,1) (Box-Muller).
    fn overlap_test_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>, Vec<f64>) {
        let mut rng = seed;
        let (mut x, mut y, mut w, mut tau) = (
            Array2::zeros((n, 3)),
            Array1::zeros(n),
            Array1::zeros(n),
            vec![0.0; n],
        );
        for i in 0..n {
            for j in 0..3 {
                x[[i, j]] = unit_uniform(&mut rng);
            }
            w[i] = if unit_uniform(&mut rng) < 0.2 + 0.6 * x[[i, 0]] {
                1.0
            } else {
                0.0
            };
            tau[i] = 1.0 + 2.0 * x[[i, 0]];
            let (u1, u2) = (unit_uniform(&mut rng), unit_uniform(&mut rng));
            let eps = (-2.0 * (1.0 - u1).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            y[i] = 5.0 + x[[i, 0]] + 0.5 * x[[i, 1]] + tau[i] * w[i] + 0.5 * eps;
        }
        (x, y, w, tau)
    }

    /// Emulation of this design (calib_H, 12 seeds, n = 300, 100 trees): |ATE - SATE| <= 0.188,
    /// SE in [0.073, 0.097], W.hat in [0.104, 0.914], every variance estimate finite.
    #[test]
    fn test_causal_forest_basic() {
        let (x, y, w, tau) = overlap_test_data(300, 42);
        let r =
            causal_forest_arrays(x.view(), y.view(), w.view(), names(3), cfg(100, 42, 3)).unwrap();
        assert_eq!(
            (
                r.n_obs,
                r.n_trees,
                r.predictions.len(),
                r.variance_estimates.len()
            ),
            (300, 100, 300, 300)
        );
        assert_eq!((r.y_hat_trees, r.w_hat_trees), (50, 50));
        assert!(
            r.variance_estimates
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
        );
        assert!(r.w_hat.iter().all(|e| *e > 0.0 && *e < 1.0));
        assert!(
            (r.ate - mean(&tau)).abs() <= 0.35,
            "ate {} sate {}",
            r.ate,
            mean(&tau)
        );
        assert!(r.ate_se > 0.03 && r.ate_se < 0.2, "se {}", r.ate_se);
        assert_eq!(r.aipw_scores.len(), 300);
        let text = r.to_string();
        for needle in ["AIPW", "Out-of-bag R-loss", "W.hat range", "z = "] {
            assert!(text.contains(needle), "{needle}");
        }
    }

    #[test]
    fn test_causal_forest_no_honesty() {
        let (x, y, w, tau) = overlap_test_data(300, 123);
        let mut c = cfg(100, 123, 3);
        c.honesty = false;
        let r = causal_forest_arrays(x.view(), y.view(), w.view(), names(3), c).unwrap();
        assert!((r.ate - mean(&tau)).abs() <= 0.35, "ate {}", r.ate);
        assert!(r.predictions.iter().all(|p| p.is_finite()));
    }

    /// Replaces the T0 golden of the old algorithm (its outputs change by design).
    #[test]
    fn test_causal_forest_is_deterministic_per_seed() {
        let (x, y, w, _) = overlap_test_data(120, 7);
        let fit = |seed: Option<u64>| {
            let mut c = cfg(60, 0, 3);
            c.seed = seed;
            causal_forest_arrays(x.view(), y.view(), w.view(), names(3), c).unwrap()
        };
        let bits = |r: &CausalForestResult| -> Vec<u64> {
            [r.ate, r.ate_se, r.oob_error]
                .into_iter()
                .chain(r.predictions.iter().copied())
                .chain(r.variance_estimates.iter().copied())
                .chain(r.y_hat.iter().copied())
                .chain(r.w_hat.iter().copied())
                .map(f64::to_bits)
                .collect()
        };
        let (a, b) = (fit(Some(7)), fit(Some(7)));
        assert_eq!(bits(&a), bits(&b));
        assert_eq!(bits(&fit(None)), bits(&fit(Some(42))));
        assert_ne!(fit(Some(8)).ate.to_bits(), a.ate.to_bits());
    }

    #[test]
    fn test_causal_forest_ci_group_size_one_has_no_variance_estimates() {
        let (x, y, w, _) = overlap_test_data(120, 11);
        let mut c = cfg(61, 11, 3);
        c.ci_group_size = 1;
        c.sample_fraction = 0.7; // allowed without little bags
        let r = causal_forest_arrays(x.view(), y.view(), w.view(), names(3), c).unwrap();
        assert_eq!(r.n_trees, 61);
        assert!(r.variance_estimates.iter().all(|v| v.is_nan()));
        assert!(
            r.warnings.iter().any(|t| t.contains("ci_group_size")),
            "{:?}",
            r.warnings
        );
        assert!(r.ate.is_finite() && r.ate_se.is_finite());
        // with groups of 2, 61 trees round down to 60 and say so
        let r2 =
            causal_forest_arrays(x.view(), y.view(), w.view(), names(3), cfg(61, 11, 3)).unwrap();
        assert_eq!(r2.n_trees, 60);
        assert!(
            r2.warnings
                .iter()
                .any(|t| t.contains("n_trees = 61") && t.contains("60")),
            "{:?}",
            r2.warnings
        );
    }

    #[test]
    fn test_causal_forest_rejects_invalid_arrays() {
        let (x, y, mut w, _) = overlap_test_data(120, 5);
        let msg = |r: EconResult<CausalForestResult>| r.unwrap_err().to_string();
        let ok = || cfg(60, 5, 3);
        assert!(
            msg(causal_forest_arrays(
                x.view(),
                y.view(),
                w.view(),
                names(2),
                ok()
            ))
            .contains("covariate_names")
        );
        let mut c = ok();
        c.mtry = Some(4);
        assert!(
            msg(causal_forest_arrays(
                x.view(),
                y.view(),
                w.view(),
                names(3),
                c
            ))
            .contains("mtry must be at most the number of covariates (3)")
        );
        let x0 = Array2::<f64>::zeros((120, 0));
        let mut c = ok();
        c.mtry = None;
        assert!(
            msg(causal_forest_arrays(
                x0.view(),
                y.view(),
                w.view(),
                vec![],
                c
            ))
            .contains("at least one covariate")
        );
        let short = Array1::from(vec![0.5; 10]);
        assert!(
            msg(causal_forest_arrays_with_nuisance(
                x.view(),
                y.view(),
                w.view(),
                Some(short.view()),
                None,
                names(3),
                ok()
            ))
            .contains("y_hat has 10 values")
        );
        let mut e = Array1::from(vec![0.5; 120]);
        e[3] = 1.0;
        assert!(
            msg(causal_forest_arrays_with_nuisance(
                x.view(),
                y.view(),
                w.view(),
                None,
                Some(e.view()),
                names(3),
                ok()
            ))
            .contains("w_hat must be strictly between 0 and 1")
        );
        e[3] = f64::NAN;
        assert!(
            msg(causal_forest_arrays_with_nuisance(
                x.view(),
                y.view(),
                w.view(),
                None,
                Some(e.view()),
                names(3),
                ok()
            ))
            .contains("finite")
        );
        w[0] = 0.5; // was counted as control and fitted
        assert!(
            msg(causal_forest_arrays(
                x.view(),
                y.view(),
                w.view(),
                names(3),
                ok()
            ))
            .contains("w must be binary")
        );
    }

    #[test]
    fn test_causal_forest_predict_checks_columns() {
        let (x, y, w, _) = overlap_test_data(120, 9);
        let r =
            causal_forest_arrays(x.view(), y.view(), w.view(), names(3), cfg(60, 9, 3)).unwrap();
        assert!(
            causal_forest_predict_arrays(&r, Array2::zeros((2, 2)).view())
                .unwrap_err()
                .to_string()
                .contains("3 covariates")
        );
        assert!(causal_forest_predict_arrays(&r, array![[0.5, f64::NAN, 0.5]].view()).is_err());
        assert!(
            causal_forest_predict_arrays(&r, array![[0.5, 0.5, 0.5]].view()).unwrap()[0]
                .is_finite()
        );
    }

    /// T2 1.6 (forests: Fail): a missing covariate is NullValues naming it (was ColumnNotFound
    /// for any problem).
    #[test]
    fn test_causal_forest_dataset_wrapper_reports_missing_values() {
        use polars::prelude::*;
        let n = 12;
        let x1: Vec<Option<f64>> = (0..n)
            .map(|i| if i == 4 { None } else { Some(i as f64 / 7.0) })
            .collect();
        let df = df! {
            "y" => (0..n).map(|i| (i as f64 * 0.37).sin()).collect::<Vec<f64>>(),
            "w" => (0..n).map(|i| (i % 2) as i64).collect::<Vec<i64>>(),
            "x0" => (0..n).map(|i| (i as f64 * 0.61).cos()).collect::<Vec<f64>>(),
            "x1" => x1,
        }
        .unwrap();
        let ds = Dataset::new(df);
        let r = causal_forest(&ds, "y", "w", &["x0", "x1"], cfg(20, 1, 2));
        assert!(
            matches!(&r, Err(EconError::NullValues { column, .. }) if column == "x1"),
            "{:?}",
            r.err()
        );
        let r = causal_forest(&ds, "y", "w", &["x0", "nope"], cfg(20, 1, 2));
        assert!(
            matches!(&r, Err(EconError::ColumnNotFound { column, .. }) if column == "nope"),
            "{:?}",
            r.err()
        );
    }

    #[test]
    fn test_causal_tree_honest_split() {
        let indices: Vec<usize> = (0..100).collect();
        let mut rng = 42u64;

        let (structure, estimation) = honest_split(&indices, 0.5, &mut rng);

        // Check sizes
        assert!(
            estimation.len() >= 45 && estimation.len() <= 55,
            "Estimation size: {}",
            estimation.len()
        );
        assert_eq!(structure.len() + estimation.len(), 100);

        // Check no overlap
        let est_set: std::collections::HashSet<_> = estimation.iter().collect();
        for &idx in &structure {
            assert!(!est_set.contains(&idx));
        }
    }

    #[test]
    fn test_insufficient_data() {
        let x = Array2::zeros((5, 2));
        let y = Array1::zeros(5);
        let w = array![0.0, 0.0, 1.0, 1.0, 1.0];

        let config = CausalForestConfig::default();

        let result = causal_forest_arrays(
            x.view(),
            y.view(),
            w.view(),
            vec!["x0".to_string(), "x1".to_string()],
            config,
        );

        assert!(result.is_err());
    }

    // ---- Task 4: average_treatment_effect on the AIPW scores ----

    /// ML-22: the helper reported sd(CATE)/sqrt(n): a 95% half-width of about 3% of the valid
    /// one. Now the AIPW scores. Emulation: SE / Welch SE 1.02-1.06 at n = 500.
    #[test]
    fn test_validate_cf_rct_average_treatment_effect_uses_the_aipw_scores() {
        let (case, ds) = crate::test_support::load_case("cf_rct");
        let c = case_columns(&ds, &["x0", "x1", "w", "y"], 500);
        let x = matrix(&c[..2]);
        let (w, y) = (Array1::from(c[2].clone()), Array1::from(c[3].clone()));
        let r = causal_forest_arrays(x.view(), y.view(), w.view(), names(2), cfg(200, 4708, 2))
            .unwrap();
        let (ate, lo, hi) = average_treatment_effect(&r, None).unwrap();
        assert!((ate - r.ate).abs() <= 1e-12 * r.ate.abs().max(1.0));
        let half = (hi - lo) / 2.0;
        let valid = 1.9599639845400536 * case.scalar("dim_se_500"); // 0.1732...
        assert!(
            (0.85..=1.3).contains(&(half / valid)),
            "half-width {half} vs {valid}"
        );
        let sub: Vec<usize> = (0..250).collect();
        let est = average_treatment_effect_with(&r, Some(&sub), 0.9).unwrap();
        let (m, se) = aipw_ate(&r.aipw_scores[..250]).unwrap();
        assert_eq!((est.estimate, est.std_err, est.n), (m, se, 250));
        assert!((est.ci_upper - (m + 1.6448536269514715 * se)).abs() <= 1e-12);
        assert!((est.ci_lower - (m - 1.6448536269514715 * se)).abs() <= 1e-12);
    }

    #[test]
    fn test_average_treatment_effect_rejects_bad_subsets() {
        let (x, y, w, _) = overlap_test_data(120, 13);
        let r =
            causal_forest_arrays(x.view(), y.view(), w.view(), names(3), cfg(60, 13, 3)).unwrap();
        let e = |s: &[usize]| {
            average_treatment_effect(&r, Some(s))
                .unwrap_err()
                .to_string()
        };
        assert!(e(&[0, 120]).contains("out of range")); // panicked: index out of bounds
        assert!(e(&[3, 3]).contains("more than once"));
        assert!(average_treatment_effect(&r, Some(&[5])).is_err()); // one unit: no SE
        assert!(average_treatment_effect(&r, Some(&[])).is_err());
        assert!(average_treatment_effect_with(&r, None, 1.2).is_err());
        let mut stale = r.clone();
        stale.aipw_scores.clear(); // deserialized from JSON written before the AIPW scores
        assert!(
            average_treatment_effect(&stale, None)
                .unwrap_err()
                .to_string()
                .contains("AIPW scores")
        );
    }

    // ---- Task 5: calibration over seeded replications ----

    /// Calibration over 50 seeded replications (release only: 50 forests of 200 trees). DGP:
    /// n = 500, x ~ U(0,1)^3, e = 0.1 + 0.8 x0, Y = 10 x0 + W + N(0, 1) (tau = 1). Planning
    /// emulation, 3 x 50 replications: coverage 50, 45, 48 of 50; mean(SE) / sd(ATE) 1.023,
    /// 0.799, 0.994; mean ATE 1.0018, 1.0036, 1.0245; mean(variance) / mean((tau_hat - 1)^2)
    /// 1.71, 1.39, 1.48; CATE 95% interval coverage 0.959, 0.939, 0.946.
    /// pbinom(41, 50, 0.95) = 0.00076. A replication refused for an out-of-bag W.hat of 0 or 1
    /// counts as not covered (1 of the 50 here); at most 2 may be.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: 50 causal forests; run with --release"
    )]
    fn test_causal_forest_ate_ci_coverage_over_replications() {
        let reps = 50;
        let (mut ates, mut ses, mut covered, mut refused) = (Vec::new(), Vec::new(), 0, 0);
        let (mut var_sum, mut mse_sum, mut cate_in, mut cate_n) = (0.0, 0.0, 0usize, 0usize);
        for rep in 0..reps {
            let mut rng = 9_000 + rep as u64;
            let n = 500;
            let (mut x, mut y, mut w) = (Array2::zeros((n, 3)), Array1::zeros(n), Array1::zeros(n));
            for i in 0..n {
                for j in 0..3 {
                    x[[i, j]] = unit_uniform(&mut rng);
                }
                w[i] = if unit_uniform(&mut rng) < 0.1 + 0.8 * x[[i, 0]] {
                    1.0
                } else {
                    0.0
                };
                let (u1, u2) = (unit_uniform(&mut rng), unit_uniform(&mut rng));
                let eps = (-2.0 * (1.0 - u1).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                y[i] = 10.0 * x[[i, 0]] + w[i] + eps;
            }
            let r = match causal_forest_arrays(
                x.view(),
                y.view(),
                w.view(),
                names(3),
                cfg(200, rep as u64, 3),
            ) {
                Ok(r) => r,
                // An out-of-bag W.hat of exactly 0 or 1 (all of a unit's out-of-bag leaves in
                // one arm, near x0 = 0 or 1) is refused, where grf returns a NaN ATE. Such a
                // replication counts as an interval that did not cover.
                Err(EconError::InvalidSpecification { message })
                    if message.contains("estimated propensity W.hat is 0 or 1") =>
                {
                    refused += 1;
                    continue;
                }
                Err(e) => panic!("replication {rep}: {e}"),
            };
            if (r.ate - 1.0).abs() <= 1.9599639845400536 * r.ate_se {
                covered += 1;
            }
            ates.push(r.ate);
            ses.push(r.ate_se);
            for (p, v) in r.predictions.iter().zip(&r.variance_estimates) {
                if v.is_finite() {
                    var_sum += v;
                    mse_sum += (p - 1.0).powi(2);
                    cate_n += 1;
                    if (p - 1.0).abs() <= 1.9599639845400536 * v.sqrt() {
                        cate_in += 1;
                    }
                }
            }
        }
        let m = mean(&ates);
        let sd = (ates.iter().map(|a| (a - m).powi(2)).sum::<f64>() / (reps as f64 - 1.0)).sqrt();
        let ratio = mean(&ses) / sd;
        let cover = cate_in as f64 / cate_n as f64;
        eprintln!(
            "coverage {covered}/50 ({refused} refused), mean ATE {m}, mean(se)/sd(ate) {ratio}, variance/MSE {}, \
             CATE coverage {cover}",
            var_sum / mse_sum
        );
        assert!(covered >= 42, "coverage {covered}/50 ({refused} refused)");
        assert!(
            refused <= 2,
            "{refused} of 50 replications refused for W.hat in {{0, 1}}"
        );
        assert!((m - 1.0).abs() <= 0.08, "mean ATE {m}");
        assert!((0.6..=1.6).contains(&ratio), "mean(se)/sd(ate) = {ratio}");
        assert!(
            (0.5..=3.0).contains(&(var_sum / mse_sum)),
            "variance/MSE = {}",
            var_sum / mse_sum
        );
        assert!((0.85..=0.995).contains(&cover), "CATE coverage {cover}");
    }
}
