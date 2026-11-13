//! Bayesian Additive Regression Trees (BART) for prediction with uncertainty.
//!
//! This module implements BART for both regression and classification, following the
//! methodology of Chipman, George, and McCulloch (2010). BART is a Bayesian
//! nonparametric regression approach using a sum-of-trees model with regularization
//! priors and MCMC sampling.
//!
//! # Model
//!
//! BART models the response as a sum of `m` regression trees:
//!
//! ```text
//! y = f(x) + epsilon = sum_{j=1}^m g_j(x; T_j, M_j) + epsilon
//! ```
//!
//! where each `g_j` is a tree with structure `T_j` and leaf parameters `M_j`,
//! and `epsilon ~ N(0, sigma^2)`.
//!
//! # Priors
//!
//! 1. **Tree structure prior**: P(node at depth d is nonterminal) = alpha * (1 + d)^(-beta)
//!    - Default: alpha = 0.95, beta = 2
//!    - This regularizes tree depth, making deeper trees less likely
//!
//! 2. **Leaf value prior**: mu_ij | T_j ~ N(0, tau^2), as `BART::wbart` calibrates it
//!    - y is centred by its mean (`offset`), and tau = (max(y) - min(y)) / (2 k sqrt(m)),
//!      fixed for the whole chain, so the prior of f(x) puts the range of y at about +-k SDs
//!    - Probit BART: tau = 3 / (k sqrt(m)) on the latent scale, as `BART::pbart`
//!
//! 3. **Residual variance prior**: nu lambda / sigma^2 ~ chi^2_nu
//!    - sigest is the residual SD of the linear fit of y on the covariates (constant columns
//!      dropped), or sd(y) when there are at least as many covariates as rows; when that fit is
//!      exact (sigest 0) p2a also falls back to sd(y), where wbart's chain would start at 0
//!    - lambda = sigest^2 qchisq(1 - q, nu) / nu, so P(sigma < sigest) = q under the prior
//!      (defaults nu = 3, q = 0.90)
//!    - A constant y is refused: the leaf prior SD would be 0
//!
//! # MCMC Algorithm
//!
//! The algorithm uses Bayesian backfitting with Gibbs sampling:
//!
//! 1. For j = 1,...,m: Draw (T_j, M_j) given all other trees and sigma, as `BART::wbart`
//!    - One Metropolis-Hastings birth (grow) or death (prune) move on the tree structure, with
//!      the Chipman-George-McCulloch (1998) acceptance ratio: the tree prior's
//!      p_d (1 - p_l)(1 - p_r) / (1 - p_d) term, the move and node-choice probabilities, and the
//!      leaf likelihood with mu integrated out
//!    - Split rules come from a fixed grid of cutpoints per covariate (`numcut`, as
//!      `BART::bartModelMatrix`); the split variable is drawn uniformly among the covariates
//!      with cutpoints, rows with `x < cut` go left, and a child with fewer than
//!      `min_node_size` rows rejects the birth (wbart's fixed 5); a node at `max_depth` cannot
//!      split, so the tree prior is truncated there (below 1e-6 of its mass at the default 10)
//!    - Draw leaf values from conjugate normal posterior
//!
//! 2. Draw sigma^2 from its conditional inverse-gamma posterior
//!
//! # References
//!
//! - Chipman, H. A., George, E. I., & McCulloch, R. E. (2010). BART: Bayesian
//!   Additive Regression Trees. *Annals of Applied Statistics*, 4(1), 266-298.
//!   <https://doi.org/10.1214/09-AOAS285>
//!
//! - R package `BART`: Sparapani, R., Spanbauer, C., & McCulloch, R. (2021).
//!   *Nonparametric Machine Learning and Efficient Computation with Bayesian
//!   Additive Regression Trees: The BART R Package*. JSS, 97(1).
//!   <https://CRAN.R-project.org/package=BART>
//!
//! R equivalent: `BART::wbart()` (regression), `BART::pbart()` (classification)

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::solve::least_squares;
use crate::resource::{checked_len, checkpoint, try_with_capacity, try_zeros2};
use crate::stats::robust::percentile_interval;
use crate::traits::estimator::{chi_squared_quantile, normal_quantile};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for BART model.
///
/// # Example
///
/// ```ignore
/// use p2a_core::ml::BartConfig;
///
/// let config = BartConfig {
///     n_trees: 200,
///     n_burn: 250,
///     n_mcmc: 1000,
///     k: 2.0,
///     alpha: 0.95,
///     beta: 2.0,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BartConfig {
    /// Number of trees in the sum (m). Default: 200.
    ///
    /// More trees generally improve prediction but increase computation.
    /// Chipman et al. (2010) recommend m=200 as a robust default.
    pub n_trees: usize,

    /// Number of MCMC burn-in iterations to discard. Default: 250.
    ///
    /// The burn-in period allows the chain to converge before collecting
    /// posterior samples.
    pub n_burn: usize,

    /// Number of posterior MCMC samples to draw. Default: 1000.
    ///
    /// More samples provide more precise uncertainty estimates but
    /// increase computation time.
    pub n_mcmc: usize,

    /// Prior parameter k for leaf value shrinkage. Default: 2.0.
    ///
    /// Controls the width of the prior on leaf values: mu ~ N(0, sigma_hat^2 / (k^2 * m)).
    /// Larger k = more shrinkage, smaller individual tree contributions.
    /// k=2 is recommended for most applications.
    pub k: f64,

    /// Tree prior alpha parameter. Default: 0.95.
    ///
    /// P(node is nonterminal | depth d) = alpha * (1 + d)^(-beta).
    /// Higher alpha allows deeper trees.
    pub alpha: f64,

    /// Tree prior beta parameter. Default: 2.0.
    ///
    /// Controls how quickly splitting probability decreases with depth.
    /// Higher beta = shallower trees.
    pub beta: f64,

    /// Prior degrees of freedom for sigma^2. Default: 3.0.
    ///
    /// sigma^2 ~ Inverse-Gamma(nu/2, nu*lambda/2).
    /// Values 3-10 are typical; nu < 3 may cause overfitting.
    pub nu: f64,

    /// Prior quantile for sigma calibration. Default: 0.90.
    ///
    /// Lambda is set so P(sigma < sigma_hat) = q under the prior.
    pub q: f64,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,

    /// Minimum observations per terminal node. Default: 5.
    pub min_node_size: usize,

    /// Maximum tree depth. Default: 10.
    pub max_depth: usize,

    /// Confidence level for prediction intervals. Default: 0.95.
    pub confidence_level: f64,

    /// Probit BART as `BART::pbart`: y coded 0/1, Albert-Chib latent draws, leaf prior SD
    /// 3 / (k sqrt(n_trees)); predictions are posterior mean probabilities. Default: false.
    pub classification: bool,

    /// Cutpoints per covariate, as `BART::bartModelMatrix(numcut, usequants = FALSE)`: a column
    /// with fewer distinct values gets the midpoints of its distinct values, any other column
    /// `numcut` equally spaced interior points of its range. Default: 100.
    #[serde(default = "default_numcut")]
    pub numcut: usize,
}

fn default_numcut() -> usize {
    100
}

impl Default for BartConfig {
    fn default() -> Self {
        Self {
            n_trees: 200,
            n_burn: 250,
            n_mcmc: 1000,
            k: 2.0,
            alpha: 0.95,
            beta: 2.0,
            nu: 3.0,
            q: 0.90,
            seed: None,
            min_node_size: 5,
            max_depth: 10,
            confidence_level: 0.95,
            classification: false,
            numcut: default_numcut(),
        }
    }
}

// ============================================================================
// Result Types
// ============================================================================

/// Result from BART model fitting.
///
/// Contains predictions with uncertainty quantification and variable importance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BartResult {
    /// Posterior mean predictions for training data.
    pub predictions: Vec<f64>,

    /// Lower bound of prediction intervals (e.g., 2.5th percentile).
    pub prediction_lower: Vec<f64>,

    /// Upper bound of prediction intervals (e.g., 97.5th percentile).
    pub prediction_upper: Vec<f64>,

    /// Posterior standard deviation of predictions.
    pub prediction_sd: Vec<f64>,

    /// Variable importance scores (based on inclusion frequency).
    ///
    /// Higher values indicate features that are more frequently used
    /// in tree splits across the MCMC samples.
    pub variable_importance: Vec<f64>,

    /// Feature names (if provided).
    pub feature_names: Option<Vec<String>>,

    /// Posterior mean of residual standard deviation sigma (probit BART: the fixed latent SD, 1).
    pub sigma: f64,

    /// Posterior standard deviation of sigma.
    pub sigma_sd: f64,

    /// The constant the trees are fitted around: mean(y) for regression (`BART::wbart`'s
    /// `fmean`, so the fit is offset + f(x)), qnorm(mean(y)) for probit BART (P(y = 1 | x) =
    /// Phi(offset + f(x))).
    #[serde(default)]
    pub offset: f64,

    /// The sigma estimate the prior is calibrated on: the residual SD of the linear fit of y on
    /// the covariates (sd(y) when there are at least as many covariates as rows, or when the
    /// linear fit is exact); 1 for probit BART.
    #[serde(default)]
    pub sigest: f64,

    /// Scale of the inverse chi-square prior on sigma^2 (nu lambda / sigma^2 ~ chi^2_nu),
    /// sigest^2 qchisq(1 - q, nu) / nu; 0 for probit BART, whose sigma is fixed at 1.
    #[serde(default)]
    pub lambda: f64,

    /// Prior SD of each leaf value (tau): (max(y) - min(y)) / (2 k sqrt(n_trees)) for
    /// regression, 3 / (k sqrt(n_trees)) for probit BART.
    #[serde(default)]
    pub leaf_prior_sd: f64,

    /// Number of observations.
    pub n_obs: usize,

    /// Number of features.
    pub n_features: usize,

    /// Number of trees.
    pub n_trees: usize,

    /// Number of MCMC samples kept.
    pub n_samples: usize,

    /// Configuration used.
    pub config: BartConfig,

    /// Full posterior samples of predictions (n_samples x n_obs).
    /// Skipped in serialization to save space.
    #[serde(skip)]
    pub posterior_samples: Option<Array2<f64>>,

    /// Posterior samples of sigma.
    #[serde(skip)]
    pub sigma_samples: Option<Vec<f64>>,
}

impl fmt::Display for BartResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "BART Model Results")?;
        writeln!(f, "==================")?;
        writeln!(
            f,
            "Observations: {}  |  Features: {}  |  Trees: {}",
            self.n_obs, self.n_features, self.n_trees
        )?;
        writeln!(
            f,
            "MCMC: {} burn-in + {} samples",
            self.config.n_burn, self.n_samples
        )?;
        writeln!(f)?;

        if self.config.classification {
            writeln!(
                f,
                "Probit BART: P(y = 1 | x) = Phi(offset + f(x)), offset = qnorm(mean y) = {:.4}, latent SD fixed at 1",
                self.offset
            )?;
            writeln!(f)?;
            writeln!(f, "PROBABILITY SUMMARY:")?;
        } else {
            writeln!(f, "RESIDUAL SD (sigma):")?;
            writeln!(
                f,
                "  Posterior mean: {:.4}  (SD: {:.4})",
                self.sigma, self.sigma_sd
            )?;
            writeln!(f)?;

            writeln!(f, "PREDICTION SUMMARY:")?;
        }
        let pred_mean: f64 = self.predictions.iter().sum::<f64>() / self.predictions.len() as f64;
        let pred_min = self
            .predictions
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let pred_max = self
            .predictions
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        writeln!(
            f,
            "  Mean: {:.4}  |  Min: {:.4}  |  Max: {:.4}",
            pred_mean, pred_min, pred_max
        )?;

        let avg_interval_width: f64 = self
            .prediction_upper
            .iter()
            .zip(self.prediction_lower.iter())
            .map(|(u, l)| u - l)
            .sum::<f64>()
            / self.n_obs as f64;
        writeln!(
            f,
            "  {:.0}% CI avg width: {:.4}",
            self.config.confidence_level * 100.0,
            avg_interval_width
        )?;
        writeln!(f)?;

        writeln!(f, "VARIABLE IMPORTANCE:")?;
        let mut indexed: Vec<(usize, f64)> = self
            .variable_importance
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

        for (rank, (i, importance)) in indexed.iter().take(10).enumerate() {
            let name = match &self.feature_names {
                Some(names) => names
                    .get(*i)
                    .cloned()
                    .unwrap_or_else(|| format!("X{}", i + 1)),
                None => format!("X{}", i + 1),
            };
            writeln!(f, "  {}. {}: {:.4}", rank + 1, name, importance)?;
        }
        if self.variable_importance.len() > 10 {
            writeln!(
                f,
                "  ... ({} more variables)",
                self.variable_importance.len() - 10
            )?;
        }

        Ok(())
    }
}

// ============================================================================
// Tree Data Structures
// ============================================================================

/// The child index of a leaf and the parent index of the root.
const NO_NODE: u32 = u32::MAX;

/// A node of a BART tree. A leaf has `left == right == NO_NODE`.
#[derive(Debug, Clone, Copy)]
struct BartNode {
    /// Parent index (`NO_NODE` for the root)
    parent: u32,
    /// Left child index (`NO_NODE` for a leaf)
    left: u32,
    /// Right child index (`NO_NODE` for a leaf)
    right: u32,
    /// Split variable (internal nodes)
    var: u32,
    /// Index of the split's cutpoint among the variable's cutpoints (internal nodes)
    cut: u32,
    /// Depth (the root is at 0)
    depth: u32,
    /// The split's cutpoint: rows with `x[var] < cutv` go left, as BART's `tree::bn`
    cutv: f64,
    /// Leaf value (leaves)
    mu: f64,
}

impl BartNode {
    fn leaf(parent: u32, depth: u32, mu: f64) -> Self {
        Self {
            parent,
            left: NO_NODE,
            right: NO_NODE,
            var: 0,
            cut: 0,
            depth,
            cutv: 0.0,
            mu,
        }
    }

    fn is_leaf(&self) -> bool {
        self.left == NO_NODE
    }
}

/// A single BART tree. `nodes` holds exactly the nodes reachable from the root (index 0):
/// a birth appends two leaves, and a death removes them, moving the last nodes into their
/// slots, so there are always `2 * internal + 1` nodes.
#[derive(Debug, Clone)]
struct BartTree {
    nodes: Vec<BartNode>,
}

impl BartTree {
    /// Create a new tree with single root node
    fn new(mu: f64) -> Self {
        Self {
            nodes: vec![BartNode::leaf(NO_NODE, 0, mu)],
        }
    }

    /// The leaf a row (its covariates in column order) falls in
    fn leaf_of(&self, row: &[f64]) -> usize {
        let mut i = 0;
        loop {
            let node = &self.nodes[i];
            if node.is_leaf() {
                return i;
            }
            i = if row[node.var as usize] < node.cutv {
                node.left as usize
            } else {
                node.right as usize
            };
        }
    }

    /// Node indices reached from the root, depth first, left before right
    fn walk(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.nodes.len());
        let mut stack = vec![0usize];
        while let Some(i) = stack.pop() {
            out.push(i);
            let node = &self.nodes[i];
            if !node.is_leaf() {
                stack.push(node.right as usize);
                stack.push(node.left as usize);
            }
        }
        out
    }

    /// Terminal node indices (leaves), walked from the root
    fn get_leaves(&self) -> Vec<usize> {
        self.walk()
            .into_iter()
            .filter(|&i| self.nodes[i].is_leaf())
            .collect()
    }

    /// Internal node indices, walked from the root
    fn get_internals(&self) -> Vec<usize> {
        self.walk()
            .into_iter()
            .filter(|&i| !self.nodes[i].is_leaf())
            .collect()
    }

    /// Internal nodes whose two children are leaves ("nog" nodes: the ones a death can remove)
    fn nogs(&self) -> Vec<usize> {
        self.get_internals()
            .into_iter()
            .filter(|&i| self.is_nog(i))
            .collect()
    }

    /// Whether node `i` is internal with two leaf children
    fn is_nog(&self, i: usize) -> bool {
        let node = &self.nodes[i];
        !node.is_leaf()
            && self.nodes[node.left as usize].is_leaf()
            && self.nodes[node.right as usize].is_leaf()
    }

    /// Node depth (stored)
    fn get_depth(&self, node_idx: usize) -> usize {
        self.nodes[node_idx].depth as usize
    }

    /// Split leaf `i` on `x[var] < cutv` (cutpoint `cut` of `var`); returns the new children
    fn birth(
        &mut self,
        i: usize,
        var: usize,
        cut: usize,
        cutv: f64,
        mu_left: f64,
        mu_right: f64,
    ) -> (usize, usize) {
        let left = self.nodes.len();
        let right = left + 1;
        let depth = self.nodes[i].depth + 1;
        self.nodes.push(BartNode::leaf(i as u32, depth, mu_left));
        self.nodes.push(BartNode::leaf(i as u32, depth, mu_right));
        let node = &mut self.nodes[i];
        node.left = left as u32;
        node.right = right as u32;
        node.var = var as u32;
        node.cut = cut as u32;
        node.cutv = cutv;
        (left, right)
    }

    /// Collapse the nog node `i` to a leaf with value `mu`, removing its two children. Returns
    /// where every old node index is now: the two children map to the collapsed node, and the
    /// nodes moved into the freed slots to those slots.
    fn death(&mut self, i: usize, mu: f64) -> Vec<u32> {
        let (l, r) = (self.nodes[i].left as usize, self.nodes[i].right as usize);
        {
            let node = &mut self.nodes[i];
            node.left = NO_NODE;
            node.right = NO_NODE;
            node.mu = mu;
        }
        let mut remap: Vec<u32> = (0..self.nodes.len() as u32).collect();
        remap[l] = i as u32;
        remap[r] = i as u32;
        for gone in [l.max(r), l.min(r)] {
            if let Some(from) = self.remove_slot(gone) {
                for t in remap.iter_mut() {
                    if *t as usize == from {
                        *t = gone as u32;
                    }
                }
            }
        }
        remap
    }

    /// Remove the (detached) node at `idx` by moving the last node into its slot and pointing
    /// that node's parent and children at the new slot. Returns the moved node's old index.
    fn remove_slot(&mut self, idx: usize) -> Option<usize> {
        let last = self.nodes.len() - 1;
        if idx == last {
            self.nodes.pop();
            return None;
        }
        let moved = self.nodes[last];
        self.nodes[idx] = moved;
        self.nodes.pop();
        let p = moved.parent as usize;
        if self.nodes[p].left as usize == last {
            self.nodes[p].left = idx as u32;
        } else {
            self.nodes[p].right = idx as u32;
        }
        if !moved.is_leaf() {
            self.nodes[moved.left as usize].parent = idx as u32;
            self.nodes[moved.right as usize].parent = idx as u32;
        }
        Some(last)
    }
}

// ============================================================================
// Prior calibration and cutpoints (BART::wbart)
// ============================================================================

/// `BART::wbart`'s data-driven prior for regression.
#[derive(Debug, Clone, Copy)]
struct WbartPrior {
    /// mean(y), the constant the trees are fitted around
    fmean: f64,
    /// The residual SD of the linear fit (or sd(y)) the sigma prior is calibrated on
    sigest: f64,
    /// sigest^2 qchisq(1 - q, nu) / nu
    lambda: f64,
    /// The leaf prior SD, (max(y) - min(y)) / (2 k sqrt(m))
    tau: f64,
}

/// Whether a column holds a single value (no cutpoint, dropped as `wbart`'s `rm.const`).
fn is_constant(col: ArrayView1<f64>) -> bool {
    col.iter().all(|&v| v == col[0])
}

/// `BART::wbart`'s prior calibration: fmean = mean(y); sigest the residual SD of
/// `lm(y ~ x)` with constant columns dropped, sd(y) when there are at least as many covariates
/// as rows; lambda = sigest^2 qchisq(1 - q, nu) / nu; tau = (max(y) - min(y)) / (2 k sqrt(m)).
/// A linear fit with no residual degrees of freedom or an exact one (sigest 0, where wbart's
/// chain would start at sigma 0) falls back to sd(y). A constant y is refused.
fn wbart_prior(
    y: ArrayView1<f64>,
    x: ArrayView2<f64>,
    config: &BartConfig,
) -> EconResult<WbartPrior> {
    let n = y.len();
    let (lo, hi) = y
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    if !(hi > lo) {
        return Err(EconError::InvalidSpecification {
            message: "y is constant: BART's leaf prior SD, (max(y) - min(y)) / (2 k sqrt(n_trees)), would be 0".to_string(),
        });
    }
    let fmean = y.sum() / n as f64;
    let sd = (y.iter().map(|&v| (v - fmean) * (v - fmean)).sum::<f64>() / (n - 1) as f64).sqrt();
    let active: Vec<usize> = (0..x.ncols())
        .filter(|&j| !is_constant(x.column(j)))
        .collect();
    let mut sigest = sd;
    if active.len() < n {
        let mut design = try_zeros2(n, active.len() + 1, "BART sigest design")?;
        design.column_mut(0).fill(1.0);
        for (k, &j) in active.iter().enumerate() {
            design.column_mut(k + 1).assign(&x.column(j));
        }
        let yc = y.mapv(|v| v - fmean);
        let fit = least_squares(design.view(), yc.view())?;
        if fit.df_residual > 0 {
            let rss = fit.residuals.iter().map(|r| r * r).sum::<f64>();
            let s = (rss / fit.df_residual as f64).sqrt();
            if s.is_finite() && s > 0.0 {
                sigest = s;
            }
        }
    }
    let lambda = sigest * sigest * chi_squared_quantile(1.0 - config.q, config.nu)? / config.nu;
    let tau = (hi - lo) / (2.0 * config.k * (config.n_trees as f64).sqrt());
    Ok(WbartPrior {
        fmean,
        sigest,
        lambda,
        tau,
    })
}

/// The cutpoints of one covariate, as `BART::bartModelMatrix(numcut, usequants = FALSE)`: none
/// for a constant column; the midpoints of the sorted distinct values when there are fewer
/// than `numcut` of them; otherwise `numcut` equally spaced interior points of
/// `[min, max]`, `min + j (max - min) / (numcut + 1)`, as R's `seq` computes them.
fn cutpoints(col: ArrayView1<f64>, numcut: usize) -> EconResult<Vec<f64>> {
    let mut xs = try_with_capacity::<f64>(col.len(), "BART cutpoints")?;
    xs.extend(col.iter().copied());
    xs.sort_by(f64::total_cmp);
    xs.dedup();
    let k = xs.len();
    if k <= 1 {
        return Ok(Vec::new());
    }
    if k < numcut {
        return Ok(xs.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect());
    }
    let (from, to) = (xs[0], xs[k - 1]);
    let by = (to - from) / (numcut as f64 + 1.0);
    let mut cuts = try_with_capacity::<f64>(numcut, "BART cutpoints")?;
    cuts.extend((1..=numcut).map(|j| from + j as f64 * by));
    Ok(cuts)
}

// ============================================================================
// MCMC Sampler
// ============================================================================

/// The fewest rows a leaf needs to be split: two children of `min_node_size` rows. It
/// saturates at `usize::MAX`, more rows than any leaf can hold, so a `min_node_size`
/// above `usize::MAX / 2` stops every split instead of overflowing; below that it is
/// exactly `2 * min_node_size`.
fn min_rows_to_split(min_node_size: usize) -> usize {
    min_node_size.saturating_mul(2)
}

/// BART MCMC sampler state
struct BartSampler {
    /// Current trees
    trees: Vec<BartTree>,
    /// Current residual SD
    sigma: f64,
    /// Leaf prior SD (tau), fixed for the chain
    sigma_mu: f64,
    /// Config
    config: BartConfig,
    /// RNG state
    rng: u64,
    /// Net births per variable, for variable importance
    var_counts: Vec<usize>,
    /// Number of features
    n_features: usize,
    /// Scale of the sigma prior (0 for probit BART)
    lambda: f64,
    /// Probit BART: sigma stays at 1 and the sweep's target is the latent draw
    probit: bool,
    /// mean(y) for regression, qnorm(mean(y)) for probit BART
    offset: f64,
    /// Cutpoints per covariate (empty for a constant column)
    cuts: Vec<Vec<f64>>,
    /// The covariates with at least one cutpoint, the ones a birth draws its variable from
    active: Vec<usize>,
    /// The current sum of the trees at every row
    allfit: Vec<f64>,
    /// Scratch: the fit of the tree being updated, before its update
    ftemp: Vec<f64>,
    /// Scratch: the partial residual of the tree being updated
    resid: Vec<f64>,
    /// Scratch: each row's leaf in the tree being updated
    leaf_of: Vec<u32>,
    /// Scratch: the sum of the updated trees, in tree order
    fresh: Vec<f64>,
    /// Scratch: row count and residual sum per node of the tree being updated
    node_n: Vec<usize>,
    node_sum: Vec<f64>,
}

/// Deepest tail draws: each accepts with probability above 0.7, so this bound is never reached by
/// a uniform source.
const TAIL_MAX_REJECTIONS: usize = 10_000;

/// R's pnorm. statrs' `Normal::cdf` errs near the 11th digit (pnorm(-1) comes out 1.36e-11 high):
/// its erfc carries the single-precision offsets of its rational fits as decimal doubles
/// (0.3440242112 for 0.34402421116828918...).
fn std_normal_cdf(x: f64) -> f64 {
    r_mathlib::normal_cdf(x, 0.0, 1.0, true, false)
}

/// A draw of z ~ N(mean, 1) truncated to z > 0 (`positive`) or z <= 0, by inversion, or by Robert's
/// (1995) exponential rejection sampler when the side has probability below 1e-100.
fn truncated_normal(
    mean: f64,
    positive: bool,
    uniform: &mut dyn FnMut() -> f64,
) -> EconResult<f64> {
    let side = std_normal_cdf(if positive { mean } else { -mean });
    if side >= 1e-100 {
        let u = uniform().clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        let q = normal_quantile((u * side).max(f64::MIN_POSITIVE))?;
        return Ok(if positive {
            (mean - q).max(f64::MIN_POSITIVE)
        } else {
            (mean + q).min(0.0)
        });
    }
    // The standardised tail beyond a = -mean (positive side) or a = mean (negative side)
    let a = if positive { -mean } else { mean };
    let lambda = (a + (a * a + 4.0).sqrt()) / 2.0;
    for _ in 0..TAIL_MAX_REJECTIONS {
        let u1 = uniform().clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        let u2 = uniform();
        let e = a - u1.ln() / lambda;
        if u2 <= (-(e - lambda).powi(2) / 2.0).exp() {
            return Ok(if positive { mean + e } else { mean - e });
        }
    }
    Err(EconError::Computation(format!(
        "probit BART: no truncated normal draw for mean {mean} in {TAIL_MAX_REJECTIONS} proposals"
    )))
}

/// Log integrated likelihood of a leaf holding `n` partial residuals with sum `s`, up to the
/// terms that cancel between a parent and its two children: the residuals are N(mu, s2) and
/// mu ~ N(0, t2) is integrated out.
fn leaf_log_lik(n: usize, s: f64, s2: f64, t2: f64) -> f64 {
    let k = s2 + n as f64 * t2;
    0.5 * (s2 / k).ln() + t2 * s * s / (2.0 * s2 * k)
}

/// The tree prior's probability that a node at depth `d` splits, alpha (1 + d)^-beta.
fn split_prob(alpha: f64, beta: f64, d: usize) -> f64 {
    alpha * (1.0 + d as f64).powf(-beta)
}

impl BartSampler {
    /// Initialize the sampler on the backfitting target's length `n` and the covariates.
    fn new(
        y: &Array1<f64>,
        x: ArrayView2<f64>,
        config: &BartConfig,
        prior: Option<WbartPrior>,
    ) -> EconResult<Self> {
        let n = x.nrows();
        let p = x.ncols();
        let m = config.n_trees;
        if p > u32::MAX as usize {
            return Err(EconError::InvalidSpecification {
                message: format!("BART takes at most {} covariates (got {p})", u32::MAX),
            });
        }
        let mut cuts = try_with_capacity::<Vec<f64>>(p, "BART cutpoints")?;
        for j in 0..p {
            let c = cutpoints(x.column(j), config.numcut)?;
            if c.len() >= u32::MAX as usize {
                return Err(EconError::InvalidSpecification {
                    message: format!("covariate {j} has more than {} cutpoints", u32::MAX - 1),
                });
            }
            cuts.push(c);
        }
        let active: Vec<usize> = (0..p).filter(|&j| !cuts[j].is_empty()).collect();
        let mut trees = try_with_capacity::<BartTree>(m, "BART trees")?;
        trees.extend((0..m).map(|_| BartTree::new(0.0)));
        let buf = |what| -> EconResult<Vec<f64>> {
            let mut v = try_with_capacity::<f64>(n, what)?;
            v.resize(n, 0.0);
            Ok(v)
        };
        let mut leaf_of = try_with_capacity::<u32>(n, "BART leaf index")?;
        leaf_of.resize(n, 0);

        let (sigma, sigma_mu, lambda, offset, probit) = if config.classification {
            // The latent SD is 1 and the leaf prior fixed; trees start at 0 around the offset
            (
                1.0,
                3.0 / (config.k * (m as f64).sqrt()),
                0.0,
                normal_quantile(y.mean().unwrap_or(0.5))?,
                true,
            )
        } else {
            // Regression fits y - fmean: the trees start at 0, sigma at sigest, and the leaf
            // prior SD stays at tau for the whole chain
            let prior = match prior {
                Some(prior) => prior,
                None => wbart_prior(y.view(), x, config)?,
            };
            (prior.sigest, prior.tau, prior.lambda, prior.fmean, false)
        };

        Ok(Self {
            trees,
            sigma,
            sigma_mu,
            config: config.clone(),
            rng: config.seed.unwrap_or(42),
            var_counts: vec![0; p],
            n_features: p,
            lambda,
            probit,
            offset,
            cuts,
            active,
            allfit: buf("BART fit")?,
            ftemp: buf("BART tree fit")?,
            resid: buf("BART residuals")?,
            leaf_of,
            fresh: buf("BART fit")?,
            node_n: Vec::new(),
            node_sum: Vec::new(),
        })
    }

    /// A uniform draw on [0, 1): an LCG state passed through the SplitMix64 finaliser, 53 bits
    fn random(&mut self) -> f64 {
        self.rng = self
            .rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Generate random integer in [0, max), max >= 1
    fn random_int(&mut self, max: usize) -> usize {
        ((self.random() * max as f64) as usize).min(max - 1)
    }

    /// Sample from standard normal (Box-Muller)
    fn random_normal(&mut self) -> f64 {
        let u1 = 1.0 - self.random(); // (0, 1]
        let u2 = self.random();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Sample from inverse-gamma(alpha, beta)
    fn random_inv_gamma(&mut self, alpha: f64, beta: f64) -> f64 {
        // Gamma via Marsaglia-Tsang for alpha >= 1
        let a = if alpha < 1.0 { alpha + 1.0 } else { alpha };
        let d = a - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        let g = loop {
            let x = self.random_normal();
            let v = (1.0 + c * x).powi(3);
            if v > 0.0 {
                let u = self.random();
                if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x.powi(2) + d * (1.0 - v + v.ln())
                {
                    break d * v;
                }
            }
        };

        let gamma = if alpha < 1.0 {
            g * (1.0 - self.random()).powf(1.0 / alpha)
        } else {
            g
        };

        beta / gamma
    }

    /// Probit BART's latent draw: z_i ~ N(offset + f(x_i), 1) truncated by y_i, returned as
    /// z - offset, the sweep's backfitting target.
    fn draw_latent(&mut self, y01: &Array1<f64>) -> EconResult<Array1<f64>> {
        let offset = self.offset;
        let fit = std::mem::take(&mut self.allfit);
        let mut latent = Array1::zeros(y01.len());
        let mut uniform = || self.random();
        let mut out = Ok(());
        for i in 0..y01.len() {
            match truncated_normal(offset + fit[i], y01[i] == 1.0, &mut uniform) {
                Ok(z) => latent[i] = z - offset,
                Err(e) => {
                    out = Err(e);
                    break;
                }
            }
        }
        self.allfit = fit;
        out.map(|()| latent)
    }

    /// The range of cutpoint indices of variable `v` still available at `node`: those strictly
    /// inside the bounds its ancestors' rules on `v` leave (BART's `tree::rg`).
    fn cut_range(&self, tree: &BartTree, node: usize, v: usize) -> Option<(usize, usize)> {
        let ncut = self.cuts[v].len();
        if ncut == 0 {
            return None;
        }
        let (mut lo, mut hi) = (0i64, ncut as i64 - 1);
        let mut child = node;
        let mut parent = tree.nodes[node].parent;
        while parent != NO_NODE {
            let pn = &tree.nodes[parent as usize];
            if pn.var as usize == v {
                let c = pn.cut as i64;
                if pn.left as usize == child {
                    hi = hi.min(c - 1);
                } else {
                    lo = lo.max(c + 1);
                }
            }
            child = parent as usize;
            parent = pn.parent;
        }
        (lo <= hi).then_some((lo as usize, hi as usize))
    }

    /// Whether `node` can split: below `max_depth`, with some variable left that has an
    /// available cutpoint (BART's `cansplit`, with the depth cap).
    fn can_split(&self, tree: &BartTree, node: usize) -> bool {
        (tree.nodes[node].depth as usize) < self.config.max_depth
            && self
                .active
                .iter()
                .any(|&v| self.cut_range(tree, node, v).is_some())
    }

    /// The tree prior's probability that `node` splits: alpha (1 + d)^-beta if it can, else 0
    fn grow_prob(&self, tree: &BartTree, node: usize) -> f64 {
        if self.can_split(tree, node) {
            split_prob(
                self.config.alpha,
                self.config.beta,
                tree.nodes[node].depth as usize,
            )
        } else {
            0.0
        }
    }

    /// One MCMC sweep through all trees on the backfitting target `y` (the centred response, or
    /// the latent draw for probit BART), then the sigma draw. Rows are in `xs`, row-major.
    fn step(&mut self, y: &Array1<f64>, xs: &[f64]) {
        let n = y.len();
        let p = self.n_features;
        self.fresh.iter_mut().for_each(|v| *v = 0.0);
        for j in 0..self.trees.len() {
            {
                let tree = &self.trees[j];
                for i in 0..n {
                    let leaf = tree.leaf_of(&xs[i * p..(i + 1) * p]);
                    self.leaf_of[i] = leaf as u32;
                    self.ftemp[i] = tree.nodes[leaf].mu;
                    self.resid[i] = y[i] - self.allfit[i] + self.ftemp[i];
                }
            }
            self.birth_death(j, xs);
            self.draw_leaf_values(j);
            let tree = &self.trees[j];
            for i in 0..n {
                let f = tree.nodes[self.leaf_of[i] as usize].mu;
                self.allfit[i] += f - self.ftemp[i];
                self.fresh[i] += f;
            }
        }
        // The sum of the trees in tree order, free of the drift of the running update
        std::mem::swap(&mut self.allfit, &mut self.fresh);

        if self.probit {
            return;
        }
        let ss: f64 = (0..n).map(|i| (y[i] - self.allfit[i]).powi(2)).sum();
        // Posterior: sigma^2 | ... ~ IG((nu + n)/2, (nu*lambda + ss)/2)
        let nu = self.config.nu;
        let post_alpha = (nu + n as f64) / 2.0;
        let post_beta = (nu * self.lambda + ss) / 2.0;
        self.sigma = self.random_inv_gamma(post_alpha, post_beta).sqrt();
    }

    /// One birth-or-death Metropolis-Hastings move on tree `j` (BART's `bd`), on the partial
    /// residuals in `resid` and the leaf indices in `leaf_of`, which it keeps current.
    fn birth_death(&mut self, j: usize, xs: &[f64]) {
        let tree = &self.trees[j];
        let goodbots: Vec<usize> = tree
            .get_leaves()
            .into_iter()
            .filter(|&l| self.can_split(tree, l))
            .collect();
        let pb = if goodbots.is_empty() {
            0.0
        } else if tree.nodes.len() == 1 {
            1.0
        } else {
            0.5
        };
        if self.random() < pb {
            self.birth(j, xs, &goodbots, pb);
        } else {
            self.death(j, goodbots.len(), pb);
        }
    }

    /// The prior-and-proposal part of a birth's MH ratio at leaf `eta` on (`v`, cut index `c`)
    /// with available range `[lo, hi]` (BART's `bprop`): p_eta (1 - p_l)(1 - p_r) PD Pnog /
    /// ((1 - p_eta) Pbot PB). The children's grow probabilities use the depth cap and whether a
    /// variable stays available in each child.
    fn birth_ratio(
        &self,
        tree: &BartTree,
        eta: usize,
        v: usize,
        (c, lo, hi): (usize, usize, usize),
        n_goodbots: usize,
        pb: f64,
    ) -> f64 {
        let (alpha, beta) = (self.config.alpha, self.config.beta);
        let d = tree.nodes[eta].depth as usize;
        let pg_eta = split_prob(alpha, beta, d);
        let child_depth_ok = d + 1 < self.config.max_depth;
        let other_good = self
            .active
            .iter()
            .any(|&w| w != v && self.cut_range(tree, eta, w).is_some());
        let pg_child = split_prob(alpha, beta, d + 1);
        let pgl = if child_depth_ok && (other_good || c > lo) {
            pg_child
        } else {
            0.0
        };
        let pgr = if child_depth_ok && (other_good || c < hi) {
            pg_child
        } else {
            0.0
        };
        let pd_new = if n_goodbots > 1 || pgl > 0.0 || pgr > 0.0 {
            0.5
        } else {
            1.0
        };
        let n_nogs = tree.nogs().len();
        let parent = tree.nodes[eta].parent;
        let pnog_new = if parent == NO_NODE {
            1.0
        } else if tree.is_nog(parent as usize) {
            1.0 / n_nogs as f64
        } else {
            1.0 / (n_nogs as f64 + 1.0)
        };
        pg_eta * (1.0 - pgl) * (1.0 - pgr) * pd_new * pnog_new
            / ((1.0 - pg_eta) * (1.0 / n_goodbots as f64) * pb)
    }

    /// The prior-and-proposal part of a death's MH ratio at the nog node `eta` (BART's
    /// `dprop`): (1 - p_eta) PB Pbot / (p_eta (1 - p_l)(1 - p_r) PD Pnog).
    fn death_ratio(&self, tree: &BartTree, eta: usize, n_goodbots: usize, pb: f64) -> f64 {
        let node = tree.nodes[eta];
        let (l, r) = (node.left as usize, node.right as usize);
        let pg_eta = split_prob(self.config.alpha, self.config.beta, node.depth as usize);
        let pgl = self.grow_prob(tree, l);
        let pgr = self.grow_prob(tree, r);
        let pb_new = if node.parent == NO_NODE { 1.0 } else { 0.5 };
        let ngood_new = n_goodbots + 1 - usize::from(pgl > 0.0) - usize::from(pgr > 0.0);
        let n_nogs = tree.nogs().len();
        (1.0 - pg_eta) * pb_new * (1.0 / ngood_new as f64)
            / (pg_eta * (1.0 - pgl) * (1.0 - pgr) * (1.0 - pb) * (1.0 / n_nogs as f64))
    }

    /// Propose and maybe accept a birth at a leaf that can split.
    fn birth(&mut self, j: usize, xs: &[f64], goodbots: &[usize], pb: f64) {
        let p = self.n_features;
        let eta = goodbots[self.random_int(goodbots.len())];
        // The variable is drawn among all covariates with cutpoints, as wbart; one exhausted
        // at eta puts every row in one child, a birth wbart always rejects
        let k = self.random_int(self.active.len());
        let v = self.active[k];
        let tree = &self.trees[j];
        let Some((lo, hi)) = self.cut_range(tree, eta, v) else {
            return;
        };
        let c = lo + self.random_int(hi - lo + 1);
        let cutv = self.cuts[v][c];

        let (mut nl, mut sl, mut nr, mut sr) = (0usize, 0.0, 0usize, 0.0);
        for i in 0..self.leaf_of.len() {
            if self.leaf_of[i] as usize == eta {
                if xs[i * p + v] < cutv {
                    nl += 1;
                    sl += self.resid[i];
                } else {
                    nr += 1;
                    sr += self.resid[i];
                }
            }
        }
        let min = self.config.min_node_size;
        if nl.saturating_add(nr) < min_rows_to_split(min) || nl < min || nr < min {
            return;
        }
        let tree = &self.trees[j];
        let ratio = self.birth_ratio(tree, eta, v, (c, lo, hi), goodbots.len(), pb);
        let (s2, t2) = (self.sigma * self.sigma, self.sigma_mu * self.sigma_mu);
        let log_alpha = ratio.ln() + leaf_log_lik(nl, sl, s2, t2) + leaf_log_lik(nr, sr, s2, t2)
            - leaf_log_lik(nl + nr, sl + sr, s2, t2);
        if (1.0 - self.random()).ln() < log_alpha {
            let (l, r) = self.trees[j].birth(eta, v, c, cutv, 0.0, 0.0);
            self.var_counts[v] += 1;
            for i in 0..self.leaf_of.len() {
                if self.leaf_of[i] as usize == eta {
                    self.leaf_of[i] = if xs[i * p + v] < cutv { l } else { r } as u32;
                }
            }
        }
    }

    /// Propose and maybe accept a death at a nog node.
    fn death(&mut self, j: usize, n_goodbots: usize, pb: f64) {
        let nogs = self.trees[j].nogs();
        if nogs.is_empty() {
            return; // a single leaf that cannot split
        }
        let eta = nogs[self.random_int(nogs.len())];
        let tree = &self.trees[j];
        let (l, r) = (
            tree.nodes[eta].left as usize,
            tree.nodes[eta].right as usize,
        );
        let (mut nl, mut sl, mut nr, mut sr) = (0usize, 0.0, 0usize, 0.0);
        for i in 0..self.leaf_of.len() {
            let k = self.leaf_of[i] as usize;
            if k == l {
                nl += 1;
                sl += self.resid[i];
            } else if k == r {
                nr += 1;
                sr += self.resid[i];
            }
        }
        let ratio = self.death_ratio(tree, eta, n_goodbots, pb);
        let (s2, t2) = (self.sigma * self.sigma, self.sigma_mu * self.sigma_mu);
        let log_alpha = ratio.ln() + leaf_log_lik(nl + nr, sl + sr, s2, t2)
            - leaf_log_lik(nl, sl, s2, t2)
            - leaf_log_lik(nr, sr, s2, t2);
        if (1.0 - self.random()).ln() < log_alpha {
            let var = self.trees[j].nodes[eta].var as usize;
            self.var_counts[var] = self.var_counts[var].saturating_sub(1);
            let remap = self.trees[j].death(eta, 0.0);
            for k in self.leaf_of.iter_mut() {
                *k = remap[*k as usize];
            }
        }
    }

    /// Draw every leaf value of tree `j` from its conditional posterior N(post_var S / s2,
    /// post_var), post_var = 1 / (n / s2 + 1 / t2), in node order.
    fn draw_leaf_values(&mut self, j: usize) {
        let len = self.trees[j].nodes.len();
        self.node_n.clear();
        self.node_n.resize(len, 0);
        self.node_sum.clear();
        self.node_sum.resize(len, 0.0);
        for i in 0..self.leaf_of.len() {
            let k = self.leaf_of[i] as usize;
            self.node_n[k] += 1;
            self.node_sum[k] += self.resid[i];
        }
        let (s2, t2) = (self.sigma * self.sigma, self.sigma_mu * self.sigma_mu);
        for k in 0..len {
            if self.trees[j].nodes[k].is_leaf() {
                let post_var = 1.0 / (self.node_n[k] as f64 / s2 + 1.0 / t2);
                let post_mean = post_var * self.node_sum[k] / s2;
                let z = self.random_normal();
                self.trees[j].nodes[k].mu = post_mean + post_var.sqrt() * z;
            }
        }
    }

    /// Get variable importance (normalized counts)
    fn get_variable_importance(&self) -> Vec<f64> {
        let total: usize = self.var_counts.iter().sum();
        if total == 0 {
            return vec![1.0 / self.n_features as f64; self.n_features];
        }
        self.var_counts
            .iter()
            .map(|&c| c as f64 / total as f64)
            .collect()
    }
}

// ============================================================================
// Main API Functions
// ============================================================================

/// Run BART from a dataset.
///
/// # Arguments
///
/// * `dataset` - The dataset containing the response and features
/// * `y_col` - Name of the response variable
/// * `x_cols` - Names of feature columns
/// * `config` - BART configuration
///
/// # Returns
///
/// `BartResult` containing predictions with uncertainty and variable importance.
///
/// # Example
///
/// ```ignore
/// use p2a_core::ml::{bart, BartConfig};
///
/// let config = BartConfig {
///     n_trees: 200,
///     n_burn: 250,
///     n_mcmc: 1000,
///     ..Default::default()
/// };
///
/// let result = bart(&dataset, "y", &["x1", "x2", "x3"], config)?;
/// println!("Predictions: {:?}", &result.predictions[..5]);
/// println!("Sigma: {:.4}", result.sigma);
/// ```
///
/// # References
///
/// - Chipman, George & McCulloch (2010). BART: Bayesian Additive Regression Trees.
/// - R package `BART` (Sparapani et al., 2021).
pub fn bart(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: BartConfig,
) -> EconResult<BartResult> {
    // Extract y
    let y = DesignMatrix::extract_column(dataset.df(), y_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    // Extract X
    let n = dataset.nrows();
    let p = x_cols.len();

    if p == 0 {
        return Err(EconError::InvalidSpecification {
            message: "At least one feature column is required".to_string(),
        });
    }

    let mut x = Array2::zeros((n, p));
    for (j, &col_name) in x_cols.iter().enumerate() {
        let col_data = DesignMatrix::extract_column(dataset.df(), col_name).map_err(|e| {
            EconError::ColumnNotFound {
                column: col_name.to_string(),
                available: vec![format!("{:?}", e)],
            }
        })?;
        for i in 0..n {
            x[[i, j]] = col_data[i];
        }
    }

    let feature_names: Vec<String> = x_cols.iter().map(|s| s.to_string()).collect();

    bart_arrays(y.view(), x.view(), Some(feature_names), config)
}

/// Check the counts and prior parameters of a BART configuration.
///
/// `n_burn = 0` is valid.
fn validate_bart_config(config: &BartConfig) -> EconResult<()> {
    let invalid = |message: String| Err(EconError::InvalidSpecification { message });
    if config.n_trees == 0 {
        return invalid("n_trees must be at least 1 (got 0)".to_string());
    }
    if config.n_mcmc == 0 {
        return invalid("n_mcmc must be at least 1 (got 0)".to_string());
    }
    for (name, v) in [("nu", config.nu), ("k", config.k)] {
        if !(v.is_finite() && v > 0.0) {
            return invalid(format!("{name} must be finite and positive (got {v})"));
        }
    }
    for (name, v) in [("alpha", config.alpha), ("q", config.q)] {
        if !(v.is_finite() && v > 0.0 && v < 1.0) {
            return invalid(format!("{name} must be in (0, 1) (got {v})"));
        }
    }
    if !(config.beta.is_finite() && config.beta >= 0.0) {
        return invalid(format!(
            "beta must be finite and non-negative (got {})",
            config.beta
        ));
    }
    let cl = config.confidence_level;
    if !(cl.is_finite() && cl > 0.0 && cl < 1.0) {
        return invalid(format!("confidence_level must be in (0, 1) (got {cl})"));
    }
    if config.numcut == 0 {
        return invalid("numcut must be at least 1 (got 0)".to_string());
    }
    Ok(())
}

/// Run BART with array inputs.
///
/// This is the core implementation, separate from data loading for flexibility.
pub fn bart_arrays(
    y: ArrayView1<f64>,
    x: ArrayView2<f64>,
    feature_names: Option<Vec<String>>,
    config: BartConfig,
) -> EconResult<BartResult> {
    validate_bart_config(&config)?;

    let n = y.len();
    let p = x.ncols();

    // Validate inputs
    if n < 20 {
        return Err(EconError::InsufficientData {
            required: 20,
            provided: n,
            context: "BART requires at least 20 observations".to_string(),
        });
    }

    if x.nrows() != n {
        return Err(EconError::InvalidSpecification {
            message: "X and y must have same number of observations".to_string(),
        });
    }

    if let Some((i, v)) = y.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!("y must contain only finite values (row {i} is {v})"),
        });
    }
    if let Some(((i, j), v)) = x.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!("x must contain only finite values (row {i}, column {j} is {v})"),
        });
    }

    if config.classification {
        if let Some((i, v)) = y.iter().enumerate().find(|&(_, &v)| v != 0.0 && v != 1.0) {
            return Err(EconError::InvalidSpecification {
                message: format!("classification needs y coded 0/1 (row {i} is {v})"),
            });
        }
        if y.iter().all(|&v| v == y[0]) {
            return Err(EconError::InvalidSpecification {
                message: "classification needs both classes in y".to_string(),
            });
        }
    }

    // Regression: wbart's prior, and the sampler fits y - fmean
    let prior = if config.classification {
        None
    } else {
        Some(wbart_prior(y, x, &config)?)
    };
    let y_owned = match prior {
        Some(prior) => y.mapv(|v| v - prior.fmean),
        None => y.to_owned(),
    };
    // Rows in one row-major buffer for the tree traversals
    let mut xs =
        try_with_capacity::<f64>(checked_len(&[n, p], "BART covariates")?, "BART covariates")?;
    for row in x.rows() {
        xs.extend(row.iter().copied());
    }

    // Initialize sampler
    let mut sampler = BartSampler::new(&y_owned, x, &config, prior)?;

    // Posterior draws, allocated before any sweep runs
    let mut posterior_preds = try_zeros2(config.n_mcmc, n, "BART posterior draws")?;
    let mut sigma_samples = try_with_capacity::<f64>(config.n_mcmc, "BART sigma draws")?;

    // Burn-in
    for _ in 0..config.n_burn {
        checkpoint()?;
        if config.classification {
            let latent = sampler.draw_latent(&y_owned)?;
            sampler.step(&latent, &xs);
        } else {
            sampler.step(&y_owned, &xs);
        }
    }

    // Reset variable importance counts after burn-in
    sampler.var_counts = vec![0; p];

    // Collect posterior samples
    for iter in 0..config.n_mcmc {
        checkpoint()?;
        if config.classification {
            let latent = sampler.draw_latent(&y_owned)?;
            sampler.step(&latent, &xs);
        } else {
            sampler.step(&y_owned, &xs);
        }

        let offset = sampler.offset;
        let mut row = posterior_preds.row_mut(iter);
        for (d, &f) in row.iter_mut().zip(&sampler.allfit) {
            *d = if config.classification {
                std_normal_cdf(offset + f)
            } else {
                offset + f
            };
        }
        sigma_samples.push(sampler.sigma);
    }

    // Compute posterior summaries
    let predictions: Vec<f64> = (0..n)
        .map(|i| posterior_preds.column(i).mean().unwrap_or(0.0))
        .collect();

    let prediction_sd: Vec<f64> = (0..n)
        .map(|i| {
            let col = posterior_preds.column(i);
            let mean = col.mean().unwrap_or(0.0);
            (col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / config.n_mcmc as f64).sqrt()
        })
        .collect();

    // Prediction intervals: R's quantile(type = 7), as BART's posterior summaries are read
    let mut prediction_lower = Vec::with_capacity(n);
    let mut prediction_upper = Vec::with_capacity(n);

    for i in 0..n {
        let col_vals: Vec<f64> = posterior_preds.column(i).to_vec();
        let (lower, upper) = percentile_interval(&col_vals, config.confidence_level)?;
        prediction_lower.push(lower);
        prediction_upper.push(upper);
    }

    // Sigma posterior summaries
    let sigma = sigma_samples.iter().sum::<f64>() / sigma_samples.len() as f64;
    let sigma_sd = {
        let var = sigma_samples
            .iter()
            .map(|&s| (s - sigma).powi(2))
            .sum::<f64>()
            / sigma_samples.len() as f64;
        var.sqrt()
    };

    // Variable importance
    let variable_importance = sampler.get_variable_importance();

    Ok(BartResult {
        predictions,
        prediction_lower,
        prediction_upper,
        prediction_sd,
        variable_importance,
        feature_names,
        sigma,
        sigma_sd,
        offset: sampler.offset,
        sigest: prior.map_or(1.0, |p| p.sigest),
        lambda: prior.map_or(0.0, |p| p.lambda),
        leaf_prior_sd: sampler.sigma_mu,
        n_obs: n,
        n_features: p,
        n_trees: config.n_trees,
        n_samples: config.n_mcmc,
        config: config.clone(),
        posterior_samples: Some(posterior_preds),
        sigma_samples: (!config.classification).then_some(sigma_samples),
    })
}

/// Run BART from dataset with convenience parameters (for MCP integration).
pub fn run_bart(
    dataset: &Dataset,
    y_col: &str,
    x_cols: Vec<String>,
    n_trees: Option<usize>,
    n_burn: Option<usize>,
    n_mcmc: Option<usize>,
    k: Option<f64>,
    alpha: Option<f64>,
    beta: Option<f64>,
    seed: Option<u64>,
) -> EconResult<BartResult> {
    let x_refs: Vec<&str> = x_cols.iter().map(|s| s.as_str()).collect();

    let config = BartConfig {
        n_trees: n_trees.unwrap_or(200),
        n_burn: n_burn.unwrap_or(250),
        n_mcmc: n_mcmc.unwrap_or(1000),
        k: k.unwrap_or(2.0),
        alpha: alpha.unwrap_or(0.95),
        beta: beta.unwrap_or(2.0),
        seed,
        ..Default::default()
    };

    bart(dataset, y_col, &x_refs, config)
}

/// Predict using BART posterior for new data.
///
/// Returns posterior mean predictions and prediction intervals.
pub fn bart_predict(
    result: &BartResult,
    x_new: ArrayView2<f64>,
) -> EconResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    // For out-of-sample prediction, we would need to store the trees
    // For now, return an error indicating this limitation
    Err(EconError::InvalidSpecification {
        message: "Out-of-sample prediction requires storing the tree ensemble. Use bart_arrays() with the new data included in training for now.".to_string(),
    })
}

// ============================================================================
// Tests
// ============================================================================

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

    use crate::data::columns::{NaPolicy, numeric_column};
    use p2a_validation::{Check, Tol};

    fn case_matrix(ds: &Dataset, cols: &[&str]) -> Array2<f64> {
        let mut m = Array2::zeros((ds.nrows(), cols.len()));
        for (j, c) in cols.iter().enumerate() {
            let (v, _) = numeric_column(ds.df(), c, NaPolicy::Fail).unwrap();
            m.column_mut(j).assign(&Array1::from_vec(v));
        }
        m
    }
    fn case_vector(ds: &Dataset, col: &str) -> Array1<f64> {
        Array1::from_vec(numeric_column(ds.df(), col, NaPolicy::Fail).unwrap().0)
    }

    const FRIEDMAN_X: [&str; 6] = ["x1", "x2", "x3", "x4", "x5", "x6"];

    #[test]
    fn test_validate_bart_wbart_calibration() {
        let (case, ds) = crate::test_support::load_case("ml_bart_wbart_calibration");
        let x = case_matrix(&ds, &FRIEDMAN_X);
        let y = case_vector(&ds, "y");
        let prior = |m| {
            let c = BartConfig {
                n_trees: m,
                ..Default::default()
            };
            wbart_prior(y.view(), x.view(), &c).unwrap()
        };
        let (p50, p200) = (prior(50), prior(200));
        let cuts: Vec<Vec<f64>> = (0..6)
            .map(|j| cutpoints(x.column(j), 100).unwrap())
            .collect();
        let numcut: Vec<f64> = cuts.iter().map(|c| c.len() as f64).collect();
        Check::new(&case)
            .scalar("fmean", p50.fmean, Tol::coef(200))
            .scalar("sigest", p50.sigest, Tol::coef(200))
            .scalar(
                "qchisq_0_1_3",
                chi_squared_quantile(0.1, 3.0).unwrap(),
                Tol::coef(200),
            )
            .scalar("lambda", p50.lambda, Tol::coef(200))
            .scalar("tau_m50", p50.tau, Tol::coef(200))
            .scalar("tau_m200", p200.tau, Tol::coef(200))
            .vector("numcut", &numcut, Tol::exact())
            .vector("cut_x1", &cuts[0], Tol::coef(200))
            .vector("cut_x6", &cuts[5], Tol::coef(200))
            .finish();
        assert_eq!(
            (p200.fmean, p200.sigest, p200.lambda),
            (p50.fmean, p50.sigest, p50.lambda)
        );
        // a constant column gets no cutpoint
        assert!(
            cutpoints(Array1::from_elem(10, 2.5).view(), 100)
                .unwrap()
                .is_empty()
        );
    }

    /// The finding's design: y = 1000 + 5 x + noise. Uncentred, the leaf prior shrank the fit
    /// toward 0 (mean fit about 38.6 below mean y) and sigma absorbed the bias (about 38.5).
    /// wbart on the same design (30 chains): bias -0.0004, sigma 0.43, coverage of f 0.93.
    #[test]
    fn bart_centres_the_response_and_recovers_sigma() {
        let n = 100;
        let mut s = 20261113u64;
        let mut u = || (crate::ml::lcg_random(&mut s) % 1_000_000 + 1) as f64 / 1_000_002.0;
        let mut x = Array2::zeros((n, 1));
        let mut y = Array1::zeros(n);
        let mut f = Array1::zeros(n);
        for i in 0..n {
            x[[i, 0]] = u();
            f[i] = 1000.0 + 5.0 * x[[i, 0]];
            // Box-Muller, noise SD 0.5
            let (u1, u2) = (u(), u());
            y[i] = f[i] + 0.5 * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
        let c = BartConfig {
            n_trees: 50,
            n_burn: 300,
            n_mcmc: 1000,
            seed: Some(7),
            ..Default::default()
        };
        let r = bart_arrays(y.view(), x.view(), None, c).unwrap();
        let mean_pred = r.predictions.iter().sum::<f64>() / n as f64;
        let mean_y = y.sum() / n as f64;
        assert!(
            (mean_pred - mean_y).abs() < 0.05,
            "bias {}",
            mean_pred - mean_y
        );
        assert!(0.3 < r.sigma && r.sigma < 0.8, "sigma {}", r.sigma);
        let covered = (0..n)
            .filter(|&i| r.prediction_lower[i] <= f[i] && f[i] <= r.prediction_upper[i])
            .count();
        assert!(covered >= 80, "coverage {covered} of {n}");
        assert_eq!(r.offset, mean_y);
    }

    #[test]
    fn bart_refuses_a_constant_response() {
        let (_, x) = generate_test_data(40, 2);
        let y = Array1::from_elem(40, 3.0);
        let e = bart_arrays(y.view(), x.view(), None, tiny()).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("y is constant")),
            "{e}"
        );
    }

    #[test]
    fn test_validate_bart_probit_single_leaf_posterior() {
        let (case, ds) = crate::test_support::load_case("ml_bart_probit_single_leaf");
        let x = case_matrix(&ds, &["x1"]);
        let y = case_vector(&ds, "y");
        let c = BartConfig {
            classification: true,
            n_trees: 1,
            max_depth: 0,
            n_burn: 200,
            n_mcmc: 4000,
            k: 2.0,
            seed: Some(46),
            ..Default::default()
        };
        let r = bart_arrays(y.view(), x.view(), None, c).unwrap();
        assert!(
            r.predictions
                .iter()
                .all(|&p| (p - r.predictions[0]).abs() < 1e-15)
        ); // the tree never splits
        assert_eq!((r.sigma, r.sigma_sd), (1.0, 0.0));
        // Monte Carlo tolerance: 200 R replications of this chain have SD 0.001745 (mean) and 0.001045 (SD); 5 SD each.
        Check::new(&case)
            .scalar("offset", r.offset, Tol::coef(40))
            .scalar("post_mean_p", r.predictions[0], Tol::new(0.0, 0.009))
            .scalar("post_sd_p", r.prediction_sd[0], Tol::new(0.0, 0.006))
            .finish();
    }

    #[test]
    fn test_validate_bart_probit_tracks_glm_probit() {
        let (case, ds) = crate::test_support::load_case("ml_bart_probit_glm");
        let x = case_matrix(&ds, &["x1", "x2", "x3"]);
        let y = case_vector(&ds, "y");
        let p_glm = case.vector("glm_prob");
        let c = BartConfig {
            classification: true,
            n_trees: 50,
            n_burn: 200,
            n_mcmc: 300,
            seed: Some(46),
            ..Default::default()
        };
        let r = bart_arrays(y.view(), x.view(), None, c).unwrap();
        assert_eq!(r.sigma, 1.0);
        for i in 0..r.n_obs {
            assert!(
                0.0 < r.prediction_lower[i]
                    && r.prediction_lower[i] <= r.predictions[i]
                    && r.predictions[i] <= r.prediction_upper[i]
                    && r.prediction_upper[i] < 1.0,
                "row {i}"
            );
        }
        let n = p_glm.len() as f64;
        let mad = r
            .predictions
            .iter()
            .zip(&p_glm)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / n;
        let (ma, mb) = (
            r.predictions.iter().sum::<f64>() / n,
            p_glm.iter().sum::<f64>() / n,
        );
        let cov: f64 = r
            .predictions
            .iter()
            .zip(&p_glm)
            .map(|(a, b)| (a - ma) * (b - mb))
            .sum();
        let sa = r
            .predictions
            .iter()
            .map(|a| (a - ma).powi(2))
            .sum::<f64>()
            .sqrt();
        let sb = p_glm.iter().map(|b| (b - mb).powi(2)).sum::<f64>().sqrt();
        // A seeded property: six emulated chains on these data give mad 0.0436-0.0591, cor 0.943-0.975.
        assert!(mad < 0.085, "mean |p_bart - p_glm| = {mad}");
        assert!(cov / (sa * sb) > 0.90, "cor = {}", cov / (sa * sb));
    }

    #[test]
    fn test_truncated_normal_draws_stay_on_their_side() {
        let mut half = || 0.5;
        // -qnorm(0.25)
        assert!(
            (truncated_normal(0.0, true, &mut half).unwrap() - 0.6744897501960817).abs() < 1e-12
        );
        // 1 + qnorm(pnorm(-1) / 2)
        let z = truncated_normal(1.0, false, &mut half).unwrap();
        assert!((z - (-0.4096087092934544)).abs() < 1e-12, "{z}");
        let mut s = 9u64;
        let mut u = move || (crate::ml::lcg_random(&mut s) % 1_000_000 + 1) as f64 / 1_000_002.0;
        for mean in [-35.0, -3.0, 0.0, 2.0, 40.0] {
            for _ in 0..200 {
                let zp = truncated_normal(mean, true, &mut u).unwrap();
                let zn = truncated_normal(mean, false, &mut u).unwrap();
                assert!(zp > 0.0 && zp.is_finite(), "mean {mean}: {zp}");
                assert!(zn <= 0.0 && zn.is_finite(), "mean {mean}: {zn}");
            }
        }
        // the tail branch (Robert 1995): E[z | z > 0] for z ~ N(-35, 1) = -35 + phi(35) / Phi(-35)
        let tail: Vec<f64> = (0..2000)
            .map(|_| truncated_normal(-35.0, true, &mut u).unwrap())
            .collect();
        let m = tail.iter().sum::<f64>() / 2000.0;
        assert!((m - 0.028524970596052412).abs() < 0.0035, "{m}"); // 5 SE: conditional SD 0.0285, 2000 draws
    }

    #[test]
    fn bart_probit_rejects_non_binary_and_single_class() {
        let (y, x) = generate_test_data(40, 2);
        let mut c = tiny();
        c.classification = true;
        let e = bart_arrays(y.view(), x.view(), None, c.clone()).unwrap_err();
        assert!(e.to_string().contains("coded 0/1"), "{e}");
        let ones = Array1::from_elem(40, 1.0);
        let e = bart_arrays(ones.view(), x.view(), None, c).unwrap_err();
        assert!(e.to_string().contains("both classes"), "{e}");
    }

    #[test]
    fn test_bart_probit_display_names_the_model() {
        let (y, x) = generate_test_data(40, 2);
        let y01 = y.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        let mut c = tiny();
        c.classification = true;
        let r = bart_arrays(y01.view(), x.view(), None, c).unwrap();
        let text = r.to_string();
        assert!(text.contains("Probit BART: P(y = 1 | x)") && text.contains("PROBABILITY SUMMARY"));
        assert!(!text.contains("RESIDUAL SD"), "{text}");
        assert!(r.sigma_samples.is_none());
    }

    #[test]
    fn bart_intervals_are_type7_percentiles() {
        let (y, x) = generate_test_data(60, 5);
        let mut c = tiny();
        c.n_mcmc = 37; // a draw count where floor-index and type-7 rules differ
        let r = bart_arrays(y.view(), x.view(), None, c).unwrap();
        let draws = r.posterior_samples.as_ref().unwrap();
        for i in 0..r.n_obs {
            let col = draws.column(i).to_vec();
            let (lo, hi) = crate::stats::robust::percentile_interval(&col, 0.95).unwrap();
            assert_eq!(
                (r.prediction_lower[i], r.prediction_upper[i]),
                (lo, hi),
                "row {i}"
            );
        }
    }

    fn tiny() -> BartConfig {
        BartConfig {
            n_trees: 5,
            n_burn: 2,
            n_mcmc: 5,
            seed: Some(1),
            ..Default::default()
        }
    }

    #[test]
    fn bart_huge_counts_are_errors_not_aborts() {
        let (y, x) = generate_test_data(40, 2);
        for cfg in [
            BartConfig {
                n_mcmc: 1usize << 44,
                ..tiny()
            },
            BartConfig {
                n_trees: 1usize << 44,
                ..tiny()
            },
        ] {
            let r =
                with_alloc_ceiling(Some(1 << 30), || bart_arrays(y.view(), x.view(), None, cfg));
            assert!(
                matches!(r, Err(EconError::ResourceLimit(_))),
                "{:?}",
                r.err()
            );
        }
    }

    #[test]
    fn bart_validates_parameters() {
        validate_bart_config(&tiny()).unwrap();
        for (cfg, param) in [
            (
                BartConfig {
                    n_trees: 0,
                    ..tiny()
                },
                "n_trees",
            ),
            (
                BartConfig {
                    n_mcmc: 0,
                    ..tiny()
                },
                "n_mcmc",
            ), // panicked on an empty draw vector
            (
                BartConfig {
                    nu: f64::NAN,
                    ..tiny()
                },
                "nu",
            ), // inverse-gamma sampler never exits
            (BartConfig { nu: 0.0, ..tiny() }, "nu"),
            (BartConfig { k: -1.0, ..tiny() }, "k"),
            (
                BartConfig {
                    alpha: 1.5,
                    ..tiny()
                },
                "alpha",
            ),
            (
                BartConfig {
                    beta: f64::INFINITY,
                    ..tiny()
                },
                "beta",
            ),
            (BartConfig { q: 1.0, ..tiny() }, "q"),
            (
                BartConfig {
                    confidence_level: 0.0,
                    ..tiny()
                },
                "confidence_level",
            ),
        ] {
            let e = validate_bart_config(&cfg).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.starts_with(&format!("{param} "))),
                "{e}"
            );
        }
        let (y, x) = generate_test_data(40, 2);
        assert!(
            bart_arrays(
                y.view(),
                x.view(),
                None,
                BartConfig {
                    n_mcmc: 0,
                    ..tiny()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn bart_honours_cancellation_and_rejects_non_finite_inputs() {
        let (mut y, x) = generate_test_data(40, 2);
        let r = with_budget(cancelled(), || {
            bart_arrays(y.view(), x.view(), None, tiny())
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "{:?}", r.err());
        y[0] = f64::NAN;
        let r = bart_arrays(y.view(), x.view(), None, tiny());
        assert!(
            matches!(&r, Err(EconError::InvalidSpecification { message }) if message.contains("finite")),
            "{:?}",
            r.err()
        );
    }

    /// Generate synthetic data: y = sin(x1) + x2 + noise
    fn generate_test_data(n: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
        let mut rng = seed;
        let lcg = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*state >> 33) ^ *state) as f64 / u64::MAX as f64
        };

        let mut x = Array2::zeros((n, 3));
        let mut y = Array1::zeros(n);

        for i in 0..n {
            x[[i, 0]] = lcg(&mut rng) * 2.0 * std::f64::consts::PI; // x1 in [0, 2pi]
            x[[i, 1]] = lcg(&mut rng) * 2.0 - 1.0; // x2 in [-1, 1]
            x[[i, 2]] = lcg(&mut rng) * 2.0 - 1.0; // x3 (noise feature)

            let noise = (lcg(&mut rng) * 2.0 - 1.0) * 0.5; // noise in [-0.5, 0.5]
            y[i] = x[[i, 0]].sin() + x[[i, 1]] + noise;
        }

        (y, x)
    }

    #[test]
    fn test_bart_basic() {
        let (y, x) = generate_test_data(100, 42);

        let config = BartConfig {
            n_trees: 20, // Small for testing
            n_burn: 50,
            n_mcmc: 100,
            seed: Some(42),
            ..Default::default()
        };

        let result = bart_arrays(
            y.view(),
            x.view(),
            Some(vec!["x1".to_string(), "x2".to_string(), "x3".to_string()]),
            config,
        )
        .unwrap();

        // Check basic properties
        assert_eq!(result.n_obs, 100);
        assert_eq!(result.n_features, 3);
        assert_eq!(result.predictions.len(), 100);
        assert_eq!(result.prediction_lower.len(), 100);
        assert_eq!(result.prediction_upper.len(), 100);

        // Sigma should be positive
        assert!(result.sigma > 0.0);

        // Prediction intervals should contain predictions
        for i in 0..100 {
            assert!(result.prediction_lower[i] <= result.predictions[i]);
            assert!(result.prediction_upper[i] >= result.predictions[i]);
        }

        println!("BART test result:\n{}", result);
    }

    #[test]
    fn test_bart_variable_importance() {
        // x1 and x2 are relevant, x3 is pure noise
        let (y, x) = generate_test_data(150, 123);

        let config = BartConfig {
            n_trees: 30,
            n_burn: 100,
            n_mcmc: 200,
            seed: Some(123),
            ..Default::default()
        };

        let result = bart_arrays(
            y.view(),
            x.view(),
            Some(vec![
                "x1".to_string(),
                "x2".to_string(),
                "noise".to_string(),
            ]),
            config,
        )
        .unwrap();

        // x1 and x2 should have higher importance than noise
        // Note: This test may be sensitive due to randomness
        println!("Variable importance: {:?}", result.variable_importance);

        // The sum of importances should be 1
        let sum: f64 = result.variable_importance.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Importances should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_bart_prediction_quality() {
        // Simple linear relationship
        let n = 100;
        let mut rng = 42u64;
        let lcg = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*state >> 33) ^ *state) as f64 / u64::MAX as f64
        };

        let mut x = Array2::zeros((n, 1));
        let mut y = Array1::zeros(n);

        for i in 0..n {
            x[[i, 0]] = lcg(&mut rng) * 10.0;
            y[i] = 2.0 * x[[i, 0]] + 1.0 + (lcg(&mut rng) - 0.5) * 2.0; // y = 2x + 1 + noise
        }

        let config = BartConfig {
            n_trees: 50,
            n_burn: 100,
            n_mcmc: 200,
            seed: Some(42),
            ..Default::default()
        };

        let result = bart_arrays(y.view(), x.view(), None, config).unwrap();

        // Compute R-squared
        let y_mean = y.mean().unwrap();
        let ss_tot: f64 = y.iter().map(|&v| (v - y_mean).powi(2)).sum();
        let ss_res: f64 = y
            .iter()
            .zip(result.predictions.iter())
            .map(|(&yi, &pi)| (yi - pi).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;

        println!("R-squared: {:.4}", r2);
        assert!(
            r2 > 0.5,
            "R² should be > 0.5 for simple linear data, got {}",
            r2
        );
    }

    #[test]
    fn test_bart_insufficient_data() {
        let y = Array1::zeros(10);
        let x = Array2::zeros((10, 2));

        let config = BartConfig::default();
        let result = bart_arrays(y.view(), x.view(), None, config);

        assert!(result.is_err());
    }

    /// The fit of one tree at each row of `x`
    fn tree_fit(tree: &BartTree, x: &Array2<f64>) -> Vec<f64> {
        x.rows()
            .into_iter()
            .map(|r| tree.nodes[tree.leaf_of(r.as_slice().unwrap())].mu)
            .collect()
    }

    #[test]
    fn test_bart_tree_operations() {
        let mut tree = BartTree::new(0.5);
        assert_eq!(tree.get_leaves().len(), 1);
        assert_eq!(tree.get_internals().len(), 0);

        // cutpoint 3 of variable 0, value 0.5
        let (l, r) = tree.birth(0, 0, 3, 0.5, 0.3, 0.7);
        assert_eq!((l, r), (1, 2));
        assert_eq!(tree.get_leaves(), vec![1, 2]);
        assert_eq!(tree.get_internals(), vec![0]);
        assert_eq!((tree.get_depth(1), tree.nodes[1].parent), (1, 0));
        assert_eq!((tree.nodes[0].cut, tree.nodes[0].cutv), (3, 0.5));

        // x < 0.5 goes left, as BART's tree::bn; 0.5 itself goes right
        let x = array![[0.3], [0.5], [0.7]];
        assert_eq!(tree_fit(&tree, &x), vec![0.3, 0.7, 0.7]);

        // grow the left leaf, then remove that split: every node is reachable afterwards
        tree.birth(1, 0, 1, 0.2, 0.1, 0.4);
        assert_eq!(tree.nogs(), vec![1]);
        let remap = tree.death(1, 0.35);
        assert_eq!(remap, vec![0, 1, 2, 1, 1]);
        assert_eq!(tree.nodes.len(), 3);
        assert_eq!(tree.walk().len(), 3);
        assert!(tree.nodes[1].is_leaf());
        assert_eq!(tree.nodes[1].mu, 0.35);
        let remap = tree.death(0, 0.5);
        assert_eq!((remap, tree.nodes.len()), (vec![0, 0, 0], 1));

        // a death in the middle of the vector moves the last nodes into the freed slots
        tree.birth(0, 0, 3, 0.5, 0.0, 0.0);
        tree.birth(1, 0, 1, 0.2, 0.0, 0.0);
        tree.birth(2, 0, 5, 0.8, 0.6, 0.9);
        let remap = tree.death(1, 0.25);
        assert_eq!(remap, vec![0, 1, 2, 1, 1, 3, 4]);
        assert_tree_invariants(&tree);
        let x = array![[0.1], [0.3], [0.7], [0.95]];
        assert_eq!(tree_fit(&tree, &x), vec![0.25, 0.25, 0.6, 0.9]);
    }

    /// Every node is reachable from the root, there are 2 * internal + 1 of them, parents and
    /// children point at each other, and each stored depth is the walked depth.
    fn assert_tree_invariants(tree: &BartTree) {
        let reached = tree.walk();
        let internal = reached
            .iter()
            .filter(|&&i| !tree.nodes[i].is_leaf())
            .count();
        assert_eq!(reached.len(), tree.nodes.len(), "unreachable nodes");
        assert_eq!(tree.nodes.len(), 2 * internal + 1);
        let mut seen = vec![false; tree.nodes.len()];
        for &i in &reached {
            assert!(!seen[i], "node {i} reached twice");
            seen[i] = true;
            let node = tree.nodes[i];
            if i == 0 {
                assert_eq!((node.parent, node.depth), (NO_NODE, 0));
            } else {
                let parent = tree.nodes[node.parent as usize];
                assert!(parent.left as usize == i || parent.right as usize == i);
                assert_eq!(node.depth, parent.depth + 1, "node {i}");
            }
        }
    }

    #[test]
    fn bart_trees_hold_only_reachable_nodes() {
        let (y, x) = generate_test_data(150, 3);
        let c = BartConfig {
            n_trees: 20,
            seed: Some(3),
            ..Default::default()
        };
        let prior = wbart_prior(y.view(), x.view(), &c).unwrap();
        let yc = y.mapv(|v| v - prior.fmean);
        let mut sampler = BartSampler::new(&yc, x.view(), &c, Some(prior)).unwrap();
        let xs: Vec<f64> = x.iter().copied().collect();
        let mut deaths_seen = false;
        for _ in 0..200 {
            let before: usize = sampler.trees.iter().map(|t| t.nodes.len()).sum();
            sampler.step(&yc, &xs);
            let after: usize = sampler.trees.iter().map(|t| t.nodes.len()).sum();
            deaths_seen |= after < before;
            for tree in &sampler.trees {
                assert_tree_invariants(tree);
            }
        }
        assert!(deaths_seen, "no death was accepted");
    }

    /// A sampler on `n` rows of two covariates with 100 distinct values each, so every node of
    /// a shallow tree has cutpoints left on both.
    fn kernel_sampler(config: &BartConfig) -> (BartSampler, Vec<f64>, Array1<f64>) {
        let (y, x) = generate_test_data(120, 11);
        let x = x.slice(s![.., 0..2]).to_owned();
        let prior = wbart_prior(y.view(), x.view(), config).unwrap();
        let yc = y.mapv(|v| v - prior.fmean);
        let sampler = BartSampler::new(&yc, x.view(), config, Some(prior)).unwrap();
        let xs: Vec<f64> = x.iter().copied().collect();
        (sampler, xs, yc)
    }

    /// CGM (1998) grow ratio of the tree prior, ln p_d + 2 ln(1 - p_{d+1}) - ln(1 - p_d) at
    /// alpha 0.95, beta 2 (R: T4-WP17-ref/lil.R).
    #[test]
    fn bart_grow_prior_ratio_matches_cgm() {
        let want = [
            2.4021334381652988,
            -1.3895398568183652,
            -2.2593868479078156,
            -2.8401529422152962,
        ];
        let (sampler, _, _) = kernel_sampler(&BartConfig::default());
        // a path of left children: the node at depth d is grown at its left leaf
        let mut tree = BartTree::new(0.0);
        let mut eta = 0;
        for (d, &w) in want.iter().enumerate() {
            let goodbots = tree.get_leaves();
            let n_goodbots = goodbots.len();
            let pb = if tree.nodes.len() == 1 { 1.0 } else { 0.5 };
            let ratio = sampler.birth_ratio(&tree, eta, 1, (50, 0, 99), n_goodbots, pb);
            // the move and node-choice factors, PD(new) Pnog(new) / (Pbot(old) PB(old)): the
            // grown node's parent stops being a nog when it is one
            let n_nogs = tree.nogs().len() as f64;
            let parent = tree.nodes[eta].parent;
            let pnog = if parent == NO_NODE {
                1.0
            } else if tree.is_nog(parent as usize) {
                1.0 / n_nogs
            } else {
                1.0 / (n_nogs + 1.0)
            };
            let moves = 0.5 * pnog / ((1.0 / n_goodbots as f64) * pb);
            let got = (ratio / moves).ln();
            assert!((got - w).abs() < 1e-12, "depth {d}: {got} vs {w}");
            let (l, _) = tree.birth(eta, 0, 50, sampler.cuts[0][50], 0.0, 0.0);
            eta = l;
        }
    }

    /// The log integrated likelihood of a leaf, the cancelling terms added back, against
    /// numeric integration over mu in R (T4-WP17-ref/lil.R).
    #[test]
    fn bart_integrated_leaf_likelihood_matches_quadrature() {
        let full = |r: &[f64], sigma: f64, t2: f64| {
            let s2 = sigma * sigma;
            let n = r.len() as f64;
            -0.5 * n * (2.0 * std::f64::consts::PI * s2).ln()
                - r.iter().map(|v| v * v).sum::<f64>() / (2.0 * s2)
                + leaf_log_lik(r.len(), r.iter().sum(), s2, t2)
        };
        for (r, sigma, t2, want) in [
            (
                vec![0.3, -1.2, 0.8, 0.1, 0.5],
                1.0,
                1.0 / 800.0,
                -5.8126526613955756,
            ),
            (
                vec![1.9, -0.4, 2.2, 0.7, 1.1, -1.5, 3.0],
                2.5,
                0.09,
                -14.568144379882822,
            ),
        ] {
            let got = full(&r, sigma, t2);
            assert!(((got - want) / want).abs() < 1e-12, "{got} vs {want}");
        }
    }

    /// A birth from the stump and the death that undoes it have reciprocal ratios.
    #[test]
    fn bart_birth_death_ratio_is_reversible() {
        let (sampler, _, _) = kernel_sampler(&BartConfig::default());
        for (v, c) in [(0, 50), (1, 0), (1, 99)] {
            let stump = BartTree::new(0.0);
            let birth = sampler.birth_ratio(&stump, 0, v, (c, 0, 99), 1, 1.0);
            let mut grown = stump.clone();
            grown.birth(0, v, c, sampler.cuts[v][c], 0.0, 0.0);
            let goodbots = grown
                .get_leaves()
                .into_iter()
                .filter(|&l| sampler.can_split(&grown, l))
                .count();
            let death = sampler.death_ratio(&grown, 0, goodbots, 0.5);
            assert!(
                (birth * death - 1.0).abs() < 1e-12,
                "{v} {c}: {}",
                birth * death
            );
        }
        // and deeper: grow the right child of a split root
        let mut tree = BartTree::new(0.0);
        tree.birth(0, 0, 40, sampler.cuts[0][40], 0.0, 0.0);
        let birth = sampler.birth_ratio(&tree, 2, 1, (10, 0, 99), 2, 0.5);
        tree.birth(2, 1, 10, sampler.cuts[1][10], 0.0, 0.0);
        let death = sampler.death_ratio(&tree, 2, 3, 0.5);
        assert!((birth * death - 1.0).abs() < 1e-12, "{}", birth * death);
    }

    #[test]
    fn bart_birth_respects_min_node_size_and_max_depth() {
        let c = BartConfig {
            n_trees: 10,
            min_node_size: 9,
            max_depth: 2,
            seed: Some(5),
            ..Default::default()
        };
        let (mut sampler, xs, yc) = kernel_sampler(&c);
        let mut deepest = 0;
        for _ in 0..150 {
            sampler.step(&yc, &xs);
            for tree in &sampler.trees {
                let mut rows = vec![0usize; tree.nodes.len()];
                for row in xs.chunks(2) {
                    rows[tree.leaf_of(row)] += 1;
                }
                for leaf in tree.get_leaves() {
                    deepest = deepest.max(tree.nodes[leaf].depth);
                    assert!(tree.nodes[leaf].depth <= 2);
                    if tree.nodes.len() > 1 {
                        assert!(rows[leaf] >= 9, "a leaf of {} rows", rows[leaf]);
                    }
                }
            }
        }
        assert_eq!(deepest, 2, "the cap was never reached");
    }

    #[test]
    fn test_config_defaults() {
        let config = BartConfig::default();
        assert_eq!(config.n_trees, 200);
        assert_eq!(config.n_burn, 250);
        assert_eq!(config.n_mcmc, 1000);
        assert_eq!(config.k, 2.0);
        assert_eq!(config.alpha, 0.95);
        assert_eq!(config.beta, 2.0);
    }

    /// A `min_node_size` above `usize::MAX / 2` cannot split any leaf: twice it (the
    /// rows a leaf needs to be split) used to overflow, a panic in a debug build, and
    /// in a release build a product wrapped to a small number that let a grow move go
    /// on to draw a covariate and a cut point for a split it could not make, which
    /// moved the random stream every later move is drawn from. No tree ever splits,
    /// and the fit is exactly the one at a `min_node_size` of all the rows.
    #[test]
    fn a_min_node_size_above_half_of_usize_neither_panics_nor_wraps() {
        let (y, x) = generate_test_data(40, 7);
        let fit = |min_node_size| {
            let cfg = BartConfig {
                n_trees: 10,
                n_burn: 5,
                n_mcmc: 20,
                min_node_size,
                ..tiny()
            };
            bart_arrays(y.view(), x.view(), None, cfg).unwrap()
        };
        let bits = |v: &[f64]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        let summary = |r: &BartResult| {
            let mut v = r.predictions.clone();
            v.extend(&r.prediction_lower);
            v.extend(&r.prediction_upper);
            v.extend(&r.prediction_sd);
            v.extend(&r.variable_importance);
            v.extend([r.sigma, r.sigma_sd]);
            bits(&v)
        };
        // With no split ever made the importance is uniform; at the default the
        // sampler does split these data.
        let uniform = bits(&[1.0 / 3.0; 3]);
        assert_ne!(bits(&fit(5).variable_importance), uniform);
        let want = fit(40);
        assert_eq!(bits(&want.variable_importance), uniform);
        for min_node_size in [usize::MAX / 2 + 1, usize::MAX - 1, usize::MAX] {
            assert_eq!(
                summary(&fit(min_node_size)),
                summary(&want),
                "min_node_size {min_node_size}"
            );
        }

        // bart, the Dataset entry point
        let ds = Dataset::new(
            polars::prelude::df! {
                "y" => y.to_vec(),
                "x1" => x.column(0).to_vec(),
                "x2" => x.column(1).to_vec(),
                "x3" => x.column(2).to_vec(),
            }
            .unwrap(),
        );
        let cfg = BartConfig {
            min_node_size: usize::MAX,
            ..tiny()
        };
        let got = bart(&ds, "y", &["x1", "x2", "x3"], cfg).unwrap();
        assert_eq!(bits(&got.variable_importance), uniform);

        // The rule itself: the product saturates above `usize::MAX / 2` and is
        // exact up to it.
        assert_eq!(min_rows_to_split(usize::MAX / 2 + 1), usize::MAX);
        assert_eq!(min_rows_to_split(usize::MAX), usize::MAX);
        assert_eq!(min_rows_to_split(usize::MAX / 2), usize::MAX - 1);
        assert_eq!((min_rows_to_split(0), min_rows_to_split(5)), (0, 10));
    }
}
