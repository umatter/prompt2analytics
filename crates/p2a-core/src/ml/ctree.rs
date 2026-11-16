//! Conditional Inference Trees (ctree) implementation.
//!
//! Pure Rust implementation of conditional inference trees based on
//! Hothorn, Hornik & Zeileis (2006).
//!
//! Follows `partykit::ctree` (partykit 1.3.1 / libcoin 1.0.13) for numeric inputs; the
//! regression, classification, null and Monte Carlo cases of `validation/r/cases/ml_trees_t4.R`
//! pin it.
//!
//! ## Algorithm
//!
//! At each node of `n` rows:
//!
//! 1. **Variable tests.** Each input is tested for independence from the response with the
//!    conditional linear statistic of Strasser and Weber (numeric input untransformed; the
//!    response itself, or class indicators for a classification). `quadratic`:
//!    `(n - 1) r^2` (regression, 1 df) or `(n - 1) SSB / Sxx` (classification, df = classes in
//!    the node - 1), with asymptotic chi-squared p-values; `maximum` (regression only): `|z|`
//!    with a two-sided normal p-value.
//! 2. **Untestable inputs.** An input whose statistic has variance below `sqrt(eps)` (constant
//!    in the node) is not tested and does not count in the adjustment.
//! 3. **P-values.** `bonferroni`: partykit's adjustment `1 - (1 - p)^m` over the `m` tested
//!    inputs; `univariate`: none; `montecarlo`: the share of `nresample` permutations of the
//!    response whose statistic reaches the observed one, unadjusted.
//! 4. **Selection and stopping.** The input with the smallest adjusted p-value (the larger
//!    statistic, then the first, on ties). The node is a leaf when no adjusted p-value is below
//!    `1 - mincriterion`, when it has fewer than `minsplit` rows, or at `maxdepth`.
//! 5. **Node minimum.** Each child keeps at least `max(minbucket, ceil(0.01 n))` rows.
//! 6. **Split point.** The observed value `u` of the selected input (left: `x <= u`) that
//!    maximises the two-sample statistic of `I(x <= u)`: the variance-reduction cut for a
//!    regression, Pearson's X^2 for a classification; the first cut on ties. When the selected
//!    input has no admissible cut, the next significant input is tried (partykit's
//!    `splittry = 2`).
//! 7. **Leaves.** The mean (regression) or the most frequent class, the first on ties.
//!
//! The task comes from the target: a numeric target is a regression unless
//! `classification = Some(true)`. Variable importance is the sum of `1 - adjusted p` over the
//! splits on each input, not partykit's permutation `varimp`.
//!
//! ## Example
//!
//! ```rust,no_run
//! use p2a_core::ml::{ctree, CtreeConfig};
//! use ndarray::array;
//!
//! let x = array![[1.0], [2.0], [3.0], [7.0], [8.0], [9.0]];
//! let y = array![1.0, 1.0, 1.0, 9.0, 9.0, 9.0];
//!
//! let config = CtreeConfig::default();
//! let result = ctree(x.view(), y.view(), &config).unwrap();
//! println!("Tree depth: {}", result.depth);
//! ```
//!
//! ## References
//!
//! - Hothorn, T., Hornik, K., & Zeileis, A. (2006). "Unbiased Recursive Partitioning:
//!   A Conditional Inference Framework". Journal of Computational and Graphical
//!   Statistics, 15(3), 651-674. <https://doi.org/10.1198/106186006X133933>
//! - Strasser, H., & Weber, C. (1999). "On the asymptotic theory of permutation
//!   statistics". Mathematical Methods of Statistics, 8, 220-250.
//! - R package `partykit`: Hothorn, T. & Zeileis, A. (2015).
//!   <https://cran.r-project.org/package=partykit>

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use super::extract::class_labels_line;
use crate::Dataset;
use crate::errors::{EconError, EconResult};
use crate::traits::estimator::{chi_squared_p_value, normal_p_value_two_sided};

/// Configuration for Conditional Inference Trees.
///
/// # Parameters
///
/// - `mincriterion`: The value of the test statistic or 1 - p-value that must be
///   exceeded to implement a split. Default is 0.95, meaning p-value < 0.05.
/// - `minsplit`: Minimum number of observations in a node required to attempt a split.
///   Default is 20.
/// - `minbucket`: Minimum number of observations in a terminal node. Default is 7.
/// - `maxdepth`: Maximum depth of the tree, at most 62 (`usize::BITS - 2`,
///   because node ids are `2^depth`); 0 means that limit. Default is 0.
/// - `teststat`: Type of test statistic: "quadratic" (chi-squared; also "quad") or "maximum"
///   (also "max"; regression trees only). Default is "quadratic".
/// - `testtype`: P-value computation method: "bonferroni" (partykit's adjustment,
///   `1 - (1 - p)^m`), "univariate" (also "none"), or "montecarlo" (permutation p-values from
///   `nresample` permutations, unadjusted). Default is "bonferroni".
/// - `seed`: Random seed of the Monte Carlo permutations (42 when `None`).
/// - `classification`: whether the target's distinct values are classes (default: regression).
/// - `nresample`: permutations per Monte Carlo test. Default is 9999 (partykit's).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtreeConfig {
    /// 1 - alpha threshold for splitting (e.g., 0.95 means p < 0.05)
    pub mincriterion: f64,
    /// Minimum observations in a node to attempt split
    pub minsplit: usize,
    /// Minimum observations in terminal nodes
    pub minbucket: usize,
    /// Maximum tree depth, at most 62 (0 = that limit)
    pub maxdepth: usize,
    /// Test statistic type: "quadratic" (also "quad") or "maximum" (also "max"; regression trees only)
    pub teststat: String,
    /// P-values: "bonferroni", "univariate" (also "none") or "montecarlo"
    pub testtype: String,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Whether the target's distinct values are classes (R: `factor(y)`). [`ctree`] fits a
    /// regression tree for `None` and `Some(false)`; [`run_ctree`] sets `Some(true)` for a
    /// text, categorical or boolean target.
    #[serde(default)]
    pub classification: Option<bool>,
    /// Permutations per test for `testtype = "montecarlo"` (partykit's `nresample`). Default 9999.
    #[serde(default = "default_nresample")]
    pub nresample: usize,
}

/// partykit's default `nresample`.
fn default_nresample() -> usize {
    9999
}

impl Default for CtreeConfig {
    fn default() -> Self {
        CtreeConfig {
            mincriterion: 0.95,
            minsplit: 20,
            minbucket: 7,
            maxdepth: 0, // unlimited
            teststat: "quadratic".to_string(),
            testtype: "bonferroni".to_string(),
            seed: None,
            classification: None,
            nresample: default_nresample(),
        }
    }
}

/// The test statistic of the variable selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CtreeTestStat {
    /// The squared standardized statistic, chi-squared with one degree of freedom.
    #[default]
    Quadratic,
    /// The absolute standardized statistic, two-sided normal (regression trees only).
    Maximum,
}

crate::option_enum!(
    CtreeTestStat,
    crate::options::Choices {
        option: "teststat",
        values: &[
            (&["quadratic", "quad"], CtreeTestStat::Quadratic),
            (&["maximum", "max"], CtreeTestStat::Maximum),
        ],
    }
);

/// The p-value adjustment of the variable selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CtreeTestType {
    /// partykit's Bonferroni adjustment, `1 - (1 - p)^m` over the m tested inputs.
    #[default]
    Bonferroni,
    /// No adjustment.
    Univariate,
    /// Monte Carlo permutation p-values (`nresample` permutations), unadjusted as partykit
    /// leaves them for this test type.
    MonteCarlo,
}

crate::option_enum!(
    CtreeTestType,
    crate::options::Choices {
        option: "testtype",
        values: &[
            (&["bonferroni"], CtreeTestType::Bonferroni),
            (&["univariate", "none"], CtreeTestType::Univariate),
            (&["montecarlo"], CtreeTestType::MonteCarlo),
        ],
    }
);

/// `teststat` and `testtype` as parsed by [`ctree`].
#[derive(Clone, Copy)]
struct CtreeOptions {
    teststat: CtreeTestStat,
    testtype: CtreeTestType,
    /// Permutations per Monte Carlo test.
    nresample: usize,
    /// The seed of the Monte Carlo permutations (42 when the config has none).
    seed: u64,
}

/// A split in the conditional inference tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtreeSplit {
    /// Feature index used for split
    pub feature: usize,
    /// Split threshold
    pub threshold: f64,
    /// Test statistic value for variable selection
    pub statistic: f64,
    /// P-value for variable selection
    pub p_value: f64,
}

/// A node in the conditional inference tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtreeNode {
    /// Node ID (1-indexed, root = 1)
    pub id: usize,
    /// Number of observations in this node
    pub n: usize,
    /// Predicted value (mean for regression, majority class for classification)
    pub prediction: f64,
    /// Class probabilities (for classification)
    pub class_probs: Option<Vec<f64>>,
    /// Split information (None for terminal nodes)
    pub split: Option<CtreeSplit>,
    /// Left child node
    pub left: Option<Box<CtreeNode>>,
    /// Right child node
    pub right: Option<Box<CtreeNode>>,
}

/// Result from conditional inference tree fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtreeResult {
    /// Root node of the tree
    pub root: CtreeNode,
    /// Total number of nodes
    pub n_nodes: usize,
    /// Number of terminal nodes
    pub n_terminal: usize,
    /// Maximum depth reached
    pub depth: usize,
    /// Sum over the splits on the variable of 1 - (adjusted) p-value; not partykit's permutation
    /// `varimp`.
    pub variable_importance: Vec<f64>,
    /// The test statistic of each input at the root (NaN: not tested, the input is constant
    /// there). Computed also when the root is not split.
    #[serde(default)]
    pub root_statistics: Vec<f64>,
    /// The unadjusted p-value of each input at the root (NaN: not tested).
    pub root_p_values: Vec<f64>,
    /// The adjusted p-value of each input at the root, the value the split decision compares with
    /// `1 - mincriterion` (NaN: not tested).
    #[serde(default)]
    pub root_p_values_adjusted: Vec<f64>,
    /// Whether the model is for classification
    pub is_classification: bool,
    /// Number of classes (for classification)
    pub n_classes: Option<usize>,
    /// Class labels (for classification)
    pub class_labels: Option<Vec<f64>>,
    /// Labels of the class codes when the target column was text, categorical or
    /// boolean; code k is `class_names[k]` ([`run_ctree`] only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class_names: Option<Vec<String>>,
    /// Configuration used
    pub config: CtreeConfig,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// Training predictions
    pub predictions: Vec<f64>,
}

impl std::fmt::Display for CtreeResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Conditional Inference Tree")?;
        writeln!(f, "==========================")?;
        writeln!(
            f,
            "Type: {}",
            if self.is_classification {
                "Classification"
            } else {
                "Regression"
            }
        )?;
        if let Some(names) = &self.class_names {
            writeln!(f, "{}", class_labels_line(names))?;
        }
        writeln!(f, "Nodes: {} ({} terminal)", self.n_nodes, self.n_terminal)?;
        writeln!(f, "Depth: {}", self.depth)?;
        writeln!(
            f,
            "Criterion: 1 - p-value > {:.2} (p < {:.4})",
            self.config.mincriterion,
            1.0 - self.config.mincriterion
        )?;

        writeln!(f)?;
        writeln!(f, "Variable Importance:")?;

        // Sort by importance
        let mut indexed: Vec<(usize, f64)> = self
            .variable_importance
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Normalize
        let total: f64 = indexed.iter().map(|(_, v)| v).sum();

        for (i, importance) in indexed.iter().take(10) {
            let name = match &self.feature_names {
                Some(names) => names.get(*i).cloned().unwrap_or_else(|| format!("X{}", i)),
                None => format!("X{}", i),
            };
            let pct = if total > 0.0 {
                importance / total * 100.0
            } else {
                0.0
            };
            writeln!(f, "  {}: {:.1}%", name, pct)?;
        }

        if self.variable_importance.len() > 10 {
            writeln!(
                f,
                "  ... ({} more features)",
                self.variable_importance.len() - 10
            )?;
        }

        // Print tree structure summary
        writeln!(f)?;
        writeln!(f, "Root test statistics:")?;
        for (i, &p) in self.root_p_values.iter().enumerate() {
            let name = match &self.feature_names {
                Some(names) => names.get(i).cloned().unwrap_or_else(|| format!("X{}", i)),
                None => format!("X{}", i),
            };
            let stat = self.root_statistics.get(i).copied().unwrap_or(f64::NAN);
            let adjusted = self
                .root_p_values_adjusted
                .get(i)
                .copied()
                .unwrap_or(f64::NAN);
            if p.is_nan() {
                writeln!(f, "  {}: not tested (constant in the node)", name)?;
            } else {
                writeln!(
                    f,
                    "  {}: statistic = {:.4}, p-value = {}, adjusted = {}",
                    name,
                    stat,
                    format_p(p),
                    format_p(adjusted)
                )?;
            }
        }

        Ok(())
    }
}

/// A p-value with 4 decimals, or in scientific notation below 1e-4.
fn format_p(p: f64) -> String {
    if p < 1e-4 {
        format!("{p:.3e}")
    } else {
        format!("{p:.4}")
    }
}

/// partykit's tolerance below which an input is not tested (`ctree_control(tol = sqrt(eps))`).
///
/// An input is untestable when every diagonal entry of the conditional covariance of its linear
/// statistic is below this absolute tolerance; it then gets statistic and p-value NaN and does not
/// count in the Bonferroni adjustment. The tolerance is absolute, as in partykit, so it depends on
/// the scale of the data.
const UNTESTABLE_TOL: f64 = 1.4901161193847656e-8;

/// The response of one node as the conditional linear statistic of partykit sees it (Strasser and
/// Weber 1999, with the identity transform for a numeric input and a numeric response, and class
/// indicators for a categorical response).
///
/// For an input x of the node's rows, with `Sxx` the centred sum of squares of x:
/// - regression: `c = (n - 1) Sxy^2 / (Sxx Syy)`, chi-squared with 1 df (`maximum`: `|z|` with
///   `z = Sxy / sqrt(Syy Sxx / (n - 1))`, two-sided normal);
/// - classification: `c = (n - 1) SSB / Sxx` with `SSB = sum_k n_k (xbar_k - xbar)^2`, chi-squared
///   with (classes present in the node) - 1 df. This is the quadratic form `d' V^+ d` of the
///   multinomial conditional covariance in closed form.
pub(crate) struct LinearTest {
    n: usize,
    /// Regression: the node's responses minus their mean, in node order.
    dy: Vec<f64>,
    syy: f64,
    /// Classification: the dense class of each node row, in node order, and the class counts.
    class: Vec<usize>,
    counts: Vec<usize>,
    teststat: CtreeTestStat,
}

/// One input's centred values in node order and their sum of squares.
pub(crate) struct CentredInput {
    dx: Vec<f64>,
    sxx: f64,
}

impl CentredInput {
    pub(crate) fn new(values: impl Iterator<Item = f64>) -> Self {
        let xs: Vec<f64> = values.collect();
        let mean = xs.iter().sum::<f64>() / xs.len().max(1) as f64;
        let dx: Vec<f64> = xs.iter().map(|v| v - mean).collect();
        let sxx = dx.iter().map(|d| d * d).sum();
        CentredInput { dx, sxx }
    }
}

impl LinearTest {
    /// The test of a node over `indices` of the response.
    pub(crate) fn new(response: &Response<'_>, indices: &[usize], teststat: CtreeTestStat) -> Self {
        let n = indices.len();
        match response {
            Response::Regression(y) => {
                let mean = indices.iter().map(|&i| y[i]).sum::<f64>() / n.max(1) as f64;
                let dy: Vec<f64> = indices.iter().map(|&i| y[i] - mean).collect();
                let syy = dy.iter().map(|d| d * d).sum();
                LinearTest {
                    n,
                    dy,
                    syy,
                    class: Vec::new(),
                    counts: Vec::new(),
                    teststat,
                }
            }
            Response::Classes { class, k } => {
                let node_class: Vec<usize> = indices.iter().map(|&i| class[i]).collect();
                let mut counts = vec![0usize; *k];
                for &c in &node_class {
                    counts[c] += 1;
                }
                LinearTest {
                    n,
                    dy: Vec::new(),
                    syy: 0.0,
                    class: node_class,
                    counts,
                    teststat,
                }
            }
        }
    }

    fn is_classification(&self) -> bool {
        !self.counts.is_empty()
    }

    /// Whether some diagonal entry of Var(T) reaches [`UNTESTABLE_TOL`] (partykit tests the input).
    fn testable(&self, input: &CentredInput) -> bool {
        if self.n < 2 {
            return false;
        }
        let nm1 = (self.n - 1) as f64;
        if self.is_classification() {
            let n = self.n as f64;
            self.counts.iter().any(|&c| {
                let c = c as f64;
                c * (n - c) / n * input.sxx / nm1 >= UNTESTABLE_TOL
            })
        } else {
            self.syy * input.sxx / nm1 >= UNTESTABLE_TOL
        }
    }

    /// The statistic of `input` against the responses in the order `perm` gives (the node order when
    /// `perm` is `None`). The caller has checked [`LinearTest::testable`].
    fn statistic(&self, input: &CentredInput, perm: Option<&[usize]>) -> f64 {
        let nm1 = (self.n - 1) as f64;
        let at = |i: usize| perm.map_or(i, |p| p[i]);
        if self.is_classification() {
            let mut sums = vec![0.0f64; self.counts.len()];
            for (i, &d) in input.dx.iter().enumerate() {
                sums[self.class[at(i)]] += d;
            }
            let ssb: f64 = sums
                .iter()
                .zip(&self.counts)
                .filter(|&(_, &c)| c > 0)
                .map(|(s, &c)| s * s / c as f64)
                .sum();
            nm1 * ssb / input.sxx
        } else {
            let sxy: f64 = input
                .dx
                .iter()
                .enumerate()
                .map(|(i, &d)| d * self.dy[at(i)])
                .sum();
            let var = self.syy * input.sxx / nm1;
            match self.teststat {
                CtreeTestStat::Quadratic => sxy * sxy / var,
                CtreeTestStat::Maximum => sxy.abs() / var.sqrt(),
            }
        }
    }

    /// The asymptotic p-value of `statistic`.
    fn p_value(&self, statistic: f64) -> f64 {
        if self.is_classification() {
            let present = self.counts.iter().filter(|&&c| c > 0).count();
            chi_squared_p_value(statistic, present.saturating_sub(1).max(1) as f64)
        } else {
            match self.teststat {
                CtreeTestStat::Quadratic => chi_squared_p_value(statistic, 1.0),
                CtreeTestStat::Maximum => normal_p_value_two_sided(statistic),
            }
        }
    }
}

/// The response of [`ctree`]: numeric (regression) or dense class indices 0..k (classification).
pub(crate) enum Response<'a> {
    Regression(ArrayView1<'a, f64>),
    Classes { class: &'a [usize], k: usize },
}

/// The tests of every input at one node: statistic, raw p and adjusted p (NaN: not tested).
#[derive(Debug, Clone)]
struct NodeTests {
    statistic: Vec<f64>,
    p: Vec<f64>,
    p_adjusted: Vec<f64>,
}

impl NodeTests {
    /// The inputs whose adjusted p is below `alpha`, best first: smallest adjusted p, then the larger
    /// statistic, then the first input (partykit's order).
    fn significant(&self, alpha: f64) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.p_adjusted.len())
            .filter(|&j| self.p_adjusted[j] < alpha)
            .collect();
        order.sort_by(|&a, &b| {
            self.p_adjusted[a]
                .total_cmp(&self.p_adjusted[b])
                .then(self.statistic[b].total_cmp(&self.statistic[a]))
                .then(a.cmp(&b))
        });
        order
    }
}

/// The variable-selection tests of the node `node_id` over `indices`, for every input.
///
/// "Bonferroni" is partykit's: `1 - (1 - p)^m` computed as `-expm1(m log1p(-p))`, with m the number
/// of tested inputs. "Univariate" leaves the p-values as they are. "MonteCarlo" draws `nresample`
/// permutations of the node's responses, one per resample shared by every input, and reports
/// `#{b : c*_b >= c} / B`, unadjusted; resample b's permutation is seeded from (seed, node, b).
fn node_tests(
    x: &ArrayView2<f64>,
    response: &Response<'_>,
    indices: &[usize],
    options: CtreeOptions,
    node_id: usize,
) -> EconResult<NodeTests> {
    let p_inputs = x.ncols();
    let test = LinearTest::new(response, indices, options.teststat);
    let mut statistic = vec![f64::NAN; p_inputs];
    let mut p = vec![f64::NAN; p_inputs];
    let monte_carlo = options.testtype == CtreeTestType::MonteCarlo;
    let mut tested: Vec<(usize, CentredInput)> = Vec::new();
    for j in 0..p_inputs {
        let input = CentredInput::new(indices.iter().map(|&i| x[[i, j]]));
        if test.testable(&input) {
            let s = test.statistic(&input, None);
            statistic[j] = s;
            p[j] = test.p_value(s);
            if monte_carlo {
                tested.push((j, input));
            }
        }
    }
    if monte_carlo && !tested.is_empty() {
        let n = indices.len();
        let mut perm = crate::resource::try_vec(n, 0usize, "ctree Monte Carlo permutation")?;
        let mut exceed = vec![0usize; p_inputs];
        let budget = options.nresample;
        for b in 0..budget {
            crate::resource::checkpoint()?;
            for (i, v) in perm.iter_mut().enumerate() {
                *v = i;
            }
            let mut rng = ChaCha8Rng::seed_from_u64(resample_seed(options.seed, node_id, b));
            perm.shuffle(&mut rng);
            for (j, input) in &tested {
                if test.statistic(input, Some(&perm)) >= statistic[*j] {
                    exceed[*j] += 1;
                }
            }
        }
        for (j, _) in &tested {
            p[*j] = exceed[*j] as f64 / budget as f64;
        }
    }
    let m = p.iter().filter(|v| !v.is_nan()).count() as f64;
    let p_adjusted = match options.testtype {
        CtreeTestType::Bonferroni => p.iter().map(|&v| -(m * (-v).ln_1p()).exp_m1()).collect(),
        CtreeTestType::Univariate | CtreeTestType::MonteCarlo => p.clone(),
    };
    Ok(NodeTests {
        statistic,
        p,
        p_adjusted,
    })
}

/// The seed of Monte Carlo resample `b` at node `node`: SplitMix64 finalisers over the three.
fn resample_seed(seed: u64, node: usize, b: usize) -> u64 {
    fn mix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    mix(mix(mix(seed) ^ node as u64) ^ b as u64)
}

/// partykit's default `minprob`: each child of a split holds at least `ceil(n * minprob)` of
/// the node's `n` rows (and at least `minbucket`).
const MINPROB: f64 = 0.01;

/// partykit's default `splittry`: the number of significant inputs, best first, tried for a
/// binary split before the node becomes a leaf.
const SPLITTRY: usize = 2;

/// The least rows of a child of a node of `n` rows: `max(minbucket, ceil(n * minprob))`, and at
/// least one (`minbucket = 0` acts as 1).
fn node_minimum(n: usize, minbucket: usize) -> usize {
    let by_share = (n as f64 * MINPROB).ceil() as usize;
    minbucket.max(by_share).max(1)
}

/// The cut of `feature` that maximises partykit's two-sample quadratic statistic of `I(x <= u)`
/// over the rows in `indices`, among the observed values u (except the largest) that leave at
/// least `mb` rows on each side; the first in ascending order on ties.
///
/// - Regression: `(S_L - n_L ybar)^2 / (n_L n_R)` up to a constant factor (the between-group sum
///   of squares, so the variance-reduction cut).
/// - Classification: `sum_k (n_Lk - n_L p_k)^2 / p_k / (n_L n_R)` up to a constant factor
///   (Pearson's X^2 of the 2 x K table), over the classes present.
///
/// The rows are sorted once and scanned with running sums (regression, centred on the node mean)
/// or running class counts. Returns the break u (left: `x <= u`) and the left and right row
/// indices, each in sorted order.
fn find_best_split(
    x: &ArrayView2<f64>,
    response: &Response<'_>,
    indices: &[usize],
    feature: usize,
    mb: usize,
) -> Option<(f64, Vec<usize>, Vec<usize>)> {
    let n = indices.len();
    let mb = mb.max(1);
    if n < mb.saturating_mul(2) {
        return None;
    }
    let mut sorted: Vec<(f64, usize)> = indices.iter().map(|&i| (x[[i, feature]], i)).collect();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let is_candidate = |pos: usize| sorted[pos - 1].0 != sorted[pos].0;
    let mut best_pos: Option<usize> = None;
    let mut best_stat = f64::NEG_INFINITY;

    match response {
        Response::Regression(y) => {
            let mean = indices.iter().map(|&i| y[i]).sum::<f64>() / n as f64;
            let mut sum_left = 0.0f64;
            let mut filled = 0usize;
            for pos in mb..=n - mb {
                while filled < pos {
                    sum_left += y[sorted[filled].1] - mean;
                    filled += 1;
                }
                if !is_candidate(pos) {
                    continue;
                }
                let stat = sum_left * sum_left / (pos as f64 * (n - pos) as f64);
                if stat > best_stat {
                    best_stat = stat;
                    best_pos = Some(pos);
                }
            }
        }
        Response::Classes { class, k } => {
            let mut total = vec![0usize; *k];
            for &(_, i) in &sorted {
                total[class[i]] += 1;
            }
            let nf = n as f64;
            let mut left = vec![0usize; *k];
            let mut filled = 0usize;
            for pos in mb..=n - mb {
                while filled < pos {
                    left[class[sorted[filled].1]] += 1;
                    filled += 1;
                }
                if !is_candidate(pos) {
                    continue;
                }
                let stat = pearson_split_statistic(&left, &total, pos, nf);
                if stat > best_stat {
                    best_stat = stat;
                    best_pos = Some(pos);
                }
            }
        }
    }

    best_pos.map(|pos| {
        let threshold = sorted[pos - 1].0;
        let left_indices: Vec<usize> = sorted[..pos].iter().map(|(_, i)| *i).collect();
        let right_indices: Vec<usize> = sorted[pos..].iter().map(|(_, i)| *i).collect();
        (threshold, left_indices, right_indices)
    })
}

/// `sum_k (n_Lk - n_L p_k)^2 / p_k / (n_L n_R)` over the classes present (`p_k = total_k / n`).
fn pearson_split_statistic(left: &[usize], total: &[usize], n_left: usize, n: f64) -> f64 {
    let nl = n_left as f64;
    let mut sum = 0.0;
    for (&l, &t) in left.iter().zip(total) {
        if t > 0 {
            let p = t as f64 / n;
            let d = l as f64 - nl * p;
            sum += d * d / p;
        }
    }
    sum / (nl * (n - nl))
}

/// Deepest `maxdepth` ctree accepts (`usize::BITS - 2`, 62 on 64-bit targets).
///
/// Node ids are `2^depth` (the root is 1, children of `k` are `2k` and
/// `2k + 1`), so a deeper chain would overflow the id; this is the same reason
/// rpart caps `maxdepth` at 30 for its 32-bit ids. It also bounds the recursion
/// depth. A `maxdepth` of 0 means this limit.
const MAX_CTREE_DEPTH: usize = usize::BITS as usize - 2;

/// A terminal node.
fn ctree_leaf(id: usize, n: usize, prediction: f64, class_probs: Option<Vec<f64>>) -> CtreeNode {
    CtreeNode {
        id,
        n,
        prediction,
        class_probs,
        split: None,
        left: None,
        right: None,
    }
}

/// Build a conditional inference tree recursively.
///
/// The root's tests are always computed and stored in `root_tests`, also when the root is not
/// split (stopped by minsplit, maxdepth or no significant input).
fn build_ctree(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    response: &Response<'_>,
    indices: &[usize],
    classes: Option<&[f64]>,
    config: &CtreeConfig,
    options: CtreeOptions,
    depth: usize,
    node_id: usize,
    importance: &mut Array1<f64>,
    root_tests: &mut Option<NodeTests>,
) -> EconResult<CtreeNode> {
    crate::resource::checkpoint()?;
    let n = indices.len();

    // Compute node prediction
    let (prediction, class_probs) = match (response, classes) {
        (Response::Classes { class, k }, Some(cls)) => {
            let mut counts = vec![0usize; *k];
            for &i in indices {
                counts[class[i]] += 1;
            }
            let total = counts.iter().sum::<usize>() as f64;
            let probs: Vec<f64> = counts.iter().map(|&c| c as f64 / total.max(1.0)).collect();
            // The first class with the largest share, as which.max.
            let mut best_class_idx = 0;
            for (i, &c) in counts.iter().enumerate() {
                if c > counts[best_class_idx] {
                    best_class_idx = i;
                }
            }
            (cls[best_class_idx], Some(probs))
        }
        _ => {
            let sum: f64 = indices.iter().map(|&i| y[i]).sum();
            (sum / n.max(1) as f64, None)
        }
    };

    let max_depth = if config.maxdepth == 0 {
        MAX_CTREE_DEPTH
    } else {
        config.maxdepth
    };
    let stop = depth >= max_depth || n < config.minsplit;
    if stop && depth > 0 {
        return Ok(ctree_leaf(node_id, n, prediction, class_probs));
    }

    // Variable selection: the conditional linear statistic of every input.
    let tests = node_tests(x, response, indices, options, node_id)?;
    if depth == 0 {
        *root_tests = Some(tests.clone());
    }
    if stop {
        return Ok(ctree_leaf(node_id, n, prediction, class_probs));
    }
    // The most significant input with an admissible cut, among the first SPLITTRY.
    let alpha = 1.0 - config.mincriterion;
    let mb = node_minimum(n, config.minbucket);
    let Some((best_var, (threshold, left_indices, right_indices))) = tests
        .significant(alpha)
        .into_iter()
        .take(SPLITTRY)
        .find_map(|j| find_best_split(x, response, indices, j, mb).map(|split| (j, split)))
    else {
        return Ok(ctree_leaf(node_id, n, prediction, class_probs));
    };
    let (best_stat, best_pval) = (tests.statistic[best_var], tests.p_adjusted[best_var]);

    // Variable importance: 1 - adjusted p-value of the split
    importance[best_var] += 1.0 - best_pval;

    let left_child = build_ctree(
        x,
        y,
        response,
        &left_indices,
        classes,
        config,
        options,
        depth + 1,
        node_id * 2,
        importance,
        root_tests,
    )?;
    let right_child = build_ctree(
        x,
        y,
        response,
        &right_indices,
        classes,
        config,
        options,
        depth + 1,
        node_id * 2 + 1,
        importance,
        root_tests,
    )?;

    Ok(CtreeNode {
        id: node_id,
        n,
        prediction,
        class_probs,
        split: Some(CtreeSplit {
            feature: best_var,
            threshold,
            statistic: best_stat,
            p_value: best_pval,
        }),
        left: Some(Box::new(left_child)),
        right: Some(Box::new(right_child)),
    })
}

/// Count tree statistics recursively.
fn count_tree_stats(node: &CtreeNode) -> (usize, usize, usize) {
    if node.split.is_none() {
        return (1, 1, 0);
    }

    let (left_nodes, left_term, left_depth) = match &node.left {
        Some(child) => count_tree_stats(child),
        None => (0, 0, 0),
    };

    let (right_nodes, right_term, right_depth) = match &node.right {
        Some(child) => count_tree_stats(child),
        None => (0, 0, 0),
    };

    (
        1 + left_nodes + right_nodes,
        left_term + right_term,
        1 + left_depth.max(right_depth),
    )
}

/// Predict for a single observation.
fn predict_one(node: &CtreeNode, x: &ArrayView1<f64>) -> f64 {
    match &node.split {
        None => node.prediction,
        Some(split) => {
            if x[split.feature] <= split.threshold {
                predict_one(node.left.as_ref().unwrap(), x)
            } else {
                predict_one(node.right.as_ref().unwrap(), x)
            }
        }
    }
}

/// Fit a Conditional Inference Tree.
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features)
/// * `y` - Target values (n_samples): a regression tree, unless `config.classification` is
///   `Some(true)`, when the distinct values are the classes (at most `max(2, min(1000, n / 2))`)
/// * `config` - Tree configuration
///
/// # Returns
///
/// CtreeResult containing the fitted tree and diagnostics.
///
/// # Errors
///
/// `Computation` for fewer than 2 rows, when `x` and `y` differ in length, or
/// when `mincriterion` is not strictly inside (0, 1), NaN included;
/// `InvalidSpecification` for an unknown `teststat` or `testtype`, for partykit's
/// `Teststatistic` test type (not implemented), for `testtype = montecarlo` with `nresample` 0,
/// for `minsplit` below
/// `2 * minbucket` (no node of that size can be split into two children of `minbucket` rows), for
/// `teststat = maximum` with a classification target, for more classes than
/// `max(2, min(1000, n / 2))`, when `x` or `y` holds a non-finite value or
/// `maxdepth` exceeds 62 (`usize::BITS - 2` on 64-bit targets, since node ids
/// double at each level); `Cancelled` once the installed computation budget
/// expires (checked per node).
///
/// # Example
///
/// ```rust,no_run
/// use p2a_core::ml::{ctree, CtreeConfig};
/// use ndarray::array;
///
/// // Regression example
/// let x = array![[1.0], [2.0], [3.0], [7.0], [8.0], [9.0]];
/// let y = array![1.0, 1.0, 1.0, 9.0, 9.0, 9.0];
///
/// let config = CtreeConfig::default();
/// let result = ctree(x.view(), y.view(), &config).unwrap();
/// ```
///
/// # References
///
/// - Hothorn, T., Hornik, K., & Zeileis, A. (2006). "Unbiased Recursive Partitioning:
///   A Conditional Inference Framework". Journal of Computational and Graphical
///   Statistics, 15(3), 651-674.
pub fn ctree(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &CtreeConfig,
) -> EconResult<CtreeResult> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 2 {
        return Err(EconError::Computation(
            "Need at least 2 samples for ctree".to_string(),
        ));
    }
    if n_samples != y.len() {
        return Err(EconError::Computation(
            "X and y must have same number of samples".to_string(),
        ));
    }
    if !(config.mincriterion > 0.0 && config.mincriterion < 1.0) {
        return Err(EconError::Computation(
            "mincriterion must be between 0 and 1 (e.g., 0.95 for p < 0.05)".to_string(),
        ));
    }
    let teststat: CtreeTestStat = config.teststat.parse()?;
    let normalized_testtype = crate::options::normalize(&config.testtype);
    if normalized_testtype == "teststatistic" {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "ctree: testtype '{}' (partykit) is not implemented; use bonferroni, univariate \
                 or montecarlo",
                config.testtype
            ),
        });
    }
    let testtype: CtreeTestType = config.testtype.parse()?;
    if testtype == CtreeTestType::MonteCarlo && config.nresample == 0 {
        return Err(EconError::InvalidSpecification {
            message: "ctree: testtype = montecarlo needs nresample of at least 1 permutation"
                .to_string(),
        });
    }
    let options = CtreeOptions {
        teststat,
        testtype,
        nresample: config.nresample,
        seed: config.seed.unwrap_or(42),
    };
    if config
        .minbucket
        .checked_mul(2)
        .is_none_or(|twice| config.minsplit < twice)
    {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "ctree: minsplit ({}) must be at least 2 * minbucket ({}); a node with fewer rows \
                 cannot be split into two children of minbucket rows",
                config.minsplit,
                config.minbucket.saturating_mul(2)
            ),
        });
    }
    if config.maxdepth > MAX_CTREE_DEPTH {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "maxdepth must be at most {MAX_CTREE_DEPTH} (node ids double at each level); got {}",
                config.maxdepth
            ),
        });
    }
    for (j, column) in x.axis_iter(Axis(1)).enumerate() {
        let k = column.iter().filter(|v| !v.is_finite()).count();
        if k > 0 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "feature column {j} (0-based) has {k} non-finite values (NaN or infinite); \
                     remove or impute them first"
                ),
            });
        }
    }
    let k = y.iter().filter(|v| !v.is_finite()).count();
    if k > 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the target has {k} non-finite values (NaN or infinite); remove or impute them first"
            ),
        });
    }

    // The task comes from the target, not from its values: classes only on request.
    let is_classification = config.classification == Some(true);
    if teststat == CtreeTestStat::Maximum && is_classification {
        return Err(EconError::InvalidSpecification {
            message: "ctree: teststat = maximum is implemented for regression trees only (the \
                      classification statistic is the quadratic form); use teststat = quadratic"
                .to_string(),
        });
    }

    let (classes, dense) = if is_classification {
        let (labels, dense) = class_indices(&y)?;
        (Some(labels), dense)
    } else {
        (None, Vec::new())
    };

    let indices: Vec<usize> = (0..n_samples).collect();
    let mut importance = Array1::zeros(n_features);

    let response = match &classes {
        Some(cls) => Response::Classes {
            class: &dense,
            k: cls.len(),
        },
        None => Response::Regression(y.view()),
    };
    let mut root_tests = None;
    let root = build_ctree(
        &x,
        &y,
        &response,
        &indices,
        classes.as_deref(),
        config,
        options,
        0,
        1,
        &mut importance,
        &mut root_tests,
    )?;
    let root_tests = root_tests.unwrap_or_else(|| NodeTests {
        statistic: vec![f64::NAN; n_features],
        p: vec![f64::NAN; n_features],
        p_adjusted: vec![f64::NAN; n_features],
    });

    let (n_nodes, n_terminal, depth) = count_tree_stats(&root);

    // Generate predictions
    let predictions: Vec<f64> = (0..n_samples)
        .map(|i| predict_one(&root, &x.row(i)))
        .collect();

    Ok(CtreeResult {
        root,
        n_nodes,
        n_terminal,
        depth,
        variable_importance: importance.to_vec(),
        root_statistics: root_tests.statistic,
        root_p_values: root_tests.p,
        root_p_values_adjusted: root_tests.p_adjusted,
        is_classification,
        n_classes: classes.as_ref().map(|c| c.len()),
        class_labels: classes,
        class_names: None,
        config: config.clone(),
        feature_names: None,
        predictions,
    })
}

/// The refusal of `k` classes in `n` rows, above the cap `max(2, min(1000, n / 2))` that C5.0 and
/// CART use, or `None` within it.
fn class_cap_error(k: usize, n: usize, target: &str) -> Option<EconError> {
    let cap = super::c50::max_classes(n);
    (k > cap).then(|| EconError::InvalidSpecification {
        message: format!(
            "ctree fits a classification tree for at most {cap} classes for {n} rows \
             (max(2, min(1000, n / 2)), n / 2 rounded down ({} for {n} rows)); {target} has {k} \
             classes. Merge rare classes, or fit a regression tree",
            n / 2
        ),
    })
}

/// The distinct values of `y` as classes, in ascending order (-0.0 and 0.0 are one class), and
/// each row's class index.
fn class_indices(y: &ArrayView1<f64>) -> EconResult<(Vec<f64>, Vec<usize>)> {
    let canonical = |v: f64| if v == 0.0 { 0.0 } else { v };
    let mut labels: Vec<f64> = y.iter().map(|&v| canonical(v)).collect();
    labels.sort_by(f64::total_cmp);
    labels.dedup_by(|a, b| a.to_bits() == b.to_bits());
    if let Some(e) = class_cap_error(labels.len(), y.len(), "the target") {
        return Err(e);
    }
    let dense = y
        .iter()
        .map(|&v| {
            let v = canonical(v);
            labels
                .binary_search_by(|c| c.total_cmp(&v))
                .unwrap_or_default()
        })
        .collect();
    Ok((labels, dense))
}

/// Predict using a fitted ctree.
///
/// # Arguments
///
/// * `result` - Fitted ctree result
/// * `x` - Feature matrix for prediction
///
/// # Returns
///
/// Predictions for each observation.
pub fn ctree_predict(result: &CtreeResult, x: ArrayView2<f64>) -> EconResult<Vec<f64>> {
    Ok((0..x.nrows())
        .map(|i| predict_one(&result.root, &x.row(i)))
        .collect())
}

/// Predict class probabilities for classification trees.
///
/// # Arguments
///
/// * `result` - Fitted ctree result (must be classification)
/// * `x` - Feature matrix for prediction
///
/// # Returns
///
/// Class probability matrix (n_samples x n_classes).
pub fn ctree_predict_proba(result: &CtreeResult, x: ArrayView2<f64>) -> EconResult<Array2<f64>> {
    if !result.is_classification {
        return Err(EconError::Computation(
            "predict_proba only available for classification trees".to_string(),
        ));
    }

    let n_classes = result.n_classes.unwrap_or(2);
    let n_samples = x.nrows();

    let mut probs = Array2::zeros((n_samples, n_classes));

    for i in 0..n_samples {
        let node = find_terminal_node(&result.root, &x.row(i));
        if let Some(cp) = &node.class_probs {
            for (j, &p) in cp.iter().enumerate() {
                probs[[i, j]] = p;
            }
        }
    }

    Ok(probs)
}

/// Find the terminal node for a given observation.
fn find_terminal_node<'a>(node: &'a CtreeNode, x: &ArrayView1<f64>) -> &'a CtreeNode {
    match &node.split {
        None => node,
        Some(split) => {
            if x[split.feature] <= split.threshold {
                find_terminal_node(node.left.as_ref().unwrap(), x)
            } else {
                find_terminal_node(node.right.as_ref().unwrap(), x)
            }
        }
    }
}

/// Run ctree on a Dataset.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `y_col` - Target column: numeric (integer and decimal columns; a regression tree unless
///   `config.classification` is `Some(true)`, as in [`ctree`]), or a text, categorical or
///   boolean column, always a classification tree, whose labels become the class codes 0..K-1
///   in R `factor()` order (reported in [`CtreeResult::class_names`]). A missing value
///   (null or NaN) in the target or a feature is an error naming the column
///   (`EconError::NullValues`)
/// * `x_cols` - Feature column names
/// * `config` - Tree configuration
///
/// # Returns
///
/// CtreeResult with fitted tree and diagnostics. A classification target with more than
/// `max(2, min(1000, n / 2))` classes (the cap of C5.0 and CART), or a text, categorical or
/// boolean target with `classification = Some(false)`, is `EconError::InvalidSpecification`.
pub fn run_ctree(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &CtreeConfig,
) -> EconResult<CtreeResult> {
    let (x, target, _missing) = super::extract::extract_x_classes(
        dataset,
        y_col,
        x_cols,
        crate::data::columns::NaPolicy::Fail,
    )?;
    let mut config = config.clone();
    if let Some(labels) = &target.labels {
        if config.classification == Some(false) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "the target '{y_col}' is a text, categorical or boolean column, so ctree fits \
                     a classification tree; classification = false needs a numeric target"
                ),
            });
        }
        if let Some(e) = class_cap_error(labels.len(), x.nrows(), &format!("the target '{y_col}'"))
        {
            return Err(e);
        }
        config.classification = Some(true);
    }
    let feature_names: Vec<String> = x_cols.iter().map(|s| s.to_string()).collect();

    let mut result = ctree(x.view(), target.y.view(), &config)?;
    result.feature_names = Some(feature_names);
    result.class_names = target.labels;

    Ok(result)
}

/// Run ctree with default configuration.
pub fn run_ctree_default(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<CtreeResult> {
    run_ctree(dataset, y_col, x_cols, &CtreeConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{Budget, CancelToken, with_budget};
    use ndarray::array;

    /// Relative-only tolerance for an upper tail: Tol::p_value's absolute floor would accept
    /// 0 for a tail far below 1e-16.
    fn tail(n: usize) -> p2a_validation::Tol {
        p2a_validation::Tol::new(p2a_validation::Tol::p_value(n).rel, 0.0)
    }

    /// The columns `names` of a case dataset as a matrix, and its column `y`.
    /// `validation/datasets/generated/<name>.csv`, for property tests that pin no R value.
    fn generated_dataset(name: &str) -> crate::Dataset {
        let path = p2a_validation::repo_path(&format!("validation/datasets/generated/{name}.csv"));
        match crate::data::DataLoader::load_csv(&path) {
            Ok(df) => crate::Dataset::new(df),
            Err(e) => panic!("cannot load {}: {e}", path.display()),
        }
    }

    fn case_xy(ds: &crate::Dataset, names: &[&str], y: &str) -> (Array2<f64>, Array1<f64>) {
        use crate::data::columns::{NaPolicy, numeric_column};
        let col = |c: &str| Array1::from(numeric_column(ds.df(), c, NaPolicy::Fail).unwrap().0);
        let cols: Vec<Array1<f64>> = names.iter().map(|c| col(c)).collect();
        let views: Vec<_> = cols.iter().map(|c| c.view()).collect();
        (ndarray::stack(Axis(1), &views).unwrap(), col(y))
    }

    #[test]
    fn test_validate_ctree_root_tail_tie_break() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("tail_ctree_root");
        let (x, y) = case_xy(&ds, &["x1", "x2", "x3"], "y");
        let r = ctree(x.view(), y.view(), &CtreeConfig::default()).unwrap();
        let split = r.root.split.as_ref().expect("root split");
        Check::new(&case)
            .vector("statistic", &r.root_statistics, Tol::stat(300))
            .vector("p", &r.root_p_values, tail(300))
            .vector("p_adjusted", &r.root_p_values_adjusted, tail(300))
            .scalar("split_break", split.threshold, Tol::exact())
            .finish();
        assert_eq!(
            split.feature, 1,
            "the more significant x2 must win; both p were 0 before"
        );
        // The chosen variable's statistic and adjusted p are element 1 (x2) of the 3-vectors.
        let (stat, padj) = (case.vector("statistic")[1], case.vector("p_adjusted")[1]);
        assert!(
            Tol::stat(300).allows(split.statistic, stat),
            "statistic {} vs R {stat}",
            split.statistic
        );
        assert!(
            tail(300).allows(split.p_value, padj),
            "adjusted p {} vs R {padj}",
            split.p_value
        );
    }

    #[test]
    fn test_validate_ctree_null_root_t4() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("ctree_null_root_t4");
        let (x, y) = case_xy(&ds, &["x1", "x2"], "y");
        let r = ctree(x.view(), y.view(), &CtreeConfig::default()).unwrap();
        assert!(r.root.split.is_none(), "partykit does not split this root");
        Check::new(&case)
            .scalar("n_nodes", r.n_nodes as f64, Tol::exact())
            .vector("root_statistic", &r.root_statistics, Tol::stat(150))
            .vector("root_p", &r.root_p_values, Tol::p_value(150))
            .vector(
                "root_p_adjusted",
                &r.root_p_values_adjusted,
                Tol::p_value(150),
            )
            .finish();
    }

    /// `n` noisy rows from a fixed LCG: one feature and a class code in 0..k that depends on it.
    fn noisy_classes(n: usize, k: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut x = Array2::zeros((n, 1));
        let mut y = Array1::zeros(n);
        for i in 0..n {
            let class = i % k;
            y[i] = class as f64;
            x[[i, 0]] = class as f64 + 1.5 * next();
        }
        (x, y)
    }

    #[test]
    fn test_ctree_classification_statistic_is_the_anova_quadratic_form() {
        // partykit's quadratic statistic for a numeric input and a K-class response is
        // (n - 1) SSB / Sxx, the between-class sum of squares of x over its total (computed here naively).
        let (x, y) = noisy_classes(40, 3, 7);
        let r = ctree(x.view(), y.view(), &classification_config()).unwrap();
        let split = r.root.split.as_ref().expect("the classes differ in x");
        let n = 40.0;
        let xs: Vec<f64> = x.column(0).to_vec();
        let mean = xs.iter().sum::<f64>() / n;
        let sxx: f64 = xs.iter().map(|v| (v - mean).powi(2)).sum();
        let mut ssb = 0.0;
        for k in 0..3 {
            let members: Vec<f64> = (0..40)
                .filter(|&i| y[i] == k as f64)
                .map(|i| xs[i])
                .collect();
            let mk = members.iter().sum::<f64>() / members.len() as f64;
            ssb += members.len() as f64 * (mk - mean).powi(2);
        }
        let expected = (n - 1.0) * ssb / sxx;
        assert!(
            (split.statistic - expected).abs() <= 1e-10 * expected,
            "statistic {} vs (n - 1) SSB / Sxx = {expected}",
            split.statistic
        );
    }

    #[test]
    fn test_ctree_null_root_split_rate() {
        // ML-26. A binary response independent of one input: at mincriterion 0.95 the root may split in about 5%
        // of the samples. The sum of the per-class rank statistics referred to chi2(K - 1) split 18% of them.
        let reps = 400;
        let mut splits = 0;
        for rep in 0..reps {
            let mut s = 0x9e37_79b9_7f4a_7c15u64 ^ rep as u64;
            let mut next = || {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64
            };
            let x = Array2::from_shape_fn((100, 1), |_| next());
            let y = Array1::from_shape_fn(100, |_| if next() < 0.5 { 0.0 } else { 1.0 });
            let cfg = CtreeConfig {
                testtype: "univariate".into(),
                ..classification_config()
            };
            if ctree(x.view(), y.view(), &cfg)
                .unwrap()
                .root
                .split
                .is_some()
            {
                splits += 1;
            }
        }
        let rate = splits as f64 / reps as f64;
        assert!(rate <= 0.08, "null root split rate {rate}");
    }

    #[test]
    fn test_ctree_constant_input_is_not_tested() {
        // partykit: an input whose Var(T) is below sqrt(eps) gets no test and does not count in the adjustment.
        let (x1, y) = noisy_regression(60, 1, 3);
        let mut x = Array2::from_elem((60, 2), 4.0);
        x.column_mut(1).assign(&x1.column(0));
        let r = ctree(x.view(), y.view(), &CtreeConfig::default()).unwrap();
        assert!(r.root_p_values[0].is_nan(), "{:?}", r.root_p_values);
        let split = r.root.split.as_ref().expect("x is strong");
        assert_eq!(split.feature, 1);
        // m = 1: the adjusted p is the raw p.
        assert_eq!(split.p_value.to_bits(), r.root_p_values[1].to_bits());
    }

    /// The nodes of `root` in preorder (partykit's node numbering).
    fn preorder(root: &CtreeNode) -> Vec<&CtreeNode> {
        let (mut out, mut stack) = (Vec::new(), vec![root]);
        while let Some(node) = stack.pop() {
            out.push(node);
            if let (Some(l), Some(r)) = (&node.left, &node.right) {
                stack.push(r);
                stack.push(l);
            }
        }
        out
    }

    /// The per-node quantities of the `ml_trees_t4` cases, in preorder.
    struct TreeValues {
        node_n: Vec<f64>,
        split_var: Vec<f64>,
        split_break: Vec<f64>,
        split_statistic: Vec<f64>,
        split_p_adjusted: Vec<f64>,
        terminal_prediction: Vec<f64>,
        terminal_class_prob: Vec<f64>,
    }

    fn tree_values(r: &CtreeResult) -> TreeValues {
        let mut v = TreeValues {
            node_n: Vec::new(),
            split_var: Vec::new(),
            split_break: Vec::new(),
            split_statistic: Vec::new(),
            split_p_adjusted: Vec::new(),
            terminal_prediction: Vec::new(),
            terminal_class_prob: Vec::new(),
        };
        for node in preorder(&r.root) {
            v.node_n.push(node.n as f64);
            match &node.split {
                Some(s) => {
                    v.split_var.push(s.feature as f64);
                    v.split_break.push(s.threshold);
                    v.split_statistic.push(s.statistic);
                    v.split_p_adjusted.push(s.p_value);
                }
                None => {
                    v.terminal_prediction.push(node.prediction);
                    if let Some(p) = &node.class_probs {
                        v.terminal_class_prob.extend(p);
                    }
                }
            }
        }
        v
    }

    #[test]
    fn test_validate_ctree_regression_tree_t4() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("ctree_reg_t4");
        let (x, y) = case_xy(&ds, &["x1", "x2", "x3"], "y");
        let r = ctree(x.view(), y.view(), &CtreeConfig::default()).unwrap();
        let v = tree_values(&r);
        Check::new(&case)
            .scalar("n_nodes", r.n_nodes as f64, Tol::exact())
            .scalar("n_terminal", r.n_terminal as f64, Tol::exact())
            .scalar("depth", r.depth as f64, Tol::exact())
            .vector("node_n", &v.node_n, Tol::exact())
            .vector("split_var", &v.split_var, Tol::exact())
            .vector("split_break", &v.split_break, Tol::exact())
            .vector("split_statistic", &v.split_statistic, Tol::stat(200))
            .vector("split_p_adjusted", &v.split_p_adjusted, tail(200))
            .vector(
                "terminal_prediction",
                &v.terminal_prediction,
                Tol::coef(200),
            )
            .vector("root_statistic", &r.root_statistics, Tol::stat(200))
            .vector("root_p", &r.root_p_values, tail(200))
            .vector("root_p_adjusted", &r.root_p_values_adjusted, tail(200))
            .vector("fitted", &r.predictions, Tol::coef(200))
            .finish();
    }

    #[test]
    fn test_validate_ctree_classification_tree_t4() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("ctree_cls_t4");
        let r = run_ctree(&ds, "cls", &["x1", "x2", "x3"], &CtreeConfig::default()).unwrap();
        assert!(r.is_classification);
        assert_eq!(
            r.class_names.as_deref(),
            Some(&["a".to_string(), "b".into(), "c".into()][..])
        );
        let v = tree_values(&r);
        Check::new(&case)
            .scalar("n_nodes", r.n_nodes as f64, Tol::exact())
            .scalar("n_terminal", r.n_terminal as f64, Tol::exact())
            .scalar("depth", r.depth as f64, Tol::exact())
            .vector("node_n", &v.node_n, Tol::exact())
            .vector("split_var", &v.split_var, Tol::exact())
            .vector("split_break", &v.split_break, Tol::exact())
            .vector("split_statistic", &v.split_statistic, Tol::stat(240))
            .vector("split_p_adjusted", &v.split_p_adjusted, tail(240))
            .vector("terminal_prediction", &v.terminal_prediction, Tol::exact())
            .vector(
                "terminal_class_prob",
                &v.terminal_class_prob,
                Tol::coef(240),
            )
            .vector("root_statistic", &r.root_statistics, Tol::stat(240))
            .vector("root_p", &r.root_p_values, tail(240))
            .vector("root_p_adjusted", &r.root_p_values_adjusted, tail(240))
            .vector("fitted_class", &r.predictions, Tol::exact())
            .finish();
    }

    #[test]
    fn test_ctree_splits_a_node_of_twice_minbucket() {
        // partykit splits a node of exactly 2 x minbucket rows (one admissible cut).
        let x = Array2::from_shape_fn((14, 1), |(i, _)| (i + 1) as f64);
        let y = Array1::from_shape_fn(14, |i| {
            (if i < 7 { 1.0 } else { 6.0 }) + 0.1 * ((i * 7 % 5) as f64 - 2.0)
        });
        let cfg = CtreeConfig {
            minsplit: 14,
            minbucket: 7,
            ..Default::default()
        };
        let r = ctree(x.view(), y.view(), &cfg).unwrap();
        let split = r
            .root
            .split
            .as_ref()
            .expect("a node of 2 x minbucket rows splits");
        assert_eq!(split.threshold, 7.0);
        assert_eq!(
            (
                r.root.left.as_ref().unwrap().n,
                r.root.right.as_ref().unwrap().n
            ),
            (7, 7)
        );
    }

    #[test]
    fn test_ctree_minprob_raises_the_node_minimum() {
        // partykit's minprob = 0.01: a child of a node of 1,000 rows holds at least 10 rows. An outlier
        // at the largest x makes the best cut of the root isolate it; with minbucket 1 alone that
        // child had 1 row.
        let (x, mut y) = noisy_regression(1000, 1, 11);
        let top = (0..1000)
            .max_by(|&a, &b| x[[a, 0]].total_cmp(&x[[b, 0]]))
            .unwrap();
        y[top] += 500.0;
        let cfg = CtreeConfig {
            minsplit: 2,
            minbucket: 1,
            ..Default::default()
        };
        let r = ctree(x.view(), y.view(), &cfg).unwrap();
        assert!(r.root.split.is_some());
        for node in preorder(&r.root) {
            if let (Some(a), Some(b)) = (&node.left, &node.right) {
                let mb = (node.n as f64 * 0.01).ceil() as usize;
                assert!(
                    a.n >= mb && b.n >= mb,
                    "node of {}: {} | {}",
                    node.n,
                    a.n,
                    b.n
                );
            }
        }
    }

    #[test]
    fn test_ctree_class_ties_take_the_first_class() {
        // which.max: a 3-3 leaf predicts the first class.
        let x = Array2::from_shape_fn((6, 1), |(i, _)| i as f64);
        let y = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let r = ctree(x.view(), y.view(), &classification_config()).unwrap();
        assert!(r.root.split.is_none());
        assert_eq!(r.root.prediction, 0.0);
    }

    #[test]
    fn test_ctree_numeric_target_is_regression() {
        // ML-87 (decision D5: the task comes from the target's type, as partykit). A numeric target
        // with ten integer values was a classification tree.
        let (x, y) = noisy_regression(200, 2, 4);
        let y = y.mapv(|v| (v.clamp(-4.0, 13.0) / 2.0).round() + 2.0); // 0..=9
        let r = ctree(x.view(), y.view(), &CtreeConfig::default()).unwrap();
        assert!(!r.is_classification);
        assert!(r.root.class_probs.is_none());
        let mean = y.sum() / 200.0;
        assert!((r.root.prediction - mean).abs() < 1e-12 * mean.abs());
    }

    #[test]
    fn test_ctree_large_values_are_regression() {
        // ML-29. `(v * 1e10) as i64` saturates above about 9.2e8, so every value of y in [1e9, 2e9]
        // fell into one "class": a single node predicting 922337203.685.
        let (x, y) = noisy_regression(200, 2, 9);
        let y = y.mapv(|v| 1.5e9 + 2.0e7 * v);
        let r = ctree(x.view(), y.view(), &CtreeConfig::default()).unwrap();
        assert!(!r.is_classification);
        assert!(r.n_nodes > 1, "a single node");
        let (lo, hi) = (
            y.iter().cloned().fold(f64::INFINITY, f64::min),
            y.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        assert!(r.predictions.iter().all(|&p| p >= lo && p <= hi));
    }

    #[test]
    fn test_ctree_classification_on_request() {
        // Numeric codes with classification = true fit the tree of the same classes as text.
        let ds = generated_dataset("ctree_cls_t4");
        let text = run_ctree(&ds, "cls", &["x1", "x2", "x3"], &CtreeConfig::default()).unwrap();
        let (x, code) = case_xy(&ds, &["x1", "x2", "x3"], "code");
        let codes = ctree(x.view(), code.view(), &classification_config()).unwrap();
        assert!(codes.is_classification);
        let sig = |r: &CtreeResult| {
            preorder(&r.root)
                .iter()
                .map(|n| {
                    (
                        n.n,
                        n.split.as_ref().map(|s| (s.feature, s.threshold.to_bits())),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(sig(&codes), sig(&text));
        assert_eq!(codes.predictions, text.predictions);
        // A text target cannot be a regression.
        let cfg = CtreeConfig {
            classification: Some(false),
            ..Default::default()
        };
        let e = run_ctree(&ds, "cls", &["x1", "x2", "x3"], &cfg)
            .unwrap_err()
            .to_string();
        assert!(e.contains("text, categorical or boolean"), "{e}");
    }

    #[test]
    fn test_ctree_class_cap() {
        // The C5.0 and CART cap max(2, min(1000, n / 2)): 15 classes in 30 rows fit, 16 do not.
        let x = Array2::from_shape_fn((30, 1), |(i, _)| i as f64);
        let fit = |k: usize| {
            let y = Array1::from_shape_fn(30, |i| (i % k) as f64);
            ctree(x.view(), y.view(), &classification_config())
        };
        assert!(fit(15).is_ok());
        let e = fit(16).unwrap_err().to_string();
        assert!(e.contains("at most 15 classes"), "{e}");
    }

    /// The Monte Carlo configuration of the `ctree_montecarlo_t4` case (`stump = TRUE`: the root only).
    fn monte_carlo_config(seed: u64) -> CtreeConfig {
        CtreeConfig {
            testtype: "MonteCarlo".into(),
            maxdepth: 1,
            seed: Some(seed),
            ..Default::default()
        }
    }

    #[test]
    fn test_validate_ctree_montecarlo_t4() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("ctree_montecarlo_t4");
        let (x, y) = case_xy(&ds, &["x1", "x2"], "y");
        let r = ctree(x.view(), y.view(), &monte_carlo_config(1)).unwrap();
        Check::new(&case)
            .vector("root_statistic", &r.root_statistics, Tol::stat(120))
            .finish();
        // p2a's and R's Monte Carlo p-values are two independent permutation estimates: each lies
        // within 4 standard errors of their difference (plus 2 / B) of R's value.
        let b = case.scalar("nresample");
        for (j, &pr) in case.vector("root_p_mc").iter().enumerate() {
            let allowed = 4.0 * (2.0 * pr * (1.0 - pr) / b).sqrt() + 2.0 / b;
            let p = r.root_p_values[j];
            assert!(
                (p - pr).abs() <= allowed,
                "x{}: p {p} vs R {pr} (allowed {allowed})",
                j + 1
            );
            // MonteCarlo p-values are not adjusted (partykit).
            assert_eq!(r.root_p_values_adjusted[j].to_bits(), p.to_bits());
            // A count of permutations out of B.
            assert_eq!((p * b).round() / b, p);
        }
    }

    #[test]
    fn test_ctree_montecarlo_is_reproducible_and_seeded() {
        let ds = generated_dataset("ctree_montecarlo_t4");
        let (x, y) = case_xy(&ds, &["x1", "x2"], "y");
        let p = |seed: u64| {
            ctree(x.view(), y.view(), &monte_carlo_config(seed))
                .unwrap()
                .root_p_values
        };
        assert_eq!(p(7), p(7));
        assert_ne!(p(7), p(8));
    }

    #[test]
    fn test_ctree_montecarlo_honours_cancellation() {
        let (x, y, _) = rpart_reference_data();
        let err = with_budget(cancelled_budget(), || {
            ctree(x.view(), y.view(), &monte_carlo_config(1))
        })
        .unwrap_err();
        assert!(matches!(err, EconError::Cancelled(_)), "{err}");
    }

    /// The default configuration for a classification of numeric class codes.
    fn classification_config() -> CtreeConfig {
        CtreeConfig {
            classification: Some(true),
            ..Default::default()
        }
    }

    fn cancelled_budget() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    /// Noisy regression data from a fixed LCG (no external RNG).
    fn noisy_regression(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut x = Array2::zeros((n, p));
        let mut y = Array1::zeros(n);
        for i in 0..n {
            for j in 0..p {
                x[[i, j]] = 10.0 * next();
            }
            y[i] = 2.0 * x[[i, 0]] - x[[i, p - 1]] + 3.0 * (next() - 0.5);
        }
        (x, y)
    }

    // R reference data (rpart 4.1.x; run with `Rscript`):
    //
    // library(rpart)
    // i <- 0:39
    // x1 <- ((i * 17) %% 41) / 4; x2 <- ((i * 29) %% 43) / 5; x3 <- (i %% 7) / 2
    // noise <- ((i * 7919) %% 101) / 100 - 0.5
    // y_reg <- 2 * (x1 > 5) + 0.3 * x2 + noise
    // y_cls <- factor(ifelse(x2 > 6, 2, ifelse(x1 + 0.5 * noise > 5, 1, 0)))
    // ctl <- rpart.control(minsplit = 10, minbucket = 5, cp = 0, maxdepth = 1, xval = 0,
    //                      maxcompete = 0, maxsurrogate = 0, usesurrogate = 0)
    // rpart(y_reg ~ x1 + x2 + x3, method = "anova", control = ctl)$splits    # x1 at 5.125 (21 | 19)
    // # one variable at a time:
    // # anova:                          x1 5.125 (21|19), x2 3.5 (17|23), x3 2.75 (35|5)
    // # class, parms=list(split="information"): x1 5.125 (21|19), x2 6.1 (29|11), x3 2.75 (35|5)
    //
    // rpart sends `x3 >= 2.75` left; the code here sends `x <= threshold` left, so the
    // sizes above are written as (<= | >).

    /// The 40 rows of the rpart reference script (see comment above).
    fn rpart_reference_data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let n = 40;
        let mut x = Array2::zeros((n, 3));
        let (mut y_reg, mut y_cls) = (Array1::zeros(n), Array1::zeros(n));
        for i in 0..n {
            let x1 = ((i * 17) % 41) as f64 / 4.0;
            let x2 = ((i * 29) % 43) as f64 / 5.0;
            let x3 = (i % 7) as f64 / 2.0;
            let noise = ((i * 7919) % 101) as f64 / 100.0 - 0.5;
            x[[i, 0]] = x1;
            x[[i, 1]] = x2;
            x[[i, 2]] = x3;
            y_reg[i] = 2.0 * (if x1 > 5.0 { 1.0 } else { 0.0 }) + 0.3 * x2 + noise;
            y_cls[i] = if x2 > 6.0 {
                2.0
            } else if x1 + 0.5 * noise > 5.0 {
                1.0
            } else {
                0.0
            };
        }
        (x, y_reg, y_cls)
    }

    #[test]
    fn test_ctree_rejects_non_finite_inputs() {
        let (x, y, _) = rpart_reference_data();
        let cfg = CtreeConfig {
            minsplit: 10,
            minbucket: 5,
            ..Default::default()
        };
        assert!(ctree(x.view(), y.view(), &cfg).is_ok());
        for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let mut xb = x.clone();
            xb[[3, 1]] = bad;
            let err = ctree(xb.view(), y.view(), &cfg).unwrap_err().to_string();
            assert!(err.contains("non-finite"), "x {bad}: {err}");
            let mut yb = y.clone();
            yb[5] = bad;
            let err = ctree(x.view(), yb.view(), &cfg).unwrap_err().to_string();
            assert!(err.contains("non-finite"), "y {bad}: {err}");
        }
    }

    #[test]
    fn test_ctree_rejects_mincriterion_outside_the_unit_interval() {
        // NaN used to pass the range check and silently grow a stump, since
        // no p-value is below 1 - NaN.
        let (x, y, _) = rpart_reference_data();
        for bad in [f64::NAN, 0.0, 1.0, f64::INFINITY] {
            let cfg = CtreeConfig {
                mincriterion: bad,
                ..Default::default()
            };
            let err = ctree(x.view(), y.view(), &cfg).unwrap_err().to_string();
            assert!(
                err.contains("mincriterion must be between 0 and 1"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn test_ctree_maxdepth_is_bounded_by_node_numbering() {
        // Assumes a 64-bit host (usize::BITS - 2 == 62), as CI does.
        let (x, y, _) = rpart_reference_data();
        let deep = CtreeConfig {
            maxdepth: usize::BITS as usize - 1,
            ..Default::default()
        };
        let err = ctree(x.view(), y.view(), &deep).unwrap_err().to_string();
        assert!(err.contains("maxdepth must be at most 62"), "{err}");
        let ok = CtreeConfig {
            maxdepth: 62,
            ..Default::default()
        };
        assert!(ctree(x.view(), y.view(), &ok).is_ok());
    }

    #[test]
    fn test_ctree_honours_cancellation() {
        let (x, y, _) = rpart_reference_data();
        let err = with_budget(cancelled_budget(), || {
            ctree(x.view(), y.view(), &CtreeConfig::default())
        })
        .unwrap_err();
        assert!(matches!(err, EconError::Cancelled(_)), "{err}");
    }

    #[test]
    fn test_ctree_regression_basic() {
        // Clear pattern with two groups
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0],
            [11.0],
            [12.0],
            [13.0],
            [14.0],
            [15.0],
            [16.0],
            [17.0],
            [18.0],
            [19.0],
            [20.0]
        ];
        // Two clear groups: low values for x <= 10, high values for x > 10
        let y = array![
            1.0, 1.1, 0.9, 1.2, 0.8, 1.1, 0.9, 1.0, 1.1, 0.9, 10.0, 10.1, 9.9, 10.2, 9.8, 10.1,
            9.9, 10.0, 10.1, 9.9
        ];

        let config = CtreeConfig {
            mincriterion: 0.95,
            minsplit: 4,
            minbucket: 2,
            maxdepth: 0,
            ..Default::default()
        };

        let result = ctree(x.view(), y.view(), &config).unwrap();

        // Should find a split
        assert!(result.n_nodes >= 1);
        assert_eq!(result.predictions.len(), 20);

        // Predictions should approximate the true pattern
        for (i, &pred) in result.predictions.iter().enumerate() {
            if i < 10 {
                assert!(pred < 5.0, "Low group prediction {} should be < 5", pred);
            } else {
                assert!(pred > 5.0, "High group prediction {} should be > 5", pred);
            }
        }
    }

    #[test]
    fn test_ctree_classification() {
        // Binary classification
        let x = array![
            [1.0, 0.0],
            [1.5, 0.5],
            [2.0, 0.0],
            [2.5, 0.5],
            [3.0, 0.0],
            [3.5, 0.5],
            [7.0, 1.0],
            [7.5, 0.5],
            [8.0, 1.0],
            [8.5, 0.5],
            [9.0, 1.0],
            [9.5, 0.5],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let config = CtreeConfig {
            mincriterion: 0.95,
            minsplit: 4,
            minbucket: 2,
            maxdepth: 5,
            classification: Some(true),
            ..Default::default()
        };

        let result = ctree(x.view(), y.view(), &config).unwrap();

        assert!(result.is_classification);

        // Check predictions match classes
        for (i, &pred) in result.predictions.iter().enumerate() {
            if i < 6 {
                assert!(
                    pred < 0.5,
                    "Sample {} should predict class 0, got {}",
                    i,
                    pred
                );
            } else {
                assert!(
                    pred > 0.5,
                    "Sample {} should predict class 1, got {}",
                    i,
                    pred
                );
            }
        }
    }

    #[test]
    fn test_ctree_predict() {
        let x_train = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [15.0],
            [16.0],
            [17.0],
            [18.0],
            [19.0]
        ];
        let y_train = array![1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0, 9.0];

        let config = CtreeConfig {
            mincriterion: 0.90,
            minsplit: 4,
            minbucket: 2,
            ..Default::default()
        };

        let result = ctree(x_train.view(), y_train.view(), &config).unwrap();

        let x_test = array![[2.5], [17.5]];
        let predictions = ctree_predict(&result, x_test.view()).unwrap();

        assert_eq!(predictions.len(), 2);
        assert!(predictions[0] < 5.0, "First prediction should be low");
        assert!(predictions[1] > 5.0, "Second prediction should be high");
    }

    #[test]
    fn test_ctree_no_split_when_no_pattern() {
        // Random noise - no real pattern
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        // Alternating values - no clear split pattern
        let y = array![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];

        let config = CtreeConfig {
            mincriterion: 0.99, // Very strict criterion
            minsplit: 4,
            minbucket: 2,
            ..Default::default()
        };

        let result = ctree(x.view(), y.view(), &config).unwrap();

        // With strict criterion and no pattern, should have few or no splits
        assert!(result.n_terminal >= 1);
    }

    #[test]
    fn test_ctree_variable_importance() {
        // First variable is predictive, second is noise
        let x = array![
            [1.0, 5.0],
            [2.0, 3.0],
            [3.0, 7.0],
            [4.0, 2.0],
            [5.0, 8.0],
            [15.0, 4.0],
            [16.0, 9.0],
            [17.0, 1.0],
            [18.0, 6.0],
            [19.0, 5.0],
        ];
        let y = array![1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0, 9.0];

        let config = CtreeConfig {
            mincriterion: 0.90,
            minsplit: 4,
            minbucket: 2,
            ..Default::default()
        };

        let result = ctree(x.view(), y.view(), &config).unwrap();

        // First variable should have higher importance
        assert!(
            result.variable_importance[0] >= result.variable_importance[1],
            "Feature 0 ({:.4}) should have >= importance than Feature 1 ({:.4})",
            result.variable_importance[0],
            result.variable_importance[1]
        );
    }

    #[test]
    fn test_ctree_depth_limit() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0],
            [11.0],
            [12.0],
            [13.0],
            [14.0],
            [15.0],
            [16.0],
            [17.0],
            [18.0],
            [19.0],
            [20.0]
        ];
        let y = array![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0
        ];

        let config = CtreeConfig {
            mincriterion: 0.50, // Low criterion to allow splits
            minsplit: 4,
            minbucket: 2,
            maxdepth: 2, // Limit depth
            ..Default::default()
        };

        let result = ctree(x.view(), y.view(), &config).unwrap();

        assert!(result.depth <= 2, "Depth {} exceeds max 2", result.depth);
    }

    #[test]
    fn test_ctree_config_defaults() {
        let config = CtreeConfig::default();
        assert_eq!(config.mincriterion, 0.95);
        assert_eq!(config.minsplit, 20);
        assert_eq!(config.minbucket, 7);
        assert_eq!(config.maxdepth, 0);
        assert_eq!(config.teststat, "quadratic");
        assert_eq!(config.testtype, "bonferroni");
        assert_eq!(config.classification, None);
        assert_eq!(config.nresample, 9999);
    }

    /// partykit's two-sample statistic of every admissible cut computed from scratch, the
    /// reference the running-sum scan of `find_best_split` must reproduce.
    fn find_best_split_reference(
        x: &ArrayView2<f64>,
        response: &Response<'_>,
        indices: &[usize],
        feature: usize,
        mb: usize,
    ) -> Option<(f64, Vec<usize>, Vec<usize>)> {
        let n = indices.len();
        let mb = mb.max(1);
        let mut sorted: Vec<(f64, usize)> = indices.iter().map(|&i| (x[[i, feature]], i)).collect();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut best: Option<(f64, usize)> = None;
        for pos in mb..(n + 1).saturating_sub(mb) {
            if sorted[pos - 1].0 == sorted[pos].0 {
                continue;
            }
            let left = &sorted[..pos];
            let stat = match response {
                Response::Regression(y) => {
                    let mean = indices.iter().map(|&i| y[i]).sum::<f64>() / n as f64;
                    let mut s = 0.0;
                    for &(_, i) in left {
                        s += y[i] - mean;
                    }
                    s * s / (pos as f64 * (n - pos) as f64)
                }
                Response::Classes { class, k } => {
                    let mut lc = vec![0usize; *k];
                    let mut tc = vec![0usize; *k];
                    for &(_, i) in left {
                        lc[class[i]] += 1;
                    }
                    for &(_, i) in &sorted {
                        tc[class[i]] += 1;
                    }
                    pearson_split_statistic(&lc, &tc, pos, n as f64)
                }
            };
            if best.is_none_or(|(b, _)| stat > b) {
                best = Some((stat, pos));
            }
        }
        best.map(|(_, pos)| {
            (
                sorted[pos - 1].0,
                sorted[..pos].iter().map(|(_, i)| *i).collect(),
                sorted[pos..].iter().map(|(_, i)| *i).collect(),
            )
        })
    }

    #[test]
    fn test_ctree_split_matches_naive_reference() {
        for seed in 0..20u64 {
            let n = 20 + 7 * seed as usize;
            let (mut x, y) = noisy_regression(n, 2, seed);
            for i in 0..n {
                x[[i, 1]] = x[[i, 1]].round() / 2.0; // tie-heavy feature
            }
            let class: Vec<usize> = y
                .iter()
                .map(|&v| {
                    if v < 3.0 {
                        0
                    } else if v < 9.0 {
                        1
                    } else {
                        2
                    }
                })
                .collect();
            let indices: Vec<usize> = (0..n).filter(|i| i % 5 != 2).collect();
            let responses = [
                Response::Regression(y.view()),
                Response::Classes {
                    class: &class,
                    k: 3,
                },
            ];
            for mb in [1usize, 3, 7] {
                for f in 0..2 {
                    for (kind, response) in responses.iter().enumerate() {
                        let key = |o: Option<(f64, Vec<usize>, Vec<usize>)>| {
                            o.map(|(t, l, r)| (t.to_bits(), l, r))
                        };
                        assert_eq!(
                            key(find_best_split(&x.view(), response, &indices, f, mb)),
                            key(find_best_split_reference(
                                &x.view(),
                                response,
                                &indices,
                                f,
                                mb
                            )),
                            "response {kind} seed {seed} mb {mb} feature {f}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_ctree_minbucket_zero_acts_as_one() {
        // ML-28. A child needs at least one row, so minbucket = 0 gives the node minimum of
        // minbucket = 1 and the same tree.
        assert_eq!(node_minimum(40, 0), node_minimum(40, 1));
        let (x, y_reg, _) = rpart_reference_data();
        let fit = |mb: usize| {
            let cfg = CtreeConfig {
                minsplit: 2,
                minbucket: mb,
                ..Default::default()
            };
            ctree(x.view(), y_reg.view(), &cfg).unwrap().predictions
        };
        assert_eq!(fit(0), fit(1));
    }

    #[test]
    fn test_ctree_rejects_minsplit_below_twice_minbucket() {
        // ML-28 (design 0.1). A node needs 2 * minbucket rows to split into two children of minbucket, so a smaller
        // minsplit has no effect (partykit accepts and ignores it); refused (rule R2).
        let (x, y, _) = rpart_reference_data();
        let cfg = |ms: usize, mb: usize| CtreeConfig {
            minsplit: ms,
            minbucket: mb,
            ..Default::default()
        };
        let e = ctree(x.view(), y.view(), &cfg(10, 6))
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("minsplit (10) must be at least 2 * minbucket (12)"),
            "{e}"
        );
        assert!(ctree(x.view(), y.view(), &cfg(12, 6)).is_ok());
        assert!(ctree(x.view(), y.view(), &cfg(2, 0)).is_ok()); // minbucket 0 acts as 1 (T0-WP15)
    }

    #[test]
    fn test_ctree_options_parse_aliases_and_reject_unknown() {
        CtreeTestStat::choices().validate().unwrap();
        CtreeTestType::choices().validate().unwrap();
        assert_eq!(
            "Maximum".parse::<CtreeTestStat>().unwrap(),
            CtreeTestStat::Maximum
        );
        assert_eq!(
            "max".parse::<CtreeTestStat>().unwrap(),
            CtreeTestStat::Maximum
        );
        assert_eq!(
            "quadratic".parse::<CtreeTestStat>().unwrap(),
            CtreeTestStat::Quadratic
        );
        assert_eq!(
            "none".parse::<CtreeTestType>().unwrap(),
            CtreeTestType::Univariate
        );
        assert_eq!(
            "Bonferroni".parse::<CtreeTestType>().unwrap(),
            CtreeTestType::Bonferroni
        );
        let (x, y, y_cls) = rpart_reference_data();
        let run = |ts: &str, tt: &str, y: &Array1<f64>, classification: bool| {
            ctree(
                x.view(),
                y.view(),
                &CtreeConfig {
                    minsplit: 10,
                    minbucket: 5,
                    teststat: ts.into(),
                    testtype: tt.into(),
                    classification: Some(classification),
                    ..Default::default()
                },
            )
        };
        assert!(
            run("qs2", "bonferroni", &y, false)
                .unwrap_err()
                .to_string()
                .contains("quadratic")
        );
        assert!(
            run("quadratic", "Teststatistic", &y, false)
                .unwrap_err()
                .to_string()
                .contains("not implemented")
        );
        assert!(run("quadratic", "MonteCarlo", &y, false).is_ok());
        assert_eq!(
            "MonteCarlo".parse::<CtreeTestType>().unwrap(),
            CtreeTestType::MonteCarlo
        );
        let none = CtreeConfig {
            testtype: "montecarlo".into(),
            nresample: 0,
            ..Default::default()
        };
        let e = ctree(x.view(), y.view(), &none).unwrap_err();
        assert!(
            matches!(e, EconError::InvalidSpecification { .. })
                && e.to_string().contains("nresample"),
            "{e}"
        );
        assert!(
            run("maximum", "bonferroni", &y_cls, true)
                .unwrap_err()
                .to_string()
                .contains("regression")
        );
    }

    #[test]
    fn test_ctree_maximum_teststat_is_not_the_quadratic_one() {
        // Before: any string other than "max" (partykit's "maximum" included) silently gave the quadratic statistic.
        let (x, y, _) = rpart_reference_data();
        let stat = |ts: &str| {
            let r = ctree(
                x.view(),
                y.view(),
                &CtreeConfig {
                    minsplit: 10,
                    minbucket: 5,
                    teststat: ts.into(),
                    ..Default::default()
                },
            )
            .unwrap();
            r.root.split.expect("the x1 step is significant").statistic
        };
        let (quad, max, maximum) = (stat("quadratic"), stat("max"), stat("maximum"));
        assert_eq!(maximum.to_bits(), max.to_bits());
        assert!(
            (max * max - quad).abs() <= 1e-12 * quad,
            "{max}^2 vs {quad}"
        );
    }

    #[test]
    fn test_ctree_class_rule_message_states_the_rounded_bound() {
        // T2 leftover minor (T2-WP17 #0). The class cap is max(2, min(1000, n / 2)) with integer division:
        // 3 classes in 5 rows are refused, and the refusal states the rounded bound. Property test of p2a's
        // own rule; no number is compared with partykit (test_na_ctree_class_labels asserts the 15-class
        // message of a 30-row fixture).
        let df = polars::prelude::df! {
            "lab" => ["a", "b", "c", "b", "a"],
            "x" => [0.1, 0.7, 0.3, 0.9, 0.2],
        }
        .unwrap();
        let e = run_ctree(
            &crate::Dataset::new(df),
            "lab",
            &["x"],
            &CtreeConfig::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("at most 2 classes"), "{e}");
        assert!(e.contains("n / 2 rounded down (2 for 5 rows)"), "{e}");
        assert!(!e.contains("half the rows"), "{e}");
    }
}
