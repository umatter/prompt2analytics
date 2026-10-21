//! The rule tables of the request-limits catalogue: [`PARAM_RULES`] classifies every cost-bearing
//! request field of the MCP tools, [`ROW_RULES`] names the tools whose cost grows faster than
//! linearly in the dataset's row count.

use super::{Family, ParamRule, RowClass, RowRule, Rule};

const fn p(tool: &'static str, path: &'static str, rule: Rule) -> ParamRule {
    ParamRule { tool, path, rule }
}

const fn row(tool: &'static str, class: RowClass) -> RowRule {
    RowRule {
        tool,
        dataset_field: "dataset",
        class,
    }
}

// Classification rules for PARAM_RULES:
//
// 1. A generic ("*") rule is used only when the field name means the same thing in every tool
//    that has it (`n_bootstrap` is always a replication count, `max_iter` always an iteration
//    limit). A tool-specific rule for the same path replaces the generic one for that tool.
// 2. Ambiguous names are always tool-specific: `p`, `q`, `d`, `n`, `k`, `order`, `lag`, `period`,
//    `start` and `values` mean a model order in one tool, a row count, a cluster count or an
//    index in another.
// 3. Every `Exempt` states why the field costs nothing (it selects, indexes or shrinks work, or a
//    closed-form formula uses it). The schema test in this file requires every integer field,
//    every array field and every `tol`/`eps` float field of every tool to be matched here.
//
// The list covers every integer property, every array property and every `tol`/`eps` float of
// the tool schemas at d6a3a3b.

/// Every classified request field (see the classification rules above).
pub static PARAM_RULES: &[ParamRule] = &[];

/// Tools whose dataset height is limited, by [`RowClass`]. All take the dataset in `dataset`.
///
/// Deliberately not row-limited, because their wave-2 packages remove the superlinear cost:
/// `hypothesis_cor_test` (Kendall), `stats_ecdf`, `timeseries_cpgram`, `ml_ctree`,
/// `treatment_twang`, `treatment_sbw`, `bacon_decomp`, `regression_gls` (AR(1) and CS closed
/// forms; a user-supplied correlation relies on the per-allocation ceiling), `panel_gls`,
/// `moran_test`, `spatial_lm_tests_tool` and `spatial_neighbors`.
pub static ROW_RULES: &[RowRule] = &[
    // Dense: clustering.rs:829-830 keeps the condensed n(n-1)/2 f64 distance matrix.
    row("ml_hierarchical", RowClass::Dense),
    row("ml_cutree", RowClass::Dense),
    // Dense: neighbour lists of up to n^2 indices (ML-23).
    row("ml_dbscan", RowClass::Dense),
    // Dense: smooth_spline.rs:377 allocates the n x n penalty `vec![vec![0.0; n]; n]`.
    row("regression_smooth_spline", RowClass::Dense),
    // Cubic: reduction.rs:474/582/641/690 builds several n x n matrices per iteration.
    row("ml_tsne", RowClass::Cubic),
    // Cubic: reduction.rs:912/934/1004 builds three n x n matrices plus an O(n^3) eigensolve.
    row("ml_cmdscale", RowClass::Cubic),
    // Cubic: weights.rs:320 computes dense eigenvalues of W (R parity, `method = "eigen"`).
    row("sar_model", RowClass::Cubic),
    row("sem_model", RowClass::Cubic),
    // Cubic: spatialprobit.rs:401 uses a dense W; the impacts do dense n^3 products per draw.
    row("sar_probit_model", RowClass::Cubic),
    row("sem_probit_model", RowClass::Cubic),
    // Cubic: splm.rs:1793 builds the dense NT x NT projection `P_Z`.
    row("spatial_panel_ml", RowClass::Cubic),
    row("spatial_panel_gmm", RowClass::Cubic),
    // Quadratic: nearest-neighbour search over all controls.
    row("propensity_matching", RowClass::Quadratic),
    // Quadratic: O(B n^2) matching permutations (ECON-10).
    row("heterogeneity_test", RowClass::Quadratic),
    // Quadratic: SMO over all pairs (ML-39; still O(n^2) after T0-WP15's error cache).
    row("ml_svm", RowClass::Quadratic),
    // Quadratic: loess.rs:240-290 fits each point on its span * n nearest neighbours.
    row("regression_loess", RowClass::Quadratic),
];
