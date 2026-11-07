# Validation: Generalized Synthetic Control (gsynth)

## What p2a computes

`run_gsynth` (`crates/p2a-core/src/econometrics/synth.rs`) reproduces gsynth 1.4.0 (`gsynth::gsynth`,
which calls fect 2.4.5) for its two estimators:

- **`ife`** (gsynth `estimator = "gsynth"`, fect `method = "gsynth"`; not gsynth 1.4's EM `"ife"`).
  Bai's (2009) interactive fixed effects are fitted on the never-treated units over all periods:
  `Y = mu + alpha_i + xi_t + X beta + F Lambda' + e`, with unit, time, two-way or no fixed effects
  (`force`). Covariates use gsynth's start (the within OLS) and iteration (factors given beta, beta
  given factors); the stopping rule is scale-free, `||X~ dbeta|| / ||Y~|| < tolerance` (gsynth stops
  on the absolute `||dbeta||`). Each treated unit's loadings (and its alpha under unit effects) are
  the OLS fit of `Y - X beta - mu - xi` on `[F, 1]` over its own pre-treatment periods; the
  counterfactual is `mu + alpha_i + xi_t + X beta + lambda_i' f_t`.
  `sigma2 = SSE / (T N_co - n_fe - p - r (N_co + T - n_dim) + r^2)` with `n_fe` = 1, N_co, T,
  N_co + T - 1 and `n_dim` = 0, 1, 1, 2 for none, unit, time and two-way effects (gsynth's); with no
  residual degrees of freedom it is NaN with a warning.
- **`mc`** (gsynth `estimator = "mc"`): `min 1/2 sum_untreated (Y - mu - alpha - xi - X beta - L)^2 +
  lambda T N ||L||_*` by soft-impute over all units' untreated cells, from fect's lambda = +inf start.
  lambda is fixed (one value), or chosen by k-fold cross-validation from a user grid or fect's
  default grid (`n_lambda` values from the largest singular value of the start residual / (T N) down
  three decades, then 0).
- **Number of factors** (`cross_validate`): leave-one-out over the treated units' pre-treatment
  periods (Xu 2017; fect `cv.method = "loo"`), r from `n_factors` to `min(max_factors, r_id)` with
  `r_id = min(N_co - v, T - u, T0_min - u - 1)` (u, v = unit and time effects); the smallest r with
  minimum MSPE. The errors are PRESS errors `e / (1 - h)` of one fit per unit.
- **Bootstrap** (`bootstrap_se`): `ife` uses gsynth's parametric bootstrap (empirical errors from
  each drawn control posing as the treated units, then resampled controls and residuals around the
  fitted counterfactual); `mc` the unit bootstrap. SE = sd of the replicates, normal CI and p-value,
  per-event-time SEs; seeded (`seed`, default 42) ChaCha8 streams.
- `dynamic_effects` is gsynth's event-time ATT (`e = t - T0_j + 1`: pre-treatment periods <= 0,
  first treated period 1) with `dynamic_counts`; `pre_mspe` is gsynth's `rmse^2` over the treated
  units' pre-treatment cells.
- Refused: non-0/1 or reversing treatments, collinear or fixed-effect-absorbed covariates, r above
  `r_id`, treated loadings that the factors do not identify.

## Reference and pins

Case file `validation/r/cases/causal_gsynth.R` writes `gsynth_simdata` (gsynth's `simdata`: 50 units,
30 periods, 5 treated from period 21; `validation/datasets/generated/gsynth_simdata.csv`) and
`gsynth_staggered` (36 units, 20 periods, 6 treated from periods 11, 13, 15, one covariate). It
writes the estimators out (`.gs_ife`, `.gs_estimate`, `.gs_loo`, `.gsmc_fit`, `.gsmc_cv`) and checks
them against gsynth and fect before the values are stored.

| Test | Pins | Tolerance |
|---|---|---|
| `test_validate_gsynth_ife_r2` | `ife_r2_nocov_*`: ATT, event-time ATT and counts, unit ATTs, mu, sigma2, rmse (two-way, r = 2) | coefficients 1e-10, statistics 1e-8 |
| `test_validate_gsynth_ife_cov` | `ife_r2_cov_*` (X1, X2) at tolerance 1e-12; gsynth's default-tolerance ATT | `Tol::iterative(1e-8)`; 5e-4 |
| `test_validate_gsynth_ife_unit` | `ife_r2_unit_*` (unit effects) | as the first row |
| `test_validate_gsynth_staggered` | `tw_r2_cov_*`, `tw_r2_nocov_*`, `unit_r2_cov_*` | iterative 1e-8 / README |
| `test_validate_gsynth_ife_cv` | LOO MSPE by r = 0..5 with and without covariates, selected r, ATT; fect's MSPEs (fitted at 1e-3) | iterative 1e-8; 1e-3 |
| `test_validate_gsynth_staggered_cv` | `loo_cov_*` | iterative 1e-8; 1e-3 |
| `test_validate_gsynth_mc` | `mc_l001_cov_*` (rank 2) and `mc_l03_cov_*` (rank 0) | iterative 1e-7 / 1e-8 |
| `test_validate_gsynth_mc_lambda_grid` | `mc_grid`, `mc_eigen_max` | coefficients 1e-10 |
| `test_validate_gsynth_mc_cv` | MC CV MSPE over a 3-value grid, lambda, ATT (fits at 1e-8 against the solution) | iterative max(1e-6, 10 x gap); exact lambda |
| `test_validate_gsynth_bootstrap_parametric_replay` | 50 replicate ATTs, SE 0.858036, CI, p, event-time SEs, gsynth's draws replayed | README |
| `test_validate_gsynth_bootstrap_unit_replay` | 20 MC replicate ATTs, SE 0.385157, CI, p | iterative 1e-7 / 1e-6 |

Pinned values (simdata): ATT 4.639593 (two-way, r = 2), 5.543287 (with X1, X2; beta 1.0213, 3.0524),
4.872566 (unit effects); LOO selects r = 2; MC ATT 5.117591 (lambda 0.01) and 5.085248 (lambda 0.3).
Before this package p2a reported 4.517288 for both two-way fits (covariates ignored), 4.618396 with
unit effects and r = 5 from its cross-validation.

Property tests (not registered): `test_gsynth_options_change_the_estimate`,
`test_gsynth_ife_convergence_is_scale_free`, `test_gsynth_mc_convergence_is_scale_free`,
`test_gsynth_reports_non_convergence`, `test_gsynth_rejects_invalid_inputs`,
`test_gsynth_loo_press_equals_refits`, `test_gsynth_cv_caps_r_at_the_identified_bound`,
`test_gsynth_mc_requires_lambda_without_cv`, `test_gsynth_bootstrap_is_seeded`,
`test_gsynth_bootstrap_rejects_invalid_parameters`, `test_gsynth_display_reports_what_was_estimated`.

## Deviations from gsynth / fect

1. Cross-validated r: LOO over treated pre-periods (fect `cv.method = "loo"`); gsynth 1.4's default
   random folds are recorded in the case (`ife_cv_cov_*`, r = 1 with `set.seed(7)`), not pinned.
2. MC cross-validation: deterministic folds over treated pre-period positions (`p mod k`), warm
   starts along the grid, minimum MSPE; fect uses random folds and its `1se` rule with time-weighted
   scores. The full default grid is not run in the tests.
3. IFE stopping rule: scale-free (above); the pins use tolerance 1e-12, where both rules agree.
4. Inputs that fect handles with a warning (beta = 0 for collinear covariates, r reset, NA loadings)
   are errors here.
5. Bootstrap draws: ChaCha8 streams per (seed, phase, replication) instead of R's RNG; the pins
   replay gsynth's own draws.
6. sigma2 is NaN (with a warning) when the control fit has no residual degrees of freedom.

## Missing values and column types

Data requirements: one row per (unit, time) of a balanced panel, with no missing value (null
or NaN) in the outcome, treatment, unit, time or covariate columns. Times are whole numbers
or dates; units may be text, integers, floats, booleans or dates; integer and boolean
treatments are read as numbers. The panel is checked before any units x periods matrix is
built:

- a missing value: `NullValues` naming the column, e.g. "Column 'outcome' contains 1 missing
  value(s) (null or NaN). Drop or impute them first.";
- a duplicate (unit, time) row: "gsynth needs one row per (unit, time): {k} duplicate row(s)
  (first: unit '{u}', time {t}). Remove the duplicates first.";
- a missing (unit, time) cell: "gsynth needs a balanced panel: {k} (unit, time) cell(s) are
  missing (first: unit '{u}', time {t}). p2a's gsynth has no estimator for unbalanced panels
  (R's gsynth imputes missing cells); fill or drop them first.";
- a fractional time: "Time column 'time' must hold whole numbers; row {i} (0-based) is {v}.
  Round or recode it first."

Before T2-WP13 a missing outcome was read as 0, a missing cell stayed NaN until the factor
step failed with a linear-algebra error, a duplicate row silently overwrote its twin and a
fractional time was truncated.

R gsynth 1.4.0 (checked for T2-WP13) behaves differently: it imputes missing cells and missing
outcomes (EM), refuses duplicates ("Observations are not uniquely defined by unit and time
indicators."), and accepts decimal times. p2a does not implement the EM step, so it refuses
those inputs with the messages above.

Units are in sorted order (R `factor()` order), so `unit_effects` and the cross-validation
folds are deterministic; they followed a hash-set order that changed between processes.

Tests (p2a-core, `econometrics::synth::gsynth_tests`): `test_na_gsynth_missing_cell`,
`test_na_gsynth_duplicate_cell`, `test_na_gsynth_null_outcome`,
`test_na_gsynth_fractional_time`, `test_na_gsynth_boolean_treatment`,
`test_gsynth_unit_order_is_deterministic`; MCP (`crates/p2a-mcp/tests/tier2/wp13_causal.rs`):
`test_mcp_gsynth_unbalanced_panel_is_explained`. Not R-pinned.

## Timing

Release build, one P core (`taskset`, `RAYON_NUM_THREADS=1`), three interleaved rounds at 811c26bb against
`fe13b889` (before the rewrite to Xu's estimator); new / base as the ratio of medians. Figures marked
(69b202ae) are not from that run: seven interleaved rounds at that commit, on a machine under load. The
base computed a different estimate on these panels: power iteration on the pre-periods, cross-validation
on in-sample fit (CAUS-43).

| Fit | panel | new / base |
|---|---|---|
| IFE r = 1, unit effects, no covariates (same call as criterion `gsynth`; 69b202ae) | 180 / 500 / 1,500 cells | 1.01 / 1.121 / 0.96 |
| criterion `new_methods_benchmarks`, `gsynth` | 180 / 500 / 1,500 cells | 1.04 / 1.19 / 1.02 |
| IFE r = 2, two-way effects, no covariates | `gsynth_simdata` (1,500 cells) / 50 x 20 panel | 1.10 / 1.105 (69b202ae) |
| LOO CV at a single r (`n_factors` = `max_factors` = 2), no covariates; this tree runs LOO, the base the plain fit (69b202ae) | `gsynth_simdata` / 50 x 20 panel | 1.149 / 1.136 |
| LOO CV over r = 0..2, no covariates; both end on r = 2 (69b202ae) | `gsynth_simdata` / 50 x 20 panel | 0.35 / 0.36 |
| LOO CV over r = 0..5, no covariates; the two pick different r, base 5, new 2 (69b202ae) | `gsynth_simdata` / 50 x 20 panel | 0.12 / 0.12 |
| default (LOO CV over r = 0..10), no covariates; the two pick different r | `gsynth_simdata` / 50 x 20 panel | 0.05 / 0.05 |
| IFE r = 2, covariates X1, X2 | `gsynth_simdata` | 7.62 |

The criterion row at 500 cells costs 1.19x, which is accepted, for the reason below. The fits with
covariates cost 7.62x, which is accepted: that row is the iteration gsynth runs and the base did not.
The other figures above 1.1x are the marked ones.

The fit with no covariates costs more than the base because the factors come from a full symmetric
eigendecomposition of the demeaned controls' Gram matrix, as gsynth's `panel_factor` does, where
the base ran power iteration. On the 500-cell panel of the IFE r = 1 row (20 x 20 Gram; at 69b202ae,
seven rounds, medians), copies of the two steps timed on their own take 0.089 ms (`gs_pca_with`: Gram,
eigendecomposition, factors, loadings and common component) and 0.006 ms (the base's
`extract_factors_svd`, on its 10 pre-periods). Of the whole fits, 0.239 ms new and 0.213 ms base,
that leaves 0.150 ms and 0.207 ms for everything else: the factor step adds 0.084 ms, the other
steps give back 0.058 ms, and the difference is the 0.026 ms above the base. A solver for the leading eigenvectors only has not been
tried: its vectors would match the full decomposition only to its own tolerance, and whether that
holds the fits to their tests has not been measured.

Without covariates the fit has no iteration, and its fixed parts are computed once: the panel is
placed cell by cell without sorting its rows, the means, demeaned outcomes and eigendecomposition
are shared by every r of a cross-validation, and treated units with the same first treated period
share a QR of their pre-period design and its leverages, one period's at a time.
`test_gsynth_shared_fits_equal_fresh_fits_bit_for_bit` holds these shared fits to a fresh fit, bit
for bit, for every force and r on five panels.

## References

- Xu, Y. (2017). "Generalized Synthetic Control Method: Causal Inference with Interactive Fixed Effects Models." *Political Analysis*, 25(1), 57-76.
- R Package: `gsynth` (Yiqing Xu). https://yiqingxu.org/packages/gsynth/
