# General GMM (linear IV and normal-distribution moments)

**File:** `crates/p2a-core/src/econometrics/general_gmm.rs` (`run_gmm_iv`, `run_general_gmm`; MCP tool `gmm_iv`).
**Reference:** `gmm::gmm` 1.9.1 and `gmm::specTest` (R 4.3.3, sandwich 3.1.3).
**R case:** `gmm_linear_iv` (`validation/r/cases/econ_gmm.R`, data `validation/datasets/generated/gmm_iv.csv`,
n = 500, y on (1, x1, x2, w) with instruments (1, z1, z2, z3, z4): q = 5, k = 4).
**Status:** R-pinned (registry `validation/methods/econ_gmm_mte.json`, entry `general_gmm`), except the two
deviations below.

## Estimators

Moments `gbar(theta) = mean_i g_i(theta)`; linear IV `g_i = z_i (y_i - x_i' theta)`, normal moments
`g_i = (x_i - mu, (x_i - mu)^2 - s2, (x_i - mu)^3)`.

- **Two-step**: 2SLS (linear) or the identity-weight Gauss-Newton minimiser (nonlinear), then
  `W = Omega(theta_1)^-1`; J uses that W.
- **Iterative**: weight updates until `relative_change_vec(theta_new, theta) < tolerance`; J uses the last update's W.
- **CUE**: `Q(theta) = gbar' Omega(theta)^-1 gbar` minimised by BFGS from the two-step estimate; `J = n Q`. For
  linear IV without prewhitening, `Omega(theta) = sum_{l,m} c_l c_m Gamma_lm` with `c = (1, -theta)` and Gamma the
  meat of the stacked functions `[z y, z x_1, .., z x_k]`, so Q and its gradient are exact and cheap; otherwise
  central differences.
- **One-step (fixed W)**: `weight_matrix = W` (q x q, finite, symmetric, positive definite) or `vcov = Fixed`
  (identity). Iterative or CUE with a fixed W is refused.
- **Variance**: optimal W: `(G' Omega(theta)^-1 G)^-1 / n`, Omega re-evaluated at the estimate. Fixed W: the sandwich
  `T Omega T' / n`, `T = (G'WG)^-1 G'W` (gmm's `FinRes`). z statistics, normal p-values.
- **Moment covariance**: IID `sigma^2 Z'Z / n` (two-step = 2SLS); MDS `u'u / n`; HAC `S = eps'eps + sum_j w_j
  (Gamma_j + Gamma_j')`, `w_j = kernel.weight(j, b)`, `b = bandwidth` or `floor(4 (n/100)^(2/9))`, with optional
  VAR(1) prewhitening (`D S D'`, `D = (I - A)^-1`). Moments are centred by default (`centeredVcov = TRUE`).

## p2a configuration and the gmm call it reproduces

| p2a | `gmm::gmm` |
|---|---|
| `method = TwoStep, vcov = MDS` | `type = "twoStep", vcov = "MDS"` |
| `method = Iterative, vcov = MDS` | `type = "iterative", vcov = "MDS", crit = tolerance` |
| `method = CUE, vcov = MDS` | `type = "cue", vcov = "MDS"` |
| `vcov = IID` | `vcov = "iid"` |
| `vcov = HAC` (default) | `vcov = "HAC", kernel = "Bartlett", bw = lags + 1, prewhite = 0` |
| `vcov = HAC, prewhite = true` | `.., prewhite = 1` |
| `vcov = Fixed` | `wmatrix = "ident"` |
| `weight_matrix = W` | `weightsMatrix = W` |
| `centered = false` | `centeredVcov = FALSE` |

`vcov = IID` with the normal-distribution moments is refused (InvalidSpecification); gmm::gmm uses the MDS covariance crossprod(gt)/n there; use `vcov = MDS`.

**p2a's HAC default against gmm's.** p2a's default HAC weight is the Bartlett kernel with
`floor(4 (n/100)^(2/9))` lags (5 at n = 500) and no prewhitening, pinned as `twostep_hac_bartlett`. gmm's default
(Quadratic Spectral kernel, `bwAndrews` bandwidth, `prewhite = 1`) is not implemented: `twostep_hac`,
`twostep_hac_bw`, `cue_hac` and `cue_hac_bw` are recorded in the case and not pinned (deviation ECON-68).

**Fixed-W J.** gmm prints `n gbar' Omega^-1 gbar` for `wmatrix = "ident"` and `weightsMatrix`; that statistic is
chi-squared(q - k) only for the optimal W, so p2a reports `j_stat = NaN` (deviation ECON-68; `ident_mds_J`,
`fixedw_mds_J` recorded, not pinned).

## Tests (`econometrics::general_gmm::tests`)

| Test | Keys |
|---|---|
| `test_validate_gmm_twostep` | `twostep_mds_*`, `twostep_mds_uncentered_*` |
| `test_validate_gmm_iid` | `twostep_iid_*`, `iid_equals_2sls_coef` |
| `test_validate_gmm_iterative` | `iter_mds_*` (Tol::iterative) |
| `test_validate_gmm_hac` | `twostep_hac_bartlett_*`, `twostep_hac_bartlett_pw_*`, `hac_nw_lags` |
| `test_validate_gmm_cue` | `cue_mds_*`, `cue_hac_bartlett_*` (Tol::iterative; p2a's objective no larger than R's) |
| `test_validate_gmm_ident` | `ident_mds_coef/se`, `fixedw_mds_coef/se` |
| `test_validate_gmm_normal_moments` | `normal_twostep_*`, `normal_iter_*` (Tol::iterative) |

Property tests (no R value): `test_gmm_iterative_scale_free`, `test_gmm_cue_quadratic_form_matches_direct`,
`test_gmm_rejects_bad_weight_matrix_and_combinations`, `test_gmm_config_serde_round_trip`, the option-parsing tests.

**Tolerance.** `validation/README.md` at n = 500: coefficients 1e-8, SE, J and p 1e-6 relative. Iterated, CUE and
Gauss-Newton fits use `Tol::iterative` with the same values (R's optimisers stop at reltol 1e-15 / crit 1e-12;
planning gaps 1.2e-15 iterative, 1.3e-10 / 6.3e-10 CUE, 1.8e-11 / 6.9e-10 normal moments).

## Values (from `validation/expected/gmm_linear_iv.json`, four digits)

| Key | (Intercept) | x1 | x2 | w | J |
|---|---|---|---|---|---|
| `twostep_mds_coef` | 0.9985 | 0.9442 | -0.5377 | 0.3503 | 0.007289 |
| `twostep_mds_se` | 0.04163 | 0.1008 | 0.06649 | 0.04648 | |
| `iter_mds_coef` | 0.9985 | 0.9442 | -0.5377 | 0.3503 | 0.007305 |
| `cue_mds_coef` | 0.9985 | 0.9442 | -0.5377 | 0.3503 | 0.007304 |
| `twostep_iid_se` | 0.04106 | 0.07618 | 0.06221 | 0.04903 | 0.006934 |
| `twostep_hac_bartlett_se` | 0.04155 | 0.1010 | 0.06482 | 0.04346 | 0.007428 |
| `twostep_hac_bartlett_pw_se` | 0.04106 | 0.1004 | 0.06426 | 0.04296 | 0.007461 |
| `ident_mds_se` | 0.04167 | 0.1036 | 0.06688 | 0.04657 | (NaN) |

| Key | mu | sigma_sq | J |
|---|---|---|---|
| `normal_twostep_coef` | 0.04861 | 0.9094 | 1.415 |
| `normal_iter_coef` | 0.04877 | 0.9069 | 1.406 |

## Timing

Release build, one P core (`taskset`, `RAYON_NUM_THREADS=1`), three interleaved rounds of the timing
harness (linear IV, q = 5, k = 4) at 003af73a against `ff8e7559` (before the GMM rework); new / base as
the ratio of medians.

| Fit | n = 100 / 1,000 / 10,000 |
|---|---|
| one-step, identity W (`vcov = Fixed`) | 0.88 / 0.90 / 0.51 |
| two-step MDS | 1.21 / 0.84 / 0.54 |
| iterative MDS | 0.75 / 0.48 / 0.38 |
| two-step IID | 1.07 / 0.63 / 0.63 |
| two-step HAC | 1.06 / 0.77 / 0.59 |
| CUE MDS (accepted as a correctness change) | 2.85 / 1.72 / 1.15 |

For q <= 8 the MDS moment covariance of linear IV is formed column by column (`LinearIv::mds_omega`:
each moment `z_j * e` as one contiguous vector, centred by its own mean, then q (q + 1) / 2 dot
products) instead of the n x q moment matrix and a matrix product of its transpose; it is about twice
as fast at q = 5 and equals the matrix form to rounding. CUE's variance reuses the Omega and the
weighting matrix it computed at the estimate.

Two-step MDS at n = 100 costs 1.21x of base, which is accepted. The variance re-evaluates Omega at the
two-step estimate, as `gmm::gmm` does (the base reused the first-step weighting matrix and did not
match R's standard errors), so a two-step fit forms two moment covariances and inverts six small
matrices (Z'Z, X'Z W Z'X twice, Omega twice, G' Omega^-1 G) where the base inverted four. At q = 5 each
`sym_inverse` costs about 2.5 µs, most of it faer's per-call Cholesky overhead (about 1.6 µs for a 5 x 5
factorisation), which is shared linear algebra outside this method; the second covariance adds about
2 µs. No cheaper form gives the same values and the same singularity refusals.
