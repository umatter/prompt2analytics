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
  (Gamma_j + Gamma_j')` for the lag count `L = bandwidth` or `floor(4 (n/100)^(2/9))`, with optional VAR(1)
  prewhitening (`D S D'`, `D = (I - A)^-1`, the OLS HAC's prewhitening shared). Moments are centred by default
  (`centeredVcov = TRUE`). The kernel weights:

  | kernel | weight of lag j | lags summed | gmm call |
  |---|---|---|---|
  | Bartlett | `1 - j/(L+1)` | 1..=L | `kernel = "Bartlett", bw = L + 1` |
  | Parzen | `k(j/(L+1))` | 1..=L | `kernel = "Parzen", bw = L + 1` |
  | Tukey-Hanning | `k(j/(L+1))` | 1..=L | `kernel = "Tukey-Hanning", bw = L + 1` |
  | Truncated | 1 | 1..=L+1 (`k(1) = 1`) | `kernel = "Truncated", bw = L + 1` |
  | Quadratic Spectral | `k(j/L)` | 1..=L | not reproduced (deviation filed under ECON-68, the closest finding; no finding covers the QS lag rule) |

  The lags summed are sandwich's `weightsAndrews` (every lag up to n - 1, cut after the last weight above 1e-7),
  computed from the n rows (n - 1 with prewhitening). For the Quadratic Spectral kernel sandwich keeps every lag up to
  n - 1 (500 of 500 at n = 500), which the direct lag loop would make an O(n^2 q^2) computation; p2a keeps its earlier
  L-lag rule there and does not match sandwich (`qs_*` and `qs_pw_*` of `gmm_hac_kernels_t4` are recorded, not
  pinned). `hac_lags` is the last lag summed.

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
| `test_validate_gmm_hac_kernels_gmm_iv` (case `gmm_hac_kernels_t4`) | `{bartlett,parzen,truncated,tukey_hanning}[_pw]_{coef,se,J,J_p,n_weights}` |
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

Release builds of three trees: `ff8e7559` (before the GMM rework), v0.5.0 (`f80b0131`) and this version.
The timing harness fits linear IV with q = 5 instruments and k = 4 coefficients on simulated data at n =
100, 1,000 and 10,000. Quiet re-time, 2026-11-15 (this tree at `2d0bb8b6`): one core (`taskset -c 2`,
`RAYON_NUM_THREADS=1`), cwd `/dev/shm`; the three builds interleaved per row with the order rotated per
round, each process the median of its calls after a warm-up (calls calibrated to about 0.8 s per
process); cells are the median over five rounds (nine or ten for rows within 1.05-1.15x), the ratio of
the minima in brackets; 1-minute load average 1.0-3.0. In `ff8e7559` the MDS rows are its `IID` setting,
which summed `z z' e^2`, the MDS form.

| Fit | n | `ff8e7559` µs | v0.5.0 µs | new µs | new / `ff8e7559` [min] | new / v0.5.0 [min] |
|---|---|---|---|---|---|---|
| one-step, identity W (`vcov = Fixed`) | 100 | 39.8 | 38.1 | 38.5 | 0.97 [0.96] | 1.01 [1.00] |
| one-step, identity W (`vcov = Fixed`) | 1,000 | 212 | 208 | 211 | 1.00 [0.99] | 1.01 [1.01] |
| one-step, identity W (`vcov = Fixed`) | 10,000 | 2,501 | 2,556 | 2,571 | 1.03 [1.03] | 1.01 [1.00] |
| two-step MDS (signed off 2026-11-13) | 100 | 50.7 | 55.7 | 56.6 | 1.12 [1.12] | 1.01 [1.02] |
| two-step MDS | 1,000 | 285 | 255 | 259 | 0.91 [0.91] | 1.02 [1.01] |
| two-step MDS | 10,000 | 3,194 | 2,892 | 2,938 | 0.92 [0.92] | 1.02 [1.01] |
| iterative MDS | 100 | 141 | 94.7 | 97.1 | 0.69 [0.70] | 1.03 [1.03] |
| iterative MDS | 1,000 | 532 | 312 | 320 | 0.60 [0.61] | 1.02 [1.02] |
| iterative MDS | 10,000 | 5,367 | 3,148 | 3,191 | 0.59 [0.60] | 1.01 [1.02] |
| two-step IID | 100 | 50.6 | 52.5 | 53.4 | 1.05 [1.05] | 1.02 [1.01] |
| two-step IID | 1,000 | 286 | 225 | 230 | 0.81 [0.81] | 1.02 [1.02] |
| two-step IID | 10,000 | 3,194 | 2,485 | 2,525 | 0.79 [0.79] | 1.02 [1.01] |
| two-step HAC | 100 | 79 | 76.4 | 77.4 | 0.98 [0.97] | 1.01 [1.02] |
| two-step HAC | 1,000 | 702 | 458 | 461 | 0.66 [0.65] | 1.00 [1.00] |
| two-step HAC | 10,000 | 10,874 | 5,701 | 5,761 | 0.53 [0.53] | 1.01 [1.01] |
| iterative HAC | 100 | 602 | 233 | 236 | 0.39 [0.39] | 1.01 [1.01] |
| iterative HAC | 1,000 | 2,701 | 843 | 856 | 0.32 [0.32] | 1.01 [1.02] |
| iterative HAC | 10,000 | 36,175 | 9,020 | 9,150 | 0.25 [0.25] | 1.01 [1.01] |
| CUE MDS (accepted as a correctness change) | 100 | 141 | 204 | 209 | 1.49 [1.52] | 1.03 [1.03] |
| CUE MDS (accepted as a correctness change) | 1,000 | 531 | 580 | 592 | 1.11 [1.14] | 1.02 [1.02] |
| CUE MDS (accepted as a correctness change) | 10,000 | 5,366 | 5,654 | 5,676 | 1.06 [1.04] | 1.00 [1.00] |
| CUE HAC | 100 | 601 | 339 | 346 | 0.58 [0.58] | 1.02 [1.02] |
| CUE HAC | 1,000 | 2,691 | 1,566 | 1,574 | 0.58 [0.58] | 1.00 [1.00] |
| CUE HAC | 10,000 | 36,091 | 23,214 | 23,189 | 0.64 [0.65] | 1.00 [1.00] |

Quiet re-time against v0.5.0 alone (2026-11-13; `ff8e7559` was not rebuilt). Release builds of
v0.5.0 (`f80b0131`) and this version, one P core (`taskset -c 2`, `RAYON_NUM_THREADS=1`), five
rounds interleaving the two builds in alternating order; each round is one process per build
reporting the median of its calls after warm-up calls. The table gives the ratio of the medians of
the five per-round medians, with the ratio of their minima in brackets (1-minute load average
1.7-2.6). Linear IV, q = 5 instruments and k = 4 coefficients:

| Fit | n | v0.5.0 µs | new µs | new / v0.5.0 [min] |
|---|---|---|---|---|
| one-step, identity W (`vcov = Fixed`) | 100 | 14.5 | 14.5 | 1.00 [0.93] |
| one-step, identity W (`vcov = Fixed`) | 1,000 | 84.3 | 82.9 | 0.98 [1.01] |
| one-step, identity W (`vcov = Fixed`) | 10,000 | 971 | 965 | 0.99 [0.99] |
| two-step MDS | 100 | 21.5 | 21.8 | 1.01 [1.02] |
| two-step MDS | 1,000 | 104 | 99.5 | 0.96 [1.00] |
| two-step MDS | 10,000 | 1,137 | 1,124 | 0.99 [1.00] |
| iterative MDS | 100 | 36.7 | 36.6 | 1.00 [1.03] |
| iterative MDS | 1,000 | 125 | 124 | 0.99 [0.99] |
| iterative MDS | 10,000 | 1,210 | 1,227 | 1.01 [0.99] |
| two-step IID | 100 | 20.5 | 20.7 | 1.01 [1.01] |
| two-step IID | 1,000 | 91.4 | 92.8 | 1.01 [0.98] |
| two-step IID | 10,000 | 968 | 950 | 0.98 [1.00] |
| two-step HAC | 100 | 29.7 | 29.7 | 1.00 [0.97] |
| two-step HAC | 1,000 | 179 | 178 | 0.99 [0.99] |
| two-step HAC | 10,000 | 2,213 | 2,234 | 1.01 [0.99] |
| iterative HAC | 100 | 89.9 | 90.2 | 1.00 [1.00] |
| iterative HAC | 1,000 | 331 | 330 | 1.00 [1.02] |
| iterative HAC | 10,000 | 3,526 | 3,484 | 0.99 [1.00] |
| CUE MDS | 100 | 79.7 | 79.8 | 1.00 [1.01] |
| CUE MDS | 1,000 | 228 | 228 | 1.00 [1.00] |
| CUE MDS | 10,000 | 2,176 | 2,098 | 0.96 [1.00] |
| CUE HAC | 100 | 131 | 131 | 1.00 [1.00] |
| CUE HAC | 1,000 | 613 | 608 | 0.99 [1.00] |
| CUE HAC | 10,000 | 8,867 | 8,816 | 0.99 [1.01] |

Every row is within 1.1x of v0.5.0 (0.96-1.01).

For q <= 8 the MDS moment covariance of linear IV is formed column by column (`LinearIv::mds_omega`:
each moment `z_j * e` as one contiguous vector, centred by its own mean, then q (q + 1) / 2 dot
products) instead of the n x q moment matrix and a matrix product of its transpose. It equals the
matrix form to rounding. CUE's variance reuses the Omega and the weighting matrix it computed at the
estimate.

Two-step MDS at n = 100 costs 1.12x of `ff8e7559` on the quiet re-time (56.6 µs against 50.7 µs; 1.25x
on a loaded machine); signed off on 2026-11-13 (rule 5).
The variance re-evaluates Omega at the two-step estimate, as `gmm::gmm` does (`ff8e7559` reused the
first-step weighting matrix and did not match R's standard errors), so a two-step fit forms two
moment covariances and inverts six small matrices (Z'Z, X'Z W Z'X twice, Omega twice,
G' Omega^-1 G) where `ff8e7559` inverted four. No moment covariance or inverse is computed twice at
the same estimate in this path. Within this method no cheaper form with the same values and
singularity refusals was found (a fused row loop and a const-generic accumulator were tried earlier); a
small-dimension Cholesky path in shared linear algebra could reduce the cost and would need its own
package and sign-off.

### Kernels and prewhitening (against `3db88b8e`)

Linear IV, k = 2, q = 4, AR(1) errors, HAC weighting, release builds, one core, interleaved, median of 11
runs, each the mean of 200 / 40 / 5 fits at n = 100 / 1,000 / 10,000. Bartlett is unchanged; the other
kernels weigh one more lag (`k(j / (L + 1))` over sandwich's lag set). Prewhitening solves the VAR(1) by the
normal equations when the lagged moments are of full rank and well conditioned, and only otherwise by the
shared pivoted QR (which took 1.31-1.41x at n >= 1,000 when used for every fit).

| Fit | n = 100 | n = 1,000 | n = 10,000 |
|---|---|---|---|
| two-step Bartlett, prewhite 0 / 1 | 1.03 / 1.01 | 1.02 / 1.02 | 1.02 / 1.03 |
| iterative Bartlett, prewhite 0 / 1 | 1.02 / 1.02 | 1.02 / 1.02 | 1.02 / 1.05 |
| CUE Bartlett, prewhite 0 / 1 | 1.01 / 1.00 | 1.01 / 1.01 | 1.01 / 1.07 |
| two-step Parzen, prewhite 0 / 1 | 1.05 / 1.03 | 1.09 / 1.05 | 1.06 / 1.07 |
| two-step Quadratic Spectral, prewhite 0 / 1 | 1.00 / 1.01 | 1.02 / 1.00 | 1.01 / 1.04 |
| two-step Truncated, prewhite 0 / 1 | 1.08 / 1.03 | 1.08 / 1.06 | 1.06 / 1.06 |
| two-step Tukey-Hanning, prewhite 0 / 1 | 1.05 / 1.03 | 1.08 / 1.06 | 1.07 / 1.07 |

Panel FE / RE / LLC after the shared label and grouping helpers (same protocol, 10 periods per unit):
FE 1.00 / 1.01 / 1.01, RE 1.02 / 1.01 / 1.01, LLC 1.02 / 1.01 / 1.02 at n = 100 / 1,000 / 10,000.
