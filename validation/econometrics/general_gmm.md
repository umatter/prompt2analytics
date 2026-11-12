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

Release builds of three trees: `ff8e7559` (before the GMM rework), v0.5.0 (`f80b0131`) and this
version. The timing harness fits linear IV with q = 5 instruments and k = 4 coefficients on simulated
data at n = 100, 1,000 and 10,000, on one core (`taskset -c 2`, `RAYON_NUM_THREADS=1`), in nine rounds
that interleave the three builds. Each round reports the median of 41 calls (21 at n = 10,000) after
one warm-up call; the table gives the median of the nine medians. The run shared the machine with
other builds (load average 33-36 on 12 logical CPUs), so ratios within about 0.05 of 1 are noise. In
`ff8e7559` the MDS rows are its `IID` setting, which summed `z z' e^2`, the MDS form.

| Fit | n | `ff8e7559` µs | v0.5.0 µs | new µs | new / `ff8e7559` | new / v0.5.0 |
|---|---|---|---|---|---|---|
| one-step, identity W (`vcov = Fixed`) | 100 | 15.5 | 14.7 | 14.3 | 0.92 | 0.97 |
| one-step, identity W (`vcov = Fixed`) | 1,000 | 52.8 | 46.0 | 46.7 | 0.88 | 1.02 |
| one-step, identity W (`vcov = Fixed`) | 10,000 | 1,662 | 920.8 | 916.8 | 0.55 | 1.00 |
| two-step MDS (not signed off) | 100 | 24.1 | 31.4 | 30.2 | 1.25 | 0.96 |
| two-step MDS | 1,000 | 111.6 | 90.2 | 88.9 | 0.80 | 0.99 |
| two-step MDS | 10,000 | 2,254 | 1,420 | 1,404 | 0.62 | 0.99 |
| iterative MDS | 100 | 126.5 | 95.4 | 91.2 | 0.72 | 0.96 |
| iterative MDS | 1,000 | 307.2 | 142.8 | 140.7 | 0.46 | 0.99 |
| iterative MDS | 10,000 | 4,128 | 2,150 | 2,071 | 0.50 | 0.96 |
| two-step IID | 100 | 24.7 | 27.3 | 26.2 | 1.06 | 0.96 |
| two-step IID | 1,000 | 110.7 | 66.4 | 65.4 | 0.59 | 0.99 |
| two-step IID | 10,000 | 2,286 | 1,636 | 1,652 | 0.72 | 1.01 |
| two-step HAC | 100 | 48.5 | 52.1 | 50.4 | 1.04 | 0.97 |
| two-step HAC | 1,000 | 507.3 | 419.2 | 402.7 | 0.79 | 0.96 |
| two-step HAC | 10,000 | 8,866 | 5,159 | 5,137 | 0.58 | 1.00 |
| iterative HAC | 100 | 926.7 | 613.3 | 592.2 | 0.64 | 0.97 |
| iterative HAC | 1,000 | 1,688 | 697.8 | 680.8 | 0.40 | 0.98 |
| iterative HAC | 10,000 | 29,525 | 8,980 | 8,886 | 0.30 | 0.99 |
| CUE MDS (accepted as a correctness change) | 100 | 126.5 | 390.3 | 373.5 | 2.95 | 0.96 |
| CUE MDS (accepted as a correctness change) | 1,000 | 305.1 | 664.2 | 648.3 | 2.13 | 0.98 |
| CUE MDS (accepted as a correctness change) | 10,000 | 4,030 | 8,399 | 8,843 | 2.19 | 1.05 |
| CUE HAC | 100 | 903.3 | 476.8 | 459.6 | 0.51 | 0.96 |
| CUE HAC | 1,000 | 1,732 | 1,557 | 1,479 | 0.85 | 0.95 |
| CUE HAC | 10,000 | 30,524 | 27,340 | 28,696 | 0.94 | 1.05 |

For q <= 8 the MDS moment covariance of linear IV is formed column by column (`LinearIv::mds_omega`:
each moment `z_j * e` as one contiguous vector, centred by its own mean, then q (q + 1) / 2 dot
products) instead of the n x q moment matrix and a matrix product of its transpose. It equals the
matrix form to rounding. CUE's variance reuses the Omega and the weighting matrix it computed at the
estimate.

Two-step MDS at n = 100 costs 1.25x of `ff8e7559` (30.2 µs against 24.1 µs) and is not signed off.
The variance re-evaluates Omega at the two-step estimate, as `gmm::gmm` does (`ff8e7559` reused the
first-step weighting matrix and did not match R's standard errors), so a two-step fit forms two
moment covariances and inverts six small matrices (Z'Z, X'Z W Z'X twice, Omega twice,
G' Omega^-1 G) where `ff8e7559` inverted four. No moment covariance or inverse is computed twice at
the same estimate in this path. Within this method no cheaper form with the same values and
singularity refusals was found (a fused row loop and a const-generic accumulator were tried earlier); a
small-dimension Cholesky path in shared linear algebra could reduce the cost and would need its own
package and sign-off.
