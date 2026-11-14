# hettx: randomization tests of treatment effect variation

**Status:** R-pinned (registry entries `causal_hettx_dfm` in `validation/methods/causal_hettx_t4.json`
and `hettx_effect_summary_and_imputation` in `validation/methods/numeric_hygiene.json`).

R's `hettx` package is not installed in the validation environment. The procedures of Ding, Feller &
Miratrix (2016, 2019) are written out in p2a and, for the reference values, in base R and `sandwich`
(`validation/r/cases/causal_hettx_t4.R`); the R pin is against that transcription, not against hettx.

## Test of a constant effect (`hettx_dfm_t4`)

H0: tau_i = tau for every unit, tau unknown. The average effect is the difference in means with the
Neyman SE sqrt(s1^2 / n1 + s0^2 / n0) (`ate_se`; it used to be sd(imputed effects) / sqrt(n), the spread
of the imputed unit effects). For each tau on `grid_size` (default 151) evenly spaced points over the
1 - gamma Neyman interval (gamma = 1e-4 by default), Y(0) = Y - tau Z is imputed, every drawn assignment Zb
(a permutation of Z, draw b from stream b of the seed) gives Yb = Y(0) + tau Zb, and
p(tau) = (1 + #{t(Yb, Zb) >= t(Y, Z)}) / (B + 1), with the same B assignments for every tau. The p-value is
min(1, max p(tau) + gamma); `plugin` tests tau = ATE only, without gamma. The old code permuted the
treatment labels with the outcomes fixed, which tests "no effect for anyone": any constant non-zero effect
moved its p-value.

Statistics: the shifted Kolmogorov-Smirnov statistic (default, as hettx; the largest distance between the
ecdfs of the arm-centred outcomes, computed on integer counts max |a n0 - b n1| / (n1 n0) by a merge of four
pieces of the outcome order, O(n) per grid point), or the variance, range, IQR or mean absolute deviation
of the imputed unit effects (matching or regression). Both imputations are linear in the outcome for a
fixed assignment, so each draw imputes once (tau_hat(tau) = imp(Y) + tau imp(Zb - Z)), and the variance is
closed form in tau. Over the grid, the range is read off the upper and lower envelopes of the lines
a_i + tau b_i (built once per draw, O(n log n)), the mean absolute deviation is linear between the kinks
-c_i / e_i of the centred lines (sorted once per draw), and the IQR selects its four order statistics per
grid point (O(n) each). `stratified` imputes the difference in means for every unit (no strata are formed),
so it is refused with a spread statistic and runs with SKS. SKS uses no imputed effects, so it runs without
covariates whatever `effect_method` is: the unit effects reported are then the difference in means
(`effect_method` reported as `stratified`, with a warning).

R values (n = 40, 20 treated, B = 199 stored assignments, 21 grid points, gamma = 1e-4):

| quantity | y_het (tau_i = 1 + 1.5 x1) | y_homo (tau = 2) |
|---|---|---|
| ATE | 1.2523141634600001 | 1.822803991685 |
| Neyman SE | 0.46096121659848616 | 0.32102126510916196 |
| SKS numerator | 80 | 80 |
| p-value (max + gamma) | 0.7851 | 0.6001 |
| plug-in p | 0.68 | 0.5 |

The SKS test has little power at n = 40 (DFM's own finding for small samples). Every p(tau) is pinned
exactly (multiples of 1/200). Test: `test_validate_hettx_frt_sks_stored_assignments`.

## Decomposition (`hettx_dfm_t4`)

**Systematic variation** (DFM 2019): `y ~ 1 + z + X + z:X` by least squares (an aliased interaction is
dropped from the test and its df, with R's aliasing warning); the interaction coefficients are tested with
the Wald chi-squared statistic on their HC2 covariance (`sandwich::vcovHC(type = "HC2")`); the systematic
variance is beta' S_xx beta (S_xx the sample covariance of X). Covariate importance is |beta_j| sd(x_j),
normalised to sum 1.

**Idiosyncratic variation** (DFM 2016, section 6; hettx's `detect_idiosyncratic` with an interaction
formula, plug-in coefficients): H0 tau_i = b_z + X_i' beta. Y(0) = Y - Z tau_i, Yb = Y(0) + Zb tau_i, and
the statistic is the KS distance between the two arms' residuals from their own regressions on X, with
the same drawn assignments as the test of a constant effect.

The old decomposition permuted the imputed effects across units: its idiosyncratic p-value was
1 + 1/(B + 1) - p_sys by construction, and with regression imputation the idiosyncratic part was
identically 0. The total and idiosyncratic variances of the imputed effects are now descriptive only.

| quantity | y_het | y_homo |
|---|---|---|
| beta (x1, x2) | 1.1137538690529611, 0.27741484918074066 | -0.38624613094174404, 0.27741484918194353 |
| HC2 SE (x1, x2) | 0.35044847974916404, 0.4037724485726621 | 0.3504484797464814, 0.40377244856776373 |
| Wald chi-squared (2 df) | 10.13136472658899 | 2.259513868801343 |
| p | 0.006309603991365877 | 0.32311178423764686 |
| systematic variance | 1.2561945372195817 | 0.21721394444931225 |
| covariate-adjusted KS numerator | 80 | 80 |
| idiosyncratic p | 0.525 | 0.525 |

The two outcomes differ by treat (1.5 x1 - 1), which is linear in X, so their covariate-adjusted
residuals and the idiosyncratic test coincide. Test: `test_validate_hettx_systematic_interaction_hc2`.

Property tests: `test_hettx_p_value_is_shift_invariant` (adding c to the treated outcomes moves the ATE and
the interval by c and leaves the p-value), `test_hettx_sks_counts_match_naive` (the merge against the ecdf
definition), `test_hettx_linear_in_tau_imputation_matches_rerun`, `test_hettx_matching_k_nearest_matches_full_sort`
(the neighbour selection against a stable sort, with tied distances), `test_hettx_ate_se_is_neyman`,
`test_spread_on_grid_matches_pointwise` (envelope range and kink-sweep MAD against the statistic at each
grid point), `test_iqr_type7_matches_sorted_quantiles`, `test_hettx_sks_without_covariates_runs`.

Tolerances: `validation/README.md` for n < 100; integer numerators and p-values exact.

## Effect summary and IQR statistic (`qtl_hettx_summary`)

p2a used to read the quantiles of the effect vector at a sorted index (`s[n / 4]`, `s[n / 2]`, ...),
which is none of R's nine quantile types. The summary and the IQR statistic now use
`stats::robust::quantile_sorted` with type 7, R's default.

R call: `quantile(tau, c(0.1, 0.25, 0.5, 0.75, 0.9)); IQR(tau)` on 38 effects with ties, chosen so that
`(n - 1) p` is fractional at 0.25, 0.5 and 0.75.

| Quantity | R (type 7) | old index rule |
|---|---|---|
| p10 | -1.745 | -2.27 |
| p25 | -0.765 | -0.81 |
| median | 0.675 | 0.76 |
| p75 | 2.1875 | 2.25 |
| p90 | 4.11 | 4.53 |
| IQR | 2.9525 | 3.06 |

Min (-3.14), max (6.96) and sd (2.3656879283013965) are pinned too.
Test: `test_validate_hettx_effect_summary_type7`.

## Outcome model on an aliased arm (`inv_hettx_outcome_model`)

Regression imputation fits `y ~ x` on one arm and predicts every row. When the arm's design is
rank-deficient, p2a used to solve the normal equations with a generalized inverse, which splits a
coefficient between collinear columns and gives predictions that differ from R's. It now solves with
`linalg::solve::least_squares` (the normal equations first, the pivoted QR when they are singular), sets
an aliased coefficient to NA as `lm` does, predicts with it at 0 as `predict.lm` does on a rank-deficient
fit, and records R's warning "Coefficients not defined because of singularities: x2".

R call: `lm(y ~ x1 + x2, subset = treat == 1); predict(fit, newdata = d)` with `x2 = 2 x1 + 1` among the
treated. Coefficients: intercept 1.5604117287238273, x1 -0.11117833449336709, x2 NA; all 40 predictions
and the R² are pinned. Test: `test_validate_hettx_outcome_model_aliased_subset`.

Tolerances: `validation/README.md` for n = 20-40.

## Timing

Release builds of v0.5.0 (`f80b0131`) and this version, one core (`taskset -c 2`, `RAYON_NUM_THREADS=1`),
base and new interleaved, median of 11 runs (3 runs for the matching rows at n = 1,000 and 10,000, whose
v0.5.0 runs take 30-55 s each). Simulated data: three normal covariates, half the units treated, effect
1 + 0.5 x1 plus noise, seed fixed; B = 1000 drawn assignments at n = 100 and 1,000, B = 99 at n = 10,000
with regression and B = 9 with matching. The other lanes' builds shared the machine, so absolute times
move between rows; the ratios come from interleaved pairs. "dec" is `decompose = true` (v0.5.0: the
projection null of the permuted effects; now the HC2 interaction test and the covariate-adjusted KS
randomization test, two arm regressions and a sort per draw).

| Configuration | n = 100: v0.5.0 / new s (ratio) | n = 1,000 | n = 10,000 |
|---|---|---|---|
| MCP default (variance, matching k = 3, decompose) | 0.494 / 0.251 (0.51) | 53.7 / 14.2 (0.27) | 42.2 / 6.65 (0.16) |
| `HetTxConfig::default()` (v0.5.0 variance + matching; now SKS + matching) | 0.494 / 0.583 (1.18) | 53.9 / 8.04 (0.15) | 34.6 / 1.03 (0.03) |
| variance, matching | 0.488 / 0.207 (0.42) | 53.6 / 13.8 (0.26) | 33.3 / 6.51 (0.20) |
| variance, matching, dec | 0.494 / 0.251 (0.51) | 54.8 / 14.0 (0.26) | |
| range, matching | 0.491 / 0.228 (0.46) | 53.6 / 14.0 (0.26) | |
| range, matching, dec | 0.494 / 0.269 (0.55) | 53.7 / 14.3 (0.27) | |
| IQR, matching | 0.497 / 0.561 (1.13) | 53.6 / 16.0 (0.30) | 33.4 / 6.55 (0.20) |
| IQR, matching, dec | 0.503 / 0.605 (1.20) | 27.6 / 8.43 (0.31) | |
| MAD, matching | 0.490 / 0.217 (0.44) | 29.8 / 13.8 (0.46) | |
| MAD, matching, dec | 0.494 / 0.258 (0.52) | 27.7 / 7.11 (0.26) | |
| variance, regression | 0.0382 / 0.0434 (1.14) | 0.223 / 0.247 (1.11) | 0.127 / 0.122 (0.96) |
| variance, regression, dec | 0.0420 / 0.0862 (2.05) | 0.268 / 0.519 (1.94) | 0.153 / 0.256 (1.67) |
| range, regression | 0.0383 / 0.0659 (1.72) | 0.224 / 0.433 (1.94) | 0.128 / 0.211 (1.65) |
| range, regression, dec | 0.0436 / 0.107 (2.46) | 0.269 / 0.706 (2.63) | 0.154 / 0.344 (2.23) |
| IQR, regression | 0.0444 / 0.403 (9.09) | 0.290 / 2.39 (8.25) | 0.164 / 1.02 (6.22) |
| IQR, regression, dec | 0.0500 / 0.448 (8.95) | 0.335 / 2.67 (7.96) | 0.202 / 1.43 (7.05) |
| MAD, regression | 0.0383 / 0.0550 (1.43) | 0.223 / 0.336 (1.51) | 0.128 / 0.172 (1.35) |
| MAD, regression, dec | 0.0440 / 0.0990 (2.25) | 0.268 / 0.610 (2.27) | 0.235 / 0.497 (2.11) |

Matching is faster because the k nearest donors are selected instead of sorted. The rows above 1.1x are
the cost of the test itself, not of an implementation choice: v0.5.0 evaluated one statistic per
permutation, while the test of a constant effect evaluates it at every one of the 151 grid values of tau
for every draw (the variance in closed form; the range from the lines' envelopes and the MAD from a kink
sweep, O(n log n) per draw; the IQR by selection, O(n) per grid value), and regression imputation fits two
outcomes per arm. The decomposition rows add the idiosyncratic randomization test, which v0.5.0 did not
run. These rows, and the new default at n = 100, await the user's sign-off.
