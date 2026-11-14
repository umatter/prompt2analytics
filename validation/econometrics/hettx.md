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
closed form in tau. `stratified` imputes the difference in means for every unit (no strata are formed),
so it is refused with a spread statistic and runs with SKS.

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
(the neighbour selection against a stable sort, with tied distances), `test_hettx_ate_se_is_neyman`.

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
