# Generalized least squares (`regression::gls`)

**Method:** GLS with identity, AR(1) or compound-symmetry errors, optionally within groups, with the
correlation parameter fixed or estimated by REML or ML (`gls_with`, `run_gls_with`; `gls`,
`gls_ar1_auto`, `run_gls` and `run_gls_raw` delegate).

**R reference:** `nlme::gls(correlation = corCompSymm(form = ~ 1 | g) | corAR1(form = ~ 1 | g),
method = "REML" | "ML")`, case `gls_compsymm` (`validation/r/cases/regression_smoothers_t3.R`); case
`gls_ar1` (`validation/r/cases/regression_legacy.R`).

## Model

`y = X b + e`, `Var(e) = sigma^2 Omega`. With groups (`form = ~ 1 | g`) Omega is block diagonal: the
correlation acts within each group, in data order, and rows of different groups are independent.
Without groups all rows are one block.

- AR(1): `Omega[i, j] = rho^|i - j|` within a block, `|rho| < 1`. Whitening: the first row of a block
  is unchanged, then `v*_t = (v_t - rho v_{t-1}) / sqrt(1 - rho^2)`.
- Compound symmetry: `Omega[i, j] = rho` off the diagonal within a block, `-1/(m_max - 1) < rho < 1`
  (`m_max` the largest block). Whitening: `v* = (v - a mean_g(v)) / sqrt(1 - rho)`,
  `a = 1 - sqrt((1 - rho) / (1 + (m - 1) rho))`.
- Both are O(n p). A known Omega (`gls()` only) uses its Cholesky factor.

## Likelihood (nlme's `logLik.gls`)

With `SSE* = |P y - P X b|^2`, `P'P = Omega^-1`, N rows and p coefficients:

- `log det Omega`: CS `sum_g [(m - 1) ln(1 - rho) + ln(1 + (m - 1) rho)]`; AR(1) `sum_g (m - 1) ln(1 - rho^2)`.
- ML: `l = -N/2 (ln 2 pi + 1 + ln(SSE*/N)) - 1/2 log det Omega`.
- REML: `l = -(N - p)/2 (ln 2 pi + 1 + ln(SSE*/(N - p))) - 1/2 log det Omega - 1/2 ln det(X*'X*)`.
- sigma: `sqrt(SSE*/(N - p))` (REML), `sqrt(SSE*/N)` (ML). Standard errors use `SSE*/(N - p)` for
  both, as nlme's `vcov`.
- `k = p + 1 + [rho estimated]` parameters; `AIC = -2 l + 2 k`; `BIC = -2 l + k ln(N - p)` (REML),
  `k ln N` (ML).

## Estimation and identification

Without a given rho, rho maximises the profile log-likelihood of the method by Brent's method (a port
of R's `Brent_fmin`) on `[lo + 1e-8, hi - 1e-8]`, `tol = 1e-12`, at most 200 evaluations. The
objective is centred on the SSE* of the starting fit, so the search is the same at any scale of y.
`convergence` reports the search; an estimate within 1e-6 of a bound adds a warning.

Compound-symmetry rho is refused when it is not identified: when every group indicator lies in the
column space of X (one group with an intercept, or group dummies among the regressors), the profile
is flat in rho. The rule: with W the within-group demeaning and G the number of groups, the fit is
refused when `G + rank(W X) == rank(X)` (columns of `W X` below `1e-7 |X_j|` set to zero). A
rank-deficient whitened design is an error, as nlme's "computed gls fit is singular".

## Values (Appendix B of T3-WP11)

`gls_compsymm` (N = 150, 30 groups of 5, `y ~ x + z`, `z` constant within groups):

| quantity | nlme | profile (optimize, tol 1e-12) |
|---|---|---|
| rho, CS REML | 0.3839430209 | 0.3839430267 |
| rho, CS ML | 0.3640065887 | 0.3640065781 |
| rho, AR(1) REML within groups | 0.3685858815 | 0.3685850885 |
| logLik, CS REML / ML | -227.1568098 / -223.9137212 | |
| logLik, CS rho = 0.3 fixed, REML | -227.5927820 | |

`gls_ar1` (N = 50, one series): REML phi 0.6819761522, logLik -79.12921201, AIC 166.2584240, BIC
173.7432281; ML phi 0.6532503857, logLik -77.23968553.

## Tolerances

rho 1e-6 relative: the maximum is flat, and Brent resolves rho to about 1.5e-8 |rho|. nlme's own
optimiser stops up to 3e-8 (CS) and 2.2e-6 (grouped AR(1)) relative short of the profile maximum,
so the profile values (`*_profile_*`) are the pins and nlme's own fit a looser cross-check
(`Tol::iterative(1e-6)`, `1e-5` for grouped AR(1) rho and SEs). Coefficients, SEs, sigma, logLik, AIC
and BIC otherwise follow `Tol::coef(n)`, `Tol::se(n)`, `Tol::stat(n)`; counts are exact.

## Tests

`test_validate_gls_compsymm_reml`, `test_validate_gls_compsymm_ml`,
`test_validate_gls_compsymm_fixed_rho_grouped`, `test_validate_gls_ar1_grouped_reml` (case
`gls_compsymm`); `test_validate_gls_ar1_nlme` (case `gls_ar1`: fixed phi, `ar1_auto` REML and ML
phi, logLik, AIC, BIC). Property tests: `test_gls_compsymm_rho_not_identified`,
`test_gls_rejects_rank_deficient_design`, `test_gls_rho_estimate_is_scale_invariant`,
`test_gls_options_parse_aliases_and_reject_unknown`, `test_gls_r_squared_follows_explicit_intercept`.
T0-WP14's `test_validate_gls_fixed_correlation_vs_nlme` pins fixed-rho fits on its own 30-point
data at 1e-8 (values quoted in the test; not registered).

## Interfaces

The MCP tool and the CLI command call `gls()` / `run_gls` and have no `group` or `method` option yet.
Recommendation for WP58/WP60: expose `group` (a column) and `method` (`reml` | `ml`), drop the
handler's 0.5 default rho (estimate instead), and append the result's notes.

## Timing (WP11-13)

Release build, same protocol as `loess.md`. Head/base: AR(1) with fixed rho 1.17 (n = 100), 1.44
(n = 1,000; the pivoted QR replaces the normal equations); `ar1_auto` 11.6 (n = 100), 15.7 (n = 1,000),
5.0 on `gls_ar1` (REML by Brent instead of one lag-1 correlation; pre-approved, WP11-8); grouped CS
REML on `gls_cs` 12.3x the old fixed-rho, ungrouped fit (a new estimator; WP11-9). Smoothing spline
(GCV) 1.00 at n = 100 and 1,000; sensemakr 1.17 (n = 1,000), 1.35 (n = 10,000).
