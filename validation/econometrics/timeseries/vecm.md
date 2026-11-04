# Validation: Vector Error Correction Model (Johansen procedure)

**Status:** pinned to R (T3-WP14, ECON-52): `urca::ca.jo` and `urca::cajorls`, urca 1.3.4, R 4.3.3.
Registry: `validation/methods/ts_vecm.json`. R code: `validation/r/cases/ts_vecm.R`
(`Rscript validation/r/run.R --write vecm_denmark vecm_sim3`; `--check` verifies).

## What p2a computes

`p2a_core::econometrics::timeseries::vecm::run_vecm_with(dataset, columns, &VecmConfig)` and the matrix form
`johansen_vecm`. The library, MCP `ts_vecm` and CLI `p2a ts vecm` call `run_vecm(dataset, columns, lags, rank)`,
which is `ecdet = none`, `K = lags`, `r = rank`.

The model is the VAR(K) in levels written in the transitory error-correction form,
`Δy_t = Π y_{t-1} + Γ_1 Δy_{t-1} + ... + Γ_{K-1} Δy_{t-K+1} + μ + e_t`, with `Π = α β'` at rank r.
The eigenproblem is solved through the Cholesky factors of `S11` and `S00` (a symmetric eigen-solve), not by
urca's code path; the results are compared at the solution.

| Step | p2a field | R |
|---|---|---|
| eigenvalues, decreasing (k, or k + 1 with a restricted term) | `eigenvalues` | `ca.jo@lambda` |
| trace / maximal-eigenvalue statistics, index r = H0 rank <= r | `trace_stat`, `eigen_stat` (`trace_stats` kept) | `rev(ca.jo(type = "trace" / "eigen")@teststat)` |
| Osterwald-Lenum critical values [10%, 5%, 1%] | `trace_cv`, `eigen_cv` (`trace_crit_values` = 5%) | `ca.jo@cval`, reversed rows |
| eigenvectors normalised on the first variable | `v_normalised` | `@V` |
| weights and unrestricted PI | `w_loadings`, `pi` | `@W`, `@PI` |
| beta, normalised by its top r x r block | `beta` (rows `beta_rows`) | `cajorls(r)$beta` |
| alpha, constant, Gamma (OLS on [ZK beta, Z1]) | `alpha`, `constant`, `gamma` | `coef(cajorls(r)$rlm)` |
| rank from the trace test at 5% (first r not rejected) | `selected_rank` | none; derived |

## Options

| `VecmConfig` | R | Values |
|---|---|---|
| `lag_order` | `K` | at least 2; `ca.jo` refuses K < 2 |
| `ecdet` | `ecdet` | `none` (default, as urca), `const` (also `constant`), `trend` |
| `rank` | `r` of `cajorls` | 1..k-1; `None`: `selected_rank`. Rank 0 or k gives no restricted estimates and a warning. |

Error-correction form: transitory only. urca's default `spec = "longrun"` gives the same eigenvalues, statistics,
beta and alpha, but Gamma_i = -(I - A_1 - .. - A_i) instead of -(A_{i+1} + .. + A_K). The references call
`ca.jo(spec = "transitory")` explicitly.

Refused, where urca warns or computes nonsense:
- more than 11 variables (urca has no critical values);
- N < K + m + k + p1 (m short-run regressors, p1 level terms), where urca returns canonical correlations of 1 and
  NaN statistics for N = 12..14 on the Danish data;
- collinear short-run regressors, for example a constant series (the error names the aliased regressor);
- exact error-correction identities (squared canonical correlation 1);
- missing values (urca drops incomplete rows with `na.omit`, joining non-adjacent periods).

An eigenvector with no weight on the first variable is scaled by its largest element, with a warning (urca divides
by zero); a restricted VECM whose first r vectors have that property is refused.

Not supported: `season`, `dumvar`, urca's long-run form, the implied levels VAR (`vars::vec2var`) and urca's
`@GAMMA` slot (re-scope 2026-10-30).

## Cases

| Case | Data | Fits |
|---|---|---|
| `vecm_denmark` | urca's `denmark` (LRM, LRY, IBO, IDE; 1974:1-1987:3, N = 55, T = 53), without the seasonal dummies of urca's example | K = 2, transitory: const (r = 1, 2), none (r = 1) |
| `vecm_sim3` | `set.seed(20261101)`: x1 random walk, x2 = 0.5 x1 + AR(1) + noise, x3 = random walk + 0.3 x1 (N = 250, T = 247) | K = 3, transitory: none and trend, r = 1 |

## Decisive values

The full arrays are in `validation/expected/vecm_*.json`.
- **`vecm_denmark`, const:**
  - trace (r <= 3 .. r = 0): 2.2878492651090614, 8.947661300821899, 19.094642159477846, 52.71086603952652, against
    5% values 9.24, 19.96, 34.91, 53.12. Rank 0 at 5%.
  - eigenvalues: 0.46967665581606705, 0.17424112670707045, 0.11808255829236168, 0.04224853642742335, -6.2e-16.
  - beta: 1, -0.9691164017188845, 5.402771872896797, -4.140325466256411, -6.478051134664917.
  - alpha (LRM.d): -0.2997842970195177.
- **`vecm_denmark`, none:** trace 0.5560157619039636, 7.144888376925063, 17.29017198139512, 48.80373095873664
  (rank 1 at 5%); beta 1, -0.9756548953250112, 5.40858766775924, -4.162443413267649.
- **`vecm_sim3`, none:** trace 1.9177282073494066, 15.733731164014204, 81.87497219937299 (rank 1); beta 1,
  -1.980147456398159, -0.06285408162507165.
- **`vecm_sim3`, trend:** trace 2.7019879928553747, 18.83414979756169, 85.48361400374223; beta 1,
  -1.8861940677285607, -0.0613308167549556, 0.003934736743594665.

## Tests and tolerances

All are in `crates/p2a-core/src/econometrics/timeseries/vecm.rs`
(`cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::timeseries::vecm::tests::`):
- `test_validate_vecm_denmark_trace_eigen`: statistics, critical values, eigenvalues, V, W, PI; none statistics.
- `test_validate_vecm_denmark_beta_alpha_gamma`: cajorls beta and coefficients, const r = 1 and 2, none r = 1.
- `test_validate_vecm_sim3_none_trend`: everything for K = 3, none and trend.
- `test_validate_vecm_beta_depends_on_data`: the legacy wrapper `run_vecm` against R, and scale invariance
  (LRM x 100 leaves the eigenvalues unchanged and multiplies beta by 100).
- `test_validate_vecm_rank_selected_by_trace_test`: the rank rule, including rank 0 and rank k (warnings, no
  restricted estimates).
- `test_validate_vecm_options_change_results`, `test_validate_vecm_rejects_degenerate_inputs`,
  `test_validate_vecm_display_shows_tests_and_notes`, `test_validate_vecm_result_serde_round_trip`: properties
  on the case data; they compare no number with R beyond what the tests above pin.

Tolerances (validation/README.md):
- denmark: coefficients 1e-6, statistics 1e-5;
- sim3: 1e-8 and 1e-6;
- critical values and counts: exact;
- W's column for the structural zero eigenvalue (const): abs 1e-11, since it is rounding noise in R.

Measured agreement (largest relative difference over the pinned values): denmark 1.5e-8 (W), 1.2e-9 (PI), 4.7e-10
(cajorls coefficients), 1.4e-10 (V) and below 1e-10 for eigenvalues, statistics and beta; sim3 3.1e-13 or less.

## Notes

- urca orders statistics from H0 r <= k-1 down to r = 0; p2a indexes them by r, and its Display prints urca's order
  and row labels.
- With `ecdet = const | trend`, the last eigenvalue is structurally zero, and it enters the trace statistics as urca
  has it.
- The rank rule is p2a's: urca has no selector.
- Before T3-WP14, `run_vecm` returned fabricated output, so this method was not validated before it:
  - eigenvalues 1/(1 + diag((Y'Y)^-1));
  - critical values 3.84 + 3(k - r);
  - beta the identity, alpha -0.1 and Gamma 0.

## References

Johansen (1988, 1991); Osterwald-Lenum (1992); Pfaff (2008), *Analysis of Integrated and Cointegrated Time Series
with R*.
