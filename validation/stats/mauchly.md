# Validation: Mauchly's test of sphericity (mauchly.test)

## Method Overview

`p2a_core::stats::mauchly::mauchly_test_with(data, &MauchlyConfig { transform, sigma })` is R's
`mauchly.test` on an intercept-only multivariate fit `lm(Y ~ 1)`: it tests whether the covariance
of the transformed measures `T y` is proportional to `T Sigma T'`. The transformations are:

| `MauchlyTransform` | R | T |
|---|---|---|
| `Contrasts` (default) | `X = ~1` | the p - 1 orthonormal Helmert contrasts |
| `Identity` | `T = diag(p)` (R's own default) | the measures |
| `Matrix(t)` | `T = t`; any `idata`, `M`, `X` design through its `T` | user rows, p columns |

W, the epsilons and the p-value depend only on the row space of T, so R's `idata`/`M`/`X` designs
are tested by passing R's `T` (`proj(M) - proj(X)` thinned to its row space). `sigma` (R's
`Sigma`, identity when absent) must be p x p, finite, symmetric and positive definite.
`mauchly_test(data, sigma)` uses the contrasts; `run_mauchly_test(data, n, p)` and
`run_mauchly_test_with_sigma(data, n, p, sigma)` take row-major slices.

With S the sample covariance, B = T S T' and Psi = T Sigma T', the l are the eigenvalues of
Psi^-1 B (R's `U = solve(Psi, B)`), computed from the symmetric Psi^-1/2 B Psi^-1/2. With pp
rows of T and nd = n - 1:

- W = prod(l) / mean(l)^pp;
- rho = 1 - (2 pp^2 + pp + 2) / (6 pp nd), z = -nd rho ln W (`chi_squared`), f = pp (pp + 1) / 2 - 1 (`df`);
- p = Pr1 + w2 (Pr(chi2_{f+4} > z) - Pr1), Pr1 = Pr(chi2_f > z) (`p_value_first_order`), with
  Anderson's w2 = (pp + 2)(pp - 1)(pp - 2)(2 pp^3 + 6 pp^2 + 3 pp + 2) / (288 (nd pp rho)^2);
  f = 0 (one contrast) gives W = 1, p = 1;
- Greenhouse-Geisser (sum l)^2 / (pp sum l^2), Huynh-Feldt ((nd + 1) pp GG - 2) / (pp (nd - pp GG))
  capped at 1, lower bound 1 / pp (R's `stats:::sphericity`).

Fewer subjects than contrasts plus one, or a singular B, is `InsufficientData` (R returns W = 0).

The MCP tool `stats_mauchly_test` reads the repeated-measures columns through the shared
column extractor (`crate::tools::common::numeric_matrix`), so integer columns are accepted. It
uses the contrasts and reports `p_value` and `p_value_first_order`.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `mauchly.test`, `stats:::sphericity` | R 4.3.3 |

## R case: `stat_mauchly_sigma` (core)

`validation/r/cases/stats_power_loglin.R`, case `stat_mauchly_sigma`, dataset
`validation/datasets/generated/mauchly_sigma.csv` (15 subjects, `V1`..`V4`). Designs: `T = I`;
`X = ~1`; `idata = data.frame(f = c("a", "a", "b", "b"))` with `X = ~f` (passed as R's `T`,
`within_f_T`); `Sigma = diag(1:4)` with `T = I` and with `X = ~1`.

| Design | W | p (Anderson) | R's p | first-order p | GG | HF (R, uncapped) |
|---|---|---|---|---|---|---|
| `identity` | 0.27312588530899923 | 0.06667846608351051 | 0.06667846608351051 | 0.06452607913443756 | 0.6479740616003805 | 0.8081633129517733 |
| `contrasts` | 0.6470236361186349 | 0.35492213704567077 | 0.3549529811761251 | 0.3536986532043151 | 0.7949341362563177 | 0.9691910351724787 |
| `within_f` | 0.9184539874936128 | 0.5752718538887099 | 0.5752718538887099 | 0.5752718538887099 | 0.92460236405716 | 1.059110549795335 |
| `psi_identity` | 0.10612736435182163 | 0.0011119808489157392 | 0.0011119808489157392 | 0.0010097098383457914 | 0.5183224978912854 | 0.6099617969435276 |
| `psi_contrasts` | 0.4651968252024281 | 0.08388320744865516 | 0.08390388736441756 | 0.08306290412341338 | 0.7729093006631138 | 0.935426617203868 |

`p (Anderson)` is the case's `{design}_p_anderson`, computed in R from R's own `Pr1`, `Pr2` and
Anderson's w2; it equals R's p whenever pp = p or pp = 2. Two measures (`two_col`): W = 1, p = 1.

Tests (`p2a-core`, `stats::mauchly::tests::`):

- `test_validate_mauchly_sigma`: per design W, chi-squared, df, pp, the first-order p, GG
  (`Tol::stat(15)`, `Tol::p_value(15)`, exact), the p-value against `p_anderson`
  (`Tol::p_value(15)`) and against R's p (1e-3 relative); HF against `min(1, R)`; R's own
  non-orthonormal contrast basis `contrasts_T` gives the same W and p; `mauchly_test(y, Some(Sigma))`
  and `run_mauchly_test_with_sigma` honour Sigma.
- `test_validate_mauchly_two_columns`: W = 1, p = 1, df 0.
- `test_mauchly_rejects_invalid_sigma_and_transform` (not R-pinned): the refusals.

## R case: `stat_mauchly_na`

`validation/r/cases/stats_mcp.R`, case `stat_mauchly_na`, dataset
`validation/datasets/generated/mauchly_na.csv` (16 subjects: `id, t1, t2, t3, t4`, all Int64).
`t2` is missing for subject 5 and `t4` for subject 11, so 14 subjects are complete. The
variances grow across the conditions, so sphericity is rejected.

```r
fit <- lm(cbind(t1, t2, t3, t4) ~ 1, data = d)    # drops the 2 incomplete subjects
mauchly.test(fit, X = ~1); stats:::sphericity(SSD(fit), X = ~1)
# z and the first-order p-value are mauchly.test.SSD's own intermediate quantities
```

| Quantity | R value |
|---|---|
| W | 0.09759122928472669 |
| chi-squared (R's `z`) / df | 27.277231938098648 / 5 |
| first-order p-value | 5.0371160244126754e-05 |
| Greenhouse-Geisser / Huynh-Feldt / lower bound | 0.561998278305735 / 0.6364951336453561 / 0.3333333333333333 |
| subjects used / in the data | 14 / 16 |
| R's reported p-value (with the second-order term) | 5.450912248903203e-05 (not pinned, STAT-82 below) |

## Missing values

A subject (row) with a missing value in any of the repeated-measures columns is dropped, as
`lm(cbind(...) ~ 1)` drops it, and reported under `missing` (`n_total`, `n_used`, `n_dropped`,
`by_column`); `n_subjects` counts the complete subjects. A text column is refused with "Column
'x' contains non-numeric values".

`sphericity_violated` is `p < 0.05` when the p-value is finite and `null` otherwise, so a
p-value that could not be computed is never read as "sphericity holds".

## Tests

- `tools::handlers::stats::tests::test_validate_mcp_stats_mauchly_test_na` (p2a-mcp: W,
  chi-squared, df, `p_value_first_order`, the three epsilons, the subjects used, the verdict and
  `missing`; the refusal of two columns)
- `tools::handlers::stats::tests::test_mcp_stats_mauchly_verdict_is_null_for_non_finite_p`
  (not R-pinned: the verdict of a NaN or infinite p-value is `null`)

**Tolerance:** `validation/README.md`, n = 14 (< 100): W, chi-squared and the epsilons 1e-5
relative (`Tol::stat`), p 1e-5 relative plus 1e-12 absolute (`Tol::p_value`), df and counts
exact.

**Status:** R-pinned with a known deviation (registry `validation/methods/mcp_stats.json`, id
`mcp_stats_mauchly`).

## Known Differences

- **STAT-82:** the p-value uses Anderson's (2003, Sec. 10.7.4) second-order coefficient w2 with
  `3 pp`; R's `mauchly.test.SSD` writes `3 * p` (p the number of measures), a typo against the
  expansion, whose every other term uses pp. The two agree when pp = p (`T = I`) or pp = 2 (w2 = 0);
  for the default contrasts they differ by about 1e-4 relative (0.354922 against R's 0.354953 on
  `stat_mauchly_sigma`). Registry deviations in `stats_power_loglin.json` (`mauchly_sigma`) and
  `mcp_stats.json` (`mcp_stats_mauchly`, where `p_value` is not pinned).
- **Default transformation:** p2a's default is R's `X = ~1` (contrasts); R's `mauchly.test`
  default is `T = I`, available as `MauchlyTransform::Identity`.
- **Huynh-Feldt** is capped at 1, as `anova.mlm` uses it; `stats:::sphericity` prints it uncapped.
- **Two columns:** with 2 repeated measures there is one contrast and sphericity holds
  trivially; R and the core function report W = 1, p = 1. The MCP tool refuses fewer than 3
  columns, as its request documents.
