# Smoothing spline (`regression::smooth_spline`)

## Method

A cubic smoothing spline: lambda is given (`lambda`, `spar`), found by bisection for a
target `df`, or, with neither, chosen by a criterion. The criterion is GCV by default, as
R's `smooth.spline(cv = FALSE)`, or ordinary leave-one-out CV, R's `cv = TRUE`. The result
reports the criterion it used (`criterion`, `None` when df, spar or lambda was given) and its
value (`cv_crit`). Inputs are validated: df in (1, number of unique x], spar finite, lambda
finite and non-negative, weights finite and non-negative. A df target that 30 bisection
steps miss by 0.05 or more is a warning.

The penalty, the spar scale and the df/leverages are p2a's own approximation, not R's
`sbart.c` (REG-29, REG-30, Tier 4), so no smooth.spline number is R-pinned.

### Criterion semantics (REG-53)

| Request | R | p2a |
|---|---|---|
| nothing given | `smooth.spline(x, y)` = `cv = FALSE`: GCV | `SmoothSplineConfig::default()`: GCV |
| GCV | `cv = FALSE` | `criterion = SplineCriterion::Gcv`, or `with_cv(true)` |
| ordinary leave-one-out CV | `cv = TRUE` | `criterion = SplineCriterion::Cv`, `cv = true` (R's meaning; overrides `criterion`), or `with_cv(false)` |

`with_cv` keeps its old argument for source compatibility: its argument has the opposite
sense of R's `cv`. Before, `SmoothSplineConfig` derived `Default` with `cv = false`, and
`cv = true` meant GCV, so a default config (MCP `regression_smooth_spline`) ran LOO-CV while
the CLI (`with_cv(true)`) ran GCV. The MCP default fit and its `cv_crit` change; the CLI is
unchanged. The unified benchmark's "GCV" row now sets `criterion` (with `cv: true` under
R's meaning it would have timed LOO-CV).

## R reference

Case `smoothers_gcv_periodic_loess` (`validation/r/cases/regression_smoothers_t3.R`),
x = sorted `runif(100)`, y = sin(2 pi x) + N(0, 0.3^2) (`set.seed(20261127)`):

```r
smooth.spline(x, y, cv = FALSE); smooth.spline(x, y, cv = TRUE); smooth.spline(x, y, df = 6)
```

## Values (recorded, not compared: REG-29, REG-30)

| Key | R |
|---|---|
| `gcv_df` | 15.321748435572115 |
| `gcv_lambda` | 3.212013000460938e-05 |
| `gcv_crit` | 0.07644935114226169 |
| `cv_df` | 14.708802804818221 |
| `cv_lambda` | 3.875462586472911e-05 |
| `cv_crit` | 0.07592175230027477 |
| `dfmatch6_df` | 5.9989809005108645 |
| `dfmatch6_lambda` | 0.002428943146826809 |

## Tolerances

None against R. The property tests compare p2a's reported criterion with R's definition on
p2a's own fit at 1e-12 relative: GCV `(RSS / n) / (1 - df / n)^2`, CV
`mean((r_i / (1 - h_ii))^2)`.

## Tests

Property tests (no registry entry: nothing is compared with R):

- `test_smooth_spline_default_criterion_is_gcv`: the default equals GCV bit for bit, `cv =
  true` equals `criterion = Cv`, the two criteria give different lambdas, `cv_crit` is R's
  formula on the returned fit, `with_cv(true)` is GCV, a df request reports no criterion
  (data: `validation/datasets/generated/smoothers_t3.csv`).
- `test_spline_criterion_parses_aliases_and_rejects_unknown`: the option table.
- `test_smooth_spline_rejects_invalid_controls`: df, spar, lambda and weights.
- `test_smooth_spline_df_target_is_met`: df = 6 within 0.05, no warning.

T0-WP14's `test_pinned_smooth_spline_outputs` and the band, cancellation and memory tests
are unchanged.

## Deviations

- REG-29: `gcv_lambda`, `cv_lambda`, `dfmatch6_lambda`, `gcv_fit_head` (p2a's discrete
  penalty and spar scale are not R's; Tier 4).
- REG-30: `gcv_df`, `cv_df`, `dfmatch6_df`, `gcv_crit`, `cv_crit` (approximate leverages;
  Tier 4).

## NA handling

`run_smooth_spline` refuses a missing x or y (`NullValues`), as R's smooth.spline; see
[loess.md](loess.md), "Column Types and Missing Values".
