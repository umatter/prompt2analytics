# Multivariate adaptive regression splines (MARS)

`crates/p2a-core/src/ml/mars.rs`. **Not R-pinned**: `earth` is not installed, so no R reference case
exists and the method has no registry entry. The tests below check formulas and properties.

## What is checked

GCV is Friedman (1991), eq. 30, as earth's `GetGcv`: `RSS / (n (1 - C/n)^2)` with `C = M + penalty
(M - 1) / 2` for an additive model, and infinite once `C >= n`.

| Test | What it pins |
|---|---|
| `test_mars_respects_nprune` | The pruned model has at most `nprune` terms, and its GCV is the best among those sizes (n = 120, and n = 240 in fast mode) |
| `test_mars_gcv_infinite_at_df_ge_n` | The formula on hand values, infinity at `d >= n`; a fit at n = 8 keeps `d < n` and a finite adjusted R-squared |
| `test_mars_gcv_matches_friedman_formula` | The returned model's GCV recomputed from its RSS, to 1e-12 relative |
| `test_mars_without_intercept` | `intercept = false`: no constant, least squares on the returned basis to 1e-10 relative, uncentred R-squared as `lm(y ~ 0 + ...)`, `predict` equal to the fitted values exactly |
| `test_mars_design_has_full_rank` | The returned basis is identified: a collinear pair adds only its identified hinge, as earth does |
| `test_mars_incremental_scoring_matches_refits` | Every forward candidate's RSS and every backward deletion's RSS equal a full `least_squares` refit to 1e-9 relative, with the same aliased set |
| `test_mars_stopping_rule_is_scale_free` | `y` rescaled by 1e-6 and 1e6 gives the same terms, with and without the intercept |
| `test_mars_validates_parameters` | Spans, `nprune`, `nk`, `degree`, `penalty`, `thresh`, `fast_k`, non-finite data, feature names |

## Known differences from earth

Not compared, and open:

- Fast mode (`n > 200`) subsamples candidate knots differently from earth's fast MARS (finding WT-3).
- The `minspan`, `endspan` and `penalty` defaults differ: the penalty is 3 for every degree (earth: 2
  for degree 1), and the span formulas differ (finding WT-11).
- Every knot of a variable is scored separately, where earth sweeps a variable's knots in one pass.
  The results of the two do not differ; the sweep would be faster.

## Wall-clock

`release`, p2a-core without default features, against `fe13b889`, five rounds with the sides interleaved per row
(order rotated per round), one P core (`taskset`), one rayon thread, 2026-11-08, load average 1.2-1.5 from the
second round on. Friedman-1 data, p = 5, noise sd 1, `MarsConfig` defaults except `fast`. Cells are the median of
the round medians; "ratio" is this tree over the base, "min ratio" the ratio of the minima.

| `fast` | n | base (ms) | this tree (ms) | ratio | min ratio |
|---|---:|---:|---:|---:|---:|
| default (fast for n > 200) | 100 | 1.48 | 1.24 | 0.83 | 0.84 |
|  | 1,000 | 39.4 | 42.1 | 1.07 | 1.07 |
|  | 10,000 | 1,150 | 520 | 0.45 | 0.45 |
| `Some(false)` | 100 | 1.48 | 1.24 | 0.84 | 0.84 |
|  | 1,000 | 172 | 224 | 1.30 | 1.31 |
|  | 10,000 | 36,200 | 16,600 | 0.46 | 0.46 |

The one row above 1.1x, fast mode switched off by the caller at n = 1,000 (1.30x), is reached only by an explicit
`fast = Some(false)` above n = 200, and was signed off by the maintainer on 2026-11-08: the forward pass detects
rank by Gram-Schmidt instead of adding a ridge, and the same path is 0.46x of the base at n = 10,000.
