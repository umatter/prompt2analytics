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
| `test_mars_fast_mode_finds_upper_kink` | Fast mode at n = 300 finds a kink at the 80th percentile of `x` (the first `fast_k` candidates used to stop near the 62nd) |
| `test_mars_fast_subsample_is_even` | The fast-mode subsample of 58 candidates keeps indices 0, 3, ..., 57 |
| `test_mars_spans_follow_friedman` | `minspan` and `endspan` against eqs. 43 and 45 computed in base R, unfloored to 1e-14 relative |
| `test_mars_default_penalty_is_earth` | Default penalty 2; `earth_penalty` 2 for degree 1, 3 above |
| `test_mars_refuses_overflowing_response` | A response whose sum of squares overflows is refused ("rescale the response"); one below it fits |
| `test_mars_stopping_rule_is_scale_free` | `y` rescaled by 1e-6 and 1e6 gives the same terms, with and without the intercept |
| `test_mars_validates_parameters` | Spans, `nprune`, `nk`, `degree`, `penalty`, `thresh`, `fast_k`, non-finite data, feature names |

## Known differences from earth

Not compared, and open:

- Fast mode (on by default above 200 rows) keeps `fast_k` (20) candidate knots per predictor, evenly spaced over
  all of its candidates, first and last included (`test_mars_fast_subsample_is_even`,
  `test_mars_fast_mode_finds_upper_kink`). earth's fast MARS (`fast.k`) instead limits the parent terms
  considered and keeps every knot.
- Spans: `minspan` is Friedman's (1991) eq. 43 and `endspan` eq. 45, alpha = 0.05, with `p` the number of
  predictors, as earth; `minspan` is computed once on all `n` rows, where earth counts the non-zero cases of
  each parent, and `endspan` is not adjusted for interaction terms.
- The default `penalty` is 2, earth's value for the default `degree = 1`; `MarsConfig::earth_penalty(degree)`
  gives earth's 3 for higher degrees, and `run_mars` uses it.
- The forward pass adds the pair with the largest GCV reduction and stops when the relative reduction is below
  `thresh`; earth selects on RSS and stops on the gain in R-squared.
- Every knot of a variable is scored separately, where earth sweeps a variable's knots in one pass.
  The results of the two do not differ; the sweep would be faster.

## Wall-clock

`release`, p2a-core without default features, against `ff8e7559`, five rounds with the sides interleaved per row
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

### After the span and penalty change (against v0.5.0)

`release`, p2a-core with the `ci` features, against `f80b0131` (v0.5.0); quiet re-time, 2026-11-15 (this tree at
`2d0bb8b6`): one core (`taskset -c 2`, `RAYON_NUM_THREADS=1`), cwd `/dev/shm`; base and this tree interleaved per row
with the order rotated per round, each process the median of its calls after a warm-up (calls calibrated to about 0.8
s per process); cells are the median of the process medians over five rounds (nine or ten for rows within 1.05-1.15x);
1-minute load average 1.0-3.0. "ratio" is this tree over the base, "min ratio" the ratio of the minima.
Friedman-1 data, p = 5, noise sd 1.

| `fast` | n | base (ms) | this tree (ms) | ratio | min ratio |
|---|---:|---:|---:|---:|---:|
| default (fast for n > 200) | 100 | 2.11 | 3.14 | 1.48 | 1.48 |
|  | 1,000 | 78.2 | 115 | 1.47 | 1.47 |
|  | 10,000 | 1,104 | 1,243 | 1.12 | 1.13 |
| `Some(false)` | 100 | 2.12 | 3.14 | 1.48 | 1.48 |
|  | 1,000 | 535 | 895 | 1.67 | 1.67 |
|  | 10,000 | 39,988 | 74,301 | 1.86 | 1.86 |

The slower rows come from Friedman's spans, signed off on 2026-11-14 (rule 5) as the price of earth's defaults (at
the loaded run's 1.45 / 1.14x default and 2.49 / 1.81x without fast mode; the quiet default at n = 1,000 (1.47x) and
at n = 10,000 (1.12x) and fast mode off at n = 10,000 (1.86x) are above those figures and go to the user): at n = 100
(no fast mode) each predictor has 17 instead of 11 candidate knots, and without fast mode 164 / 1,426 instead of
98 / 768 at n = 1,000 / 10,000; the smaller default penalty (2) can also lengthen the forward pass. Fast mode keeps
`fast_k` knots per predictor in both trees, so its cost at n = 10,000 is unchanged.
