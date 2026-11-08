# Projection Pursuit Regression Validation

## Method

Friedman's SMART, as R's `stats::ppr` (`ppr.f`): `y = ybar + sy sum_k b_k f_k(alpha_k' x) + e`, with
unit directions and standardised ridge functions. A forward pass from linear starts, a Gauss-Newton
direction search with step halving, backfitting, and backward elimination from `max_terms` to `nterms`.
The estimate is the local optimum this algorithm reaches, so the algorithm and its stopping rules follow
`ppr.f`; the constants are its single-precision literals (`conv = 0.005_f32`, for one).

Ridge smoothers (`crates/p2a-core/src/ml/ppr/`):

- `supsmu` (`supsmu.rs`): Friedman's super smoother with `bass` and a cross-validated or fixed `span`.
- `spline` and `gcvspline` (`spline.rs`): a 15-coefficient penalised cubic B-spline on `ppr.f`'s knots,
  with the smoothing parameter matched to `df` or chosen by GCV with penalty `gcvpen`. It is a standard
  penalised spline, not a port of `sbart`.

`optlevel`: 0 is the forward pass only; 1 backfits `b` and `f`; 2 adds one direction step per pass (the
default); 3 runs the full direction search per term.

Fitted values, residuals and `predict` come from the stored ridge functions (`pppred`): exact at training
projections, linear between them, constant beyond.

## R reference

`validation/r/cases/ml_ppr_svm.R`:

- `ppr_methods`, dataset `validation/datasets/generated/ppr_sim.csv`: `ppr(nterms = 2, max.terms = 3)`
  with `supsmu` (cv span, bass 5, span 0.3), `spline` (df 5), `gcvspline` (gcvpen 1 and 2), optlevel 0,
  1, 3, weights; one predictor; `predict` on new rows inside and outside the training range.
- `ppr_smoothers`, dataset `validation/datasets/generated/ppr_smooth.csv`: `stats::supsmu` (cv, bass 5,
  span 0.3, weights), and `smooth.spline` on `ppr.f`'s knots (df 5, GCV, GCV penalty 2), with each fit
  also refitted at R's own `spar`.

## Tests and measured agreement

| Test | Tolerance | Largest measured gap |
|---|---|---|
| `test_validate_ppr_supsmu_b0`, `test_validate_ppr_supsmu_b5`, `test_validate_ppr_supsmu_span03`, `test_validate_ppr_optlevels`, `test_validate_ppr_weighted`, `test_validate_ppr_one_predictor` | `Tol::new(1e-8, 1e-10)`; `yb`, `ys` `Tol::coef(n)` | 9.6e-11 relative (a fitted value near 0); 1e-12 absolute |
| `test_validate_ppr_supsmu_smoother` | `Tol::coef(n)` | 0 |
| `test_validate_ppr_spline_df5`, `test_validate_ppr_gcvspline`, `test_validate_ppr_gcvspline_pen2` | `Tol::new(0.03, 0.02)` | see below |
| `test_validate_ppr_spline_smoother` | about 10x measured: `spar` 5e-3, `ratio` 3e-2, last evaluation 0.15 relative + 0.02, refit at R's `spar` 5e-3 relative + 4e-3 | see below |

The spline runs, maximum gap over the vector's largest value:

| Quantity | `spline` df 5 | `gcvspline` | `gcvspline` penalty 2 |
|---|---|---|---|
| alpha | 1.2e-2 | 1.1e-3 | 2.2e-3 |
| beta | 4.7e-3 | 6.1e-4 | 1.4e-3 |
| gofn | 2.3e-3 | 8.5e-3 | 6.6e-3 |
| gof | 2.1e-3 | 2.6e-3 | 2.4e-3 |
| fitted values | 2.6e-3 | 9.0e-4 | 1.5e-3 |
| edf per term | 1.2e-2 | 9.7e-3 | 1.1e-2 |

Three of these gaps exceed 1e-2 (1.20e-2 `spline` alpha, 1.15e-2 `spline` edf, 1.09e-2 `gcvspline`
penalty 2 edf). The pins stay at `Tol::new(0.03, 0.02)` and are recorded as a registry deviation (ML-34):
`sbart` is not ported.

Without R: `test_ppr_fitted_equals_predict_on_training_rows`, `test_ppr_ridge_value_interpolates_and_extrapolates_flat`,
`test_ppr_pprder_pools_and_differentiates` (hand-computed from `ppr.f`'s rule; R exposes no `pprder`),
`test_ppr_pprdir_solves_gauss_newton_step`, `test_ppr_options_change_the_fit`,
`test_smoothing_method_parses_aliases_and_rejects_unknown`, `test_ppr_rejects_invalid_options`,
`test_ppr_reports_options_the_smoother_ignores`, `test_ppr_spline_penalty_is_exact`,
`test_ppr_spline_fit_is_consistent`, `test_ppr_spline_brent_finds_the_minimum`,
`test_ppr_spline_refuses_what_it_cannot_scale`, `test_ppr_spline_fits_tied_abscissae`,
`test_ppr_spline_knots_strictly_increase`, `test_ppr_spline_fits_discrete_predictors`,
`test_ppr_supsmu_ties_and_constant_x`.

## Deviations from R

- **The spline smoothers differ from R by about 1%, on purpose.** R's `sgram.f` writes `0.333` for 1/3 in
  the penalty, and `smooth.spline` returns the fit, df and `lambda` of its last Brent evaluation rather
  than of the `spar` it reports. p2a integrates the penalty exactly and returns everything at the reported
  `spar`. Refitted at R's own `spar`, the two smooths agree to 3e-4 and the df to 3.5e-4, which is the
  `0.333` alone; `spar` agrees to 4.6e-4, `ratio` to 2.5e-3. `lambda` is not compared: it is about 6% from
  R's, which belongs to R's last evaluation. Fits at the reported `spar` are checked inside p2a
  (`test_ppr_spline_fit_is_consistent`).
- **`edf` per term** belongs to the stored ridge function. R reports that of the last smoother call, a
  rejected trial included, and at `optlevel = 0` only for the first term.
- **`optlevel = 0` with `max_terms > nterms`** keeps every forward term and warns. R's object lists only
  `nterms` directions but predicts with all of them.
- **Errors where R has none:** `max_terms < nterms`, fewer than 4 rows (6 for the spline smoothers), and
  projections pooling into fewer than three groups (R stops with `br is too large` or reads out of
  bounds).
- **`gof`** is the RSS the search tracked, as in R: just above the RSS of `fitted` when the last term of the
  final pass kept its old fit.
- **Tied projections with a spline smoother.** About 1/12 of the rows or more tied at either end of a sorted
  projection make `ppr.f`'s inner knots coincide with a boundary knot. R then logs non-finite `spar` values and
  returned a `gofn` of 80.205 on six runs of a discrete-predictor test (not compared further). p2a drops the
  repeated knots, so `nk` follows the shorter knot vector, and fits. Repeats are judged at the rounding of the
  values (64 machine epsilons of the largest, as a share of the range), so untied data keep R's knots unless the
  largest value is about 1e13 times the gap between knots; such a value is refused where R fits it. Too few
  distinct values (4 always suffice) at the start of a search are an error, and any refusal by the smoother later
  in a search is a rejected step.

## Missing and non-finite values

- **Core.** `ppr` refuses non-finite x, y and weights before any work, so projections are
  finite. Ties in a projection keep row order.
- **MCP.** `ml_ppr` drops the rows with a missing value in the target or a feature
  (`NaPolicy::Omit`), as R's formula interface does: `ppr(y_na ~ x1_na + x2, nterms = 1)`
  uses 28 of the 30 rows of the WP17 fixture (`na.omit` drops rows 4 and 7, 0-based;
  R 4.3.3). The JSON reports them under `"missing"`
  (`{"n_total": 30, "n_used": 28, "n_dropped": 2, "by_column": {...}}`), and the fit equals
  the fit on the complete rows.
- **Tests.** `test_ppr_rejects_bad_values_and_honours_cancellation`
  (`crates/p2a-core/src/ml/ppr.rs`), `test_na_mcp_ppr_omits_incomplete_rows`
  (`crates/p2a-mcp/src/tools/handlers/ml.rs`). They compare p2a with p2a and pin no R
  number.

## References

1. Friedman, J. H., & Stuetzle, W. (1981). Projection Pursuit Regression. JASA, 76(376), 817-823.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer.
3. R Core Team. ppr() documentation.
