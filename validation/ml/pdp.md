# Partial Dependence and ICE Validation

## Method overview

Partial dependence (Friedman 2001, eq. 10.51): PD(x_s) = (1/n) sum_i f(x_s, x_c,i), the average prediction with the
feature(s) of interest set to a grid value for every observation. ICE (Goldstein et al. 2015): one curve per
observation, ICE_i(x_s) = f(x_s, x_c,i); centred ICE subtracts each curve's value at the smallest grid value. p2a
computes all of them with one engine (`crates/p2a-core/src/ml/pdp.rs`): `feature_grid` builds every grid and
`evaluate_grid` evaluates every model; prediction errors and wrong-length predictions are errors.

| Entry point | Default grid | Notes |
|---|---|---|
| `partial_dependence`, `partial_dependence_named`, `try_partial_dependence` | `PdpConfig::default()`: pdp's `min(length(unique(x)), 51)` uniform points | `center`: PD minus its mean over the grid (no pdp equivalent) |
| `partial_dependence_2d`, `partial_dependence_2d_named`, `try_partial_dependence_2d` | as above per feature; `grid_resolution_2` for the second | `pdp_values[i][j]` = PD(grid_values_1[i], grid_values_2[j]) |
| `compute_ice_curves`, `try_compute_ice_curves` | `IceConfig::default()`: the same rule over the sampled rows; `grid_values` sorted | `center`: pdp's `center = TRUE`; `frac_to_plot` samples rows (grid from the sample) |
| `gbm_partial_dependence`, `cart_partial_dependence` | the caller's uniform grid | `PdpResult` with `model` ("GBM", "CART") |
| `ice_curves_gbm`, `ice_curves_cart`, `ice_curves_rf` | as `compute_ice_curves` | `ice_curves_rf` takes the `RandomForestModel` of `random_forest_with_trees` |

The default call does the same work as `pdp::partial`: up to 51 model evaluations on the whole data, where the
former `PdpConfig` default did 20.

## Grid rules (`pdp:::pred_grid`, pdp 0.8.3)

| `GridSpec` | R | Notes |
|---|---|---|
| `Uniform(n)` | `seq(min(x), max(x), length.out = n)` | R 4.3's `seq.default` length.out branch, including the exact endpoint and the overflow branch (`(from/4 + ...) * 4`); n >= 1. With no `grid_resolution` the configs use `n = min(length(unique(x)), 51)` |
| `Quantiles(probs)` | `quantile(x, probs, type = 7, na.rm = TRUE)` | through `stats::robust::quantile_sorted`; probs in [0, 1]. `use_quantiles` with no `grid_resolution` uses pdp's `1:9/10`, with one `seq(0, 1, length.out = n)` |
| `Values(v)` | `pred.grid` sorted by `order_grid` | non-empty, finite |

NaN feature values are ignored (pdp's `na.rm = TRUE`); infinite values and a feature without finite values are errors.

**Documented differences from `pdp::partial`.** The first three are registry deviations of `pdp_partial_dependence`:

1. Exact duplicate grid values are listed once; pdp repeats the row (same yhat). The ties pin compares with R's
   `unique()`.
2. NaN predictions propagate into the PD mean; pdp's `mean(na.rm = TRUE)` drops them, which would average over a
   subset of the rows without saying so.
3. The default grid size counts non-missing values; pdp's `length(unique(x))` counts a missing value as one more
   point.
4. Grids are sorted ascending, so a quantile grid with unsorted probs is sorted; pdp sorts only a user `pred.grid`.
5. `PdpConfig.center` (PD minus its mean) and `IceConfig.frac_to_plot` have no pdp equivalent.

ICE spread per grid point: population SD (divisor n), range, `IQR()` (type 7); a grid point with a NaN prediction gives
NaN for all three.

## R reference cases (`validation/r/cases/ml_pdp.R`)

| Case | Data | Values |
|---|---|---|
| `pdp_lm` | `generated/pdp_lm.csv`, n = 60: x1 ~ U(-2, 2), x2 ~ N(0, 1), x3 in {0.5, 1.5, 2.5, 3.5}, y from a quadratic with an interaction plus N(0, 0.4^2) noise; fit `lm(y ~ x1 * x2 + I(x1^2) + x3)` | coefficients; the default grid (x1: 51 points, x3: 4 points) and PD; uniform (25) and quantile (1:9/10, `seq(0, 1, length.out = 20)`) grids and PD; 5 x 4 two-feature PD; ICE and centred ICE on the user grid (1.3, -1.7, 0.25, -0.6, 1.9); 8-point ICE with SD, range, IQR; tied x3 grid |
| `pdp_small_scale` | `generated/pdp_small_scale.csv`, n = 40: x = 1e-12 U(1, 3), z ~ N(0, 1); fit `lm(y ~ x + z)` | 9-point quantile and uniform grids and PD; 6-point centred ICE |

The Rust tests rebuild the fits' prediction functions from the committed coefficients. These cases therefore pin the
grid rules, the averaging, the two-feature layout and the centring, not a model. A randomForest fit on `pdp_lm` checks
in R that the grid does not depend on the model.

Decisive values:

| Quantity | R |
|---|---|
| `pd_uniform_grid` [1], [25] | -1.9741500103846192, 1.832898922264576 |
| `pd_uniform_yhat` [1], [25] | -0.013502221001567956, 6.9824359205486166 |
| `pd_quantile20_grid` [11] | 0.009164202223090917 |
| `pd2_yhat` [1, 1], [5, 4] | 5.2891831991166196, 8.3646094977476775 |
| `cice_custom_pd` | 0, 0.63706251067117281, 2.0956062524262227, 5.0603563586533609, 7.3316368232087816 |
| `ice_uniform_iqr7` [1] | 2.4128154243818587 |
| `pd_ties_grid_unique` | 0.5, 1.5, 2.5, 2.6000000000000014 |
| `pd_default_x3_grid` | 0.5, 1.5, 2.5, 3.5 |
| `pdp_small_scale` `pd_quantile_grid` [1], [9] | 1.1289574653841557e-12, 2.7610676951240748e-12 |

## Tests

| Test | Case | Quantities | Tolerance |
|---|---|---|---|
| `test_validate_pdp_grid_rules` | `pdp_lm` | default, uniform and quantile grids and PD; ties | grids exact; PD 1e-12 rel + 1e-12 abs |
| `test_validate_pdp_lm_2d` | `pdp_lm` | 5 x 4 grids and PD | grids exact; PD 1e-12 |
| `test_validate_pdp_small_scale` | `pdp_small_scale` | 9-point grids and PD | grids exact; PD 1e-12 |
| `test_validate_pdp_config_grid_spec` | `pdp_lm` | `PdpConfig`'s default size, pdp's `1:9/10` and R's `seq(0, 1, length.out = 20)` | exact |
| `test_validate_pdp_lm_model_helpers_uniform_grid` | `pdp_lm` | GBM and CART helpers use pdp's uniform grid and the engine's PD | grid exact; PD identical |
| `test_validate_pdp_lm_ice_custom_grid` | `pdp_lm` | sorted grid, ICE, centred ICE, means | grid exact; values 1e-12 |
| `test_validate_pdp_lm_ice_uniform` | `pdp_lm` | grid, ICE, mean, SD, range, IQR; the default grid and its mean curve | grid exact; values 1e-12 |
| `test_validate_pdp_small_scale_cice` | `pdp_small_scale` | grid, centred ICE | grid exact; values 1e-12 |

The tolerance of the values is three orders above the measured difference between Rust's and R's summation order
(8.9e-16 * max(|R|, 1)).
The README coefficient row (n < 100: 1e-6) would also pass, but it would hide a wrong grid point.

Behaviour tests (no R value): `test_pdp_2d_rejects_wrong_prediction_length`, `test_pdp_prediction_error_propagates`,
`test_pdp_config_rejects_invalid_grids`, `test_feature_grid_rules`, `test_pdp_huge_grid_is_a_resource_error`,
`test_pdp_stops_when_cancelled`, `test_model_feature_counts_match_training_columns`,
`test_ice_empty_custom_grid_is_error`, `test_ice_gbm_prediction_error_propagates`,
`test_ice_model_wrappers_refuse_column_mismatch`, `test_ice_parameter_rules`, `test_ice_prediction_error_propagates`,
`test_ice_spread_nan_column`, `test_ice_curves_rf_uses_forest_predictions` (random-forest ICE equals the tree average;
at the observation's own x1, its fitted value, bit for bit), `test_model_pd_rejects_zero_grid_and_column_mismatch`.

## Running

```bash
Rscript validation/r/run.R --check pdp_lm pdp_small_scale
cargo test -p p2a-core --lib --no-default-features --features ci -- ml::pdp ml::ice ml::evaluation
```
