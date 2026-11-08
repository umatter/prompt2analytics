# Kalman filter, smoother and forecast

Status: **R-pinned** (`validation/methods/ts_statespace.json`, method `forecasting_kalman`, case
`kalman_intercepts`, `validation/r/cases/ts_statespace.R`). Recompute with
`Rscript validation/r/run.R --check kalman_intercepts`.

## The model

```
state:        a_{t+1} = T a_t + c + R eta_t,   eta_t ~ N(0, Q)
observation:  y_t     = Z' a_t + d + eps_t,    eps_t ~ N(0, H)
```

`state_intercept` is `c` and `obs_intercept` is `d`; `StateSpaceModel::with_intercepts` sets them and
`StateSpaceModel::validate` checks every field (a state dimension of at least 1, matching dimensions, finite
entries, `H >= 0`, a non-negative diagonal of `Q`, `c` of length m). The filter, the smoother and the
forecast call it, because the fields are public.

## Conventions

- **Predict first.** `init_state` is `a_0` and `init_cov` is `P_0`; the first step predicts
  `a_1 = T a_0 + c`. This is R's `KalmanRun(nit = -1)`.
- **Update.** `v = y - Z'a⁻ - d`, `M = P⁻ Z`, `F = Z'M + H`, `a = a⁻ + M v / F`, `P = P⁻ - M M' / F`.
  The reason for this form is cost and differentiability: it takes O(m²) per update where the Joseph form
  takes O(m³), and `kalman_likelihood_terms` differentiates exactly this form. Agreement with R's digits is
  not the reason. `P` is symmetrised by copying the upper triangle.
- **`F` must be positive and finite.** Otherwise the filter returns `EconError::Computation` naming the
  observation. There is no absolute threshold, so data on any scale behaves the same
  (`test_validate_kalman_filter_scale_invariant`).
- **Missing values.** A NaN observation is a prediction without an update; its innovation and variance are
  NaN and it adds nothing to the log-likelihood. An infinite observation is `InvalidSpecification`.
- **Smoother.** The Durbin-Koopman state smoother (Durbin and Koopman 2012, eq. 4.44): backward from `r = 0`,
  `N = 0`, with `K = T P Z / F` and `L = T - K Z'`, `r ← Z v / F + L' r` and `N ← Z Z' / F + L' N L`
  (`T' r` and `T' N T` at a missing value), then `a + P r` and `P - P N P`. It inverts nothing, so a singular
  `P_{t+1|t}` (a zero variance) is an ordinary case.
- **Forecast.** `a ← T a + c`, `P ← T P T' + R Q R'`, `Z'a + d` and `Z'P Z + H`, from the last filtered
  state.
- **Likelihood gradient (`pub(crate)`).** `kalman_likelihood_terms` returns `Σ v²/F`, `Σ ln F` and their
  exact derivatives: the recursions differentiated (Harvey 1990, sec. 3.4.6). StructTS fits on it.

No step of the filter or of `kalman_likelihood_terms` allocates except the history the filter keeps.
Products with `T` and `Z` go through their nonzeros, computed once per pass.

## The reference

KFAS is not installed. The case runs the model through stats' own C code, with the intercepts carried by a
constant state: `T* = [T c; 0 1]`, `Z* = (Z, d)`, `a*_0 = (a_0, 1)`, through `KalmanRun`, `KalmanLike`,
`KalmanSmooth` and `KalmanForecast` with `nit = -1`. It checks the recursion written out against that at
`1e-10`. The data are a seeded local linear trend, n = 50 (`T = [[1, 1], [0, 1]]`, `Z = (1, 0)`, `R = I`,
`Q = diag(0.3, 0.01)`, `H = 0.5`, `c = (0.1, 0)`, `d = 2`, `a_0 = 0`, `P_0 = 10 I`).

| Key | Value | Test | Tolerance |
|---|---|---|---|
| `loglik` | `-84.7163224699111` | `test_validate_kalman_intercepts` | `Tol::stat(50)` |
| `loglik_no_intercept` | `-84.83460141625507` | `test_validate_kalman_intercepts_change_the_fit` | `Tol::stat(50)` |
| `filtered_level` | `-0.3524627721451925, …, -4.456684694192082` | `test_validate_kalman_intercepts` | `Tol::coef(50)` |
| `filtered_slope` | `-0.22288806509615391, …, -0.14890738534382994` | same | same |
| `innovations` | `-0.46360717540000007, …, -1.9006728466679061` | same | same |
| `innovation_var` | `20.8, …, 1.2580139883379287` | same | same |
| `forecast` | `-2.505592079535912, …, -2.701221620911234` | same | same |
| `forecast_var` | `1.258013984820352, …, 4.736624394455935` | same | same |
| `smoothed_level` | `-0.10765936819228236, …, -4.456684694192082` | `test_validate_kalman_smoother_intercepts` | same |
| `smoothed_slope` | `0.15270596400490016, …, -0.14890738534382994` | same | same |
| `smoothed_level_var` | `0.2899886926390596, …, 0.30127406982946453` | same | same |
| `smoothed_level_fixed_slope` | `2.1282531239469473, …, -2.5123807991292417` | `test_validate_kalman_smoother_singular_covariance` | same |

`test_validate_kalman_smoother_singular_covariance` is the guard for a singular `P_{t+1|t}`: a deterministic
slope (`Q = diag(0.3, 0)`, `a_0 = (0, -0.1)`, `P_0 = diag(10, 0)`, no intercepts), whose smoothed slope stays
at `-0.1`. `test_validate_kalman_likelihood_terms_match_filter_and_differences` compares
`kalman_likelihood_terms` with the filter and with central differences of its own objective.

The tests without a reference: `test_kalman_local_level_tracks_signal_recovers_dgp`,
`test_kalman_smoother_variance_not_above_filter`, `test_kalman_rejects_bad_intercepts`,
`test_kalman_with_intercepts_checks_them`, `test_kalman_filter_rejects_nonpositive_innovation_variance`,
`test_kalman_likelihood_terms_skip_missing_values_and_check_their_input`.

## References

- Durbin, J. and Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.
- Harvey, A. C. (1990). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
