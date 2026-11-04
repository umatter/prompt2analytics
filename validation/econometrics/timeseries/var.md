# Validation: Vector Autoregression (VAR)

Registry: [`validation/methods/ts_var.json`](../../methods/ts_var.json) (method `ts_var`).
R case: `var_canada` in [`validation/r/cases/ts_var.R`](../../r/cases/ts_var.R).
Impulse responses: [irf.md](irf.md). Granger causality: [granger.md](granger.md).

## Method

`run_var(dataset, columns, lags)` fits a VAR(p) with a constant,

```
y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t,
```

equation by equation on the common design `Z = [1, y_{t-1}', .., y_{t-p}']` with `T = n - p`
rows and `m = 1 + K p` columns, as `lm` does inside `vars::VAR(y, p, type = "const")`. `Z` is
the same for the K equations, so one solve serves them all: OLS's normal equations on a
full-rank, well-conditioned `Z`, otherwise one pivoted QR decomposition (`linalg::solve::Qr`,
the `dqrdc2` algorithm behind `lm.fit`, tolerance 1e-7), which decides the rank (see "Solve and
timing").

**Coefficient layout.** `coefficients[eq]` is `[const, y1.l1, .., yK.l1, .., y1.lp, .., yK.lp]`;
`coef_names` carries vars' names in that order (vars itself prints `const` last).

**Reported quantities.** With `U` the T x K residual matrix:

| Field | Formula | R |
|---|---|---|
| `sigma_u` | `U'U / (T - m)` | `summary(m)$covres` |
| `std_errors[eq][j]` | `sqrt(sigma_u[eq][eq] * ((Z'Z)^-1)_jj)` | `coef(summary(m$varresult[[eq]]))[, 2]` |
| `t_values`, `p_values` | `coef / se`; two-sided on `t(T - m)` | columns 3 and 4 |
| `df_residual` | `T - m` | `df.residual(m$varresult[[1]])` |
| `sigma_u_ml` | `U'U / T` | `crossprod(resid(m)) / m$obs` |
| `log_det_sigma_ml` | `ln det sigma_u_ml`, by its Cholesky factor | `log(det(..))` |
| `log_likelihood` | `-T/2 (K ln 2 pi + ln det sigma_u_ml + K)` | `logLik(m)` (`logLik.varest`) |
| `aic`, `bic` | `-2 logLik + 2 K m`, `-2 logLik + ln(T) K m` | `AIC(m)`, `BIC(m)` |

AIC and BIC of fits with different p use different samples (T = n - p); compare lag orders
on a common sample, as `vars::VARselect` does. p2a has no lag-selection routine.

## Requirements and errors

| Condition | Error | What `vars` does |
|---|---|---|
| `lags == 0` | `InvalidSpecification` | |
| `n <= p + 1` | `InsufficientTimePeriods` | |
| `T - m < K`, i.e. `n < (K + 1)(p + 1)` | `InsufficientData { required: (K + 1)(p + 1) }`, the context names `1 + K p` | NA coefficients and NaN SEs, `logLik()` stops ("system is exactly singular"); at `T = m` the residual SE is NaN |
| an aliased lag (a series proportional to another, or constant) | `SingularMatrix`, the context names the lag (`b.l1`) | the coefficient is NA |
| singular `sigma_u_ml` | `SingularMatrix` | |

With `T - m < K` the K residual columns lie in a space of dimension `T - m < K`, so the
residual covariance is singular and neither the likelihood nor the orthogonalised impulse
responses exist. Integer and Boolean columns are accepted, missing values and text are
refused by name, as for Granger (see [../granger.md](../granger.md) for the extractor rule).

## Reference case

`var_canada`: `vars::VAR(Canada, p = 2, type = "const")` on
`validation/datasets/generated/var_canada.csv` (`vars::Canada`, 84 rows; T = 82, K = 4,
p = 2, m = 9, residual df 73).

| Quantity | R value |
|---|---|
| equation e, coefficients (e.l1 .. U.l2, const) | 1.6378206022871895, 0.16727166854704162, -0.06311863134491014, 0.2655847772119808, -0.4971337747484717, -0.10165006721151987, 0.0038444920542162852, 0.13268931262949596, -136.99844936947017 |
| equation e, SEs | 0.15000904816881805, 0.061137825267054136, 0.055238725134005684, 0.20279708449069428, 0.1595260357375151, 0.06606917678329811, 0.05552228231226825, 0.20732747449850167, 55.848073200020195 |
| e.l1: t, p | 10.918145420428303, 5.2825877991282224e-17 |
| const: t, p | -2.45305596987759, 0.01655354150954215 |
| `sigma_u[1,1]` (divisor 73), `sigma_u_ml[1,1]` (divisor 82) | 0.13163473833393352, 0.11718702315094082 |
| ln det Sigma_ML | -7.063250506198347 |
| logLik, AIC, BIC | -175.8185681370004, 423.6371362740008, 510.2790291755139 |

## Tests and tolerances

- `test_validate_var_canada_coefficients_se`: coefficients, SEs, t and p of all four
  equations by name, `sigma_u`, the counts.
- `test_validate_var_canada_loglik_ic`: logLik, AIC, BIC, ln det Sigma_ML, `sigma_u_ml`.
- `test_validate_var_display_prints_coefficients`: the text `ts_var` returns contains every
  equation's coefficient table (estimate and SE at their 6 printed decimals, t at 3).
- Not R-pinned (behaviour tests): `test_var_rejects_saturated` (n = 10, 13, 15 refused, 16
  runs for K = 3, p = 3), `test_var_rejects_collinear_series` (`b = 2 a`),
  `test_var_rejects_nearly_collinear_series`, `test_var_normal_equations_match_pivoted_qr`
  (see "Solve and timing"), and the DGP and shape tests `test_var_recovers_dgp`,
  `test_var_irf_shapes_and_decay`.

Tolerances (validation/README.md, n = 82 < 100): coefficients 1e-6 relative; SEs, t, Sigma,
logLik and criteria 1e-5 relative; p 1e-5 relative / 1e-12 absolute; counts exact.

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::timeseries::var
```

## Solve and timing

`run_var` and `granger_test` take OLS's solve decision (`regression::ols::normal_equations_inverse`):
the normal equations, refined once when the equilibrated reciprocal condition is below 1e-2, when
the design is of full rank and that condition is at least 1e-8; otherwise `lm.fit`'s pivoted QR,
which decides the rank and names the aliased lags. Every VAR equation shares the design, so one
inverse and two matrix products (four when refined) give all K coefficient vectors. Granger's
gain RSS_R - RSS_U is `||X_U b_U - X_R b_R||^2`, a sum of squares (never negative).
`test_var_normal_equations_match_pivoted_qr` and `test_granger_normal_equations_match_pivoted_qr`
hold the two solves to 1e-10 relative on noisy simulated designs (K up to 10, p up to 6);
`test_var_rejects_nearly_collinear_series` and `test_granger_rejects_nearly_proportional_series`
check that a series proportional to another up to 1e-12 noise is still refused, naming the lags.

Timing (release, one P core, `RAYON_NUM_THREADS=1`, medians of 6 interleaved runs, each a median
of 21 samples; base = fe13b889, before the Tier 3 VAR and Granger changes; QR = the pivoted QR
for every fit, 3070299f; now = the solve above). Simulated stationary VAR(1) with Gaussian
noise; Granger on its first two series:

| Fit | n | base µs | QR µs (ratio) | now µs (ratio) |
|---|---|---|---|---|
| VAR K = 2, p = 1 | 100 | 5.51 | 7.82 (1.42) | 7.50 (**1.36**) |
| | 1,000 | 35.5 | 41.9 (1.18) | 35.9 (1.01) |
| | 10,000 | 420 | 479 (1.14) | 372 (0.89) |
| VAR K = 3, p = 2 | 100 | 9.19 | 15.5 (1.68) | 12.9 (**1.40**) |
| | 1,000 | 64.6 | 102 (1.58) | 68.3 (1.06) |
| | 10,000 | 821 | 1,024 (1.25) | 661 (0.81) |
| VAR K = 10, p = 12 | 1,000 | 2,016 | 5,428 (2.69) | 1,820 (0.90) |
| | 10,000 | 28,815 | 64,628 (2.24) | 17,187 (0.60) |
| Granger, 2 lags | 100 | 7.44 | 5.36 (0.72) | 6.93 (0.93) |
| | 500 | 22.1 | 21.5 (0.97) | 21.2 (0.96) |
| | 1,000 | 40.1 | 41.3 (1.03) | 39.1 (0.98) |
| | 5,000 | 245 | 256 (1.04) | 223 (0.91) |
| Granger, 4 lags | 100 | 9.94 | 8.14 (0.82) | 9.76 (0.98) |
| | 1,000 | 55.9 | 75.1 (1.34) | 53.9 (0.97) |
| | 10,000 | 779 | 975 (1.25) | 721 (0.93) |

The results of the two solves agree to 1e-14 relative on these fits. At n = 100 the VAR is
still 1.36x (K = 2, p = 1) and 1.40x (K = 3, p = 2) of base. That is the cost of the t and p
values it now reports and base did not compute: one `t_test_p_value` per coefficient,
1.31 µs for 6 and 4.17 µs for 21 (same in both builds). Without that cost the fits take 6.19 µs
(1.12x) and 8.73 µs (0.95x). This slowdown at small n was accepted by the maintainer
(2026-11-04).

## History

Before Tier 3 (review finding ids in parentheses):
- the log-likelihood used the product of the absolute values of all K^2 entries of Sigma_u,
  each floored at 1e-10, instead of its determinant (ECON-50): on this case logLik was
  1586.1325945384274, AIC -3100.2651890768548 and BIC -3013.6232961753417;
- `sigma_u` had the divisor T, so SEs and orthogonalised responses were too small by
  `sqrt((T - m) / T)` (ECON-90; se(e.l1) 0.14153764305264271). The JSON field `sigma_u` of
  `p2a ts var` now has the divisor T - Kp - 1; the old matrix is `sigma_u_ml`;
- saturated and collinear designs were fitted through a pseudo-inverse and returned as a
  perfect fit (ECON-79; AIC -1331 for any 10-row VAR(3) of three series);
- the Display printed no coefficient (ECON-49).

## Known differences

- `type = "trend" | "both" | "none"`, exogenous variables and seasonal dummies are not
  implemented: there is always one constant.
- There is no lag-selection routine over a range of orders (`VARselect`).
- Aliased lags and samples with fewer than K residual degrees of freedom are errors; vars
  returns NA coefficients, NaN SEs or a degenerate logLik there.
