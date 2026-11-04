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
rows and `m = 1 + K p` columns. `Z` is the same for the K equations, so one pivoted QR
decomposition (`linalg::solve::Qr`, the `dqrdc2` algorithm behind `lm.fit`, tolerance 1e-7)
serves them all, as `lm` does inside `vars::VAR(y, p, type = "const")`.

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
  runs for K = 3, p = 3), `test_var_rejects_collinear_series` (`b = 2 a`), and the DGP and
  shape tests `test_var_recovers_dgp`, `test_var_irf_shapes_and_decay`.

Tolerances (validation/README.md, n = 82 < 100): coefficients 1e-6 relative; SEs, t, Sigma,
logLik and criteria 1e-5 relative; p 1e-5 relative / 1e-12 absolute; counts exact.

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::timeseries::var
```

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
