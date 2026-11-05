# Validation: Probit Regression

R reference: `validation/r/cases/discrete_mlogit_asc.R` (T3-WP18).

## Method Overview

Probit regression models binary outcomes using the standard normal CDF. It estimates the probability that Y=1 given X via maximum likelihood.

**Model**:
```
P(Y=1|X) = Φ(Xβ)
```

where Φ(·) is the standard normal CDF.

**Estimation**: IRLS (Fisher scoring) as R's `glm.fit`, through the shared
`econometrics::glm` (`fit_glm`). Standard errors come from the expected information at the
estimate, as `summary.glm`.

## Reference Implementations

| Package | Language | Function |
|---------|----------|----------|
| stats | R | `glm(family = binomial("probit"))`, `summary.glm` |

## R reference: `disc_binary_glm`

Case `disc_binary_glm` in [`validation/r/cases/discrete_mlogit_asc.R`](../r/cases/discrete_mlogit_asc.R)
(registry `validation/methods/discrete_t3.json`, id `discrete_logit_probit_glm`), the data of
[logit.md](logit.md) (n = 400):

```r
pt <- glm(y ~ x1 + x2 + d, binomial("probit"), g, control = glm.control(epsilon = 1e-14, maxit = 100))
```

| Key | (Intercept) | x1 | x2 | d |
|---|---|---|---|---|
| `probit_coef_tight` | -0.3052715196 | 0.4741663935 | -0.4531388146 | 0.5044827737 |
| `probit_se_tight` | 0.0877577883 | 0.0775209775 | 0.0721602549 | 0.1385658648 |
| `probit_z_tight` | -3.4785689736 | 6.1166204172 | -6.2796177094 | 3.6407435160 |

`probit_loglik_tight` -230.1802488494, `probit_aic_tight` 468.3604976987.

`econometrics::discrete::binary_choice::tests::test_validate_probit_binary_glm` fits at
`tolerance = 1e-14` and pins, at the MLE: coefficients `Tol::coef(400)` (1e-8), SEs, z, logLik
and AIC 1e-6, p 1e-6 + 1e-12. The default control (1e-10) is checked against the same MLE at
1e-6: Fisher scoring converges linearly for the probit link, so glm's rule at 1e-10 stops about
1.4e-7 from the MLE (R too, `probit_coef`).

Aliasing, separation, the outcome domain and the controls are shared with the logit
([logit.md](logit.md), sections 2-6): aliased columns are `NaN` with R's message, y outside
{0, 1} is refused, and `MleSettings` defaults to 100 iterations and 1e-10 (R: 25 and 1e-8;
decided 2026-10-30, registry deviation `ECON-60`). Not R-pinned: the iteration count
(`probit_iter` 5).

## Logit vs Probit Comparison

| Aspect | Logit | Probit |
|--------|-------|--------|
| Link function | Logistic CDF | Normal CDF |
| Tail behavior | Heavier tails | Lighter tails |
| Coefficient scale | ≈ 1.6× Probit | ≈ 0.625× Logit |
| Interpretation | Odds ratios | Marginal effects |
| Computation | Slightly faster | Requires Φ evaluation |

In practice, predicted probabilities are nearly identical.

## Missing values

As for the logit ([logit.md](logit.md)): rows with a missing outcome or regressor are
dropped (`glm`'s `na.omit`) and reported in `missing`; a text outcome is
`NonNumericColumn`. Case `disc_choice_na` in
[`validation/r/cases/discrete_mixed.R`](../r/cases/discrete_mixed.R):

```r
glm(choice_na ~ x_na + w, binomial("probit"), d, control = glm.control(epsilon = 1e-14))  # nobs 297
```

| Key | (Intercept) | x_na | w | log-likelihood | nobs |
|---|---|---|---|---|---|
| `probit_coef` | -0.4761633999415939 | 0.1960342515712869 | -0.2664571109637172 | -180.21133533224827 | 297 |
| `probit_se` | 0.07782488226903958 | 0.0813932906245501 | 0.07869008940022174 | | |

R-pinned: `econometrics::discrete::binary_choice::tests::test_validate_probit_na_rows_dropped`
(coefficients 1e-6, SEs 1e-5 relative, log-likelihood 1e-6, nobs exact; registry
`validation/methods/discrete_mixed.json`, id `probit_missing_values`).

## Timing

Each IRLS step of the shared GLM fit
(`econometrics::glm`) takes OLS's solve decision: the normal equations of the weighted design
when it is of full rank with reciprocal condition at least 1e-8 (refined once below 1e-2),
the pivoted QR otherwise, which still decides rank and aliasing; both paths agree to 1e-10
relative (`econometrics::glm::tests::test_glm_normal_equations_match_qr`), and a nearly collinear design still
reaches the QR (`econometrics::glm::tests::test_glm_near_collinear_design_takes_qr`).

Release build, one core (`taskset`, `RAYON_NUM_THREADS=1`), three interleaved rounds against
`fe13b889` (before these changes); new / base as the ratio of medians (`unified_benchmarks`,
criterion `econometrics_benchmarks`).

| Bench | n | new / base |
|---|---|---|
| unified `Probit` | 100 / 1,000 / 10,000 | 0.53 / 0.47 / 0.60 |
| criterion `Probit` | 100 / 500 / 1,000 | 0.79 / 0.65 / 0.63 |

## Running the Tests

```bash
# Run Probit validation tests
cargo test -p p2a-core -- probit

# The R pins of the logit and probit
cargo test -p p2a-core --lib --no-default-features --features ci -- binary_choice::tests::test_validate
```

## References

- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed. MIT Press. Chapter 15.
- Greene, W.H. (2018). *Econometric Analysis*, 8th ed. Pearson. Chapter 17.
