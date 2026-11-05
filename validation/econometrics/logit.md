# Validation: Logistic Regression (Logit)

R reference: `validation/r/cases/discrete_mlogit_asc.R` (T3-WP18).

## Method Overview

Logistic regression models binary outcomes using the logistic (sigmoid) function. It estimates the probability that Y=1 given X via maximum likelihood.

**Model**:
```
P(Y=1|X) = exp(Xβ) / (1 + exp(Xβ)) = Λ(Xβ)
```

where Λ(·) is the logistic CDF.

**Estimation**: IRLS (Fisher scoring) as R's `glm.fit`, through the shared
`econometrics::glm` (`fit_glm`); for the logit link Fisher scoring is Newton's method.

## Reference Implementations

| Package | Language | Function |
|---------|----------|----------|
| stats | R | `glm(family = binomial("logit"))`, `summary.glm` |

## R reference: `disc_binary_glm`

Case `disc_binary_glm` in [`validation/r/cases/discrete_mlogit_asc.R`](../r/cases/discrete_mlogit_asc.R)
(registry `validation/methods/discrete_t3.json`, id `discrete_logit_probit_glm`), dataset
`validation/datasets/generated/binary_glm.csv` (n = 400, seeded):

```r
pc    <- glm.control(epsilon = 1e-10, maxit = 100)   # p2a's control (MleSettings::default())
tight <- glm.control(epsilon = 1e-14, maxit = 100)   # the MLE
fit   <- glm(y ~ x1 + x2 + d, binomial, g, control = pc)          # logit_*
ft    <- glm(y ~ x1 + x2 + d, binomial, g, control = tight)       # logit_*_tight
fa    <- glm(y ~ x1 + x1b + x2 + d, binomial, g, control = pc)    # x1b = 2 x1: NA (aliased_*)
ftr   <- glm(y ~ x1 + x2 + d + nd, binomial, g, control = pc)     # nd = 1 - d: dummy trap (trap_*)
fs    <- glm(ysep ~ x1 + x2, binomial, g, control = pc)           # separated by x1 + x2 (sep_*)
fnc   <- glm(y ~ x1 + xnc + x2, binomial, g, control = tight)     # near-collinear pair (nc_*)
```

### 1. Pins

| Key | (Intercept) | x1 | x2 | d |
|---|---|---|---|---|
| `logit_coef` | -0.5117536827 | 0.7787234865 | -0.7638229575 | 0.8426756885 |
| `logit_se` | 0.1470577482 | 0.1334714134 | 0.1248839467 | 0.2316119985 |
| `logit_z` | -3.4799504883 | 5.8343840590 | -6.1162621585 | 3.6383075745 |

`logit_loglik` -230.2221418993, `logit_aic` 468.4442837986, `logit_null_loglik`
-276.6535668335.

`econometrics::discrete::binary_choice::tests::test_validate_logit_binary_glm` pins, at the
MLE (the tight glm refit; at epsilon 1e-10 the logit is about 1e-13 from it): coefficients
`Tol::coef(400)` (1e-8 relative), SEs, z, logLik, AIC and the null logLik 1e-6, p 1e-6 +
1e-12, and the tight coefficients 1e-8.

### 2. Aliased regressors

`x1b = 2 x1` and the dummy trap `nd = 1 - d`: R reports the later column as `NA` and summary.glm
drops its row. p2a's coefficient, SE, z and p are `NaN`, the name is in
`DiscreteResult.aliased`, the warning is R's "Coefficients not defined because of
singularities: x1b", and the other coefficients, SEs and the AIC (which counts the rank) are
the full-rank fit's: `econometrics::discrete::binary_choice::tests::test_validate_logit_aliased_is_na`
(`aliased_*`, `trap_*`, rank 4).

### 3. Separation

- `ysep = 1[x1 + x2 > 0.5]` is separated by x1 + x2 and by neither regressor alone: no MLE.
  glm warns "fitted probabilities numerically 0 or 1 occurred". p2a sets
  `DiscreteResult.separation`, repeats glm's warning and adds a sentence that counts the
  extreme fitted values; most of them reach glm's clamp before its criterion is met (R: 397 of
  400 at p2a's control). A fit stopped by `max_iter` reports "glm.fit: algorithm did not
  converge" and `Converged: no`.
  `econometrics::discrete::binary_choice::tests::test_validate_logit_multivariate_separation_warns`.
- A single regressor that separates the outcome completely is refused (`PerfectSeparation`);
  quasi-complete separation by one regressor is a warning. `yq` is quasi-separated by the
  dummy `d` (R warns nothing at its default); p2a warns, and the identified coefficients
  approach the fit without the `d = 1` rows (`quasi_limit_*`, coefficients 1e-6, SEs 1e-5):
  `econometrics::discrete::binary_choice::tests::test_validate_logit_quasi_separated_dummy_warns`.

### 4. Near-collinearity and convergence (ECON-60)

`xnc = x1 + 1e-7 noise` is not aliased at glm's QR tolerance. The former Newton loop with a line
search stopped at the restricted log-likelihood -237.0063665657; glm reaches -236.984530672235.
`econometrics::discrete::binary_choice::tests::test_validate_logit_near_collinear_reaches_optimum`
pins `nc_loglik_tight` and the coefficients at 1e-6.

### 5. Outcome domain

R's glm refuses y outside [0, 1] (`y012`). p2a's binary models need a 0/1 outcome and refuse
any other value with `InvalidSpecification` naming the column and the count; R also fits
proportions in (0, 1) with a warning, which p2a's binary models refuse:
`econometrics::discrete::binary_choice::tests::test_validate_logit_rejects_y_outside_0_1`.

### 6. Controls

`MleSettings` is glm.control: `max_iter` 100 and `tolerance` 1e-10, against R's defaults (25,
1e-8). Decided (user, 2026-10-30): at R's default the probit stops about 1.6e-6 from the MLE;
R's values remain settable. Registry deviation `ECON-60`, quantity "default control". The line
search fields are gone (glm.fit has none).

Not R-pinned: the iteration counts (`logit_iter` 4, `sep_iter` 63, `sep_default_iter` 25,
`quasi_iter` 23) and the separated and quasi-separated fits' properties; the controls'
validation (`econometrics::discrete::binary_choice::tests::test_mle_settings_reject_invalid_controls`).

## Missing values

Rows with a missing outcome or regressor are dropped, as `glm`'s default
`na.action = na.omit` does: y and X come from one request with joint listwise deletion,
and the drop is reported in `DiscreteResult.missing` (and as the last line of the Display,
`Missing values: 2 of 300 rows dropped because of missing values (choice_na: 2)`). The
outcome must be numeric 0/1; Int64 and Boolean columns are read, a text outcome is
`NonNumericColumn` (it was cast to nulls and read as all 0). `marginal_effects_discrete`
builds X from the same rows and refuses a dataset whose complete rows differ from the
model's `n_obs`.

R reference: case `disc_choice_na` in [`validation/r/cases/discrete_mixed.R`](../r/cases/discrete_mixed.R)
(registry `validation/methods/discrete_mixed.json`, id `binary_multinomial_missing_values`),
300 rows; `choice_na` has 2 missing values, `x_na` 1.

```r
glm(choice_na ~ x + w,    binomial("logit"),  d, control = glm.control(epsilon = 1e-14))  # nobs 298
glm(choice_na ~ x_na + w, binomial("logit"),  d, control = glm.control(epsilon = 1e-14))  # nobs 297
glm(choice_na ~ x_na + w, binomial("probit"), d, control = glm.control(epsilon = 1e-14))  # nobs 297
```

| Key | (Intercept) | x / x_na | w | log-likelihood | nobs |
|---|---|---|---|---|---|
| `logit_y_coef` | -0.7632037025188213 | 0.3263142248068704 | -0.41797529908201525 | -181.78835192726527 | 298 |
| `logit_y_se` | 0.12932766842678328 | 0.13538466078787076 | 0.1298067241281601 | | |
| `logit_coef` | -0.7800254609952163 | 0.33128854168078986 | -0.44345203659331467 | -180.10204540326504 | 297 |
| `logit_se` | 0.13033629774151329 | 0.13609335254316823 | 0.13141778024322312 | | |

R-pinned tests (coefficients 1e-6, SEs 1e-5 relative, log-likelihood 1e-6, nobs exact):

- `econometrics::discrete::binary_choice::tests::test_validate_logit_na_outcome_dropped`:
  a missing outcome only (298 rows); the missing-value summary and the note.
- `econometrics::discrete::binary_choice::tests::test_validate_logit_na_rows_dropped`:
  outcome and regressor missing (297 rows).
- `p2a-cli` `tier2::wp18_discrete::test_validate_cli_logit_missing_note`: the CLI table at
  print precision and the `Note:` line.

Not R-pinned: `econometrics::discrete::binary_choice::tests::test_na_logit_text_outcome_refused`,
`econometrics::discrete::binary_choice::tests::test_na_logit_absent_column` and
`regression::marginal_effects::tests::test_na_marginal_effects_logit_uses_estimation_rows`
(the marginal effects equal those on the complete rows). The probit pins are in
[probit.md](probit.md) (registry id `probit_missing_values`).

### Multinomial logit

`run_multinom` reads the outcome as a factor in R `factor()` order (numbers numerically, text
by byte) and drops rows with a missing outcome or regressor, as `nnet::multinom`'s `na.omit`.
On `disc_choice_na`: `multinom_n` 298, `multinom_counts` [77, 149, 72]; the integer outcome
`k` has `k_levels` [2, 10, 33] (2 < 10 < 33, not text order). Tests:
`econometrics::discrete::multinomial_logit::tests::test_validate_multinom_na_outcome_dropped` and
`econometrics::discrete::multinomial_logit::tests::test_validate_multinom_integer_outcome_level_order`.

## Interpretation

### Coefficients
- β represents change in log-odds for one-unit increase in X
- Not directly interpretable as probability change

### Marginal Effects
```
∂P/∂X = Λ(Xβ) × (1 - Λ(Xβ)) × β
```

At mean of X: P × (1-P) × β

### Odds Ratio
```
OR = exp(β)
```

One-unit increase in X multiplies odds by OR.

## Model Fit Statistics

| Statistic | Formula | Interpretation |
|-----------|---------|----------------|
| Log-likelihood | Σ[yᵢlog(p̂ᵢ) + (1-yᵢ)log(1-p̂ᵢ)] | Higher is better |
| McFadden R² | 1 - LL/LL₀ | 0.2-0.4 is good |
| AIC | -2LL + 2k | Lower is better |
| BIC | -2LL + k×log(n) | Lower is better |

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
| unified `Logit` | 100 / 1,000 / 10,000 | 0.69 / 0.54 / 0.63 |
| criterion `Logit` | 100 / 500 / 1,000 | 0.83 / 0.69 / 0.75 |

## Running the Tests

```bash
# Run Logit validation tests
cargo test -p p2a-core -- logit

# The R pins of this page
cargo test -p p2a-core --lib --no-default-features --features ci -- binary_choice::tests::test_validate
```

## References

- McFadden, D. (1974). "Conditional Logit Analysis of Qualitative Choice Behavior". In *Frontiers in Econometrics*.
- Train, K.E. (2009). *Discrete Choice Methods with Simulation*, 2nd ed. Cambridge University Press.
