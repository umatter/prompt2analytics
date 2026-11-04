# Validation: Logistic Regression (Logit)

R reference: `validation/r/cases/discrete_mlogit_asc.R` (T3-WP18).

## Method Overview

Logistic regression models binary outcomes using the logistic (sigmoid) function. It estimates the probability that Y=1 given X via maximum likelihood.

**Model**:
```
P(Y=1|X) = exp(Xβ) / (1 + exp(Xβ)) = Λ(Xβ)
```

where Λ(·) is the logistic CDF.

**Estimation**: Newton-Raphson iteration for MLE.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `glm(family=binomial)` | 4.3.x |
| statsmodels | Python | `Logit()` | 0.14.x |

## Test Cases

### Test 1: Simple Logistic Regression

**Data Generating Process**:
```
P(Y=1|X) = Λ(−1 + 2×X)
```

**R Code**:
```r
set.seed(42)
n <- 500

x <- rnorm(n)
latent <- -1 + 2*x
prob <- 1 / (1 + exp(-latent))
y <- rbinom(n, 1, prob)

data <- data.frame(y = y, x = x)

logit_fit <- glm(y ~ x, data = data, family = binomial(link = "logit"))
summary(logit_fit)

# Expected output:
#             Estimate Std. Error z value Pr(>|z|)
# (Intercept)  -1.xx     0.xx      -x.xx   < 0.05
# x             2.xx     0.xx       x.xx   < 0.001
```

**Results Comparison**:

| Parameter | True Value | R Estimate | p2a Rust | Tolerance |
|-----------|------------|------------|----------|-----------|
| β₀ (Intercept) | -1.0 | ~-1.0 | ~-1.0 | 0.3 |
| β₁ (x) | 2.0 | ~2.0 | ~2.0 | 0.3 |
| Log-likelihood | - | varies | varies | 0.1 |

**Rust Test**: superseded by `econometrics::discrete::binary_choice::tests::test_validate_logit_binary_glm` (R-pinned on `disc_binary_glm`; section rewritten with the validation docs)

---

### Test 2: Multiple Predictors

**R Code**:
```r
set.seed(42)
n <- 1000

x1 <- rnorm(n)
x2 <- rnorm(n)
latent <- 0.5 + 1.5*x1 - 0.8*x2
prob <- 1 / (1 + exp(-latent))
y <- rbinom(n, 1, prob)

data <- data.frame(y = y, x1 = x1, x2 = x2)

logit_fit <- glm(y ~ x1 + x2, data = data, family = binomial)
summary(logit_fit)

# Coefficients should be close to (0.5, 1.5, -0.8)
```

**Validation Criteria**:
- All coefficients within 0.3 of true values
- Correct signs
- Significant p-values

---

### Test 3: Perfect Separation (Edge Case)

When a predictor perfectly separates outcomes, MLE doesn't exist.

**R Code**:
```r
# Perfect separation: x perfectly predicts y
data <- data.frame(
  y = c(rep(0, 10), rep(1, 10)),
  x = c(rep(-1, 10), rep(1, 10))
)

logit_fit <- glm(y ~ x, data = data, family = binomial)
# Warning: "fitted probabilities numerically 0 or 1 occurred"
```

**Validation Criteria**:
- Warning or error about separation
- Very large coefficient estimates (diverging)
- Very large standard errors

---

### Test 4: Odds Ratios

**R Code**:
```r
set.seed(42)
n <- 500

x <- rnorm(n)
latent <- -0.5 + 1*x
prob <- 1 / (1 + exp(-latent))
y <- rbinom(n, 1, prob)

data <- data.frame(y = y, x = x)
logit_fit <- glm(y ~ x, data = data, family = binomial)

# Odds ratios
exp(coef(logit_fit))
# OR for x should be ≈ exp(1) ≈ 2.72
```

**Validation Criteria**:
- exp(β) gives correct odds ratio
- One-unit increase in x multiplies odds by exp(β)

---

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

## Numerical Precision Summary

| Test Case | n | Coefficient Precision | LL Precision |
|-----------|---|----------------------|--------------|
| Simple | 500 | < 0.1 | < 0.01 |
| Multiple | 1000 | < 0.1 | < 0.01 |

## Known Differences

1. **Starting values**: Different initial values may affect convergence path.
2. **Convergence criteria**: Tolerance for gradient/likelihood may differ.
3. **Marginal effects**: R requires additional package (margins); p2a computes directly.

## Running the Tests

```bash
# Run Logit validation tests
cargo test -p p2a-core -- logit

# Run with output
cargo test -p p2a-core -- discrete::tests::test_logit --nocapture
```

## References

- McFadden, D. (1974). "Conditional Logit Analysis of Qualitative Choice Behavior". In *Frontiers in Econometrics*.
- Train, K.E. (2009). *Discrete Choice Methods with Simulation*, 2nd ed. Cambridge University Press.
