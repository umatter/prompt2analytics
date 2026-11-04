# Validation: Probit Regression

R reference: `validation/r/cases/discrete_mlogit_asc.R` (T3-WP18).

## Method Overview

Probit regression models binary outcomes using the standard normal CDF. It estimates the probability that Y=1 given X via maximum likelihood.

**Model**:
```
P(Y=1|X) = Φ(Xβ)
```

where Φ(·) is the standard normal CDF.

**Estimation**: Newton-Raphson iteration for MLE.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `glm(family=binomial(link="probit"))` | 4.3.x |
| statsmodels | Python | `Probit()` | 0.14.x |

## Test Cases

### Test 1: Simple Probit Regression

**Data Generating Process**:
```
P(Y=1|X) = Φ(−0.5 + 1.2×X)
```

**R Code**:
```r
set.seed(42)
n <- 500

x <- rnorm(n)
latent <- -0.5 + 1.2*x
prob <- pnorm(latent)
y <- rbinom(n, 1, prob)

data <- data.frame(y = y, x = x)

probit_fit <- glm(y ~ x, data = data, family = binomial(link = "probit"))
summary(probit_fit)

# Expected: coefficients close to (-0.5, 1.2)
```

**Results Comparison**:

| Parameter | True Value | R Estimate | p2a Rust | Tolerance |
|-----------|------------|------------|----------|-----------|
| β₀ (Intercept) | -0.5 | ~-0.5 | ~-0.5 | 0.2 |
| β₁ (x) | 1.2 | ~1.2 | ~1.2 | 0.2 |
| Log-likelihood | - | varies | varies | 0.1 |

**Rust Test**: superseded by `econometrics::discrete::binary_choice::tests::test_validate_probit_binary_glm` (R-pinned on `disc_binary_glm`; section rewritten with the validation docs)

---

### Test 2: Multiple Predictors

**R Code**:
```r
set.seed(42)
n <- 1000

x1 <- rnorm(n)
x2 <- rnorm(n)
latent <- 0.3 + 0.8*x1 - 0.5*x2
prob <- pnorm(latent)
y <- rbinom(n, 1, prob)

data <- data.frame(y = y, x1 = x1, x2 = x2)

probit_fit <- glm(y ~ x1 + x2, data = data, family = binomial(link = "probit"))
summary(probit_fit)
```

**Validation Criteria**:
- All coefficients within 0.2 of true values
- Correct signs
- Significant p-values

---

### Test 3: Comparison with Logit

Probit and Logit give similar results, with coefficient ratio ≈ 1.6.

**R Code**:
```r
set.seed(42)
n <- 1000

x <- rnorm(n)
prob <- pnorm(0.5*x)  # Probit DGP
y <- rbinom(n, 1, prob)

data <- data.frame(y = y, x = x)

logit_fit <- glm(y ~ x, data = data, family = binomial(link = "logit"))
probit_fit <- glm(y ~ x, data = data, family = binomial(link = "probit"))

# Compare coefficients
coef_logit <- coef(logit_fit)["x"]
coef_probit <- coef(probit_fit)["x"]

# Ratio should be approximately 1.6 (π/√3)
ratio <- coef_logit / coef_probit
print(ratio)  # ≈ 1.6
```

**Validation Criteria**:
- Logit coefficient ≈ 1.6 × Probit coefficient
- Predicted probabilities very similar

---

### Test 4: Marginal Effects at Mean

**R Code**:
```r
set.seed(42)
n <- 500

x <- rnorm(n)
latent <- -0.2 + 0.8*x
prob <- pnorm(latent)
y <- rbinom(n, 1, prob)

data <- data.frame(y = y, x = x)
probit_fit <- glm(y ~ x, data = data, family = binomial(link = "probit"))

# Marginal effect at mean
x_mean <- mean(x)
latent_at_mean <- coef(probit_fit)[1] + coef(probit_fit)[2] * x_mean
marginal_effect <- dnorm(latent_at_mean) * coef(probit_fit)[2]

print(marginal_effect)
```

**Marginal Effect Formula**:
```
∂P/∂X = φ(Xβ) × β
```

where φ is the standard normal PDF.

---

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

## Numerical Precision Summary

| Test Case | n | Coefficient Precision | LL Precision |
|-----------|---|----------------------|--------------|
| Simple | 500 | < 0.1 | < 0.01 |
| Multiple | 1000 | < 0.1 | < 0.01 |

## Known Differences

1. **Normal CDF computation**: Numerical approximation may vary.
2. **Starting values**: Different initialization may affect convergence.
3. **Gradient tolerance**: Convergence criteria may differ slightly.

## Running the Tests

```bash
# Run Probit validation tests
cargo test -p p2a-core -- probit

# Compare Logit and Probit
cargo test -p p2a-core -- discrete::tests --nocapture
```

## References

- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed. MIT Press. Chapter 15.
- Greene, W.H. (2018). *Econometric Analysis*, 8th ed. Pearson. Chapter 17.
