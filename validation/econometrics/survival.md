# Validation: Survival Analysis

## Method Overview

Survival analysis methods handle time-to-event data with censoring. The module implements:
- **Kaplan-Meier**: Non-parametric survival curve estimation
- **Log-Rank Test**: Comparing survival between groups
- **Cox Proportional Hazards**: Semi-parametric regression
- **AFT Models**: Parametric survival regression (Weibull, Log-Normal, etc.)
- **Competing Risks**: Aalen-Johansen cumulative incidence functions

## Reference Implementations

| Package | Language | Functions | Version Tested |
|---------|----------|-----------|----------------|
| survival | R | `survfit()`, `coxph()`, `survreg()` | 3.5+ |
| cmprsk | R | `cuminc()` | 2.2+ |

---

## Test Case 1: Kaplan-Meier Estimator

### R Code
```r
library(survival)

# Simple dataset with censoring
time <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
event <- c(1, 1, 0, 1, 1, 0, 1, 0, 1, 1)  # 1=event, 0=censored

surv_obj <- Surv(time, event)
km_fit <- survfit(surv_obj ~ 1, conf.type = "log-log")

# Extract results
summary(km_fit)
```

### Expected Results

| Time | N at Risk | Events | Survival | SE | 95% CI Lower | 95% CI Upper |
|------|-----------|--------|----------|-------|--------------|--------------|
| 1 | 10 | 1 | 0.900 | 0.0949 | 0.6827 | 0.9698 |
| 2 | 9 | 1 | 0.800 | 0.1265 | 0.5384 | 0.9184 |
| 4 | 7 | 1 | 0.686 | 0.1533 | 0.4005 | 0.8540 |
| 5 | 6 | 1 | 0.571 | 0.1664 | 0.2920 | 0.7785 |
| 7 | 4 | 1 | 0.429 | 0.1813 | 0.1634 | 0.6880 |
| 9 | 2 | 1 | 0.214 | 0.1712 | 0.0308 | 0.5765 |
| 10 | 1 | 1 | 0.000 | NA | NA | NA |

### Validation Criteria

| Statistic | Tolerance |
|-----------|-----------|
| Survival estimates | 0.001 |
| Standard errors | 0.01 |
| Median survival | Exact match or within 0.5 |

---

## Test Case 2: Stratified Kaplan-Meier

### R Code
```r
library(survival)

# Two-group comparison
time <- c(1, 2, 3, 5, 6, 7, 2, 3, 4, 5, 8, 9)
event <- c(1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1)
group <- c(rep("A", 6), rep("B", 6))

surv_obj <- Surv(time, event)
km_fit <- survfit(surv_obj ~ group, conf.type = "log-log")
summary(km_fit)

# Median survival by group
print(km_fit)
```

### Expected Results

**Group A:**
- Median survival: 5.5 (or NA if not reached)
- Final survival: varies

**Group B:**
- Median survival: 4.0
- Final survival: varies

---

## Test Case 3: Log-Rank Test

### R Code
```r
library(survival)

# Two groups with different survival
time <- c(1, 2, 3, 5, 6, 7, 2, 3, 4, 5, 8, 9)
event <- c(1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1)
group <- c(rep(0, 6), rep(1, 6))

surv_obj <- Surv(time, event)
logrank <- survdiff(surv_obj ~ group)
print(logrank)

# Chi-squared and p-value
logrank$chisq
1 - pchisq(logrank$chisq, df = 1)
```

### Expected Results

| Statistic | R Value | Tolerance |
|-----------|---------|-----------|
| Chi-squared | ~0.5-2.0 | 0.1 |
| df | 1 | Exact |
| p-value | >0.05 | 0.01 |

### Validation Notes
- Log-rank test assumes proportional hazards
- Chi-squared approximation may be poor with small samples

---

## Test Case 4: Cox Proportional Hazards (Basic)

### R Code
```r
library(survival)

# Simulated data with known effect
set.seed(42)
n <- 100
treatment <- rbinom(n, 1, 0.5)
age <- rnorm(n, 50, 10)

# Generate survival times (Weibull)
# True hazard ratio for treatment = 0.5 (protective)
# True hazard ratio for age = 1.02 per year
shape <- 1.5
scale <- exp(5 - 0.7*treatment + 0.02*age)
time <- rweibull(n, shape = shape, scale = scale)

# Add random censoring (~30%)
censor_time <- runif(n, 0, max(time) * 1.2)
observed_time <- pmin(time, censor_time)
event <- as.integer(time <= censor_time)

data <- data.frame(time = observed_time, event = event,
                   treatment = treatment, age = age)

# Fit Cox model
cox_fit <- coxph(Surv(time, event) ~ treatment + age, data = data)
summary(cox_fit)

# Key outputs
coef(cox_fit)           # Log hazard ratios
exp(coef(cox_fit))      # Hazard ratios
sqrt(diag(vcov(cox_fit))) # Standard errors
cox_fit$concordance     # C-index
```

### Expected Results

| Parameter | True Value | Expected Estimate | Tolerance |
|-----------|------------|-------------------|-----------|
| β(treatment) | -0.7 | ~-0.7 | 0.3 |
| β(age) | 0.02 | ~0.02 | 0.02 |
| HR(treatment) | 0.5 | ~0.5 | 0.2 |
| HR(age) | 1.02 | ~1.02 | 0.02 |
| Concordance | - | 0.6-0.8 | - |

### Validation Criteria

| Statistic | Tolerance |
|-----------|-----------|
| Coefficients | 0.1 |
| Standard errors | 0.05 |
| Hazard ratios | 0.15 |
| Concordance | 0.05 |
| Log-likelihood | 1.0 |

---

## Test Case 5: Cox PH with Breslow vs Efron Ties

### R Code
```r
library(survival)

# Data with ties
time <- c(1, 1, 2, 2, 2, 3, 4, 4, 5, 5)
event <- c(1, 1, 1, 0, 1, 1, 1, 0, 1, 1)
x <- c(0, 1, 0, 0, 1, 1, 0, 1, 0, 1)

# Breslow method (default in some software)
cox_breslow <- coxph(Surv(time, event) ~ x, ties = "breslow")
coef(cox_breslow)

# Efron method (R default, more accurate)
cox_efron <- coxph(Surv(time, event) ~ x, ties = "efron")
coef(cox_efron)

summary(cox_breslow)
summary(cox_efron)
```

### Expected Results

| Method | Coefficient | SE | p-value |
|--------|-------------|-----|---------|
| Breslow | ~0.3-0.6 | ~0.6-0.9 | >0.3 |
| Efron | ~0.3-0.6 | ~0.6-0.9 | >0.3 |

### Validation Notes
- Efron is more accurate when there are many tied event times
- Breslow is faster but can be biased with heavy ties
- Both should give similar results with few ties

---

## Test Case 6: Cox Model Tests (Wald, Score, LR)

### R Code
```r
library(survival)

set.seed(123)
n <- 200
x1 <- rnorm(n)
x2 <- rnorm(n)

# Generate survival with x1 effect only
time <- rexp(n, rate = exp(0.5 * x1))
event <- rep(1, n)  # All events for simplicity

cox_fit <- coxph(Surv(time, event) ~ x1 + x2)
summary(cox_fit)

# Extract tests
# Wald test
cox_fit$wald.test

# Score test
cox_fit$score

# Likelihood ratio test
# 2 * (logLik(cox_fit) - logLik(null_model))
```

### Expected Results

| Test | Statistic | df | p-value |
|------|-----------|-----|---------|
| Wald | ~25-35 | 2 | <0.001 |
| Score | ~25-35 | 2 | <0.001 |
| LR | ~25-35 | 2 | <0.001 |

---

## Test Case 7: AFT Weibull Model

### R Code
```r
library(survival)

set.seed(42)
n <- 200
x <- rnorm(n, 0, 1)

# True model: log(T) = 2 + 0.5*x + sigma*epsilon
# sigma = 0.5 (shape = 1/sigma = 2)
true_intercept <- 2
true_beta <- 0.5
true_scale <- 0.5

# Generate Weibull times
epsilon <- -log(runif(n))  # Standard Gumbel
log_time <- true_intercept + true_beta * x + true_scale * epsilon
time <- exp(log_time)

# Random censoring
censor_time <- runif(n, 0, quantile(time, 0.9))
observed_time <- pmin(time, censor_time)
event <- as.integer(time <= censor_time)

data <- data.frame(time = observed_time, event = event, x = x)

# Fit AFT Weibull
aft_fit <- survreg(Surv(time, event) ~ x, data = data, dist = "weibull")
summary(aft_fit)

# Note: R's survreg parameterization differs slightly
# Coefficients are for log(T) = mu + beta*x + scale*epsilon
coef(aft_fit)
aft_fit$scale
```

### Expected Results

| Parameter | True Value | Expected Estimate | Tolerance |
|-----------|------------|-------------------|-----------|
| Intercept | 2.0 | ~2.0 | 0.3 |
| β(x) | 0.5 | ~0.5 | 0.2 |
| Scale (σ) | 0.5 | ~0.5 | 0.15 |
| AIC | - | varies | - |

### Validation Criteria

| Statistic | Tolerance |
|-----------|-----------|
| Coefficients | 0.2 |
| Scale parameter | 0.1 |
| Log-likelihood | 5.0 |
| AIC | 10.0 |

---

## Test Case 8: AFT Log-Normal Model

### R Code
```r
library(survival)

set.seed(42)
n <- 200
x <- rnorm(n, 0, 1)

# True model: log(T) = 3 + 0.3*x + sigma*epsilon, epsilon ~ N(0,1)
true_intercept <- 3
true_beta <- 0.3
true_sigma <- 0.8

log_time <- true_intercept + true_beta * x + true_sigma * rnorm(n)
time <- exp(log_time)

# Random censoring
censor_time <- runif(n, 0, quantile(time, 0.85))
observed_time <- pmin(time, censor_time)
event <- as.integer(time <= censor_time)

data <- data.frame(time = observed_time, event = event, x = x)

# Fit AFT Log-Normal
aft_fit <- survreg(Surv(time, event) ~ x, data = data, dist = "lognormal")
summary(aft_fit)
```

### Expected Results

| Parameter | True Value | Expected Estimate | Tolerance |
|-----------|------------|-------------------|-----------|
| Intercept | 3.0 | ~3.0 | 0.4 |
| β(x) | 0.3 | ~0.3 | 0.2 |
| Scale (σ) | 0.8 | ~0.8 | 0.2 |

---

## Test Case 9: Competing Risks (Aalen-Johansen)

### R Code
```r
library(survival)
library(cmprsk)

# Competing risks data
# Event type: 0=censored, 1=event type 1, 2=event type 2
set.seed(42)
n <- 100

time <- c(rexp(n/2, 0.5), rexp(n/2, 0.3))
event_type <- c(
  sample(c(0, 1, 2), n/2, replace = TRUE, prob = c(0.3, 0.5, 0.2)),
  sample(c(0, 1, 2), n/2, replace = TRUE, prob = c(0.2, 0.3, 0.5))
)

# Using cmprsk package
cif <- cuminc(time, event_type)
print(cif)

# Extract CIF values at specific times
cif$`1 1`  # CIF for event type 1
cif$`1 2`  # CIF for event type 2
```

### Alternative with survival package
```r
library(survival)

# Create multi-state survival object
# status: 0=censored, 1=type1, 2=type2
time <- c(1, 2, 3, 4, 5, 6, 7, 8)
status <- c(1, 2, 0, 1, 2, 1, 0, 2)

# Fit cumulative incidence
fit <- survfit(Surv(time, factor(status)) ~ 1)
summary(fit)
```

### Expected Results

| Time | CIF (Type 1) | CIF (Type 2) | Overall Survival |
|------|--------------|--------------|------------------|
| 2.0 | ~0.15 | ~0.10 | ~0.75 |
| 5.0 | ~0.35 | ~0.25 | ~0.40 |
| 10.0 | ~0.45 | ~0.35 | ~0.20 |

### Validation Criteria

| Statistic | Tolerance |
|-----------|-----------|
| CIF estimates | 0.05 |
| Standard errors | 0.03 |
| Sum of CIFs ≤ 1 - S(t) | Must hold |

---

## Test Case 10: Lung Cancer Dataset (Classic)

### R Code
```r
library(survival)

# Use built-in lung dataset
data(lung)
head(lung)

# Kaplan-Meier
km_fit <- survfit(Surv(time, status) ~ sex, data = lung)
print(km_fit)

# Log-rank test
survdiff(Surv(time, status) ~ sex, data = lung)

# Cox model
cox_fit <- coxph(Surv(time, status) ~ age + sex + ph.karno, data = lung)
summary(cox_fit)
```

### Expected Results (from R)

**Kaplan-Meier Median Survival:**
- Male (sex=1): ~270 days
- Female (sex=2): ~426 days

**Log-Rank Test:**
- Chi-squared: ~10.3
- df: 1
- p-value: ~0.001

**Cox Model Coefficients:**

| Variable | Coefficient | SE | HR | p-value |
|----------|-------------|-----|-----|---------|
| age | ~0.017 | ~0.009 | ~1.017 | ~0.07 |
| sex | ~-0.51 | ~0.17 | ~0.60 | ~0.003 |
| ph.karno | ~-0.016 | ~0.006 | ~0.984 | ~0.01 |

---

## Implementation Notes

### Numerical Precision

1. **Newton-Raphson Convergence**:
   - Default tolerance: 1e-9
   - Maximum iterations: 25 (Cox), 100 (AFT)

2. **Matrix Operations**:
   - Use `safe_inverse` with condition number checking
   - Fall back to pseudoinverse for ill-conditioned Hessians

3. **Tie Handling**:
   - Breslow: Faster, less accurate with many ties
   - Efron: More accurate, slightly slower

### Known Differences from R

1. **Parameterization**:
   - R's `survreg` uses different parameterization for scale
   - Our implementation uses standard AFT parameterization

2. **Confidence Intervals**:
   - Kaplan-Meier: Uses log-log transformation (matches R's `conf.type="log-log"`)
   - Cox HR CIs: Uses Wald-based intervals

3. **Baseline Hazard**:
   - Currently not estimated (only relative hazards)
   - Future: Add Breslow estimator for H₀(t)

---

## Tolerance Summary

| Method | Statistic | Tolerance |
|--------|-----------|-----------|
| Kaplan-Meier | Survival | 0.001 |
| Kaplan-Meier | SE | 0.01 |
| Log-Rank | Chi-squared | 0.1 |
| Log-Rank | p-value | 0.01 |
| Cox PH | Coefficients | 0.1 |
| Cox PH | SE | 0.05 |
| Cox PH | Concordance | 0.05 |
| AFT | Coefficients | 0.2 |
| AFT | Scale | 0.15 |
| Competing Risks | CIF | 0.05 |

---

## References

1. Therneau, T.M. (2023). "A Package for Survival Analysis in R". R package version 3.5+.
2. Gray, R.J. (2022). "cmprsk: Subdistribution Analysis of Competing Risks". R package.
3. Klein, J.P. & Moeschberger, M.L. (2003). *Survival Analysis*. Springer.
