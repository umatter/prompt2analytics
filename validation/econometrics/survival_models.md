# Validation: Cox PH, AFT Models and Competing Risks (not R-pinned)

**Status: not R-pinned.** No entry in `validation/methods/*.json` compares these estimates with
committed R output. The tests in `crates/p2a-core/src/econometrics/survival.rs`
(`test_cox_ph_basic`, `test_validate_cox_ties_against_r`, `test_validate_cox_concordance`,
`test_aft_weibull`, `test_validate_aft_scale_parameter`, `test_validate_aft_distributions`,
`test_competing_risks`, `test_validate_competing_risks_cif_sum`) check properties (signs, ranges,
monotone cumulative incidence), not R's numbers. The expected values below are approximate and
are kept as a guide until R reference cases exist (Tier 3: ECON-45 robust SEs, ECON-46 Breslow
row order, X-15 Cox convergence, ECON-47/ECON-48 AFT optimiser and exponential, ECON-78
competing-risks SE, ECON-89 concordance SE).

What these methods share with Kaplan-Meier and the log-rank test is pinned in
[survival.md](survival.md), "Column Types and Missing Values": integer columns, rows with missing
values dropped and reported (`missing`), status coded as R's `Surv()` (0/1, 1/2, logical; Cox and
AFT), times >= 0 (AFT > 0), competing-risk codes 0-255. On R's `lung`, `coxph(Surv(time, status) ~
age + sex + ph.karno)` uses 227 rows with 164 events and `survreg(Surv(time, status) ~ age + sex)`
228 rows with 165 events, and p2a uses the same rows
(`test_validate_na_survival_missing_rows_are_dropped_and_reported`,
`test_validate_mcp_survival_tools_accept_integer_columns`,
`test_validate_cli_survival_cox_aft_integer_columns`); the coefficients are not compared.

## Reference Implementations

| Package | Language | Functions | Version |
|---------|----------|-----------|---------|
| survival | R | `coxph()`, `survreg()`, `survfit()` (multi-state) | 3.5+ |
| cmprsk | R | `cuminc()` | 2.2+ |

---

## Cox Proportional Hazards (Basic)

```r
library(survival)

set.seed(42)
n <- 100
treatment <- rbinom(n, 1, 0.5)
age <- rnorm(n, 50, 10)

# Weibull survival times: hazard ratio 0.5 for treatment, 1.02 per year of age
shape <- 1.5
scale <- exp(5 - 0.7*treatment + 0.02*age)
time <- rweibull(n, shape = shape, scale = scale)

# Random censoring (~30%)
censor_time <- runif(n, 0, max(time) * 1.2)
observed_time <- pmin(time, censor_time)
event <- as.integer(time <= censor_time)

data <- data.frame(time = observed_time, event = event, treatment = treatment, age = age)
cox_fit <- coxph(Surv(time, event) ~ treatment + age, data = data)
summary(cox_fit)
```

| Parameter | True value | Expected estimate (approximate) |
|-----------|------------|---------------------------------|
| β(treatment) | -0.7 | about -0.7 |
| β(age) | 0.02 | about 0.02 |
| HR(treatment) | 0.5 | about 0.5 |
| HR(age) | 1.02 | about 1.02 |
| Concordance | - | 0.6-0.8 |

## Cox PH with Breslow vs Efron Ties

```r
library(survival)

time <- c(1, 1, 2, 2, 2, 3, 4, 4, 5, 5)
event <- c(1, 1, 1, 0, 1, 1, 1, 0, 1, 1)
x <- c(0, 1, 0, 0, 1, 1, 0, 1, 0, 1)

coef(coxph(Surv(time, event) ~ x, ties = "breslow"))
coef(coxph(Surv(time, event) ~ x, ties = "efron"))   # R's default
```

Approximate: both coefficients about 0.3-0.6 with SE about 0.6-0.9 and p > 0.3. Efron is more
accurate with many tied event times; both give similar results with few ties. p2a's test
(`test_validate_cox_ties_against_r`) checks only that the two estimates are within 0.5 of each
other.

## Cox Model Tests (Wald, Score, LR)

```r
library(survival)

set.seed(123)
n <- 200
x1 <- rnorm(n)
x2 <- rnorm(n)
time <- rexp(n, rate = exp(0.5 * x1))
event <- rep(1, n)

cox_fit <- coxph(Surv(time, event) ~ x1 + x2)
summary(cox_fit)   # Wald, score (log-rank) and likelihood-ratio tests
```

Approximate: each statistic about 25-35 on 2 df, p < 0.001.

## Cox Model on R's `lung`

```r
library(survival)

lung <- read.csv("validation/datasets/generated/lung.csv")   # survival::lung; 1 = censored, 2 = dead
coxph(Surv(time, status) ~ age + sex + ph.karno, data = lung)  # 227 rows: one ph.karno is NA
```

| Variable | Coefficient | SE | HR | p-value |
|----------|-------------|-----|-----|---------|
| age | about 0.017 | about 0.009 | about 1.017 | about 0.07 |
| sex | about -0.51 | about 0.17 | about 0.60 | about 0.003 |
| ph.karno | about -0.016 | about 0.006 | about 0.984 | about 0.01 |

Approximate, not pinned (Tier 3: ECON-45, ECON-46, X-15). The status coding and the dropped row
are pinned in [survival.md](survival.md).

---

## AFT Weibull Model

```r
library(survival)

set.seed(42)
n <- 200
x <- rnorm(n, 0, 1)

# log(T) = 2 + 0.5 x + 0.5 epsilon, epsilon standard Gumbel (shape 1 / 0.5 = 2)
epsilon <- -log(runif(n))
time <- exp(2 + 0.5 * x + 0.5 * epsilon)
censor_time <- runif(n, 0, quantile(time, 0.9))
observed_time <- pmin(time, censor_time)
event <- as.integer(time <= censor_time)

aft_fit <- survreg(Surv(observed_time, event) ~ x, dist = "weibull")
coef(aft_fit); aft_fit$scale
```

| Parameter | True value | Expected estimate (approximate) |
|-----------|------------|---------------------------------|
| Intercept | 2.0 | about 2.0 |
| β(x) | 0.5 | about 0.5 |
| Scale (σ) | 0.5 | about 0.5 |

## AFT Log-Normal Model

```r
library(survival)

set.seed(42)
n <- 200
x <- rnorm(n, 0, 1)
time <- exp(3 + 0.3 * x + 0.8 * rnorm(n))
censor_time <- runif(n, 0, quantile(time, 0.85))
observed_time <- pmin(time, censor_time)
event <- as.integer(time <= censor_time)

survreg(Surv(observed_time, event) ~ x, dist = "lognormal")
```

| Parameter | True value | Expected estimate (approximate) |
|-----------|------------|---------------------------------|
| Intercept | 3.0 | about 3.0 |
| β(x) | 0.3 | about 0.3 |
| Scale (σ) | 0.8 | about 0.8 |

R's `survreg` parameterises the scale differently from p2a's standard AFT form; p2a refuses times
<= 0, as `survreg` does ("Invalid survival times for this distribution").

---

## Competing Risks (Aalen-Johansen)

```r
library(survival)
library(cmprsk)

set.seed(42)
n <- 100
time <- c(rexp(n/2, 0.5), rexp(n/2, 0.3))
event_type <- c(
  sample(c(0, 1, 2), n/2, replace = TRUE, prob = c(0.3, 0.5, 0.2)),
  sample(c(0, 1, 2), n/2, replace = TRUE, prob = c(0.2, 0.3, 0.5))
)
cuminc(time, event_type)

# With survival only: a multi-state fit
fit <- survfit(Surv(time, factor(event_type)) ~ 1)
summary(fit)
```

Approximate cumulative incidence at t = 2, 5 and 10: type 1 about 0.15, 0.35, 0.45; type 2 about
0.10, 0.25, 0.35. The sum of the cumulative incidences must not exceed 1 - S(t)
(`test_validate_competing_risks_cif_sum` checks that the final sum is at most 1). The confidence
limits use z from statrs; the standard-error formula is ECON-78 (Tier 3).

---

## Implementation Notes

- Newton-Raphson with tolerance 1e-9 and at most 25 (Cox) or 100 (AFT) iterations;
  `safe_inverse` with condition-number checking, falling back to a pseudoinverse for
  ill-conditioned Hessians.
- Ties: Breslow (the default of `CoxConfig` and of CLI `survival cox`) or Efron (the default of
  MCP `cox_ph`, as R's `coxph`).
- Cox hazard-ratio intervals are Wald intervals; no baseline hazard is estimated.

## References

1. Therneau, T.M. (2024). "A Package for Survival Analysis in R". R package version 3.5-8.
2. Gray, R.J. (2022). "cmprsk: Subdistribution Analysis of Competing Risks". R package.
3. Klein, J.P. & Moeschberger, M.L. (2003). *Survival Analysis*. Springer.
