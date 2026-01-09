# Validation: Difference-in-Differences (DiD)

## Method Overview

Difference-in-Differences is a quasi-experimental design that estimates causal effects by comparing changes over time between a treatment group and a control group.

**Model**:
```
Y = β₀ + β₁×Treatment + β₂×Post + β₃×(Treatment × Post) + ε
```

**ATT (Average Treatment Effect on Treated)** = β₃

**Key Assumption**: Parallel trends - absent treatment, treated and control groups would have followed the same trend.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| base | R | Manual interaction regression | - |
| did | R | `att_gt()` for staggered DiD | 2.1-x |
| statsmodels | Python | Manual interaction regression | 0.14.x |

## Test Cases

### Test 1: Classic 2×2 DiD

**Data Generating Process**:
```
Treatment group: Y increases by 5 units after treatment
Control group: No change
Both groups have parallel pre-trends
```

**R Code**:
```r
set.seed(42)
n_per_group <- 100

# Create balanced panel
data <- expand.grid(
  id = 1:(2*n_per_group),
  time = c("pre", "post")
)

# Assign treatment (first n_per_group are treated)
data$treat <- ifelse(data$id <= n_per_group, 1, 0)
data$post <- ifelse(data$time == "post", 1, 0)

# Outcome with treatment effect = 5
data$y <- 10 +                           # Baseline
          2 * data$treat +               # Treated group higher
          3 * data$post +                # Time trend
          5 * data$treat * data$post +   # Treatment effect (ATT)
          rnorm(nrow(data), 0, 1)

# DiD regression
did_fit <- lm(y ~ treat + post + treat:post, data = data)
summary(did_fit)

# ATT = coefficient on treat:post
# Expected: ~5
```

**Results Comparison**:

| Statistic | True Value | R Estimate | p2a Rust | Tolerance |
|-----------|------------|------------|----------|-----------|
| ATT (β₃) | 5.0 | ~5.0 | ~5.0 | 0.5 |
| SE(ATT) | - | varies | varies | 0.1 |
| β(treat) | 2.0 | ~2.0 | ~2.0 | 0.3 |
| β(post) | 3.0 | ~3.0 | ~3.0 | 0.3 |

**Rust Test**: `crates/p2a-core/src/econometrics/did.rs::tests::test_validate_classic_did`

---

### Test 2: With Covariates

**R Code**:
```r
set.seed(42)
n_per_group <- 200

data <- expand.grid(
  id = 1:(2*n_per_group),
  time = c("pre", "post")
)

data$treat <- ifelse(data$id <= n_per_group, 1, 0)
data$post <- ifelse(data$time == "post", 1, 0)

# Add covariate
data$x <- rnorm(nrow(data))

# Outcome with covariate effect
data$y <- 10 + 2*data$treat + 3*data$post +
          5*data$treat*data$post +
          1.5*data$x +  # Covariate effect
          rnorm(nrow(data), 0, 1)

did_fit <- lm(y ~ treat + post + treat:post + x, data = data)
summary(did_fit)
```

**Validation Criteria**:
- ATT still close to 5
- Covariate coefficient close to 1.5
- Standard errors may be smaller with covariates

---

### Test 3: Panel DiD with Entity Fixed Effects

**R Code**:
```r
library(plm)

set.seed(42)
n_entities <- 100
n_periods <- 10
treatment_period <- 6

data <- expand.grid(
  id = 1:n_entities,
  t = 1:n_periods
)

# Half are treated (after period 6)
data$treat <- ifelse(data$id <= 50, 1, 0)
data$post <- ifelse(data$t >= treatment_period, 1, 0)

# Entity fixed effects
entity_fe <- rnorm(n_entities)
time_fe <- seq(0, 1, length.out = n_periods)

data$y <- entity_fe[data$id] + time_fe[data$t] +
          5 * data$treat * data$post +  # Treatment effect
          rnorm(nrow(data), 0, 0.5)

pdata <- pdata.frame(data, index = c("id", "t"))

# DiD with entity FE
did_fe <- plm(y ~ treat:post, data = pdata, model = "within")
summary(did_fe)
```

**Validation Criteria**:
- ATT (interaction coefficient) close to 5
- Entity FE absorbed
- Correct standard errors

---

### Test 4: No Treatment Effect (Null Case)

**R Code**:
```r
set.seed(42)
n <- 400

data <- data.frame(
  treat = rep(c(0, 1), each = n/2),
  post = rep(c(0, 1), n/2),
  y = rnorm(n)  # No actual treatment effect
)

did_fit <- lm(y ~ treat + post + treat:post, data = data)
summary(did_fit)

# ATT should not be significantly different from 0
```

**Validation Criteria**:
- ATT close to 0
- p-value > 0.05 (fail to reject no effect)

---

## ATT Calculation

The ATT can be computed directly from group means:

```
ATT = (Ȳ_treat,post - Ȳ_treat,pre) - (Ȳ_control,post - Ȳ_control,pre)
```

| Group | Pre | Post | Δ |
|-------|-----|------|---|
| Control | Ȳ₀₀ | Ȳ₀₁ | Ȳ₀₁ - Ȳ₀₀ |
| Treated | Ȳ₁₀ | Ȳ₁₁ | Ȳ₁₁ - Ȳ₁₀ |

ATT = (Ȳ₁₁ - Ȳ₁₀) - (Ȳ₀₁ - Ȳ₀₀)

## Numerical Precision Summary

| Test Case | n | ATT Precision |
|-----------|---|---------------|
| Classic 2×2 | 400 | < 0.1 |
| With covariates | 800 | < 0.1 |
| Panel DiD | 1000 | < 0.05 |

## Known Differences

1. **Standard errors**: Default vs. clustered by entity.
2. **Fixed effects**: Some implementations require explicit FE specification.
3. **Weights**: Unweighted vs. weighted by group size.

## Parallel Trends Assumption

Cannot be tested directly, but can assess plausibility:

**R Code for Pre-Trend Test**:
```r
# Check pre-treatment trends are parallel
pre_data <- subset(data, post == 0)
trend_test <- lm(y ~ treat + as.factor(t) + treat:as.factor(t), data = pre_data)
# Interaction terms should be insignificant
```

## Running the Tests

```bash
# Run DiD validation tests
cargo test -p p2a-core -- did::tests::test_validate

# Run with output
cargo test -p p2a-core -- did --nocapture
```

## References

- Card, D. & Krueger, A.B. (1994). "Minimum Wages and Employment". *American Economic Review*, 84(4), 772-793.
- Angrist, J.D. & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press. Chapter 5.
- Goodman-Bacon, A. (2021). "Difference-in-Differences with Variation in Treatment Timing". *Journal of Econometrics*, 225(2), 254-277.
