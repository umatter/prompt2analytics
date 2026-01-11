# Causal Mediation Analysis Validation

## Method Overview

IPW-based causal mediation analysis for decomposing treatment effects into natural direct and indirect effects, following Huber (2014).

**p2a Function**: `run_mediation_analysis()`

**Estimands**:
- Total Effect (ATE)
- Natural Direct Effect (NDE)
- Natural Indirect Effect (NIE)
- Proportion Mediated (NIE/ATE)

## Reference Implementations

| Package | Function | Language | Notes |
|---------|----------|----------|-------|
| causalweight | `medweight()` | R | **Primary reference** |
| mediation | `mediate()` | R | Simulation-based |
| medflex | `neWeight()` | R | Natural effect models |

## Test Case 1: Synthetic Data with Known Mediation Structure

### Data Generating Process

```
X ~ Uniform(-1, 1)
D ~ Bernoulli(0.5)  # Random treatment
M = 0.5*D + 0.3*X + epsilon_m, epsilon_m ~ N(0, 0.3)
Y = 0.4*D + 0.6*M + 0.2*X + epsilon_y, epsilon_y ~ N(0, 0.5)

True effects:
- Direct effect (NDE): 0.4
- Indirect effect (NIE): 0.5 * 0.6 = 0.3
- Total effect (ATE): 0.4 + 0.3 = 0.7
- Proportion mediated: 0.3 / 0.7 ≈ 42.9%
```

### R Code (causalweight::medweight)

```r
library(causalweight)

set.seed(42)
n <- 1000

# Covariates
x <- runif(n, -1, 1)

# Treatment (random assignment)
d <- rbinom(n, 1, 0.5)

# Mediator: M = 0.5*D + 0.3*X + noise
m <- 0.5*d + 0.3*x + rnorm(n, 0, 0.3)

# Outcome: Y = 0.4*D + 0.6*M + 0.2*X + noise
y <- 0.4*d + 0.6*m + 0.2*x + rnorm(n, 0, 0.5)

# Run mediation analysis
result <- medweight(
  y = y,
  d = d,
  m = m,
  x = as.matrix(x),
  boot = 999,
  trim = 0.05
)

cat("=== R causalweight Results ===\n")
cat("Total Effect:", result$total, "\n")
cat("Direct Effect (NDE):", result$dir0, "\n")  # dir0 is NDE
cat("Indirect Effect (NIE):", result$indir0, "\n")  # indir0 is NIE
cat("Proportion Mediated:", result$indir0 / result$total, "\n")
```

### Expected Results

| Statistic | True Value | R (causalweight) | p2a (Rust) | Tolerance |
|-----------|------------|------------------|------------|-----------|
| Total Effect | 0.7 | 0.65-0.75 | 0.65-0.75 | 0.1 |
| Direct Effect (NDE) | 0.4 | 0.35-0.45 | 0.35-0.45 | 0.1 |
| Indirect Effect (NIE) | 0.3 | 0.25-0.35 | 0.25-0.35 | 0.1 |
| Proportion Mediated | 0.43 | 0.35-0.50 | 0.35-0.50 | 0.1 |

## Test Case 2: Confounded Setting

### DGP with Confounding

```
X ~ Uniform(-1, 1)
D ~ Bernoulli(logit(0.3 + 0.5*X))  # Treatment depends on X
M = 0.5*D + 0.3*X + epsilon_m
Y = 0.4*D + 0.6*M + 0.2*X + epsilon_y
```

### R Code

```r
set.seed(42)
n <- 1000

x <- runif(n, -1, 1)

# Treatment assignment depends on X (confounding)
ps <- plogis(0.3 + 0.5*x)
d <- rbinom(n, 1, ps)

m <- 0.5*d + 0.3*x + rnorm(n, 0, 0.3)
y <- 0.4*d + 0.6*m + 0.2*x + rnorm(n, 0, 0.5)

# Must control for X to get unbiased estimates
result <- medweight(
  y = y,
  d = d,
  m = m,
  x = as.matrix(x),
  boot = 999,
  trim = 0.05
)

print(result)
```

## Test Case 3: No Mediation (Zero Indirect Effect)

### DGP

```
D does not affect M
M = 0.3*X + epsilon_m
Y = 0.7*D + 0.6*M + 0.2*X + epsilon_y

True: NIE = 0, NDE = Total = 0.7
```

### R Code

```r
set.seed(42)
n <- 1000

x <- runif(n, -1, 1)
d <- rbinom(n, 1, 0.5)

# M does NOT depend on D
m <- 0.3*x + rnorm(n, 0, 0.3)

# Y depends on both D and M
y <- 0.7*d + 0.6*m + 0.2*x + rnorm(n, 0, 0.5)

result <- medweight(
  y = y, d = d, m = m,
  x = as.matrix(x),
  boot = 499
)

cat("NIE should be ~0:", result$indir0, "\n")
cat("NDE should be ~0.7:", result$dir0, "\n")
```

### Expected Results

| Statistic | True Value | Expected Range |
|-----------|------------|----------------|
| NIE | 0.0 | -0.05 to 0.05 |
| NDE | 0.7 | 0.65 to 0.75 |
| Total | 0.7 | 0.65 to 0.75 |

## Test Case 4: Full Mediation (Zero Direct Effect)

### DGP

```
D only affects Y through M
M = 0.8*D + 0.3*X + epsilon_m
Y = 0.0*D + 0.6*M + 0.2*X + epsilon_y

True: NDE = 0, NIE = Total = 0.48
```

### R Code

```r
set.seed(42)
n <- 1000

x <- runif(n, -1, 1)
d <- rbinom(n, 1, 0.5)

m <- 0.8*d + 0.3*x + rnorm(n, 0, 0.3)

# No direct effect of D on Y
y <- 0.0*d + 0.6*m + 0.2*x + rnorm(n, 0, 0.5)

result <- medweight(
  y = y, d = d, m = m,
  x = as.matrix(x),
  boot = 499
)

cat("NDE should be ~0:", result$dir0, "\n")
cat("NIE should be ~0.48:", result$indir0, "\n")
```

## Numerical Precision

| Sample Size | Effect Tolerance | SE Tolerance |
|-------------|------------------|--------------|
| n < 500 | 0.15 | 0.05 |
| n = 500-2000 | 0.10 | 0.03 |
| n > 2000 | 0.05 | 0.015 |

## Known Differences from causalweight

1. **Effect definitions**: causalweight returns `dir0`, `dir1`, `indir0`, `indir1`. p2a returns NDE and NIE (corresponding to `dir0` and `indir1` in the paper's notation, but there are subtleties in the identification strategy).

2. **Bootstrap**: causalweight may use different resampling strategy.

3. **Propensity score model**: Both use logistic regression, but implementation details may differ.

4. **Trimming**: Both use symmetric trimming on propensity scores.

## Comparing p2a to causalweight

To run direct comparison:

```r
# R code
library(causalweight)

# Generate data
set.seed(42)
n <- 1000
x <- runif(n, -1, 1)
d <- rbinom(n, 1, 0.5)
m <- 0.5*d + 0.3*x + rnorm(n, 0, 0.3)
y <- 0.4*d + 0.6*m + 0.2*x + rnorm(n, 0, 0.5)

# Save for Rust
write.csv(data.frame(y=y, d=d, m=m, x=x), "mediation_test_data.csv")

# R results
result <- medweight(y=y, d=d, m=m, x=as.matrix(x), boot=999, trim=0.05)
cat("Total:", result$total, "\n")
cat("NDE (dir0):", result$dir0, "\n")
cat("NIE (indir0):", result$indir0, "\n")
```

```rust
// Rust code (pseudo)
let ds = Dataset::from_csv("mediation_test_data.csv")?;
let result = run_mediation_analysis(&ds, "y", "d", "m", &["x"], config)?;
println!("Total: {}", result.total_effect);
println!("NDE: {}", result.direct_effect);
println!("NIE: {}", result.indirect_effect);
```

## References

- Huber, M. (2014). "Identifying Causal Mechanisms (Primarily) Based on Inverse Probability Weighting." *Journal of Applied Econometrics*, 29, 920-943.
- Imai, K., Keele, L., & Tingley, D. (2010). "A General Approach to Causal Mediation Analysis." *Psychological Methods*, 15(4), 309-334.
- Pearl, J. (2001). "Direct and Indirect Effects." *Proceedings of the 17th Conference on Uncertainty in Artificial Intelligence*.
- Bodory, H. & Huber, M. (2018). "causalweight: Estimation Methods for Causal Inference Based on Inverse Probability Weighting." R package.
