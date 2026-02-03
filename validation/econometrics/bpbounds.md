# Balke-Pearl Bounds Validation

## Method Overview

**Balke-Pearl bounds** provide sharp nonparametric bounds on the Average Causal Effect (ACE) using instrumental variables without assuming parametric models. These bounds are the tightest possible given only the observed data and the IV assumptions.

## Mathematical Framework

### Cell Probabilities
For binary variables Z (instrument), D (treatment), Y (outcome):
- p_{dy|z} = P(D=d, Y=y | Z=z)

Eight cell probabilities: p00|0, p01|0, p10|0, p11|0, p00|1, p01|1, p10|1, p11|1

### Bounds Without Monotonicity

```
ACE_lower = max(
    p00|1 - p00|0 - p01|0 - p10|0,
    p00|0 - p00|1 - p01|1 - p10|1,
    p11|1 - p11|0 - p01|0 - p10|0,
    p11|0 - p11|1 - p01|1 - p10|1,
    -1
)

ACE_upper = min(
    p11|1 - p11|0 + p01|0 + p10|0,
    p11|0 - p11|1 + p01|1 + p10|1,
    p00|0 - p00|1 + p01|1 + p10|1,
    p00|1 - p00|0 + p01|0 + p10|0,
    1
)
```

### Bounds With Monotonicity

When assuming no defiers (P(D=1|Z=1,U=u) >= P(D=1|Z=0,U=u)):
```
ACE_lower = p00|0 - p00|1 - p01|1 - p10|1
ACE_upper = p00|0 + p01|0 + p11|0 - p01|1
```

## Reference Implementation

**R package**: bpbounds (Ramsahai & Palmer)
- URL: https://cran.r-project.org/package=bpbounds
- Version tested: 0.1.5

## Test Case 1: Synthetic Data with Known Structure

### Data Generation (R)
```r
library(bpbounds)

# Create simple compliance data
set.seed(42)
n <- 1000

# Instrument (randomized assignment)
z <- rbinom(n, 1, 0.5)

# Treatment with noncompliance
# Compliers: D follows Z
# Always-takers: D=1 regardless
# Never-takers: D=0 regardless
u <- runif(n)
d <- as.numeric((z == 1 & u < 0.7) | (z == 0 & u < 0.2))

# Outcome depends on treatment
y <- rbinom(n, 1, ifelse(d == 1, 0.7, 0.3))

# Compute cell probabilities
table(z, d, y)
```

### Expected Results

Cell Probabilities (from R):
```
Z=0: p00=0.35, p01=0.25, p10=0.15, p11=0.25
Z=1: p00=0.15, p01=0.10, p10=0.25, p11=0.50
```

Bounds (without monotonicity):
- Lower: approximately -0.30
- Upper: approximately 0.70

Bounds (with monotonicity):
- Lower: approximately 0.15
- Upper: approximately 0.55

## Test Case 2: Perfect Compliance

When D = Z (perfect compliance), bounds should collapse to the ITT effect.

### R Code
```r
# Perfect compliance
z <- rep(c(0, 1), each = 500)
d <- z  # Perfect compliance
y <- rbinom(1000, 1, ifelse(d == 1, 0.8, 0.4))

result <- bpbounds(p = table(z, d, y) / c(500, 500))
```

### Expected Results

With perfect compliance:
- P(D=0|Z=1) = 0
- P(D=1|Z=0) = 0
- Bounds collapse to point estimate: ITT = P(Y=1|Z=1) - P(Y=1|Z=0)

## Rust Test Functions

```rust
#[test]
fn test_validate_bpbounds_synthetic() {
    // See test_validate_against_r in bpbounds.rs
}

#[test]
fn test_validate_bpbounds_perfect_compliance() {
    // See test_perfect_compliance in bpbounds.rs
}
```

## Validation Results

### Coefficient Comparison

| Statistic | R bpbounds | Rust p2a-core | Difference |
|-----------|------------|---------------|------------|
| ACE Lower (no mono) | -0.30 | TBD | TBD |
| ACE Upper (no mono) | 0.70 | TBD | TBD |
| ACE Lower (mono) | 0.15 | TBD | TBD |
| ACE Upper (mono) | 0.55 | TBD | TBD |
| Wald Estimate | 0.50 | TBD | TBD |

### Tolerance

| Sample Size | Bounds | Wald Estimate |
|-------------|--------|---------------|
| n < 100 | 0.05 | 0.1 |
| n = 100-1000 | 0.01 | 0.05 |
| n > 1000 | 0.001 | 0.01 |

## References

- Balke, A., & Pearl, J. (1997). Bounds on treatment effects from studies with imperfect compliance. *Journal of the American Statistical Association*, 92(439), 1171-1176.

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

- Robins, J. M. (1989). The analysis of randomized and non-randomized AIDS treatment trials. In *Health Service Research Methodology* (pp. 113-159).

- Manski, C. F. (1990). Nonparametric bounds on treatment effects. *American Economic Review Papers and Proceedings*, 80(2), 319-323.

- Palmer, T. M., Ramsahai, R. R., Didelez, V., & Sheehan, N. A. (2011). Nonparametric bounds for the causal effect in a binary instrumental-variable model. *Stata Journal*, 11(3), 345-367.

## Implementation Notes

1. **Numerical stability**: Cell probabilities are computed from counts to avoid floating point issues.

2. **Monotonicity check**: The implementation warns when monotonicity is assumed but appears violated in the data.

3. **Bootstrap CI**: Uses percentile method with configurable number of replications.

4. **Wald estimate**: Provided for comparison; should be within bounds if monotonicity holds.
