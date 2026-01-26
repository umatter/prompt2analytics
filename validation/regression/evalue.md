# E-Value Sensitivity Analysis Validation

## Overview

E-values quantify the minimum strength of association that an unmeasured confounder would need to have with both the treatment and outcome to fully explain away an observed treatment-outcome association. This implementation follows VanderWeele & Ding (2017).

## Reference Implementation

- **R Package**: EValue (https://CRAN.R-project.org/package=EValue)
- **Version**: 4.1.3 (or later)
- **Authors**: Maya B. Mathur, Peng Ding, Tyler J. VanderWeele

## Test Cases

### Test 1: Risk Ratio - Basic E-value

```r
# R code
library(EValue)

# RR = 2.5
evalue(RR = 2.5)
# E-value: 4.44

# RR = 3.9 (example from paper)
evalue(RR = 3.9)
# E-value: 7.26
```

**Rust results**:
- `evalue_rr(2.5)` = 4.44
- `evalue_rr(3.9)` = 7.26

**Tolerance**: |diff| < 0.01

### Test 2: Risk Ratio with Confidence Interval

```r
library(EValue)

evalue(RR = 2.5, lo = 1.8, hi = 3.5)
# Point estimate E-value: 4.44
# CI limit E-value: 2.99 (for lower limit 1.8)
```

**Rust results**:
```rust
let result = evalue_rr_ci(2.5, Some(1.8), Some(3.5)).unwrap();
assert!((result.evalue_point - 4.44).abs() < 0.01);
assert!((result.evalue_ci.unwrap() - 3.0).abs() < 0.02);
```

### Test 3: Odds Ratio - Rare Outcome

```r
library(EValue)

# For rare outcome, OR approximates RR
evalues.OR(2.5, rare = TRUE)
# E-value: 4.44 (same as RR = 2.5)
```

**Rust results**:
```rust
let result = evalue_or(2.5, None, None, true).unwrap();
assert!((result.evalue_point - 4.44).abs() < 0.01);
assert!((result.risk_ratio - 2.5).abs() < 1e-10);
```

### Test 4: Odds Ratio - Common Outcome

```r
library(EValue)

# For common outcome, apply sqrt transformation
# OR = 4 -> RR_approx = sqrt(4) = 2
evalues.OR(4, rare = FALSE)
# E-value: 3.41 (for RR = 2)
```

**Rust results**:
```rust
let result = evalue_or(4.0, None, None, false).unwrap();
assert!((result.risk_ratio - 2.0).abs() < 1e-10);  // sqrt(4) = 2
assert!((result.evalue_point - 3.41).abs() < 0.01);
```

### Test 5: Standardized Mean Difference

```r
library(EValue)

# SMD = 0.5
evalues.SMD(0.5)
# Uses exp(0.91 * d) conversion
# RR_approx = exp(0.91 * 0.5) = 1.576
# E-value = 1.576 + sqrt(1.576 * 0.576) = 2.53
```

**Rust results**:
```rust
let result = evalue_smd(0.5, None).unwrap();
let expected_rr = (0.91 * 0.5_f64).exp();
assert!((result.risk_ratio - expected_rr).abs() < 1e-10);
assert!((result.evalue_point - 2.53).abs() < 0.05);
```

### Test 6: Protective Effect (RR < 1)

```r
library(EValue)

# RR = 0.5 should give same E-value as RR = 2
evalue(RR = 0.5)
# E-value: 3.41 (same as RR = 2)
```

**Rust results**:
```rust
let ev_05 = evalue_rr(0.5);
let ev_20 = evalue_rr(2.0);
assert!((ev_05 - ev_20).abs() < 1e-10);
```

## Formulas Implemented

### Risk Ratio E-value
For RR >= 1:
```
E-value = RR + sqrt(RR * (RR - 1))
```

For RR < 1, use 1/RR and apply the formula.

### Odds Ratio Conversion
- Rare outcome: OR directly approximates RR
- Common outcome: RR_approx = sqrt(OR)

Then apply RR E-value formula.

### Hazard Ratio Conversion
Same as odds ratio conversion.

### SMD Conversion (Chinn 2000)
```
RR_approx = exp(0.91 * d)
```
where d is the standardized mean difference.

### Risk Difference Conversion
```
RR = (baseline_risk + RD) / baseline_risk
```

## Edge Cases

1. **RR = 1**: E-value = 1 (no confounding needed)
2. **CI includes null**: E-value for CI = 1
3. **Invalid inputs**: Returns error for RR <= 0, OR <= 0, etc.

## References

1. VanderWeele, T. J., & Ding, P. (2017). "Sensitivity Analysis in Observational
   Research: Introducing the E-Value". Annals of Internal Medicine, 167(4), 268-274.
   https://doi.org/10.7326/M16-2607

2. VanderWeele, T. J. (2017). "On a Square-Root Transformation of the Odds Ratio
   for a Common Outcome". Epidemiology, 28(6), e58-e59.

3. Chinn, S. (2000). "A simple method for converting an odds ratio to effect size
   for use in meta-analysis". Statistics in Medicine, 19(22), 3127-3131.

4. Linden, A., Mathur, M. B., & VanderWeele, T. J. (2020). "Conducting sensitivity
   analysis for unmeasured confounding in observational studies using E-values:
   The evalue package". The Stata Journal, 20(1), 162-175.
