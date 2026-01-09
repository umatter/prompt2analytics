# Econometrics Guide: Assumptions and Interpretation

This guide explains the assumptions underlying each econometric method in prompt2analytics and how to interpret the results.

## Table of Contents
- [OLS Regression](#ols-regression)
- [Robust Standard Errors](#robust-standard-errors)
- [Panel Data Models](#panel-data-models)
- [High-Dimensional Fixed Effects](#high-dimensional-fixed-effects)
- [Instrumental Variables](#instrumental-variables)
- [Difference-in-Differences](#difference-in-differences)
- [Discrete Choice Models](#discrete-choice-models)
- [Regression Diagnostics](#regression-diagnostics)

---

## OLS Regression

### Model
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ + ε
```

### Gauss-Markov Assumptions

For OLS to be BLUE (Best Linear Unbiased Estimator):

1. **Linearity**: The relationship between y and x is linear in parameters
   - *Violation*: Biased and inconsistent estimates
   - *Fix*: Transform variables (log, polynomial) or use non-linear models

2. **Random Sampling**: Observations are independently drawn
   - *Violation*: Incorrect standard errors
   - *Fix*: Use clustered standard errors or panel methods

3. **No Perfect Multicollinearity**: No exact linear relationship among regressors
   - *Violation*: Cannot estimate coefficients
   - *Fix*: Remove redundant variables, check VIF

4. **Zero Conditional Mean**: E(ε|X) = 0
   - *Violation*: Omitted variable bias, simultaneity bias
   - *Fix*: Add controls, use IV, use panel FE

5. **Homoskedasticity**: Var(ε|X) = σ²
   - *Violation*: Inefficient estimates, incorrect SEs
   - *Fix*: Use robust standard errors (HC0-HC3)

6. **No Autocorrelation**: Cov(εᵢ, εⱼ) = 0 for i ≠ j
   - *Violation*: Incorrect standard errors
   - *Fix*: Use Newey-West SEs, cluster by time

### Interpreting Output

| Statistic | Interpretation |
|-----------|----------------|
| Coefficient (β) | One-unit increase in x → β-unit change in y (holding others constant) |
| Standard Error | Precision of coefficient estimate |
| t-statistic | β / SE; measures distance from zero in SE units |
| p-value | P(|t| > observed) if true β = 0 |
| R² | Proportion of variance in y explained by model |
| Adjusted R² | R² penalized for number of regressors |
| F-statistic | Tests if all coefficients (except intercept) = 0 |

### Significance Stars
- `*` : p < 0.10 (marginally significant)
- `**` : p < 0.05 (significant)
- `***` : p < 0.01 (highly significant)

---

## Robust Standard Errors

### Why Use Them?
When homoskedasticity is violated, OLS standard errors are biased. Robust (heteroskedasticity-consistent) standard errors correct for this.

### Types Available

| Type | Formula | When to Use |
|------|---------|-------------|
| HC0 | Basic White estimator | Large samples |
| HC1 | HC0 × n/(n-k) | Default; small-sample correction |
| HC2 | Weights by leverage | Moderate leverage |
| HC3 | More conservative | Small samples, high leverage |

**Recommendation**: Use HC1 as default. Use HC3 for small samples.

### Clustered Standard Errors

When observations are correlated within groups (firms, states, individuals over time):

```
/regression_clustered dataset:panel y:outcome x:treatment cluster1:firm
```

**Two-way clustering** (e.g., firm + year) accounts for correlation in both dimensions:
```
/regression_clustered dataset:panel y:outcome x:treatment cluster1:firm cluster2:year
```

---

## Panel Data Models

### Fixed Effects (FE)

**Model**:
```
yᵢₜ = αᵢ + Xᵢₜβ + εᵢₜ
```

**Assumptions**:
1. Strict exogeneity: E(εᵢₜ|Xᵢ₁, ..., Xᵢₜ, αᵢ) = 0
2. No perfect multicollinearity in time-varying regressors

**Key Property**: Eliminates time-invariant unobserved heterogeneity (αᵢ)

**Interpretation**:
- Coefficients represent **within-entity** effects
- "When firm i's x increases by 1, its y changes by β"
- Cannot estimate effects of time-invariant variables

**When to Use**:
- Unobserved factors likely correlated with regressors
- Interest in within-unit variation
- Hausman test rejects RE

### Random Effects (RE)

**Model**:
```
yᵢₜ = α + Xᵢₜβ + uᵢ + εᵢₜ
```

**Assumptions**:
1. E(uᵢ|Xᵢₜ) = 0 (unobserved effects uncorrelated with regressors)
2. uᵢ is random with constant variance

**Key Property**: More efficient than FE if assumptions hold

**Interpretation**:
- Coefficients represent both within- and between-entity effects
- Can estimate effects of time-invariant variables

**When to Use**:
- Unobserved effects likely random and uncorrelated with regressors
- Hausman test fails to reject RE

### Hausman Test

**Null Hypothesis**: RE is consistent (use RE)
**Alternative**: RE is inconsistent (use FE)

| p-value | Decision |
|---------|----------|
| < 0.05 | Reject H0 → Use Fixed Effects |
| ≥ 0.05 | Fail to reject → Random Effects is acceptable |

---

## High-Dimensional Fixed Effects

### The Challenge

When you have multiple categorical variables to control for (e.g., firm, year, industry), creating dummy variables becomes infeasible:
- 10,000 firms × 20 years = 200,000 dummy variables
- Memory and computation issues
- Standard panel FE only handles one dimension

### Solution: Method of Alternating Projections (MAP)

HDFE uses the **Method of Alternating Projections** to efficiently "demean" data across multiple dimensions without creating dummies.

**Algorithm**:
```
repeat until convergence:
    for each fixed effect dimension:
        subtract group means from data
```

### Model

```
yᵢₜⱼ = Xᵢₜⱼβ + αᵢ + γₜ + δⱼ + εᵢₜⱼ
```

Where:
- αᵢ = firm fixed effects
- γₜ = time fixed effects
- δⱼ = industry fixed effects
- All FE terms are "absorbed" (not estimated, but removed from variation)

### Using HDFE

```
panel_hdfe dataset:panel y:outcome x:treatment,control fe:firm,year,industry
```

**Parameters**:
- `fe`: List of columns to absorb as fixed effects
- `tolerance`: Convergence threshold (default: 1e-8)
- `max_iterations`: Maximum MAP iterations (default: 10000)
- `se_type`: Standard error type ('standard', 'hc0'-'hc3')

### Interpreting Output

| Statistic | Interpretation |
|-----------|----------------|
| Coefficient (β) | Effect of X on Y, controlling for all absorbed FE |
| Within R² | Variance explained after removing FE |
| Iterations | Number of demeaning passes (higher = slower convergence) |
| Convergence | Final change in demeaned values (should be < tolerance) |
| DF Absorbed | Degrees of freedom consumed by FE |

### Degrees of Freedom

For multi-way FE, degrees of freedom are adjusted:
```
df_residual = n - k - (Σ levels for each FE) + (redundant terms)
```

The redundant terms account for the fact that the grand mean is absorbed multiple times.

### Key Assumptions

1. **Strict exogeneity**: E(εᵢₜ|Xᵢ₁, ..., Xᵢₜ, αᵢ, γₜ, ...) = 0
2. **Sufficient within-group variation**: After removing FE, X must still vary
3. **No perfect multicollinearity**: X cannot be perfectly explained by FE combinations

### Common Issues

**Collinearity after demeaning**: If X is constant within FE groups:
```
# BAD: Same X values for all firms in each year
x = [1, 2, 3, 4] for firm A
x = [1, 2, 3, 4] for firm B  ← After year demeaning, X becomes 0!

# GOOD: Different patterns across firms
x = [1, 3, 2, 5] for firm A
x = [2, 1, 4, 3] for firm B
```

**Singleton observations**: Observations that are the only one in their FE cell provide no identifying variation.

### When to Use HDFE

| Situation | Recommended Method |
|-----------|-------------------|
| One FE dimension | `panel_fixed_effects` |
| Two+ FE dimensions | `panel_hdfe` |
| Very large FE dimensions (>1000 levels) | `panel_hdfe` |
| Need to absorb industry, region, etc. | `panel_hdfe` |

### References

- Gaure, S. (2013). "lfe: Linear Group Fixed Effects". *The R Journal*, 5(2), 104-117.
- Guimarães, P. & Portugal, P. (2010). "A Simple Feasible Procedure to Fit Models with High-Dimensional Fixed Effects". *Stata Journal*, 10(4), 628-649.
- Correia, S. (2017). "Linear Models with Multi-Way Fixed Effects: An Efficient and Feasible Estimator". Working Paper.

---

## Instrumental Variables

### The Problem: Endogeneity

When E(ε|X) ≠ 0, OLS is biased and inconsistent. Causes:
- Omitted variables correlated with x
- Measurement error in x
- Simultaneity (x and y determined together)

### Solution: 2SLS

**Requirements for instrument Z**:
1. **Relevance**: Corr(Z, X) ≠ 0 (check via first-stage F)
2. **Exclusion**: Cov(Z, ε) = 0 (not directly testable)

**Two Stages**:
1. Regress endogenous X on Z (and exogenous variables) → get X̂
2. Regress Y on X̂ (and exogenous variables) → get β

### First-Stage Diagnostics

| Statistic | Rule of Thumb |
|-----------|---------------|
| F-statistic | > 10 for strong instruments |
| Partial R² | Higher is better |

**Weak Instruments**: F < 10 suggests weak instruments, leading to:
- Biased IV estimates (toward OLS)
- Unreliable inference

### Interpretation

IV estimates are **Local Average Treatment Effects (LATE)**:
- Effect for "compliers" whose treatment status is changed by the instrument
- May differ from Average Treatment Effect (ATE)

---

## Difference-in-Differences

### Model
```
Y = β₀ + β₁·Treatment + β₂·Post + β₃·(Treatment × Post) + ε
```

### Key Assumption: Parallel Trends

Without treatment, treated and control groups would have followed the same trend.

**Cannot be tested directly**, but can assess plausibility:
- Check pre-treatment trends are similar
- Use event study to visualize

### Interpreting the ATT

**ATT (Average Treatment Effect on Treated) = β₃**

```
ATT = (Treated_Post - Treated_Pre) - (Control_Post - Control_Pre)
```

| Group | Pre | Post | Change |
|-------|-----|------|--------|
| Control | Y₀₀ | Y₀₁ | Y₀₁ - Y₀₀ (trend) |
| Treated | Y₁₀ | Y₁₁ | Y₁₁ - Y₁₀ (trend + effect) |

ATT = (Y₁₁ - Y₁₀) - (Y₀₁ - Y₀₀)

### Common Pitfalls

1. **Violation of parallel trends**: Include group-specific trends or use synthetic control
2. **Anticipation effects**: Treatment affects behavior before official start
3. **Spillovers**: Treatment affects control group
4. **Heterogeneous timing**: Use staggered DiD methods

---

## Discrete Choice Models

### Logit (Logistic Regression)

**Model**:
```
P(Y=1|X) = exp(Xβ) / (1 + exp(Xβ)) = Λ(Xβ)
```

**Interpretation**:
- **Coefficients (β)**: Change in log-odds for one-unit increase in X
- **Odds Ratio (exp(β))**: Multiplicative change in odds
- **Marginal Effect**: ∂P/∂X = Λ(Xβ)(1-Λ(Xβ))β

**Example**:
- β = 0.5 → exp(0.5) = 1.65
- One-unit increase in X multiplies odds by 1.65 (65% increase)

### Probit

**Model**:
```
P(Y=1|X) = Φ(Xβ)
```

where Φ is the standard normal CDF.

**Interpretation**:
- Coefficients not directly interpretable
- Use marginal effects for interpretation
- Marginal Effect: ∂P/∂X = φ(Xβ)β

### Logit vs Probit

| Aspect | Logit | Probit |
|--------|-------|--------|
| Distribution | Logistic | Normal |
| Coefficient ratio | ~1.6 × Probit | ~0.625 × Logit |
| Marginal effects | Similar | Similar |
| Tail behavior | Heavier tails | Thinner tails |

**In practice**: Results are usually very similar. Logit preferred for odds ratio interpretation.

### Pseudo R²

McFadden's Pseudo R²:
```
R² = 1 - (Log-Likelihood / Log-Likelihood_null)
```

**Interpretation**:
- 0.2-0.4 is considered good fit
- Not directly comparable to OLS R²

---

## Regression Diagnostics

### Jarque-Bera Test (Normality)

**Tests**: Whether residuals are normally distributed

| Result | Interpretation |
|--------|----------------|
| p > 0.05 | Residuals approximately normal |
| p < 0.05 | Non-normal residuals |

**Implications**: Non-normality affects inference in small samples; less important in large samples (CLT).

### Breusch-Pagan Test (Heteroskedasticity)

**Tests**: Whether variance of residuals is constant

| Result | Interpretation |
|--------|----------------|
| p > 0.05 | Homoskedasticity (constant variance) |
| p < 0.05 | Heteroskedasticity present |

**Fix**: Use robust standard errors (HC1-HC3)

### Durbin-Watson Test (Autocorrelation)

**Tests**: First-order autocorrelation in residuals

| DW Value | Interpretation |
|----------|----------------|
| ≈ 2 | No autocorrelation |
| < 2 | Positive autocorrelation |
| > 2 | Negative autocorrelation |

**Rule of thumb**: 1.5-2.5 is acceptable

**Fix**: Use Newey-West standard errors or model time structure

### Variance Inflation Factor (VIF)

**Measures**: Multicollinearity for each variable

| VIF | Interpretation |
|-----|----------------|
| 1 | No correlation with other variables |
| 1-5 | Moderate correlation |
| 5-10 | High correlation |
| > 10 | Severe multicollinearity |

**Fix**: Remove highly correlated variables, combine into index, or use regularization

### Condition Number

**Measures**: Overall multicollinearity in design matrix

| Value | Interpretation |
|-------|----------------|
| < 30 | Acceptable |
| 30-100 | Moderate concern |
| > 100 | Severe multicollinearity |

---

## Quick Reference: Choosing a Method

| Situation | Recommended Method |
|-----------|-------------------|
| Cross-sectional, exogenous X | OLS with robust SEs |
| Cross-sectional, endogenous X | IV/2SLS |
| Panel, one FE dimension | Fixed Effects |
| Panel, multiple FE dimensions | HDFE |
| Panel, random unobserved effects | Random Effects |
| Before/after treatment + control | Difference-in-Differences |
| Binary outcome | Logit or Probit |
| Clustered data | Clustered SEs |
| Time series correlation | Newey-West SEs |

---

## Further Reading

- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*
- Angrist, J.D. & Pischke, J.S. (2009). *Mostly Harmless Econometrics*
- Greene, W.H. (2018). *Econometric Analysis*
- Cameron, A.C. & Trivedi, P.K. (2005). *Microeconometrics*
