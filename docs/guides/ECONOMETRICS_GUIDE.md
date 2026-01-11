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
- [Treatment Effect Estimation](#treatment-effect-estimation)
- [Causal Mediation Analysis](#causal-mediation-analysis)
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

## Treatment Effect Estimation

These methods estimate causal treatment effects in observational studies where treatment assignment is not random.

### Inverse Probability Weighting (IPW)

**The Problem**: In observational data, treated and control groups differ systematically. Simple comparisons are biased.

**Solution**: Weight observations by inverse of treatment probability to create "pseudo-populations" where treatment is independent of covariates.

**Propensity Score**:
```
p(X) = P(D=1|X)
```
Estimated using logistic regression of treatment D on covariates X.

**Estimands**:

| Estimand | Description | Weight (Treated) | Weight (Control) |
|----------|-------------|------------------|------------------|
| ATE | Average Treatment Effect (population) | 1/p(X) | 1/(1-p(X)) |
| ATT | Average Treatment Effect on Treated | 1 | p(X)/(1-p(X)) |

**Normalized (Hajek) Estimator**:
```
ATE = Σ[w₁·Y] / Σ[w₁] - Σ[w₀·Y] / Σ[w₀]
```
Uses normalized weights that sum to 1, providing more stable estimates.

**Trimming**: Extreme propensity scores (near 0 or 1) create unstable weights. Trim observations with p(X) < trim or p(X) > 1-trim.

**Key Assumptions**:
1. **Unconfoundedness (Selection on Observables)**: Treatment assignment independent of potential outcomes conditional on X
   ```
   (Y(0), Y(1)) ⊥ D | X
   ```
2. **Positivity (Common Support)**: All units have positive probability of treatment
   ```
   0 < P(D=1|X) < 1 for all X
   ```
3. **Correct Propensity Model**: Logit model for p(X) is correctly specified

**Usage**:
```
treatment_ipw dataset:data outcome:earnings treatment:training
    covariates:age,education,experience estimand:ate trim:0.05
```

**Interpreting Output**:

| Statistic | Interpretation |
|-----------|----------------|
| Effect | Estimated treatment effect (ATE or ATT) |
| Std Error | Bootstrap standard error |
| 95% CI | Confidence interval from bootstrap distribution |
| p-value | Two-sided test of H0: effect = 0 |
| n_obs | Observations after trimming |
| n_trimmed | Observations removed due to extreme propensity scores |
| PS Mean (Treated/Control) | Propensity score diagnostics |

### Doubly Robust Estimation (AIPW)

**The Problem**: IPW is consistent only if propensity model is correct. Outcome regression is consistent only if outcome model is correct.

**Solution**: Augmented IPW (AIPW) combines both, achieving consistency if **either** model is correct.

**AIPW Estimator**:
```
τ_AIPW = (1/n) Σ [
    μ̂₁(X) - μ̂₀(X)
    + D/p̂(X) · (Y - μ̂₁(X))
    - (1-D)/(1-p̂(X)) · (Y - μ̂₀(X))
]
```

Where:
- μ̂₁(X) = predicted outcome if treated (from OLS on treated group)
- μ̂₀(X) = predicted outcome if control (from OLS on control group)
- p̂(X) = estimated propensity score

**Methods Available**:

| Method | Description | Consistency Requires |
|--------|-------------|---------------------|
| AIPW | Doubly robust (default) | Either PS or outcome model correct |
| IPW | Inverse probability weighting only | Correct propensity score model |
| Regression | Outcome regression only | Correct outcome model |

**Usage**:
```
treatment_doubly_robust dataset:data outcome:wages treatment:job_training
    covariates:education,age,prior_wages method:aipw estimand:att
```

**Why Doubly Robust?**:
- More efficient than IPW when both models are correct
- Provides insurance against model misspecification
- Standard approach in modern causal inference

**Interpreting Output**:

| Statistic | Interpretation |
|-----------|----------------|
| Effect | Estimated treatment effect |
| Method | Estimation method used (AIPW/IPW/Regression) |
| Outcome R² (Treated) | Fit of outcome model for treated |
| Outcome R² (Control) | Fit of outcome model for control |

**Key Considerations**:
1. **Overlap**: Check propensity score distributions for treated/control overlap
2. **Model Specification**: Include relevant confounders in both models
3. **Sample Size**: Requires sufficient observations in both treatment groups
4. **Bootstrap**: Used for inference to account for two-step estimation

### Choosing Between Methods

| Situation | Recommended Method |
|-----------|-------------------|
| Large sample, good overlap | AIPW (doubly robust) |
| Simple setting, trust PS model | IPW |
| Many covariates, complex relationships | AIPW |
| Need insurance against misspecification | AIPW |
| Limited overlap (many extreme PS) | Consider trimming, matching, or different design |

### Common Pitfalls

1. **Unmeasured Confounding**: These methods cannot address unobserved confounders
2. **Poor Overlap**: Extreme propensity scores indicate lack of common support
3. **Model Misspecification**: Neither method corrects for both models being wrong
4. **Positivity Violations**: If some covariate patterns only appear in one group
5. **Post-Treatment Variables**: Don't include variables affected by treatment

### References

- Horvitz, D.G. & Thompson, D.J. (1952). "A Generalization of Sampling Without Replacement from a Finite Universe." *JASA*, 47(260), 663-685.
- Robins, J.M., Rotnitzky, A. & Zhao, L.P. (1994). "Estimation of Regression Coefficients When Some Regressors Are Not Always Observed." *JASA*, 89(427), 846-866.
- Bang, H. & Robins, J.M. (2005). "Doubly Robust Estimation in Missing Data and Causal Inference Models." *Biometrics*, 61(4), 962-973.
- Lunceford, J.K. & Davidian, M. (2004). "Stratification and Weighting Via the Propensity Score in Estimation of Causal Treatment Effects." *Statistics in Medicine*, 23, 2937-2960.

---

## Causal Mediation Analysis

Mediation analysis decomposes the total treatment effect into the portion that operates through a specific mediator variable and the portion that operates through other pathways.

### The Mediation Framework

Consider a treatment D affecting outcome Y potentially through a mediator M:

```
D ─────────────────────────────────► Y    (Direct Effect)
│                                    ▲
│           D ──► M ──► Y           │
└───────────► M ──────────────────────┘    (Indirect Effect)
```

**Total Effect (ATE)**: Overall effect of treatment on outcome
```
ATE = E[Y(1,M(1))] - E[Y(0,M(0))]
```

**Natural Direct Effect (NDE)**: Effect of treatment NOT operating through the mediator
```
NDE = E[Y(1,M(0))] - E[Y(0,M(0))]
```
This measures what would happen if we changed treatment status but kept the mediator at its control level.

**Natural Indirect Effect (NIE)**: Effect of treatment operating THROUGH the mediator
```
NIE = E[Y(1,M(1))] - E[Y(1,M(0))]
```
This measures the effect of the mediator change induced by treatment.

**Decomposition**:
```
ATE = NDE + NIE
```

**Proportion Mediated**: Fraction of total effect explained by the mediator
```
% Mediated = NIE / ATE
```

### IPW-Based Identification

prompt2analytics uses the inverse probability weighting approach of Huber (2014):

1. **Propensity Score p(D=1|X)**: Probability of treatment given covariates
2. **Extended Propensity Score p(D=1|M,X)**: Probability of treatment given mediator and covariates
3. **Reweighting**: Control observations are reweighted to have the mediator distribution that would have occurred under treatment

This approach:
- Does not require parametric models for the mediator or outcome
- Provides consistent estimates under the identification assumptions
- Works with continuous or discrete mediators

### Key Assumptions

1. **Sequential Ignorability**:
   - Treatment assignment is ignorable given covariates: (Y(d,m), M(d)) ⊥ D | X
   - Mediator is ignorable given treatment and covariates: Y(d,m) ⊥ M | D, X

2. **No Treatment-Mediator Confounding by Treatment**: No unobserved variables that are affected by treatment and affect both mediator and outcome

3. **Positivity**: All covariate values have positive probability of all treatment-mediator combinations
   - p(D=1|X) > 0 and p(D=0|X) > 0
   - p(D=1|M,X) > 0 and p(D=0|M,X) > 0

4. **Correct Propensity Models**: Logistic models for propensity scores are correctly specified

### Usage

```
mediation_analysis dataset:data outcome:wages treatment:training
    mediator:job_skills covariates:age,education trim:0.05 bootstrap:999
```

**Parameters**:
- `outcome`: Outcome variable Y
- `treatment`: Binary treatment indicator (0/1)
- `mediator`: Mediator variable M
- `covariates`: Confounders to adjust for
- `trim`: Remove observations with extreme propensity scores (default: 0.05)
- `bootstrap`: Number of bootstrap replications for inference (default: 999)

### Interpreting Output

| Statistic | Interpretation |
|-----------|----------------|
| Total Effect (ATE) | Overall causal effect of treatment on outcome |
| Direct Effect (NDE) | Effect bypassing the mediator |
| Indirect Effect (NIE) | Effect through the mediator |
| Proportion Mediated | NIE / ATE (what fraction is mediated) |
| Std Error | Bootstrap standard error |
| 95% CI | Percentile confidence interval |
| p-value | Two-sided test of H0: effect = 0 |
| n_obs | Observations used after trimming |
| n_trimmed | Observations removed due to extreme propensity scores |

### Example Interpretation

Suppose you're studying the effect of job training on wages:
- **Treatment**: Job training program (D = 1 if trained)
- **Mediator**: Job skills score (M)
- **Outcome**: Wages (Y)

Results:
```
Total Effect:     $2,500/year (p < 0.01)
Direct Effect:    $1,500/year (p < 0.05)
Indirect Effect:  $1,000/year (p < 0.05)
Proportion Med:   40%
```

**Interpretation**:
- Training increases wages by $2,500/year on average
- $1,500 (60%) comes from direct effects (e.g., signaling, network effects)
- $1,000 (40%) comes through improved job skills
- Both pathways are statistically significant

### Common Issues

1. **Extreme Propensity Scores**: If many observations are trimmed, the positivity assumption may be violated. Consider:
   - Reducing the covariate set
   - Using a different trimming threshold
   - Reconsidering the causal model

2. **Confounded Mediator**: If there are unobserved variables affecting both M and Y, the indirect effect is biased. Solutions:
   - Include additional confounders
   - Use sensitivity analysis
   - Consider instrumental variable approaches for mediation

3. **Post-Treatment Confounding**: Don't adjust for variables that are affected by treatment and affect the mediator

4. **Proportion Mediated Issues**:
   - Can be > 100% if direct and indirect effects have opposite signs
   - Unstable when total effect is small
   - Not well-defined when total effect is zero

### When to Use Mediation Analysis

| Situation | Recommended |
|-----------|-------------|
| Want to understand treatment mechanisms | Yes |
| Planning to intervene on mediator directly | Yes |
| Mediator is post-treatment variable | Yes (that's the point) |
| Strong confounding of mediator-outcome | Consider alternatives |
| Only care about total effect | Use standard treatment effects |

### References

- Huber, M. (2014). "Identifying Causal Mechanisms (Primarily) Based on Inverse Probability Weighting." *Journal of Applied Econometrics*, 29, 920-943.
- Imai, K., Keele, L., & Tingley, D. (2010). "A General Approach to Causal Mediation Analysis." *Psychological Methods*, 15(4), 309-334.
- VanderWeele, T.J. (2015). *Explanation in Causal Inference: Methods for Mediation and Interaction*. Oxford University Press.
- Pearl, J. (2001). "Direct and Indirect Effects." *Proceedings of the 17th Conference on Uncertainty in Artificial Intelligence*, 411-420.

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
| Observational treatment effect | IPW or Doubly Robust (AIPW) |
| Selection on observables | IPW (simple) or AIPW (robust) |
| Understand treatment mechanisms | Causal Mediation Analysis |
| Decompose direct/indirect effects | Causal Mediation Analysis |

---

## Further Reading

- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*
- Angrist, J.D. & Pischke, J.S. (2009). *Mostly Harmless Econometrics*
- Greene, W.H. (2018). *Econometric Analysis*
- Cameron, A.C. & Trivedi, P.K. (2005). *Microeconometrics*
