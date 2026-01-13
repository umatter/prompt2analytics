# Econometrics Guide: Assumptions and Interpretation

This guide explains the assumptions underlying each econometric method in prompt2analytics and how to interpret the results.

## Table of Contents
- [OLS Regression](#ols-regression)
- [Robust Standard Errors](#robust-standard-errors)
- [Panel Data Models](#panel-data-models)
- [High-Dimensional Fixed Effects](#high-dimensional-fixed-effects)
- [Instrumental Variables](#instrumental-variables)
- [Difference-in-Differences](#difference-in-differences)
- [Synthetic Control Method](#synthetic-control-method)
- [Regression Discontinuity Design](#regression-discontinuity-design)
- [Discrete Choice Models](#discrete-choice-models)
- [Treatment Effect Estimation](#treatment-effect-estimation)
- [Causal Mediation Analysis](#causal-mediation-analysis)
- [Survival Analysis](#survival-analysis)
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

## Synthetic Control Method

The Synthetic Control Method (SCM) is a data-driven approach for comparative case studies with a single treated unit, developed by Abadie, Diamond, and Hainmueller.

### The Problem

When studying the effect of an intervention on a single unit (state, country, firm):
- Cannot use standard treatment-control comparison
- Parallel trends assumption may be too strong
- Need a principled way to construct a counterfactual

### Solution: Synthetic Control

Construct a **weighted combination** of untreated (donor) units that best matches the treated unit's pre-treatment characteristics.

**Key Insight**: A weighted average of donors can better approximate the treated unit than any single donor.

### Mathematical Formulation

**Setup**:
- J+1 units: unit 1 is treated, units 2,...,J+1 are donors
- T time periods, treatment occurs at T₀
- Y_{jt} = outcome for unit j at time t
- X_j = predictor vector for unit j (pre-treatment characteristics)

**Optimization Problem**:

Find weights W* = (w₂, ..., w_{J+1}) that minimize:

```
||X₁ - X₀W||_V = √[(X₁ - X₀W)' V (X₁ - X₀W)]
```

Subject to:
- w_j ≥ 0 for all j (non-negative weights)
- Σw_j = 1 (weights sum to 1)

Where:
- X₁ = treated unit predictors (k × 1)
- X₀ = donor predictors (k × J)
- V = diagonal matrix of predictor importance weights

**Nested Optimization**:

1. **Outer loop**: Optimize V to minimize pre-treatment MSPE:
   ```
   V* = argmin_V Σ_{t<T₀} (Y_{1t} - Σ_j w_j*(V) Y_{jt})²
   ```

2. **Inner loop**: For given V, solve constrained QP for W:
   ```
   W*(V) = argmin_W (X₁ - X₀W)' V (X₁ - X₀W)
   ```

### Treatment Effect Estimation

The estimated treatment effect at time t > T₀ is:

```
τ_t = Y_{1t} - Σ_j w_j* Y_{jt}
     = Actual - Synthetic
```

**Average Effect**: Mean of τ_t over all post-treatment periods

**Cumulative Effect**: Sum of τ_t over all post-treatment periods

### Usage

```
synthetic_control dataset:panel outcome:gdp unit_col:country time_col:year
    treated_unit:Germany treatment_time:1990
    predictors:gdp_lag,population,trade_openness run_placebos:true
```

**Parameters**:
- `outcome`: Outcome variable
- `unit_col`: Column identifying units
- `time_col`: Column identifying time periods
- `treated_unit`: Name/ID of the treated unit
- `treatment_time`: First post-treatment period
- `predictors`: Pre-treatment characteristics to match on
- `run_placebos`: Whether to run placebo tests for inference

### Predictor Specification

Predictors can be aggregated in different ways:

| Aggregation | Description |
|-------------|-------------|
| Mean | Average over pre-treatment periods (default) |
| First | First observation |
| Last | Last observation |
| Sum | Sum over periods |

Time windows allow focusing on specific pre-treatment periods:
```
PredictorSpec::with_window("gdp", 1980, 1985)  // Mean of GDP from 1980-1985
```

### V Matrix Optimization

| Method | Description | When to Use |
|--------|-------------|-------------|
| DataDriven | Minimize pre-treatment MSPE | Default; most flexible |
| Equal | Equal weights for all predictors | When all predictors equally important |
| Custom | User-specified weights | Expert knowledge of predictor importance |

### Interpreting Output

| Statistic | Interpretation |
|-----------|----------------|
| Unit Weights | Which donors contribute to the synthetic control |
| Predictor Balance | How well synthetic matches treated on predictors |
| Pre-Treatment RMSPE | Fit quality; lower is better |
| Treatment Effects | Effect at each post-treatment period |
| Average Effect | Mean effect over all post-periods |
| Cumulative Effect | Total accumulated effect |

### Inference: Placebo Tests

Since SCM is designed for a single treated unit, standard inference doesn't apply. Instead, use **placebo tests**:

1. Apply SCM to each donor unit as if it were treated
2. Compute RMSPE ratio for each unit:
   ```
   Ratio = Post-Treatment RMSPE / Pre-Treatment RMSPE
   ```
3. Rank all units by their ratios
4. **P-value** = Rank of treated unit / Total units

**Interpretation**:
- Large ratio → Large effect relative to fit quality
- If treated unit has highest ratio → Effect is significant
- P-value = 1/N if treated has highest ratio

### Key Assumptions

1. **No Anticipation**: Treated unit doesn't change behavior before treatment

2. **No Interference (SUTVA)**: Treatment of one unit doesn't affect donors

3. **Convex Hull**: Treated unit's characteristics lie within the range of donors
   - *Violation sign*: All weight on one unit, or poor predictor balance

4. **Sufficient Pre-Treatment Fit**: Low pre-treatment RMSPE
   - *Rule of thumb*: RMSPE should be small relative to outcome scale

5. **Common Shocks**: All units affected by same aggregate shocks
   - Ensures donors provide valid counterfactual

### Diagnostics and Warnings

| Issue | Warning | Action |
|-------|---------|--------|
| High weight concentration | One donor has >90% weight | Check predictor choice |
| Poor predictor balance | Large % difference | Add/modify predictors |
| Few non-zero weights | Only 1 donor significant | May indicate extrapolation |
| Non-convergence | Max iterations reached | Increase max_iter |
| Low post/pre RMSPE ratio | Treated unit ranks low in placebo | Effect may not be significant |

### Example

**California's Tobacco Control Program** (Classic example from Abadie et al. 2010):

```
# California implemented Proposition 99 in 1988
# Outcome: Per-capita cigarette sales

synthetic_control dataset:tobacco outcome:cigsale unit_col:state time_col:year
    treated_unit:California treatment_time:1989
    predictors:cigsale,retprice,income,age15to24,beer run_placebos:true
```

**Expected results**:
- Synthetic California is ~weighted average of Utah, Nevada, Montana, etc.
- Pre-1989: Synthetic closely tracks actual California
- Post-1989: Gap opens → estimated effect of tobacco control

### Comparison: SCM vs Difference-in-Differences

| Aspect | DiD | Synthetic Control |
|--------|-----|-------------------|
| Treated units | Multiple allowed | Designed for single unit |
| Control selection | Simple comparison group | Optimized weighted average |
| Parallel trends | Assumed | Constructed via matching |
| Inference | Standard | Placebo-based |
| When to use | Multiple treated, similar trends | Single treated unit, aggregate data |

### Common Pitfalls

1. **Extrapolation**: If treated unit is outside donor range, synthetic will be poor
   - Check: All weights > 0, good predictor balance

2. **Overfitting Pre-Treatment**: Perfect pre-treatment fit may not predict well
   - Solution: Use subset of pre-periods for optimization

3. **Too Few Donors**: Need enough donors for good synthetic
   - Rule of thumb: At least 5-10 comparable donors

4. **Treatment Spillovers**: If treatment affects donors, bias results
   - Solution: Remove affected donors from pool

5. **Choosing Predictors**: Include variables that predict outcome
   - Don't include post-treatment variables
   - Include lagged outcomes (e.g., outcome in years T₀-1, T₀-2)

### References

- Abadie, A. & Gardeazabal, J. (2003). "The Economic Costs of Conflict: A Case Study of the Basque Country." *American Economic Review*, 93(1), 112-132.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program." *JASA*, 105(490), 493-505.
- Abadie, A. (2021). "Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects." *Journal of Economic Literature*, 59(2), 391-425.

---

## Regression Discontinuity Design

Regression Discontinuity Design (RDD) is a quasi-experimental method that exploits discontinuities in treatment assignment rules to estimate causal effects.

### The Setup

Treatment is assigned based on whether a **running variable** (X) crosses a **cutoff** (c):
- **Sharp RD**: Treatment deterministically assigned at cutoff: D = I(X ≥ c)
- **Fuzzy RD**: Treatment probability jumps at cutoff but not from 0 to 1

### Identification Strategy

**Key Insight**: Units just below and just above the cutoff are similar except for treatment status.

```
τ_RD = lim_{x→c⁺} E[Y|X=x] - lim_{x→c⁻} E[Y|X=x]
```

This is a **local average treatment effect** (LATE) for units at the cutoff.

### Local Polynomial Estimation

We estimate the conditional expectation on each side using **local polynomial regression**:

1. Fit weighted least squares on each side:
   ```
   min Σᵢ [yᵢ - Σⱼ βⱼ(xᵢ - c)ʲ]² K((xᵢ - c)/h)
   ```
2. The treatment effect is the difference in intercepts: τ̂ = β̂₀⁺ - β̂₀⁻

**Kernel Functions** weight observations by distance from cutoff:
- **Triangular** (default): K(u) = (1 - |u|)I(|u| ≤ 1)
- **Epanechnikov**: K(u) = 0.75(1 - u²)I(|u| ≤ 1)
- **Uniform**: K(u) = 0.5·I(|u| ≤ 1)

### Bandwidth Selection

The **bandwidth** (h) controls the window around the cutoff:
- Too small → high variance (few observations)
- Too large → high bias (smoothing over discontinuity)

**MSE-optimal bandwidth** (Imbens & Kalyanaraman 2012):
```
h_opt ∝ [σ² / (f(c) × (m''(c))²)]^(1/5) × n^(-1/5)
```

Balances squared bias against variance.

### Robust Bias-Corrected Inference

Following Calonico, Cattaneo & Titiunik (2014):

1. **Conventional estimate**: Local linear (p=1) regression
2. **Bias estimate**: Higher-order polynomial (q=p+1) with larger bandwidth
3. **Bias-corrected estimate**: τ̂_bc = τ̂ - B̂
4. **Robust standard errors**: Account for uncertainty in bias estimation

### Usage

**Sharp RD**:
```
rd_estimate dataset:mydata outcome:score running_var:age cutoff:65
    kernel:triangular bwselect:mserd
```

**Parameters**:
- `outcome`: Outcome variable (Y)
- `running_var`: Running/forcing variable (X)
- `cutoff`: Threshold value (default: 0)
- `p`: Polynomial order (1=local linear, 2=local quadratic)
- `kernel`: triangular, epanechnikov, or uniform
- `bwselect`: mserd (MSE-optimal) or msetwo (separate left/right)
- `h`: Main bandwidth (auto if not specified)
- `level`: Confidence level (default: 0.95)

**Fuzzy RD** (for imperfect compliance):
```
rd_fuzzy dataset:mydata outcome:score running_var:age treatment:enrolled
    cutoff:65
```

Returns LATE = (reduced form effect) / (first stage effect)

### Interpreting Output

| Statistic | Description |
|-----------|-------------|
| tau_conventional | Standard local polynomial estimate |
| tau_bc | Bias-corrected estimate |
| tau_robust | Robust estimate (same point, different SE) |
| se_robust | Standard error accounting for bias uncertainty |
| ci_robust | Confidence interval for robust inference |
| h_left, h_right | Bandwidths used for estimation |
| n_eff_left, n_eff_right | Effective sample sizes within bandwidth |
| p, q | Polynomial orders for estimation and bias |

**Recommended**: Use **robust** estimates and confidence intervals.

### Key Assumptions

1. **Continuity of potential outcomes** at cutoff:
   - E[Y(0)|X=x] and E[Y(1)|X=x] are continuous at c
   - No other interventions occur at the threshold

2. **No manipulation** of running variable:
   - Units cannot precisely control X to select into treatment
   - Test: Check for bunching at cutoff (McCrary test)

3. **Local randomization** (stronger):
   - Near cutoff, assignment is "as good as random"

### Validity Checks

1. **Covariate balance**: Baseline covariates should not jump at cutoff
2. **Density continuity**: No bunching/manipulation of running variable
3. **Placebo tests**: No effect at fake cutoffs
4. **Robustness**: Stability across bandwidth choices

### Common Pitfalls

| Problem | Solution |
|---------|----------|
| Too few observations near cutoff | Consider larger bandwidth or different design |
| Covariate imbalance at cutoff | Include covariates or reconsider design |
| Discrete running variable | Use local randomization inference |
| Bandwidth-sensitive results | Report range of bandwidths |
| Weak first stage (fuzzy) | Check first-stage discontinuity |

### References

- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). "Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs." *Econometrica*, 82(6), 2295-2326.
- Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2020). "Optimal Bandwidth Choice for Robust Bias Corrected Inference in RD Designs." *Econometrics Journal*, 23(2), 192-210.
- Imbens, G. & Kalyanaraman, K. (2012). "Optimal Bandwidth Choice for the Regression Discontinuity Estimator." *Review of Economic Studies*, 79(3), 933-959.
- Lee, D. S. & Lemieux, T. (2010). "Regression Discontinuity Designs in Economics." *Journal of Economic Literature*, 48(2), 281-355.

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

## Survival Analysis

Survival analysis deals with time-to-event data where the outcome is the time until an event occurs (death, failure, recovery, etc.). A key feature is **censoring**: we may not observe the event for all subjects.

### Censoring

**Right Censoring** (most common): The event hasn't occurred by the end of observation
- Subject still alive at study end
- Subject lost to follow-up
- Subject withdraws from study

**Key Notation**:
- T = true event time
- C = censoring time
- Observed: min(T, C) and event indicator δ = I(T ≤ C)

### Survival and Hazard Functions

**Survival Function**: Probability of surviving beyond time t
```
S(t) = P(T > t)
```

**Hazard Function**: Instantaneous risk of event given survival to time t
```
h(t) = lim_{Δt→0} P(t ≤ T < t+Δt | T ≥ t) / Δt
```

**Cumulative Hazard**:
```
H(t) = ∫₀ᵗ h(u) du = -log(S(t))
```

---

### Kaplan-Meier Estimator

A non-parametric estimator of the survival function.

**Product-Limit Estimator**:
```
Ŝ(t) = ∏_{tᵢ ≤ t} (1 - dᵢ/nᵢ)
```

Where:
- tᵢ = distinct event times
- dᵢ = number of events at time tᵢ
- nᵢ = number at risk just before tᵢ

**Greenwood's Formula** (variance):
```
Var(Ŝ(t)) = Ŝ(t)² × Σ_{tᵢ ≤ t} dᵢ / (nᵢ × (nᵢ - dᵢ))
```

**Confidence Intervals**: Uses log-log transformation for better coverage:
```
CI = Ŝ(t)^{exp(±z_{α/2} × SE(log(-log(Ŝ(t)))))}
```

**Usage**:
```rust
run_kaplan_meier(dataset, "time", "event", Some("group"), 0.95)
```

**Interpreting Output**:

| Statistic | Interpretation |
|-----------|----------------|
| Survival | Ŝ(t) at each distinct event time |
| Std Error | Greenwood SE at each time |
| CI Lower/Upper | 95% confidence interval |
| N at Risk | Number still at risk at each time |
| N Events | Number of events at each time |
| Median Survival | Time when Ŝ(t) = 0.5 |

**Assumptions**:
1. **Independent censoring**: Censoring is unrelated to event risk
2. **No informative censoring**: Censored subjects have same prognosis as those remaining

---

### Log-Rank Test

Compares survival curves between two or more groups.

**Null Hypothesis**: H₀: S₁(t) = S₂(t) = ... = Sₖ(t) for all t

**Test Statistic**:
```
χ² = U'V⁻¹U
```

Where:
- U = observed - expected events per group
- V = variance-covariance matrix

For two groups:
```
χ² = (Σ(O₁ᵢ - E₁ᵢ))² / Σ Var(O₁ᵢ)
```

**Usage**:
```rust
log_rank_test(dataset, "time", "event", "treatment_group")
```

**Interpreting Output**:

| Statistic | Interpretation |
|-----------|----------------|
| Chi-squared | Test statistic |
| df | Degrees of freedom (k-1 groups) |
| p-value | P(χ² > observed) under H₀ |

**Interpretation**:
- p < 0.05: Significant difference in survival between groups
- Large χ²: Greater departure from equal survival

**Limitations**:
- Sensitive to differences in middle of distribution
- May miss differences if curves cross
- Assumes proportional hazards

---

### Cox Proportional Hazards Model

A semi-parametric regression model relating covariates to hazard.

**Model**:
```
h(t|X) = h₀(t) × exp(β'X)
```

Where:
- h₀(t) = baseline hazard (unspecified)
- exp(β'X) = relative risk
- β = regression coefficients

**Partial Likelihood** (Cox, 1972):
```
L(β) = ∏ᵢ: δᵢ=1 [exp(β'xᵢ) / Σⱼ∈R(tᵢ) exp(β'xⱼ)]
```

Where R(tᵢ) is the risk set at time tᵢ.

**Tie Handling**:

| Method | Description | When to Use |
|--------|-------------|-------------|
| Breslow | Approximate, faster | Few ties (default) |
| Efron | More accurate | Many ties |

**Usage**:
```rust
let config = CoxConfig {
    ties: TiesMethod::Breslow,
    max_iter: 25,
    tolerance: 1e-9,
    robust_se: false,
};
run_cox_ph(dataset, "time", "event", &["age", "treatment"], Some(config))
```

**Interpreting Output**:

| Statistic | Interpretation |
|-----------|----------------|
| Coefficient (β) | Log hazard ratio |
| Hazard Ratio (exp(β)) | Multiplicative change in hazard per unit increase |
| SE | Standard error of β |
| z-stat | β / SE |
| p-value | Two-sided test of H₀: β = 0 |
| HR CI | 95% CI for hazard ratio |
| Concordance | C-index; probability that predictions agree with observed ordering |
| Wald/Score/LR tests | Overall model significance tests |

**Example**:
- β = 0.5 → HR = exp(0.5) = 1.65
- Interpretation: One-unit increase in X multiplies hazard by 1.65 (65% higher risk)

**Key Assumptions**:

1. **Proportional Hazards**: Hazard ratios are constant over time
   ```
   h(t|X=1) / h(t|X=0) = exp(β) for all t
   ```
   - *Test*: Schoenfeld residuals, log-log plots
   - *Violation*: Time-varying coefficients, stratification

2. **Log-linear relationship**: log(h) is linear in X
   - *Violation*: Transform covariates, use splines

3. **Independent censoring**: Censoring unrelated to covariates

**Diagnostics**:
- **Concordance (C-index)**: 0.5 = random, 1.0 = perfect discrimination
  - C > 0.7 is generally good
- **Likelihood Ratio Test**: Compares model to null
- **Wald Test**: Tests all coefficients = 0

---

### Accelerated Failure Time (AFT) Models

Parametric survival models where covariates accelerate or decelerate time to event.

**Model**:
```
log(T) = μ + β'X + σε
```

Where ε follows a specified distribution.

**Interpretation**:
- **Acceleration Factor**: exp(β)
- exp(β) > 1: Longer survival (deceleration)
- exp(β) < 1: Shorter survival (acceleration)

**Available Distributions**:

| Distribution | ε Distribution | Hazard Shape |
|--------------|----------------|--------------|
| Exponential | Gumbel (min) | Constant |
| Weibull | Gumbel (min) | Monotonic increasing/decreasing |
| Log-Normal | Normal | Non-monotonic (peak then decline) |
| Log-Logistic | Logistic | Non-monotonic |

**Usage**:
```rust
let config = AftConfig {
    distribution: AftDistribution::Weibull,
    max_iter: 100,
    tolerance: 1e-8,
};
run_aft(dataset, "time", "event", &["age", "treatment"], Some(config))
```

**Interpreting Output**:

| Statistic | Interpretation |
|-----------|----------------|
| Coefficient (β) | Effect on log(time) |
| Acceleration Factor | exp(β); multiplier on survival time |
| Scale (σ) | Dispersion parameter |
| Shape | Distribution shape parameter |
| Log-Likelihood | Model fit |
| AIC/BIC | Model comparison (lower = better) |

**Example** (Weibull with β = 0.3):
- Acceleration factor = exp(0.3) = 1.35
- Expected survival time multiplied by 1.35 (35% longer survival)

**Choosing Distribution**:
- **Exponential**: Constant hazard (memoryless)
- **Weibull**: Monotonic hazard (aging effects)
- **Log-Normal**: Hazard increases then decreases
- **Log-Logistic**: Similar to Log-Normal, heavier tails

**Model Comparison**: Use AIC/BIC to select distribution
```
AIC = -2 × log(L) + 2k
BIC = -2 × log(L) + k × log(n)
```

---

### Competing Risks (Aalen-Johansen)

When subjects can experience one of several mutually exclusive event types.

**Example**: Studying time to death where cause can be:
- Event type 1: Cardiovascular disease
- Event type 2: Cancer
- Event type 0: Censored (still alive)

**Cumulative Incidence Function (CIF)**:
```
F̂ₖ(t) = Σ_{tᵢ ≤ t} Ŝ(tᵢ₋₁) × dₖᵢ / nᵢ
```

Where:
- Ŝ(t) = Kaplan-Meier for all-cause survival
- dₖᵢ = events of type k at time tᵢ

**Key Property**: ΣF̂ₖ(t) ≤ 1 - Ŝ(t)

**Usage**:
```rust
run_competing_risks(dataset, "time", "event_type", 0.95)
// event_type: 0 = censored, 1, 2, ... = event types
```

**Interpreting Output**:

| Statistic | Interpretation |
|-----------|----------------|
| Cumulative Incidence | Probability of event k by time t |
| SE | Standard error of CIF |
| CI | Confidence interval for CIF |
| N Events by Type | Number of each event type |

**Why Not Use Kaplan-Meier?**

Standard KM treats competing events as censoring, which:
1. Overestimates the probability of the event of interest
2. Assumes competing events are independent (often false)
3. CIFs don't sum to 1 - S(t)

**CIF Interpretation**:
- CIF₁(5) = 0.15 means: 15% probability of event type 1 by year 5
- Accounts for the "competing" nature of other event types

---

### Quick Reference: Survival Methods

| Situation | Recommended Method |
|-----------|-------------------|
| Single group survival curve | Kaplan-Meier |
| Compare survival between groups | Log-rank test |
| Adjust for covariates | Cox PH (default) |
| Non-proportional hazards | Stratified Cox, AFT |
| Parametric survival model | AFT (Weibull, etc.) |
| Multiple event types | Competing Risks (Aalen-Johansen) |
| Time-varying covariates | Extended Cox model |

### Assumptions Summary

| Method | Key Assumptions |
|--------|-----------------|
| Kaplan-Meier | Independent censoring |
| Log-rank | Proportional hazards |
| Cox PH | Proportional hazards, log-linear, independent censoring |
| AFT | Correct distribution, independent censoring |
| Competing Risks | Independent censoring |

### References

- Cox, D.R. (1972). "Regression Models and Life Tables." *JRSS B*, 34:187-220.
- Kaplan, E.L. & Meier, P. (1958). "Nonparametric Estimation from Incomplete Observations." *JASA*, 53:457-481.
- Aalen, O.O. & Johansen, S. (1978). "An Empirical Transition Matrix." *Scandinavian J. Statistics*, 5:141-150.
- Klein, J.P. & Moeschberger, M.L. (2003). *Survival Analysis: Techniques for Censored and Truncated Data*. Springer.
- Therneau, T.M. & Grambsch, P.M. (2000). *Modeling Survival Data*. Springer.

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
| Single treated unit, case study | Synthetic Control Method |
| Binary outcome | Logit or Probit |
| Clustered data | Clustered SEs |
| Time series correlation | Newey-West SEs |
| Observational treatment effect | IPW or Doubly Robust (AIPW) |
| Selection on observables | IPW (simple) or AIPW (robust) |
| Understand treatment mechanisms | Causal Mediation Analysis |
| Decompose direct/indirect effects | Causal Mediation Analysis |
| Time-to-event (survival curve) | Kaplan-Meier |
| Compare survival between groups | Log-Rank Test |
| Survival with covariates | Cox Proportional Hazards |
| Parametric survival model | AFT (Weibull, Log-Normal, etc.) |
| Multiple competing event types | Competing Risks (Aalen-Johansen) |

---

## Further Reading

- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*
- Angrist, J.D. & Pischke, J.S. (2009). *Mostly Harmless Econometrics*
- Greene, W.H. (2018). *Econometric Analysis*
- Cameron, A.C. & Trivedi, P.K. (2005). *Microeconometrics*
