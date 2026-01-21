# MCP Tool Usage Examples

This guide provides usage examples for all 55 MCP tools available in prompt2analytics.

## Table of Contents
- [Data Management](#data-management)
- [Descriptive Statistics](#descriptive-statistics)
- [ANOVA (Analysis of Variance)](#anova-analysis-of-variance)
- [Regression Analysis](#regression-analysis)
- [Panel Data](#panel-data)
- [Instrumental Variables](#instrumental-variables)
- [Causal Inference](#causal-inference)
- [Survival Analysis](#survival-analysis)
- [Discrete Choice Models](#discrete-choice-models)
- [Time Series](#time-series)
- [Multivariate Analysis](#multivariate-analysis)
- [Machine Learning](#machine-learning)
- [Database Queries](#database-queries)
- [Visualization](#visualization)
- [Utilities](#utilities)

---

## Data Management

### load_dataset
Load a dataset from a file (CSV, Parquet, Excel, Stata, SAS).

```
/load_dataset path:/path/to/data.csv
/load_dataset path:/data/survey.xlsx
/load_dataset path:/data/panel.dta
```

**Parameters:**
- `path` (required): File path to load

**Supported formats:**
- CSV (.csv)
- Parquet (.parquet)
- Excel (.xlsx, .xls)
- Stata (.dta) - versions 117-119
- SAS (.sas7bdat)

### list_datasets
Show all currently loaded datasets.

```
/list_datasets
```

### describe_dataset
Get summary statistics for a dataset.

```
/describe_dataset dataset:mydata
```

**Output includes:**
- Row and column counts
- For numeric columns: count, mean, std, min, 25%, 50%, 75%, max
- For categorical columns: unique values, top values

### head_dataset
Preview the first N rows of a dataset.

```
/head_dataset dataset:mydata n:10
```

---

## Descriptive Statistics

### compute_correlation
Calculate the Pearson correlation matrix.

```
/compute_correlation dataset:mydata
/compute_correlation dataset:mydata columns:price,sqft,bedrooms
```

**Parameters:**
- `dataset` (required): Dataset name
- `columns` (optional): Specific columns to include

---

## ANOVA (Analysis of Variance)

### anova_one_way
Test whether means differ across groups (one categorical factor).

```
/anova_one_way dataset:experiment response:yield factor:treatment
```

**Parameters:**
- `dataset` (required): Dataset name
- `response` (required): Numeric response variable column
- `factor` (required): Categorical grouping variable column

**Output includes:**
- ANOVA table (SS, DF, MS, F, p-value)
- Effect sizes (η², ω²)
- Group statistics (n, mean, std for each group)
- Grand mean

**Example use cases:**
- Compare treatment effects across groups
- Test if regional sales differ significantly
- Analyze experimental results

### anova_two_way
Test effects of two factors and their interaction.

```
# With interaction (factorial design)
/anova_two_way dataset:experiment response:yield factor_a:fertilizer factor_b:irrigation interaction:true

# Additive model (no interaction)
/anova_two_way dataset:experiment response:yield factor_a:fertilizer factor_b:irrigation interaction:false
```

**Parameters:**
- `dataset` (required): Dataset name
- `response` (required): Numeric response variable column
- `factor_a` (required): First categorical factor column
- `factor_b` (required): Second categorical factor column
- `interaction` (optional): Include interaction term (default: true)

**Output includes:**
- ANOVA table with F-tests for each factor and interaction
- Degrees of freedom for each term
- Significance levels

**Example use cases:**
- 2×2 factorial experiments
- Testing main effects and interactions in experimental designs
- Analyzing treatment × covariate effects

---

## Hypothesis Testing

### hypothesis_t_test
Run Student's or Welch's t-test for comparing means.

**One-sample test** (compare to hypothesized mean):
```
/hypothesis_t_test dataset:data x:weight mu:70
```

**Two-sample test** (compare two groups, Welch's by default):
```
/hypothesis_t_test dataset:data x:treatment y:control
```

**Paired test** (matched pairs):
```
/hypothesis_t_test dataset:data x:before y:after paired:true
```

**One-sided test:**
```
/hypothesis_t_test dataset:data x:scores mu:50 alternative:greater
```

**Parameters:**
- `dataset` (required): Dataset name
- `x` (required): First variable column (numeric)
- `y` (optional): Second variable column for two-sample/paired tests
- `mu` (optional): Null hypothesis value (default: 0)
- `alternative` (optional): `two.sided` (default), `greater`, or `less`
- `paired` (optional): If `true`, perform paired t-test (default: `false`)
- `var_equal` (optional): If `true`, use Student's t-test with pooled variance (default: `false`, Welch's)
- `conf_level` (optional): Confidence level (default: 0.95)

**Output includes:**
- t-statistic and degrees of freedom
- p-value with significance stars
- Confidence interval
- Sample mean(s) or difference

**Example use cases:**
- Test if a sample mean differs from a target value
- Compare treatment vs control groups
- Analyze before/after measurements
- A/B testing with continuous outcomes

### hypothesis_wilcoxon
Run Wilcoxon non-parametric test for location. Does not assume normality.

**Two-sample rank sum test** (Mann-Whitney U, compare distributions):
```
/hypothesis_wilcoxon dataset:data x:treatment y:control
```

**One-sample signed rank test** (test median against value):
```
/hypothesis_wilcoxon dataset:data x:measurements mu:50
```

**Paired signed rank test** (matched pairs):
```
/hypothesis_wilcoxon dataset:data x:before y:after paired:true
```

**With confidence interval:**
```
/hypothesis_wilcoxon dataset:data x:group1 y:group2 conf_int:true
```

**Parameters:**
- `dataset` (required): Dataset name
- `x` (required): First variable column (numeric)
- `y` (optional): Second variable column for two-sample/paired tests
- `mu` (optional): Hypothesized location shift or median (default: 0)
- `alternative` (optional): `two.sided` (default), `greater`, or `less`
- `paired` (optional): If `true`, perform paired signed rank test (default: `false`)
- `exact` (optional): If `true`, compute exact p-value (auto-decides if omitted)
- `correct` (optional): Apply continuity correction (default: `true`)
- `conf_int` (optional): Compute confidence interval (default: `false`)
- `conf_level` (optional): Confidence level (default: 0.95)

### hypothesis_shapiro_wilk
Run the Shapiro-Wilk test for normality. Tests whether data comes from a normally distributed population.

**Basic usage:**
```
/hypothesis_shapiro_wilk dataset:data column:values
```

**Test regression residuals:**
```
/hypothesis_shapiro_wilk dataset:model_output column:residuals
```

**Parameters:**
- `dataset` (required): Dataset name
- `column` (required): Numeric column to test for normality

**Output:**
- `w_statistic`: Test statistic (0-1, values close to 1 indicate normality)
- `p_value`: P-value for the test
- `n`: Sample size
- `reject_normality`: Boolean indicating whether to reject normality at α=0.05
- `interpretation`: Human-readable conclusion

**Example Output:**
```json
{
  "test": "Shapiro-Wilk Normality Test",
  "w_statistic": 0.9843,
  "p_value": 0.5621,
  "significance": "",
  "n": 50,
  "reject_normality": false,
  "interpretation": "No evidence against normality (fail to reject H₀ at α = 0.05)"
}
```

**Notes:**
- Sample size must be between 3 and 5000
- Small p-values (< 0.05) suggest data is not normally distributed
- Use before parametric tests (t-test, ANOVA) to verify assumptions

**Output includes:**
- W (rank sum) or V (signed rank) statistic
- Mann-Whitney U statistic (for rank sum test)
- p-value with significance stars
- Z-score (for normal approximation)
- Hodges-Lehmann estimate (if `conf_int:true`)
- Confidence interval (if `conf_int:true`)

**Example use cases:**
- Compare distributions when normality is questionable
- Analyze ordinal or non-normal continuous data
- Robust alternative to t-test
- Compare two groups with outliers

### hypothesis_chisq_gof
Run Pearson's chi-squared goodness-of-fit test for categorical data.

**Test against uniform distribution:**
```
/hypothesis_chisq_gof dataset:data column:category
```

**Test against specific probabilities:**
```
/hypothesis_chisq_gof dataset:data column:blood_type probs:[0.44,0.42,0.10,0.04]
```

**Parameters:**
- `dataset` (required): Dataset name
- `column` (required): Categorical column to count
- `probs` (optional): Expected probabilities (must sum to 1.0; uniform if omitted)

**Output includes:**
- Chi-squared statistic and degrees of freedom
- p-value with significance level
- Observed and expected counts
- Pearson residuals

**Example use cases:**
- Test if a die is fair
- Check if survey responses match expected demographics
- Validate random number generator output

### hypothesis_chisq_independence
Run Pearson's chi-squared test of independence for two categorical variables.

```
/hypothesis_chisq_independence dataset:survey row_var:gender col_var:party
```

**With Yates' correction disabled (for 2×2 tables):**
```
/hypothesis_chisq_independence dataset:data row_var:treatment col_var:outcome correct:false
```

**Parameters:**
- `dataset` (required): Dataset name
- `row_var` (required): Row variable for contingency table
- `col_var` (required): Column variable for contingency table
- `correct` (optional): Apply Yates' correction for 2×2 tables (default: `true`)

**Output includes:**
- Chi-squared statistic and degrees of freedom
- p-value with significance level
- Table dimensions
- Expected values and residuals
- Standardized residuals (for identifying associations)

**Example use cases:**
- Test if party preference is independent of gender
- Check association between treatment and outcome
- Analyze categorical survey data

### hypothesis_fisher_exact
Run Fisher's exact test for 2×2 contingency tables. More accurate than chi-squared for small samples.

```
/hypothesis_fisher_exact dataset:clinical row_var:treatment col_var:outcome
```

**With one-sided alternative:**
```
/hypothesis_fisher_exact dataset:data row_var:exposed col_var:disease alternative:greater
```

**With confidence interval for odds ratio:**
```
/hypothesis_fisher_exact dataset:data row_var:group col_var:response alternative:two.sided conf_level:0.95
```

**Parameters:**
- `dataset` (required): Dataset name
- `row_var` (required): Row variable (must have exactly 2 unique values)
- `col_var` (required): Column variable (must have exactly 2 unique values)
- `alternative` (optional): `"two.sided"` (default), `"greater"`, or `"less"`
- `conf_level` (optional): Confidence level for odds ratio CI (e.g., `0.95`)

**Output includes:**
- p-value with significance level
- Sample odds ratio
- Optional: Confidence interval for odds ratio
- Contingency table with marginals

**Example use cases:**
- Test treatment effect with small samples (n < 20)
- Clinical trials with rare events
- When chi-squared approximation is unreliable (expected counts < 5)
- Exact hypothesis testing for 2×2 tables

**When to use Fisher vs Chi-Squared:**
- Use Fisher when: small samples, any expected cell < 5, need exact p-values
- Use chi-squared when: larger samples, r×c tables (not just 2×2), quick approximation is sufficient

---

## Regression Analysis

### regression_ols
Run Ordinary Least Squares regression with robust standard errors.

```
/regression_ols dataset:housing y:price x:sqft,bedrooms,bathrooms
```

**Parameters:**
- `dataset` (required): Dataset name
- `y` (required): Dependent variable column
- `x` (required): Comma-separated independent variable columns

**Output includes:**
- Coefficients with standard errors, t-values, p-values
- R-squared and adjusted R-squared
- F-statistic and p-value
- Significance stars (*, **, ***)

### regression_diagnostics
Run diagnostic tests on a regression model.

```
/regression_diagnostics dataset:housing y:price x:sqft,bedrooms
```

**Tests performed:**
- **Jarque-Bera**: Normality of residuals
- **Breusch-Pagan**: Heteroskedasticity
- **Durbin-Watson**: Autocorrelation
- **VIF**: Multicollinearity for each variable
- **Condition Number**: Overall multicollinearity

### regression_clustered
OLS with clustered standard errors.

```
/regression_clustered dataset:panel y:outcome x:treatment cluster1:firm
/regression_clustered dataset:panel y:outcome x:treatment cluster1:firm cluster2:year
```

**Parameters:**
- `cluster1` (required): First clustering variable
- `cluster2` (optional): Second clustering variable for two-way clustering

### regression_nls
Fit a nonlinear regression model using Levenberg-Marquardt.

```
/regression_nls dataset:kinetics y:velocity x:substrate model:michaelis_menten start:[200,0.1]
/regression_nls dataset:decay y:concentration x:time model:exponential_decay start:[10,0.5,2]
```

**Parameters:**
- `model` (required): Model type - one of:
  - `exponential_decay`: y = a·exp(-bx) + c → start: [a, b, c]
  - `exponential_growth`: y = a·exp(bx) → start: [a, b]
  - `michaelis_menten`: y = Vmax·x/(Km+x) → start: [Vmax, Km]
  - `logistic`: y = K/(1+exp(-r(x-x₀))) → start: [K, r, x₀]
  - `power`: y = a·x^b → start: [a, b]
  - `asymptotic`: y = a - b·exp(-cx) → start: [a, b, c]
- `start` (required): Initial parameter values as JSON array
- `algorithm` (optional): `levenberg_marquardt` (default) or `gauss_newton`
- `max_iter` (optional): Maximum iterations (default: 200)

**Output includes:**
- Parameter estimates with standard errors, t-values, p-values
- Residual sum of squares (RSS)
- Residual standard error (sigma)
- Convergence status and iteration count

**Example - Enzyme Kinetics:**
```
/regression_nls dataset:enzyme y:rate x:concentration model:michaelis_menten start:[200,0.1]
```

**Example - Drug Elimination:**
```
/regression_nls dataset:pharma y:plasma_conc x:time model:exponential_decay start:[100,0.5,5]
```

### regression_loess
Fit a LOESS (local polynomial regression) smoothing model.

```
/regression_loess dataset:sales y:revenue x:time span:0.5
/regression_loess dataset:weather y:temperature x:day span:0.3 degree:2 robust:true
```

**Parameters:**
- `span` (optional): Smoothing parameter (default: 0.75). Range (0,1] uses proportion of data; >1 uses all points. Smaller = more wiggly, larger = smoother.
- `degree` (optional): Local polynomial degree - 1 (linear) or 2 (quadratic, default)
- `robust` (optional): Use robust fitting with iterative reweighting (default: false). Set true to downweight outliers.

**Output includes:**
- Fitted values at each x point
- Residuals (y - fitted)
- Equivalent number of parameters (ENP)
- Residual standard error
- R-squared

**Example - Trend Smoothing:**
```
/regression_loess dataset:monthly_sales y:revenue x:month span:0.4 degree:2
```

**Example - Robust Smoothing with Outliers:**
```
/regression_loess dataset:sensor_data y:reading x:timestamp span:0.3 robust:true
```

**Interpretation:**
- LOESS captures nonlinear trends without assuming a specific functional form
- Fitted values represent the smooth underlying trend
- ENP indicates model complexity (higher = more flexible)
- For prediction, use the fitted values at original x points

---

## Panel Data

### panel_fixed_effects
Fixed Effects (within) estimation for panel data.

```
/panel_fixed_effects dataset:panel y:wage x:education,experience entity_col:person_id
```

**Parameters:**
- `entity_col` (required): Column identifying panel entities (firms, individuals, etc.)

**Interpretation:**
- Coefficients represent within-entity effects
- Controls for time-invariant unobserved heterogeneity

### panel_random_effects
Random Effects (GLS) estimation for panel data.

```
/panel_random_effects dataset:panel y:wage x:education,experience entity_col:person_id
```

**When to use:**
- When unobserved effects are uncorrelated with regressors
- More efficient than FE if RE assumptions hold

### hausman_test
Specification test to choose between Fixed and Random Effects.

```
/hausman_test dataset:panel y:wage x:education,experience entity_col:person_id
```

**Interpretation:**
- H0: Random Effects is consistent (use RE)
- H1: Random Effects is inconsistent (use FE)
- Reject H0 if p-value < 0.05 → use Fixed Effects

---

## Instrumental Variables

### iv_2sls
Two-Stage Least Squares for endogeneity.

```
/iv_2sls dataset:data y:earnings x_exog:education x_endog:work_experience instruments:distance_to_college
```

**Parameters:**
- `x_exog`: Exogenous regressors (not endogenous)
- `x_endog`: Endogenous regressors (correlated with error)
- `instruments`: Instruments for endogenous variables
- `robust` (optional): Use robust standard errors

**Requirements:**
- Number of instruments ≥ number of endogenous variables
- Instruments must be correlated with endogenous variables (relevance)
- Instruments must be uncorrelated with error (exclusion)

### iv_first_stage
First-stage diagnostics for instrumental variables.

```
/iv_first_stage dataset:data endogenous:work_experience instruments:distance_to_college
```

**Output includes:**
- First-stage F-statistic (rule of thumb: F > 10 for strong instruments)
- R-squared of first-stage regression
- Instrument strength assessment

---

## Causal Inference

### diff_in_diff
Difference-in-Differences estimation.

```
/diff_in_diff dataset:policy y:outcome treatment:treated post:after_policy
```

**Parameters:**
- `treatment`: Binary indicator (1 = treated group, 0 = control)
- `post`: Binary indicator (1 = after treatment, 0 = before)
- `controls` (optional): Additional control variables

**Output includes:**
- ATT (Average Treatment Effect on Treated)
- Standard error and p-value
- Group means (control pre, control post, treated pre, treated post)

**Interpretation:**
- ATT = (Treated_Post - Treated_Pre) - (Control_Post - Control_Pre)
- Assumes parallel trends: without treatment, both groups would have same trend

---

## Survival Analysis

Survival analysis tools handle time-to-event data with censoring.

### kaplan_meier
Estimate non-parametric survival curves.

```
/kaplan_meier dataset:patients time:survival_time event:death
/kaplan_meier dataset:patients time:survival_time event:death strata:treatment_group
/kaplan_meier dataset:patients time:survival_time event:death confidence_level:0.95
```

**Parameters:**
- `dataset` (required): Dataset name
- `time` (required): Column with survival times
- `event` (required): Binary event indicator (1=event, 0=censored)
- `strata` (optional): Column for group stratification
- `confidence_level` (optional): CI level (default: 0.95)

**Output includes:**
- Survival probabilities at each event time
- Greenwood standard errors
- Confidence intervals (log-log transformed)
- Median survival time
- Number at risk at each time point

### log_rank_test
Compare survival curves between groups.

```
/log_rank_test dataset:trial time:days event:death group:treatment
```

**Parameters:**
- `dataset` (required): Dataset name
- `time` (required): Column with survival times
- `event` (required): Binary event indicator (1=event, 0=censored)
- `group` (required): Column defining groups to compare

**Output includes:**
- Chi-squared statistic
- Degrees of freedom
- P-value
- Observed vs expected events per group

### cox_ph
Fit Cox Proportional Hazards regression model.

```
/cox_ph dataset:patients time:survival_time event:death x:age,stage,treatment
/cox_ph dataset:patients time:survival_time event:death x:age,stage ties:efron
/cox_ph dataset:patients time:survival_time event:death x:age,stage robust_se:true
```

**Parameters:**
- `dataset` (required): Dataset name
- `time` (required): Column with survival times
- `event` (required): Binary event indicator (1=event, 0=censored)
- `x` (required): Comma-separated covariate columns
- `ties` (optional): Tie handling method (`breslow` or `efron`, default: breslow)
- `robust_se` (optional): Use robust standard errors (default: false)
- `max_iter` (optional): Maximum Newton-Raphson iterations (default: 25)
- `tolerance` (optional): Convergence tolerance (default: 1e-9)

**Output includes:**
- Coefficients (log hazard ratios)
- Hazard ratios with 95% CI
- Standard errors, z-statistics, p-values
- Concordance (C-index)
- Wald, Score, and Likelihood Ratio tests

**Interpretation:**
- HR > 1: Higher risk of event
- HR < 1: Lower risk of event (protective)
- HR = exp(coefficient)

### aft
Fit Accelerated Failure Time parametric survival model.

```
/aft dataset:patients time:survival_time event:death x:age,treatment dist:weibull
/aft dataset:patients time:survival_time event:death x:age,treatment dist:lognormal
/aft dataset:patients time:survival_time event:death x:age,treatment dist:exponential
/aft dataset:patients time:survival_time event:death x:age,treatment dist:loglogistic
```

**Parameters:**
- `dataset` (required): Dataset name
- `time` (required): Column with survival times
- `event` (required): Binary event indicator (1=event, 0=censored)
- `x` (required): Comma-separated covariate columns
- `dist` (optional): Distribution (`weibull`, `lognormal`, `exponential`, `loglogistic`; default: weibull)
- `max_iter` (optional): Maximum iterations (default: 100)
- `tolerance` (optional): Convergence tolerance (default: 1e-8)

**Output includes:**
- Coefficients for log(survival time)
- Acceleration factors (exp(coefficient))
- Scale and shape parameters
- Standard errors, z-statistics, p-values
- Log-likelihood, AIC, BIC

**Interpretation:**
- Acceleration factor > 1: Longer survival (slower time to event)
- Acceleration factor < 1: Shorter survival (faster time to event)

### competing_risks
Estimate cumulative incidence functions with competing events.

```
/competing_risks dataset:patients time:time event_type:cause_of_death
/competing_risks dataset:patients time:time event_type:cause_of_death confidence_level:0.95
```

**Parameters:**
- `dataset` (required): Dataset name
- `time` (required): Column with event/censoring times
- `event_type` (required): Column with event type (0=censored, 1=event type 1, 2=event type 2, etc.)
- `confidence_level` (optional): CI level (default: 0.95)

**Output includes:**
- Cumulative incidence functions for each event type
- Standard errors and confidence intervals
- Number of events by type

**Use case:**
When subjects can experience multiple types of events (e.g., death from cancer vs. death from other causes), standard Kaplan-Meier overestimates event probabilities. Competing risks properly accounts for the mutually exclusive nature of different event types.

---

## Discrete Choice Models

### logit
Logistic regression for binary outcomes.

```
/logit dataset:survey y:purchased x:income,age,gender
```

**Requirements:**
- `y` must be binary (0/1)

**Output includes:**
- Coefficients (log-odds)
- McFadden's Pseudo R-squared
- Marginal effects at means (optional)

**Interpretation:**
- Coefficient β: one-unit increase in x increases log-odds by β
- Odds ratio: exp(β)

### probit
Probit regression for binary outcomes.

```
/probit dataset:survey y:purchased x:income,age,gender
```

**Similar to logit but uses normal CDF instead of logistic CDF.**

---

## Time Series

### ts_arima_fit
Fit an ARIMA(p,d,q) model.

```
/ts_arima_fit dataset:sales column:revenue p:1 d:1 q:1
```

**Parameters:**
- `p`: Autoregressive order
- `d`: Differencing order
- `q`: Moving average order

### ts_arima_forecast
Generate forecasts from a fitted ARIMA model.

```
/ts_arima_forecast dataset:sales column:revenue p:1 d:1 q:1 h:12
```

**Parameters:**
- `h`: Forecast horizon (number of periods ahead)

### ts_mstl
MSTL seasonal-trend decomposition.

```
/ts_mstl dataset:sales column:revenue periods:12
```

**Output includes:**
- Trend component
- Seasonal component(s)
- Remainder (irregular)

### ts_var
Vector Autoregression for multivariate time series.

```
/ts_var dataset:macro columns:gdp,inflation,unemployment lags:2
```

**Parameters:**
- `columns`: Multiple time series columns
- `lags`: Number of lags in the VAR

### ts_varma
VARMA(p,q) model via Hannan-Rissanen.

```
/ts_varma dataset:macro columns:gdp,inflation p:1 q:1
```

### ts_vecm
Vector Error Correction Model for cointegrated series.

```
/ts_vecm dataset:macro columns:gdp,consumption,investment lags:2 rank:1
```

**Parameters:**
- `rank`: Cointegration rank (number of cointegrating relationships)

### ts_var_irf
Impulse Response Functions from a VAR model.

```
/ts_var_irf dataset:macro columns:gdp,inflation,unemployment lags:2 horizons:20 shock:inflation
```

**Parameters:**
- `horizons`: Number of periods to trace the response
- `shock`: Variable receiving the shock

### ts_changepoint
Detect structural breaks in time series.

```
/ts_changepoint dataset:sales column:revenue method:pelt
```

**Methods:**
- `pelt`: Pruned Exact Linear Time (faster)
- `binseg`: Binary Segmentation

### ts_holt_winters
Holt-Winters exponential smoothing for time series with trend and seasonality.

```
# Fit model with automatic parameter optimization
/ts_holt_winters dataset:sales column:revenue period:12

# Multiplicative seasonality with forecasting
/ts_holt_winters dataset:sales column:revenue period:12 seasonal:multiplicative horizon:6

# Specify smoothing parameters manually
/ts_holt_winters dataset:sales column:revenue period:4 alpha:0.2 beta:0.1 gamma:0.3
```

**Parameters:**
- `dataset` (required): Dataset name
- `column` (required): Time series column
- `period` (required): Seasonal period (e.g., 12 for monthly, 4 for quarterly)
- `seasonal` (optional): `additive` (default) or `multiplicative`
- `alpha` (optional): Level smoothing (0-1), optimized if not specified
- `beta` (optional): Trend smoothing (0-1), optimized if not specified
- `gamma` (optional): Seasonal smoothing (0-1), optimized if not specified
- `horizon` (optional): Number of periods to forecast ahead

**Output includes:**
- Optimized smoothing parameters (α, β, γ)
- Sum of squared errors (SSE)
- Final level, trend, and seasonal coefficients
- Fitted values and residuals
- Forecasts (if horizon specified)

**Use cases:**
- Demand forecasting with seasonal patterns
- Sales projections
- Capacity planning
- Budget forecasting

**Notes:**
- Requires at least 2 full seasonal cycles (2×period observations)
- Multiplicative seasonality requires all positive values
- For non-seasonal data, consider ARIMA instead

### timeseries_acf
Compute autocorrelation (ACF) or partial autocorrelation (PACF) functions.

```
# Autocorrelation function
/timeseries_acf dataset:data column:returns lag_max:20 acf_type:correlation

# Autocovariance function (unnormalized)
/timeseries_acf dataset:data column:returns lag_max:20 acf_type:covariance

# Partial autocorrelation function
/timeseries_acf dataset:data column:returns lag_max:20 acf_type:partial
```

**Parameters:**
- `dataset` (required): Dataset name
- `column` (required): Time series column
- `lag_max` (optional): Maximum lag (default: min(10×log₁₀(n), n-1))
- `acf_type` (optional): `correlation` (default), `covariance`, or `partial`

**Output includes:**
- ACF/PACF values at each lag
- 95% confidence bounds (white noise assumption)
- For PACF: Uses Durbin-Levinson algorithm

**Applications:**
- Identify AR/MA orders for ARIMA modeling
- Check residuals for autocorrelation
- Detect seasonality patterns

### timeseries_ccf
Compute cross-correlation function between two time series.

```
/timeseries_ccf dataset:data x:gdp y:unemployment lag_max:10
```

**Parameters:**
- `dataset` (required): Dataset name
- `x` (required): First time series column
- `y` (required): Second time series column
- `lag_max` (optional): Maximum lag in both directions (default: min(10×log₁₀(n), n-1))

**Output includes:**
- CCF values at lags from -lag_max to +lag_max
- Lag 0 equals Pearson correlation between x and y
- 95% confidence bounds

**Applications:**
- Identify lead-lag relationships between variables
- Determine if one series predicts another
- Transfer function modeling

### timeseries_spectrum
Estimate the power spectral density of a time series using periodogram or AR-based methods.

```
# Raw periodogram (no smoothing)
/timeseries_spectrum dataset:data column:returns

# Smoothed periodogram with Daniell kernel
/timeseries_spectrum dataset:data column:returns spans:3,3 taper:0.1 detrend:true

# AR-based spectrum estimation
/timeseries_spectrum dataset:data column:returns method:ar ar_order:10
```

**Parameters:**
- `dataset` (required): Dataset name
- `column` (required): Time series column
- `spans` (optional): Odd integers for Daniell kernel smoothing (e.g., "3,3" or "5")
- `taper` (optional): Proportion of data to taper (0 to 0.5, default: 0.1)
- `detrend` (optional): Remove linear trend before analysis (default: true)
- `method` (optional): "pgram" (default) or "ar" for AR-based spectrum
- `ar_order` (optional): AR model order for method=ar (auto-selected if omitted)

**Output includes:**
- Frequency values (0 to Nyquist = 0.5)
- Spectral density estimates at each frequency
- Bandwidth (smoothing resolution)
- Degrees of freedom (for confidence intervals)
- Peak frequency detection

**Applications:**
- Identify dominant frequencies/cycles in time series
- Detect seasonality or periodic patterns
- Signal vs. noise analysis
- Pre-whitening for transfer function models

---

### timeseries_box_test
Box-Pierce and Ljung-Box tests for autocorrelation (portmanteau tests).

```
# Basic Ljung-Box test with default lag=1
/timeseries_box_test dataset:data column:residuals

# Ljung-Box test with 10 lags
/timeseries_box_test dataset:data column:residuals lag:10

# Box-Pierce test (simpler, classic version)
/timeseries_box_test dataset:data column:residuals lag:10 test_type:box-pierce

# Testing ARMA(1,1) residuals - adjust degrees of freedom
/timeseries_box_test dataset:data column:residuals lag:10 fitdf:2
```

**Parameters:**
- `dataset` (required): Dataset name
- `column` (required): Time series column to test
- `lag` (optional): Number of autocorrelation lags to include (default: 1, common choices: 10, 20)
- `test_type` (optional): "ljung-box" (default) or "box-pierce"
- `fitdf` (optional): Degrees of freedom adjustment for ARMA residuals (set to p+q for ARMA(p,q), default: 0)

**Output includes:**
- X-squared test statistic
- Degrees of freedom (lag - fitdf)
- P-value from chi-squared distribution
- Sample autocorrelations used in test
- Interpretation of results

**Applications:**
- Check if ARIMA model residuals are white noise
- Test for autocorrelation in regression residuals
- Validate model specification
- Diagnostic checking in time series analysis

---

### timeseries_pp_test
Phillips-Perron unit root test for stationarity.

```
# Basic PP test with default short truncation lag
/timeseries_pp_test dataset:data column:gdp

# PP test with long truncation lag (more robust to autocorrelation)
/timeseries_pp_test dataset:data column:gdp lshort:false
```

**Parameters:**
- `dataset` (required): Dataset name
- `column` (required): Time series column to test
- `lshort` (optional): Use short truncation lag formula (default: true)
  - true: trunc(4*(n/100)^0.25)
  - false: trunc(12*(n/100)^0.25)

**Output includes:**
- Z(τ) test statistic (Dickey-Fuller with PP correction)
- Truncation lag parameter
- P-value (interpolated from critical value tables)
- Diagnostics: γ̂, t-statistic, σ², λ²
- Interpretation and recommendation

**Applications:**
- Test for unit root before time series modeling
- Check stationarity before ARIMA/VAR analysis
- Cointegration analysis (test individual series)
- Alternative to ADF test when heteroskedasticity suspected

**Interpreting Results:**
- **p < 0.05**: Reject unit root → series is stationary
- **p ≥ 0.05**: Cannot reject unit root → consider differencing

---

## Multivariate Analysis

### multivariate_factanal
Maximum Likelihood Factor Analysis with optional rotation.

```
/multivariate_factanal dataset:survey columns:q1,q2,q3,q4,q5,q6 n_factors:2
/multivariate_factanal dataset:survey columns:q1,q2,q3,q4,q5,q6 n_factors:2 rotation:varimax
/multivariate_factanal dataset:survey columns:q1,q2,q3,q4,q5,q6 n_factors:3 rotation:promax scores:regression
```

**Parameters:**
- `dataset` (required): Dataset name
- `columns` (required): Comma-separated list of numeric columns to analyze
- `n_factors` (required): Number of latent factors to extract
- `rotation` (optional): Rotation method - "none", "varimax" (default), or "promax"
- `scores` (optional): Factor score method - "none" (default), "regression", or "bartlett"

**Output includes:**
- Factor loadings matrix (p × k)
- Uniquenesses (variance unexplained by factors)
- Communalities (variance explained by factors)
- Chi-squared goodness-of-fit statistic
- Degrees of freedom and p-value
- Factor scores (if requested)
- Rotation matrix (for varimax)
- Factor correlation matrix (for promax)

**Use cases:**
- Dimensionality reduction for survey/questionnaire data
- Identifying latent constructs underlying observed variables
- Scale construction and psychometric analysis
- Exploratory factor analysis (EFA)

**Example interpretation:**
```
Loadings (varimax rotated):
           Factor1  Factor2
item1        0.82     0.15
item2        0.79     0.22
item3        0.75     0.18
item4        0.19     0.85
item5        0.24     0.78
item6        0.21     0.80

Items 1-3 load on Factor 1, items 4-6 load on Factor 2
```

---

## Machine Learning

### ml_kmeans
K-means clustering with k-means++ initialization.

```
/ml_kmeans dataset:customers columns:income,spending k:3
```

**Parameters:**
- `k`: Number of clusters
- `max_iter` (optional): Maximum iterations (default: 100)
- `n_init` (optional): Number of random initializations (default: 10)

**Output includes:**
- Cluster assignments
- Cluster centroids
- Inertia (within-cluster sum of squares)

### ml_dbscan
Density-based clustering.

```
/ml_dbscan dataset:customers columns:income,spending eps:0.5 min_samples:5
```

**Parameters:**
- `eps`: Maximum distance between points in a neighborhood
- `min_samples`: Minimum points to form a dense region

**Advantages:**
- Does not require specifying number of clusters
- Identifies outliers as noise (cluster = -1)

### ml_hierarchical
Hierarchical (agglomerative) clustering.

```
/ml_hierarchical dataset:customers columns:income,spending n_clusters:3 linkage:ward
```

**Linkage options:**
- `ward`: Minimizes within-cluster variance (recommended)
- `single`: Minimum distance between clusters
- `complete`: Maximum distance between clusters
- `average`: Average distance between clusters

### ml_pca
Principal Component Analysis for dimensionality reduction.

```
/ml_pca dataset:features columns:x1,x2,x3,x4,x5 n_components:2
```

**Output includes:**
- Principal component scores
- Explained variance ratios
- Loadings (feature contributions)

### ml_tsne
t-SNE for visualization of high-dimensional data.

```
/ml_tsne dataset:features columns:x1,x2,x3,x4,x5 n_components:2 perplexity:30
```

**Parameters:**
- `perplexity`: Balance between local and global structure (typical: 5-50)

### ml_random_forest
Random Forest regression.

```
/ml_random_forest dataset:housing y:price x:sqft,bedrooms,bathrooms n_trees:100
```

**Output includes:**
- Predictions
- Feature importance scores
- Out-of-bag error estimate

### ml_svm
Linear Support Vector Machine classification.

```
/ml_svm dataset:binary_data y:class x:feature1,feature2 C:1.0
```

**Parameters:**
- `C`: Regularization parameter (higher = less regularization)

---

## Database Queries

### db_sqlite_query
Execute SQL query on SQLite database.

```
/db_sqlite_query path:/data/database.db query:SELECT * FROM customers WHERE age > 30
```

### db_sqlite_tables
List all tables in SQLite database.

```
/db_sqlite_tables path:/data/database.db
```

### db_sqlite_schema
Get schema for a specific table.

```
/db_sqlite_schema path:/data/database.db table:customers
```

### db_duckdb_query
Execute SQL query on DuckDB database.

```
/db_duckdb_query path:/data/analytics.duckdb query:SELECT AVG(revenue) FROM sales GROUP BY region
```

DuckDB can also query files directly:
```
/db_duckdb_query path::memory: query:SELECT * FROM '/data/sales.parquet' WHERE year = 2024
```

### db_duckdb_tables / db_duckdb_schema
Same as SQLite equivalents but for DuckDB.

---

## Visualization

### viz_histogram
Distribution of a single variable.

```
/viz_histogram dataset:sales column:revenue bins:20
```

### viz_scatter
Scatter plot with correlation coefficient.

```
/viz_scatter dataset:housing x_column:sqft y_column:price
```

### viz_line
Time series or multi-series line chart.

```
/viz_line dataset:sales x_column:date y_columns:revenue,costs
```

### viz_boxplot
Box plots for comparing distributions.

```
/viz_boxplot dataset:survey columns:income,spending,savings
```

### viz_heatmap
Correlation heatmap.

```
/viz_heatmap dataset:features columns:x1,x2,x3,x4,x5
```

### viz_event_study
Event study plot for DiD or dynamic treatment effects.

```
/viz_event_study dataset:event_data estimates:coefs std_errors:ses periods:periods
```

### viz_coefficient
Coefficient plot with confidence intervals.

```
/viz_coefficient dataset:results variables:var1,var2,var3 coefficients:coef_col std_errors:se_col
```

### viz_irf
Impulse Response Function plot from VAR.

```
/viz_irf dataset:irf_results horizons:horizon_col response:response_col
```

### viz_residual_diagnostics
Four diagnostic plots: Residuals vs Fitted, Q-Q, Scale-Location, Leverage.

```
/viz_residual_diagnostics dataset:housing y:price x:sqft,bedrooms
```

### viz_dendrogram
Tree diagram for hierarchical clustering.

```
/viz_dendrogram dataset:customers columns:income,spending
```

---

## Utilities

### generate_report
Create a self-contained HTML report.

```
/generate_report title:Sales Analysis sections:summary,regression,charts output:/path/to/report.html
```

### batch_process
Run the same analysis across multiple datasets.

```
/batch_process datasets:data1,data2,data3 tool:regression_ols y:outcome x:treatment
```

### compare_datasets
Compare columns across datasets.

```
/compare_datasets datasets:train,test columns:price,sqft comparison:summary
```

**Comparison types:**
- `summary`: Descriptive statistics
- `distribution`: Distribution comparison
- `correlation`: Correlation matrices

### export_session
Save current session (loaded datasets) to JSON.

```
/export_session path:/path/to/session.json
```

### import_session
Restore a previously saved session.

```
/import_session path:/path/to/session.json
```

### set_seed
Set random seed for reproducibility.

```
/set_seed seed:42
```

### get_seed
Get current seed and list of affected tools.

```
/get_seed
```

---

## Tips for Best Results

1. **Always load data first** using `load_dataset` before running analyses.

2. **Check data quality** with `describe_dataset` to identify missing values or outliers.

3. **Run diagnostics** after OLS to check model assumptions.

4. **For panel data**, ensure your entity column uniquely identifies cross-sectional units.

5. **For IV**, always check first-stage F-statistic (should be > 10 for strong instruments).

6. **For DiD**, verify parallel trends assumption holds before interpreting ATT.

7. **For clustering**, try multiple values of k and compare using inertia or silhouette scores.

8. **Set a seed** before any ML analysis for reproducibility.
