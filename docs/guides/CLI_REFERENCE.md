# p2a CLI Reference

The `p2a` command-line interface provides direct access to all prompt2analytics capabilities for scripted workflows and reproducible analysis.

## Installation

```bash
cargo build --release -p p2a-cli
# Binary at: target/release/p2a
```

## Global Options

```
-o, --output <FORMAT>    Output format: text (default), json, table
--session <FILE>         Record commands to session file for reproducibility
-h, --help               Print help
-V, --version            Print version
```

## Commands

### Data Commands

```bash
# Load a dataset
p2a data load <PATH> [--name NAME] [--format FORMAT]

# List loaded datasets
p2a data list

# Show dataset summary statistics
p2a data describe <DATASET>

# Preview first N rows
p2a data head <DATASET> [-n ROWS]
```

**Examples:**
```bash
p2a --session analysis.json data load /data/panel.csv --name firms
p2a --session analysis.json data describe firms
p2a --session analysis.json data head firms -n 20
```

### Regression Commands

```bash
# OLS regression with robust standard errors
p2a reg ols <DATASET> -y <DEP_VAR> -x <INDEP_VARS...> [--intercept] [--robust TYPE]

# Clustered standard errors
p2a reg clustered <DATASET> -y <DEP_VAR> -x <INDEP_VARS...> --cluster <CLUSTER_COL>

# Regression diagnostics
p2a reg diagnostics <DATASET> -y <DEP_VAR> -x <INDEP_VARS...>
```

**Standard errors:** `--robust standard|hc0|hc1|hc2|hc3` (default `hc1`, heteroskedasticity-robust;
`standard` reproduces R's `summary(lm())`).

Every coefficient table states its estimator on the line `Standard errors: ...`:
`HC1 (heteroskedasticity-robust)` (and likewise `HC0`, `HC2`, `HC3`), `classical`, or for
`reg clustered` `clustered by <var> (CR1, G = <n>)`, where CR1 is the factor
G/(G-1)·(n-1)/(n-k) of `sandwich::vcovCL(type = "HC1")` and G the number of clusters. The
t-values are coefficient / standard error as the estimator reports them (an aliased regressor
shows the estimator's `NaN`). Estimator warnings, for example the few-clusters warning of
`reg clustered`, follow under `Warnings:`. `reg ols grunfeld -y inv -x value capital` on
`validation/datasets/grunfeld.csv` prints:

```
OLS Regression Results
============================================================

Variable         Coefficient   Std. Error    t-value    p-value
------------------------------------------------------------
(Intercept)       -42.714369    11.574701     -3.690     0.0003***
value               0.115562     0.006811     16.967     0.0000***
capital             0.230678     0.048866      4.721     0.0000***
------------------------------------------------------------
R-squared: 0.812408
Adj. R-squared: 0.810504
Observations: 200
Standard errors: HC1 (heteroskedasticity-robust)

Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1
```

`reg diagnostics` lists the variance inflation factors of the regressors only: the intercept
has none, and `car::vif` omits it too. The `-F json` output keeps the intercept entry with
`null`.

**Examples:**
```bash
# Basic OLS with HC1 robust standard errors
p2a reg ols mydata -y price -x sqft bedrooms bathrooms --robust hc1

# Classical standard errors, as R's summary(lm())
p2a reg ols mydata -y price -x sqft bedrooms bathrooms --robust standard

# Clustered by firm
p2a reg clustered mydata -y revenue -x employees R_D --cluster firm_id

# Run diagnostics (Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF)
p2a reg diagnostics mydata -y price -x sqft bedrooms
```

### Panel Data Commands

```bash
# Fixed effects
p2a panel fe <DATASET> -y <DEP_VAR> -x <INDEP_VARS...> --entity <ENTITY_COL> [--vcov <TYPE>]

# Random effects
p2a panel re <DATASET> -y <DEP_VAR> -x <INDEP_VARS...> --entity <ENTITY_COL> [--vcov <TYPE>]

# Hausman specification test
p2a panel hausman <DATASET> -y <DEP_VAR> -x <INDEP_VARS...> --entity <ENTITY_COL>

# High-dimensional fixed effects
p2a panel hdfe <DATASET> -y <DEP_VAR> -x <INDEP_VARS...> --fe <FE_COLS...>
```

**--vcov:** `classical` (default; plm's `vcov`), `cluster` (entity-clustered Arellano with the
G/(G-1)·(n-1)/(n-k) factor: plm `vcovHC(method = "arellano", type = "sss")`, Stata's
`vce(cluster)`), `cluster-hc0` (no factor), `cluster-hc1` (n/(n-k)). Any other value is a usage
error (exit 2) that lists these four.

What each command prints, besides the coefficient table and its `Standard errors:` line:

- `panel fe`: the within R-squared, labelled `R-squared (within):`, and its adjusted value
  `Adj. R-squared (within):`, adjusted with (n-1)/df_resid as plm does; `Number of groups:` and
  the F test of the slopes. Under `--vcov cluster*` the F line is
  `F-statistic (Wald, clustered): F on K and G-1 DF (p-value: p)`, the Wald F plm's `summary`
  reports with a cluster covariance.
- `panel re`: Swamy-Arora variance components, z-values with normal p-values (as plm), the
  R-squared of plm (`cor(y*, X*b)^2` of the quasi-demeaned model), `Number of groups:`,
  `Wald chi-squared: W on K DF (p-value: p)` (`Wald chi-squared (clustered):` under
  `--vcov cluster*`), then `sigma_u: a, sigma_e: b, theta: c`. On an unbalanced panel theta
  differs by entity; the line then reads `theta (mean): c, range lo-hi`. When the estimated
  individual-effect variance is negative it is set to 0 (as plm), random effects reduces to
  pooled OLS, and a warning says so.
- `panel hausman`: always the classical FE and RE covariances (`plm::phtest`), stated on the
  line `Covariance: classical FE and RE covariance (plm::phtest default)`, with warnings when
  Var(b_FE) - Var(b_RE) is not positive definite or is singular.
- `panel hdfe`: HC1 standard errors (fixest `vcov = "hetero"`), the within R-squared labelled
  `R-squared (within):` and `Adj. R-squared (within):` (adjusted with (n-1)/df_resid as plm;
  fixest's `war2` differs: 0.716833 vs 0.670467 on Grunfeld), the levels of each fixed effect
  and the within F test.

**Examples:**
```bash
# Within-firm fixed effects
p2a panel fe firms -y revenue -x employees capital --entity firm_id

# Firm-clustered (Arellano) standard errors
p2a panel fe firms -y revenue -x employees capital --entity firm_id --vcov cluster

# Random effects model
p2a panel re firms -y revenue -x employees capital --entity firm_id

# Hausman test to choose between FE and RE
p2a panel hausman firms -y revenue -x employees capital --entity firm_id

# Two-way fixed effects (firm + year)
p2a panel hdfe firms -y revenue -x employees --fe firm_id year
```

### Causal Inference Commands

```bash
# Instrumental variables (2SLS)
p2a causal iv <DATASET> -y <DEP_VAR> --exog <EXOG_VARS...> --endog <ENDOG_VARS...> --instruments <INST_VARS...>

# Difference-in-differences
p2a causal did <DATASET> -y <DEP_VAR> --treat <TREAT_COL> --post <POST_COL>

# Synthetic Control Method (Abadie et al.)
p2a causal synth <DATASET> -y <OUTCOME> --unit <UNIT_COL> --time <TIME_COL> \
    --treated <TREATED_UNIT> --treatment-time <TIME> -p <PREDICTORS...> \
    [--v-method <METHOD>] [--placebos]

# Regression Discontinuity (Sharp RD)
p2a causal rd <DATASET> -y <DEP_VAR> --running <RUNNING_VAR> --cutoff <CUTOFF> \
    [--bandwidth <H>] [--kernel <TYPE>] [--polynomial <P>]

# Regression Discontinuity (Fuzzy RD)
p2a causal rd-fuzzy <DATASET> -y <DEP_VAR> --treatment <TREAT_VAR> --running <RUNNING_VAR> \
    --cutoff <CUTOFF> [--bandwidth <H>] [--kernel <TYPE>]

# RD Bandwidth Selection
p2a causal rd-bw <DATASET> -y <DEP_VAR> --running <RUNNING_VAR> --cutoff <CUTOFF> \
    [--method <METHOD>]
```

**V-method options:** `datadriven` (default), `equal`

**RD kernel types:** `triangular` (default), `epanechnikov`, `uniform`

**RD bandwidth methods:** `mserd` (default), `msetwo`, `cerrd`, `certwo`

**Examples:**
```bash
# 2SLS with instrument
p2a causal iv mydata -y wage --exog experience --endog education --instruments parent_education

# DiD estimation
p2a causal did mydata -y outcome --treat treated --post post_treatment

# Synthetic control (California tobacco study style)
p2a causal synth smoking -y cigsale --unit state --time year --treated California \
    --treatment-time 1988 -p lnincome retprice

# Synthetic control with placebo tests for inference
p2a causal synth mydata -y gdp --unit country --time year --treated Germany \
    --treatment-time 1990 -p exports imports --placebos

# Sharp RD: effect of crossing eligibility threshold
p2a causal rd mydata -y test_score --running age --cutoff 65 --bandwidth 5

# Fuzzy RD: treatment probability changes at cutoff
p2a causal rd-fuzzy mydata -y earnings --treatment enrolled --running score --cutoff 50

# Bandwidth selection for RD
p2a causal rd-bw mydata -y outcome --running running_var --cutoff 0 --method mserd
```

### Discrete Choice Commands

```bash
# Logit model
p2a discrete logit <DATASET> -y <DEP_VAR> -x <INDEP_VARS...>

# Probit model
p2a discrete probit <DATASET> -y <DEP_VAR> -x <INDEP_VARS...>
```

**Examples:**
```bash
p2a discrete logit mydata -y purchased -x income age education
p2a discrete probit mydata -y voted -x age income party_id
```

### Time Series Commands

```bash
# ARIMA model
p2a ts arima <DATASET> --col <COL> -p <AR> -d <DIFF> -q <MA> [--horizon <N>]

# MSTL decomposition
p2a ts mstl <DATASET> --col <COL> --periods <PERIODS...>

# Vector autoregression
p2a ts var <DATASET> --cols <COLS...> --lags <N>
```

**Examples:**
```bash
# ARIMA(1,1,1) with 12-step forecast
p2a ts arima sales --col revenue -p 1 -d 1 -q 1 --horizon 12

# Seasonal decomposition (weekly + yearly)
p2a ts mstl sales --col revenue --periods 7 365

# VAR(2) for multiple series
p2a ts var macro --cols gdp inflation unemployment --lags 2
```

### Machine Learning Commands

```bash
# K-means clustering
p2a ml kmeans <DATASET> --cols <COLS...> -k <N> [--seed <N>] [--max-iter <N>]

# Principal Component Analysis
p2a ml pca <DATASET> --cols <COLS...> [--n-components <N>] [--transform]
```

**Examples:**
```bash
# Cluster customers into 5 segments
p2a ml kmeans customers --cols income spending age -k 5 --seed 42

# PCA with top 3 components
p2a ml pca customers --cols var1 var2 var3 var4 var5 --n-components 3
```

### Visualization Commands

```bash
# Histogram
p2a viz histogram <DATASET> --col <COL> -f <OUTPUT> [--bins <N>] [--title <TITLE>]

# Scatter plot
p2a viz scatter <DATASET> -x <X_COL> -y <Y_COL> -f <OUTPUT> [--title <TITLE>]

# Line chart
p2a viz line <DATASET> -x <X_COL> -y <Y_COL> -f <OUTPUT> [--title <TITLE>]

# Box plot
p2a viz box <DATASET> -y <VALUE_COL> [-g <GROUP_COL>] -f <OUTPUT> [--title <TITLE>]

# Correlation heatmap
p2a viz heatmap <DATASET> --cols <COLS...> -f <OUTPUT> [--title <TITLE>]

# Coefficient plot (forest plot from regression)
p2a viz coefplot <DATASET> -y <DEP_VAR> -x <INDEP_VARS...> -f <OUTPUT> [--conf-level <N>] [--title <TITLE>]

# Residual diagnostic plots
p2a viz residuals <DATASET> -y <DEP_VAR> -x <INDEP_VARS...> -f <OUTPUT> [--title <TITLE>]

# Dendrogram (hierarchical clustering)
p2a viz dendrogram <DATASET> --cols <COLS...> -f <OUTPUT> [--linkage <METHOD>] [--title <TITLE>]

# Event study plot
p2a viz event-study <DATASET> --time-col <COL> --estimate-col <COL> --ci-lower-col <COL> --ci-upper-col <COL> -f <OUTPUT> [--title <TITLE>]

# Impulse response function plot
p2a viz irf <DATASET> --horizon-col <COL> --response-col <COL> [--ci-lower-col <COL>] [--ci-upper-col <COL>] -f <OUTPUT> [--title <TITLE>]
```

**Linkage methods:** `single`, `complete`, `average`, `ward` (default)

**Examples:**
```bash
# Basic distribution plots
p2a viz histogram mydata --col price -f price_dist.png --bins 50
p2a viz scatter mydata -x income -y spending -f scatter.png --title "Income vs Spending"
p2a viz line timeseries -x date -y value -f trend.png

# Box plot by category
p2a viz box sales -y revenue -g region -f revenue_by_region.png

# Correlation heatmap for selected variables
p2a viz heatmap mydata --cols price sqft bedrooms bathrooms -f correlations.png

# Coefficient plot from regression results
p2a viz coefplot mydata -y price -x sqft bedrooms bathrooms -f coef_plot.png --conf-level 0.95

# Residual diagnostics (creates 4 plots: main, _qq, _scale_location, _leverage)
p2a viz residuals mydata -y price -x sqft bedrooms -f residual_diag.png

# Dendrogram for clustering analysis
p2a viz dendrogram customers --cols spending income age -f dendrogram.png --linkage ward

# Event study plot (pre-prepared coefficient data)
p2a viz event-study event_coeffs --time-col period --estimate-col coef --ci-lower-col ci_lo --ci-upper-col ci_hi -f event_study.png

# IRF plot from VAR analysis
p2a viz irf irf_data --horizon-col h --response-col response --ci-lower-col lower --ci-upper-col upper -f irf.png
```

### Script Commands

```bash
# Export session to bash script
p2a script export <SESSION_FILE> -o <SCRIPT_FILE>

# Run a script
p2a script run <SCRIPT_FILE>
```

**Examples:**
```bash
# Record a session
p2a --session analysis.json data load data.csv --name d
p2a --session analysis.json reg ols d -y y -x x1 x2

# Export to reproducible script
p2a script export analysis.json -o analysis.sh

# Re-run the analysis
bash analysis.sh
```

## Output Formats

### Text (default)
Human-readable formatted output with tables and summaries.

### JSON (`-F json`)
Structured JSON output for programmatic use:
```bash
p2a -F json reg ols mydata -y y -x x1 x2 | jq '.coefficients'
```

The estimation commands emit objects with alphabetically sorted keys; numbers that are not
finite (an aliased regressor's `NaN`) are `null`.

| Command | Keys |
|---|---|
| `reg ols` | `adj_r_squared`, `coefficients` (objects with `coefficient`, `name`, `p_value`, `std_error`, `t_value`), `method`, `n_observations`, `r_squared`, `r_squared_type` (`"ordinary"`), `se_type` (`"classical"`, `"HC0"` ... `"HC3"`), `statistic` (`"t"`), `warnings` (only when there are any) |
| `reg clustered` | as `reg ols`, plus `n_clusters`; `se_type` is `"cluster:<var>"` |
| `reg hac` | `bandwidth`, `coefficients`, `kernel`, `method`, `n_obs`, `p_values`, `r_squared`, `std_errors`, `t_values` (coefficient / HAC SE), `warnings` (only when there are any) |
| `panel fe` | as `reg ols`, plus `n_groups`, `f_statistic`, `f_p_value`; `r_squared_type` is `"within"`; `se_type` is `"classical"`, `"cluster:<entity>"`, `"cluster-hc0:<entity>"` or `"cluster-hc1:<entity>"` |
| `panel re` | as `panel fe`, but the coefficient objects have `z_value` instead of `t_value` and `statistic` is `"z"`; `r_squared_type` is `"cor(y*, X*b)^2 (plm)"`; `f_statistic`/`f_p_value` are the F form W/K of the Wald test; plus `wald_chi2`, `wald_df`, `wald_p_value`, `sigma_u`, `sigma_e`, `theta` (the observation-weighted mean on an unbalanced panel) and, on an unbalanced panel, `theta_min` and `theta_max` |
| `panel hdfe` | as `panel fe`, plus `fe_dimensions` (objects with `levels`, `name`, one per `--fe` column); `n_groups` is the number of levels of the first `--fe` column; `se_type` is `"HC1"` |
| `panel hausman` | `chi2_statistic`, `df`, `p_value`, `recommendation`, `test`, `vcov_method`, `warnings` (always, possibly `[]`) |

### Table (`--output table`)
ASCII table format using tabled.

## Session Recording

Use `--session <file>` to record all commands for reproducibility:

```bash
# Start a session
p2a --session my_analysis.json data load data.csv --name mydata

# All subsequent commands with same session file are recorded
p2a --session my_analysis.json data describe mydata
p2a --session my_analysis.json reg ols mydata -y y -x x1 x2
p2a --session my_analysis.json viz scatter mydata -x x1 -y y -f plot.png

# Export to bash script
p2a script export my_analysis.json -o my_analysis.sh
```

The generated script can be shared and re-executed for reproducible results.

## Error Handling

Errors are displayed with context:
```
Error: Dataset 'missing' not found
Error: Column 'unknown' not found in dataset
Error: Regression failed: Matrix is singular
```

With `--output json`, errors are returned as JSON:
```json
{"error": "Dataset 'missing' not found"}
```
