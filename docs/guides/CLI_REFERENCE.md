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

**Robust SE types:** `standard`, `hc0`, `hc1` (default), `hc2`, `hc3`

**Examples:**
```bash
# Basic OLS with HC1 robust standard errors
p2a reg ols mydata -y price -x sqft bedrooms bathrooms --robust hc1

# Clustered by firm
p2a reg clustered mydata -y revenue -x employees R_D --cluster firm_id

# Run diagnostics (Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF)
p2a reg diagnostics mydata -y price -x sqft bedrooms
```

### Panel Data Commands

```bash
# Fixed effects
p2a panel fe <DATASET> -y <DEP_VAR> -x <INDEP_VARS...> --entity <ENTITY_COL>

# Random effects
p2a panel re <DATASET> -y <DEP_VAR> -x <INDEP_VARS...> --entity <ENTITY_COL>

# Hausman specification test
p2a panel hausman <DATASET> -y <DEP_VAR> -x <INDEP_VARS...> --entity <ENTITY_COL>

# High-dimensional fixed effects
p2a panel hdfe <DATASET> -y <DEP_VAR> -x <INDEP_VARS...> --fe <FE_COLS...>
```

**Examples:**
```bash
# Within-firm fixed effects
p2a panel fe firms -y revenue -x employees capital --entity firm_id

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

### JSON (`--output json`)
Structured JSON output for programmatic use:
```bash
p2a --output json reg ols mydata -y y -x x1 x2 | jq '.coefficients'
```

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
