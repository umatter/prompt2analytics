# prompt2analytics Cookbook

A practical guide with real-world examples for common analytics workflows.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Loading and Exploration](#data-loading-and-exploration)
3. [Regression Analysis](#regression-analysis)
4. [Panel Data Analysis](#panel-data-analysis)
5. [Causal Inference](#causal-inference)
6. [Time Series Analysis](#time-series-analysis)
7. [Machine Learning](#machine-learning)
8. [Visualization](#visualization)

---

## Getting Started

### Installation

```bash
# Build the CLI
cargo build --release -p p2a-cli

# Build the MCP server
cargo build --release -p p2a-mcp

# The CLI binary is at target/release/p2a
# The MCP server is at target/release/p2a-mcp
```

### CLI Basics

All CLI commands use sessions for reproducibility:

```bash
# Start a new analysis session
p2a --session my_analysis.json data load mydata.csv --name mydata

# Continue the session
p2a --session my_analysis.json data describe mydata
```

### MCP Server

Configure your AI assistant to use the MCP server:

```json
{
  "mcpServers": {
    "prompt2analytics": {
      "command": "/path/to/target/release/p2a-mcp"
    }
  }
}
```

---

## Data Loading and Exploration

### Loading Data

```bash
# Load CSV file
p2a --session analysis.json data load /path/to/data.csv --name mydata

# Load with custom delimiter
p2a --session analysis.json data load data.tsv --name mydata --delimiter "\t"

# Load Parquet file
p2a --session analysis.json data load data.parquet --name mydata

# Load Stata file
p2a --session analysis.json data load data.dta --name mydata
```

### Exploring Data

```bash
# List all loaded datasets
p2a --session analysis.json data list

# Get summary statistics
p2a --session analysis.json data describe mydata

# Preview first 10 rows
p2a --session analysis.json data head mydata -n 10
```

### Data Quality

```bash
# Generate quality profile (missing values, outliers, duplicates)
p2a --session analysis.json data quality mydata

# Get cleaning suggestions
p2a --session analysis.json data suggest mydata
```

---

## Regression Analysis

### Basic OLS

```bash
# Simple linear regression
p2a --session analysis.json reg ols mydata -y price -x sqft

# Multiple regression
p2a --session analysis.json reg ols mydata -y price -x sqft bedrooms bathrooms

# Without intercept
p2a --session analysis.json reg ols mydata -y price -x sqft --no-intercept
```

### Robust Standard Errors

```bash
# HC1 robust standard errors (default)
p2a --session analysis.json reg ols mydata -y price -x sqft bedrooms --robust hc1

# HC3 (more conservative)
p2a --session analysis.json reg ols mydata -y price -x sqft bedrooms --robust hc3

# Clustered standard errors
p2a --session analysis.json reg clustered mydata -y outcome -x treatment control --cluster firm_id
```

### Diagnostics

```bash
# Run diagnostic tests (VIF, Breusch-Pagan, Durbin-Watson)
p2a --session analysis.json reg diagnostics mydata -y price -x sqft bedrooms bathrooms
```

### Quantile Regression

```bash
# Median regression (tau=0.5)
p2a --session analysis.json reg quantile mydata -y price -x sqft bedrooms --tau 0.5

# 25th percentile
p2a --session analysis.json reg quantile mydata -y price -x sqft bedrooms --tau 0.25
```

---

## Panel Data Analysis

### Fixed Effects

```bash
# Entity (individual) fixed effects
p2a --session analysis.json panel fe mydata \
    -y revenue -x employees marketing \
    --entity firm_id

# Two-way fixed effects (entity + time)
p2a --session analysis.json panel fe mydata \
    -y revenue -x employees marketing \
    --entity firm_id --time year
```

### Random Effects

```bash
# Random effects model
p2a --session analysis.json panel re mydata \
    -y revenue -x employees marketing \
    --entity firm_id
```

### Hausman Test

```bash
# Test FE vs RE
p2a --session analysis.json panel hausman mydata \
    -y revenue -x employees marketing \
    --entity firm_id
```

### High-Dimensional Fixed Effects

```bash
# Multiple fixed effects (firm + year + industry)
p2a --session analysis.json panel hdfe mydata \
    -y revenue -x employees marketing \
    --fe firm_id year industry_code
```

---

## Causal Inference

### Difference-in-Differences

```bash
# Basic DiD
p2a --session analysis.json causal did mydata \
    -y outcome \
    --treat treatment \
    --post post_period

# With controls
p2a --session analysis.json causal did mydata \
    -y outcome \
    --treat treatment \
    --post post_period \
    -x control1 control2
```

### Staggered DiD (Callaway-Sant'Anna)

```bash
# Staggered treatment timing
p2a --session analysis.json causal staggered-did mydata \
    -y outcome \
    --gvar treatment_year \
    --tvar year \
    --idvar unit_id
```

### Instrumental Variables

```bash
# 2SLS with a single instrument
p2a --session analysis.json causal iv mydata \
    -y outcome \
    -x treatment \
    --instruments instrument1

# Multiple instruments
p2a --session analysis.json causal iv mydata \
    -y outcome \
    -x treatment \
    --instruments instrument1 instrument2
```

### Regression Discontinuity

```bash
# Sharp RD
p2a --session analysis.json causal rd mydata \
    -y outcome \
    -x running_variable \
    --cutoff 50

# Fuzzy RD
p2a --session analysis.json causal rd-fuzzy mydata \
    -y outcome \
    -x running_variable \
    --cutoff 50 \
    --treatment actual_treatment
```

### Propensity Score Matching

```bash
# Nearest neighbor matching
p2a --session analysis.json causal match mydata \
    -y outcome \
    --treat treatment \
    --covariates age income education \
    --method nearest
```

---

## Time Series Analysis

### ARIMA

```bash
# Fit ARIMA(1,1,1)
p2a --session analysis.json ts arima mydata --col sales -p 1 -d 1 -q 1

# Forecast 12 periods ahead
p2a --session analysis.json ts arima mydata --col sales -p 1 -d 1 -q 1 --horizon 12
```

### Decomposition

```bash
# STL decomposition
p2a --session analysis.json ts stl mydata --col sales --period 12

# MSTL (multiple seasonality)
p2a --session analysis.json ts mstl mydata --col sales --periods 7 365
```

### VAR Models

```bash
# Vector autoregression
p2a --session analysis.json ts var mydata --cols gdp inflation unemployment --lags 2

# Impulse response functions
p2a --session analysis.json ts irf mydata --cols gdp inflation unemployment --lags 2 --horizon 20
```

### Changepoint Detection

```bash
# Detect structural breaks
p2a --session analysis.json ts changepoint mydata --col price --method pelt
```

---

## Machine Learning

### Clustering

```bash
# K-means clustering
p2a --session analysis.json ml kmeans mydata --cols x1 x2 x3 -k 3

# DBSCAN
p2a --session analysis.json ml dbscan mydata --cols x1 x2 x3 --eps 0.5 --min-samples 5

# Hierarchical clustering
p2a --session analysis.json ml hierarchical mydata --cols x1 x2 x3 --method ward
```

### Dimensionality Reduction

```bash
# PCA
p2a --session analysis.json ml pca mydata --cols x1 x2 x3 x4 x5 --n-components 2

# t-SNE
p2a --session analysis.json ml tsne mydata --cols x1 x2 x3 x4 x5 --n-components 2 --perplexity 30
```

### Random Forest

```bash
# Classification
p2a --session analysis.json ml random-forest mydata \
    -y label -x feature1 feature2 feature3 \
    --n-trees 100

# Regression
p2a --session analysis.json ml random-forest mydata \
    -y price -x sqft bedrooms bathrooms \
    --n-trees 100 --task regression
```

---

## Visualization

### Static Charts (PNG)

```bash
# Histogram
p2a --session analysis.json viz histogram mydata -x price -f histogram.png

# Scatter plot
p2a --session analysis.json viz scatter mydata -x sqft -y price -f scatter.png

# Line chart
p2a --session analysis.json viz line mydata -x date -y sales -f trend.png

# Box plot
p2a --session analysis.json viz boxplot mydata -x category -y value -f boxplot.png
```

### Interactive Charts (HTML)

```bash
# Interactive scatter with zoom/pan
p2a --session analysis.json viz scatter-interactive mydata -x sqft -y price -f scatter.html

# Interactive line chart
p2a --session analysis.json viz line-interactive mydata -x date -y sales -f trend.html
```

### Specialized Plots

```bash
# Correlation heatmap
p2a --session analysis.json viz heatmap mydata --cols x1 x2 x3 x4 x5 -f correlation.png

# Coefficient plot from regression
p2a --session analysis.json viz coefficient mydata -y price -x sqft bedrooms bathrooms -f coef.png
```

---

## MCP Tool Examples

When using prompt2analytics through an AI assistant, you can use natural language:

### Example Prompts

**Data exploration:**
> "Load the file sales_data.csv and show me summary statistics"

**Regression:**
> "Run an OLS regression of sales on advertising and price, with robust standard errors"

**Panel data:**
> "Estimate a fixed effects model with firm and year fixed effects"

**Causal inference:**
> "Run a difference-in-differences analysis comparing the treatment and control groups before and after the policy"

**Visualization:**
> "Create a scatter plot of income vs spending, colored by education level"

---

## Tips and Best Practices

### Reproducibility

1. **Always use sessions**: Sessions record all operations for reproducibility
2. **Set random seeds**: Use `set_seed` before any stochastic operations
3. **Export scripts**: Use `p2a script export session.json -o analysis.sh`

### Performance

1. **Use Parquet**: For large datasets, Parquet is much faster than CSV
2. **Pre-filter data**: Filter to relevant observations before complex analyses
3. **Use HDFE**: For multiple fixed effects, HDFE is more efficient than dummy variables

### Best Practices

1. **Check diagnostics**: Always run regression diagnostics
2. **Validate assumptions**: Check parallel trends for DiD, smoothness for RD
3. **Report robustness**: Run multiple specifications with different controls

---

## Quick Reference

| Task | CLI Command |
|------|-------------|
| Load CSV | `data load file.csv --name mydata` |
| Describe data | `data describe mydata` |
| OLS regression | `reg ols mydata -y y -x x1 x2` |
| Robust SEs | `reg ols mydata -y y -x x1 x2 --robust hc1` |
| Fixed effects | `panel fe mydata -y y -x x1 --entity id` |
| DiD | `causal did mydata -y y --treat t --post p` |
| ARIMA | `ts arima mydata --col y -p 1 -d 1 -q 1` |
| K-means | `ml kmeans mydata --cols x1 x2 -k 3` |
| Scatter plot | `viz scatter mydata -x x -y y -f plot.png` |
