# MCP Tool Usage Examples

This guide provides usage examples for all 55 MCP tools available in prompt2analytics.

## Table of Contents
- [Data Management](#data-management)
- [Descriptive Statistics](#descriptive-statistics)
- [Regression Analysis](#regression-analysis)
- [Panel Data](#panel-data)
- [Instrumental Variables](#instrumental-variables)
- [Causal Inference](#causal-inference)
- [Discrete Choice Models](#discrete-choice-models)
- [Time Series](#time-series)
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
