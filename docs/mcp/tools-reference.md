# MCP Tools Reference

Total tools: 106

## Data

*Data loading, listing, describing, and export*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `list_datasets` | List all loaded datasets with dimensions and column types | ls() |
| `load_dataset` | Load data from CSV, Parquet, Excel, Stata, or SAS files | read.csv, read_parquet, read_dta |
| `export_dataset` | Export dataset to CSV, Parquet, or JSON format | write.csv, write_parquet |
| `create_dataset` | Create dataset from inline CSV content | - |
| `describe_dataset` | Get descriptive statistics for all columns | summary() |
| `head_dataset` | Preview first N rows of a dataset | head() |
| `compare_datasets` | Compare two datasets for differences | all.equal() |

## Cleaning

*Data quality profiling, cleaning sessions, rollback*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `data_quality_profile` | Generate quality profile: missing values, outliers, duplicates | - |
| `suggest_cleaning` | Get AI-powered cleaning suggestions with priorities | - |
| `preview_cleaning` | Preview cleaning operation without applying | - |
| `verify_cleaning` | Verify cleaning results match expectations | - |
| `cleaning_session_start` | Start a cleaning session with checkpoints | - |
| `cleaning_session_apply` | Apply a cleaning operation within session | - |
| `cleaning_rollback` | Rollback to previous checkpoint | - |

## Munging

*Data transformation: filter, join, pivot, mutate*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `munge_filter` | Filter rows based on conditions | dplyr::filter() |
| `munge_select` | Select specific columns | dplyr::select() |
| `munge_mutate` | Create or modify columns with expressions | dplyr::mutate() |
| `munge_join` | Join two datasets (left, inner, full) | dplyr::left_join() |
| `munge_pivot` | Reshape from long to wide format | tidyr::pivot_wider() |
| `munge_melt` | Reshape from wide to long format | tidyr::pivot_longer() |
| `munge_group_by` | Group data and compute aggregations | dplyr::group_by() %>% summarize() |
| `munge_drop_na` | Remove rows with missing values | tidyr::drop_na() |
| `munge_fill_na` | Fill missing values with specified strategy | tidyr::fill() |

## Descriptive

*Descriptive statistics, correlation, ANOVA*

| Tool | Description | R Equivalent |
|------|-------------|-------------|

## Statistics

*Hypothesis tests, power analysis, multivariate*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `hypothesis_t_test` | One-sample, two-sample, or paired t-test | t.test() |
| `hypothesis_wilcoxon` | Wilcoxon rank-sum or signed-rank test | wilcox.test() |
| `hypothesis_chisq_gof` | Chi-squared goodness of fit test | chisq.test() |
| `anova_one_way` | One-way ANOVA | aov() |
| `power_t_test` | Power analysis for t-tests | power.t.test() |

## Regression

*OLS, GLS, NLS, quantile, robust standard errors*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `regression_ols` | OLS regression with optional robust standard errors (HC0-HC3) | lm(), sandwich::vcovHC() |
| `regression_clustered` | OLS with clustered standard errors | sandwich::vcovCL() |
| `regression_hac` | HAC (Newey-West) standard errors for time series | sandwich::vcovHAC() |
| `regression_gls` | Generalized least squares with AR1 or custom correlation | nlme::gls() |
| `regression_nls` | Nonlinear least squares (Levenberg-Marquardt) | nls() |
| `regression_quantreg` | Quantile regression (interior point/simplex) | quantreg::rq() |
| `regression_loess` | Local polynomial regression (LOESS/LOWESS) | loess() |
| `regression_diagnostics` | Regression diagnostics: VIF, Breusch-Pagan, Durbin-Watson | car::vif(), lmtest::bptest() |

## Panel

*Fixed effects, random effects, Hausman, GMM, HDFE*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `panel_fixed_effects` | Fixed effects (within) estimator | plm::plm(model='within') |
| `panel_random_effects` | Random effects (GLS) estimator | plm::plm(model='random') |
| `hausman_test` | Hausman specification test (FE vs RE) | plm::phtest() |
| `panel_hdfe` | High-dimensional fixed effects (multiple FE) | fixest::feols() |
| `panel_gmm` | Arellano-Bond dynamic panel GMM | plm::pgmm() |

## IV

*Instrumental variables, 2SLS, Sargan, bounds*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `iv_2sls` | Two-stage least squares IV regression | AER::ivreg() |
| `iv_first_stage` | First-stage diagnostics: F-stat, partial R² | lmtest::waldtest() |
| `iv_sargan_test` | Sargan overidentification test | AER::summary.ivreg() |
| `bp_bounds` | Balke-Pearl bounds for IV with binary outcomes | bpbounds package |
| `iv_mte` | Marginal treatment effects for IV | ivmte package |

## DiD

*Difference-in-differences, staggered DiD, synthetic control*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `diff_in_diff` | Canonical 2x2 difference-in-differences | Basic DiD in lm() |
| `staggered_did` | Callaway-Sant'Anna staggered treatment DiD | did::att_gt() |
| `etwfe` | Extended TWFE (Wooldridge approach) | etwfe package |
| `bacon_decomp` | Goodman-Bacon TWFE decomposition | bacondecomp::bacon() |
| `synthetic_control` | Synthetic control method | Synth package |
| `gsynth` | Generalized synthetic control (matrix completion) | gsynth package |

## RD

*Regression discontinuity (sharp, fuzzy, multi-cutoff)*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `rd_estimate` | Sharp RD with CCT robust inference | rdrobust::rdrobust() |
| `rd_fuzzy` | Fuzzy RD design (two-stage) | rdrobust::rdrobust(fuzzy=) |
| `rd_bw` | Optimal bandwidth selection (MSE, CCT) | rdrobust::rdbwselect() |
| `rd_multi` | Multi-cutoff or multi-score RD | rdmulti package |

## Matching

*Propensity score matching, CEM, nearest neighbor*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `propensity_matching` | Propensity score matching (1:1, 1:k, caliper) | MatchIt::matchit() |

## Treatment

*IPW, doubly robust, TMLE, CBPS, entropy balancing*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `treatment_cbps` | Covariate balancing propensity scores | CBPS::CBPS() |
| `treatment_weightit` | Flexible propensity score weighting | WeightIt::weightit() |
| `treatment_entropy_balance` | Entropy balancing weights | ebal::ebalance() |
| `treatment_twang` | GBM-based propensity score estimation | twang::ps() |
| `treatment_ipw` | Inverse probability weighting | ipw package |
| `treatment_doubly_robust` | Doubly robust (AIPW) estimation | drtmle package |
| `treatment_tmle` | Targeted maximum likelihood estimation | tmle::tmle() |
| `collaborative_tmle` | Collaborative TMLE for high-dimensional data | ctmle package |
| `ltmle` | Longitudinal TMLE for time-varying treatments | ltmle::ltmle() |
| `treatment_double_ml` | Double/debiased machine learning | DoubleML package |

## Mediation

*Causal mediation, natural effects*

| Tool | Description | R Equivalent |
|------|-------------|-------------|

## Discrete

*Logit, probit, multinomial, ordered, count models*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `logit` | Logistic regression (binary outcome) | glm(family=binomial) |
| `probit` | Probit regression (binary outcome) | glm(family=binomial(probit)) |
| `multinom` | Multinomial logit for unordered categories | nnet::multinom() |
| `ordered_model` | Ordered logit/probit for ordinal outcomes | MASS::polr() |
| `negbin` | Negative binomial for overdispersed counts | MASS::glm.nb() |
| `zeroinfl` | Zero-inflated Poisson/negative binomial | pscl::zeroinfl() |
| `feglm` | GLM with high-dimensional fixed effects | fixest::feglm() |

## TimeSeries

*ARIMA, VAR, GARCH, Kalman, changepoint detection*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `ts_arima_fit` | Fit ARIMA(p,d,q) model | arima() |
| `ts_arima_forecast` | Generate ARIMA forecasts with confidence intervals | predict.Arima() |
| `ts_var` | Vector autoregression | vars::VAR() |
| `ts_var_irf` | Impulse response functions | vars::irf() |
| `ts_garch_fit` | GARCH(p,q) volatility model | rugarch::ugarchfit() |
| `ts_mstl` | Multiple seasonal decomposition (MSTL) | forecast::mstl() |
| `ts_changepoint` | Changepoint detection (PELT, binary segmentation) | changepoint::cpt.*() |

## Spatial

*SAR, SEM, Moran's I, spatial weights, panel spatial*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `spatial_neighbors` | Create spatial neighbors (k-NN, distance) | spdep::knearneigh() |
| `moran_test` | Moran's I test for spatial autocorrelation | spdep::moran.test() |
| `sar_model` | Spatial autoregressive lag model (SAR) | spatialreg::lagsarlm() |
| `sem_model` | Spatial error model (SEM) | spatialreg::errorsarlm() |
| `spatial_panel_ml` | Spatial panel ML estimation | splm::spml() |

## Survival

*Kaplan-Meier, Cox PH, AFT, competing risks*

| Tool | Description | R Equivalent |
|------|-------------|-------------|

## MachineLearning

*K-means, DBSCAN, PCA, t-SNE, Random Forest, SVM*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `ml_kmeans` | K-means clustering (k-means++ initialization) | kmeans() |
| `ml_dbscan` | Density-based clustering (DBSCAN) | dbscan::dbscan() |
| `ml_hierarchical` | Hierarchical clustering (Ward, complete, single) | hclust() |
| `ml_pca` | Principal component analysis | prcomp() |
| `ml_tsne` | t-SNE dimensionality reduction | Rtsne::Rtsne() |
| `ml_random_forest` | Random forest classification/regression | randomForest::randomForest() |
| `ml_causal_forest` | Causal forest for heterogeneous treatment effects | grf::causal_forest() |

## Visualization

*Histograms, scatter plots, heatmaps, interactive charts*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `viz_histogram` | Create histogram plot (PNG) | hist() |
| `viz_scatter` | Create scatter plot (PNG) | plot() |
| `viz_line` | Create line chart (PNG) | plot(type='l') |
| `viz_heatmap` | Create correlation heatmap | heatmap() |
| `viz_scatter_interactive` | Interactive scatter plot (Plotly HTML) | plotly::plot_ly() |
| `viz_coefficient` | Coefficient plot with confidence intervals | coefplot::coefplot() |

## Database

*SQLite and DuckDB queries, schema inspection*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `db_sqlite_query` | Execute SQL query on SQLite database | DBI::dbGetQuery() |
| `db_duckdb_query` | Execute SQL query on DuckDB (can query Parquet/CSV directly) | duckdb::dbGetQuery() |
| `db_query_file` | Query Parquet/CSV files directly with SQL | - |

## Utility

*Random seed, reports, session export/import*

| Tool | Description | R Equivalent |
|------|-------------|-------------|
| `set_seed` | Set random seed for reproducibility | set.seed() |
| `generate_random_data` | Generate synthetic dataset with various distributions | Various rnorm, runif, etc. |
| `generate_report` | Generate HTML report from analysis results | rmarkdown::render() |
| `export_session` | Export session state for later import | save(), saveRDS() |

