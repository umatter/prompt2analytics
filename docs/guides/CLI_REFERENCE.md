# p2a CLI Reference

The `p2a` command-line interface provides direct access to all prompt2analytics capabilities for scripted workflows and reproducible analysis.

This page has a hand-written part (conventions, worked examples per task, output formats) and, at the end, the
[Command reference](#command-reference): every command, argument and option, generated from the binary's own help.

## Installation

```bash
cargo build --release -p p2a-cli
# Binary at: target/release/p2a
```

## Conventions

- **Sessions.** Commands that read a dataset need `--session <file>`, the session in which `data load` registered
  it. Without one they stop with `Error: No session active. Use --session <file> to enable dataset storage.`
  (exit 1). `script`, `chat` and `smoke-test` need no session.
- **Global options** (`-F/--format`, `--session`, `--quiet`, `--verbose`, `--max-memory`) can come before or after
  the subcommand. `-q` and `-v` are not global: inside subcommands they are the subcommand's own options
  (`ts arima -q` is the MA order, `panel unit-root -v` the variable).
- **Memory warning.** The session's datasets can be large, so a global option warns about them:
  `--max-memory <MB>` warns at most once per command, on stderr; no effect without --session.
  It compares the estimated memory of the session's datasets with MB MiB (0 = no limit, the default).
- **Output formats.** `-F text` (default), `-F json`, `-F table` (printed like `text` in this version).
- **Boolean options with a value.** Options such as `--intercept`, `--asc` or `ts stl --robust` take
  `true`/`false` (also y/yes/t/on/1 and n/no/f/off/0, any case); a bare `--intercept` means true. A bare boolean
  option directly before a positional argument takes that argument as its value
  (`reg ols --intercept mydata ...` fails with "invalid value 'mydata'"), so write `--intercept=true` or put the
  option after the positional arguments.
- **Exit codes.** 0 success, 1 an error the command reported (a missing dataset, column or session, a refused
  option value), 2 an invalid command line (clap: unknown command or option, a missing required argument).
- The complete list of commands, arguments and options is the generated [Command reference](#command-reference)
  at the end of this page; `p2a <command> --help` prints the long descriptions.

## Examples

Every example below names a session; load the dataset into it first (`data load ... --name <dataset>`).

### Data

```bash
p2a --session analysis.json data load /data/panel.csv --name firms
p2a --session analysis.json data list
p2a --session analysis.json data describe firms
p2a --session analysis.json data head firms -n 20

# A CSV export that writes missing values as "."
p2a --session analysis.json data load stata_export.csv --name survey --na-strings .

# Save (the format comes from the extension unless --file-format is given; both are case-insensitive)
p2a --session analysis.json data save firms --output firms.parquet
p2a --session analysis.json data save firms --output firms.dat --file-format csv
```

`data load` reads CSV, Parquet, Excel, Stata and SAS files; the format comes from the extension. `data describe`
prints the row and column counts and each column's name and type.

**Missing values in CSV files.** `data load` reads the field `NA` and the empty field as missing,
as R's `read.csv(na.strings = c("NA", ""))` does. `--na-strings` replaces that list with its own,
comma-separated (`--na-strings .` for a Stata or SAS export, `--na-strings NA,,.` for all three);
an empty unquoted field is missing in every case. Write a marker that starts with `-` in the `=`
form, `--na-strings=-99`. The list is stored in the session file, so later commands of the same
session reload the file with the same markers. `--na-strings` on any file other than CSV is an
error (Parquet, Excel, Stata and SAS files store missing values natively).

### Data manipulation (munge)

`p2a munge <COMMAND> <DATASET> ...` writes its result to a new dataset (the name flag differs by command; see the
command reference).

```bash
# Bin a numeric column into quartiles, as R's cut()
p2a --session analysis.json munge bin firms -c revenue -m quantile -n 4 -o firms_binned

# Convert a column to another type; --strict refuses values that cannot be converted
p2a --session analysis.json munge cast firms -c year --dtype int --strict

# Fill missing values
p2a --session analysis.json munge fill-na firms -m mean -c revenue employees
p2a --session analysis.json munge fill-na firms -m constant --value 0 -c capital
```

- `munge bin`: intervals are closed on the right, `(a, b]`, as `cut()`; `--left-closed` closes
  them on the left, `[a, b)` (`cut(right = FALSE)`), and `--include-lowest` puts a value equal
  to the lowest edge in the first bin (the highest edge in the last bin with `--left-closed`),
  as `cut(include.lowest = TRUE)`. `--edges` gives the strictly increasing break points of
  `-m custom`. A missing value, or one outside the edges, is binned as missing.
- `munge cast`: values that cannot be converted become missing, and the command reports how
  many; `--strict` refuses the cast instead.
- `munge fill-na`: every strategy fills NaN cells as well as nulls.

### Regression

```bash
# OLS with HC1 robust standard errors (the default)
p2a --session analysis.json reg ols mydata -y price -x sqft bedrooms bathrooms --robust hc1

# Classical standard errors, as R's summary(lm())
p2a --session analysis.json reg ols mydata -y price -x sqft bedrooms bathrooms --robust standard

# Regression through the origin (uncentered R-squared, as lm(y ~ 0 + x))
p2a --session analysis.json reg ols mydata -y price -x sqft --intercept false

# Clustered by firm; two-way clustering with --cluster2
p2a --session analysis.json reg clustered mydata -y revenue -x employees r_and_d --cluster firm_id
p2a --session analysis.json reg clustered mydata -y revenue -x employees r_and_d --cluster firm_id --cluster2 year

# HAC (Newey-West) standard errors
p2a --session analysis.json reg hac mydata -y returns -x mkt_rf smb hml --lags 4

# Quantile regression
p2a --session analysis.json reg quantreg mydata -y wage -x education experience --tau 0.25

# Diagnostics (Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF)
p2a --session analysis.json reg diagnostics mydata -y price -x sqft bedrooms
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

- `--intercept false` fits the model through the origin; R-squared is then R's uncentered one, labelled
  `R-squared (uncentered)` (JSON `r_squared_type: "uncentered"`). `reg hac` and `reg gls` JSON carry the
  coefficient names (`names`).
- `reg quantreg --method` is `auto` (default: Barrodale-Roberts up to 5,000 rows, Frisch-Newton above), `br`,
  `fn` or `irls`; `--se` is `nid` (default at every n), `iid`, `ker` or `boot` (`--seed` for the bootstrap).
  `rank` is not offered.
- `reg diagnostics` lists the variance inflation factors of the regressors only: the intercept
  has none, and `car::vif` omits it too. The `-F json` output keeps the intercept entry with
  `null`.
- The bootstrap covariance is an MCP tool (`regression_bootstrap_cov`), not a CLI command.

### Panel data

```bash
# Within-firm fixed effects
p2a --session analysis.json panel fe firms -y revenue -x employees capital --entity firm_id

# Firm-clustered (Arellano) standard errors
p2a --session analysis.json panel fe firms -y revenue -x employees capital --entity firm_id --vcov cluster

# Random effects model
p2a --session analysis.json panel re firms -y revenue -x employees capital --entity firm_id

# Hausman test to choose between FE and RE
p2a --session analysis.json panel hausman firms -y revenue -x employees capital --entity firm_id

# Two-way fixed effects (firm + year)
p2a --session analysis.json panel hdfe firms -y revenue -x employees --fe firm_id year

# Two-step difference GMM with one lag of the dependent variable, collapsed instruments
p2a --session analysis.json panel gmm firms -y inv -x value capital --entity firm --time year --collapse

# Panel unit-root test with per-unit lag selection
p2a --session analysis.json panel unit-root firms -v revenue --unit firm_id --time year --test ips --lag-selection sic
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
- `panel gmm`: Arellano-Bond difference GMM, as R's `plm::pgmm` (`effect = "individual"`).
  `--lags` (default 1) lags of the dependent variable are added to `-x`; the instruments are the
  levels of y lagged `--min-lag` (default 2) to `--max-lag` (default: all available), one column
  per lag depth with `--collapse`. `--step` is `twostep` by default; `--vcov robust` (default)
  gives the one-step robust or the two-step Windmeijer-corrected standard errors, `--vcov
  classical` the classical ones (these values differ from `panel fe`'s `--vcov`). The time column
  holds whole numbers, and lags are steps of 1. It prints the coefficient table, the
  Sargan/Hansen test of the overidentifying restrictions and the Arellano-Bond AR(1) and AR(2)
  tests of serial correlation in the differenced residuals, with warnings when a weighting
  matrix is rank-deficient; the JSON adds the homoskedastic Sargan statistic. The JSON keys:
  `coefficients`, `variables`, `std_errors`, `z_stats`, `p_values`, `se_type`, `sigma2`,
  `ar1_statistic`, `ar1_p_value`, `ar2_statistic`, `ar2_p_value`, `sargan_statistic`,
  `sargan_df`, `sargan_p_value`, `sargan_homoskedastic`, `sargan_homoskedastic_p_value`,
  `one_step_weight_rank`, `two_step_weight_rank`, `n_obs`, `n_groups`, `n_instruments`, `lags`,
  `min_lag`, `max_lag`, `collapse`, `step`, `transform`, `method`, `dep_var`, `warnings` and
  `missing`. System (Blundell-Bond) GMM is the MCP tool `panel_gmm` with `transform:system`.
- `panel unit-root`: the LLC, IPS, Maddala-Wu (Fisher) or Hadri test, as `plm::purtest`; rows with a
  missing value are dropped, and a unit's remaining periods are joined in time order with a
  warning. `--lags` fixes every unit's lag order (default floor(T_max^(1/3))); `--lag-selection aic|sic|hall`
  chooses it per unit up to `--pmax` (default 10), as `purtest(lags = ..)`. The JSON keys: `test_type`,
  `statistic`, `p_value`, `null_hypothesis`, `alternative_hypothesis`, `model`, `lags_used`, `n_panels`,
  `avg_time_periods`, `warnings`, `unit_statistics` and `unit_p_values` (per unit, where the test has them)
  and `missing`.

Every panel command drops the rows with a missing value in the columns it uses. `panel hdfe`,
`feglm`, `gmm`, `pvcm`, `pmg`, `gls` and `unit-root` print the `Note:` line last and carry
`missing` in JSON (see [Missing values in results](#missing-values-in-results)); `panel fe` and
`re` list the drop under `Warnings:` and in the JSON `warnings`.

### Causal inference

```bash
# 2SLS with an instrument
p2a --session analysis.json causal iv mydata -y wage --exog experience --endog education --instruments parent_education

# Difference-in-differences
p2a --session analysis.json causal did mydata -y outcome --treat treated --post post_treatment

# Synthetic control (California tobacco study style), with placebo tests for inference
p2a --session analysis.json causal synth smoking -y cigsale --unit state --time year --treated California \
    --treatment-time 1988 -p lnincome retprice
p2a --session analysis.json causal synth mydata -y gdp --unit country --time year --treated Germany \
    --treatment-time 1990 -p exports imports --placebos

# Sharp RD: the default MSE-optimal bandwidth, a coverage-error-optimal one, or a fixed one with HC1 SEs
p2a --session analysis.json causal rd mydata -y test_score --running age --cutoff 65
p2a --session analysis.json causal rd mydata -y test_score --running age --cutoff 65 --bwselect cerrd
p2a --session analysis.json causal rd mydata -y outcome --running running_var --cutoff 0 --h 2.5 --vce hc1

# Fuzzy RD: treatment probability changes at the cutoff
p2a --session analysis.json causal fuzzy-rd mydata -y earnings --treatment enrolled --running score --cutoff 50

# Callaway-Sant'Anna staggered DiD, bootstrap and analytic standard errors
p2a --session analysis.json causal staggered-did mydata -y outcome --treatment-time first_treat --time year --unit state
p2a --session analysis.json causal staggered-did panel -y y --treatment-time first_treat --time year --unit state --bootstrap 0

# Propensity-score matching, inverse probability weighting
p2a --session analysis.json causal matching mydata --treatment treated -x age income education --method nearest
p2a --session analysis.json causal ipw mydata -y outcome --treatment treated -x age income
```

**V-method options (`causal synth`):** `datadriven` (default), `equal`

**RD kernel types:** `triangular` (default; also `tri`), `epanechnikov` (`epa`), `uniform`
(`uni`), in any case; another name is an error

**RD bandwidths (`causal rd`, `causal fuzzy-rd`):** `--bwselect` takes rdrobust's selector names (mserd,
msetwo, msesum, msecomb1, msecomb2, cerrd, certwo, cersum, cercomb1, cercomb2; default mserd); another name
is an error. `--h <H>` and `--b <B>` fix the bandwidths instead: `--h` is the main bandwidth on both sides
(alias `--bandwidth`), `--b` the bias bandwidth (alias `--bias-bandwidth`; default: h). `--vce` is the
variance estimator: `nn` (default, as rdrobust), `hc0`, `hc1`, `hc2` or `hc3`. The output reports the robust
bias-corrected estimate and the conventional one, as `rdrobust`, with the bandwidths and the
effective observations on each side.

**`causal iv`** fits robust standard errors and prints the first-stage diagnostics as
`First-stage diagnostics (HC1 Wald F of the excluded instruments):`, one
`<endog> F-statistic: F on df1 and df2 DF` line per endogenous regressor; its JSON adds a
`first_stage` object (`f_type`, `f_stats`, `df`, `p_values`, `strong_instruments`).

**`causal staggered-did`:** `--bootstrap 0` gives analytic influence-function standard errors
(did's `bstrap = FALSE`), with or without covariates; otherwise at least 50 bootstrap
replicates (default 999): a unit bootstrap, or with covariates a multiplier bootstrap of the
influence functions. The JSON's `pretrend_test` holds `chi2`, `df`, `p_value`, `f_stat`
and `df2` (the F form of a bootstrap covariance; `null` with the analytic one) and `method`.

**`causal ipw`:** `--bootstrap 0` (default) gives the M-estimation standard error with the propensity score
estimated, as WeightIt; N >= 2 a bootstrap of the weighted means with the fitted scores held fixed.

`causal iv`, `mediation`, `rd`, `fuzzy-rd` and `staggered-did` drop the rows with a missing value
in the columns they use and report them (`Note:` line and `missing`).

### Discrete choice

```bash
p2a --session analysis.json discrete logit mydata -y purchased -x income age education
p2a --session analysis.json discrete probit mydata -y voted -x age income party_id

# Ordered logit (a text outcome needs its order, lowest first)
p2a --session analysis.json discrete o-logit survey -y satisfaction -x age income --levels low,mid,high

# Multinomial logit with a reference category
p2a --session analysis.json discrete mlogit trips -y mode -x income distance --base car

# Conditional logit without alternative-specific constants; mixed logit with repeated choices
p2a --session analysis.json discrete clogit trips --choice-id trip --alt-id mode -y chosen -x cost time --asc false
p2a --session analysis.json discrete mixed-logit trips --choice-id trip --alt-id mode -y chosen -x cost time --random-vars cost --individual person
```

`logit`, `probit`, `mlogit` (multinomial), `o-logit`, `o-probit`, `negbin`, `zip`, `zinb`
and `hurdle` drop rows with a missing outcome or regressor and say so (see
[Missing values in results](#missing-values-in-results)); `clogit` and `mixed-logit`
refuse a missing id, alternative, choice or attribute, as R's mlogit does. The ordered,
count and mixed-logit commands print `Converged: Yes|No (k iterations)`. `mlogit --base` names the
reference category (default: the lowest value); `clogit` fits alternative-specific constants by default, as
mlogit does (`--asc false` drops them), with `--reference` (alias `--reflevel`) as the reference alternative.

### Survival

```bash
p2a --session analysis.json survival km lung -t time -e status --conf-type log-log
p2a --session analysis.json survival log-rank lung -t time -e status -g sex
p2a --session analysis.json survival cox lung -t time -e status -x age sex --ties efron
p2a --session analysis.json survival cox lung -t time -e status -x age sex --cluster inst
```

The status column `-e` is coded as R's `Surv()`: 0/1 (1 = event), 1/2 (2 = event, as R's `lung`)
or TRUE/FALSE; any other code is an error. Integer times are accepted. `--conf-type` is R
`survfit`'s `conf.type`: `log` (the default, as R), `log-log` or `plain`; the median follows R's
`quantile.survfit`. `survival cox --ties` is `efron` (default, as `coxph`) or `breslow`; `--robust` gives the
Lin-Wei sandwich standard errors and `--cluster <col>` clustered ones (implies `--robust`). Rows with a missing
time, status, group or covariate are dropped and reported (`Note:` line, `missing`).

JSON keys: `survival km` prints one object per group with `n_obs`, `total_events`,
`total_censored`, `conf_level`, `conf_type`, `median_survival`, `median_ci`, `times`,
`n_at_risk`, `n_events`, `n_censored`, `survival`, `std_errors`, `ci_lower`, `ci_upper` (a
value that is not defined, such as the SE where S(t) = 0, is `null`) and `missing`;
`survival log-rank` has `chi_squared`, `degrees_of_freedom`, `p_value`, `groups`,
`n_per_group`, `events_per_group`, `expected_per_group`, `obs_minus_exp`, `variance` and
`missing`; `survival cox` has `variables`, `coefficients`, `hazard_ratios`, `std_errors`,
`z_stats`, `p_values`, `log_likelihood`, `concordance`, `n_obs`, `n_events`, `method` and
`missing`.

### Time series

```bash
# ARIMA(1,1,1) with a 12-step forecast
p2a --session analysis.json ts arima sales --col revenue -p 1 -d 1 -q 1 --horizon 12

# Seasonal decomposition (weekly + yearly)
p2a --session analysis.json ts mstl sales --col revenue --periods 7 365

# VAR(2) and its impulse responses
p2a --session analysis.json ts var macro --cols gdp inflation unemployment --lags 2
p2a --session analysis.json ts irf macro --cols gdp inflation unemployment --lags 2 --steps 20

# VECM with a constant in the cointegration relation
p2a --session analysis.json ts vecm macro --cols gdp consumption investment --lags 2 --ecdet const

# VARMA(1,1) with a longer step-1 VAR
p2a --session analysis.json ts varma macro --cols gdp inflation -p 1 -q 1 --long-var-order 8

# GARCH(1,1) with Student-t errors and a 10-step variance forecast
p2a --session analysis.json ts garch returns --col ret --distribution std --horizon 10

# Holt-Winters with 90% prediction intervals
p2a --session analysis.json ts holt-winters sales --col revenue --period 12 --horizon 12 --level 0.9
```

**`ts arima`** fits by conditional sum of squares, as R's `arima(method = "CSS")`. The mean (R's
`intercept`) is estimated by default when d = 0; `--include-mean false` drops it, and it must be
false or omitted when d > 0. `--drift` fits a linear trend in the levels, as
`forecast::Arima(include.drift = TRUE)`, for d = 0 or 1. With `--horizon` it prints point
forecasts with 80% and 95% prediction intervals, as `forecast(Arima(...))`. The JSON keys include
`model`, `label` (forecast::Arima's model label), `coefficients`, `std_errors`, `sigma2`, `loglik`,
`aic`, `converged`, `iterations`, `include_mean`, `drift`, `warnings` and, with `--horizon`, the
forecasts with `lower_80`, `upper_80`, `lower_95` and `upper_95`. A series with missing values
is refused, with how to fill the gaps first (`munge fill-na`): dropping a value would splice
the series.

- `ts vecm` takes `--lags` (the VAR order K in levels, at least 2; default 2), `--ecdet none|const|trend` and an
  optional `--rank` (default: the trace test's rank at 5%), and prints the trace and maximum-eigenvalue tables.
- `ts varma --long-var-order` sets the order of the Hannan-Rissanen step-1 VAR (default
  max(floor(2 ln n), 2 max(p, q))); set it larger when an MA root is near the unit circle.
- `ts garch --distribution` is `norm` (default) or `std` (standardised Student t); `--horizon` prints the
  conditional-variance forecasts.
- `ts holt-winters --horizon` prints forecasts with prediction intervals (point forecasts only for a
  multiplicative seasonal fit); `--level`, strictly between 0 and 1 (default 0.95), needs `--horizon`.

### Statistics

```bash
p2a --session analysis.json stats t-test-two trial --col1 treated --col2 control
p2a --session analysis.json stats anova-two crops -r yield --factor-a fertilizer --factor-b irrigation --interaction --ss-type II
p2a --session analysis.json stats fisher survey --row-var smoker --col-var outcome
p2a --session analysis.json stats acf sales -c revenue --lag-max 24
```

Missing values are dropped as R drops them, and every `stats` data command reports the drop:
text output ends with the `Note:` line and JSON carries `missing` (see
[Missing values in results](#missing-values-in-results)). Integer columns are numeric. A
two-sample test (`t-test-two`, the rank-sum `wilcoxon`, `ks-test`) drops each sample's missing
values on its own, so its note counts values: `Note: 30 of 44 values dropped because of missing
values (two samples of 22; x: 15, y: 15)`. Paired tests drop incomplete pairs. `acf` and `pacf`
refuse a column with missing values, as R's `acf`/`pacf` (`na.action = na.fail`), and say how to
fill the gaps first; `ccf`, `box-test` and `pp-test` refuse them too.

- `t-test-two` prints `Means: x = ..., y = ...` and `Mean difference: ...`; its JSON has
  `mean_x`, `mean_y`, `mean_difference` (x minus y), `n_x`, `n_y`, `std_error`, `t_statistic`,
  `df`, `p_value`, `ci_lower` and `ci_upper`. `t-test-one` and `t-test-paired` add `std_error`
  and `n`.
- `anova-two --ss-type`: `I` (default; sequential, as R's `aov`), `II` or `III` (as
  `car::Anova`; Type III with sum-to-zero contrasts). A term the data cannot estimate is shown
  as `not estimable (aliased with the other factor)` or, when empty cells leave the interaction
  no degrees of freedom, `not estimable (empty cells leave no interaction df)`.
- `chisq-indep` and `fisher` print the labelled table (observed and expected counts for
  `chisq-indep`), up to 50 rows and 12 columns; a larger table is summarised as
  `Table not shown: ...`. `chisq-indep` states `Continuity correction: Yates` on a 2 x 2 table
  and R's `Warning:` lines (expected counts below 5). `fisher` reports the conditional maximum
  likelihood odds ratio, as R's `fisher.test`, next to the sample odds ratio.
- `fivenum`'s JSON has `n`, the values used.

### Machine learning

```bash
# Cluster customers into 5 segments
p2a --session analysis.json ml kmeans customers --cols income spending age -k 5 --seed 42

# PCA with the top 3 components and their scores
p2a --session analysis.json ml pca customers --cols var1 var2 var3 var4 var5 --n-components 3 --transform

# Hierarchical clustering (ward means ward.D2)
p2a --session analysis.json ml hierarchical customers --cols income spending age -n 4 --linkage ward
```

`ml svm` drops the rows with a missing feature or target and prints the `Note:` line (`missing`
in JSON); `kmeans` and `pca` refuse missing values. `ml pca --transform` prints the principal-component scores
(the first 10 rows in text, all in `-F json`); `--scores-file <CSV>` writes every score to a file.

### Visualization

```bash
# Basic distribution plots
p2a --session analysis.json viz histogram mydata --col price -f price_dist.png --bins 50
p2a --session analysis.json viz scatter mydata -x income -y spending -f scatter.png --title "Income vs Spending"
p2a --session analysis.json viz line timeseries -x date -y value -f trend.png

# Box plot by category
p2a --session analysis.json viz box sales -y revenue -g region -f revenue_by_region.png

# Correlation heatmap for selected variables
p2a --session analysis.json viz heatmap mydata --cols price sqft bedrooms bathrooms -f correlations.png

# Coefficient plot from regression results
p2a --session analysis.json viz coefplot mydata -y price -x sqft bedrooms bathrooms -f coef_plot.png --conf-level 0.95

# Residual diagnostics (creates 4 plots: main, _qq, _scale_location, _leverage)
p2a --session analysis.json viz residuals mydata -y price -x sqft bedrooms -f residual_diag.png

# Dendrogram for clustering analysis
p2a --session analysis.json viz dendrogram customers --cols spending income age -f dendrogram.png --linkage ward

# Event study plot (pre-prepared coefficient data)
p2a --session analysis.json viz event-study event_coeffs --time-col period --estimate-col coef --ci-lower-col ci_lo --ci-upper-col ci_hi -f event_study.png

# IRF plot from VAR analysis
p2a --session analysis.json viz irf irf_data --horizon-col h --response-col response --ci-lower-col lower --ci-upper-col upper -f irf.png
```

The data commands drop the rows that are incomplete in the plotted columns, together, and print
the `Note:` line; integer columns are accepted. `viz box -g` orders the groups as R's `factor()`
(numbers numerically, text in byte order). `viz heatmap` correlates the complete rows, as R's
`cor(use = "complete.obs")`: a column that is constant on them has NA correlations (shown as
`NA` in its cells; its diagonal stays 1), and the message names it. `viz dendrogram` refuses
missing values, and `viz irf` refuses a CI column it cannot read as numbers. With `-F json` a
plot command prints `{"message": ...}`, plus `missing` when rows were dropped.

### Scripts

```bash
# Record a session
p2a --session analysis.json data load data.csv --name d
p2a --session analysis.json reg ols d -y y -x x1 x2

# Export to a reproducible script, and run it
p2a script export analysis.json -o analysis.sh
p2a script run analysis.sh
```

## Output formats

### Text (default)
Human-readable formatted output with tables and summaries.

### JSON (`-F json`)
Structured JSON output for programmatic use:
```bash
p2a --session analysis.json -F json reg ols mydata -y y -x x1 x2 | jq '.coefficients'
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

### Table (`-F table`)
Printed like `text` in this version; a separate table rendering is not implemented.

### Missing values in results
A method that drops rows with missing values in the columns it uses reports them. Text and table
output end with one line, `Note: 3 of 120 rows dropped because of missing values (y: 2, x1: 1)`
(rows dropped, rows in the dataset, and the missing cells per column; a row missing in two columns
counts in both). JSON output has the key `missing` instead,
`{"n_total": 120, "n_used": 117, "n_dropped": 3, "by_column": {"y": 2, "x1": 1}}`. Neither appears
when no row was dropped.

## Session recording

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

What is recorded: each command from its subcommand on, as typed, without the global options
before the subcommand and without `--session` (wherever it appears before a `--`). The exported
script supplies its own session file, `p2a --session "$SESSION_FILE" <command>`. Global options
typed after the subcommand (`-F json`, `--quiet`) are recorded and replayed. The commands
`script`, `chat` and `smoke-test` are not recorded.

## Error handling

Errors are displayed with context:
```
Error: Dataset 'missing' not found
Error: Column 'unknown' not found in dataset
Error: Regression failed: Matrix is singular
```

With `-F json`, errors are written to stderr as JSON:
```json
{"error": "Dataset 'missing' not found"}
```

## Where p2a's defaults differ from R

p2a uses R as its test oracle and follows R's conventions where that is cheap. Where a default was kept as
p2a's own, it is listed here (CLI and MCP alike), with R's default and the document that records it.

| Method | p2a default | R default (function) | Recorded in |
|---|---|---|---|
| Quantile regression SEs | `nid` at every n | `rank` below 1,001 rows, `nid` above (`summary.rq`) | `validation/regression/quantreg.md` |
| Quantile regression algorithm | `auto`: Barrodale-Roberts up to 5,000 rows, Frisch-Newton above | `br` at every n (`rq`) | `validation/regression/quantreg.md` |
| General GMM weight (HAC) | Bartlett kernel, bandwidth lags + 1, no prewhitening | Quadratic Spectral kernel, Andrews bandwidth, VAR(1) prewhitening (`gmm`) | `validation/econometrics/general_gmm.md` |
| CBPS estimand | ATE | ATT (`CBPS::CBPS`) | `validation/econometrics/cbps.md` |
| ETWFE standard errors | clustered by unit (CR1) | iid (`fixest` as called by `etwfe`) | `validation/econometrics/etwfe.md` |
| Driscoll-Kraay t tests | T - rank degrees of freedom | n - rank (`coeftest` with `sandwich::vcovPL`) | `validation/regression/ols_robust_se.md` |
| Panel unit-root lags | floor(T_max^(1/3)) for every unit; `dfcor = TRUE` | SIC with `pmax` 10; `dfcor = FALSE` (`plm::purtest`) | `validation/econometrics/panel_unit_root.md` |
| Causal forest `mtry` | ceil(sqrt(p)) | a Poisson draw with mean sqrt(p) + 20 per tree (`grf`) | `validation/ml/causal_forest.md` |
| Affinity propagation | damping 0.5, 200 iterations, convergence after 15 | 0.9, 1000, 100 (`apcluster`) | `validation/ml/affinity_propagation.md` |
| Gaussian mixture stopping rule | relative log-likelihood change below 1e-8 | 1e-5 (`mclust`) | `validation/ml/gmm.md` |
| Discrete-choice and GLM fits (`MleSettings`, `GlmSpec`) | `max_iter` 100, tolerance 1e-10 | 25, 1e-8 (`glm.control`) | `validation/econometrics/logit.md` |
| Hierarchical clustering `ward` | ward.D2 | ward.D (the deprecated `hclust(method = "ward")`) | `validation/ml/hclust.md` |
| LTMLE variance | influence curve (`variance.method = "ic"`) | `"tmle"` substitution variance (`ltmle`) | `validation/econometrics/ltmle.md` |

Options that are refused rather than approximated: `reg quantreg --se rank` (MCP `se:rank`), System GMM with
non-robust standard errors, prediction intervals of a multiplicative Holt-Winters fit (point forecasts only),
spatial-panel random effects with time or two-way effects and the random-effects GM estimator, a medflex
treatment that is not coded 0/1, and a `balance_penalty` other than the default in the weighting tools. Each
refusal names the accepted values.

<!-- BEGIN GENERATED: p2a command reference (cargo run -p p2a-cli --example generate_cli_reference; do not edit by hand) -->
## Command reference

Rendered from this build's `p2a <command> -h`; `p2a <command> --help` prints the long descriptions.

### Global options

Every command accepts these, before or after the subcommand.

| Option | Description |
|---|---|
| `-F, --format <FORMAT>` | Output format: text, json, or table [default: text] [possible values: text, json, table] |
| `--session <SESSION>` | Session file for recording commands (enables reproducibility) |
| `--quiet` | Suppress non-essential output |
| `--verbose...` | Enable verbose logging on stderr (--verbose for info, twice for debug, three times for trace) |
| `--max-memory <MB>` | Warn on stderr, at most once per command, when the estimated memory of the --session datasets exceeds this many MB (MiB); 0 = no limit [default: 0] |
| `-h, --help` | Print help (see more with '--help') |
| `-V, --version` | Print version |

### Commands

| Command | Aliases | Description |
|---|---|---|
| `data` |  | Data loading and inspection |
| `munge` |  | Data munging (filter, join, reshape, clean, aggregate) |
| `regression` | `reg` | Regression analysis |
| `panel` |  | Panel data estimation |
| `causal` |  | Causal inference methods |
| `discrete` |  | Discrete choice models |
| `stats` |  | Statistical tests and descriptive statistics |
| `timeseries` | `ts` | Time series analysis |
| `survival` |  | Survival analysis |
| `spatial` |  | Spatial econometrics |
| `machine-learning` | `ml` | Machine learning |
| `visualize` | `viz` | Visualization (output to file) |
| `script` |  | Script generation and session management |
| `chat` |  | Interactive chat with a p2a-mcp server (natural language analytics) |
| `smoke-test` |  | Run a smoke test to verify the CLI works correctly |

### `p2a data`

Data loading and inspection

```text
p2a data [OPTIONS] <COMMAND>
```

| Command | Aliases | Description |
|---|---|---|
| `load` |  | Load a dataset from a file |
| `list` |  | List all loaded datasets |
| `describe` |  | Show dataset summary and statistics |
| `head` |  | Show first N rows of a dataset |
| `generate` |  | Generate random data with specified distributions |
| `save` |  | Save/export a dataset to a file |

#### `p2a data load`

Load a dataset from a file

```text
p2a data load [OPTIONS] <PATH>
```

| Argument or option | Description |
|---|---|
| `<PATH>` | Path to the data file (CSV, Parquet, Excel, Stata, SAS) |
| `-n, --name <NAME>` | Name to assign to the dataset |
| `--na-strings <LIST>` | Strings read as missing values in a CSV file, comma-separated (e.g. "NA,,." = NA, the empty field and "."). Replaces the default list, NA and the empty field (R read.csv); an empty unquoted field is missing in every case. Use --na-strings=-99 for a marker that starts with '-' |

#### `p2a data list`

List all loaded datasets

```text
p2a data list [OPTIONS]
```

#### `p2a data describe`

Show dataset summary and statistics

```text
p2a data describe [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |

#### `p2a data head`

Show first N rows of a dataset

```text
p2a data head [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-n, --n <N>` | Number of rows to display [default: 10] |

#### `p2a data generate`

Generate random data with specified distributions

```text
p2a data generate [OPTIONS] --columns <COLUMNS>
```

| Argument or option | Description |
|---|---|
| `-n, --rows <ROWS>` | Number of rows to generate [default: 100] |
| `-d, --name <NAME>` | Name to assign to the generated dataset [default: generated] |
| `-c, --columns <COLUMNS>` | Column specifications in JSON format Example: '[{"name": "x", "distribution": {"type": "normal", "mean": 0, "std": 1}}]' |
| `-s, --seed <SEED>` | Random seed for reproducibility |

#### `p2a data save`

Save/export a dataset to a file

```text
p2a data save [OPTIONS] --output <OUTPUT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-o, --output <OUTPUT>` | Output file path (format inferred from extension: .csv, .parquet, .json) |
| `-f, --file-format <FILE_FORMAT>` | File format (csv, parquet, json; any case); overrides the file extension [possible values: csv, parquet, json] |

### `p2a munge`

Data munging (filter, join, reshape, clean, aggregate)

```text
p2a munge [OPTIONS] <COMMAND>
```

| Command | Aliases | Description |
|---|---|---|
| `filter` |  | Filter rows based on a condition |
| `select` |  | Select specific columns |
| `drop` |  | Drop columns from a dataset |
| `rename` |  | Rename columns |
| `sort` |  | Sort by columns |
| `mutate` |  | Create or compute a new column |
| `sample` |  | Take a random sample of rows |
| `join` |  | Join two datasets |
| `concat` |  | Concatenate datasets vertically |
| `group-by` |  | Group by and aggregate |
| `value-counts` |  | Count value frequencies |
| `pivot` |  | Pivot from long to wide format |
| `melt` |  | Melt from wide to long format |
| `drop-na` |  | Drop rows with missing values |
| `fill-na` |  | Fill missing values |
| `deduplicate` |  | Remove duplicate rows |
| `lag` |  | Create lag of a column |
| `lead` |  | Create lead of a column |
| `diff` |  | Compute difference of a column |
| `standardize` |  | Standardize (z-score) columns |
| `bin` |  | Bin a numeric column into intervals, as R's cut() |
| `cast` |  | Convert a column to another type (reports values that fail to convert) |
| `one-hot` |  | One-hot encode a categorical column |

#### `p2a munge filter`

Filter rows based on a condition

```text
p2a munge filter [OPTIONS] --column <COLUMN> --op <OP> --value <VALUE> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --column <COLUMN>` | Column to filter on |
| `-o, --op <OP>` | Comparison operator (eq, ne, gt, ge, lt, le, contains, startswith, endswith) |
| `-v, --value <VALUE>` | Value to compare against |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge select`

Select specific columns

```text
p2a munge select [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --columns <COLUMNS>...` | Columns to select |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge drop`

Drop columns from a dataset

```text
p2a munge drop [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --columns <COLUMNS>...` | Columns to drop |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge rename`

Rename columns

```text
p2a munge rename [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-r, --renames <RENAMES>...` | Column renames in OLD=NEW format |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge sort`

Sort by columns

```text
p2a munge sort [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-b, --by <BY>...` | Columns to sort by |
| `-d, --desc` | Sort in descending order |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge mutate`

Create or compute a new column

```text
p2a munge mutate [OPTIONS] --new-col <NEW_COL> --expr <EXPR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-n, --new-col <NEW_COL>` | Name of the new column |
| `-e, --expr <EXPR>` | Expression: copy:COL, constant:VALUE, add:COL1:COL2, sub:COL1:COL2, mul:COL1:COL2, div:COL1:COL2 |
| `-N, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge sample`

Take a random sample of rows

```text
p2a munge sample [OPTIONS] --n <N> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-n, --n <N>` | Number of rows to sample |
| `--replace` | Sample with replacement |
| `--seed <SEED>` | Random seed for reproducibility |
| `-o, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge join`

Join two datasets

```text
p2a munge join [OPTIONS] --on <ON>... <LEFT> <RIGHT>
```

| Argument or option | Description |
|---|---|
| `<LEFT>` | Left dataset name |
| `<RIGHT>` | Right dataset name |
| `-o, --on <ON>...` | Join key columns (at least one) |
| `--right-on <RIGHT_ON>...` | Right key columns (if different from left) |
| `-t, --join-type <JOIN_TYPE>` | Join type [default: left] [possible values: left, right, inner, full] |
| `--suffix <SUFFIX>` | Suffix for duplicate column names [default: _right] |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge concat`

Concatenate datasets vertically

```text
p2a munge concat [OPTIONS] [DATASETS] [DATASETS]...
```

| Argument or option | Description |
|---|---|
| `[DATASETS] [DATASETS]...` | Dataset names to concatenate |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge group-by`

Group by and aggregate

```text
p2a munge group-by [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-b, --by <BY>...` | Columns to group by |
| `-a, --aggs <AGGS>...` | Aggregations in COLUMN:FUNCTION format (sum, mean, count, min, max, std, var, first, last, median) |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge value-counts`

Count value frequencies

```text
p2a munge value-counts [OPTIONS] --column <COLUMN> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --column <COLUMN>` | Column to count |
| `--normalize` | Add a proportion column, count / rows, missing values included (dplyr::count + n / sum(n)) |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge pivot`

Pivot from long to wide format

```text
p2a munge pivot [OPTIONS] --on <ON> --values <VALUES> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-i, --index <INDEX>...` | Index columns (row identifiers) |
| `-o, --on <ON>` | Column whose values become new column names |
| `-v, --values <VALUES>` | Column containing values |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge melt`

Melt from wide to long format

```text
p2a munge melt [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-i, --id-vars <ID_VARS>...` | ID columns to keep |
| `-v, --value-vars <VALUE_VARS>...` | Value columns to unpivot |
| `--var-name <VAR_NAME>` | Name for the variable column [default: variable] |
| `--val-name <VAL_NAME>` | Name for the value column [default: value] |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge drop-na`

Drop rows with missing values

```text
p2a munge drop-na [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --columns <COLUMNS>...` | Columns to check (all if not specified) |
| `--how <HOW>` | How to drop: any (default) or all [default: any] |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge fill-na`

Fill missing values

```text
p2a munge fill-na [OPTIONS] --method <METHOD> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-m, --method <METHOD>` | Fill method [possible values: mean, median, mode, constant, forward, backward, zero] |
| `-c, --columns <COLUMNS>...` | Columns to fill (all if not specified) |
| `--value <VALUE>` | Fill value for --method constant; typed to the column, as text for String columns |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge deduplicate`

Remove duplicate rows

```text
p2a munge deduplicate [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-s, --subset <SUBSET>...` | Columns to consider for duplicates (all if not specified) |
| `--keep <KEEP>` | Which duplicate to keep: first (default), last, or none [default: first] |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge lag`

Create lag of a column

```text
p2a munge lag [OPTIONS] --column <COLUMN> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --column <COLUMN>` | Column to lag |
| `-p, --periods <PERIODS>` | Number of periods to lag [default: 1] |
| `-g, --group-by <GROUP_BY>...` | Group by columns (for panel data) |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge lead`

Create lead of a column

```text
p2a munge lead [OPTIONS] --column <COLUMN> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --column <COLUMN>` | Column to lead |
| `-p, --periods <PERIODS>` | Number of periods to lead [default: 1] |
| `-g, --group-by <GROUP_BY>...` | Group by columns (for panel data) |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge diff`

Compute difference of a column

```text
p2a munge diff [OPTIONS] --column <COLUMN> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --column <COLUMN>` | Column to difference |
| `-p, --periods <PERIODS>` | Number of periods [default: 1] |
| `--pct` | Compute percentage change instead |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge standardize`

Standardize (z-score) columns

```text
p2a munge standardize [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --columns <COLUMNS>...` | Columns to standardize |
| `--normalize` | Normalize to 0-1 instead of z-score |
| `-n, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge bin`

Bin a numeric column into intervals, as R's cut()

```text
p2a munge bin [OPTIONS] --column <COLUMN> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --column <COLUMN>` | Column to bin |
| `-m, --method <METHOD>` | Binning method [default: equal-width] [possible values: equal-width, quantile, custom] |
| `-n, --n-bins <N_BINS>` | Number of bins [default: 10] |
| `--edges <EDGES>` | Strictly increasing break points for --method custom, comma-separated |
| `--left-closed` | Intervals closed on the left, [a, b) (R's cut(right = FALSE)) |
| `--include-lowest` | Include a value equal to the lowest edge in the first bin (the highest edge in the last bin with --left-closed), as R's cut(include.lowest = TRUE) |
| `-o, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge cast`

Convert a column to another type (reports values that fail to convert)

```text
p2a munge cast [OPTIONS] --column <COLUMN> --dtype <DTYPE> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --column <COLUMN>` | Column to cast |
| `--dtype <DTYPE>` | Target type: int, float, string or bool |
| `--strict` | Refuse the cast when any value cannot be converted |
| `-o, --name <NAME>` | Name for the resulting dataset |

#### `p2a munge one-hot`

One-hot encode a categorical column

```text
p2a munge one-hot [OPTIONS] --column <COLUMN> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --column <COLUMN>` | Column to encode |
| `--drop-first` | Drop first category (avoid multicollinearity) |
| `-n, --name <NAME>` | Name for the resulting dataset |

### `p2a regression` (alias `reg`)

Regression analysis

```text
p2a regression [OPTIONS] <COMMAND>
```

| Command | Aliases | Description |
|---|---|---|
| `ols` |  | Ordinary Least Squares regression |
| `clustered` |  | Clustered standard errors regression |
| `diagnostics` |  | Run regression diagnostics |
| `quantreg` |  | Quantile regression |
| `loess` |  | LOESS (local polynomial regression) |
| `nls` |  | Nonlinear Least Squares regression |
| `gls` |  | Generalized Least Squares regression |
| `step` |  | Stepwise regression model selection |
| `smooth-spline` |  | Smoothing spline regression |
| `supsmu` |  | Friedman's SuperSmoother |
| `line` |  | Tukey's resistant line (robust regression) |
| `hac` |  | HAC (Newey-West) standard errors for OLS |

#### `p2a regression ols`

Ordinary Least Squares regression

```text
p2a regression ols [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--intercept [<BOOL>]` | Include an intercept (true/false; default true). With false the model goes through the origin and R² is uncentered, as R's lm(y ~ 0 + x) [default: true] [possible values: true, false] |
| `-r, --robust <ROBUST>` | Standard errors: hc1 (default, heteroskedasticity-robust), hc0, hc2, hc3, or standard (classical, as R's summary(lm)) [default: hc1] [possible values: standard, hc0, hc1, hc2, hc3] |

#### `p2a regression clustered`

Clustered standard errors regression

```text
p2a regression clustered [OPTIONS] --dep-var <DEP_VAR> --cluster <CLUSTER> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--cluster <CLUSTER>` | Cluster variable column |
| `--cluster2 <COLUMN>` | Second cluster variable: two-way clustered SEs (Cameron, Gelbach and Miller 2011), as sandwich::vcovCL(cluster = ~c1 + c2, type = "HC1") |
| `--intercept [<BOOL>]` | Include an intercept (true/false; default true). With false the model goes through the origin and R² is uncentered, as R's lm(y ~ 0 + x) [default: true] [possible values: true, false] |

#### `p2a regression diagnostics`

Run regression diagnostics

```text
p2a regression diagnostics [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |

#### `p2a regression quantreg`

Quantile regression

```text
p2a regression quantreg [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `-t, --tau <TAU>` | Quantile (tau) value (default: 0.5 = median) [default: 0.5] |
| `--method <METHOD>` | Algorithm: auto (default: br up to 5,000 rows, fn above; rq's default is br at every n), br (Barrodale-Roberts simplex), fn (Frisch-Newton interior point) or irls (an approximation) [default: auto] |
| `--se <SE>` | Standard errors as summary.rq: nid (default at every n), iid, ker, boot [default: nid] |
| `--seed <SEED>` | Seed of the xy-pair bootstrap (--se boot only) |
| `--intercept [<BOOL>]` | Include an intercept (true/false; default true). With false the model goes through the origin and R² is uncentered, as R's lm(y ~ 0 + x) [default: true] [possible values: true, false] |

#### `p2a regression loess`

LOESS (local polynomial regression)

```text
p2a regression loess [OPTIONS] --dep-var <DEP_VAR> --indep-var <INDEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-var <INDEP_VAR>` | Independent variable column (single predictor) |
| `--span <SPAN>` | Span (smoothing parameter, 0-1, default: 0.75) [default: 0.75] |
| `--degree <DEGREE>` | Polynomial degree (1 = local linear, 2 = local quadratic) [default: 2] |

#### `p2a regression nls`

Nonlinear Least Squares regression

```text
p2a regression nls [OPTIONS] --dep-var <DEP_VAR> --indep-var <INDEP_VAR> --start <START> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-var <INDEP_VAR>` | Independent variable column (single predictor) |
| `--model <MODEL>` | Model type: exponential_decay (y = a \* exp(-b \* x) + c), exponential_growth, michaelis_menten, logistic_growth, power, asymptotic [default: exponential_decay] |
| `--start <START>` | Comma-separated start values, as many as the model has parameters (R's nls needs them) |

#### `p2a regression gls`

Generalized Least Squares regression

```text
p2a regression gls [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--correlation <CORRELATION>` | Correlation structure: ar1_auto (default; AR(1) with rho estimated), ar1, compound_symmetry (cs) or identity [default: ar1_auto] |
| `--rho <RHO>` | Correlation parameter (rho for ar1/cs; estimated when not given; refused with ar1_auto and identity) |
| `--method <METHOD>` | Likelihood for rho and the reported fit: reml (default, as nlme) or ml [default: reml] |
| `--group <COL>` | Groups for the correlation (nlme form = ~ 1 \| g); compound symmetry without groups needs --rho |
| `--intercept [<BOOL>]` | Include an intercept (true/false; default true). With false the model goes through the origin and R² is uncentered, as R's lm(y ~ 0 + x) [default: true] [possible values: true, false] |

#### `p2a regression step`

Stepwise regression model selection

```text
p2a regression step [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Candidate predictor variables (upper scope) |
| `--required-vars [<REQUIRED_VARS>...]` | Variables always included (lower scope) |
| `--direction <DIRECTION>` | Direction: forward, backward, or both (default: both) [default: both] |
| `--use-bic [<BOOL>]` | Use BIC (k = log n) instead of AIC (true/false; default false) [default: false] [possible values: true, false] |
| `--intercept [<BOOL>]` | Include an intercept (true/false; default true). With false the model goes through the origin and R² is uncentered, as R's lm(y ~ 0 + x) [default: true] [possible values: true, false] |

#### `p2a regression smooth-spline`

Smoothing spline regression

```text
p2a regression smooth-spline [OPTIONS] --dep-var <DEP_VAR> --indep-var <INDEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-var <INDEP_VAR>` | Independent variable column (single predictor) |
| `--df <DF>` | Equivalent degrees of freedom (if not set, --criterion chooses the smoothing) |
| `--criterion <CRITERION>` | Smoothing-parameter criterion when --df is not given: gcv (R's default cv = FALSE) or cv (leave-one-out, R's cv = TRUE) [default: gcv] |

#### `p2a regression supsmu`

Friedman's SuperSmoother

```text
p2a regression supsmu [OPTIONS] --dep-var <DEP_VAR> --indep-var <INDEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-var <INDEP_VAR>` | Independent variable column (single predictor) |
| `--span <SPAN>` | Fixed span (0-1), if not set uses cross-validation |
| `--bass <BASS>` | Bass parameter (0-10, higher = smoother) [default: 0] |

#### `p2a regression line`

Tukey's resistant line (robust regression)

```text
p2a regression line [OPTIONS] --dep-var <DEP_VAR> --indep-var <INDEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-var <INDEP_VAR>` | Independent variable column (single predictor) |
| `--iter <ITER>` | Number of polishing iterations [default: 1] |

#### `p2a regression hac`

HAC (Newey-West) standard errors for OLS

```text
p2a regression hac [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--lags <LAGS>` | Number of lags (if not set, uses automatic bandwidth) |
| `--intercept [<BOOL>]` | Include an intercept (true/false; default true). With false the model goes through the origin and R² is uncentered, as R's lm(y ~ 0 + x) [default: true] [possible values: true, false] |

### `p2a panel`

Panel data estimation

```text
p2a panel [OPTIONS] <COMMAND>
```

| Command | Aliases | Description |
|---|---|---|
| `fe` |  | Fixed Effects estimation |
| `re` |  | Random Effects estimation |
| `hausman` |  | Hausman test (Fixed vs Random Effects) |
| `hdfe` |  | High-dimensional Fixed Effects |
| `feglm` |  | Fixed Effects GLM (logit, probit, poisson with HDFE) |
| `gmm` |  | Arellano-Bond difference GMM for dynamic panels (as plm::pgmm) |
| `pvcm` |  | Panel Variable Coefficients Model (Swamy estimator) |
| `pmg` |  | Pooled Mean Group estimator |
| `gls` |  | Panel GLS (Feasible GLS) |
| `unit-root` |  | Panel unit root tests (LLC, IPS, Hadri) |

#### `p2a panel fe`

Fixed Effects estimation

```text
p2a panel fe [OPTIONS] --dep-var <DEP_VAR> --entity <ENTITY> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--entity <ENTITY>` | Entity (group) column for fixed effects |
| `--vcov <VCOV>` | Standard errors: classical (default, as plm), cluster (entity-clustered Arellano, Stata small-sample factor), cluster-hc0, cluster-hc1 [default: classical] [possible values: classical, cluster, cluster-hc0, cluster-hc1] |

#### `p2a panel re`

Random Effects estimation

```text
p2a panel re [OPTIONS] --dep-var <DEP_VAR> --entity <ENTITY> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--entity <ENTITY>` | Entity (group) column |
| `--vcov <VCOV>` | Standard errors: classical (default, as plm), cluster (entity-clustered Arellano, Stata small-sample factor), cluster-hc0, cluster-hc1 [default: classical] [possible values: classical, cluster, cluster-hc0, cluster-hc1] |

#### `p2a panel hausman`

Hausman test (Fixed vs Random Effects)

```text
p2a panel hausman [OPTIONS] --dep-var <DEP_VAR> --entity <ENTITY> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--entity <ENTITY>` | Entity (group) column |

#### `p2a panel hdfe`

High-dimensional Fixed Effects

```text
p2a panel hdfe [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--fe <FE>...` | Fixed effect columns |

#### `p2a panel feglm`

Fixed Effects GLM (logit, probit, poisson with HDFE)

```text
p2a panel feglm [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--fe <FE>...` | Fixed effect columns |
| `--family <FAMILY>` | GLM family: logit (default; also binomial), probit, poisson, gaussian [default: logit] [possible values: logit, probit, poisson, gaussian] |

#### `p2a panel gmm`

Arellano-Bond difference GMM for dynamic panels (as plm::pgmm)

```text
p2a panel gmm [OPTIONS] --dep-var <DEP_VAR> --entity <ENTITY> --time <TIME> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Strictly exogenous regressors (the lagged dependent variable is added by --lags) |
| `--entity <ENTITY>` | Entity (group) column for panel structure |
| `--time <TIME>` | Time column (whole numbers; lags are steps of 1) |
| `--lags <LAGS>` | Lags of the dependent variable [default: 1] |
| `--step <STEP>` | One-step or two-step GMM [default: twostep] [possible values: onestep, twostep] |
| `--min-lag <MIN_LAG>` | Shallowest lag of y used as instrument [default: 2] |
| `--max-lag <MAX_LAG>` | Deepest lag of y used as instrument (default: all available) |
| `--collapse` | One instrument column per lag depth |
| `--vcov <VCOV>` | robust (one-step robust, two-step Windmeijer) or classical [default: robust] [possible values: robust, classical] |

#### `p2a panel pvcm`

Panel Variable Coefficients Model (Swamy estimator)

```text
p2a panel pvcm [OPTIONS] --dep-var <DEP_VAR> --entity <ENTITY> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--entity <ENTITY>` | Entity (group) column |
| `--model <MODEL>` | Model type: within (default) or random (Swamy; also swamy) [default: within] [possible values: within, random] |

#### `p2a panel pmg`

Pooled Mean Group estimator

```text
p2a panel pmg [OPTIONS] --dep-var <DEP_VAR> --entity <ENTITY> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--entity <ENTITY>` | Entity (group) column |

#### `p2a panel gls`

Panel GLS (Feasible GLS)

```text
p2a panel gls [OPTIONS] --dep-var <DEP_VAR> --entity <ENTITY> --time <TIME> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--entity <ENTITY>` | Entity (group) column |
| `--time <TIME>` | Time column |
| `--model <MODEL>` | Model type: fe (fixed effects, default), fd (first-difference) or pooling [default: fe] [possible values: fe, fd, pooling] |

#### `p2a panel unit-root`

Panel unit root tests (LLC, IPS, Hadri)

```text
p2a panel unit-root [OPTIONS] --var <VAR> --unit <UNIT> --time <TIME> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-v, --var <VAR>` | Variable to test for unit root |
| `--unit <UNIT>` | Unit (entity) column |
| `--time <TIME>` | Time column |
| `--test <TEST>` | Test: llc (Levin-Lin-Chu; also levinlin), ips (Im-Pesaran-Shin), madwu (Maddala-Wu Fisher; also fisher) or hadri, as plm::purtest's test [default: llc] |
| `--model <MODEL>` | Deterministic terms: none, intercept (default) or trend, as purtest's exo [default: intercept] |
| `--lags <LAGS>` | Lag order of every unit (default floor(T_max^(1/3)) for every unit; plm::purtest's default is SIC with pmax 10); see --lag-selection |
| `--lag-selection <LAG_SELECTION>` | Lag order per unit: fixed (needs --lags), aic, sic or hall, as plm::purtest(lags = ..) |
| `--pmax <N>` | Largest lag the criteria try (default 10, purtest's pmax); needs --lag-selection aic, sic or hall |

### `p2a causal`

Causal inference methods

```text
p2a causal [OPTIONS] <COMMAND>
```

| Command | Aliases | Description |
|---|---|---|
| `iv` |  | Two-Stage Least Squares (Instrumental Variables) |
| `did` |  | Difference-in-Differences |
| `synth` |  | Synthetic Control Method (Abadie et al.) |
| `ipw` |  | Inverse Probability Weighting treatment effect |
| `doubly-robust` |  | Doubly robust treatment effect (AIPW) |
| `mediation` |  | Causal mediation analysis |
| `rd` |  | Regression Discontinuity (sharp design) |
| `fuzzy-rd` |  | Fuzzy Regression Discontinuity |
| `staggered-did` |  | Staggered Difference-in-Differences (Callaway-Sant'Anna) |
| `tmle` |  | Targeted Maximum Likelihood Estimation (TMLE) |
| `gsynth` |  | Generalized Synthetic Control (gsynth) |
| `matching` |  | Propensity Score Matching (MatchIt) |
| `stdreg` |  | Regression Standardization / G-computation (stdReg) |

#### `p2a causal iv`

Two-Stage Least Squares (Instrumental Variables)

```text
p2a causal iv [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `--exog <EXOG>...` | Exogenous variable columns |
| `--endog <ENDOG>...` | Endogenous variable columns |
| `--instruments <INSTRUMENTS>...` | Instrument columns |

#### `p2a causal did`

Difference-in-Differences

```text
p2a causal did [OPTIONS] --outcome <OUTCOME> --treat <TREAT> --post <POST> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --outcome <OUTCOME>` | Outcome variable column |
| `--treat <TREAT>` | Treatment indicator column |
| `--post <POST>` | Post-treatment period indicator column |
| `-x, --controls [<CONTROLS>...]` | Control variable columns (optional) |

#### `p2a causal synth`

Synthetic Control Method (Abadie et al.)

```text
p2a causal synth [OPTIONS] --outcome <OUTCOME> --unit <UNIT> --time <TIME> --treated <TREATED> --treatment-time <TREATMENT_TIME> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --outcome <OUTCOME>` | Outcome variable column |
| `--unit <UNIT>` | Unit identifier column (e.g., "state", "country") |
| `--time <TIME>` | Time period column (must be integer) |
| `--treated <TREATED>` | Name/ID of the treated unit |
| `--treatment-time <TREATMENT_TIME>` | Treatment time (first post-treatment period) |
| `-p, --predictors <PREDICTORS>...` | Predictor columns (will use pre-treatment mean for matching) |
| `--v-method <V_METHOD>` | V matrix: datadriven (nested optimisation, default; also data_driven, optimal, nested) or equal (also uniform; equal weights on the standardised predictors); custom V is MCP-only [default: datadriven] |
| `--placebos` | Run placebo tests for inference |

#### `p2a causal ipw`

Inverse Probability Weighting treatment effect

```text
p2a causal ipw [OPTIONS] --outcome <OUTCOME> --treatment <TREATMENT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --outcome <OUTCOME>` | Outcome variable column |
| `-t, --treatment <TREATMENT>` | Treatment indicator column |
| `-x, --covariates <COVARIATES>...` | Covariate columns |
| `--estimand <ESTIMAND>` | Estimand: ate (default) or att [default: ate] |
| `--trim <TRIM>` | Trimming threshold for propensity scores, in [0, 0.5) (default: 0.05) [default: 0.05] |
| `--bootstrap <BOOTSTRAP>` | Standard errors: 0 (default) gives the M-estimation SE with the propensity score estimated, as WeightIt; N &gt;= 2 a bootstrap of the weighted means with the fitted propensity scores held fixed (conservative) [default: 0] |

#### `p2a causal doubly-robust`

Doubly robust treatment effect (AIPW)

```text
p2a causal doubly-robust [OPTIONS] --outcome <OUTCOME> --treatment <TREATMENT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --outcome <OUTCOME>` | Outcome variable column |
| `-t, --treatment <TREATMENT>` | Treatment indicator column |
| `-x, --covariates <COVARIATES>...` | Covariate columns |
| `--method <METHOD>` | Method: aipw (default; also dr, doubly_robust), ipw or regression (also reg, ra, outcome_regression) [default: aipw] |
| `--estimand <ESTIMAND>` | Estimand: ate (default) or att [default: ate] |
| `--trim <TRIM>` | Trimming threshold for propensity scores, in [0, 0.5) (default: 0.05) [default: 0.05] |
| `--bootstrap <BOOTSTRAP>` | Bootstrap replications for the SEs of --method ipw and regression (at least 2; default: 999); aipw uses its analytic influence-function SE and ignores it [default: 999] |

#### `p2a causal mediation`

Causal mediation analysis

```text
p2a causal mediation [OPTIONS] --outcome <OUTCOME> --treatment <TREATMENT> --mediator <MEDIATOR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --outcome <OUTCOME>` | Outcome variable column |
| `-t, --treatment <TREATMENT>` | Treatment indicator column (binary 0/1) |
| `-m, --mediator <MEDIATOR>` | Mediator variable column |
| `-x, --covariates [<COVARIATES>...]` | Covariate columns |
| `--bootstrap <BOOTSTRAP>` | Number of bootstrap replications (default: 999) [default: 999] |

#### `p2a causal rd`

Regression Discontinuity (sharp design)

```text
p2a causal rd [OPTIONS] --outcome <OUTCOME> --running <RUNNING> --cutoff <CUTOFF> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --outcome <OUTCOME>` | Outcome variable column |
| `-r, --running <RUNNING>` | Running variable column |
| `-c, --cutoff <CUTOFF>` | Cutoff value |
| `-p, --poly-order <POLY_ORDER>` | Polynomial order (default: 1 = local linear) [default: 1] |
| `--kernel <KERNEL>` | Kernel, as rdrobust: triangular (default), epanechnikov or uniform (aliases tri, epa, uni) [default: triangular] |
| `--bwselect <BWSELECT>` | Bandwidth selector, as rdrobust::rdbwselect: mserd (default), msetwo, msesum, msecomb1, msecomb2, cerrd, certwo, cersum, cercomb1 or cercomb2; not used for h when --h is given [default: mserd] |
| `--h <H>` | Main bandwidth h (both sides); default: p2a's MSE-type selector [aliases: --bandwidth] |
| `--b <B>` | Bias bandwidth b; default: h [aliases: --bias-bandwidth] |
| `--vce <VCE>` | Variance estimator: nn (default, as rdrobust), hc0, hc1, hc2 or hc3 [default: nn] |

#### `p2a causal fuzzy-rd`

Fuzzy Regression Discontinuity

```text
p2a causal fuzzy-rd [OPTIONS] --outcome <OUTCOME> --running <RUNNING> --treatment <TREATMENT> --cutoff <CUTOFF> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --outcome <OUTCOME>` | Outcome variable column |
| `-r, --running <RUNNING>` | Running variable column |
| `-t, --treatment <TREATMENT>` | Treatment indicator column |
| `-c, --cutoff <CUTOFF>` | Cutoff value |
| `-p, --poly-order <POLY_ORDER>` | Polynomial order (default: 1) [default: 1] |
| `--kernel <KERNEL>` | Kernel, as rdrobust: triangular (default), epanechnikov or uniform (aliases tri, epa, uni) [default: triangular] |
| `--bwselect <BWSELECT>` | Bandwidth selector for both stages, as rdrobust::rdbwselect(fuzzy = ..): mserd (default), msetwo, msesum, msecomb1, msecomb2, cerrd, certwo, cersum, cercomb1 or cercomb2; not used for h when --h is given [default: mserd] |
| `--h <H>` | Main bandwidth h (both sides); default: p2a's MSE-type selector [aliases: --bandwidth] |
| `--b <B>` | Bias bandwidth b; default: h [aliases: --bias-bandwidth] |
| `--vce <VCE>` | Variance estimator: nn (default, as rdrobust), hc0, hc1, hc2 or hc3 [default: nn] |

#### `p2a causal staggered-did`

Staggered Difference-in-Differences (Callaway-Sant'Anna)

```text
p2a causal staggered-did [OPTIONS] --outcome <OUTCOME> --treatment-time <TREATMENT_TIME> --time <TIME> --unit <UNIT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --outcome <OUTCOME>` | Outcome variable column |
| `--treatment-time <TREATMENT_TIME>` | Treatment timing column: the period a unit is first treated (0 or negative: never) |
| `--time <TIME>` | Time period column |
| `--unit <UNIT>` | Unit identifier column |
| `-x, --covariates [<COVARIATES>...]` | Covariate columns for conditional parallel trends (optional) |
| `--comparison <COMPARISON>` | Comparison group, as did's control_group [default: never_treated] |
| `--method <METHOD>` | Estimation method, as did's est_method (DRDID) [default: or] |
| `--anticipation <ANTICIPATION>` | Periods before treatment in which units may anticipate it (did's anticipation) [default: 0] |
| `--bootstrap <BOOTSTRAP>` | Unit-bootstrap replications for SEs (at least 50), or 0 for analytic SEs [default: 999] |

#### `p2a causal tmle`

Targeted Maximum Likelihood Estimation (TMLE)

```text
p2a causal tmle [OPTIONS] --outcome <OUTCOME> --treatment <TREATMENT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --outcome <OUTCOME>` | Outcome variable column |
| `-t, --treatment <TREATMENT>` | Treatment indicator column (binary 0/1) |
| `-x, --covariates <COVARIATES>...` | Covariate columns |
| `--qmodel <QMODEL>` | Outcome (Q) model: logistic (default; also logit, binomial, binary) or linear (also gaussian, continuous, ols); logistic: Y outside [0, 1] is rescaled to [0, 1] first, as the tmle package does [default: logistic] |
| `--truncate-lower <TRUNCATE_LOWER>` | Propensity score truncation lower bound (default: 0.01) [default: 0.01] |
| `--truncate-upper <TRUNCATE_UPPER>` | Propensity score truncation upper bound (default: 0.99) [default: 0.99] |

#### `p2a causal gsynth`

Generalized Synthetic Control (gsynth)

```text
p2a causal gsynth [OPTIONS] --outcome <OUTCOME> --treatment <TREATMENT> --unit <UNIT> --time <TIME> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --outcome <OUTCOME>` | Outcome variable column |
| `-d, --treatment <TREATMENT>` | Treatment indicator column (0/1) |
| `--unit <UNIT>` | Unit identifier column |
| `--time <TIME>` | Time period column |
| `-x, --covariates [<COVARIATES>...]` | Covariate columns (optional) |
| `--n-factors <N_FACTORS>` | Number of latent factors (0 = auto-select via cross-validation) [default: 0] |
| `--force <FORCE>` | Fixed effects (gsynth force): none, unit (also individual), time or two_way (default; also twoway, two-way) [default: two_way] |
| `--estimator <ESTIMATOR>` | Estimator: ife (interactive fixed effects, default; also gsynth) or mc (matrix completion; also matrix_completion) [default: ife] |
| `--bootstrap-se` | Compute bootstrap standard errors |
| `--n-bootstrap <N_BOOTSTRAP>` | Number of bootstrap replications (default: 200) [default: 200] |

#### `p2a causal matching`

Propensity Score Matching (MatchIt)

```text
p2a causal matching [OPTIONS] --treatment <TREATMENT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-t, --treatment <TREATMENT>` | Treatment indicator column (binary 0/1) |
| `-x, --covariates <COVARIATES>...` | Covariate columns for matching |
| `--method <METHOD>` | Matching method, as MatchIt: nearest (default; also nn), cem (coarsened exact, Sturges bins), full (optimal full matching; also optimal) or subclass (also subclassification) [default: nearest] |
| `--distance <DISTANCE>` | Distance: logit (default; also glm, logistic, ps), probit, mahalanobis or euclidean [default: logit] |
| `--m-order <M_ORDER>` | Order in which treated units are matched (MatchIt m.order): largest (also descending), smallest (also ascending), data or random; default largest for logit/probit, data otherwise |
| `--ratio <RATIO>` | Matching ratio for nearest neighbor (default: 1) [default: 1] |
| `--caliper <CALIPER>` | Caliper for nearest neighbor (in SD units of propensity score; optional) |
| `--replace` | Sample with replacement (for nearest neighbor) |
| `--n-subclasses <N_SUBCLASSES>` | Number of subclasses (for subclass method, default: 5) [default: 5] |

#### `p2a causal stdreg`

Regression Standardization / G-computation (stdReg)

```text
p2a causal stdreg [OPTIONS] --outcome <OUTCOME> --treatment <TREATMENT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --outcome <OUTCOME>` | Outcome variable column |
| `-t, --treatment <TREATMENT>` | Treatment indicator column (binary 0/1) |
| `-x, --covariates <COVARIATES>...` | Covariate columns |
| `--model <MODEL>` | Outcome model: linear (default; also ols, gaussian), logistic (also logit, binomial) or poisson (also count, log_linear) [default: linear] |
| `--estimand <ESTIMAND>` | Estimand: ate (default), att, atc or levels (also potential_outcomes) [default: ate] |
| `--se-method <SE_METHOD>` | SE method: bootstrap (default; also boot), delta or sandwich (also robust) [default: bootstrap] |
| `--bootstrap <BOOTSTRAP>` | Number of bootstrap replications (default: 999) [default: 999] |
| `--interactions` | Include treatment-covariate interactions in outcome model |

### `p2a discrete`

Discrete choice models

```text
p2a discrete [OPTIONS] <COMMAND>
```

| Command | Aliases | Description |
|---|---|---|
| `logit` |  | Logit (logistic regression) |
| `probit` |  | Probit regression |
| `o-logit` |  | Ordered Logit (proportional odds model) |
| `o-probit` |  | Ordered Probit |
| `mlogit` |  | Multinomial Logit |
| `negbin` |  | Negative Binomial regression (count data with overdispersion) |
| `poisson` |  | Poisson regression (count data) |
| `zip` |  | Zero-Inflated Poisson (ZIP) model |
| `zinb` |  | Zero-Inflated Negative Binomial (ZINB) model |
| `hurdle` |  | Hurdle model (two-part: binary + truncated count) |
| `clogit` |  | Conditional Logit (McFadden's choice model for panel/choice data) |
| `mixed-logit` |  | Mixed Logit / Random Parameters Logit |

#### `p2a discrete logit`

Logit (logistic regression)

```text
p2a discrete logit [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column (binary 0/1) |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |

#### `p2a discrete probit`

Probit regression

```text
p2a discrete probit [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column (binary 0/1) |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |

#### `p2a discrete o-logit`

Ordered Logit (proportional odds model)

```text
p2a discrete o-logit [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column (ordered categorical, e.g., 1,2,3,4,5) |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--levels <LEVELS>...` | Ordinal order of a text outcome, lowest first (e.g. --levels low,mid,high) |

#### `p2a discrete o-probit`

Ordered Probit

```text
p2a discrete o-probit [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column (ordered categorical) |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--levels <LEVELS>...` | Ordinal order of a text outcome, lowest first (e.g. --levels low,mid,high) |

#### `p2a discrete mlogit`

Multinomial Logit

```text
p2a discrete mlogit [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column (unordered categorical) |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--base <BASE>` | Reference category (default: lowest value) |

#### `p2a discrete negbin`

Negative Binomial regression (count data with overdispersion)

```text
p2a discrete negbin [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column (non-negative integer counts) |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |

#### `p2a discrete poisson`

Poisson regression (count data)

```text
p2a discrete poisson [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column (non-negative integer counts) |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |

#### `p2a discrete zip`

Zero-Inflated Poisson (ZIP) model

```text
p2a discrete zip [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column (non-negative counts with excess zeros) |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns for count model |
| `-z, --zero-vars <ZERO_VARS>...` | Independent variable columns for zero-inflation model (defaults to same as count model) |

#### `p2a discrete zinb`

Zero-Inflated Negative Binomial (ZINB) model

```text
p2a discrete zinb [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column (non-negative counts with excess zeros and overdispersion) |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns for count model |
| `-z, --zero-vars <ZERO_VARS>...` | Independent variable columns for zero-inflation model (defaults to same as count model) |

#### `p2a discrete hurdle`

Hurdle model (two-part: binary + truncated count)

```text
p2a discrete hurdle [OPTIONS] --dep-var <DEP_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column (non-negative counts) |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns for count model |
| `-z, --zero-vars <ZERO_VARS>...` | Independent variable columns for binary (hurdle) model (defaults to same as count model) |
| `--dist <DIST>` | Count distribution: poisson (default) or negbin (aliases nb, negative-binomial) [default: poisson] [possible values: poisson, negbin] |

#### `p2a discrete clogit`

Conditional Logit (McFadden's choice model for panel/choice data)

```text
p2a discrete clogit [OPTIONS] --choice-id <CHOICE_ID> --alt-id <ALT_ID> --choice <CHOICE> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--choice-id <CHOICE_ID>` | Choice situation ID column (groups alternatives) |
| `--alt-id <ALT_ID>` | Alternative ID column |
| `-y, --choice <CHOICE>` | Choice indicator column (1 = chosen, 0 = not chosen) |
| `-x, --indep-vars <INDEP_VARS>...` | Alternative-specific variable columns |
| `--reference <REFERENCE>` | Reference alternative (mlogit's reflevel); default: the first alternative in sorted order [aliases: --reflevel] |
| `--asc [<BOOL>]` | Alternative-specific constants (true/false; default true), as mlogit's default; false fits mlogit(choice ~ x \| 0) [default: true] [possible values: true, false] |

#### `p2a discrete mixed-logit`

Mixed Logit / Random Parameters Logit

```text
p2a discrete mixed-logit [OPTIONS] --choice-id <CHOICE_ID> --alt-id <ALT_ID> --choice <CHOICE> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--choice-id <CHOICE_ID>` | Choice situation ID column |
| `--alt-id <ALT_ID>` | Alternative ID column |
| `-y, --choice <CHOICE>` | Choice indicator column (1 = chosen, 0 = not chosen) |
| `-x, --indep-vars <INDEP_VARS>...` | Variable columns |
| `--random-vars <RANDOM_VARS>...` | Variables to treat as random (with normal distribution) |
| `--n-draws <N_DRAWS>` | Number of simulation draws (default: 500) [default: 500] |
| `--individual <INDIVIDUAL>` | Column identifying the decision maker for repeated choices (mlogit panel = TRUE) |

### `p2a stats`

Statistical tests and descriptive statistics

```text
p2a stats [OPTIONS] <COMMAND>
```

| Command | Aliases | Description |
|---|---|---|
| `t-test-one` |  | One-sample t-test |
| `t-test-two` |  | Two-sample t-test (independent samples) |
| `t-test-paired` |  | Paired t-test |
| `chisq-gof` |  | Chi-squared goodness-of-fit test |
| `chisq-indep` |  | Chi-squared test of independence |
| `fisher` |  | Fisher's exact test for 2x2 tables |
| `shapiro` |  | Shapiro-Wilk normality test |
| `wilcoxon` |  | Wilcoxon rank-sum test (Mann-Whitney U) |
| `bartlett` |  | Bartlett's test for homogeneity of variances |
| `kruskal` |  | Kruskal-Wallis rank-sum test |
| `fivenum` |  | Five-number summary (min, Q1, median, Q3, max) |
| `iqr` |  | Interquartile range |
| `mad` |  | Median Absolute Deviation |
| `acf` |  | Autocorrelation function |
| `anova` |  | One-way ANOVA |
| `anova-two` |  | Two-way ANOVA |
| `tukey` |  | Tukey's HSD post-hoc test |
| `friedman` |  | Friedman rank sum test for blocked data |
| `pacf` |  | Partial autocorrelation function |
| `ccf` |  | Cross-correlation function |
| `box-test` |  | Ljung-Box or Box-Pierce test for autocorrelation |
| `ks-test` |  | Kolmogorov-Smirnov two-sample test |
| `pp-test` |  | Phillips-Perron unit root test |
| `power-t` |  | Power analysis for t-test |

#### `p2a stats t-test-one`

One-sample t-test

```text
p2a stats t-test-one [OPTIONS] --col <COL> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --col <COL>` | Column to test |
| `-m, --mu <MU>` | Hypothesized mean (null hypothesis) [default: 0.0] |
| `-a, --alternative <ALTERNATIVE>` | Alternative hypothesis [default: two-sided] [possible values: two-sided, less, greater] |
| `--conf-level <CONF_LEVEL>` | Confidence level [default: 0.95] |

#### `p2a stats t-test-two`

Two-sample t-test (independent samples)

```text
p2a stats t-test-two [OPTIONS] --col1 <COL1> --col2 <COL2> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--col1 <COL1>` | First column |
| `--col2 <COL2>` | Second column |
| `-a, --alternative <ALTERNATIVE>` | Alternative hypothesis [default: two-sided] [possible values: two-sided, less, greater] |
| `--equal-var` | Assume equal variances (use pooled variance) |

#### `p2a stats t-test-paired`

Paired t-test

```text
p2a stats t-test-paired [OPTIONS] --col1 <COL1> --col2 <COL2> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--col1 <COL1>` | First column (before/treatment) |
| `--col2 <COL2>` | Second column (after/control) |
| `-a, --alternative <ALTERNATIVE>` | Alternative hypothesis [default: two-sided] [possible values: two-sided, less, greater] |

#### `p2a stats chisq-gof`

Chi-squared goodness-of-fit test

```text
p2a stats chisq-gof [OPTIONS] --observed <OBSERVED> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-o, --observed <OBSERVED>` | Column with observed counts |

#### `p2a stats chisq-indep`

Chi-squared test of independence

```text
p2a stats chisq-indep [OPTIONS] --row-var <ROW_VAR> --col-var <COL_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--row-var <ROW_VAR>` | Row variable column |
| `--col-var <COL_VAR>` | Column variable column |

#### `p2a stats fisher`

Fisher's exact test for 2x2 tables

```text
p2a stats fisher [OPTIONS] --row-var <ROW_VAR> --col-var <COL_VAR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--row-var <ROW_VAR>` | Row variable column |
| `--col-var <COL_VAR>` | Column variable column |

#### `p2a stats shapiro`

Shapiro-Wilk normality test

```text
p2a stats shapiro [OPTIONS] --col <COL> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --col <COL>` | Column to test |

#### `p2a stats wilcoxon`

Wilcoxon rank-sum test (Mann-Whitney U)

```text
p2a stats wilcoxon [OPTIONS] --col1 <COL1> --col2 <COL2> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--col1 <COL1>` | First column |
| `--col2 <COL2>` | Second column |
| `--paired` | Paired test (signed-rank) instead of rank-sum |
| `-a, --alternative <ALTERNATIVE>` | Alternative hypothesis [default: two-sided] [possible values: two-sided, less, greater] |

#### `p2a stats bartlett`

Bartlett's test for homogeneity of variances

```text
p2a stats bartlett [OPTIONS] --col <COL> --group <GROUP> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --col <COL>` | Value column |
| `-g, --group <GROUP>` | Grouping column |

#### `p2a stats kruskal`

Kruskal-Wallis rank-sum test

```text
p2a stats kruskal [OPTIONS] --col <COL> --group <GROUP> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --col <COL>` | Value column |
| `-g, --group <GROUP>` | Grouping column |

#### `p2a stats fivenum`

Five-number summary (min, Q1, median, Q3, max)

```text
p2a stats fivenum [OPTIONS] --col <COL> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --col <COL>` | Column to summarize |

#### `p2a stats iqr`

Interquartile range

```text
p2a stats iqr [OPTIONS] --col <COL> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --col <COL>` | Column to compute IQR |

#### `p2a stats mad`

Median Absolute Deviation

```text
p2a stats mad [OPTIONS] --col <COL> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --col <COL>` | Column to compute MAD |

#### `p2a stats acf`

Autocorrelation function

```text
p2a stats acf [OPTIONS] --col <COL> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --col <COL>` | Time series column |
| `--lag-max <LAG_MAX>` | Maximum lag [default: 20] |
| `--partial` | Compute partial ACF instead |

#### `p2a stats anova`

One-way ANOVA

```text
p2a stats anova [OPTIONS] --response <RESPONSE> --factor <FACTOR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-r, --response <RESPONSE>` | Response (dependent) variable column |
| `-f, --factor <FACTOR>` | Factor (grouping) variable column |

#### `p2a stats anova-two`

Two-way ANOVA

```text
p2a stats anova-two [OPTIONS] --response <RESPONSE> --factor-a <FACTOR_A> --factor-b <FACTOR_B> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-r, --response <RESPONSE>` | Response (dependent) variable column |
| `--factor-a <FACTOR_A>` | First factor (grouping) variable column |
| `--factor-b <FACTOR_B>` | Second factor (grouping) variable column |
| `--interaction` | Include interaction term |
| `--ss-type <TYPE>` | Sums of squares: I (sequential, as R aov; default), II or III (as car::Anova; III with sum-to-zero contrasts) [default: I] |

#### `p2a stats tukey`

Tukey's HSD post-hoc test

```text
p2a stats tukey [OPTIONS] --response <RESPONSE> --factor <FACTOR> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-r, --response <RESPONSE>` | Response (dependent) variable column |
| `-f, --factor <FACTOR>` | Factor (grouping) variable column |
| `--conf-level <CONF_LEVEL>` | Confidence level [default: 0.95] |

#### `p2a stats friedman`

Friedman rank sum test for blocked data

```text
p2a stats friedman [OPTIONS] --value <VALUE> --group <GROUP> --block <BLOCK> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-v, --value <VALUE>` | Value column (measurements) |
| `-g, --group <GROUP>` | Group/treatment column |
| `-b, --block <BLOCK>` | Block column (e.g., subject ID) |

#### `p2a stats pacf`

Partial autocorrelation function

```text
p2a stats pacf [OPTIONS] --col <COL> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --col <COL>` | Time series column |
| `--lag-max <LAG_MAX>` | Maximum lag [default: 20] |

#### `p2a stats ccf`

Cross-correlation function

```text
p2a stats ccf [OPTIONS] --x <X> --y <Y> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--x <X>` | First time series column |
| `--y <Y>` | Second time series column |
| `--lag-max <LAG_MAX>` | Maximum lag [default: 20] |

#### `p2a stats box-test`

Ljung-Box or Box-Pierce test for autocorrelation

```text
p2a stats box-test [OPTIONS] --col <COL> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --col <COL>` | Time series column |
| `--lag <LAG>` | Maximum lag to test [default: 10] |
| `--box-pierce` | Use Box-Pierce instead of Ljung-Box |
| `--fitdf <FITDF>` | Degrees of freedom to subtract (for ARMA residuals) [default: 0] |

#### `p2a stats ks-test`

Kolmogorov-Smirnov two-sample test

```text
p2a stats ks-test [OPTIONS] --x <X> --y <Y> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--x <X>` | First sample column |
| `--y <Y>` | Second sample column |
| `-a, --alternative <ALTERNATIVE>` | Alternative hypothesis [default: two-sided] [possible values: two-sided, less, greater] |

#### `p2a stats pp-test`

Phillips-Perron unit root test

```text
p2a stats pp-test [OPTIONS] --col <COL> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-c, --col <COL>` | Time series column |
| `--lshort` | Use short truncation lag |

#### `p2a stats power-t`

Power analysis for t-test

```text
p2a stats power-t [OPTIONS]
```

| Argument or option | Description |
|---|---|
| `-n, --n <N>` | Sample size per group (leave unset to solve for it) |
| `-d, --delta <DELTA>` | True difference in means (leave unset to solve for it) |
| `--sd <SD>` | Standard deviation [default: 1.0] |
| `--sig-level <SIG_LEVEL>` | Significance level [default: 0.05] |
| `-p, --power <POWER>` | Desired power (leave unset to solve for it) |
| `--test-type <TEST_TYPE>` | Test type: one-sample, two-sample, or paired [default: two-sample] [possible values: one-sample, two-sample, paired] |
| `--alternative <ALTERNATIVE>` | Alternative: two-sided or one-sided [default: two-sided] [possible values: two-sided, one-sided] |

### `p2a timeseries` (alias `ts`)

Time series analysis

```text
p2a timeseries [OPTIONS] <COMMAND>
```

| Command | Aliases | Description |
|---|---|---|
| `arima` |  | ARIMA model estimation and forecasting |
| `mstl` |  | MSTL seasonal decomposition |
| `var` |  | Vector Autoregression |
| `varma` |  | Vector ARMA model |
| `vecm` |  | Vector Error Correction Model (cointegration) |
| `irf` |  | Impulse Response Function from VAR model |
| `changepoint` |  | Changepoint detection (PELT algorithm) |
| `garch` |  | GARCH volatility model |
| `holt-winters` |  | Holt-Winters exponential smoothing |
| `granger` |  | Granger causality test |
| `ar` |  | Autoregressive (AR) model fitting |
| `stl` |  | STL decomposition (Seasonal-Trend using LOESS) |
| `decompose` |  | Classical seasonal decomposition (moving averages) |
| `struct-ts` |  | Structural time series (state space) model |
| `causal-impact` |  | Bayesian structural time series causal inference (CausalImpact) |
| `spectrum` |  | Spectral density estimation (periodogram) |

#### `p2a timeseries arima`

ARIMA model estimation and forecasting

```text
p2a timeseries arima [OPTIONS] --col <COL> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--col <COL>` | Time series column |
| `-p, --ar <AR>` | Autoregressive order (p) [default: 1] |
| `-d, --diff <DIFF>` | Differencing order (d) [default: 0] |
| `-q, --ma <MA>` | Moving average order (q) [default: 1] |
| `--horizon <HORIZON>` | Forecast horizon (optional) |
| `--include-mean <BOOL>` | Estimate the mean (default: true when d = 0, as R); --include-mean false drops it [possible values: true, false] |
| `--drift` | Fit a drift (linear trend in the levels), as forecast::Arima(include.drift = TRUE); d = 0 or 1 |

#### `p2a timeseries mstl`

MSTL seasonal decomposition

```text
p2a timeseries mstl [OPTIONS] --col <COL> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--col <COL>` | Time series column |
| `--periods <PERIODS>...` | Seasonal periods (e.g., 7 for weekly, 365 for yearly) |

#### `p2a timeseries var`

Vector Autoregression

```text
p2a timeseries var [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--cols <COLS> <COLS>...` | Variable columns |
| `--lags <LAGS>` | Number of lags [default: 1] |

#### `p2a timeseries varma`

Vector ARMA model

```text
p2a timeseries varma [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--cols <COLS> <COLS>...` | Variable columns |
| `-p, --ar <AR>` | AR order (p) [default: 1] |
| `-q, --ma <MA>` | MA order (q) [default: 1] |
| `--long-var-order <LONG_VAR_ORDER>` | Order m of the step-1 long VAR (Hannan-Rissanen); default max(floor(2 ln n), 2 max(p, q)), 0 when q = 0. Set it larger when an MA root is near the unit circle |

#### `p2a timeseries vecm`

Vector Error Correction Model (cointegration)

```text
p2a timeseries vecm [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--cols <COLS> <COLS>...` | Variable columns |
| `--lags <LAGS>` | Lag order K of the VAR in levels (urca::ca.jo K; at least 2) [default: 2] |
| `--rank <RANK>` | Cointegration rank r for the restricted VECM (1..k-1); default: the trace test's rank at 5% |
| `--ecdet <ECDET>` | Deterministic term in the cointegration relation, as ca.jo ecdet: none (default), const or trend [default: none] |

#### `p2a timeseries irf`

Impulse Response Function from VAR model

```text
p2a timeseries irf [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--cols <COLS> <COLS>...` | Variable columns |
| `--lags <LAGS>` | Number of lags for VAR [default: 1] |
| `--steps <STEPS>` | Number of horizons including the impact h = 0 (vars::irf n.ahead = steps - 1) [default: 10] |

#### `p2a timeseries changepoint`

Changepoint detection (PELT algorithm)

```text
p2a timeseries changepoint [OPTIONS] --col <COL> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--col <COL>` | Time series column |
| `--penalty <PENALTY>` | Penalty; default changepoint's MBIC: (k + 2) ln n plus ln(segment length) per segment (k = 1 for mean or variance, 2 for meanvar) |
| `--min-segment <MIN_SEGMENT>` | Minimum segment length (default: 2) |
| `--change-type <CHANGE_TYPE>` | Type of change, as the changepoint package: mean (cpt.mean, unit variance; default), variance (cpt.var) or meanvar (cpt.meanvar; alias both) [default: mean] |

#### `p2a timeseries garch`

GARCH volatility model

```text
p2a timeseries garch [OPTIONS] --col <COL> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--col <COL>` | Time series column |
| `-p, --arch <ARCH>` | ARCH order (p) - number of lagged squared residuals [default: 1] |
| `-q, --garch <GARCH>` | GARCH order (q) - number of lagged variances [default: 1] |
| `--horizon <HORIZON>` | Forecast the conditional variance h steps ahead (GARCH recursion) |
| `--distribution <DISTRIBUTION>` | Conditional distribution, as fGarch cond.dist: norm (default) or std (standardised Student t; aliases t, student_t) [default: norm] |

#### `p2a timeseries holt-winters`

Holt-Winters exponential smoothing

```text
p2a timeseries holt-winters [OPTIONS] --col <COL> --period <PERIOD> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--col <COL>` | Time series column |
| `--period <PERIOD>` | Seasonal period (e.g., 12 for monthly, 4 for quarterly) |
| `--seasonal <SEASONAL>` | Seasonal component, as HoltWinters: additive (default) or multiplicative (alias mult) [default: additive] |
| `--alpha <ALPHA>` | Level smoothing alpha in (0, 1]; default: optimised |
| `--beta <BETA>` | Trend smoothing beta in [0, 1]; 0 fits no trend (R beta = FALSE); default: optimised |
| `--gamma <GAMMA>` | Seasonal smoothing gamma in [0, 1]; 0 fits no seasonal component (R gamma = FALSE); default: optimised |
| `--horizon <HORIZON>` | Forecast h steps ahead with prediction intervals (predict.HoltWinters); a multiplicative seasonal fit gets point forecasts only |
| `--level <LEVEL>` | Level of the prediction intervals (predict.HoltWinters level), strictly between 0 and 1; default 0.95. Needs --horizon |

#### `p2a timeseries granger`

Granger causality test

```text
p2a timeseries granger [OPTIONS] --x <X> --y <Y> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--x <X>` | First variable (potential cause) |
| `--y <Y>` | Second variable (potential effect) |
| `--lags <LAGS>` | Number of lags [default: 1] |

#### `p2a timeseries ar`

Autoregressive (AR) model fitting

```text
p2a timeseries ar [OPTIONS] --col <COL> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--col <COL>` | Time series column |
| `--order <ORDER>` | Fit exactly this order (AIC selection is then not used) |
| `--method <METHOD>` | Method, as ar(): yule-walker (default; alias yw), burg or ols [default: yule-walker] |
| `--aic [<BOOL>]` | Select the order by AIC up to min(n - 1, floor(10 log10 n)) (true/false; default true); false fits that maximum order, as R's ar(aic = FALSE). --order fixes the order and overrides both [default: true] [possible values: true, false] |

#### `p2a timeseries stl`

STL decomposition (Seasonal-Trend using LOESS)

```text
p2a timeseries stl [OPTIONS] --col <COL> --period <PERIOD> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--col <COL>` | Time series column |
| `--period <PERIOD>` | Seasonal period (e.g., 12 for monthly, 4 for quarterly) |
| `--robust [<BOOL>]` | Robust (outlier-resistant) fitting (true/false; default false) [default: false] [possible values: true, false] |

#### `p2a timeseries decompose`

Classical seasonal decomposition (moving averages)

```text
p2a timeseries decompose [OPTIONS] --col <COL> --period <PERIOD> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--col <COL>` | Time series column |
| `--period <PERIOD>` | Seasonal period |
| `--decompose-type <DECOMPOSE_TYPE>` | Decomposition, as decompose(): additive (default) or multiplicative [default: additive] |

#### `p2a timeseries struct-ts`

Structural time series (state space) model

```text
p2a timeseries struct-ts [OPTIONS] --col <COL> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--col <COL>` | Time series column |
| `--model-type <MODEL_TYPE>` | Model, as StructTS: level (default), trend or BSM (needs --period) [default: level] |
| `--period <PERIOD>` | Seasonal period (required for "bsm" model type) |

#### `p2a timeseries causal-impact`

Bayesian structural time series causal inference (CausalImpact)

```text
p2a timeseries causal-impact [OPTIONS] --response <RESPONSE> --time <TIME> --pre-start <PRE_START> --pre-end <PRE_END> --post-start <POST_START> --post-end <POST_END> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--response <RESPONSE>` | Response variable column |
| `--time <TIME>` | Time column |
| `--pre-start <PRE_START>` | Pre-intervention period start (inclusive) |
| `--pre-end <PRE_END>` | Pre-intervention period end (inclusive) |
| `--post-start <POST_START>` | Post-intervention period start (inclusive) |
| `--post-end <POST_END>` | Post-intervention period end (inclusive) |
| `--controls <CONTROLS>` | Control series columns (comma-separated) |

#### `p2a timeseries spectrum`

Spectral density estimation (periodogram)

```text
p2a timeseries spectrum [OPTIONS] --col <COL> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--col <COL>` | Time series column |
| `--spans <SPANS>...` | Modified Daniell smoothing spans, comma-separated odd integers, e.g. 3,3 |
| `--taper <TAPER>` | Taper proportion (0.0 to 0.5, default: 0.1) [default: 0.1] |
| `--detrend [<BOOL>]` | Remove a least-squares linear trend first (true/false; default true, as spec.pgram) [default: true] [possible values: true, false] |

### `p2a survival`

Survival analysis

```text
p2a survival [OPTIONS] <COMMAND>
```

| Command | Aliases | Description |
|---|---|---|
| `km` |  | Kaplan-Meier survival curve estimation |
| `log-rank` |  | Log-rank test comparing survival curves between groups |
| `cox` |  | Cox Proportional Hazards regression |
| `aft` |  | Accelerated Failure Time model |
| `competing-risks` |  | Competing risks analysis (cumulative incidence) |

#### `p2a survival km`

Kaplan-Meier survival curve estimation

```text
p2a survival km [OPTIONS] --time <TIME> --event <EVENT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-t, --time <TIME>` | Time variable column (duration/time to event) |
| `-e, --event <EVENT>` | Status column, coded as R's Surv(): 0/1 (1 = event), 1/2 (2 = event) or TRUE/FALSE |
| `-g, --group <GROUP>` | Optional grouping variable for stratified analysis |
| `--conf <CONF>` | Confidence level (default 0.95) [default: 0.95] |
| `--conf-type <CONF_TYPE>` | Confidence interval type, as R survfit's conf.type: log (default), log-log, plain [default: log] |

#### `p2a survival log-rank`

Log-rank test comparing survival curves between groups

```text
p2a survival log-rank [OPTIONS] --time <TIME> --event <EVENT> --group <GROUP> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-t, --time <TIME>` | Time variable column |
| `-e, --event <EVENT>` | Status column, coded as R's Surv(): 0/1 (1 = event), 1/2 (2 = event) or TRUE/FALSE |
| `-g, --group <GROUP>` | Grouping variable column |

#### `p2a survival cox`

Cox Proportional Hazards regression

```text
p2a survival cox [OPTIONS] --time <TIME> --event <EVENT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-t, --time <TIME>` | Time variable column |
| `-e, --event <EVENT>` | Status column, coded as R's Surv(): 0/1 (1 = event), 1/2 (2 = event) or TRUE/FALSE |
| `-x, --covariates <COVARIATES>...` | Covariate columns |
| `--ties <TIES>` | Ties handling method, as coxph's ties: "efron" (default) or "breslow" [default: Efron] |
| `--robust` | Use robust (Lin-Wei sandwich) standard errors, as coxph(robust = TRUE) |
| `--cluster <CLUSTER>` | Cluster column for clustered sandwich standard errors, as coxph's cluster() (implies --robust) |

#### `p2a survival aft`

Accelerated Failure Time model

```text
p2a survival aft [OPTIONS] --time <TIME> --event <EVENT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-t, --time <TIME>` | Time variable column |
| `-e, --event <EVENT>` | Status column, coded as R's Surv(): 0/1 (1 = event), 1/2 (2 = event) or TRUE/FALSE |
| `-x, --covariates <COVARIATES>...` | Covariate columns |
| `--dist <DIST>` | Distribution, as survreg's dist: "weibull" (default), "exponential", "lognormal", "loglogistic" [default: Weibull] |

#### `p2a survival competing-risks`

Competing risks analysis (cumulative incidence)

```text
p2a survival competing-risks [OPTIONS] --time <TIME> --event-type <EVENT_TYPE> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-t, --time <TIME>` | Time variable column |
| `-e, --event-type <EVENT_TYPE>` | Event type column (0 = censored, 1,2,... = event types) |
| `--conf <CONF>` | Confidence level (default 0.95) [default: 0.95] |

### `p2a spatial`

Spatial econometrics

```text
p2a spatial [OPTIONS] <COMMAND>
```

| Command | Aliases | Description |
|---|---|---|
| `sar` |  | Spatial Autoregressive Model (SAR / lagsarlm) |
| `sem` |  | Spatial Error Model (SEM / errorsarlm) |
| `sac` |  | Spatial Autoregressive Combined Model (SAC / SARAR) |
| `moran` |  | Moran's I test for spatial autocorrelation |

#### `p2a spatial sar`

Spatial Autoregressive Model (SAR / lagsarlm)

```text
p2a spatial sar [OPTIONS] --dep-var <DEP_VAR> --coord-x <COORD_X> --coord-y <COORD_Y> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--coord-x <COORD_X>` | Coordinate X column (for building spatial weights) |
| `--coord-y <COORD_Y>` | Coordinate Y column (for building spatial weights) |
| `-k, --k-neighbors <K_NEIGHBORS>` | Number of nearest neighbors for spatial weights, 1 to n - 1 where n is the number of rows (default: 5) [default: 5] |
| `--weight-style <WEIGHT_STYLE>` | Weight style, as spdep::nb2listw: W (row-standardised, default), B (binary), C (global), U (unit sum), minmax or S (variance-stabilising); old spellings rowstd, binary accepted [default: W] |
| `--durbin` | Estimate Spatial Durbin Model (include spatially lagged X) |
| `--impacts` | Compute spatial impacts (direct, indirect, total effects) |

#### `p2a spatial sem`

Spatial Error Model (SEM / errorsarlm)

```text
p2a spatial sem [OPTIONS] --dep-var <DEP_VAR> --coord-x <COORD_X> --coord-y <COORD_Y> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--coord-x <COORD_X>` | Coordinate X column (for building spatial weights) |
| `--coord-y <COORD_Y>` | Coordinate Y column (for building spatial weights) |
| `-k, --k-neighbors <K_NEIGHBORS>` | Number of nearest neighbors for spatial weights, 1 to n - 1 where n is the number of rows (default: 5) [default: 5] |
| `--weight-style <WEIGHT_STYLE>` | Weight style, as spdep::nb2listw: W (row-standardised, default), B (binary), C (global), U (unit sum), minmax or S (variance-stabilising); old spellings rowstd, binary accepted [default: W] |

#### `p2a spatial sac`

Spatial Autoregressive Combined Model (SAC / SARAR)

```text
p2a spatial sac [OPTIONS] --dep-var <DEP_VAR> --coord-x <COORD_X> --coord-y <COORD_Y> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `--coord-x <COORD_X>` | Coordinate X column (for building spatial weights) |
| `--coord-y <COORD_Y>` | Coordinate Y column (for building spatial weights) |
| `-k, --k-neighbors <K_NEIGHBORS>` | Number of nearest neighbors for spatial weights, 1 to n - 1 where n is the number of rows (default: 5) [default: 5] |
| `--weight-style <WEIGHT_STYLE>` | Weight style, as spdep::nb2listw: W (row-standardised, default), B (binary), C (global), U (unit sum), minmax or S (variance-stabilising); old spellings rowstd, binary accepted [default: W] |

#### `p2a spatial moran`

Moran's I test for spatial autocorrelation

```text
p2a spatial moran [OPTIONS] --variable <VARIABLE> --coord-x <COORD_X> --coord-y <COORD_Y> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --variable <VARIABLE>` | Variable column to test for spatial autocorrelation |
| `--coord-x <COORD_X>` | Coordinate X column (for building spatial weights) |
| `--coord-y <COORD_Y>` | Coordinate Y column (for building spatial weights) |
| `-k, --k-neighbors <K_NEIGHBORS>` | Number of nearest neighbors for spatial weights, 1 to n - 1 where n is the number of rows (default: 5) [default: 5] |
| `--weight-style <WEIGHT_STYLE>` | Weight style, as spdep::nb2listw: W (row-standardised, default), B (binary), C (global), U (unit sum), minmax or S (variance-stabilising); old spellings rowstd, binary accepted [default: W] |
| `--alternative <ALTERNATIVE>` | Alternative hypothesis, as spdep::moran.test: two.sided (default), greater or less [default: two.sided] |

### `p2a machine-learning` (alias `ml`)

Machine learning

```text
p2a machine-learning [OPTIONS] <COMMAND>
```

| Command | Aliases | Description |
|---|---|---|
| `kmeans` |  | K-means clustering |
| `pca` |  | Principal Component Analysis |
| `tsne` |  | t-SNE dimensionality reduction |
| `random-forest` |  | Random Forest regression (MSE splits; a class label is averaged, not voted) |
| `dbscan` |  | DBSCAN density-based clustering |
| `hierarchical` |  | Hierarchical (agglomerative) clustering |
| `svm` |  | Linear Support Vector Machine (SVM) |

#### `p2a machine-learning kmeans`

K-means clustering

```text
p2a machine-learning kmeans [OPTIONS] --k <K> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--cols <COLS>...` | Feature columns |
| `-k, --k <K>` | Number of clusters |
| `--seed <SEED>` | Random seed |
| `--max-iter <MAX_ITER>` | Maximum iterations [default: 100] |

#### `p2a machine-learning pca`

Principal Component Analysis

```text
p2a machine-learning pca [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--cols <COLS>...` | Feature columns |
| `-n, --n-components <N_COMPONENTS>` | Number of components to keep |
| `--transform` | Compute the principal-component scores (prcomp x): first 10 rows in text, all in -F json |
| `--scores-file <CSV>` | Write every score to this CSV file (columns PC1..PCk); implies --transform |

#### `p2a machine-learning tsne`

t-SNE dimensionality reduction

```text
p2a machine-learning tsne [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--cols <COLS>...` | Feature columns |
| `-n, --n-components <N_COMPONENTS>` | Number of output dimensions (default: 2) [default: 2] |
| `--perplexity <PERPLEXITY>` | Perplexity parameter (default: 30.0) [default: 30.0] |
| `--max-iter <MAX_ITER>` | Maximum iterations (default: 1000) [default: 1000] |
| `--learning-rate <LEARNING_RATE>` | Learning rate (default: 200.0) [default: 200.0] |
| `--seed <SEED>` | Random seed |

#### `p2a machine-learning random-forest`

Random Forest regression (MSE splits; a class label is averaged, not voted)

```text
p2a machine-learning random-forest [OPTIONS] --target <TARGET> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--cols <COLS>...` | Feature columns |
| `-y, --target <TARGET>` | Target column |
| `--n-trees <N_TREES>` | Number of trees (default: 100) [default: 100] |
| `--max-depth <MAX_DEPTH>` | Maximum tree depth (default: 10) [default: 10] |
| `--min-samples-split <MIN_SAMPLES_SPLIT>` | Minimum samples to split (default: 2) [default: 2] |
| `--max-features <MAX_FEATURES>` | Features tried per split: sqrt (default), log2, all, or a number &gt;= 1 [default: sqrt] |
| `--seed <SEED>` | Random seed |

#### `p2a machine-learning dbscan`

DBSCAN density-based clustering

```text
p2a machine-learning dbscan [OPTIONS] --eps <EPS> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--cols <COLS>...` | Feature columns |
| `--eps <EPS>` | Maximum distance between samples for neighborhood (epsilon) |
| `--min-samples <MIN_SAMPLES>` | Minimum samples in neighborhood for a core point [default: 5] |

#### `p2a machine-learning hierarchical`

Hierarchical (agglomerative) clustering

```text
p2a machine-learning hierarchical [OPTIONS] <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--cols <COLS>...` | Feature columns |
| `-n, --n-clusters <N_CLUSTERS>` | Number of clusters to form |
| `--linkage <LINKAGE>` | Linkage, as stats::hclust: single, complete, average (also upgma), ward.D2 (default; also ward), mcquitty (also weighted), ward.D, centroid or median. R's deprecated hclust(method = "ward") is ward.D [default: ward.D2] |
| `--distance-threshold <DISTANCE_THRESHOLD>` | Distance threshold for cutting the dendrogram |

#### `p2a machine-learning svm`

Linear Support Vector Machine (SVM)

```text
p2a machine-learning svm [OPTIONS] --target <TARGET> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--cols <COLS>...` | Feature columns |
| `-y, --target <TARGET>` | Target column (binary classification) |
| `-c, --c <C>` | Regularization parameter C (default: 1.0) [default: 1.0] |
| `--max-iter <MAX_ITER>` | Maximum iterations (default: 1000) [default: 1000] |
| `--tolerance <TOLERANCE>` | Convergence tolerance (default: 1e-3) [default: 0.001] |

### `p2a visualize` (alias `viz`)

Visualization (output to file)

```text
p2a visualize [OPTIONS] <COMMAND>
```

| Command | Aliases | Description |
|---|---|---|
| `histogram` |  | Create a histogram |
| `scatter` |  | Create a scatter plot |
| `line` |  | Create a line plot |
| `box` |  | Create a box plot |
| `heatmap` |  | Create a correlation heatmap |
| `coefplot` |  | Create a coefficient plot (forest plot) from regression |
| `residuals` |  | Create residual diagnostic plots from regression |
| `dendrogram` |  | Create a dendrogram from hierarchical clustering |
| `event-study` |  | Create an event study plot |
| `irf` |  | Create an impulse response function plot |
| `scatter-interactive` |  | Create an interactive scatter plot (HTML output with Plotly) |
| `histogram-interactive` |  | Create an interactive histogram (HTML output with Plotly) |
| `line-interactive` |  | Create an interactive line plot (HTML output with Plotly) |

#### `p2a visualize histogram`

Create a histogram

```text
p2a visualize histogram [OPTIONS] --col <COL> --file <OUTPUT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--col <COL>` | Column to plot |
| `-f, --file <OUTPUT>` | Output file path |
| `--bins <BINS>` | Number of bins [default: 30] |
| `--title <TITLE>` | Chart title |

#### `p2a visualize scatter`

Create a scatter plot

```text
p2a visualize scatter [OPTIONS] --x-col <X_COL> --y-col <Y_COL> --file <OUTPUT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-x, --x-col <X_COL>` | X-axis column |
| `-y, --y-col <Y_COL>` | Y-axis column |
| `-f, --file <OUTPUT>` | Output file path |
| `--title <TITLE>` | Chart title |

#### `p2a visualize line`

Create a line plot

```text
p2a visualize line [OPTIONS] --x-col <X_COL> --y-col <Y_COL> --file <OUTPUT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-x, --x-col <X_COL>` | X-axis column |
| `-y, --y-col <Y_COL>` | Y-axis column |
| `-f, --file <OUTPUT>` | Output file path |
| `--title <TITLE>` | Chart title |

#### `p2a visualize box`

Create a box plot

```text
p2a visualize box [OPTIONS] --value-col <VALUE_COL> --file <OUTPUT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --value-col <VALUE_COL>` | Value column to plot |
| `-g, --group-col <GROUP_COL>` | Group column (optional - if omitted, single box) |
| `-f, --file <OUTPUT>` | Output file path |
| `--title <TITLE>` | Chart title |

#### `p2a visualize heatmap`

Create a correlation heatmap

```text
p2a visualize heatmap [OPTIONS] --file <OUTPUT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--cols <COLS> <COLS>...` | Columns to include in correlation matrix |
| `-f, --file <OUTPUT>` | Output file path |
| `--title <TITLE>` | Chart title |

#### `p2a visualize coefplot`

Create a coefficient plot (forest plot) from regression

```text
p2a visualize coefplot [OPTIONS] --dep-var <DEP_VAR> --file <OUTPUT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `-f, --file <OUTPUT>` | Output file path |
| `--conf-level <CONF_LEVEL>` | Confidence level (default: 0.95) [default: 0.95] |
| `--title <TITLE>` | Chart title |

#### `p2a visualize residuals`

Create residual diagnostic plots from regression

```text
p2a visualize residuals [OPTIONS] --dep-var <DEP_VAR> --file <OUTPUT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-y, --dep-var <DEP_VAR>` | Dependent variable column |
| `-x, --indep-vars <INDEP_VARS>...` | Independent variable columns |
| `-f, --file <OUTPUT>` | Output file path |
| `--title <TITLE>` | Chart title |

#### `p2a visualize dendrogram`

Create a dendrogram from hierarchical clustering

```text
p2a visualize dendrogram [OPTIONS] --file <OUTPUT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--cols <COLS>...` | Feature columns |
| `--linkage <LINKAGE>` | Linkage, as stats::hclust: single, complete, average (also upgma), ward.D2 (default; also ward), mcquitty (also weighted), ward.D, centroid or median. R's deprecated hclust(method = "ward") is ward.D [default: ward.D2] |
| `-f, --file <OUTPUT>` | Output file path |
| `--title <TITLE>` | Chart title |

#### `p2a visualize event-study`

Create an event study plot

```text
p2a visualize event-study [OPTIONS] --time-col <TIME_COL> --estimate-col <ESTIMATE_COL> --ci-lower-col <CI_LOWER_COL> --ci-upper-col <CI_UPPER_COL> --file <OUTPUT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--time-col <TIME_COL>` | Time/period column (relative to event) |
| `--estimate-col <ESTIMATE_COL>` | Estimate column |
| `--ci-lower-col <CI_LOWER_COL>` | Lower CI column |
| `--ci-upper-col <CI_UPPER_COL>` | Upper CI column |
| `-f, --file <OUTPUT>` | Output file path |
| `--title <TITLE>` | Chart title |

#### `p2a visualize irf`

Create an impulse response function plot

```text
p2a visualize irf [OPTIONS] --horizon-col <HORIZON_COL> --response-col <RESPONSE_COL> --file <OUTPUT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--horizon-col <HORIZON_COL>` | Horizon column |
| `--response-col <RESPONSE_COL>` | Response column |
| `--ci-lower-col <CI_LOWER_COL>` | Lower CI column (optional) |
| `--ci-upper-col <CI_UPPER_COL>` | Upper CI column (optional) |
| `--shock-label <SHOCK_LABEL>` | Shock variable label |
| `--response-label <RESPONSE_LABEL>` | Response variable label |
| `-f, --file <OUTPUT>` | Output file path |
| `--title <TITLE>` | Chart title |

#### `p2a visualize scatter-interactive`

Create an interactive scatter plot (HTML output with Plotly)

```text
p2a visualize scatter-interactive [OPTIONS] --x-col <X_COL> --y-col <Y_COL> --file <OUTPUT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-x, --x-col <X_COL>` | X-axis column |
| `-y, --y-col <Y_COL>` | Y-axis column |
| `-g, --group-col <GROUP_COL>` | Group column for separate traces (optional) |
| `-f, --file <OUTPUT>` | Output HTML file path |
| `--title <TITLE>` | Chart title |

#### `p2a visualize histogram-interactive`

Create an interactive histogram (HTML output with Plotly)

```text
p2a visualize histogram-interactive [OPTIONS] --col <COL> --file <OUTPUT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `--col <COL>` | Column to plot |
| `-g, --group-col <GROUP_COL>` | Group column for separate traces (optional) |
| `-f, --file <OUTPUT>` | Output HTML file path |
| `--title <TITLE>` | Chart title |

#### `p2a visualize line-interactive`

Create an interactive line plot (HTML output with Plotly)

```text
p2a visualize line-interactive [OPTIONS] --x-col <X_COL> --y-col <Y_COL> --file <OUTPUT> <DATASET>
```

| Argument or option | Description |
|---|---|
| `<DATASET>` | Dataset name |
| `-x, --x-col <X_COL>` | X-axis column |
| `-y, --y-col <Y_COL>` | Y-axis column |
| `-f, --file <OUTPUT>` | Output HTML file path |
| `--title <TITLE>` | Chart title |

### `p2a script`

Script generation and session management

```text
p2a script [OPTIONS] <COMMAND>
```

| Command | Aliases | Description |
|---|---|---|
| `export` |  | Export session to a reproducible bash script |
| `history` |  | Show session history |
| `run` |  | Run a p2a script |

#### `p2a script export`

Export session to a reproducible bash script

```text
p2a script export [OPTIONS] --output <OUTPUT> <SESSION_FILE>
```

| Argument or option | Description |
|---|---|
| `<SESSION_FILE>` | Session file to export |
| `-o, --output <OUTPUT>` | Output script file |
| `--comments [<BOOL>]` | Write each recorded command line as a comment above it (true/false; default true). The header and the cleanup comment are always written [default: true] [possible values: true, false] |

#### `p2a script history`

Show session history

```text
p2a script history [OPTIONS] <SESSION_FILE>
```

| Argument or option | Description |
|---|---|
| `<SESSION_FILE>` | Session file to display |

#### `p2a script run`

Run a p2a script

```text
p2a script run [OPTIONS] <SCRIPT_FILE>
```

| Argument or option | Description |
|---|---|
| `<SCRIPT_FILE>` | Script file to execute |

### `p2a chat`

Interactive chat with a p2a-mcp server (natural language analytics)

```text
p2a chat [OPTIONS]
```

| Argument or option | Description |
|---|---|
| `--url <URL>` | Server URL [env: P2A_SERVER_URL=] [default: http://localhost:8080] |
| `--provider <PROVIDER>` | LLM provider (openai, anthropic, ollama) [env: P2A_PROVIDER=] [default: ollama] |
| `--model <MODEL>` | Model name [env: P2A_MODEL=] [default: llama3.2] |
| `--api-key <API_KEY>` | API key for the provider [env: P2A_API_KEY] |
| `--access-token <ACCESS_TOKEN>` | Bearer token for a p2a-mcp server started with --access-token (visible ASCII, no spaces) [env: P2A_ACCESS_TOKEN] |
| `--base-url <BASE_URL>` | Override the LLM API base URL [env: P2A_BASE_URL=] |
| `--output-dir <OUTPUT_DIR>` | Directory for saved images/charts [default: .] |
| `--record <RECORD>` | Save a plain-text transcript to this file |
| `--script <SCRIPT>` | Read prompts from a file instead of interactive input (one prompt per line, or multi-line prompts separated by blank lines) |

### `p2a smoke-test`

Run a smoke test to verify the CLI works correctly

```text
p2a smoke-test [OPTIONS]
```
<!-- END GENERATED: p2a command reference -->
