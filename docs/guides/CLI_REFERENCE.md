# p2a CLI Reference

The `p2a` command-line interface provides direct access to all prompt2analytics capabilities for scripted workflows and reproducible analysis.

## Installation

```bash
cargo build --release -p p2a-cli
# Binary at: target/release/p2a
```

## Global Options

```
-F, --format <FORMAT>    Output format: text (default), json, table
--session <FILE>         Record commands to session file for reproducibility
--quiet                  Suppress non-essential output (progress spinners)
--verbose                Log to stderr: info; repeat for debug (twice) and trace (three times)
--max-memory <MB>        Memory limit in MB (0 = unlimited, the default); parsed, not yet enforced
-h, --help               Print help
-V, --version            Print version
```

Global options can come before or after the subcommand; `-q` and `-v` are not global: inside
subcommands they are the subcommand's own options (`ts arima -q` is the MA order,
`panel unit-root -v` the variable).

## Commands

### Data Commands

```bash
# Load a dataset (CSV, Parquet, Excel, Stata, SAS; the format comes from the extension)
p2a data load <PATH> [--name NAME] [--na-strings LIST]

# List loaded datasets
p2a data list

# Show dataset summary statistics
p2a data describe <DATASET>

# Preview first N rows
p2a data head <DATASET> [-n ROWS]

# Save a dataset (the format comes from the extension unless --file-format is given;
# both are case-insensitive)
p2a data save <DATASET> --output <FILE> [--file-format csv|parquet|json]
```

**Missing values in CSV files.** `data load` reads the field `NA` and the empty field as missing,
as R's `read.csv(na.strings = c("NA", ""))` does. `--na-strings` replaces that list with its own,
comma-separated (`--na-strings .` for a Stata or SAS export, `--na-strings NA,,.` for all three);
an empty unquoted field is missing in every case. Write a marker that starts with `-` in the `=`
form, `--na-strings=-99`. The list is stored in the session file, so later commands of the same
session reload the file with the same markers. `--na-strings` on any file other than CSV is an
error (Parquet, Excel, Stata and SAS files store missing values natively).

**Examples:**
```bash
p2a --session analysis.json data load /data/panel.csv --name firms
p2a --session analysis.json data describe firms
p2a --session analysis.json data head firms -n 20

# A CSV export that writes missing values as "."
p2a --session analysis.json data load stata_export.csv --name survey --na-strings .

# Save under a name without a known extension
p2a --session analysis.json data save firms --output firms.dat --file-format csv
```

### Data Manipulation (munge) Commands

`p2a munge <COMMAND> <DATASET> ... [-n|-o NAME]` writes its result to a new dataset (the name
flag differs by command; see `p2a munge <COMMAND> --help`). Commands: `filter`, `select`,
`drop`, `rename`, `sort`, `mutate`, `sample`, `join`, `concat`, `group-by`, `value-counts`,
`pivot`, `melt`, `drop-na`, `fill-na`, `deduplicate`, `lag`, `lead`, `diff`, `standardize`,
`bin`, `cast`, `one-hot`.

```bash
# Bin a numeric column into intervals, as R's cut()
p2a munge bin <DATASET> -c <COLUMN> [-m equal-width|quantile|custom] [-n <N_BINS>] \
    [--edges <E1,E2,...>] [--left-closed] [--include-lowest] [-o <NAME>]

# Convert a column to another type
p2a munge cast <DATASET> -c <COLUMN> --dtype int|float|string|bool [--strict] [-o <NAME>]

# Fill missing values
p2a munge fill-na <DATASET> -m mean|median|constant|forward|backward|zero [-c <COLUMNS...>] [--value <V>] [-n <NAME>]
```

- `munge bin`: intervals are closed on the right, `(a, b]`, as `cut()`; `--left-closed` closes
  them on the left, `[a, b)` (`cut(right = FALSE)`), and `--include-lowest` puts a value equal
  to the lowest edge in the first bin (the highest edge in the last bin with `--left-closed`),
  as `cut(include.lowest = TRUE)`. `--edges` gives the strictly increasing break points of
  `-m custom`. A missing value, or one outside the edges, is binned as missing.
- `munge cast`: values that cannot be converted become missing, and the command reports how
  many; `--strict` refuses the cast instead.
- `munge fill-na`: every strategy fills NaN cells as well as nulls.

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

# Arellano-Bond difference GMM
p2a panel gmm <DATASET> -y <DEP_VAR> [-x <INDEP_VARS...>] --entity <ENTITY_COL> --time <TIME_COL> \
    [--lags <N>] [--step onestep|twostep] [--min-lag <N>] [--max-lag <N>] [--collapse] \
    [--vcov robust|classical]

# Panel unit-root tests
p2a panel unit-root <DATASET> -v <VAR> --unit <UNIT_COL> --time <TIME_COL> [--test llc|ips|hadri|fisher] [--lags <N>]
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
  `missing`.
- `panel unit-root`: the LLC, IPS, Hadri or Fisher test, as `plm::purtest`; rows with a
  missing value are dropped, and a unit's remaining periods are joined in time order with a
  warning. The JSON keys: `test_type`, `statistic`, `p_value`, `null_hypothesis`,
  `alternative_hypothesis`, `model`, `lags_used`, `n_panels`, `avg_time_periods`, `warnings`,
  `unit_statistics` and `unit_p_values` (per unit, where the test has them) and `missing`.

Every panel command drops the rows with a missing value in the columns it uses. `panel hdfe`,
`feglm`, `gmm`, `pvcm`, `pmg`, `gls` and `unit-root` print the `Note:` line last and carry
`missing` in JSON (see [Missing values in results](#missing-values-in-results)); `panel fe` and
`re` list the drop under `Warnings:` and in the JSON `warnings`.

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

# Two-step difference GMM with one lag of the dependent variable, collapsed instruments
p2a panel gmm firms -y inv -x value capital --entity firm --time year --collapse
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
    [-p <POLY_ORDER>] [--kernel <TYPE>] [--bwselect <METHOD>] [--h <H>] [--b <B>] [--vce <VCE>]

# Regression Discontinuity (Fuzzy RD)
p2a causal fuzzy-rd <DATASET> -y <DEP_VAR> --treatment <TREAT_VAR> --running <RUNNING_VAR> \
    --cutoff <CUTOFF> [-p <POLY_ORDER>] [--kernel <TYPE>] [--h <H>] [--b <B>] [--vce <VCE>]

# Callaway-Sant'Anna staggered DiD
p2a causal staggered-did <DATASET> -y <OUTCOME> --treatment-time <FIRST_TREAT_COL> --time <TIME_COL> \
    --unit <UNIT_COL> [-x <COVARIATES...>] [--comparison nevertreated|notyettreated] \
    [--method or|ipw|dr] [--bootstrap <N>]

# Causal mediation analysis
p2a causal mediation <DATASET> -y <OUTCOME> -t <TREATMENT> -m <MEDIATOR> [-x <COVARIATES...>] [--bootstrap <N>]
```

**V-method options:** `datadriven` (default), `equal`

**RD kernel types:** `triangular` (default), `epanechnikov`, `uniform`

**RD bandwidths (`causal rd`, `causal fuzzy-rd`):** `--h` is the main bandwidth on both sides
(alias `--bandwidth`; default: p2a's MSE-type selector, chosen with `--bwselect mserd` (default),
`msetwo` or `cerrd` in `causal rd`), `--b` the bias bandwidth (alias `--bias-bandwidth`;
default: h). `--vce` is the variance estimator: `nn` (default, as rdrobust), `hc0`, `hc1`, `hc2`
or `hc3`. The output reports the robust bias-corrected estimate and the conventional one, as
`rdrobust`, with the bandwidths and the effective observations on each side.

**`causal iv`** fits robust standard errors and prints the first-stage diagnostics as
`First-stage diagnostics (HC1 Wald F of the excluded instruments):`, one
`<endog> F-statistic: F on df1 and df2 DF` line per endogenous regressor; its JSON adds a
`first_stage` object (`f_type`, `f_stats`, `df`, `p_values`, `strong_instruments`).

**`causal staggered-did`:** `--bootstrap 0` gives analytic influence-function standard errors
(did's `bstrap = FALSE`; without covariates only); otherwise at least 50 unit-bootstrap
replicates (default 999). The JSON's `pretrend_test` holds `chi2`, `df`, `p_value`, `f_stat`
and `df2` (the F form of a bootstrap covariance; `null` with the analytic one) and `method`.

`causal iv`, `mediation`, `rd`, `fuzzy-rd` and `staggered-did` drop the rows with a missing value
in the columns they use and report them (`Note:` line and `missing`).

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
p2a causal rd mydata -y test_score --running age --cutoff 65 --h 5

# Fuzzy RD: treatment probability changes at cutoff
p2a causal fuzzy-rd mydata -y earnings --treatment enrolled --running score --cutoff 50

# Sharp RD with a fixed main bandwidth and HC1 standard errors
p2a causal rd mydata -y outcome --running running_var --cutoff 0 --h 2.5 --vce hc1

# Staggered DiD with analytic standard errors
p2a causal staggered-did panel -y y --treatment-time first_treat --time year --unit state --bootstrap 0
```

### Discrete Choice Commands

```bash
# Logit model
p2a discrete logit <DATASET> -y <DEP_VAR> -x <INDEP_VARS...>

# Probit model
p2a discrete probit <DATASET> -y <DEP_VAR> -x <INDEP_VARS...>

# Ordered logit / probit (a text outcome needs its order, lowest first)
p2a discrete o-logit <DATASET> -y <DEP_VAR> -x <INDEP_VARS...> [--levels <L1,L2,...>]
p2a discrete o-probit <DATASET> -y <DEP_VAR> -x <INDEP_VARS...> [--levels <L1,L2,...>]

# Mixed logit (long format; --individual for repeated choices by one person)
p2a discrete mixed-logit <DATASET> --choice-id <COL> --alt-id <COL> -y <CHOICE> -x <VARS...> [--random-vars <VARS...>] [--n-draws <N>] [--individual <COL>]
```

**Examples:**
```bash
p2a discrete logit mydata -y purchased -x income age education
p2a discrete probit mydata -y voted -x age income party_id
p2a discrete o-logit survey -y satisfaction -x age income --levels low,mid,high
p2a discrete mixed-logit trips --choice-id trip --alt-id mode -y chosen -x cost time --random-vars cost --individual person
```

`logit`, `probit`, `mlogit` (multinomial), `o-logit`, `o-probit`, `negbin`, `zip`, `zinb`
and `hurdle` drop rows with a missing outcome or regressor and say so (see
[Missing values in results](#missing-values-in-results)); `clogit` and `mixed-logit`
refuse a missing id, alternative, choice or attribute, as R's mlogit does. The ordered,
count and mixed-logit commands print `Converged: Yes|No (k iterations)`.

### Survival Commands

```bash
# Kaplan-Meier survival curves
p2a survival km <DATASET> -t <TIME> -e <EVENT> [-g <GROUP>] [--conf <LEVEL>] [--conf-type log|log-log|plain]

# Log-rank test
p2a survival log-rank <DATASET> -t <TIME> -e <EVENT> -g <GROUP>

# Cox proportional hazards
p2a survival cox <DATASET> -t <TIME> -e <EVENT> -x <COVARIATES...> [--ties breslow|efron] [--robust]
```

The status column `-e` is coded as R's `Surv()`: 0/1 (1 = event), 1/2 (2 = event, as R's `lung`)
or TRUE/FALSE; any other code is an error. Integer times are accepted. `--conf-type` is R
`survfit`'s `conf.type`: `log` (the default, as R), `log-log` or `plain`; the median follows R's
`quantile.survfit`. Rows with a missing time, status, group or covariate are dropped and reported
(`Note:` line, `missing`).

JSON keys: `survival km` prints one object per group with `n_obs`, `total_events`,
`total_censored`, `conf_level`, `conf_type`, `median_survival`, `median_ci`, `times`,
`n_at_risk`, `n_events`, `n_censored`, `survival`, `std_errors`, `ci_lower`, `ci_upper` (a
value that is not defined, such as the SE where S(t) = 0, is `null`) and `missing`;
`survival log-rank` has `chi_squared`, `degrees_of_freedom`, `p_value`, `groups`,
`n_per_group`, `events_per_group`, `expected_per_group`, `obs_minus_exp`, `variance` and
`missing`; `survival cox` has `variables`, `coefficients`, `hazard_ratios`, `std_errors`,
`z_stats`, `p_values`, `log_likelihood`, `concordance`, `n_obs`, `n_events`, `method` and
`missing`.

**Examples:**
```bash
p2a survival km lung -t time -e status --conf-type log-log
p2a survival log-rank lung -t time -e status -g sex
p2a survival cox lung -t time -e status -x age sex --ties efron
```

### Time Series Commands

```bash
# ARIMA model
p2a ts arima <DATASET> --col <COL> -p <AR> -d <DIFF> -q <MA> [--horizon <N>] [--include-mean <BOOL>] [--drift]

# MSTL decomposition
p2a ts mstl <DATASET> --col <COL> --periods <PERIODS...>

# Vector autoregression
p2a ts var <DATASET> --cols <COLS...> --lags <N>
```

**`ts arima`** fits by conditional sum of squares, as R's `arima(method = "CSS")`. The mean (R's
`intercept`) is estimated by default when d = 0; `--include-mean false` drops it, and it must be
false or omitted when d > 0. `--drift` fits a linear trend in the levels, as
`forecast::Arima(include.drift = TRUE)`, for d = 0 or 1. With `--horizon` it prints point
forecasts with 80% and 95% prediction intervals, as `forecast(Arima(...))`. The JSON keys include
`model`, `label` (forecast::Arima's model label), `coefficients`, `std_errors`, `sigma2`, `loglik`,
`aic`, `converged`, `iterations`, `include_mean`, `drift`, `warnings` and, with `--horizon`, the
forecasts with `lower_80`, `upper_80`, `lower_95` and `upper_95`. A series with missing values
is refused, with how to fill the gaps first (`p2a munge fill-na`): dropping a value would splice
the series.

**Examples:**
```bash
# ARIMA(1,1,1) with 12-step forecast
p2a ts arima sales --col revenue -p 1 -d 1 -q 1 --horizon 12

# Seasonal decomposition (weekly + yearly)
p2a ts mstl sales --col revenue --periods 7 365

# VAR(2) for multiple series
p2a ts var macro --cols gdp inflation unemployment --lags 2
```

### Statistics Commands

```bash
# t-tests
p2a stats t-test-one <DATASET> -c <COL> [-m <MU>] [-a two-sided|less|greater] [--conf-level <L>]
p2a stats t-test-two <DATASET> --col1 <COL1> --col2 <COL2> [-a ...] [--equal-var]
p2a stats t-test-paired <DATASET> --col1 <COL1> --col2 <COL2> [-a ...]

# ANOVA and rank tests
p2a stats anova <DATASET> -r <RESPONSE> -f <FACTOR>
p2a stats anova-two <DATASET> -r <RESPONSE> --factor-a <A> --factor-b <B> [--interaction] [--ss-type I|II|III]
p2a stats kruskal <DATASET> -c <COL> -g <GROUP>
p2a stats wilcoxon <DATASET> --col1 <COL1> --col2 <COL2> [--paired] [-a ...]
p2a stats friedman <DATASET> -v <VALUE> -g <GROUP> -b <BLOCK>

# Contingency tables
p2a stats chisq-indep <DATASET> --row-var <ROW_VAR> --col-var <COL_VAR>
p2a stats chisq-gof <DATASET> -o <OBSERVED_COUNTS>
p2a stats fisher <DATASET> --row-var <ROW_VAR> --col-var <COL_VAR>

# Distributions and summaries
p2a stats shapiro <DATASET> -c <COL>
p2a stats ks-test <DATASET> --x <X> --y <Y> [-a ...]
p2a stats fivenum <DATASET> -c <COL>      # also iqr, mad

# Serial correlation and unit roots
p2a stats acf <DATASET> -c <COL> [--lag-max <N>] [--partial]
p2a stats pacf <DATASET> -c <COL> [--lag-max <N>]
p2a stats box-test <DATASET> -c <COL> [--lag <N>] [--box-pierce] [--fitdf <N>]
p2a stats pp-test <DATASET> -c <COL> [--lshort]
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

**Examples:**
```bash
p2a stats t-test-two trial --col1 treated --col2 control
p2a stats anova-two crops -r yield --factor-a fertilizer --factor-b irrigation --interaction --ss-type II
p2a stats fisher survey --row-var smoker --col-var outcome
p2a stats acf sales -c revenue --lag-max 24
```

### Machine Learning Commands

```bash
# K-means clustering
p2a ml kmeans <DATASET> --cols <COLS...> -k <N> [--seed <N>] [--max-iter <N>]

# Principal Component Analysis
p2a ml pca <DATASET> --cols <COLS...> [--n-components <N>] [--transform]

# Linear support vector machine (binary target)
p2a ml svm <DATASET> --cols <COLS...> -y <TARGET> [-c <C>] [--max-iter <N>] [--tolerance <TOL>]
```

`ml svm` drops the rows with a missing feature or target and prints the `Note:` line (`missing`
in JSON); `kmeans` and `pca` refuse missing values.

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

The data commands drop the rows that are incomplete in the plotted columns, together, and print
the `Note:` line; integer columns are accepted. `viz box -g` orders the groups as R's `factor()`
(numbers numerically, text in byte order). `viz heatmap` correlates the complete rows, as R's
`cor(use = "complete.obs")`: a column that is constant on them has NA correlations (shown as
`NA` in its cells; its diagonal stays 1), and the message names it. `viz dendrogram` refuses
missing values, and `viz irf` refuses a CI column it cannot read as numbers. With `-F json` a
plot command prints `{"message": ...}`, plus `missing` when rows were dropped.

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

### Table (`-F table`)
Currently the text layout; a separate table rendering is not implemented yet.

### Missing values in results
A method that drops rows with missing values in the columns it uses reports them. Text and table
output end with one line, `Note: 3 of 120 rows dropped because of missing values (y: 2, x1: 1)`
(rows dropped, rows in the dataset, and the missing cells per column; a row missing in two columns
counts in both). JSON output has the key `missing` instead,
`{"n_total": 120, "n_used": 117, "n_dropped": 3, "by_column": {"y": 2, "x1": 1}}`. Neither appears
when no row was dropped.

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

What is recorded: each command from its subcommand on, as typed, without the global options
before the subcommand and without `--session` (wherever it appears before a `--`). The exported
script supplies its own session file, `p2a --session "$SESSION_FILE" <command>`. Global options
typed after the subcommand (`-F json`, `--quiet`) are recorded and replayed. The commands
`script`, `chat` and `smoke-test` are not recorded.

## Error Handling

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
