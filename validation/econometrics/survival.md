# Validation: Survival Analysis (Kaplan-Meier, log-rank, Cox PH, AFT, competing risks, survival columns)

## Method Overview

This document covers what is pinned to R's `survival` package:
- **Kaplan-Meier**: `run_kaplan_meier`, `run_kaplan_meier_with` (MCP `kaplan_meier`, CLI
  `survival km`): the survival curve, Greenwood standard errors, pointwise confidence limits for
  `conf_type` "log" (default, as R), "log-log" and "plain", the median and its limits.
- **Log-rank test**: `log_rank_test` (MCP `log_rank`, CLI `survival log-rank`), as
  `survdiff(rho = 0)`.
- **Cox proportional hazards**: `run_cox_ph` (MCP `cox_ph`, CLI `survival cox`), as `coxph`: Efron
  (default) and Breslow ties, model-based, robust (Lin-Wei) and clustered SEs, the LR, Wald, score and robust
  score tests, convergence and aliased covariates.
- **Accelerated failure time**: `run_aft` (MCP `aft`, CLI `survival aft`), as `survreg` with
  `dist = "weibull"`, `"exponential"`, `"lognormal"`, `"loglogistic"`.
- **Competing risks**: `run_competing_risks` (MCP `competing_risks`, CLI `survival competing-risks`): the
  Aalen-Johansen cumulative incidence with survfit's multi-state standard errors and log-scale limits.
- **Survival columns** of all five survival entry points (Kaplan-Meier, log-rank, Cox PH, AFT,
  competing risks): integer, float and logical columns, rows with missing values dropped and
  reported, status coded as `Surv()` codes it.

The Cox model's concordance is not R-pinned (ECON-89).

## Reference Implementation

| Package | Language | Functions | Version |
|---------|----------|-----------|---------|
| survival | R | `Surv()`, `survfit()`, `quantile.survfit()`, `survdiff()`, `coxph()`, `survreg()` | 3.5.8 on R 4.3.3 |

The R reference cases are in `validation/r/cases/survival.R` and (Cox, AFT) `validation/r/cases/surv_cox_aft.R`;
`Rscript validation/r/run.R --check` recomputes them and compares with the committed
`validation/expected/<case>.json`. The registries (`validation/methods/survival.json`,
`validation/methods/surv_cox_aft.json`) list the tests that load them:

| Registry id | Cases | Tests |
|-------------|-------|-------|
| `surv_km` | `surv_km_lung`, `surv_km_lung_by_sex`, `surv_km_small`, `surv_km_median_rules` | `test_validate_km_lung`, `test_validate_km_lung_by_sex`, `test_validate_km_small`, `test_validate_km_median_rules`, `test_validate_km_conf_level_and_type_validation`; MCP `test_validate_mcp_kaplan_meier_lung`, `test_validate_mcp_kaplan_meier_rejects_unknown_conf_type`; CLI `test_validate_cli_survival_km_lung`, `test_validate_cli_survival_km_rejects_unknown_conf_type` |
| `surv_logrank` | `surv_logrank_lung`, `surv_logrank_small`, `surv_km_median_rules` | `test_validate_logrank_lung`, `test_validate_logrank_small`, `test_validate_logrank_zero_expected_group`; MCP `test_validate_mcp_log_rank_lung`; CLI `test_validate_cli_survival_log_rank_lung`, `test_validate_cli_survival_km_groups_missing_note` |
| `surv_data_handling` | `surv_km_lung`, `surv_logrank_lung`, `surv_logrank_small`, `surv_km_small` | `test_validate_na_survival_int64_columns`, `test_validate_surv_status_coding_lung`, `test_validate_surv_status_invalid_codes_are_errors`, `test_validate_na_survival_missing_rows_are_dropped_and_reported`, `test_validate_na_survival_bad_columns_name_the_column`; MCP `test_validate_mcp_survival_tools_accept_integer_columns`; CLI `test_validate_cli_survival_cox_aft_integer_columns` |
| `surv_cox` | `surv_cox_robust`, `surv_cox_ties_small` | `test_validate_cox_ties_small`, `test_validate_cox_breslow_row_order_invariant`, `test_validate_cox_efron_model`, `test_validate_cox_breslow_model`, `test_validate_cox_converges_at_large_scale`, `test_validate_cox_aliased_covariate`, `test_validate_cox_default_ties_is_efron` |
| `surv_cox_robust` | `surv_cox_robust` | `test_validate_cox_efron_robust`, `test_validate_cox_breslow_robust`, `test_validate_cox_cluster` |
| `surv_competing_risks` | `surv_competing_risks` | `test_validate_competing_risks_se_survfit` |
| `surv_aft` | `surv_aft` | `test_validate_aft_weibull`, `test_validate_aft_exponential`, `test_validate_aft_lognormal`, `test_validate_aft_loglogistic`, `test_validate_aft_converges_at_any_covariate_scale`, `test_validate_aft_aliased_covariate` |

The core tests are in `crates/p2a-core/src/econometrics/survival.rs`, the MCP tests in
`crates/p2a-mcp/src/tools/handlers/survival.rs`, the CLI tests in
`crates/p2a-cli/tests/tier2/wp09_survival.rs`.

---

## Test Case 1: Kaplan-Meier Estimator (`surv_km_small`)

### R Code
```r
library(survival)

# validation/datasets/generated/km_small.csv
d <- data.frame(time = 1:10, event = c(1, 1, 0, 1, 1, 0, 1, 0, 1, 1))
for (ct in c("log", "log-log")) {
  f <- survfit(Surv(time, event) ~ 1, data = d, conf.type = ct)
  summary(f, times = f$time)   # surv, std.err (S scale), lower, upper
  quantile(f, 0.5)             # median and its limits
}
```

### Expected Results (R, rounded to 4 decimals)

| Time | N at Risk | Events | S(t) | SE | log lower | log upper | log-log lower | log-log upper |
|------|-----------|--------|------|------|-----------|-----------|---------------|---------------|
| 1 | 10 | 1 | 0.9000 | 0.0949 | 0.7320 | 1.0000 | 0.4730 | 0.9853 |
| 2 | 9 | 1 | 0.8000 | 0.1265 | 0.5868 | 1.0000 | 0.4087 | 0.9459 |
| 3 | 8 | 0 | 0.8000 | 0.1265 | 0.5868 | 1.0000 | 0.4087 | 0.9459 |
| 4 | 7 | 1 | 0.6857 | 0.1515 | 0.4447 | 1.0000 | 0.3046 | 0.8871 |
| 5 | 6 | 1 | 0.5714 | 0.1638 | 0.3258 | 1.0000 | 0.2172 | 0.8146 |
| 6 | 5 | 0 | 0.5714 | 0.1638 | 0.3258 | 1.0000 | 0.2172 | 0.8146 |
| 7 | 4 | 1 | 0.4286 | 0.1743 | 0.1931 | 0.9512 | 0.1140 | 0.7185 |
| 8 | 3 | 0 | 0.4286 | 0.1743 | 0.1931 | 0.9512 | 0.1140 | 0.7185 |
| 9 | 2 | 1 | 0.2143 | 0.1748 | 0.0433 | 1.0000 | 0.0129 | 0.5795 |
| 10 | 1 | 1 | 0.0000 | NA | NA | NA | NA | NA |

Median survival 7; its limits are (4, NA) with log and (1, NA) with log-log: the upper band never
falls to 0.5. At S = 0 the standard error and both limits are NA (NaN in p2a). The table this
document showed before (log-log limits 0.6827-0.9698 at t = 1, SE 0.1533 at t = 4) was not R's.

Test: `test_validate_km_small` (times exact, S(t) `Tol::coef(10)`, SE and limits `Tol::se(10)` /
`Tol::stat(10)`, medians exact).

---

## Test Case 2: Stratified Kaplan-Meier and the median rules (`surv_km_median_rules`)

For the 12-row data of Test Case 3 (`logrank_small.csv`), R's medians by group are 5 (group 0)
and 6 (group 1, where S = 0.5 from t = 4 to t = 8, so the midpoint) - not 5.5 and 4.0 as this
document used to say.

The stratified fit and `quantile.survfit`'s rules are pinned on a 16-row generated dataset
(`validation/datasets/generated/km_median_rules.csv`) with three groups:

```r
d <- read.csv("validation/datasets/generated/km_median_rules.csv")
survfit(Surv(time, event) ~ group, data = d)            # strata last, mid, never
quantile(survfit(Surv(time, event) ~ 1, data = d[d$group == "mid", ], conf.type = ct), 0.5)
```

| Group | n | S(t) at its times | Median | log limits | log-log limits | plain limits |
|-------|---|-------------------|--------|------------|----------------|--------------|
| last | 4 | 0.75, 0.5, 0.5, 0.5 (t = 1-4) | 3 = (2 + 4) / 2, the curve ends at 0.5 (last-point rule) | (1, NA) | (1, NA) | (1, NA) |
| mid | 7 | 1, 0.8333, 0.6667, 0.5, 0.5, 0.25, 0 (t = 1-7) | 5 = (4 + 6) / 2, S = 0.5 over [4, 6) (midpoint rule) | (3, NA) | (2, NA) | (3, NA) |
| never | 5 | 0.8 throughout (t = 1-5) | NA | (NA, NA) | (1, NA) | (1, NA) |

The log-log limits are NA where S = 1 (`mid`, t = 1); the plain lower limit is clipped to 0
(`mid`, t = 6). The `never` group has a lower limit for a median that is never reached. Strata
follow R's `factor()` order (`last`, `mid`, `never`).

Test: `test_validate_km_median_rules` (every limit of the three types for each group, the medians
and their limits exact).

---

## Test Case 3: Log-Rank Test (`surv_logrank_small`)

### R Code
```r
library(survival)

# validation/datasets/generated/logrank_small.csv
time <- c(1, 2, 3, 5, 6, 7, 2, 3, 4, 5, 8, 9)
event <- c(1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1)
group <- c(rep("0", 6), rep("1", 6))
s <- survdiff(Surv(time, event) ~ group)
pchisq(s$chisq, 1, lower.tail = FALSE)
```

### Expected Results (R)

| Statistic | R value | Tolerance |
|-----------|---------|-----------|
| Chi-squared | 0.3052976438092901 | `Tol::stat(12)` (1e-5 relative) |
| df | 1 | exact |
| p-value | 0.58058017726554 | `Tol::p_value(12)` |
| Observed | 4, 5 | exact |
| Expected | 3.282106782106782, 5.717893217893217 | `Tol::stat(12)` |

Before ECON-6 was fixed, p2a gave 6.23 (p 0.0126) here: the rows tied at an event time left the
risk set twice. Test: `test_validate_logrank_small`.

A group whose members all leave before the first event has no expected events; `survdiff`
excludes it from the test and from the degrees of freedom. On `km_median_rules.csv`,
`survdiff(Surv(time_early, event_early) ~ group)` gives chi-squared 0.23795720545865132 on 1 df
(p 0.6256854372911773), expected 1.5160, 5.4840 and 0 for `last`, `mid`, `never`, the same
statistic as the test without the `never` rows. Test: `test_validate_logrank_zero_expected_group`.

When every subject at risk at the first event time has the event then, nobody is at risk at a
later event time and the variance of O - E is zero. `survdiff` stops ("Lapack routine dgesv:
system is exactly singular", survival 3.5.8), and p2a refuses the test with an
`InvalidSpecification` error (before this was fixed it reported chi-squared 0 on 1 df, p 1).
Property test, not registered (no R value): `test_logrank_zero_variance_is_an_error` (two and
three groups that all have the event at t = 1, one with a row censored before it; and a layout
with one subject left after t = 1, chi-squared 1 on 1 df, as survdiff).

---

## Test Case 4: Lung Cancer Dataset (R's `lung`)

### R Code
```r
library(survival)

# validation/datasets/generated/lung.csv: survival::lung, integer days, status 1 = censored, 2 = dead
lung <- read.csv("validation/datasets/generated/lung.csv")
survfit(Surv(time, status) ~ 1, data = lung)                          # default conf.type "log"
survfit(Surv(time, status) ~ sex, data = lung, conf.type = "log-log")
survdiff(Surv(time, status) ~ sex, data = lung)
survdiff(Surv(time, status) ~ ph.ecog, data = lung)                   # one ph.ecog is NA
```

### Expected Results (R)

**Kaplan-Meier, all 228 patients** (165 events, 186 distinct times; summary at the given times):

| Time | N at Risk | S(t) | SE | log lower | log upper | log-log lower | log-log upper | plain lower | plain upper |
|------|-----------|------|------|-----------|-----------|---------------|---------------|-------------|-------------|
| 5 | 228 | 0.9956 | 0.0044 | 0.9871 | 1.0000 | 0.9693 | 0.9994 | 0.9870 | 1.0000 |
| 11 | 227 | 0.9825 | 0.0087 | 0.9656 | 0.9996 | 0.9539 | 0.9934 | 0.9654 | 0.9995 |
| 100 | 196 | 0.8640 | 0.0227 | 0.8206 | 0.9096 | 0.8122 | 0.9023 | 0.8195 | 0.9085 |
| 310 | 85 | 0.4950 | 0.0352 | 0.4306 | 0.5691 | 0.4242 | 0.5618 | 0.4260 | 0.5641 |
| 500 | 41 | 0.2933 | 0.0351 | 0.2320 | 0.3707 | 0.2265 | 0.3630 | 0.2245 | 0.3620 |
| 765 | 10 | 0.0881 | 0.0257 | 0.0498 | 0.1560 | 0.0463 | 0.1466 | 0.0378 | 0.1384 |

Times 100 and 500 are not observed times: S(t) is the value at the last time before them, and the
number at risk is the risk set at the next observed time. Median survival 310 days, with limits
(285, 363) under the default log type and (284, 361) under log-log and plain. Before ECON-7 was
fixed, p2a's log-log band was [0, 0] at t = 5 and the median CI was "5 - 5".

**Kaplan-Meier by sex (log-log):** median 270 days (210, 306) for sex 1 and 426 days (345, 524)
for sex 2.

**Log-rank test by sex:** chi-squared 10.326741954885632 on 1 df, p 0.0013111645203554882;
observed 112 and 53, expected 91.5817390295728 and 73.41826097042721 (p2a gave 252.6 before the
ECON-6 fix). **By ph.ecog:** 227 rows (the row with a missing ph.ecog is dropped), groups 0-3,
chi-squared 21.962131682476006 on 3 df.

`status` is coded 1/2 (2 = dead), as `Surv()` reads it; p2a reads it the same way (ECON-77: it
used to count every row as an event).

Tests: `test_validate_km_lung` (the six summary times for all three types, and the whole curve,
SE on the log scale and both limits at all 186 times under the default type),
`test_validate_km_lung_by_sex`, `test_validate_logrank_lung` (statistic, df, p, observed,
expected, n per group and the variance matrix for both groupings).

The Cox and AFT models on lung are Test Cases 5 to 7.

## Test Case 5: Cox Proportional Hazards (`surv_cox_robust`, `surv_cox_ties_small`)

### R Code
```r
library(survival)
# validation/r/cases/surv_cox_aft.R; lung as in Test Case 4 (227 rows once ph.ecog's NA is dropped)
ctl <- coxph.control(eps = 1e-12, toler.chol = 1e-13, iter.max = 100)
coxph(Surv(time, status) ~ age + sex + ph.ecog, data = lung, ties = "efron",   control = ctl)
coxph(Surv(time, status) ~ age + sex + ph.ecog, data = lung, ties = "breslow", control = ctl)
coxph(Surv(time, event) ~ x, data = cox_ties_small, ties = "efron" | "breslow", control = ctl)
coxph(Surv(time, status) ~ I(age * s) + sex, data = lung)        # s = 1e-6, 1, 1/365.25, 1e6; default eps
coxph(Surv(time, status) ~ age + sex + age2, data = lung)        # age2 = 2 age + sex: NA
```

### Expected Results (R, 6 significant digits; full digits in `validation/expected/surv_cox_robust.json`)

Model-based standard errors, n = 227, 164 events:

| Ties | Variable | coef | se(coef) | z | p | HR lower | HR upper |
|------|----------|------|----------|---|---|----------|----------|
| Efron | age | 0.0110668 | 0.00926741 | 1.19416 | 0.232416 | 0.992928 | 1.02966 |
| Efron | sex | -0.552612 | 0.167739 | -3.29448 | 0.000986051 | 0.414213 | 0.799435 |
| Efron | ph.ecog | 0.463728 | 0.113577 | 4.08293 | 4.44707e-05 | 1.27268 | 1.98642 |
| Breslow | age | 0.0110411 | 0.00926677 | 1.19148 | 0.233467 | 0.992904 | 1.02963 |
| Breslow | sex | -0.55189 | 0.167742 | -3.2901 | 0.00100151 | 0.41451 | 0.800019 |
| Breslow | ph.ecog | 0.462947 | 0.113574 | 4.07617 | 4.57837e-05 | 1.27169 | 1.98486 |

| Ties | logLik | null logLik | LR test | Wald test | Score test (df 3) |
|------|--------|-------------|---------|-----------|-------------------|
| Efron | -729.23 | -744.48 | 30.5007 (p 1.08282e-06) | 29.9293 (p 1.42817e-06) | 30.4999 (p 1.08321e-06) |
| Breslow | -729.489 | -744.693 | 30.4082 (p 1.13242e-06) | 29.839 (p 1.49197e-06) | 30.4064 (p 1.13342e-06) |

**Tie fixture** (`validation/datasets/generated/cox_ties_small.csv`: 10 rows, 5 distinct times, censored rows
tied with deaths), in file order and reversed:

| Ties | coef | se | logLik | null logLik | score test |
|------|------|----|--------|-------------|------------|
| Efron | -0.0502312 | 0.708672 | -12.2115 | -12.214 | 0.00502513 |
| Breslow | -0.0502312 | 0.708672 | -13.1436 | -13.1461 | 0.00502513 |

Before ECON-46 was fixed, Breslow let rows enter the risk set one at a time, so tied deaths saw different
denominators: the fixture gave 0.366 in file order and -0.480 reversed, lung's age 0.011071 and 0.011049.

**Ties.** Rows are grouped in blocks of exactly equal times, processed in descending time. Breslow scores every
death of a block against the full risk set {t_j >= t} (one d-weighted term); Efron removes the fraction (r - 1)/d of
the block's deaths' risk from the r-th death's denominator. Efron is the default, as in coxph.

**Convergence** (coxph.fit / coxfit6). Covariates are centred and scaled by their mean absolute deviation; Newton
starts at beta = 0, halves the step while the log partial likelihood falls, and stops when
|1 - loglik_old / loglik| <= `tolerance` (coxph.control `eps`, default 1e-9; `max_iter` defaults to 25,
coxph's `iter.max` is 20). Coefficients and the information are mapped back to the original units. Lung's age multiplied by
1e-6, 1, 1/365.25 and 1e6 converges in 3 iterations at every scale with the default eps (coxph: 3; before X-15 was
fixed the absolute max |score| < 1e-9 rule ran 25 iterations and reported no convergence at the large scales). At the
limit the result carries a non-convergence warning; a monotone likelihood gives coxph's "Loglik converged before
variable ...; coefficient may be infinite".

**Aliasing.** A covariate that is constant or collinear with earlier ones (pivoted QR on [1 | X], the later column
dropped as R does) has NaN in every per-variable vector, is listed in `aliased`, and the warning "Coefficients not
defined because of singularities: ..." is added; the tests have df = the identified coefficients. Lung with
age2 = 2 age + sex: logLik -742.848, LR test 14.1231 on 2 df.

Tests: `test_validate_cox_efron_model`, `test_validate_cox_breslow_model` (every quantity above),
`test_validate_cox_ties_small`, `test_validate_cox_breslow_row_order_invariant`,
`test_validate_cox_converges_at_large_scale`, `test_validate_cox_aliased_covariate`,
`test_validate_cox_default_ties_is_efron`; properties: `test_cox_iteration_limit_reports_non_convergence`,
`test_cox_monotone_likelihood_warns`, `test_cox_rejects_bad_iteration_settings`.

---

## Test Case 6: Robust and Clustered Cox Standard Errors (`surv_cox_robust`)

### R Code
```r
coxph(Surv(time, status) ~ age + sex + ph.ecog, data = lung, ties, robust = TRUE, control = ctl)
coxph(Surv(time, status) ~ age + sex + ph.ecog + cluster(inst), data = lung, ties, control = ctl)
fit$rscore                                                       # robust score test
```

`CoxConfig { robust_se: true, .. }` is `robust = TRUE`; `cluster: Some("inst")` is `cluster(inst)` and implies
robust. With V the inverse information (scaled space) and U the n x q score residuals (Efron weights within tie
blocks), dfbeta = U V, summed within clusters when there are clusters, and V_r = dfbeta' dfbeta. The SEs are
sqrt(diag(V_r)); z, p, the HR interval and the Wald test b' V_r^-1 b use them; `naive_std_errors` keeps the
model-based SEs. The robust score test is u0' (U0' U0)^-1 u0 with the score residuals at beta = 0, summed within
clusters. A cluster column with a single level is refused; with no more clusters than coefficients the robust Wald
and score tests are NaN with a warning.

**Robust** (n = 227):

| Ties | Variable | robust se | z | p | HR lower | HR upper |
|------|----------|-----------|---|---|----------|----------|
| Efron | age | 0.00988139 | 1.11996 | 0.262731 | 0.991734 | 1.0309 |
| Efron | sex | 0.164748 | -3.35428 | 0.000795717 | 0.416648 | 0.794763 |
| Efron | ph.ecog | 0.124159 | 3.73496 | 0.000187744 | 1.24655 | 2.02805 |
| Breslow | age | 0.00986141 | 1.11963 | 0.262871 | 0.991747 | 1.03083 |
| Breslow | sex | 0.164484 | -3.35529 | 0.000792826 | 0.417166 | 0.794925 |
| Breslow | ph.ecog | 0.123963 | 3.73454 | 0.000188055 | 1.24606 | 2.02569 |

| Ties | robust Wald (df 3) | robust score (df 3) |
|------|--------------------|---------------------|
| Efron | 23.8023 (p 2.74702e-05) | 27.4263 (p 4.79211e-06) |
| Breslow | 23.7919 (p 2.76075e-05) | 27.4252 (p 4.79458e-06) |

**Clustered by `inst`** (226 rows, 163 events, 18 clusters; the row with a missing `inst` is dropped and reported):

| Ties | Variable | coef | model se | clustered se | z | p |
|------|----------|------|----------|--------------|---|---|
| Efron | age | 0.0112322 | 0.00926211 | 0.00688873 | 1.63051 | 0.102993 |
| Efron | sex | -0.556593 | 0.168071 | 0.112287 | -4.9569 | 7.16282e-07 |
| Efron | ph.ecog | 0.469216 | 0.11429 | 0.117616 | 3.98939 | 6.62423e-05 |
| Breslow | age | 0.0112049 | 0.00926152 | 0.00686751 | 1.63158 | 0.102767 |
| Breslow | sex | -0.555825 | 0.168074 | 0.111998 | -4.96281 | 6.94816e-07 |
| Breslow | ph.ecog | 0.468379 | 0.114286 | 0.117123 | 3.99902 | 6.36045e-05 |

| Ties | robust Wald (df 3) | robust score (df 3) |
|------|--------------------|---------------------|
| Efron | 33.8617 (p 2.11894e-07) | 7.26863 (p 0.0638112) |
| Breslow | 34.0614 (p 1.92295e-07) | 7.28292 (p 0.0634064) |

Before ECON-45 was fixed, `robust_se` was never read: the "robust" fit printed the model-based SEs.

Tests: `test_validate_cox_efron_robust`, `test_validate_cox_breslow_robust`, `test_validate_cox_cluster`;
properties: `test_cox_cluster_needs_two_levels`, `test_cox_robust_tests_need_more_clusters_than_coefficients`.

---

## Test Case 7: Accelerated Failure Time Models (`surv_aft`)

### R Code
```r
ctl <- survreg.control(rel.tolerance = 1e-12, maxiter = 100)
survreg(Surv(time, status) ~ age + sex + ph.ecog, data = lung, dist = d, control = ctl)
# d = "weibull", "exponential", "lognormal", "loglogistic"; AIC(fit), BIC(fit)
```

The model is ln T = x'beta + sigma eps with eps extreme-value (Weibull, exponential), standard normal (log-normal)
or logistic (log-logistic); the exponential fixes sigma = 1. The log-likelihood is reported on the time scale, as
survreg's: the log-density of ln T minus the sum of ln t over the events. AIC and BIC count p + 1 parameters (p for
the exponential). Newton starts from least squares on ln t, works on (beta, ln sigma) with the information
equilibrated and damped (Levenberg) when it is not positive definite, halves the step while the log-likelihood falls,
and stops when the relative change is at most `tolerance` (survreg.control `rel.tolerance`, default 1e-9). Normal
tails use a log survival function that does not underflow.

| Distribution | (Intercept) | age | sex | ph.ecog | scale | Log(scale) se | logLik | AIC | BIC |
|--------------|-------------|-----|-----|---------|-------|---------------|--------|-----|-----|
| weibull | 6.27344 | -0.00747544 | 0.401091 | -0.339638 | 0.731109 | 0.0613465 | -1132.44 | 2274.88 | 2292 |
| exponential | 6.37342 | -0.0102174 | 0.509061 | -0.405017 | 1 | fixed | -1143.56 | 2295.13 | 2308.83 |
| lognormal | 6.49479 | -0.0191819 | 0.521953 | -0.355567 | 1.02863 | 0.0559606 | -1146.88 | 2303.76 | 2320.89 |
| loglogistic | 5.93669 | -0.00807992 | 0.486624 | -0.404616 | 0.536142 | 0.0658134 | -1137.49 | 2284.98 | 2302.1 |

Standard errors (same order):

| Distribution | (Intercept) | age | sex | ph.ecog |
|--------------|-------------|-----|-----|---------|
| weibull | 0.453578 | 0.00676351 | 0.123733 | 0.0834784 |
| exponential | 0.620755 | 0.00917689 | 0.167161 | 0.112697 |
| lognormal | 0.582756 | 0.00832786 | 0.152775 | 0.103308 |
| loglogistic | 0.512073 | 0.00747791 | 0.134894 | 0.0930137 |

Before ECON-47 and ECON-48 were fixed, no Newton step was ever taken (the Cholesky of the negative-definite Hessian
failed), 100 gradient steps diverged (Weibull age 54.48, scale inf), and the exponential was the Weibull fit
relabelled. Lung's age multiplied by 1e-6 and 1e6 gives survreg's rescaled coefficients in the same number of
iterations (within one); age2 = 2 age + sex is aliased (NA), as in survreg.

Tests: `test_validate_aft_weibull`, `test_validate_aft_exponential`, `test_validate_aft_lognormal`,
`test_validate_aft_loglogistic`, `test_validate_aft_converges_at_any_covariate_scale`,
`test_validate_aft_aliased_covariate`; properties: `test_aft_exponential_differs_from_weibull`,
`test_aft_iteration_limit_is_a_warning`, `test_aft_rejects_bad_iteration_settings`,
`test_aft_tail_helpers_match_r`, `test_aft_non_finite_result_is_an_error`.

Normal tails: ln S(z) and the hazard phi(z) / S(z) come from one evaluation of Cody's rational approximations, as
R's `pnorm_both` (within about 1e-13 relative of R 4.3.3 over z in [-38, 38]).

Wall clock (release build, one P core, `RAYON_NUM_THREADS=1`, three interleaved rounds at 811c26bb against
`fe13b889`, medians; criterion `AFT` group, 30% censoring, two covariates):

| Fit | n | new / base (wall clock) |
|---|---|---|
| Weibull | 100 / 500 / 1,000 / 2,000 | 0.16 / 0.13 / 0.15 / 0.12 |
| log-normal | 100 / 500 / 1,000 / 2,000 | 1.39 / 2.38 / 1.65 / 0.91 |

The log-normal rows are not like for like: on these data the `fe13b889` fit diverged (coefficients up to 1e95,
log-likelihood NaN) and still reported convergence, after 11 / 7 / 9 / 20 Newton steps; the current fit converges in
11 / 13 / 11 / 13. The log-normal cost of 1.39 / 2.38 / 1.65x at n = 100 / 500 / 1,000 is accepted, because the
current fit converges where the base diverged.

---

## Test Case 8: Competing Risks (`surv_competing_risks`)

### R Code
```r
# validation/r/cases/surv_cox_aft.R: two simulated datasets with causes 1 and 2 and censoring (0)
# surv_cr_cont.csv: 200 rows, continuous times (no ties); surv_cr_ties.csv: 40 rows, integer times with ties
survfit(Surv(time, factor(event_type, levels = 0:2)) ~ 1, data = d, conf.int = 0.95 | 0.5)
# multi-state Aalen-Johansen: f$pstate, f$std.err, f$lower, f$upper (conf.type "log")
```

`run_competing_risks` (MCP `competing_risks`, CLI `survival competing-risks`) estimates each cause's cumulative
incidence F_k(t) by Aalen-Johansen at every distinct time. Its standard error is survfit's multi-state
`std.err`, the delta-method variance: at the distinct times t_j (n_j at risk, d_j events of any cause, d_kj of
cause k, S_{j-1} the overall Kaplan-Meier before t_j, F_j = F_k(t_j)),

```text
Var F_k(t) = sum_{t_j <= t} [ a_j (F - F_j)^2 + b_j - 2 c_j (F - F_j) ],   F = F_k(t)
  a_j = d_j / (n_j (n_j - d_j))   (0 when n_j = d_j)
  b_j = S_{j-1}^2 d_kj (n_j - d_kj) / n_j^3
  c_j = S_{j-1} d_kj / n_j^2
```

computed from six running sums per cause (O(1) per time). The limits are log-scale, exp(log F -/+ z se / F),
capped at 1, and NaN while F = 0 (R's NA). Before ECON-78 was fixed the variance was sum S_{j-1}^2 d_kj / n_j^2,
which leaves out the overall-survival terms: the SEs were up to 42 % (continuous times) and 50 % (tied times)
larger than survfit's on these data.

**Selected values** (level 0.95; full vectors in `validation/expected/surv_competing_risks.json`):

| Data | t | F_1 | se_1 | 95% limits | F_2 | se_2 | 95% limits |
|------|---|-----|------|------------|-----|------|------------|
| cont | 1.38412 | 0.17 | 0.0265612 | (0.125157, 0.23091) | 0.085 | 0.0197199 | (0.0539437, 0.133936) |
| cont | 3.10479 | 0.302923 | 0.0327116 | (0.24514, 0.374327) | 0.161857 | 0.0262233 | (0.117821, 0.222351) |
| cont | 17.1796 | 0.551658 | 0.041043 | (0.476805, 0.638261) | 0.352108 | 0.0383854 | (0.284368, 0.435984) |
| ties | 5 | 0.275 | 0.0706001 | (0.166267, 0.454841) | 0.15 | 0.0564579 | (0.0717316, 0.313669) |
| ties | 9 | 0.380625 | 0.0776403 | (0.255193, 0.56771) | 0.281875 | 0.0722212 | (0.170594, 0.465746) |
| ties | 17 | 0.520125 | 0.0866932 | (0.375173, 0.721081) | 0.479875 | 0.0866932 | (0.336784, 0.683761) |

Tests: `test_validate_competing_risks_se_survfit` (times, CIF, SE and the limits at levels 0.95 and 0.5 at every
time of both datasets); property: `test_competing_risks_variance_running_sums_match_direct_sum` (the running sums
equal the direct double sum on the tied data).

---

## Column Types and Missing Values

Every survival entry point (Kaplan-Meier, log-rank, Cox PH, AFT, competing risks) reads its
columns with the shared extractor (`crates/p2a-core/src/data/columns.rs`):

- **Types.** Integer and float columns are accepted for time, status, covariates and competing-risk
  codes; a logical status column is read as 0/1. A logical or text time column is an error naming
  the column. The group column may be text, integer, float or logical; its levels follow R's
  `factor()` order (numbers by value, so `2` before `10`; text by byte order).
- **Missing values.** A row with a missing value (null, or NaN in a float column) in any column a
  call uses is dropped, as R's `na.omit` drops it in `survfit`, `survdiff`, `coxph` and `survreg`.
  The drop is reported: a `missing` field on the result (`n_total`, `n_used`, `n_dropped`,
  `by_column`), a last `Missing values: ...` line in the MCP text, and a `Note: ...` line and a
  `missing` key in the CLI. A missing group label is not a group of its own.
- **Status coding**, as `Surv()`:

  | Values in the status column | Event |
  |-----------------------------|-------|
  | all in {0, 1} | 1 |
  | all in {1, 2} | 2 (R's `lung`) |
  | TRUE/FALSE | TRUE |
  | anything else (0/1/2 mixed, 3, -1, 1.5, ...) | error naming the column and its values |

- **Times** must be >= 0 (AFT models: > 0, as `survreg` requires). Competing-risk codes are whole
  numbers 0 (censored), 1, 2, ... up to 255.

Tests: `test_validate_na_survival_int64_columns` (Int64 lung columns give bit-identical results
to their Float64 copies in all five estimators, R's n and events),
`test_validate_surv_status_coding_lung` (1/2, 0/1 and logical status give identical results),
`test_validate_surv_status_invalid_codes_are_errors`,
`test_validate_na_survival_missing_rows_are_dropped_and_reported` (lung by ph.ecog: 227 of 228
rows, R's n and observed events per group; Cox on age + sex + ph.karno: 227 rows as R `coxph`),
`test_validate_na_survival_bad_columns_name_the_column`.

---

## Implementation Notes

- **Kaplan-Meier.** Rows are sorted by time and grouped in blocks of exactly equal times. At each
  block S(t) is multiplied by 1 - d/n and Greenwood's sum d / (n (n - d)) is accumulated (infinite
  once d = n, as in R). The standard error of S(t) is S(t) times the square root of the sum
  (`summary(fit)$std.err`).
- **Confidence limits** (`survival:::survfit_confint`, se of log S): log
  `exp(log S -/+ z se)`, upper capped at 1, NA at S = 0; log-log
  `exp(-exp(log(-log S) -/+ z se / log S))`, NA at S = 0 and S = 1; plain `S -/+ z S se` clipped to
  [0, 1]. z is `qnorm(1 - (1 - conf_level) / 2)` from statrs.
- **Median and its limits** port `survival:::quantile.survfit` (`doquant`, `findq`, tolerance
  sqrt(machine epsilon)): the first time 1 - S reaches 0.5, the midpoint where S equals 0.5 over an
  interval, the midpoint with the last time when the curve ends at 0.5, and NA where a curve never
  reaches 0.5. The lower limit comes from the lower band and the upper limit from the upper band,
  whether or not the median exists.
- **Log-rank.** The same tie blocks; at each event block group g expects d n_g / n events and the
  hypergeometric variance d (n - d) / (n^2 (n - 1)) times n_g (n - n_g) (diagonal) or -n_g n_h is
  added; the block then leaves the risk sets once. Groups with no expected events are excluded,
  the chi-square is (O - E)' V^- (O - E) over the remaining groups less one (a generalized inverse
  when V is singular but not zero; a zero V is an error, as above), df is their number minus one
  and p the upper tail.

## Known Differences from R

1. **Invalid status codes are an error.** R's `Surv()` turns codes other than 0/1, 1/2 or logical
   into NA with a warning and drops those rows; p2a refuses the column.
2. **Negative times are an error.** R computes with them.
3. **Ties are exact equality.** R's `timefix = TRUE` merges times that differ by less than
   sqrt(machine epsilon); this matters only for times produced by arithmetic.
4. **A log-rank test with fewer than two groups with expected events is an error.** R reports a
   chi-square of 0 on 0 df.
5. **Cox aliasing** is detected by pivoted QR on [1 | X] (tolerance 1e-7, as `lm`), where coxph uses `cholesky2`
   on the information (`toler.chol`). Both drop the later of exactly collinear columns; near-singular designs may
   differ.
6. **Robust Cox tests with too few clusters.** With no more clusters than coefficients the robust Wald and robust
   score tests are NaN with a warning; coxph reports generalized-inverse statistics. A single-level cluster column
   is refused (coxph returns a variance of about 0).
7. **`cluster` implies robust SEs.** coxph ignores the cluster with a warning when `robust = FALSE` is given
   explicitly, which the core cannot see; the CLI and MCP surfaces decide.
8. **AFT start values** (least squares on ln t) differ from survreg's, so iteration counts can differ; the
   estimates agree to the tolerances below.
9. **Cox concordance** and its SE are not R-pinned (ECON-89: the SE formula and the pairing of tied times differ
   from `survival::concordance`).

The confidence interval type is `conf_type` "log" (default, as R), "log-log" or "plain"; before
ECON-7 was fixed p2a used an undisclosed, broken log-log band.

---

## Tolerance Summary

Tolerances follow the table in `validation/README.md`.

| Method | Statistic | Tolerance |
|--------|-----------|-----------|
| Kaplan-Meier | S(t) | `Tol::coef(n)`: 1e-6 (n < 100), 1e-8 (n = 100-1000) relative |
| Kaplan-Meier | SE, confidence limits | `Tol::se(n)` / `Tol::stat(n)`: 1e-5 (n < 100), 1e-6 (n = 100-1000) relative |
| Kaplan-Meier | times, n at risk, median and its limits | exact |
| Log-rank | chi-squared, expected, variance | `Tol::stat(n)` |
| Log-rank | p-value | `Tol::p_value(n)` |
| Log-rank | observed, n per group, df | exact |
| Cox, AFT (lung, n = 226-228) | coefficients | `Tol::coef(n)`: 1e-8 relative |
| Cox, AFT | SE, z, HR limits, tests, log-likelihood, scale, AIC, BIC | `Tol::se(n)` / `Tol::stat(n)`: 1e-6 relative |
| Cox, AFT | p-values | `Tol::p_value(n)`: 1e-6 relative + 1e-12 absolute |
| Cox tie fixture (n = 10) | coefficient; SE and statistics | 1e-6; 1e-5 relative |
| Cox at the default eps | coefficients against the tight fit | `Tol::iterative(1e-7)` |
| Cox, AFT | n, events, clusters, aliased columns | exact |
| Competing risks (n = 40; 200) | CIF | `Tol::coef(n)`: 1e-6; 1e-8 relative |
| Competing risks | SE, limits | `Tol::se(n)` / `Tol::stat(n)`: 1e-5; 1e-6 relative |
| Competing risks | times | exact |
| Survival columns | rows used, events, error types | exact |
| CLI and MCP text | printed values | print precision |

---

## References

1. Therneau, T.M. (2024). "A Package for Survival Analysis in R". R package version 3.5-8.
2. Kaplan, E.L. & Meier, P. (1958). "Nonparametric Estimation from Incomplete Observations".
   JASA, 53:457-481.
3. Mantel, N. (1966). "Evaluation of survival data and two new rank order statistics arising in
   its consideration". Cancer Chemotherapy Reports, 50:163-170.
4. Klein, J.P. & Moeschberger, M.L. (2003). *Survival Analysis*. Springer.
5. Cox, D.R. (1972). "Regression Models and Life-Tables". JRSS B, 34:187-220.
6. Efron, B. (1977). "The Efficiency of Cox's Likelihood Function for Censored Data". JASA, 72:557-565.
7. Lin, D.Y. & Wei, L.J. (1989). "The Robust Inference for the Cox Proportional Hazards Model". JASA,
   84:1074-1078.
8. Therneau, T.M. & Grambsch, P.M. (2000). *Modeling Survival Data: Extending the Cox Model*. Springer.
9. Kalbfleisch, J.D. & Prentice, R.L. (2002). *The Statistical Analysis of Failure Time Data*, 2nd ed. Wiley.
