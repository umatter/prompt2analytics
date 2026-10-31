# Validation: Survival Analysis (Kaplan-Meier, log-rank test, survival columns)

## Method Overview

This document covers what is pinned to R's `survival` package:
- **Kaplan-Meier**: `run_kaplan_meier`, `run_kaplan_meier_with` (MCP `kaplan_meier`, CLI
  `survival km`): the survival curve, Greenwood standard errors, pointwise confidence limits for
  `conf_type` "log" (default, as R), "log-log" and "plain", the median and its limits.
- **Log-rank test**: `log_rank_test` (MCP `log_rank`, CLI `survival log-rank`), as
  `survdiff(rho = 0)`.
- **Survival columns** of all five survival entry points (Kaplan-Meier, log-rank, Cox PH, AFT,
  competing risks): integer, float and logical columns, rows with missing values dropped and
  reported, status coded as `Surv()` codes it.

Cox proportional hazards, AFT models and competing risks are documented in
[survival_models.md](survival_models.md); their estimates are not R-pinned.

## Reference Implementation

| Package | Language | Functions | Version |
|---------|----------|-----------|---------|
| survival | R | `Surv()`, `survfit()`, `quantile.survfit()`, `survdiff()` | 3.5.8 on R 4.3.3 |

The R reference cases are in `validation/r/cases/survival.R`; `Rscript validation/r/run.R --check`
recomputes them and compares with the committed `validation/expected/<case>.json`. The registry
(`validation/methods/survival.json`) lists the tests that load them:

| Registry id | Cases | Tests |
|-------------|-------|-------|
| `surv_km` | `surv_km_lung`, `surv_km_lung_by_sex`, `surv_km_small`, `surv_km_median_rules` | `test_validate_km_lung`, `test_validate_km_lung_by_sex`, `test_validate_km_small`, `test_validate_km_median_rules`, `test_validate_km_conf_level_and_type_validation`; MCP `test_validate_mcp_kaplan_meier_lung`, `test_validate_mcp_kaplan_meier_rejects_unknown_conf_type`; CLI `test_validate_cli_survival_km_lung`, `test_validate_cli_survival_km_rejects_unknown_conf_type` |
| `surv_logrank` | `surv_logrank_lung`, `surv_logrank_small`, `surv_km_median_rules` | `test_validate_logrank_lung`, `test_validate_logrank_small`, `test_validate_logrank_zero_expected_group`; MCP `test_validate_mcp_log_rank_lung`; CLI `test_validate_cli_survival_log_rank_lung`, `test_validate_cli_survival_km_groups_missing_note` |
| `surv_data_handling` | `surv_km_lung`, `surv_logrank_lung`, `surv_logrank_small`, `surv_km_small` | `test_validate_na_survival_int64_columns`, `test_validate_surv_status_coding_lung`, `test_validate_surv_status_invalid_codes_are_errors`, `test_validate_na_survival_missing_rows_are_dropped_and_reported`, `test_validate_na_survival_bad_columns_name_the_column`; MCP `test_validate_mcp_survival_tools_accept_integer_columns`; CLI `test_validate_cli_survival_cox_aft_integer_columns` |

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

The Cox model on lung is in [survival_models.md](survival_models.md).

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
