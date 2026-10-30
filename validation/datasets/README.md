# Reference Datasets

This directory holds the data the validation tests and the paper's examples read. Committed
datasets are real data or synthetic files kept as canonical artifacts; `generated/` holds the
datasets that R reference cases write (`validation/r/cases/*.R`).

## Committed datasets

| File | Rows × columns | Used by | Source |
|---|---|---|---|
| `grunfeld.csv` | 200 × 5 (firm, year, inv, value, capital) | `ols_grunfeld`, `ols_diagnostics_grunfeld`, `panel_*_grunfeld`, `panel_hdfe_grunfeld`, `reg_cluster_na`, `harness_selftest`, the CLI exhibit tests and the paper's CLI captures | `plm::Grunfeld` |
| `employee_training.csv` | 1200 × 9 (employee_id, time, post_training, training_group, department, tenure, years_since_hire, performance, bonus) | the paper's chat transcripts (`paper/transcripts/section4_3_*.txt` in the paper repo); no validation case reads it | synthetic DiD panel, added in commit 4dd2c53; its generator is not in this repository |
| `gdp_quarterly.csv` | 280 × 4 (date, year, quarter, gdp_growth) | the paper's time-series chat example (`paper/transcripts/section4_4_*.txt` in the paper repo); no validation case reads it | synthetic (see below) |

### Grunfeld (1958)
**File**: `grunfeld.csv`

| Variable | Description |
|----------|-------------|
| firm | Firm identifier (1-10) |
| year | Year (1935-1954) |
| inv | Gross investment |
| value | Market value |
| capital | Stock of capital |

**Size**: 200 observations (10 firms × 20 years)

**Source**: Grunfeld, Y. (1958). "The Determinants of Corporate Investment". Unpublished Ph.D. dissertation, University of Chicago. Available in R's `plm` package.

**R Code**:
```r
library(plm)
data(Grunfeld)
write.csv(Grunfeld, "grunfeld.csv", row.names = FALSE)
```

### Employee training (synthetic)
**File**: `employee_training.csv`

A synthetic difference-in-differences panel: 200 employees observed in 6 periods (1200 rows)
before and after a training programme (`post_training`, `training_group`), with department
(5 levels), tenure and the outcomes `performance` and `bonus`. It was added with the recorded chat sessions of the paper
(commit 4dd2c53); the script that generated it was not committed, so the checked-in CSV is the
canonical artifact. It is not real personnel data.

### Quarterly GDP growth (synthetic)
**File**: `gdp_quarterly.csv`

| Variable | Description |
|----------|-------------|
| date | Quarter label, `YYYY-Qn` |
| year | Year (1954-2023) |
| quarter | Quarter within year (1-4) |
| gdp_growth | Quarterly growth rate (synthetic) |

**Size**: 280 observations (quarters labeled 1954-Q1 to 2023-Q4)

**Source**: **Synthetic.** This is *not* real national accounts data. The values
are a simulated stationary series (roughly mean 0.5, sd 0.8, no recession
structure) dressed with real-looking quarterly date labels; e.g. 2020-Q2 shows
no COVID collapse, so it cannot be matched to any actual GDP series. It exists
only to exercise the time series tools on a GDP-shaped, stationary series. The
original generation script/seed was not preserved; treat the checked-in CSV as
the canonical artifact. Do not cite it as empirical GDP data.

## Generated datasets (`generated/`)

Written only by `ctx$gen_dataset()` inside a case, with 15-17 significant digits, and verified
byte for byte by `Rscript validation/r/run.R --check`. Never edit them by hand; change the case
and run `Rscript validation/r/run.R --write <case>`.

| File | Rows × columns | Case(s) | R file | Source |
|---|---|---|---|---|
| `generated/longley.csv` | 16 × 7 (GNP.deflator, GNP, Unemployed, Armed.Forces, Population, Year, Employed) | `ols_longley` | `r/cases/regression.R` | `datasets::longley` |
| `generated/mtcars.csv` | 32 × 12 (model, mpg, cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb) | `ols_mtcars` | `r/cases/regression.R` | `datasets::mtcars`, row names as `model` |
| `generated/step_legacy.csv` | 100 × 4 (y, x1, x2, x3) | `step_legacy` | `r/cases/regression.R` | seed 42 (DGP of `scripts/validate_regression_diag.R`) |
| `generated/step_borderline.csv` | 120 × 4 (y, x1, x2, x3) | `step_borderline` | `r/cases/regression.R` | seed 5 |
| `generated/bgtest_ar1.csv` | 100 × 2 (y, x) | `diag_bgtest_ar1` | `r/cases/regression_legacy.R` | seed 42, DGP of `scripts/validate_regression_diag.R` |
| `generated/reset_quad.csv` | 100 × 2 (y, x) | `diag_reset_quad` | `r/cases/regression_legacy.R` | seed 42, DGP of `scripts/validate_regression_diag.R` |
| `generated/harvtest_nonlin.csv` | 100 × 2 (y, x) | `diag_harvtest_nonlin` | `r/cases/regression_legacy.R` | seed 42, DGP of `scripts/validate_regression_diag.R` |
| `generated/wald_sim.csv` | 100 × 4 (y, x1, x2, x3) | `diag_wald_sim` | `r/cases/regression_legacy.R` | seed 42, DGP of `scripts/validate_regression_diag.R` |
| `generated/hac_ar1.csv` | 100 × 2 (y, x) | `hac_ar1` | `r/cases/regression_legacy.R` | seed 42, DGP of `scripts/validate_regression_diag.R` |
| `generated/quantreg_hetero.csv` | 100 × 2 (y, x) | `quantreg_hetero` | `r/cases/regression_legacy.R` | seed 42, DGP of `scripts/validate_regression_diag.R` |
| `generated/dk_panel.csv` | 200 × 4 (y, x, firm, time; 10 firms × 20 periods) | `dk_panel` | `r/cases/regression_legacy.R` | seed 42, DGP of `scripts/validate_regression_diag.R` |
| `generated/gls_ar1.csv` | 50 × 3 (y, x, t) | `gls_ar1` | `r/cases/regression_legacy.R` | seed 42, DGP of `scripts/validate_regression_diag.R` |
| `generated/grunfeld_unbalanced.csv` | 171 × 5 (firm, year, inv, value, capital) | `panel_fe_grunfeld_unbalanced`, `panel_re_grunfeld_unbalanced`, `panel_hausman_grunfeld_unbalanced` | `r/cases/panel.R` | `grunfeld.csv` without the rows where (firm + year) %% 7 == 0 |
| `generated/panel_corr_effects.csv` | 300 × 5 (firm, year, inv, value, capital) | `panel_fe_sim_corr`, `panel_re_sim_corr`, `panel_hausman_sim_corr` | `r/cases/panel.R` | seed 20261021, entity effects correlated with `value` |
| `generated/panel_no_effect.csv` | 120 × 5 (firm, year, value, capital, inv) | `panel_re_sim_no_effect` | `r/cases/panel.R` | seed 3, no entity effect |
| `generated/multinom_mn8.csv` | 500 × 5 (y, x1, x2, x3, x4) | `disc_multinom_mn8` | `r/cases/discrete_newton.R` | seed 8, 8 categories |
| `generated/multinom_sep5.csv` | 30 × 2 (y, x) | `disc_multinom_sep5` | `r/cases/discrete_newton.R` | seed 12, 5 well-separated categories |
| `generated/multinom_j4.csv` | 400 × 3 (y, x1, x2) | `disc_multinom_j4` | `r/cases/discrete_newton.R` | seed 4, 4 overlapping categories |
| `generated/ordered_j5.csv` | 30 × 2 (y, x) | `disc_polr_j5` | `r/cases/discrete_newton.R` | seed 5, 5 categories |
| `generated/ordered_j16.csv` | 96 × 2 (y, x) | `disc_polr_j16` | `r/cases/discrete_newton.R` | seed 16, 16 categories |
| `generated/ordered_j30.csv` | 180 × 2 (y, x) | `disc_polr_j30` | `r/cases/discrete_newton.R` | seed 30, 30 categories |
| `generated/columns_conventions.csv` | 10 × 5 (id, y, n, s, flag) | `columns_conventions`; read by `cli_na_strings` | `r/cases/data_columns.R`; read by `r/cases/data_na_strings.R` | literal values (no RNG): empty fields for NA, integer `n`, logical `flag`, mixed-case text `s` |
| `generated/quality_na.csv` | 40 × 3 (id, x, z) | `data_quality_profile_na` | `r/cases/data_quality.R` | seed 12; x = round(rnorm(40, 10, 3), 4) with NA in rows 3, 17, 29; z = round(runif(37, 0, 5), 4) then three NaN (written as empty fields) |
| `generated/bin_cut.csv` | 12 × 2 (id, x) | `data_bin_cut` | `r/cases/data_quality.R` | literal values (no RNG): -5, 0, 0.5, 1, 2.5, 3, 4.9, 5, 7, NA, 2, NA |
| `generated/fisher_tables.csv` | 11 × 5 (table, a, b, c, d) | `stat_fisher_tables` | `r/cases/stats_contingency.R` | literal values: eleven 2 × 2 count tables (N = 8 to 4000) |
| `generated/fisher_rows_2000.csv` | 2000 × 2 (treat, outcome) | `stat_fisher_rows_2000`; read by the GOF and independence tests | `r/cases/stats_contingency.R` | seed 20261021; table [[520, 480], [480, 520]] |
| `generated/contingency_na_codes.csv` | 12 × 4 (g, h, code, out) | `stat_contingency_na_codes` | `r/cases/stats_contingency.R` | literal values: 3 missing `h`, 1 missing integer `code` (levels 2, 10) |
| `generated/chisq_gof.csv` | 52 × 2 (id, cat) | `stat_chisq_gof` | `r/cases/stats_contingency.R` | seed 7; cat = a ×5, b ×30, c ×15, 2 missing |
| `generated/chisq_small_expected.csv` | 13 × 2 (x, y) | `stat_chisq_small_expected` | `r/cases/stats_contingency.R` | literal values; expected counts below 5 |
| `generated/mantelhaen_na.csv` | 240 × 3 (treat, outcome, stratum) | `stat_mantelhaen_na` | `r/cases/stats_contingency.R` | seed 11; 3 missing strata |
| `generated/shapiro_cases.csv` | 5000 × 18 (id and 17 sample columns n3_124 ... n5000_rnorm, each padded with NA to 5000 rows; n3_124, n3_123, n25_int read as Int64) | `stat_shapiro_wilk` | `r/cases/stats_onesample.R` | seed 42; `qnorm(ppoints(100))` for `n100_qnorm` |
| `generated/two_sample_na.csv` | 22 × 8 (id, x, y, xi, yi, g, mi, mj; xi, yi, mi, mj Int64 with NA, g text with one NA) | `stat_two_sample_na` | `r/cases/stats_onesample.R` | fixed x, y; seed 3 (xi, yi), seed 17 (mi, mj) |
| `generated/pairwise_mahalanobis_na.csv` | 30 × 5 (id, y, g, a, b; y Int64 with NA in rows 4 and 18, a NA in row 7) | `stat_pairwise_mahalanobis_na` | `r/cases/stats_onesample.R` | seed 9 |
| `generated/rd_sim.csv` | 1000 × 6 (x, y, d, yf, xm, ym) | `rd_sharp_fixed_bw`, `rd_mass_points_nn`, `rd_fuzzy_fixed_bw` | `r/cases/causal_rd.R` | seed 42, cutoff 0, fuzzy take-up 0.2/0.8, x on a 0.01 grid in xm |
| `generated/iv_weak_controls.csv` | 500 × 5 (y, x, w, z1, z2) | `iv_weak_controls` | `r/cases/econ_iv.R` | seed 20261021 |
| `generated/iv_two_endog.csv` | 400 × 8 (y, x1, x2, w1, w2, z1, z2, z3) | `iv_two_endog` | `r/cases/econ_iv.R` | seed 77 |
| `generated/iv_no_controls.csv` | 300 × 4 (y, x, z1, z2) | `iv_no_controls` | `r/cases/econ_iv.R` | seed 5 |
| `generated/iv_na_rows.csv` | 300 × 8 (id, label, y, x, w, k, z1, z2); NA in y (2), x, w, k, z1, z2 (1 each); `k` integer, `label` text | `iv_na_rows` | `r/cases/econ_iv.R` | seed 4242 |
| `generated/grunfeld_na.csv` | 200 × 7 (firm, firm_id, firm_name, year, inv, value, capital); NA in inv (2), value (1), firm_id and firm_name (2, same rows), year (2) | `panel_grunfeld_na` | `r/cases/econ_panel_na.R` | `grunfeld.csv` with missing cells and text/numeric firm ids |
| `generated/arima_series.csv` | 300 × 10 (t, a, c, m, b, dr, ai, a_na, ai_na, lab) | `arima_css_forecast`, `arima_fixed_forecast`, `ts_granger_int` | `r/cases/forecasting_arima.R` | seed 20261021; a AR(1) around 100, c random walk, m ARMA(1,1) around 20, b AR(2), dr drift + MA(1) in differences; ai = round(10 a) (Int64); a_na, ai_na missing at rows 150-151; lab text |
| `generated/causal_na.csv` | 400 × 8 (id, y, d, m, x1, x2, d_na, d_txt) | `causal_mediation_na` | `r/cases/causal_mediation_na.R` | seed 20261021; empty fields for NA (15 in y, 25 in m, 10 in x1, 6 in d_na); integer d, d_na and id; text d_txt |
| `generated/cs_panel.csv` | 720 × 4 (unit, time, first_treat, y) | `did_cs_analytic` | `r/cases/causal_did.R` | seed 20261021, 90 units x 8 periods, cohorts 3/5/7/never, AR(1) errors |
| `generated/grunfeld_cluster_na.csv` | 200 × 7 (firm, year, firm_label, year_na, inv, value, capital) | `reg_cluster_na` | `r/cases/regression_na.R` | `grunfeld.csv` with `firm_label` (`firm_01` ... `firm_10`) missing in rows 3, 44, 101 and `year_na` (the year) missing in rows 10, 190 |
| `generated/smoothers_na.csv` | 60 × 3 (id, x, y) | `reg_smoothers_na` | `r/cases/regression_na.R` | seed 8; x ~ U(0, 10), y = sin(x) + 0.1 x + N(0, 0.3²), rounded to 4 decimals; x missing in rows 5, 33 and y in rows 12, 48 |
| `generated/ordered.csv` | 400 × 7 (y, y_label, x1, x2, y_wide, y_na, x1_na) | `disc_ordered` | `r/cases/discrete.R` | seed 2026, 3 ordered categories, 13-level y_wide, 7 incomplete rows |
| `generated/negbin.csv` | 400 × 5 (y, x1, x2, y_na, x1_na) | `disc_negbin` | `r/cases/discrete.R` | seed 404, NB2 with size 1.8, 5 incomplete rows |
| `generated/zeroinfl.csv` | 500 × 6 (y, y_nb, x1, x2, y_na, x2_na) | `disc_zeroinfl` | `r/cases/discrete.R` | seed 808, zero-inflated Poisson and NB, 5 incomplete rows |

## Adding a dataset

- Committed real data gets a file in this directory plus a row above with its source and the
  R code that regenerates it.
- Simulated data is never committed by hand: it goes through a seeded case
  (`set.seed()` + `ctx$gen_dataset()` in `validation/r/cases/<area>.R`).
- Format: UTF-8, comma-separated, a header row, no row names.

## Integrity

sha256 of the committed files (recompute with `sha256sum validation/datasets/*.csv`):

```
dfcbbe38315e570872483c1064eb1c6f229515cc792284aeb612e6d5356b4ffe  employee_training.csv
e57b6a8b32296f23ba711cafcab6ea3286e23f3e0bc3e4e79de0b072135e2213  gdp_quarterly.csv
00cd21d72068b5564a509d0752417f62761c35afb7e2abed9b115a430d5d960d  grunfeld.csv
```

Generated files are checked by `Rscript validation/r/run.R --check`, not by a checksum list.
