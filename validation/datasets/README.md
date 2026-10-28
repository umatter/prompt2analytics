# Reference Datasets

This directory holds the data the validation tests and the paper's examples read. Committed
datasets are real data or synthetic files kept as canonical artifacts; `generated/` holds the
datasets that R reference cases write (`validation/r/cases/*.R`).

## Committed datasets

| File | Rows × columns | Used by | Source |
|---|---|---|---|
| `grunfeld.csv` | 200 × 5 (firm, year, inv, value, capital) | `ols_grunfeld`, `ols_diagnostics_grunfeld`, `panel_*_grunfeld`, `panel_hdfe_grunfeld`, `harness_selftest`, the CLI exhibit tests and the paper's CLI captures | `plm::Grunfeld` |
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
