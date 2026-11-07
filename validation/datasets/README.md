# Reference Datasets

This directory holds the data the validation tests and the paper's examples read. Committed
datasets are real data or synthetic files kept as canonical artifacts; `generated/` holds the
datasets that R reference cases write (`validation/r/cases/*.R`).

## Committed datasets

| File | Rows × columns | Used by | Source |
|---|---|---|---|
| `grunfeld.csv` | 200 × 5 (firm, year, inv, value, capital) | `ols_grunfeld`, `ols_diagnostics_grunfeld`, `panel_*_grunfeld`, `panel_hdfe_grunfeld`, `panel_grunfeld_na`, `reg_cluster_na`, `dpd_ab_grunfeld`, `ols_no_intercept_grunfeld`, `ols_twoway_cluster_grunfeld`, `harness_selftest`, the CLI exhibit tests and the paper's CLI captures | `plm::Grunfeld` |
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
| `generated/ap_blobs.csv` | 115 × 2 (x, y) | `apcluster_blobs`; read by `pam_clara_blobs`, `fuzzy_reference_blobs` | `r/cases/ml_mixtures.R` | seed 20261109; the same blobs and noise as `blobs_noise.csv` |
| `generated/flexmix_lines.csv` | 200 × 3 (y, x, start) | `flexmix_minprior` | `r/cases/ml_mixtures.R` | seed 20261112; `start` is the fixed 3-cluster start |
| `generated/gmm_iv.csv` | 500 × 7 (y, x1, x2, w, z1, z2, z3) | `gmm_linear_iv` | `r/cases/econ_gmm.R` | seed 20261115 |
| `generated/kalman_intercepts.csv` | 50 × 2 (t, y) | `kalman_intercepts` | `r/cases/ts_statespace.R` | seed 20261124 |
| `generated/longley.csv` | 16 × 7 (GNP.deflator, GNP, Unemployed, Armed.Forces, Population, Year, Employed) | `ols_longley` | `r/cases/regression.R` | `datasets::longley` |
| `generated/mixture_3d.csv` | 300 × 3 (x1, x2, x3) | `gmm_mclust_3d`; read by `mixtools_em` | `r/cases/ml_mixtures.R` | seed 20261111 |
| `generated/mtcars.csv` | 32 × 12 (model, mpg, cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb) | `ols_mtcars` | `r/cases/regression.R` | `datasets::mtcars`, row names as `model` |
| `generated/mte_boot_indices.csv` | 400 × 25 (b01-b25; 1-based row indices into rows 1-400 of `mte_sim.csv`) | `mte_bootstrap` | `r/cases/econ_mte.R` | seed 20261207 |
| `generated/mte_sim.csv` | 2000 × 4 (y, d, z1, z2) | `mte_local_iv`, `mte_bootstrap` | `r/cases/econ_mte.R` | seed 20261205 |
| `generated/step_legacy.csv` | 100 × 4 (y, x1, x2, x3) | `step_legacy` | `r/cases/regression.R` | seed 42 (DGP of `scripts/validate_regression_diag.R`) |
| `generated/step_borderline.csv` | 120 × 4 (y, x1, x2, x3) | `step_borderline` | `r/cases/regression.R` | seed 5 |
| `generated/bgtest_ar1.csv` | 100 × 2 (y, x) | `diag_bgtest_ar1` | `r/cases/regression_legacy.R` | seed 42, DGP of `scripts/validate_regression_diag.R` |
| `generated/reset_quad.csv` | 100 × 2 (y, x) | `diag_reset_quad` | `r/cases/regression_legacy.R` | seed 42, DGP of `scripts/validate_regression_diag.R` |
| `generated/harvtest_nonlin.csv` | 100 × 2 (y, x) | `diag_harvtest_nonlin` | `r/cases/regression_legacy.R` | seed 42, DGP of `scripts/validate_regression_diag.R` |
| `generated/structts_series.csv` | 108 × 4 (t, nile, ukgas, nile_na) | `structts_nile_ukgas` | `r/cases/ts_statespace.R` | `datasets::Nile`, `log10(datasets::UKgas)`; `nile_na` missing in rows 21-25 and 61 |
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
| `generated/two_sample_na.csv` | 22 × 8 (id, x, y, xi, yi, g, mi, mj; xi, yi, mi, mj Int64 with NA, g text with one NA) | `stat_two_sample_na`; read by `cli_stat_descriptive_na` | `r/cases/stats_onesample.R`; read by `r/cases/cli_stats.R` | fixed x, y; seed 3 (xi, yi), seed 17 (mi, mj) |
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
| `generated/cs_cov_panel.csv` | 1200 × 8 (unit, time, first_treat, x1, x2, y, x3, D) | `did_cs_covariates`, `bacon_twfe` | `r/cases/causal_did_cov.R` | seed 20261129, 150 units x 8 periods, cohorts 3/5/7/never drawn on x1, x2; one period of anticipation; x3 time-varying and shifted by treatment; D the absorbing treatment indicator |
| `generated/grunfeld_cluster_na.csv` | 200 × 7 (firm, year, firm_label, year_na, inv, value, capital) | `reg_cluster_na` | `r/cases/regression_na.R` | `grunfeld.csv` with `firm_label` (`firm_01` ... `firm_10`) missing in rows 3, 44, 101 and `year_na` (the year) missing in rows 10, 190 |
| `generated/smoothers_na.csv` | 60 × 3 (id, x, y) | `reg_smoothers_na` | `r/cases/regression_na.R` | seed 8; x ~ U(0, 10), y = sin(x) + 0.1 x + N(0, 0.3²), rounded to 4 decimals; x missing in rows 5, 33 and y in rows 12, 48 |
| `generated/ordered.csv` | 400 × 7 (y, y_label, x1, x2, y_wide, y_na, x1_na) | `disc_ordered` | `r/cases/discrete.R` | seed 2026, 3 ordered categories, 13-level y_wide, 7 incomplete rows |
| `generated/negbin.csv` | 400 × 5 (y, x1, x2, y_na, x1_na) | `disc_negbin` | `r/cases/discrete.R` | seed 404, NB2 with size 1.8, 5 incomplete rows |
| `generated/zeroinfl.csv` | 500 × 6 (y, y_nb, x1, x2, y_na, x2_na) | `disc_zeroinfl` | `r/cases/discrete.R` | seed 808, zero-inflated Poisson and NB, 5 incomplete rows |
| `generated/anova_two_way_unbalanced.csv` | 13 × 6 (y, fa, fb, y3, fa3, fb3) | `stat_anova_two_way_unbalanced` | `r/cases/stats_anova.R` | seed 101; 2 x 2 cells of 2/4/4/2 rows plus one row with y and fa missing; 3 x 2 layout y3 ~ fa3 * fb3 with the (C, L) cell empty |
| `generated/oneway_na_labels.csv` | 9 × 5 (id, y, g, v9, g9) | `stat_oneway_na_labels` | `r/cases/stats_anova.R` | fixed values; y Int64 missing in 3 rows, g in 2, g9 in 3 |
| `generated/wilcoxon_na.csv` | 10 × 3 (id, x, y) | `stat_wilcoxon_na` | `r/cases/stats_anova.R` | seed 17; x missing in row 3, y in row 8; no ties |
| `generated/friedman_quade.csv` | 24 × 4 (y, y_na, treat, block) | `stat_friedman_quade`; read by `cli_stat_acf_int` | `r/cases/stats_anova.R`; read by `r/cases/cli_stats.R` | seed 23; 8 blocks × 3 treatments, Int64 y, y_na missing in row 5 (block S2) |
| `generated/manova_na.csv` | 36 × 4 (id, y1, y2, g) | `stat_manova_na` | `r/cases/stats_anova.R` | seed 31; 3 groups of 12; y1 missing in rows 3 and 20, y2 in row 30 |
| `generated/lung.csv` | 228 × 8 (inst, time, status, age, sex, ph.ecog, ph.karno, wt.loss) | `surv_km_lung`, `surv_km_lung_by_sex`, `surv_logrank_lung`, `surv_cox_robust`, `surv_aft` | `r/cases/survival.R` | `survival::lung`: integer days, status 1 = censored, 2 = dead; empty cells inst 1, ph.ecog 1, ph.karno 1, wt.loss 14 |
| `generated/logrank_small.csv` | 12 × 3 (time, event, group) | `surv_logrank_small` | `r/cases/survival.R` | fixed values: the 12-row data of the former log-rank test |
| `generated/km_small.csv` | 10 × 2 (time, event) | `surv_km_small` | `r/cases/survival.R` | fixed values: the 10-row data of the former KM test |
| `generated/cox_ties_small.csv` | 10 × 3 (time, event, x) | `surv_cox_ties_small` | `r/cases/surv_cox_aft.R` | fixed values: 5 distinct times, censored rows tied with deaths (the data of the former Cox ties test) |
| `generated/surv_cr_cont.csv` | 200 × 3 (id, time, event_type) | `surv_competing_risks` | `r/cases/surv_cox_aft.R` | simulated (seed 20261101): exponential causes 1 (rate 0.10) and 2 (0.06), uniform(2, 18) censoring; times rounded to 6 decimals, no ties |
| `generated/surv_cr_ties.csv` | 40 × 3 (id, time, event_type) | `surv_competing_risks` | `r/cases/surv_cox_aft.R` | simulated (same seed, after `surv_cr_cont`): causes 1 (rate 0.08) and 2 (0.05), uniform(5, 30) censoring; integer times (ceiling) with ties |
| `generated/km_median_rules.csv` | 16 × 6 (time, event, group, time_early, event_early, group_no_mid) | `surv_km_median_rules` | `r/cases/survival.R` | fixed values; group_no_mid has 7 empty cells |
| `generated/arima_misspecified_seasonal.csv` | 1000 × 1 (y) | `arima_misspecified_seasonal` | `r/cases/forecasting_arima.R` | seed 42; y = 0.01 t + 2 sin(pi t / 6) + runif(1000, 0, 0.5), t = 0..999 (the unified benchmark's time-series DGP); the n = 100 fit reads the first 100 rows |
| `generated/mixed_logit.csv` | 900 × 6 (person, id, alt, choice, x, w) | `disc_mixed_logit` | `r/cases/discrete_mixed.R` | seed 1234, 300 choice situations × 3 alternatives, normal coefficient on x |
| `generated/mixed_logit_panel.csv` | 1200 × 6 (person, id, alt, choice, x, w) | `disc_mixed_logit_panel` | `r/cases/discrete_mixed.R` | seed 1234, 100 people × 4 situations × 3 alternatives |
| `generated/choice_na.csv` | 300 × 12 (row, id, alt, choice, x, w, x_int, g, k, choice_na, x_na, id_na) | `disc_choice_na` | `r/cases/discrete_mixed.R` | seed 4321, 100 situations × 3 alternatives, 6 incomplete rows (5, 17, 23, 44, 101, 240) |
| `generated/ab_balanced.csv` | 800 × 4 (id, year, y, x) | `dpd_ab_balanced` | `r/cases/econ_dynamic_panel.R` | seed 606; N = 100, T = 8 (2000-2007), y_t = 0.5 y_t-1 + 0.3 x_t + a_i + e_t, rounded to 10 decimals |
| `generated/ab_gaps.csv` | 775 × 4 (id, year, y, x) | `dpd_ab_gaps` | `r/cases/econ_dynamic_panel.R` | `ab_balanced` without 2004 for ids divisible by 9 and without 2000-2001 for ids divisible by 13 |
| `generated/ab_common_gap.csv` | 700 × 5 (id, unit, year, y, x); NA in y (10) and x (10); text `unit` (u001..u100) | `dpd_ab_common_gap`, `panel_pur_extraction` | `r/cases/econ_dynamic_panel.R` | seed 606, same DGP without 2003 for every unit |
| `generated/loglin_na.csv` | 16 × 5 (id, f1, f2, f3, cnt; f2 and cnt Int64; f1 missing in row 14, f2 in row 15, cnt in row 16; rows 11 and 13 are the same cell (b, 10, y)) | `stat_loglin_na` | `r/cases/stats_mcp.R` | seed 211, a 3 × 2 × 2 table in long form |
| `generated/weighted_na.csv` | 12 × 7 (id, v, w, a, b, wc, b_na; a, b and b_na Int64; v missing in row 3, w in row 8, b_na in row 5) | `stat_weighted_na` | `r/cases/stats_mcp.R` | seed 307 |
| `generated/mauchly_na.csv` | 16 × 5 (id, t1-t4, all Int64; t2 missing in row 5, t4 in row 11) | `stat_mauchly_na` | `r/cases/stats_mcp.R` | seed 419, variances growing across conditions |
| `generated/descriptive_na.csv` | 15 × 6 (id, v, vi, s, xs, ys; vi Int64, s text; v missing in rows 2, 9 and 13, vi in rows 4 and 11, xs in row 5, ys in row 10) | `stat_descriptive_na` | `r/cases/stats_mcp.R` | seed 613 |
| `generated/isoreg_medpolish_int.csv` | 10 × 8 (id, x, y, y_na, c1-c4; y, y_na and c1-c4 Int64; y_na missing in row 6, c3 in row 4; x has no ties) | `stat_isoreg_medpolish_int` | `r/cases/stats_mcp.R` | seed 523 |
| `generated/prop_trend_na.csv` | 7 × 6 (group, dose, successes, trials, trials_neg, successes_frac; all Int64 but successes_frac; successes missing in row 4; trials_neg is -1 in row 5; successes_frac is 5.5 in row 2) | `stat_prop_trend_na` | `r/cases/stats_mcp.R` | fixed values |
| `generated/na_reviewer.csv` | 8 × 7 (id, x, y, g, i, time, event; g, i, time and event Int64; x missing in row 3, y in row 2) | `na_x2_x8_reviewer` | `r/cases/na_acceptance.R` | fixed values (the X-2 reviewer data and integer survival columns) |
| `generated/ts_flags_sim.csv` | 120 × 3 (t, y_ar, y_seas) | `ts_ar_yw_aic` | `r/cases/cli_flags.R` | seed 20261022: AR(2) (0.6, −0.3) plus 5; trend + monthly season with outliers at t = 6 and 75 |
| `generated/linalg_glm.csv` | 120 × 12 (x1, x2, x3, x4, off, yb, yp, wq, yq, ysep, yc, w0) | `linalg_qr_glm` | `r/cases/linalg_glm.R` | seed 20261210; x3 = x1 - 2 x2 exactly (collinear); yb binomial (logit), yp Poisson with offset off, yq fractional with weights wq, ysep separated by x4, yc gaussian, w0 weights with 5 zeros |
| `generated/quantile_sample.csv` | 28 × 3 (i, x, y; y missing in rows 8-28) | `stat_quantile_types` | `r/cases/stats_quantile.R` | seed 20261106; x: 23 normal draws (mean 10, sd 3, 3 decimals) plus ties 7.5 × 2 and 12 × 3; y: 7 fixed values |
| `generated/glmnet_sim.csv` | 150 × 43 (y, yb, X1-X8, foldid, foldid7, K1-K30, yk) | `glmnet_paths` | `r/cases/reg_glmnet.R` | seed 20261122: X normal with X2 correlated with X1 and X5 scaled, y gaussian, yb binomial, foldid 5 random folds, foldid7 7 deterministic folds; seed 20261130: K1-K30 AR(0.85) design, yk with 8 nonzero coefficients (the KKT check) |
| `generated/anova_oneway_legacy.csv` | 15 × 2 (yield, fertilizer) | `stat_anova_oneway_legacy` | `r/cases/stats_power_loglin.R` | literal values: the one-way fertilizer data of `anova.rs`'s legacy test |
| `generated/anova_twoway_legacy.csv` | 12 × 3 (yield, fertilizer, water) | `stat_anova_twoway_legacy` | `r/cases/stats_power_loglin.R` | literal values: the two-way fertilizer × water data of `anova.rs`'s legacy test |
| `generated/binary_glm.csv` | 400 × 10 (y, x1, x2, d, x1b, nd, xnc, ysep, yq, y012) | `disc_binary_glm` | `r/cases/discrete_mlogit_asc.R` | seed 20261021 |
| `generated/blobs_noise.csv` | 115 × 2 (x, y) | `hdbscan_blobs`; read by `optics_blobs`, `silhouette_singletons` | `r/cases/ml_density_clustering.R` | seed 20261109 |
| `generated/cbps_sim.csv` | 400 × 5 (treat, x1, x2, x3, y; treat and x3 Int64) | `cbps_sim_methods` | `r/cases/causal_cbps.R` | seed 20261102; 178 treated, 222 controls; logit propensity in x1, x2, x3; y = 1 + 2 treat + x1 + 0.5 x2 + N(0, 1) |
| `generated/cf_aipw_formula.csv` | 40 × 5 (y, w, y_hat, w_hat, tau_hat) | `cf_aipw_formula` | `r/cases/ml_causal_forest.R` | seed 4704 |
| `generated/cf_confounded.csv` | 1000 × 7 (x0, x1, x2, w, y0, y1, yh) | `cf_confounded` | `r/cases/ml_causal_forest.R` | seed 4794; P(W = 1) = 0.1 + 0.8 x0; tau = 0 (y0), 1 (y1), 1 + 2 x1 (yh) |
| `generated/cf_little_bags.csv` | 12 × 6 (group, n, y, w, yw, ww) | `cf_little_bags` | `r/cases/ml_causal_forest.R` | seed 4705; leaf means of 12 trees in 6 groups of 2 |
| `generated/cf_rare.csv` | 200 × 4 (x0, x1, w, y) | `cf_rare` | `r/cases/ml_causal_forest.R` | seed 4703; 10 treated units, tau = 5 |
| `generated/cf_rct.csv` | 2000 × 4 (x0, x1, w, y) | `cf_rct` | `r/cases/ml_causal_forest.R` | seed 4702; randomised W, tau = 0.2 |
| `generated/cf_split_node.csv` | 30 × 5 (x, x2, y, z, r) | `cf_split_node` | `r/cases/ml_causal_forest.R` | seed 4710; x2 has ties |
| `generated/ctmle_sim.csv` | 300 × 7 (yb, yc, a, w1, w2, w3, w4) | `tmle_glm`, `ctmle_cv` | `r/cases/causal_stdreg.R` | seed 20261210; w1 confounder, w2 instrument, w3 precision variable, w4 noise |
| `generated/doubleml_sim.csv` | 400 × 17 (X1-X6, dcont, y, dbin, yb, z, dz, yz, zb, db, yzb, fold) | `doubleml_models` | `r/cases/causal_doubleml.R` | seed 20261201; continuous (PLR), binary (IRM) and instrumented (PLIV, IIVM) treatments with their outcomes; `fold` the fixed 5-fold labels |
| `generated/etwfe_panel.csv` | 960 × 6 (id, year, first_treat, xc, y, treat) | `etwfe_staggered` | `r/cases/causal_etwfe.R` | seed 20261103, 120 units x 8 periods, cohorts 4/6/8/never (0), time-invariant control xc, treat the absorbing treatment indicator |
| `generated/etwfe_panel_all_treated.csv` | 960 × 6 (id, year, first_treat, xc, y, treat) | `etwfe_unbalanced` | `r/cases/causal_etwfe.R` | seed 20261103, 120 units x 8 periods, cohorts 4/6/8, no never-treated unit |
| `generated/etwfe_panel_unbalanced.csv` | 923 × 6 (id, year, first_treat, xc, y, treat) | `etwfe_unbalanced` | `r/cases/causal_etwfe.R` | seed 20261103, `etwfe_panel` without year 3 of every 7th unit and years 1-2 of every 11th |
| `generated/eurodist_matrix.csv` | 21 × 21 (Athens ... Vienna) | `kmeans_lloyd_pca_mds` | `r/cases/ml_hierarchical.R` | `datasets::eurodist` as a 21 × 21 distance matrix (city names as columns) |
| `generated/evalue_inputs.csv` | 8 × 7 (id, type, est, lo, hi, rare, se) | `evalue_formulas` | `r/cases/regression_smoothers_t3.R` | literal values: eight RR, OR, HR and MD estimates with limits, rare-outcome flags and one SE |
| `generated/factanal_promax.csv` | 500 × 12 (v1-v12) | `stat_factanal_promax` | `r/cases/stats_power_loglin.R` | seed 20261118 |
| `generated/feglm_panel.csv` | 480 × 7 (id, time, x, x2, y_gauss, y_bin, y_cnt) | `feglm_families` | `r/cases/econ_panel_t3.R` | seed 20261106 |
| `generated/fisher_legacy.csv` | 4 × 5 (table, a, b, c, d) | `stat_fisher_legacy` | `r/cases/stats_power_loglin.R` | literal values: the four 2 × 2 tables of `fisher.rs`'s legacy tests |
| `generated/garch_returns.csv` | 1500 × 5 (t, r_norm, r_t, r_g21, r_norm_dm) | `garch_normal_t` | `r/cases/ts_garch.R` | seed 20261105 |
| `generated/gformula_nc.csv` | 2000 × 5 (B, L_t0, A_t0, Y, Yc) | `gformula_natural_course` | `r/cases/causal_gformula.R` | seed 20261209; one period; binary Y and continuous Yc |
| `generated/gformula_surv.csv` | 400 × 6 (id, L, Lb, A, time, event) | `gformula_hazard` | `r/cases/causal_gformula.R` | seed 20261207; 5 periods, time-fixed L and A; 175 events |
| `generated/gformula_surv_tv.csv` | 1000 × 10 (id, B, L_t0-L_t2, A_t0-A_t2, time, event) | `gformula_hazard_tv` | `r/cases/causal_gformula.R` | seed 20261208; 3 periods, time-varying L and A, loss to follow-up after period 2; cells after a subject's last period are empty (NA) |
| `generated/gls_cs.csv` | 150 × 4 (g, y, x, z) | `gls_compsymm` | `r/cases/regression_smoothers_t3.R` | seed 20261120 |
| `generated/granger_short.csv` | 20 × 2 (x, y) | `granger_short` | `r/cases/ts_var.R` | seed 20261113 |
| `generated/gsynth_simdata.csv` | 1500 × 6 (id, time, Y, D, X1, X2) | `gsynth_simdata` | `r/cases/causal_gsynth.R` | `gsynth::simdata` (50 units × 30 periods, 5 treated from t = 21), the model columns |
| `generated/gsynth_staggered.csv` | 720 × 5 (id, time, Y, D, X) | `gsynth_staggered` | `r/cases/causal_gsynth.R` | seed 4231; 30 controls and 6 treated units (adoption at t = 11, 13, 15) × 20 periods, two factors |
| `generated/hclust_60.csv` | 60 × 3 (a, b, c) | `hclust_methods`; read by `kmeans_lloyd_pca_mds`, `dynamic_tree_cut` | `r/cases/ml_hierarchical.R` | seed 20261110 |
| `generated/hclust_ties.csv` | 40 × 2 (u, v) | `hclust_methods` | `r/cases/ml_hierarchical.R` | seed 7 |
| `generated/hdfe_disconnected.csv` | 80 × 5 (firm, year, x, z, y) | `hdfe_disconnected` | `r/cases/econ_panel_t3.R` | seed 20261105 |
| `generated/hdfe_scale.csv` | 1000 × 5 (firm, year, x, y, y_big) | `hdfe_map_scale` | `r/cases/econ_panel_t3.R` | seed 20261103 |
| `generated/line_outliers.csv` | 30 × 2 (x, y) | `line_tukey_outliers` | `r/cases/regression_smoothers_t3.R` | literal values: x = 1:30, the old line test's y with outliers at x = 5, 15 and 25 |
| `generated/loess_skewed.csv` | 100 × 2 (x, y) | `loess_skewed_direct` | `r/cases/regression_smoothers_t3.R` | seed 20261128 |
| `generated/loess_x10.csv` | 10 × 2 (x, y) | `loess_x10_direct` | `r/cases/regression_smoothers_t3.R` | literal values: x = 1:10, near-linear y |
| `generated/loglin_2x3x4.csv` | 24 × 4 (Var1, Var2, Var3, Freq) | `stat_loglin_nodecomp` | `r/cases/stats_power_loglin.R` | seed 20261116 |
| `generated/ltmle_sim.csv` | 600 × 13 (L0, A1, L1, A2, Y, Yc, R1, Lr, R2, Yr, L2, A3, Y3) | `ltmle_two_period` | `r/cases/causal_ltmle.R` | seed 20261206; integer treatments and binary outcomes |
| `generated/matching_sim.csv` | 240 × 6 (id, treat, x1, x2, x3, y) | `matchit_nearest`; read by `matchit_full_cem`, `matchit_subclass` | `r/cases/causal_matching.R` | seed 20261128 |
| `generated/mauchly_sigma.csv` | 15 × 4 (V1, V2, V3, V4) | `stat_mauchly_sigma` | `r/cases/stats_power_loglin.R` | seed 20261117 |
| `generated/mcf_networks.csv` | 580 × 6 (network, kind, a, b, cap, cost) | `mcf_networks` | `r/cases/causal_matching.R` | seed 20261023 |
| `generated/medflex_sim.csv` | 500 × 12 (a, m, c1, c2, yc, yb, yp, yi, yo, m_lin, y_lin, c1dup; a, c2, yb and yp Int64) | `medflex_scales`; read by `mediation_ipw_t3` | `r/cases/causal_medflex.R` | seed 20261130; binary a, gaussian m, yc (gaussian), yb (binary), yp (Poisson), yi (A:M interaction), yo (NDE and NIE of opposite sign), m_lin = 0.1 + 0.3 c1 exactly, c1dup a copy of c1 |
| `generated/mlogit_fishing.csv` | 4728 × 6 (id, alt, chosen, price, catch, income) | `disc_mlogit_fishing` | `r/cases/discrete_mlogit_asc.R` | `mlogit::Fishing` in long form (`dfidx`), 1182 choice situations × 4 modes |
| `generated/multinom_overlap.csv` | 300 × 3 (y, x1, x2) | `disc_multinom_overlap` | `r/cases/discrete_mlogit_asc.R` | seed 20261022 |
| `generated/nls_expdecay.csv` | 11 × 2 (x, y) | `nls_expdecay` | `r/cases/regression_smoothers_t3.R` | literal values: the old nls test's exponential-decay data, x = 0, 0.5, ..., 5 |
| `generated/nls_mm.csv` | 8 × 2 (S, V) | `nls_mm` | `r/cases/regression_smoothers_t3.R` | literal values: the old Michaelis-Menten test data |
| `generated/ols_aliased.csv` | 120 × 9 (y, x1, x2, x3, da, db, dc, k5, cl) | `ols_aliased` | `r/cases/reg_rank_qr.R` | seed 20261118 |
| `generated/ols_illcond.csv` | 80 × 4 (y, x1, x2, x3) | `ols_illcond` | `r/cases/reg_rank_qr.R` | seed 20261119 |
| `generated/ols_year_trend.csv` | 5000 × 3 (y, year, z) | `ols_year_trend` | `r/cases/reg_rank_qr.R` | seed 20261104 |
| `generated/p_adjust_vectors.csv` | 47 × 2 (vector, p) | `stat_p_adjust` | `r/cases/stats_power_loglin.R` | literal values: named p-value vectors (legacy, Hommel, ties, NA, single) |
| `generated/pairwise_legacy.csv` | 36 × 3 (set, x, g) | `stat_pairwise_legacy` | `r/cases/stats_power_loglin.R` | literal values: the three sets of `pairwise.rs`'s legacy tests |
| `generated/panel_absorbed.csv` | 180 × 8 (id, t, x1, x2, z, w, x3, y) | `panel_absorbed_regressors` | `r/cases/econ_panel_t3.R` | seed 20261104 |
| `generated/pggls_panel.csv` | 200 × 5 (id, t, x1, x2, y) | `pggls_sim` | `r/cases/econ_panel_t3.R` | seed 20261101 |
| `generated/purtest_panel.csv` | 480 × 3 (id, t, y) | `purtest_lag_selection` | `r/cases/econ_purtest.R` | seed 20261204 |
| `generated/pvcm_panel.csv` | 240 × 5 (id, t, x1, x2, y) | `pvcm_pmg_sim` | `r/cases/econ_panel_t3.R` | seed 20261102 |
| `generated/rank_binom_grid.csv` | 23 × 5 (label, x, n, p, stat15) | `stat_rank_binom` | `r/cases/stats_rank_t3.R` | literal values: 23 (x, n, p) cells; `stat15` marks the cells of STAT-15 |
| `generated/rank_fligner.csv` | 102 × 3 (set, group, value) | `stat_rank_fligner` | `r/cases/stats_rank_t3.R` | literal values: the two examples the old tests quoted, and `datasets::InsectSprays` |
| `generated/rank_friedman_rounding.csv` | 66 × 4 (block, method, time, first12) | `stat_rank_friedman_rounding` | `r/cases/stats_rank_t3.R` | literal values: the `?friedman.test` RoundingTimes data (22 players × 3 methods) in long form; `first12` marks the first 12 players |
| `generated/rank_isoreg.csv` | 58 × 3 (set, x, y) | `stat_rank_isoreg` | `r/cases/stats_rank_t3.R` | seed 20261026 |
| `generated/rank_kruskal_hw.csv` | 14 × 2 (value, group) | `stat_rank_kruskal_hw` | `r/cases/stats_rank_t3.R` | literal values: Hollander & Wolfe (1973) p. 116, the `?kruskal.test` example |
| `generated/rank_quade_store.csv` | 35 × 3 (store, brand, sales) | `stat_rank_quade_store` | `r/cases/stats_rank_t3.R` | literal values: Conover (1999) p. 375, the `?quade.test` hand-lotion example |
| `generated/rank_small_blocks.csv` | 32 × 4 (block, treat, y, y_unit) | `stat_rank_small_blocks` | `r/cases/stats_rank_t3.R` | seed 20261023 |
| `generated/rank_small_groups.csv` | 21 × 4 (g, y, y_unit, ynt) | `stat_rank_small_groups` | `r/cases/stats_rank_t3.R` | seed 20261022 |
| `generated/rank_two_sample.csv` | 126 × 3 (set, group, value) | `stat_rank_two_sample` | `r/cases/stats_rank_t3.R` | seed 20261024 |
| `generated/rank_wilcoxon.csv` | 101 × 3 (set, group, value) | `stat_rank_wilcoxon` | `r/cases/stats_rank_t3.R` | seed 20261025 |
| `generated/reset_two.csv` | 150 × 3 (y, x1, x2) | `reset_princomp` | `r/cases/reg_rank_qr.R` | seed 20261121 |
| `generated/rq_boot_indices.csv` | 100 × 50 (b01-b50) | `rq_boot_xy` | `r/cases/reg_quantreg_br.R` | seed 20261021 |
| `generated/sensemakr_darfur.csv` | 1276 × 8 (peacefactor, directlyharmed, age, farmer_dar, herder_dar, pastvoted, hhsize_darfur, female) | `sensemakr_darfur` | `r/cases/regression_smoothers_t3.R` | `sensemakr::darfur`, the eight model columns |
| `generated/shap_kernel_sim.csv` | 20 × 12 (x0-x11) | `shap_kernel_sim` | `r/cases/ml_shap.R` | seed 4802 |
| `generated/shap_tree_sim.csv` | 80 × 5 (x0, x1, x2, x3, y) | `shap_tree_rpart` | `r/cases/ml_shap.R` | seed 4801 |
| `generated/smoothers_t3.csv` | 100 × 4 (x, y, xm, ym) | `smoothers_gcv_periodic_loess` | `r/cases/regression_smoothers_t3.R` | seed 20261127 |
| `generated/stdreg_sim.csv` | 300 × 8 (yc, yb, yp, ys, a, x1, x2, x1dup) | `stdreg_glm`, `stdreg_linear`, `stdreg_separation` | `r/cases/causal_stdreg.R` | seed 20261209; yp Poisson with mean about 220; ys = 1{x1 > 0.3} (complete separation); x1dup = 2 x1 + 1 (aliased) |
| `generated/supsmu_sine.csv` | 100 × 4 (x, y, w, xr) | `supsmu_sine` | `r/cases/regression_smoothers_t3.R` | seed 42 |
| `generated/synth_qp_panel.csv` | 180 × 9 (unit, time, y, x1, x2, x3, x1_big, x2_big, x3_big) | `synth_weights_qp`; read by `scpi_quantities` | `r/cases/causal_synth_scpi.R` | seed 3101; 8 donors and the treated unit TR × 20 periods (treatment at 15); `*_big` = predictors × 1e4 |
| `generated/sysgmm_empluk.csv` | 1031 × 5 (firm, year, lemp, lwage, lcap) | `dpd_sys_empluk` | `r/cases/econ_sysgmm.R` | `plm::EmplUK`, logs of emp, wage and capital |
| `generated/var_canada.csv` | 84 × 4 (e, prod, rw, U) | `var_canada` | `r/cases/ts_var.R` | `vars::Canada` |
| `generated/varma_sim.csv` | 400 × 2 (y1, y2) | `varma_hr` | `r/cases/ts_varma.R` | seed 20261114 |
| `generated/vecm_denmark.csv` | 55 × 4 (LRM, LRY, IBO, IDE) | `vecm_denmark` | `r/cases/ts_vecm.R` | `urca::denmark`, the four model columns |
| `generated/vecm_sim3.csv` | 250 × 3 (x1, x2, x3) | `vecm_sim3` | `r/cases/ts_vecm.R` | seed 20261101 |
| `generated/weighting_sim.csv` | 300 × 5 (treat, x1, x2, x3, y; treat and x3 Int64) | `weights_ebal`; read by `weights_sbw`, `weights_energy`, `ipw_aipw_sim` | `r/cases/causal_weighting.R`; read by `r/cases/causal_ipw_aipw.R` | seed 20261104; 151 treated, 149 controls; logit propensity in x1, x2, x3; y = 2 + 1.5 treat + x1 + x2 + N(0, 1) |

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
