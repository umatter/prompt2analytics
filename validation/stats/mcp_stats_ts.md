# MCP statistics, hypothesis and time-series handlers

Tests of the MCP path (request parsing, option names, output fields and notes) for the
`stats_*`, hypothesis and `ts_*`/`timeseries_*` tools, checked against R where a reference
exists. Registry: `validation/methods/mcp_stats_ts_t3.json`. Every test is in
`crates/p2a-mcp/tests/tier3/wp56_stats_ts.rs` and drives `call_tool_with_session` on a case
dataset (missing cells and `Int64` columns included), so what is pinned is the tool's output,
not only the core function behind it.

## Scope

The tools of `crates/p2a-mcp/src/tools/handlers/{stats,hypothesis,timeseries}.rs`:

- option spellings (R's and the CLI's) for every option string of the three files, with the
  refusal of unknown values;
- the cross-correlation peak and R's lead/lag reading (`timeseries_ccf`);
- `stats_spline` defaults (`n`), `multivariate_factanal` scores (inline rows and
  `output_name`);
- the power solvers (`solve_for`), Mauchly with a `sigma` matrix, loglin notes;
- the spectrum interval and the Mahalanobis threshold;
- VAR/IRF/Granger, VARMA and VECM JSON; GARCH distributions and variance forecasts;
- Holt-Winters intervals, cpgram, StructTS and CausalImpact notes, the changepoint, cpgram
  and filter descriptions;
- the remaining tools no earlier test pinned (Tukey HSD through arima.sim, the
  `mcp_t3_remaining_handlers` entry).

The earlier MCP-path tests of these handler files (the `tier2` binaries and the unit tests in
the handler modules, registered in `validation/methods/mcp_stats.json`) pin rows used, missing
value notes and Int64 columns; the tests here pin the values and options those do not cover.
`test_mcp_ccf_nan_no_panic` and `test_mcp_stats_spline_default_cap` (the default `n` is capped
at 200 points on a generated series) load no case and are not registered.

## Cases

| Case | Dataset | R call |
|---|---|---|
| `mcp_groups_t3` (new) | `generated/mcp_groups_t3.csv` | `aov(vi ~ g, data = d[complete.cases(d$vi, d$g), ]); TukeyHSD(fit); wilcox.test(s1, s2, exact = FALSE)` |
| `mcp_series_t3` (new) | `generated/mcp_series_t3.csv` | `Box.test(w, lag = 10); decompose(ts(y, frequency = 12)); ar(z); runmed(z, 5, endrule = 'keep'); embed(z, 3)` |
| `mcp_factanal_t3` (new) | `generated/mcp_factanal_t3.csv` | `factanal(X, 2, rotation = 'varimax', scores = 'Bartlett')` |
| `mcp_misc_t3` (new) | `generated/robust_weighted_t3.csv` | `cor(r[, c('x1','x2','w')]); mad(x1, center = mean(x1)); mcnemar.test(matrix(c(40, 5, 15, 30), 2))` |
| `mcp_hw_additive_t3` (new) | `generated/holtwinters_series.csv` | `predict(HoltWinters(AirPassengers, 0.3, 0.05, 0.4, seasonal = 'additive'), 12, prediction.interval = TRUE)` |
| `stat_power_solve` | `grunfeld.csv` | `power.t.test(n, delta, sd, sig.level, power, type, alternative, strict, tol)` |
| `stat_ks_t3` | `generated/ks_t3.csv` | `ks.test(x, 'pnorm', 2, 2[, exact = FALSE, alternative]); ks.test(x, y[, exact = FALSE, alternative])` |
| `stat_ks_t3_exact_t4` | `generated/ks_t3.csv` | `ks.test(x, 'punif', -1.5, 3, alternative)` and the other one-sided tests of `stat_ks_t3` at R's default `exact = NULL` (case file `validation/r/cases/stats_rank_t4.R`) |
| `stat_ttest_levels_t3` | `generated/ttest_levels_t3.csv` | `t.test(x, mu = 5, conf.level = 0.9, alternative); t.test(x, y, conf.level = 0.99, alternative); t.test(x, y[1:12], paired = TRUE, ...)` |
| `stat_poisson_t3` | `grunfeld.csv` | `poisson.test(137, 24.19893); poisson.test(10, 5, alternative); poisson.test(c(11, 21), c(800, 3011), ...)` |
| `stat_fisher_rows_2000` | `generated/fisher_rows_2000.csv` | `fisher.test(table(d$treat, d$outcome))` |
| `stat_mantelhaen_na` | `generated/mantelhaen_na.csv` | `mantelhaen.test(table(treat, outcome, stratum))` |
| `stat_ts_acf_spline_t3` | `generated/ts_acf_spline_t3.csv` | `acf(x); pacf(x); acf(x, type = 'covariance'); ccf(x, y, lag.max = 5); spline(kx, ky, method = 'natural'[, n = 1])` |
| `ts_filter_ar_cpt` | `generated/ts_utils.csv` | `filter(x, f, sides = 2); ar(s, method = 'ols', intercept = FALSE); cpt.var(lev, method = 'PELT'); diffinv(x[1:12], xi = 1.5); ARMAacf(ar = c(0.5, -0.3)); cpgram(wn)` |
| `stat_robust_weighted_t3` | `generated/robust_weighted_t3.csv` | `mad(x1[, constant, center]); weighted.mean(x1, w); cov.wt(cbind(x1, x2), wt = w, cor = TRUE[, method = 'ML'])` |
| `stat_mauchly_sigma` | `generated/mauchly_sigma.csv` | `mauchly.test(lm(Y ~ 1), ...)` |
| `stat_loglin_nodecomp` | `generated/loglin_2x3x4.csv` | `loglin(xtabs(Freq ~ Var1 + Var2 + Var3), list(c(1,2), c(1,3), c(2,3)), eps = 1e-12)` |
| `stat_spectrum_t3` | `generated/spectrum_t3.csv` | `spec.pgram(x, spans = c(3, 3), taper = 0.1, detrend = TRUE); spec_ci(df, coverage)` |
| `stat_mahalanobis_t3` | `generated/mahalanobis_t3.csv` | `mahalanobis(X, colMeans(X), cov(X)); rowSums((C %*% MASS::ginv(cov(X))) * C)` |
| `var_canada` | `generated/var_canada.csv` | `VAR(Canada, p = 2, type = 'const')` |
| `granger_short` | `generated/granger_short.csv` | `grangertest(y ~ x, order = p)` |
| `varma_hr` | `generated/varma_sim.csv` | `long VAR(m) by lm.fit; lm(y_t ~ 0 + [1, y_{t-1..p}, ehat_{t-1..q}]), t = m + max(p, q) + 1..n; VAR(Y, p = 2, type = 'const')` |
| `vecm_denmark` | `generated/vecm_denmark.csv` | `ca.jo(denmark[, c('LRM','LRY','IBO','IDE')], ecdet = 'const', K = 2, spec = 'transitory')` |
| `garch_normal_t` | `generated/garch_returns.csv` | `optim + Newton on the written-out GARCH log-likelihood; tseries::garch(x - mean(x), order = c(1, 1))` |
| `ts_holtwinters_forecast` | `generated/holtwinters_series.csv` | `HoltWinters(AirPassengers, alpha = 0.3, beta = 0.05, gamma = 0.4, seasonal = 'multiplicative')` |
| `structts_nile_ukgas` | `generated/structts_series.csv` | `StructTS(Nile, 'level'); StructTS(Nile, 'trend'); StructTS(log10(UKgas), 'BSM'); StructTS(Nile, 'level', fixed = c(1500, NA)); StructTS(nile_na, 'level')` |
| `stat_cancor_t3` | `generated/cancor_t3.csv` | `cancor(as.matrix(d[, c('x1','x2','x3')]), as.matrix(d[, c('y1','y2')]))` |
| `stat_quantile_types` | `generated/quantile_sample.csv` | `quantile(x, probs, type = t); boxplot.stats(x, coef); 2 * pt(-\|t\|, df); qt(p, df)` |
| `stat_pp_t3` | `generated/pp_t3.csv` | `PP.test(x, lshort = TRUE \| FALSE)` |

## Tests and tolerances

| Test | Registry entry | Quantities | Tolerance |
|---|---|---|---|
| `test_validate_mcp_hypothesis_alternative_aliases` | `mcp_t3_option_spellings` | power at R's type/alternative; KS D, D+ and exact p against punif(-1.5, 3) and pexp(0.8) (R's default exact p for the uniform sample, two-sided and greater); one-sided t, Mood and Poisson p; acf, acvf, pacf and the white-noise bound; Box-Pierce X2; Yule-Walker AR; multiplicative seasonal figure; runmed keep; recursive filter; cpt.var changepoints and cost; mad(center = mean); cov.wt ML; refusal of unknown values | validation/README.md by n (Tol::coef/se/stat/p_value); power 1e-10 relative (WP38's ports, <= 1.8e-14); printed values at their decimals (t-test 6, decompose and changepoint 4) |
| `test_validate_mcp_ks_distribution_aliases` | `mcp_t3_option_spellings` | power at R's type/alternative; KS D, D+ and exact p against punif(-1.5, 3) and pexp(0.8) (R's default exact p for the uniform sample, two-sided and greater); one-sided t, Mood and Poisson p; acf, acvf, pacf and the white-noise bound; Box-Pierce X2; Yule-Walker AR; multiplicative seasonal figure; runmed keep; recursive filter; cpt.var changepoints and cost; mad(center = mean); cov.wt ML; refusal of unknown values | validation/README.md by n (Tol::coef/se/stat/p_value); power 1e-10 relative (WP38's ports, <= 1.8e-14); printed values at their decimals (t-test 6, decompose and changepoint 4) |
| `test_validate_mcp_power_option_spellings` | `mcp_t3_option_spellings` | power at R's type/alternative; KS D, D+ and exact p against punif(-1.5, 3) and pexp(0.8) (R's default exact p for the uniform sample, two-sided and greater); one-sided t, Mood and Poisson p; acf, acvf, pacf and the white-noise bound; Box-Pierce X2; Yule-Walker AR; multiplicative seasonal figure; runmed keep; recursive filter; cpt.var changepoints and cost; mad(center = mean); cov.wt ML; refusal of unknown values | validation/README.md by n (Tol::coef/se/stat/p_value); power 1e-10 relative (WP38's ports, <= 1.8e-14); printed values at their decimals (t-test 6, decompose and changepoint 4) |
| `test_validate_mcp_stats_option_aliases` | `mcp_t3_option_spellings` | power at R's type/alternative; KS D, D+ and exact p against punif(-1.5, 3) and pexp(0.8) (R's default exact p for the uniform sample, two-sided and greater); one-sided t, Mood and Poisson p; acf, acvf, pacf and the white-noise bound; Box-Pierce X2; Yule-Walker AR; multiplicative seasonal figure; runmed keep; recursive filter; cpt.var changepoints and cost; mad(center = mean); cov.wt ML; refusal of unknown values | validation/README.md by n (Tol::coef/se/stat/p_value); power 1e-10 relative (WP38's ports, <= 1.8e-14); printed values at their decimals (t-test 6, decompose and changepoint 4) |
| `test_validate_mcp_acf_type_aliases` | `mcp_t3_option_spellings` | power at R's type/alternative; KS D, D+ and exact p against punif(-1.5, 3) and pexp(0.8) (R's default exact p for the uniform sample, two-sided and greater); one-sided t, Mood and Poisson p; acf, acvf, pacf and the white-noise bound; Box-Pierce X2; Yule-Walker AR; multiplicative seasonal figure; runmed keep; recursive filter; cpt.var changepoints and cost; mad(center = mean); cov.wt ML; refusal of unknown values | validation/README.md by n (Tol::coef/se/stat/p_value); power 1e-10 relative (WP38's ports, <= 1.8e-14); printed values at their decimals (t-test 6, decompose and changepoint 4) |
| `test_validate_mcp_timeseries_option_aliases` | `mcp_t3_option_spellings` | power at R's type/alternative; KS D, D+ and exact p against punif(-1.5, 3) and pexp(0.8) (R's default exact p for the uniform sample, two-sided and greater); one-sided t, Mood and Poisson p; acf, acvf, pacf and the white-noise bound; Box-Pierce X2; Yule-Walker AR; multiplicative seasonal figure; runmed keep; recursive filter; cpt.var changepoints and cost; mad(center = mean); cov.wt ML; refusal of unknown values | validation/README.md by n (Tol::coef/se/stat/p_value); power 1e-10 relative (WP38's ports, <= 1.8e-14); printed values at their decimals (t-test 6, decompose and changepoint 4) |
| `test_validate_mcp_ccf_direction` | `mcp_t3_ccf` | ccf at lags -5..5; lags; white-noise bound; peak lag and its reading; tool description | validation/README.md, n = 64: values 1e-5 relative, bound 1e-6, lags exact |
| `test_validate_mcp_stats_spline_default_n` | `mcp_t3_spline` | x and y at the 30 default points; n = 1; n_points; n_pairs; missing; xout description | validation/README.md, n = 10: 1e-6 relative; counts exact |
| `test_validate_mcp_factanal_scores_returned` | `mcp_t3_factanal_scores` | Bartlett scores at rows 1-3, 999-1001, 1199-1200; regression scores at rows 1-3, 999-1000; rows returned (1000 of 1200); score columns of the output dataset | iterative: Tol::iterative(1e-4) (STAT-62's factanal rule); an R emulation of p2a's default fixed point gives 3.77e-5 |
| `test_validate_mcp_factanal_scores_output_name` | `mcp_t3_factanal_scores` | Bartlett scores at rows 1-3, 999-1001, 1199-1200; regression scores at rows 1-3, 999-1000; rows returned (1000 of 1200); score columns of the output dataset | iterative: Tol::iterative(1e-4) (STAT-62's factanal rule); an R emulation of p2a's default fixed point gives 3.77e-5 |
| `test_validate_mcp_power_solve_defaults` | `mcp_t3_power_mauchly_loglin` | power; solved n, sd, groups, within_var; Cohen's f; Mauchly W, chi2, df, p (second order), first-order p, GG epsilon with and without Sigma; loglin G2, X2, df, max_deviation, non-convergence warning | power 1e-10 and roots 1e-9 relative (WP38's ports reproduce R to <= 2.2e-14); validation/README.md for Mauchly (n = 15) and loglin (720 counts) |
| `test_validate_mcp_power_solve_for` | `mcp_t3_power_mauchly_loglin` | power; solved n, sd, groups, within_var; Cohen's f; Mauchly W, chi2, df, p (second order), first-order p, GG epsilon with and without Sigma; loglin G2, X2, df, max_deviation, non-convergence warning | power 1e-10 and roots 1e-9 relative (WP38's ports reproduce R to <= 2.2e-14); validation/README.md for Mauchly (n = 15) and loglin (720 counts) |
| `test_validate_mcp_stats_mauchly_sigma` | `mcp_t3_power_mauchly_loglin` | power; solved n, sd, groups, within_var; Cohen's f; Mauchly W, chi2, df, p (second order), first-order p, GG epsilon with and without Sigma; loglin G2, X2, df, max_deviation, non-convergence warning | power 1e-10 and roots 1e-9 relative (WP38's ports reproduce R to <= 2.2e-14); validation/README.md for Mauchly (n = 15) and loglin (720 counts) |
| `test_validate_mcp_stats_loglin_nondecomposable` | `mcp_t3_power_mauchly_loglin` | power; solved n, sd, groups, within_var; Cohen's f; Mauchly W, chi2, df, p (second order), first-order p, GG epsilon with and without Sigma; loglin G2, X2, df, max_deviation, non-convergence warning | power 1e-10 and roots 1e-9 relative (WP38's ports reproduce R to <= 2.2e-14); validation/README.md for Mauchly (n = 15) and loglin (720 counts) |
| `test_validate_mcp_timeseries_spectrum_ci` | `mcp_t3_spectrum_mahalanobis` | spec.ci multipliers at df 18; peak frequency; AR: no interval; threshold 7.814727903251179; first ten distances; refusal of a singular covariance (rank 1 of 2) | multipliers 1e-10 relative (WP40); validation/README.md otherwise; distances at their 4 printed decimals |
| `test_validate_mcp_mahalanobis_threshold` | `mcp_t3_spectrum_mahalanobis` | spec.ci multipliers at df 18; peak frequency; AR: no interval; threshold 7.814727903251179; first ten distances; refusal of a singular covariance (rank 1 of 2) | multipliers 1e-10 relative (WP40); validation/README.md otherwise; distances at their 4 printed decimals |
| `test_validate_mcp_ts_var_coefficients` | `mcp_t3_var_models` | VAR coefficients, SEs, t, p, Sigma_u, logLik, AIC, BIC, VARselect criteria; orthogonalised IRF h = 0..12; Granger F, p, df at order 2 and the capped default 6; VARMA coefficients, SEs, p, logLik, AIC, BIC at m = 8 and the default 35; VECM trace and max-eigen statistics, 5% critical values, eigenvalues, beta, alpha, Gamma, selected rank (const, none, longrun) | validation/README.md by n (VAR 82, Granger 20, VARMA 391/364, VECM 53) |
| `test_validate_mcp_ts_var_irf` | `mcp_t3_var_models` | VAR coefficients, SEs, t, p, Sigma_u, logLik, AIC, BIC, VARselect criteria; orthogonalised IRF h = 0..12; Granger F, p, df at order 2 and the capped default 6; VARMA coefficients, SEs, p, logLik, AIC, BIC at m = 8 and the default 35; VECM trace and max-eigen statistics, 5% critical values, eigenvalues, beta, alpha, Gamma, selected rank (const, none, longrun) | validation/README.md by n (VAR 82, Granger 20, VARMA 391/364, VECM 53) |
| `test_validate_mcp_ts_granger` | `mcp_t3_var_models` | VAR coefficients, SEs, t, p, Sigma_u, logLik, AIC, BIC, VARselect criteria; orthogonalised IRF h = 0..12; Granger F, p, df at order 2 and the capped default 6; VARMA coefficients, SEs, p, logLik, AIC, BIC at m = 8 and the default 35; VECM trace and max-eigen statistics, 5% critical values, eigenvalues, beta, alpha, Gamma, selected rank (const, none, longrun) | validation/README.md by n (VAR 82, Granger 20, VARMA 391/364, VECM 53) |
| `test_validate_mcp_ts_varma` | `mcp_t3_var_models` | VAR coefficients, SEs, t, p, Sigma_u, logLik, AIC, BIC, VARselect criteria; orthogonalised IRF h = 0..12; Granger F, p, df at order 2 and the capped default 6; VARMA coefficients, SEs, p, logLik, AIC, BIC at m = 8 and the default 35; VECM trace and max-eigen statistics, 5% critical values, eigenvalues, beta, alpha, Gamma, selected rank (const, none, longrun) | validation/README.md by n (VAR 82, Granger 20, VARMA 391/364, VECM 53) |
| `test_validate_mcp_ts_vecm` | `mcp_t3_var_models` | VAR coefficients, SEs, t, p, Sigma_u, logLik, AIC, BIC, VARselect criteria; orthogonalised IRF h = 0..12; Granger F, p, df at order 2 and the capped default 6; VARMA coefficients, SEs, p, logLik, AIC, BIC at m = 8 and the default 35; VECM trace and max-eigen statistics, 5% critical values, eigenvalues, beta, alpha, Gamma, selected rank (const, none, longrun) | validation/README.md by n (VAR 82, Granger 20, VARMA 391/364, VECM 53) |
| `test_validate_mcp_ts_garch_fit_t` | `mcp_t3_garch_statespace_forecasting` | GARCH parameters, SEs, logLik, 10-step variance forecasts (norm, std); Holt-Winters 12-step forecasts and 95% bounds, SSE; cpgram band, Stephens statistic, p, maximum deviation, first 20 frequencies and ordinates; StructTS BSM variances and logLik; CausalImpact clipping warnings (property; no R); AR warnings key (mcp_series_t3), MBIC, cpgram and filter descriptions | GARCH parameters and forecasts 1e-8, SEs 1e-6 relative (WP35); Holt-Winters at 4 printed decimals; validation/README.md otherwise; StructTS Tol::iterative(1e-6) (WP36) |
| `test_validate_mcp_ts_holt_winters_forecast` | `mcp_t3_garch_statespace_forecasting` | GARCH parameters, SEs, logLik, 10-step variance forecasts (norm, std); Holt-Winters 12-step forecasts and 95% bounds, SSE; cpgram band, Stephens statistic, p, maximum deviation, first 20 frequencies and ordinates; StructTS BSM variances and logLik; CausalImpact clipping warnings (property; no R); AR warnings key (mcp_series_t3), MBIC, cpgram and filter descriptions | GARCH parameters and forecasts 1e-8, SEs 1e-6 relative (WP35); Holt-Winters at 4 printed decimals; validation/README.md otherwise; StructTS Tol::iterative(1e-6) (WP36) |
| `test_validate_mcp_timeseries_cpgram` | `mcp_t3_garch_statespace_forecasting` | GARCH parameters, SEs, logLik, 10-step variance forecasts (norm, std); Holt-Winters 12-step forecasts and 95% bounds, SSE; cpgram band, Stephens statistic, p, maximum deviation, first 20 frequencies and ordinates; StructTS BSM variances and logLik; CausalImpact clipping warnings (property; no R); AR warnings key (mcp_series_t3), MBIC, cpgram and filter descriptions | GARCH parameters and forecasts 1e-8, SEs 1e-6 relative (WP35); Holt-Winters at 4 printed decimals; validation/README.md otherwise; StructTS Tol::iterative(1e-6) (WP36) |
| `test_validate_mcp_timeseries_structts_bsm` | `mcp_t3_garch_statespace_forecasting` | GARCH parameters, SEs, logLik, 10-step variance forecasts (norm, std); Holt-Winters 12-step forecasts and 95% bounds, SSE; cpgram band, Stephens statistic, p, maximum deviation, first 20 frequencies and ordinates; StructTS BSM variances and logLik; CausalImpact clipping warnings (property; no R); AR warnings key (mcp_series_t3), MBIC, cpgram and filter descriptions | GARCH parameters and forecasts 1e-8, SEs 1e-6 relative (WP35); Holt-Winters at 4 printed decimals; validation/README.md otherwise; StructTS Tol::iterative(1e-6) (WP36) |
| `test_validate_mcp_causal_impact_reports_clipped_periods` | `mcp_t3_garch_statespace_forecasting` | GARCH parameters, SEs, logLik, 10-step variance forecasts (norm, std); Holt-Winters 12-step forecasts and 95% bounds, SSE; cpgram band, Stephens statistic, p, maximum deviation, first 20 frequencies and ordinates; StructTS BSM variances and logLik; CausalImpact clipping warnings (property; no R); AR warnings key (mcp_series_t3), MBIC, cpgram and filter descriptions | GARCH parameters and forecasts 1e-8, SEs 1e-6 relative (WP35); Holt-Winters at 4 printed decimals; validation/README.md otherwise; StructTS Tol::iterative(1e-6) (WP36) |
| `test_validate_mcp_timeseries_wp37_texts` | `mcp_t3_garch_statespace_forecasting` | GARCH parameters, SEs, logLik, 10-step variance forecasts (norm, std); Holt-Winters 12-step forecasts and 95% bounds, SSE; cpgram band, Stephens statistic, p, maximum deviation, first 20 frequencies and ordinates; StructTS BSM variances and logLik; CausalImpact clipping warnings (property; no R); AR warnings key (mcp_series_t3), MBIC, cpgram and filter descriptions | GARCH parameters and forecasts 1e-8, SEs 1e-6 relative (WP35); Holt-Winters at 4 printed decimals; validation/README.md otherwise; StructTS Tol::iterative(1e-6) (WP36) |
| `test_validate_mcp_tukey_hsd` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_pairwise_wilcox` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_wilcoxon_two_sample` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_mcnemar` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_poisson_one_sided` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_model_tables_one_way` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_se_contrast` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_compute_correlation` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_cancor_centres` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_stats_iqr_type6` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_pp_test` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_box_test_ljung` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_decompose_additive` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_ar_burg` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_runmed_constant` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_series_utilities` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_arma_acf` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |
| `test_validate_mcp_lag_mstl_arima_sim` | `mcp_t3_remaining_handlers` | see validation/stats/mcp_stats_ts.md, one row per test | validation/README.md by n; printed values at their decimals; counts exact |

## Deviations

Registry deviations (quantities kept out of the comparisons, with a `// deviation <id>`
comment in the test):

- **STAT-69** (`mcp_t3_spline`), method fmm: only method = natural is pinned.
- **STAT-82** (`mcp_t3_power_mauchly_loglin`), Mauchly p (R's pchisq form): p_value is pinned to the Anderson second-order p (psi_contrasts_p_anderson, contrasts_p_anderson); R's mauchly.test p is a 1e-3 cross-check only.
- **STAT-39** (`mcp_t3_spectrum_mahalanobis`), spectrum df: p2a 18 for spans c(3, 3), spec.pgram 6.552102376599634; multipliers pinned at p2a's df.
- **STAT-68** (`mcp_t3_spectrum_mahalanobis`), spectral ordinates: not pinned.
- **TS-22** (`mcp_t3_garch_statespace_forecasting`), airpass_mult_upr, airpass_mult_lwr: predict.HoltWinters' multiplicative interval formula is wrong; the tool prints point forecasts and a note for multiplicative fits, intervals only for additive and non-seasonal fits (hw_add_upr, hw_add_lwr pinned).
- **STAT-16** (`mcp_t3_remaining_handlers`), cancor correlations and coefficients: centres pinned only.
- **STAT-75** (`mcp_t3_remaining_handlers`), model-table SEs: means, grand mean and group sizes pinned.
- **TS-14** (`mcp_t3_remaining_handlers`), ARMAacf for mixed ARMA: pure AR and MA pinned.
- **TS-26** (`mcp_t3_remaining_handlers`), MSTL components: observation count and periods only.
- **TS-29** (`mcp_t3_remaining_handlers`), timeseries_lag values: the tool returns x[t + k]; R's dplyr-style lag is lag_dplyr3_head.
- **TS-34** (`mcp_t3_remaining_handlers`), runmed endrule median: constant pinned here (keep in the option test); R's median is runmed_median_head.
- **TOOL-28** (`mcp_t3_remaining_handlers`), arima_sim sd: seed determinism only.

R values recorded for the deviations: `lag_dplyr3_head` (TS-29) and `runmed_median_head`
(TS-34) in `mcp_series_t3`; the spectrum's `ci_df[0]` is 6.552102376599634 in R where p2a
uses 18 (STAT-39, `stat_spectrum_t3`); R's canonical correlations are `cor` in
`stat_cancor_t3` (STAT-16). The PP statistic and p-value are pinned to PP.test (no
deviation).

## Wall-clock

Measured on 2026-11-09 against ff8e7559 (release builds, one core, eight interleaved rounds, load average
1.4-2.1) and accepted by the maintainer the same day:

- `stats_spline` on 10,000 rows: 0.74 ms, 1.16x of the base (1.09x on minima), from the larger default
  output (200 points with x and y, 9.6 KB).
- `multivariate_factanal` with regression scores: 1.20 ms on 1,200 rows (1.56x) and 2.51 ms on 10,000
  (1.36x). The handler change adds 1.04x and 1.01x; the rest is the core fit, as changed before it.

## Notes

- factanal regression scores are pinned for varimax only. For promax, R multiplies by the
  factor correlation Phi (`sc %*% Phi`) and p2a's `compute_factor_scores` does not (new
  observation, no finding id).
- `hypothesis_poisson` with a one-sided `alternative` returns the one-sided interval, and the
  two-sided p-value is R's (outcomes no more likely than the observed one); the less and greater
  intervals, the rate-ratio interval and the two-sided p are pinned in
  `test_validate_mcp_poisson_one_sided` (R: `poisson.test(10, 5, alternative = "less")$conf.int`
  is [0, 3.392443847144379]).
- Holt-Winters prediction intervals are printed for additive and non-seasonal fits only;
  predict.HoltWinters' multiplicative formula is wrong (TS-22), so a multiplicative fit gets
  point forecasts and a note.
