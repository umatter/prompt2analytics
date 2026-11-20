# Validation: CLI model commands (`reg`, `discrete`, `panel`, `survival`, `stats`), persisted datasets, notes

## Method Overview

T3-WP60 makes the computing commands of `p2a reg`, `discrete`, `panel`, `survival` and `stats`
report what their core result says about itself (convergence status and every warning, in text and
JSON), parses their options into the cores' enums, and exposes the options the Tier 3 cores added.
The numbers come from the cores; the tests here pin them through the binary against R.

## What is pinned

| Test | Case(s) | Keys | Tolerance |
|---|---|---|---|
| `test_validate_cli_logit_prints_converged_and_warnings` | `disc_binary_glm` | `aliased_coef`; the separation and glm.fit warnings in text and JSON | coefficients `Tol::coef(400)` |
| `test_validate_cli_logit_probit_glm` | `disc_binary_glm` | `logit_coef/se/z/p/loglik/aic/null_loglik`; `probit_coef_tight`, `probit_se_tight`, `probit_loglik_tight`, `probit_aic_tight` | logit `Tol::coef/se/stat/p_value(400)`; probit coefficients and SEs `Tol::iterative(1e-6)` (p2a's default control), log-likelihood and AIC `Tol::stat(400)` |
| `test_validate_cli_mlogit_base` | `disc_multinom_overlap` | `refB_coef_tight`, `refB_se_tight`, `coef_tight`, `loglik` | `Tol::iterative(1e-8)` coefficients, `1e-6` SEs |
| `test_validate_cli_clogit_asc_reflevel` | `disc_mlogit_fishing` | `generic_asc_*_tight`, `generic_noasc_*_tight`, the reflevel-pier reparametrisation | `Tol::iterative(1e-9)` coefficients, `1e-8` SEs |
| `test_validate_cli_discrete_poisson` | `cli_discrete_poisson` | `coef`, `se`, `z`, `p`, `loglik`, `aic`, `deviance`, `null_deviance`, `n`, `na_coef`, `na_se`, `na_n_dropped` | `Tol::coef/se/stat/p_value(400)`; counts exact |
| `test_validate_cli_cox_ties_robust_cluster` | `surv_cox_robust` | `{efron,breslow}_coef/se/z/p/loglik`, `efron_se_robust/z_robust/p_robust/rscore_robust/rscore_p_robust`, `efron_coef_cluster/se_cluster/se_model_cluster/z_cluster/p_cluster/rscore_cluster`, `n`, `n_clusters`, `n_cluster_obs` | `Tol::coef/se/stat/p_value(227)`; counts exact |
| `test_validate_cli_aft_distributions` | `surv_aft` | `{weibull,exponential,lognormal,loglogistic}_coef/se/z/p/scale/loglik/aic/bic`, `*_se_log_scale` (not exponential: scale fixed), `weibull_z_log_scale` in the printed Log(scale) row | `Tol::coef/se/stat/p_value(227)`; printed row `Tol::printed(4)` |
| `test_validate_cli_competing_risks_ci` | `cli_competing_risks` | `times`, `cif_k`, `survfit_se_k`, `survfit_ci95_*_k`, `survfit_ci50_*_k` (null where the CIF is 0) | CIF `Tol::coef(200)`, SEs `Tol::se(200)`, limits `Tol::stat(200)`; first printed row `Tol::printed(4)` |
| `test_validate_cli_unit_root_lag_selection` | `purtest_lag_selection` | `ips_{aic,sic,hall}_dfcor_lags`, `trho_{aic,sic,hall}_dfcor`, `trend_lags_aic_dfcor`, `trend_trho_aic_dfcor`; the `Unit lags:` line | lags exact; statistics `Tol::stat(480)` |
| `test_validate_cli_panel_absorbed_regressors` | `panel_absorbed_regressors` | `fe_coef`, `fe_se`, `hdfe2_coef`; `dropped` names z (fe) and z, w (hdfe) | `Tol::coef/se(180)` |
| `test_validate_cli_quantreg_se` | `rq_br_se` | `tau50_coef`, `tau50_{se,t,p,ci_lower,ci_upper}_{nid,iid,ker}`, `tau50_objective`; the default fit has no rank note and runs br at n = 100 | `Tol::coef/se/stat/p_value(100)` |
| `test_validate_cli_gls_compsymm_group` | `gls_compsymm` | `{reml,ml}_profile_{coef,se,rho,sigma,loglik}`, `{reml,ml}_{coef,aic,bic}`, `n_groups`, `cs03_reml_{coef,se}` | `Tol::coef/se/stat(150)`; rho `Tol::iterative(1e-6)`; nlme's own coefficients (cross-check) `Tol::iterative(1e-6)` |
| `test_validate_cli_loess_masspoint_warns` | `smoothers_gcv_periodic_loess` | `loess_masspoint_expected` (first 10 fitted values); the zero-width-neighbourhood warning in text and JSON | `Tol::coef(100)` |
| `test_validate_cli_hac_aliased_df` | `cli_hac_aliased` | `coef`, `se_nw`, `t_nw`, `p_nw` (identified columns), `df_resid`, `r2` | `Tol::coef/se/stat/p_value(120)`; df exact |
| `test_validate_cli_nls_expdecay` | `nls_expdecay` | `coef`, `se`, `rss` | `Tol::iterative(1e-6)` coefficients, `1e-5` SEs, `1e-8` RSS |
| `test_validate_cli_pp_test_internals` | `stat_pp_t3` | `rw_short_gamma`, `rw_short_t`, `rw_short_sigma2`, `rw_short_lambda2`, `rw_short_lag` for the default run and bare `--lshort`; `rw_long_gamma`, `rw_long_t`, `rw_long_sigma2`, `rw_long_lambda2`, `rw_long_lag` for `--lshort false` (Z(tau) and its p-value not pinned: STAT-6, STAT-35) | gamma `Tol::coef(200)`, the rest `Tol::stat(200)`; lag exact |
| `test_validate_cli_munge_persists_across_invocations`, `test_validate_cli_derived_dataset_names_replacement_and_errors` | `cli_munge_ols_grunfeld` | `sel_coef`, `sel_se_hc1`, `sel_t_hc1`, `sel_p_hc1`, `sel_r2`, `sel_adj_r2`, `sel_n` on the subset `munge select` stored in one invocation and `reg ols` read in the next; `sorted_{inv,firm,year}_head` | `Tol::coef/se/stat/p_value(200)`; n and rows exact |
| `test_validate_cli_exported_script_with_munge_runs` | `cli_munge_ols_grunfeld` | `post_coef`, `post_se_hc1`, `post_t_hc1`, `post_p_hc1`, `post_rows` on `munge filter` (year >= 1945); the exported script, run with bash from another directory, prints the same table and leaves nothing behind | `Tol::printed(6)` coefficients and SEs, `printed(3)` t, `printed(4)` p; rows exact |
| `test_validate_cli_iv_report` | `cli_iv_report` | `causal iv` coefficients, HC1 and (`--robust false`) classical SEs, t, p, R-squared, ivreg's adjusted R-squared 1 - (1 - R²)(n - 1)/df | `Tol::coef/se/stat/p_value(n)` |
| `test_validate_cli_gsynth_defaults` | `cli_gsynth_defaults` | `unit_r0_att_avg` (`--r-max 0`), `unit_r1_att_avg` (`--n-factors 1`); default `force` unit, `r_max` 5; the default run's ATT at the r core chooses | relative 5e-4 (as the r spellings test); deviation CLI-33: gsynth's default CV (random folds, then fect's 1-SE rule) chooses r = 0 under the case's seed (`default_r`, ATT `default_att_avg` 6.948), core's leave-one-out CV chooses r = 1 (ATT 2.967), as fect's leave-one-out CV does under each of its rules; both keys are not compared |
| `test_validate_cli_panel_fe_missing_note` | `panel_grunfeld_na` | fe and re: the `Note:` line last, not among the warnings; JSON `missing.n_dropped` 5 | exact |
| `test_validate_cli_staggered_universal_simple` | `did_cs_varying`, `did_cs_small_cohorts` | `never_varying_simple_overall_att/_se` (`overall_simple`), `never_universal_group_overall_att/_se` (`--base-period-type universal`), `n_dropped_first_period` | `Tol::coef/se(n)`; counts exact |
| `test_validate_cli_hurdle_theta_se` | `disc_hurdle_t4` | `hnb_theta`, `hnb_se_theta`, `hnb_se_log_theta` (JSON `theta`, `theta_std_error`, `log_theta_std_error`; the text line as zinb's) | `Tol::iterative(1e-7)` theta, `1e-5` SEs |

`test_cli_smooth_spline_criterion` loads no case: it checks that `--criterion gcv|cv` gives the core's
own fit (p2a's smoothing spline is not R's yet), so it is not in the registry.

Not listed above, because they read no R case: the table tests `test_cli_json_has_warnings_and_diagnostics`
(every computing command of these files prints one JSON document with its notes) and
`test_cli_option_values_parse_aliases_and_reject_unknown` (option values parse through the cores' spellings and
unknown ones are usage errors), plus `test_cli_smooth_spline_criterion`, `test_cli_munge_bin_and_cast_print_one_json_document`,
`test_cli_munge_fill_na_mode_and_value_counts_normalize`, `test_cli_data_load_jsonl_persists` and
`test_cli_data_save_json_round_trip`. Grouped Kaplan-Meier prints one JSON object, its groups in a `groups`
array (CLI-38).

## Persisted datasets

Every `p2a` invocation is its own process. With `--session <file>`, a dataset that a command produces
(`munge *`, `data generate`) is written as Parquet to `<file>.datasets/<name>.parquet` (a temporary file, then a
rename) and registered in the session with `"derived": true`, `"format": "parquet"` and a `source_path` relative to
the session file's directory, so a session directory can be moved with its sidecar.

- Names are percent-encoded into file names (`%XX` for every byte outside `[A-Za-z0-9._-]`, and for a leading dot);
  a name whose encoding exceeds 200 bytes is refused, and the error shows it shortened.
- A result that cannot be written is an error and is not registered (no "Created dataset" line).
- Loading a file over a derived dataset's name removes the stored copy.
- Session datasets are read when a command first asks for one. With `--max-memory` set, every dataset is read
  at start-up so the estimate covers the whole session, and the warning is printed once per invocation.
- Exported scripts (`script export`) replay into a session in a `mktemp -d` directory that an `EXIT` trap removes,
  the derived datasets with it; they `cd` into the recorded directory (`P2A_WORKDIR` overrides) and skip commands
  that failed when recorded (`--include-failed` replays them).
- `munge bin` and `munge cast` print one JSON document: `message`, plus `details` (bin's edges and counts) or
  `warnings` (cast's conversion note).
- `data load` and the session reload read every format `DataLoader::load` reads (CSV with `--na-strings`
  through `load_with_na`); an unknown extension is an error listing the readable ones. `data save --output x.json`
  writes text cells as plain JSON strings, so the file loads back to the same values.

## Notes contract

- Text: after the result, `Converged: yes|no (<n> iterations)` for iterative fits and one `Warning: <text>` line per
  core warning, then the `Missing values:` line, as earlier tiers print it.
- JSON: every computing command prints a `warnings` array (empty when there is none), plus `converged` and
  `iterations` for iterative fits. The exception: `reg ols`, `reg clustered` and `panel fe/re/hdfe`, whose JSON is
  quoted in the paper, carry `warnings` only when it is non-empty.
- Formats of earlier tiers are kept; the only golden change is the aliased-regressor warning line in
  `crates/p2a-cli/tests/golden/cli_diagnostics.txt`.
- `panel hdfe` text has no `Converged:` line, as fixest prints none; its JSON has `converged` and `iterations`.

## Options

Core enums are parsed with their own `FromStr` (case- and separator-insensitive, with the aliases the core lists in
its error message); an unknown value is a usage error (exit 2) naming the accepted values.

| Command | Option | Accepted values (default first) |
|---|---|---|
| `survival cox` | `--ties` | `efron`, `breslow` (`TiesMethod`) |
| `survival aft` | `--dist` | `weibull`, `exponential`, `lognormal`, `loglogistic` (`AftDistribution`) |
| `reg quantreg` | `--method`, `--se` | `auto`, `br` (`simplex`, ...), `fn` (`interior_point`, ...), `irls` (`QuantRegMethod`); `nid` (`sandwich`, ...), `iid`, `ker` (`kernel`, `powell`), `boot` (`QuantRegSe`) |
| `reg gls` | `--correlation`, `--method` | `ar1_auto` or a `GlsCorrelation` structure; `reml`, `ml` (`GlsMethod`) |
| `reg step` | `--direction` | `both`, `forward`, `backward` (`StepDirection`) |
| `reg smooth-spline` | `--criterion` | `gcv` (aliases `generalized_cv`, ...), `cv` (aliases `loocv`, `loo`, ...) (`SplineCriterion`) |
| `panel unit-root` | `--test`, `--model`, `--lag-selection` | `llc` (aliases `levinlin`, `levin_lin_chu`), `ips`, `madwu` (aliases `maddala_wu`, `fisher`), `hadri` (`PanelUnitRootTest`); `intercept` (aliases `drift`, `c`, `const`), `none` (`nc`), `trend` (`ct`) (`PanelModel`); `aic`, `sic`, `hall` |
| `stats anova-two` | `--ss-type` | `I`, `II`, `III` (`SsType`) |
| `panel feglm` | `--family` | `logit` (alias `binomial`), `probit`, `poisson`, `gaussian` |
| `panel pvcm`, `panel pmg` | `--model` | `within`, `random` (alias `swamy`) |
| `panel gls` | `--model` | `fe` (aliases `fixed-effects`, `within`), `fd` (aliases `first-difference`, `first_difference`), `pooling` |
| `panel gmm` | `--step`, `--vcov` | `onestep`, `twostep`; `robust`, `classical` |
| `munge fill-na` | `--method` | `mean`, `median`, `mode` (alias `most-frequent`), `constant`, `forward`, `backward`, `zero` |

`munge fill-na --method constant` needs `--value` (typed to the column, text for String columns); `--value` with
any other method is an error.

## Known differences

- `smooth-spline` numbers are not R-pinned (REG-29, REG-30); its default criterion is now GCV, as R's.
- `data head -F json` prints typed cells (numbers, booleans, null, text and dates as strings) and `data describe
  -F json` one object with `column_types` (CLI-15); grouped Kaplan-Meier prints one object with a `groups` array
  (CLI-38).
- `discrete logit/probit` report McFadden's pseudo R-squared (`pseudo_r_squared`) and z statistics (`z_value`,
  `statistic` `"z"`), as `summary(glm)`; there is no `r_squared` key.
- `discrete logit/probit` use p2a's default glm control (ECON-60); the probit pins compare at the tight MLE, which
  the default control reaches within 1e-6.
- `panel unit-root` keeps the core's defaults for lags and the lag rule, which differ from `plm::purtest`'s
  defaults; the tests pass the options explicitly.
- `reg quantreg` defaults to `--method auto` and `--se nid` at every n (R's `summary.rq` switches to `iid`-based
  rules at small n).
- `survival cox` now defaults to Efron ties, as `survival::coxph`; before, the CLI used Breslow.
- `stats pp-test` defaults to `--lshort true`, `PP.test`'s short truncation lag trunc(4 (n/100)^(1/4)) (CLI-21);
  `--lshort false` is the long lag. Z(tau) and its p-value are not pinned (STAT-6, STAT-35).


## Timing

Release builds of `p2a` (`--no-default-features --features spectral-analysis`) at the v0.5.0 base f80b0131 and at the merged head 156e744f, one process per invocation (`p2a --session s.json <command>`, the session's datasets loaded from CSV each time, so about 12-25 ms of every number is start-up and loading), pinned with `taskset -c 2`, `RAYON_NUM_THREADS=1`, `POLARS_MAX_THREADS=1`; per row and n, 9 rounds per side interleaved, each the median of 11 invocations after 3 warm-ups; the table shows the median of the medians, the ratio of the minima and the 1-minute load average at the row's start and end. Data: n rows of simulated noise data (head: 5 mixed columns; describe: 20 columns; acf/pacf/ccf/box/pp: AR(1) series; moran: n random points, k = 4; gsynth: n/20 units x 20 periods, at least 5 units; iv: 2 instruments; km: 4 groups; chisq-gof: 6 categories; coefplot and hac: 3 regressors, hac with 4 lags). Quiet re-time at the integration (2026-11-18, nothing else running): a separate 5-round run gave ratios within 0.05 of these, with absolute times about 6 ms lower on both sides.

| Command | n | f80b0131 (ms) | head (ms) | head/base | minima | load |
|---|---:|---:|---:|---:|---:|---:|
| `causal gsynth default` | 100 | 24.9 | 25.8 | 1.03x | 1.00x | 1.80-1.73 |
| `causal gsynth default` | 1,000 | 40.5 | 33.6 | 0.83x | 0.82x | 1.73-1.93 |
| `causal gsynth default` | 10,000 | 53.9 | 43.5 | 0.81x | 0.80x | 1.93-2.01 |
| `causal iv default` | 100 | 24.8 | 25.2 | 1.02x | 1.02x | 2.01-1.85 |
| `causal iv default` | 1,000 | 26.4 | 26.9 | 1.02x | 1.02x | 1.85-1.78 |
| `causal iv default` | 10,000 | 35.1 | 35.9 | 1.02x | 1.02x | 1.78-1.88 |
| `survival km --group -F json` | 100 | 25.5 | 26.0 | 1.02x | 1.02x | 1.88-1.81 |
| `survival km --group -F json` | 1,000 | 26.9 | 27.8 | 1.03x | 1.03x | 1.81-1.68 |
| `survival km --group -F json` | 10,000 | 37.3 | 43.0 | 1.15x | 1.15x | 1.68-1.71 |
| `reg hac` | 100 | 27.0 | 27.7 | 1.02x | 1.01x | 2.02-1.94 |
| `reg hac` | 1,000 | 28.2 | 28.9 | 1.02x | 1.03x | 1.94-1.95 |
| `reg hac` | 10,000 | 33.7 | 34.0 | 1.01x | 1.01x | 1.95-1.96 |

`survival km --group -F json` at n = 10,000 is 1.15x here and 1.11x in the 5-round run (30.98 ms base, 34.35 ms head); peak RSS 28.3 MiB base, 35.7 MiB head. The head prints one JSON document where the base printed one per group; the nested, pretty-printed curves are about 24% more bytes (1.67 MB against 1.35 MB), the likely cause of the difference. Above 1.10x, with the higher peak RSS: signed off by the user on 2026-11-18. The v0.5.2 (90a99318) column was not measured.

## Reproduce

```bash
Rscript validation/r/run.R --check
cargo test -p p2a-cli --no-default-features --features spectral-analysis --test tier3 wp60_models
```
