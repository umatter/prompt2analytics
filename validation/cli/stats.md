# Validation: CLI `stats` defaults, chi-squared counts and coefficient plots

## Method Overview

The `stats` commands follow R's defaults and report R's statistics:

| Command | Default or output | R reference |
|---|---|---|
| `stats acf`, `pacf` | `--lag-max` floor(10 log10 n), at most n - 1; `acf` is R's estimator (denominator n, no n/(n - k) factor); pinned on `cli_stat_acf_int` (lags 1-5) and `stat_ts_acf_spline_t3` (n = 64, default lag.max 18) by `test_validate_cli_acf_pacf_int64` | `acf(x)`, `pacf(x)` |
| `stats ccf` | `--lag-max` floor(10 log10(n / 2)), at most n - 1 | `ccf(x, y)` |
| `stats box-test` | `--lag 1` | `Box.test(x)` |
| `stats pp-test` | `--lshort true` (short truncation lag); `--lshort false` the long one | `PP.test(x)` |
| `stats chisq-gof` | `--observed` (a categorical column, tallied) or `--counts` (one count per row), `--probs`, `--rescale-p` | `chisq.test(table(x))`, `chisq.test(x, p = , rescale.p = )` |
| `stats wilcoxon` | `V` (signed rank), `W` (rank sum); JSON `statistic_name` | `wilcox.test` |
| `stats fivenum` | Tukey's hinges, `lower_hinge`, `upper_hinge` | `fivenum` |
| `viz coefplot` | `--se hc1` (default) or `standard`; title and message state the level and the standard errors | `confint` with `sandwich::vcovHC` |

Before: acf and pacf used 20 lags and scaled acf by n/(n - k), ccf used 20 lags, `box-test` lag 10, `pp-test` the
long lag; `chisq-gof` tallied a column whatever it held and took no probabilities; coefficient plots used HC1
without saying so.

## What is pinned

| Test | Case | Keys | Tolerance |
|---|---|---|---|
| `test_validate_cli_chisq_gof_counts` | `cli_stat_chisq_gof_counts` | X-squared, df, p, expected counts and residuals on a counts column, uniform and with `--probs` | `Tol::stat/p_value(n)`; counts, df exact |
| `test_validate_cli_chisq_gof_observed_probs` | `stat_chisq_gof` | `p1_*`, `p2_*`: X-squared, p and expected counts of the tallied categorical column with two `--probs` | `Tol::stat/p_value(50)` |
| `test_validate_cli_ts_default_lags` | `cli_stat_ts_defaults` | default lag counts and values of acf, pacf and ccf on a short series | `Tol::stat(n)`; lag counts exact |
| `test_validate_cli_box_test_default_lag` | `cli_stat_ts_defaults` | Box-Pierce and Ljung-Box at the default lag 1, and Ljung-Box at `--lag 10`: statistic, df, p | `Tol::stat/p_value(n)` |
| `test_validate_cli_viz_coefplot_ci` | `cli_viz_coefplot_ci` | estimates (`coef`) and interval limits at 95% (HC1) and 90% (HC1 and classical) | `Tol::coef/se(n)` |

`test_cli_chisq_gof_refusals` compares no R value: missing and negative counts, a `--probs` of the wrong length,
probabilities that do not sum to 1 and neither or both of `--observed`/`--counts` are refused, naming the flags.


## Timing

Release builds of `p2a` (`--no-default-features --features spectral-analysis`) at the v0.5.0 base f80b0131 and at the merged head 156e744f, one process per invocation (`p2a --session s.json <command>`, the session's datasets loaded from CSV each time, so about 12-25 ms of every number is start-up and loading), pinned with `taskset -c 2`, `RAYON_NUM_THREADS=1`, `POLARS_MAX_THREADS=1`; per row and n, 9 rounds per side interleaved, each the median of 11 invocations after 3 warm-ups; the table shows the median of the medians, the ratio of the minima and the 1-minute load average at the row's start and end. Data: n rows of simulated noise data (head: 5 mixed columns; describe: 20 columns; acf/pacf/ccf/box/pp: AR(1) series; moran: n random points, k = 4; gsynth: n/20 units x 20 periods, at least 5 units; iv: 2 instruments; km: 4 groups; chisq-gof: 6 categories; coefplot and hac: 3 regressors, hac with 4 lags). Quiet re-time at the integration (2026-11-18, nothing else running): a separate 5-round run gave ratios within 0.05 of these, with absolute times about 6 ms lower on both sides. The default acf/pacf/ccf rows run R's longer default lags; at n = 10,000 the extra lags cost 1.06-1.09x (acf 20.87 ms base, 22.82 ms head), within the pre-approved ceilings of 1.5x / 2.0x (acf, pacf) and 1.3x / 1.8x (ccf) at n = 1,000 / 10,000; signed off 2026-11-17.

| Command | n | f80b0131 (ms) | head (ms) | head/base | minima | load |
|---|---:|---:|---:|---:|---:|---:|
| `data head -F json -n n` | 100 | 17.9 | 18.0 | 1.01x | 1.01x | 1.10-1.10 |
| `data head -F json -n n` | 1,000 | 21.6 | 20.5 | 0.95x | 0.94x | 1.10-1.25 |
| `data head -F json -n n` | 10,000 | 57.7 | 43.6 | 0.76x | 0.76x | 1.25-1.37 |
| `data describe -F json` | 100 | 18.2 | 18.5 | 1.01x | 1.01x | 1.37-1.42 |
| `data describe -F json` | 1,000 | 22.0 | 22.2 | 1.01x | 1.01x | 1.42-1.55 |
| `data describe -F json` | 10,000 | 34.6 | 35.1 | 1.01x | 1.03x | 1.55-1.66 |
| `stats acf default` | 100 | 18.4 | 18.6 | 1.01x | 1.03x | 1.66-1.61 |
| `stats acf default` | 1,000 | 18.9 | 19.2 | 1.02x | 1.03x | 1.61-1.64 |
| `stats acf default` | 10,000 | 20.9 | 22.8 | 1.09x | 1.09x | 1.64-1.59 |
| `stats acf --lag-max 20` | 100 | 18.7 | 19.0 | 1.01x | 1.04x | 1.59-1.59 |
| `stats acf --lag-max 20` | 1,000 | 19.2 | 19.5 | 1.02x | 1.01x | 1.59-1.62 |
| `stats acf --lag-max 20` | 10,000 | 21.3 | 22.4 | 1.05x | 1.04x | 1.62-1.49 |
| `stats pacf default` | 100 | 19.4 | 19.5 | 1.00x | 1.02x | 1.49-1.45 |
| `stats pacf default` | 1,000 | 19.8 | 20.2 | 1.02x | 1.03x | 1.45-1.50 |
| `stats pacf default` | 10,000 | 22.5 | 23.8 | 1.06x | 1.05x | 1.50-1.54 |
| `stats pacf --lag-max 20` | 100 | 19.7 | 20.1 | 1.02x | 1.03x | 1.54-1.49 |
| `stats pacf --lag-max 20` | 1,000 | 20.2 | 20.7 | 1.02x | 1.03x | 1.49-1.49 |
| `stats pacf --lag-max 20` | 10,000 | 22.8 | 23.2 | 1.02x | 1.01x | 1.49-1.45 |
| `stats ccf default` | 100 | 20.3 | 20.8 | 1.03x | 1.05x | 1.45-1.42 |
| `stats ccf default` | 1,000 | 20.8 | 21.5 | 1.03x | 1.03x | 1.42-1.38 |
| `stats ccf default` | 10,000 | 24.9 | 26.8 | 1.07x | 1.08x | 1.38-1.35 |
| `stats ccf --lag-max 20` | 100 | 20.9 | 21.3 | 1.02x | 1.06x | 1.35-1.32 |
| `stats ccf --lag-max 20` | 1,000 | 21.6 | 22.0 | 1.02x | 1.01x | 1.32-1.38 |
| `stats ccf --lag-max 20` | 10,000 | 25.5 | 25.8 | 1.01x | 1.03x | 1.38-1.43 |
| `stats box-test default` | 100 | 21.2 | 21.9 | 1.03x | 1.03x | 1.43-1.63 |
| `stats box-test default` | 1,000 | 21.7 | 22.4 | 1.03x | 1.02x | 1.63-1.58 |
| `stats box-test default` | 10,000 | 23.7 | 24.1 | 1.02x | 1.03x | 1.58-1.62 |
| `stats pp-test default` | 100 | 21.8 | 22.5 | 1.03x | 1.02x | 1.62-1.62 |
| `stats pp-test default` | 1,000 | 22.2 | 22.9 | 1.03x | 1.01x | 1.62-1.65 |
| `stats pp-test default` | 10,000 | 24.8 | 25.1 | 1.01x | 1.01x | 1.65-1.60 |
| `spatial moran default` | 100 | 22.9 | 23.8 | 1.04x | 1.04x | 1.60-1.55 |
| `spatial moran default` | 1,000 | 25.3 | 26.0 | 1.03x | 1.03x | 1.55-1.66 |
| `spatial moran default` | 10,000 | 46.1 | 47.0 | 1.02x | 1.02x | 1.66-1.80 |
| `stats chisq-gof --observed` | 100 | 25.6 | 26.0 | 1.02x | 1.03x | 1.71-1.81 |
| `stats chisq-gof --observed` | 1,000 | 25.6 | 26.2 | 1.02x | 1.00x | 1.81-1.75 |
| `stats chisq-gof --observed` | 10,000 | 26.9 | 27.6 | 1.03x | 1.04x | 1.75-1.77 |
| `viz coefplot` | 100 | 69.3 | 70.6 | 1.02x | 1.02x | 1.77-1.96 |
| `viz coefplot` | 1,000 | 71.0 | 72.3 | 1.02x | 1.03x | 1.96-2.03 |
| `viz coefplot` | 10,000 | 74.4 | 76.0 | 1.02x | 1.02x | 2.03-2.02 |

## Reproduce

```bash
Rscript validation/r/run.R --check
cargo test -p p2a-cli --no-default-features --features spectral-analysis --test tier2 wp21_stats_viz
```
