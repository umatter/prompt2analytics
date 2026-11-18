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

Release builds of `p2a` (`--no-default-features --features spectral-analysis`) at the v0.5.0 base f80b0131 and at the lane head, one process per invocation (`p2a --session s.json <command>`, the session's datasets loaded from CSV each time, so about 20-40 ms of every number is start-up and loading), pinned with `taskset -c 10`, `RAYON_NUM_THREADS=1`; per row and n, 9 rounds per side interleaved, each the median of 21 invocations after 3 warm-ups; the table shows the median of the medians and the 1-minute load average of the (shared, busy) machine. Data: n rows of simulated noise data (head: 5 mixed columns; describe: 20 columns; acf/pacf/ccf/box/pp: AR(1) series; moran: n random points, k = 4; gsynth: n/20 units x 20 periods, at least 5 units; iv: 2 instruments; km: 4 groups; chisq-gof: 6 categories; coefplot and hac: 3 regressors, hac with 4 lags). Lane timing, preliminary until the integration re-times it. The default acf/pacf/ccf rows run R's longer default lags; at these sizes the extra lags cost no measurable time.

| Command | n | f80b0131 (ms) | lane head (ms) | head/base | load |
|---|---:|---:|---:|---:|---:|
| `data head -F json -n n` | 100 | 25.0 | 23.5 | 0.94x | 43 |
| `data head -F json -n n` | 1,000 | 30.1 | 26.4 | 0.88x | 42 |
| `data head -F json -n n` | 10,000 | 96.1 | 71.4 | 0.74x | 38 |
| `data describe -F json` | 100 | 23.1 | 22.5 | 0.98x | 37 |
| `data describe -F json` | 1,000 | 29.9 | 28.9 | 0.97x | 36 |
| `data describe -F json` | 10,000 | 51.0 | 51.1 | 1.00x | 36 |
| `stats acf default` | 100 | 27.1 | 26.2 | 0.97x | 36 |
| `stats acf default` | 1,000 | 28.4 | 28.4 | 1.00x | 36 |
| `stats acf default` | 10,000 | 30.2 | 32.0 | 1.06x | 37 |
| `stats acf --lag-max 20` | 100 | 27.2 | 25.8 | 0.95x | 37 |
| `stats acf --lag-max 20` | 1,000 | 29.1 | 27.6 | 0.95x | 38 |
| `stats acf --lag-max 20` | 10,000 | 31.0 | 31.1 | 1.00x | 39 |
| `stats pacf default` | 100 | 26.4 | 25.4 | 0.96x | 39 |
| `stats pacf default` | 1,000 | 29.5 | 27.9 | 0.95x | 40 |
| `stats pacf default` | 10,000 | 34.4 | 34.0 | 0.99x | 41 |
| `stats pacf --lag-max 20` | 100 | 29.7 | 27.9 | 0.94x | 41 |
| `stats pacf --lag-max 20` | 1,000 | 30.1 | 30.6 | 1.02x | 41 |
| `stats pacf --lag-max 20` | 10,000 | 35.0 | 36.3 | 1.04x | 40 |
| `stats ccf default` | 100 | 29.0 | 28.2 | 0.97x | 39 |
| `stats ccf default` | 1,000 | 29.9 | 31.1 | 1.04x | 38 |
| `stats ccf default` | 10,000 | 33.8 | 35.9 | 1.06x | 39 |
| `stats ccf --lag-max 20` | 100 | 31.2 | 29.9 | 0.96x | 40 |
| `stats ccf --lag-max 20` | 1,000 | 32.5 | 32.0 | 0.98x | 40 |
| `stats ccf --lag-max 20` | 10,000 | 38.0 | 39.3 | 1.03x | 41 |
| `stats box-test default` | 100 | 32.5 | 29.8 | 0.92x | 42 |
| `stats box-test default` | 1,000 | 38.1 | 36.1 | 0.95x | 45 |
| `stats box-test default` | 10,000 | 36.9 | 34.0 | 0.92x | 47 |
| `stats pp-test default` | 100 | 37.4 | 34.9 | 0.93x | 49 |
| `stats pp-test default` | 1,000 | 40.0 | 35.0 | 0.88x | 50 |
| `stats pp-test default` | 10,000 | 38.9 | 39.1 | 1.01x | 50 |
| `spatial moran default` | 100 | 34.1 | 32.7 | 0.96x | 50 |
| `spatial moran default` | 1,000 | 47.9 | 50.1 | 1.05x | 53 |
| `spatial moran default` | 10,000 | 78.9 | 76.1 | 0.96x | 53 |
| `stats chisq-gof --observed` | 100 | 39.9 | 39.6 | 0.99x | 26 |
| `stats chisq-gof --observed` | 1,000 | 37.8 | 37.4 | 0.99x | 24 |
| `stats chisq-gof --observed` | 10,000 | 42.9 | 42.1 | 0.98x | 24 |
| `viz coefplot` | 100 | 120.0 | 118.9 | 0.99x | 35 |
| `viz coefplot` | 1,000 | 114.9 | 114.4 | 1.00x | 34 |
| `viz coefplot` | 10,000 | 117.0 | 120.0 | 1.03x | 32 |

## Reproduce

```bash
Rscript validation/r/run.R --check
cargo test -p p2a-cli --no-default-features --features spectral-analysis --test tier2 wp21_stats_viz
```
