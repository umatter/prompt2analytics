# Validation: CLI `stats` defaults, chi-squared counts and coefficient plots

## Method Overview

The `stats` commands follow R's defaults and report R's statistics:

| Command | Default or output | R reference |
|---|---|---|
| `stats acf`, `pacf` | `--lag-max` floor(10 log10 n), at most n - 1; `acf` is R's estimator (denominator n, no n/(n - k) factor) | `acf(x)`, `pacf(x)` |
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
| `test_validate_cli_viz_coefplot_ci` | `cli_viz_coefplot_ci` | interval limits at 95% (HC1) and 90% (HC1 and classical) | `Tol::coef/se(n)` |

`test_cli_chisq_gof_refusals` compares no R value: missing and negative counts, a `--probs` of the wrong length,
probabilities that do not sum to 1 and neither or both of `--observed`/`--counts` are refused, naming the flags.

## Reproduce

```bash
Rscript validation/r/run.R --check
cargo test -p p2a-cli --no-default-features --features spectral-analysis --test tier2 wp21_stats_viz
```
