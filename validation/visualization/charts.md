# Static charts and table exporters

R code: `validation/r/cases/data_viz_t3.R` (cases `viz_scatter_cor`, `viz_boxplot_groups`,
`viz_histogram_bins`, `viz_residual_diag`, `export_ols_grunfeld`; regenerate with
`Rscript validation/r/run.R --write <id>`). Registry: `validation/methods/data_viz_t3.json`. R 4.3.3, base R
only. Code: `crates/p2a-core/src/visualization/charts.rs`, `crates/p2a-core/src/export/{latex,html,markdown}.rs`.

## Scatter-plot correlation (`scatter_plot().correlation`)

Pearson r as `stats::cor`: centred sums with R's corrected mean, deviations scaled by their largest
magnitude, None (R: NA) for a spread at or below 4 eps times the data's magnitude. Case `viz_scatter_cor`
(n = 200): r = 0.8828950961695764 (x), 0.8828950961459953 (x + 1.7e9), 0.8828950806679923 (x + 1e12),
0.8828950961695765 ((x, y) * 1e-9), NA (constant y). Tests `test_validate_scatter_correlation_large_offset`,
`test_validate_scatter_correlation_tiny_scale`; tolerance 1e-6 relative (measured 4e-16). The one-pass
formula used before gave 0.6902 and 0.0010 at the two offsets and None on the 1e-9 scale (DATA-61).

R-pinned with known deviations. A spread at or below 4 eps times the data's magnitude gives None (printed
NaN; R: NA), where the centred values are rounding noise; `stats::cor` returns NA only for an exactly zero
spread. Pairs with a non-finite value are dropped before r, as the plot drops them (like
`cor(use = "complete.obs")` for NA); `cor` returns NaN when a value is +-Inf.

## Box plot (`box_plot().statistics`)

`grDevices::boxplot.stats(x, coef = 1.5)` through `stats::robust::boxplot_stats`: `min`/`max` are the
whisker ends, `q1`/`q3` Tukey's hinges (fivenum), outliers in data order, `n` the non-NaN values. NaN is
dropped; +-Inf are counted and listed as outliers, as R; they are not drawn. A group without values keeps
its slot with n = 0 and NaN statistics, as R's `boxplot`. Case `viz_boxplot_groups`: e.g. `c(1, 7, 12, 15, 15, 16, 18, 18)`
-> 1, 9.5, 15, 17, 18, no outliers. Test `test_validate_boxplot_stats_chart`, exact.

