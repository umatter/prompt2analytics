# Static charts and table exporters

R code: `validation/r/cases/data_viz_t3.R` (cases `viz_scatter_cor`, `viz_boxplot_groups`,
`viz_histogram_bins`, `viz_residual_diag`, `export_ols_grunfeld`) and `validation/r/cases/data_viz_t4.R` (case
`export_ols_grunfeld_usd`); regenerate with `Rscript validation/r/run.R --write <id>`. Registries:
`validation/methods/data_viz_t3.json`, `validation/methods/data_viz_t4.json`. R 4.3.3, base R only. Code:
`crates/p2a-core/src/visualization/charts.rs`, `crates/p2a-core/src/export/{csv,latex,html,markdown}.rs`.

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
-> 1, 9.5, 15, 17, 18, no outliers. Test `test_validate_boxplot_stats_chart`: counts exact, statistics and outliers at the statistics tolerance.

## Histogram (`histogram()`)

Default bin count `nclass.Sturges` = ceiling(log2(n) + 1). Bins have equal width from min to max,
right-closed with the lowest value included, and values within 1e-7 bin widths of a break are assigned as
R's `hist()` does: `hist(x, breaks = seq(min(x), max(x), length.out = k + 1))` with its defaults.
R-pinned with known deviations: R's default `hist()` passes the Sturges count to `pretty()` (n = 100: 10
bins where p2a has 8); p2a does not. Constant data (range at most 4 eps times
the magnitude) get k bins of width 1 starting at the value. Case `viz_histogram_bins`: k = 1, 4, 4, 5, 8,
11, 11 for n = 1, 5, 8, 16, 100, 1000, 1024; counts `4 7 9 13 31 17 12 7` (n = 100), `3 6 10 14 9 4 4`
(n = 50 on a 1e-11 scale) and `3 3 2 3 2` (a decimal grid with values on the breaks). Test `test_validate_histogram_sturges_and_counts`: bin number and counts exact, breaks at the statistics tolerance.

## Residual diagnostics (`residual_diagnostics`, `residual_diagnostics_with`)

R's `plot.lm` quantities: `rstandard` r_i = e_i / (s sqrt(1 - h_i)) with s^2 = sum e^2 / (n - p) and
`cooks.distance` D_i = r_i^2 h_i / (p (1 - h_i)); leverage one (h >= 1 - 10 eps) gives NaN and is not
plotted. p is `n_params`, or the rounded sum of the hat values (= the model rank). Without hat values the
one-predictor formula 1/n + (f - mean f)^2 / sum (f - mean f)^2 is used (exact for a model with one
regressor and an intercept) and a warning says so. Q-Q abscissae: `qnorm(ppoints(n))`. Case
`viz_residual_diag` (mtcars): lm(mpg ~ wt + hp + qsec) with R's hat values: p = 4, sigma =
2.5777848825208176, max Cook's distance 0.3328494305479045 (observation 16, not > 0.5); lm(mpg ~ wt)
without leverage: 0.5319056255997687 at observation 16. Tests `test_validate_cooks_distance_rstandard`
(1e-5 relative; measured 7.4e-15), `test_validate_qq_theoretical_quantiles` (1e-11),
`test_residual_diagnostics_rejects_bad_leverage`.

## Regression-table exporters

`LatexTableBuilder`, `HtmlTableBuilder` and `MarkdownTableBuilder` for lm(inv ~ value + capital) and
lm(inv ~ value) on Grunfeld (case `export_ols_grunfeld`): coefficients and SEs printed at 4 decimals equal
R's rounded values, stars follow R's p-values, N/R2/adj. R2/F as summary.lm. Tests
`test_validate_latex_table_ols_grunfeld`, `test_validate_html_table_ols_grunfeld` and
`test_validate_markdown_table_ols_grunfeld` (printed precision); `test_latex_table_golden_grunfeld`
compares the whole output byte for byte with
`crates/p2a-core/tests/golden/export_latex_ols_grunfeld.tex`, written from R's values. A LaTeX table with
notes opens with `% Requires \usepackage{threeparttable}` and puts caption, tabular and notes in a
`threeparttable`; its legend is built from the star mapping (`\textdagger{}` p < 0.10, * p < 0.05, ** p < 0.01,
*** p < 0.001) with `$p<0.05$` in math mode, and the HTML and Markdown legends come from the same mapping
(`test_latex_notes_structure`, `test_html_and_markdown_legends`). "Standard errors in parentheses" appears only
when the SEs are. The ignored test `test_latex_table_compiles_with_pdflatex` compiles the golden with
`threeparttable` loaded and checks the log and the text for undefined commands and inverted exclamation marks.
Undefined values print as `NA`.

### Numbers that keep their digits

CSV cells are written with the shortest digits that read back as the same double (scientific notation below
1e-5 and from 1e15 on in magnitude), with `NA`, `Inf` and `-Inf` as R's `read.csv` reads them; they were six
fixed decimals, so a coefficient of 1e-7 and its SE were written as 0 (DATA-77). In the LaTeX, HTML and
Markdown tables an estimate, SE or marginal-effect cell that would round to zero at the style's decimals is
written with that many significant digits in R's exponent form (`1.156e-07`, as R's `format(x, digits = 4)`;
trailing zeros are kept, which R trims); other cells keep their fixed decimals. Case `export_ols_grunfeld_usd`:
lm(inv ~ value_usd + capital) on Grunfeld with `value_usd = value * 1e6`, coefficient 1.1556215636055214e-07,
SE 5.8357095572206334e-09. Tests `test_validate_csv_ols_grunfeld_usd` (coefficients, SEs, t, p, confint
bounds, R2, adj. R2, F and sigma read back from the CSV, at the validation/README.md tolerances for n = 200)
and `test_validate_latex_small_coefficient_grunfeld_usd` (`1.156e-07***` and `(5.836e-09)`, 5e-4 relative).
