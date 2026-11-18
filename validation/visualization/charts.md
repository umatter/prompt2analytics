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

Through the MCP path, `viz_boxplot` prints each group's statistics at four decimals; the tier-3 test
`test_validate_mcp_boxplot_groups` (registry `data_viz_t4.json`, `mcp_viz_boxplot_stats`) filters each
group of the case's dataset with `munge_filter` and pins n, the five statistics (printed precision) and the
outliers against the same case; a box plot whose only column has no values is refused. The chart tools
read their columns with `data::columns`: a row missing a plotted value is left out and counted in a
`Missing values:` line (each line-chart series and each box-plot column on its own).

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

## Correlation heatmap (`correlation_heatmap`, `viz_heatmap`)

Not R-pinned: the correlations are `stats::correlation_matrix` (R's `cor(use = "pairwise.complete.obs")`,
pinned there); the chart only lays them out. Cells are square and shrink to fit the image beside a reserved
colour-bar band; below 10 px the labels are hidden (the result's `warnings` say so) and below 4 px the matrix is
refused, so at the default 800 x 600 at most 127 variables fit (`heatmap_capacity`). Column labels are
rotated. Tests: `test_heatmap_layout_fits_the_canvas` (k = 2 to 127, every band inside the canvas; k = 128
refused) and `test_heatmap_draws_every_cell` (the PNG's last cell has its colour). `viz_heatmap` refuses unknown
and non-numeric names (listing the numeric ones), refuses more variables than fit, correlates only the selected
columns and ends with the pairwise-complete sentence of `CorrelationMatrix` when a pair dropped rows.

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

## Timing (against v0.5.0)

Release builds of `f80b0131` (base) and this branch (head), p2a-core `ci` and p2a-mcp
`http,llm,auth,websocket`, one harness crate per side calling the tools through
`AnalyticsServer::call_tool_with_session` in process (and the core `histogram` and
`correlation_heatmap` directly). Nine interleaved processes per side, `taskset -c 2`,
`RAYON_NUM_THREADS=1`, `POLARS_MAX_THREADS=1`; each value is the median over the nine processes of the
per-process median of 61 calls (21 at n = 10,000) after 5 warm-ups; "minima" is the ratio of the fastest
processes. Datasets: Int64 and f64 columns with 2% missing cells; the heatmap rows use 10 and 40 numeric
columns of 1,000 rows. Quiet re-time at the integration (2026-11-18, merged head 156e744f, nothing else
running), load average 1.4-2.3.

| Workload | n | base (ms) | head (ms) | head/base | minima |
|---|---|---|---|---|---|
| `viz_histogram` | 100 | 3.43 | 3.48 | 1.01x | 1.02x |
| `viz_scatter` | 100 | 5.07 | 5.16 | 1.02x | 1.01x |
| `viz_boxplot` | 100 | 3.42 | 3.49 | 1.02x | 1.02x |
| `viz_line` | 100 | 5.12 | 5.22 | 1.02x | 1.02x |
| `viz_event_study` | 100 | 5.41 | 5.55 | 1.03x | 1.03x |
| `viz_coefficient` | 100 | 5.51 | 5.64 | 1.02x | 1.03x |
| `viz_irf` | 100 | 5.44 | 5.55 | 1.02x | 1.02x |
| `viz_residual_diagnostics` | 100 | 16.36 | 16.70 | 1.02x | 1.02x |
| `viz_scatter_interactive` | 100 | 0.20 | 0.20 | 1.03x | 1.06x |
| `viz_scatter_interactive_grouped` | 100 | 0.57 | 0.21 | 0.38x | 0.38x |
| `viz_histogram_interactive` | 100 | 0.19 | 0.19 | 0.97x | 0.97x |
| `viz_line_interactive` | 100 | 0.20 | 0.20 | 0.98x | 0.98x |
| `core_histogram` | 100 | 3.50 | 3.55 | 1.01x | 1.01x |
| `viz_histogram` | 1000 | 4.16 | 4.23 | 1.02x | 1.02x |
| `viz_scatter` | 1000 | 6.66 | 6.78 | 1.02x | 1.02x |
| `viz_boxplot` | 1000 | 3.40 | 3.46 | 1.02x | 1.02x |
| `viz_line` | 1000 | 5.59 | 5.72 | 1.02x | 1.02x |
| `viz_event_study` | 1000 | 6.55 | 6.74 | 1.03x | 1.03x |
| `viz_coefficient` | 1000 | 17.10 | 17.47 | 1.02x | 1.01x |
| `viz_irf` | 1000 | 6.12 | 6.27 | 1.03x | 1.03x |
| `viz_residual_diagnostics` | 1000 | 26.29 | 25.22 | 0.96x | 1.02x |
| `viz_scatter_interactive` | 1000 | 0.37 | 0.39 | 1.04x | 1.04x |
| `viz_scatter_interactive_grouped` | 1000 | 0.80 | 0.44 | 0.55x | 0.54x |
| `viz_histogram_interactive` | 1000 | 0.29 | 0.30 | 1.02x | 1.00x |
| `viz_line_interactive` | 1000 | 0.37 | 0.38 | 1.04x | 1.03x |
| `core_histogram` | 1000 | 4.34 | 4.39 | 1.01x | 1.01x |
| `viz_histogram` | 10000 | 5.27 | 5.49 | 1.04x | 1.04x |
| `viz_scatter` | 10000 | 20.81 | 21.55 | 1.04x | 1.04x |
| `viz_boxplot` | 10000 | 4.67 | 5.18 | 1.11x | 1.11x |
| `viz_line` | 10000 | 11.78 | 10.99 | 0.93x | 1.06x |
| `viz_event_study` | 10000 | 15.67 | 16.35 | 1.04x | 1.15x |
| `viz_coefficient` | 10000 | 132.17 | 133.78 | 1.01x | 1.01x |
| `viz_irf` | 10000 | 14.06 | 14.96 | 1.06x | 1.06x |
| `viz_residual_diagnostics` | 10000 | 185.63 | 186.84 | 1.01x | 1.01x |
| `viz_scatter_interactive` | 10000 | 2.07 | 2.30 | 1.11x | 1.12x |
| `viz_scatter_interactive_grouped` | 10000 | 3.03 | 2.65 | 0.87x | 0.87x |
| `viz_histogram_interactive` | 10000 | 1.24 | 1.37 | 1.11x | 1.10x |
| `viz_line_interactive` | 10000 | 2.10 | 2.29 | 1.09x | 1.08x |
| `core_histogram` | 10000 | 4.64 | 4.70 | 1.01x | 1.01x |
| `viz_heatmap_k10` | 1000 | 6.22 | 6.11 | 0.98x | 0.98x |
| `viz_heatmap_k40_sel10` | 1000 | 12.71 | 6.22 | 0.49x | 0.49x |
| `core_correlation_heatmap_k10` | - | 6.18 | 6.48 | 1.05x | 1.05x |
| `core_correlation_heatmap_k40` | - | 6.24 | 7.38 | 1.18x | 1.18x |

Head/base above 1.10x:
- `viz_scatter_interactive` and `viz_histogram_interactive` at n = 10,000 (1.11x, 2.07 to 2.30 ms and
  1.24 to 1.37 ms; `viz_line_interactive` 1.09x): the figure now carries every value as a
  double in its shortest round-trip digits (DATA-64) instead of single precision, so the page and its JSON
  are about twice as long; the grouped scatter is faster (0.38x to 0.87x) because groups are coded once.
- `core_correlation_heatmap_k40` (1.18x, 6.24 to 7.38 ms): the head draws all 1,600 cells, their borders, the
  rotated labels and the colour bar inside the image; the base drew 20-pixel cells past the edge, where
  most of the matrix and the colour bar were clipped away (DATA-63). `viz_heatmap` with 10 of 40 columns
  selected is 0.49x, since only the selection is correlated.
- `viz_boxplot` at n = 10,000 (1.11x, 4.67 to 5.18 ms; 1.02x below): not profiled; the tool now reads its
  columns with their missing values counted.

Rule 5: the interactive charts, the 40-variable heatmap and the box plot at 10,000 rows were in no earlier
sign-off: signed off by the user on 2026-11-18 at these figures.
