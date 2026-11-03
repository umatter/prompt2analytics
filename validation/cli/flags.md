# Validation: CLI Flags (boolean values, `--cluster2`, `--max-memory`)

## Method Overview

T3-WP06 made the following CLI options do what their help says (CLI-5, CLI-18, DATA-30).

**Boolean flags.** Each takes a value: `true`/`false`, `TRUE`/`FALSE`, `T`/`F`, `yes`/`no`,
`on`/`off`, `1`/`0` (case-insensitive; clap's `BoolishValueParser`), as `--flag false` or
`--flag=false`. A bare `--flag` means `true`, as recorded sessions contain it. Help shows
`--flag [<BOOL>]`. A positional placed directly after a bare flag is read as its value.

| Flag | Commands | Default | What `false` / `true` does |
|---|---|---|---|
| `--intercept` | `reg ols`, `clustered`, `quantreg`, `gls`, `step`, `hac` | true | false: regression through the origin, R² uncentered (labels `R-squared (uncentered)`, JSON `r_squared_type: "uncentered"`) |
| `--use-bic` | `reg step` | false | true: BIC, k = log n in `extractAIC` |
| `--comments` | `script export` | true | false: no comment line above each recorded command (the header and the cleanup comment stay) |
| `--aic` | `ts ar` | true | false: fits the maximum order min(n − 1, floor(10 log10 n)), as `ar(aic = FALSE)`; `--order` fixes the order and wins over both |
| `--detrend` | `ts spectrum` | true | false: no least-squares linear trend removed first |
| `--robust` | `ts stl` | false | true: robust (outlier-resistant) STL |

Before: a `bool` field with `default_value = "true"` was a clap `SetTrue` flag, so `--intercept
false` was a usage error (exit 2) and the flag could never be false; `--robust true`, the help
example of `ts stl`, failed the same way. `reg clustered`, `reg quantreg` and `reg hac` ignored
the intercept setting even where it parsed. The bin test `no_boolean_flag_is_stuck_at_true`
fails if any `SetTrue` flag defaults to true.

**`reg clustered --cluster2 <COLUMN>`**: two-way cluster-robust SEs,
`sandwich::vcovCL(cluster = ~c1 + c2, type = "HC1")`. The method reads `Clustered OLS
(clusters: firm, year)`, the SE line `clustered by firm and year (two-way CR1, G = 10 and 20)`,
JSON `se_type` `cluster:firm+year`, `n_clusters` and `n_clusters_2`. The estimator is
documented in [../regression/ols_clustered.md](../regression/ols_clustered.md).

**`--max-memory <MB>`** (global; MiB; 0 = no limit, the default): with `--session`, the
estimated memory of the session's datasets (`p2a_core::memory::estimate_dataset_memory`: bytes
per value by dtype, plus 10 %) is counted when the session file's datasets are reloaded and
whenever a command stores a dataset (a dataset replaced under the same name is counted once).
When the total exceeds the limit, one line goes to stderr, at most once per invocation:
`Warning: estimated dataset memory 1.17 MB exceeds the limit of 1.00 MB (--max-memory 1)`.
Nothing is refused. Without `--session` no dataset is kept, and a non-zero value prints
`Warning: --max-memory has no effect without --session (no datasets are kept)`.

## Reference Implementation

R 4.3.3, sandwich 3.1.3, cases in [`validation/r/cases/cli_flags.R`](../r/cases/cli_flags.R):

- `ols_no_intercept_grunfeld`: `lm(inv ~ 0 + value + capital)` and its HC1, CR1, Newey-West,
  `step` and `extractAIC` values ([../regression/no_intercept.md](../regression/no_intercept.md)).
- `ols_twoway_cluster_grunfeld`: t and p of `vcovCL(lm(inv ~ value + capital), cluster = ~firm +
  year, type = "HC1")` (t −2.1664077296564725, 7.0485576671437791, 2.900040730052512; p
  0.031480735322878178, 2.9731400355190913e-11, 0.0041548694769250822; G 10 and 20). The SEs
  are `se_cluster_firm_year_hc1` of `ols_grunfeld`.
- `ts_ar_yw_aic`: `ar(method = "yule-walker")` with `aic = TRUE` (order 2), `aic = FALSE`
  (order 20 = floor(10 log10 120)) and `aic = FALSE, order.max = 2`, on the simulated AR(2)
  column `y_ar` of `generated/ts_flags_sim.csv` (seed 20261022, n = 120). The column `y_seas`
  (trend, monthly season, outliers at t = 6 and 75) feeds the STL and spectrum tests.

## Rust tests

Binary `tier3`, module `wp06_flags` (`cargo test -p p2a-cli --no-default-features --features
spectral-analysis --test tier3`):

| Test | Case | Checks | Tolerance |
|---|---|---|---|
| `test_validate_cli_ols_no_intercept` | `ols_no_intercept_grunfeld` | text table (HC1) and uncentered R² labels; JSON with `--intercept=FALSE --robust standard`; bare `--intercept` equals the default | text `Tol::printed` (6 decimals for coefficients, SE, R²; 3 for t; 4 for p), JSON README |
| `test_validate_cli_clustered_two_way` | `ols_grunfeld`, `ols_twoway_cluster_grunfeld` | `--cluster2 year`: coefficients, SE, t, p, labels, cluster counts; one-way unchanged | as above |
| `test_validate_cli_clustered_no_intercept` | `ols_no_intercept_grunfeld` | one- and two-way CR1 without intercept, R² and adjusted R² | README |
| `test_validate_cli_hac_no_intercept` | `ols_no_intercept_grunfeld` | `--lags 4 --intercept false`: names, coefficients, Newey-West SE, t, p, R² | README |
| `test_validate_cli_gls_identity_no_intercept` | `ols_no_intercept_grunfeld` | identity GLS: names, coefficients, SE, t, p, R², adjusted R², sigma | README |
| `test_validate_cli_step_no_intercept` | `ols_no_intercept_grunfeld` | backward step keeps both terms; `extractAIC` (k = 2 and log n), final R² and AIC; bare `--use-bic` | README |
| `test_cli_quantreg_no_intercept_reaches_core` | `ols_no_intercept_grunfeld` | `--intercept false` reaches `quantreg()`: CLI coefficients equal the in-process call | `Tol::new(1e-12, 1e-15)` |
| `test_validate_cli_ts_ar_aic_flag` | `ts_ar_yw_aic` | AIC order, coefficients, relative AIC and mean; `--aic false` order 20 and its coefficients; `--order 2 --aic false` | README, n = 120; orders exact |
| `test_cli_ts_stl_robust_flag` | `ts_ar_yw_aic` | `--robust true/false`, bare and default; remainder equals `run_stl` in process | exact |
| `test_cli_ts_spectrum_detrend_flag` | `ts_ar_yw_aic` | `--detrend false` equals `run_spectrum` in process and differs from the default | exact |
| `test_cli_script_export_comments_flag` | none | `--comments false` drops only the per-command comment lines | exact |
| `test_cli_max_memory_warns_once` | none | one warning per invocation, values 1.17 and 2.35 MB, none without a limit or under it | exact |
| `test_cli_max_memory_without_session` | none | the no-session note | exact |
| `test_tier3_cli_helpers` | none | the tier3 table helpers | `Tol::printed(6)` |

Unit tests in the `p2a` binary: `bool_flag_tests::no_boolean_flag_is_stuck_at_true`,
`bool_flag_tests::bool_flags_take_true_false_and_r_spellings`,
`session::memory_limit_tests::memory_limit_warns_once_and_counts_replacements_once`,
`session::memory_limit_tests::memory_limit_zero_never_warns`,
`commands::script::tests::render_script_without_comments_omits_command_lines`,
`cli_definition_tests::max_memory_help_says_what_it_does`; in p2a-core
`memory::tests::test_near_limit_does_not_overflow_at_usize_max` and
`memory::tests::test_take_warning_once_per_crossing`.

## Status

R-pinned (registry `validation/methods/cli_flags.json`, ids `cli_regression_flags` and
`cli_timeseries_flags`), with the known deviations below.

## Known Differences

- **TS-15** (`var_pred` of `ts ar`): p2a's Yule-Walker prediction variance omits `ar.yw`'s
  n/(n − (order + 1)) factor: 1.012860245443493 (order 2) and 0.9090641227233529 (order 20)
  against R's 1.0388310209676848 and 1.1018959063313365. Not pinned.
- **TS-26** (`ts stl`): STL runs in f32 with p2a's default windows; R's `stl` needs `s.window`
  and differs. The test compares the CLI with `run_stl` in process.
- **STAT-68** (`ts spectrum`): the raw periodogram is twice `spec.pgram`'s, with a different
  taper. The test compares the CLI with `run_spectrum` in process.
- **REG-52** (`reg quantreg`): p2a's quantile regression is not the exact LP of
  `rq(method = "br")`; R's no-intercept coefficients are recorded (`rq_coef`), not pinned.
- `reg step --intercept false --direction forward` without `--required-vars` fails ("No
  independent variables specified"): p2a cannot fit the empty model y ~ 0 that R's `step`
  starts from.

## References

- clap 4.5 `BoolishValueParser`, `Arg::default_missing_value`.
- Cameron, A. C., Gelbach, J. B. and Miller, D. L. (2011). Robust Inference With Multiway
  Clustering. *Journal of Business & Economic Statistics*, 29(2), 238-249.
