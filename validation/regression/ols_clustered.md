# Validation: Clustered Standard Errors

## Method Overview

`run_ols_clustered(dataset, y, x, cluster1, cluster2)` fits OLS with an intercept and
replaces the classical covariance by a cluster-robust one: one-way when `cluster2` is
`None`, two-way (Cameron, Gelbach and Miller 2011) otherwise. Coefficients and fit
statistics are those of `run_ols` (see [ols.md](ols.md)); `p2a reg clustered` prints them.

**One-way (CR1)**, with n observations, k coefficients and G clusters:
```
V̂ = G/(G − 1) · (n − 1)/(n − k) · (X'X)⁻¹ (Σ_g X_g'û_g û_g'X_g) (X'X)⁻¹
```

**Two-way**: V̂ = V̂₁ + V̂₂ − V̂₁₂, where V̂₁ and V̂₂ cluster on each dimension and V̂₁₂ on
their intersection, each scaled by its own G/(G − 1) and by (n − 1)/(n − k). This is
`sandwich::vcovCL(fit, cluster = ~g1 + g2, type = "HC1")` with its default
`multi0 = FALSE`. With `multi0 = TRUE` R gives 19.767577058817910 / 0.016416356538547092 /
0.079768017252114673 on Grunfeld; that is **not** what p2a computes, so the reference does
not pass it.

t, p and the 95% CI use the clustered SE with n − k degrees of freedom, as
`lmtest::coeftest(fit, vcov = vcovCL(...))` does.

**Wald F.** The result's `wald_f` is the overall Wald test of the slopes with the clustered
covariance (`lmtest::waldtest(fit, vcov = vcovCL(...), test = "F")`, n − k denominator
degrees of freedom), printed as "F-statistic (Wald, clustered): ..." or "(Wald, two-way
clustered)" before the classical F, which is labelled "F-statistic (classical): ...".

**Indefinite two-way covariance.** V̂₁ + V̂₂ − V̂₁₂ need not be positive semi-definite. When
its smallest eigenvalue is negative, p2a adds a warning naming it; a negative variance then
gives a NaN standard error and no Wald F is reported. `OlsClusteredResult::fit_with(..,
ClusteredOptions { fix: true })` sets the negative eigenvalues to 0 and rebuilds the
covariance, as `vcovCL(fix = TRUE)` (case `ols_twoway_indefinite`,
`test_validate_ols_twoway_indefinite`). One-way clustering is positive semi-definite by
construction and is never changed.

## Reference Implementation

`sandwich::vcovCL(m, cluster = ~firm | ~year | ~firm + year, type = "HC1")`, sandwich
3.1.3 (R 4.3.3), in the case `ols_grunfeld` of `validation/r/cases/regression.R`
([source](../r/cases/regression.R)): `lm(inv ~ value + capital)` on
`validation/datasets/grunfeld.csv` (n = 200, 10 firms, 20 years).

| SE | (Intercept) | value | capital |
|---|---|---|---|
| clustered by firm | 20.425202928473897 | 0.015894336687058763 | 0.08496711263554008 |
| clustered by year | 10.272890949869057 | 0.007909493497278841 | 0.03867233748282798 |
| two-way firm + year | 19.716680683803034 | 0.016395149450111585 | 0.07954318928748033 |

p-values clustered by firm: 0.03778713089939461, 8.223158574063165e-12,
0.007217873610888879. The case also holds `se_cluster_firm_hc0` (`vcovCL(type = "HC0")`,
20.322304526199204 / 0.015814263952602135 / 0.08453906400529106), which applies
G/(G − 1) without (n − 1)/(n − k); p2a has no such option, so it is not pinned.

## Tests

In `crates/p2a-core/src/regression/ols.rs` (registry entry `ols_clustered` in
`validation/methods/regression.json`):

| Test | Clusters | Pins |
|---|---|---|
| `test_validate_ols_clustered_grunfeld_firm` | firm (10) | coefficients, SE, p, and the adjusted R² through `LinearEstimator` (what `p2a reg clustered` prints) |
| `test_validate_ols_clustered_grunfeld_year` | year (20) | coefficients, SE |
| `test_validate_ols_clustered_grunfeld_firm_year` | firm (10) and year (20) | coefficients, SE |

Each also asserts the cluster counts. Tolerances (`validation/README.md`, n = 200,
relative): coefficients 1e-8, SE and statistics 1e-6, p-values 1e-6 or 1e-12 absolute.

Measured agreement: SEs within 5.9e-15 (firm), 5.5e-15 (year) and 6.8e-15 (firm + year)
relative; firm p-values within 7.3e-15 absolute. The cluster meat is summed in parallel,
so the last digits can vary between runs.

Dropping the (n − 1)/(n − k) factor from the CR1 scaling makes all three tests fail (the
firm SEs then equal `se_cluster_firm_hc0`).

`test_clustered_se` (string cluster ids, cluster count) is a property test, not pinned to R.

## Missing Cluster and Time Identifiers

`run_ols_clustered` reads cluster ids as factors (`p2a_core::data::columns`: strings,
integers, floats, booleans, dates). A row whose id is missing (null, NaN or an empty CSV
cell), in either dimension of a two-way clustering, is dropped and the model is
re-estimated on the remaining rows. That is what `fixest::feols(cluster = )` does ("NOTE: 3
observations removed because of NA values (vcov: 3)"); `sandwich::vcovCL` refuses missing
clusters, so the reference is `lm` + `vcovCL(type = "HC1")` on the rows that have an id.
The drop is reported in `OlsResult::missing` (JSON key `missing`, present only when rows
were dropped) and as the first warning, `3 of 200 rows dropped because of missing values
(firm_label: 3)`; `p2a reg clustered` prints it in its `Warnings:` block and adds the
`missing` object to `-F json`. A missing y or regressor in a kept row is still an error
(`NullValues`), as in `run_ols`. Without a missing id the code path and the output are
those of the section above.

R code: case `reg_cluster_na` of `validation/r/cases/regression_na.R`
([source](../r/cases/regression_na.R)), on
`validation/datasets/generated/grunfeld_cluster_na.csv`: Grunfeld with `firm_label`
(`firm_01` ... `firm_10`) missing in rows 3, 44 and 101 and `year_na` (the year) missing in
rows 10 and 190 (1-based). R 4.3.3, sandwich 3.1.3, fixest 0.14.2.

| Quantity | (Intercept) | value | capital |
|---|---|---|---|
| coefficients (197 rows with a firm label) | -42.61125157678997 | 0.12051018457056514 | 0.21788811575756084 |
| SE CR1 by `firm_label` (`vcovCL`, HC1) | 19.96126646444534 | 0.01656539894414541 | 0.08187544348335857 |
| SE `feols(cluster = ~firm_label)` | 19.96126646444534 | 0.01656539894414548 | 0.08187544348335866 |
| coefficients (195 rows with both ids) | -42.96779261498538 | 0.12028550976765875 | 0.2190107999499585 |
| SE two-way CR1, `firm_label` + `year_na` | 19.27067533979663 | 0.01811945502433293 | 0.07822181957790249 |

Rows used 197 (3 dropped) one-way and 195 two-way; G = 10 firm labels and 20 years in the
kept rows. Before the fix the three missing labels were pooled into an eleventh cluster:
all 200 rows, SE 20.30368 / 0.014981 / 0.077620. fixest's two-way SEs (19.0498 / ...) use
another small-G adjustment and are not the reference.

Tests (registry entry `ols_clustered_missing_ids` in `validation/methods/regression_na.json`):

| Test | Path | Pins |
|---|---|---|
| `test_validate_reg_cluster_na_fixture` | `p2a-core` `regression/ols.rs` | the dtypes and missing cells polars reads; R's row counts |
| `test_validate_ols_cluster_na` | `run_ols_clustered` | coefficients and SEs one- and two-way, rows used and dropped, cluster counts, the warning (printed once), no `missing` key without missing ids, `ColumnNotFound` for an absent id, `InsufficientData` naming the method and the id columns when no row has an id |
| `test_validate_mcp_regression_clustered_na` | MCP `regression_clustered` | the printed table, observations, clusters and the warning line |
| `test_validate_cli_reg_clustered_na` | `p2a reg clustered` | the printed table, and with `-F json` coefficients, SEs, `n_observations`, `n_clusters` and `missing` |

Tolerances (`validation/README.md`, n = 195-197, relative): coefficients 1e-8, SE 1e-6;
counts exact; MCP text at its 4 printed decimals, CLI text at 6.

### Driscoll-Kraay time ids

`run_vcov_driscoll_kraay` reads `time_col` as a whole-number time index: integers, floats
that are whole numbers, or dates. A fractional id is an `InvalidSpecification` naming the
row and value (1935.25 was truncated to period 1935 before); a text column is refused. A
row whose time id is missing is dropped, the model is re-estimated, and the drop is
reported in `DriscollKraayResult::missing` and as the first warning. On `reg_cluster_na`
with `time_col = year_na`: 198 rows in 20 periods (before: 200 rows and a bogus 21st
period). The Driscoll-Kraay SEs themselves are R-pinned on complete data (see
[ols_robust_se.md](ols_robust_se.md)); two property tests, not registered as R-pinned,
pin the estimation sample on `validation/datasets/generated/grunfeld_cluster_na.csv`
(loaded without a case):

- `test_na_driscoll_kraay_time_ids` (`crates/p2a-core/src/regression/ols.rs`): the result
  on the data with missing years equals the estimator on the complete rows (coefficients,
  SEs, bandwidth); integral float ids give the Int64 result; fractional and text ids are
  refused; with no time id in any row the error is `InsufficientData` naming the method
  and the time column;
- `test_mcp_regression_driscoll_kraay_missing_time`
  (`crates/p2a-mcp/src/tools/handlers/regression.rs`): the dropped rows and the refused
  fractional ids through MCP `regression_driscoll_kraay`.

## Known Differences

1. **Degrees of freedom**: t inference uses n − k (as `coeftest` with `vcovCL`), not
   G − 1, for one- and two-way clustering.
2. **No HC0-type option**: p2a always applies G/(G − 1) and (n − 1)/(n − k) (CR1).
3. **Missing cluster ids**: rows with a missing id are dropped and the model is
   re-estimated (as fixest); see "Missing Cluster and Time Identifiers".
4. **Few clusters**: with fewer than 50 clusters a warning is added (Cameron and Miller
   2015); the SEs are unchanged.

## Running the Tests

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- regression::ols::tests::test_validate_ols_clustered
```

## References

- Arellano, M. (1987). "Computing Robust Standard Errors for Within-Groups Estimators". *Oxford Bulletin of Economics and Statistics*, 49(4), 431-434.
- Cameron, A.C., Gelbach, J.B., & Miller, D.L. (2011). "Robust Inference with Multiway Clustering". *Journal of Business & Economic Statistics*, 29(2), 238-249.
- Cameron, A.C. & Miller, D.L. (2015). "A Practitioner's Guide to Cluster-Robust Inference". *Journal of Human Resources*, 50(2), 317-372.
- Zeileis, A., Köll, S., & Graham, N. (2020). "Various Versatile Variances: An Object-Oriented Implementation of Clustered Covariances in R". *Journal of Statistical Software*, 95(1), 1-36.
