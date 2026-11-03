# Validation: Regression Through the Origin (no intercept)

## Method Overview

`run_ols(.., intercept = false, ..)`, `run_ols_raw` on a design without a constant column,
`run_gls(.., intercept = false, ..)`, `OlsClusteredResult::fit(.., intercept = false)` and
`HacResult::fit(.., intercept = false)` fit y = Xβ + ε without an intercept. The CLI reaches
them with `--intercept false` on `reg ols`, `reg clustered`, `reg hac`, `reg gls`, `reg step`
and `reg quantreg`, and MCP with `regression_ols` `intercept: false`.

Coefficients, standard errors, t and p are computed as with an intercept (k counts the slope
columns only). The fit statistics follow R's `summary.lm` when `attr(terms, "intercept") == 0`
(REG-19):

| Quantity | With an intercept | Without an intercept |
|---|---|---|
| total sum of squares SST | Σ (y − ȳ)² | Σ y² |
| explained sum of squares | SST − SSR | SST − SSR = Σ ŷ² (the residuals are orthogonal to ŷ) |
| R² | 1 − SSR/SST | 1 − SSR/SST = Σ ŷ² / (Σ ŷ² + SSR) |
| adjusted R² | 1 − (1 − R²)(n − 1)/(n − k) | 1 − (1 − R²) n/(n − k) |
| F | ((SST − SSR)/(k − 1)) / σ̂² on (k − 1, n − k) | ((SST − SSR)/k) / σ̂² on (k, n − k) |

`OlsResult.sst` holds the SST of the row that applies, `df_model` the F numerator df. The CLI
labels the uncentered statistics `R-squared (uncentered)` and `Adj. R-squared (uncentered)` and
sets JSON `"r_squared_type": "uncentered"`.

Which rule applies: `run_ols`, the clustered and the HAC fits use their `intercept` argument;
`run_ols_raw` and `gls` see only a design matrix and treat the model as having an intercept
when some column is constant and non-zero, in any position (`has_constant_column`: first
entry finite and non-zero, every entry within 1e-10 · max(|v₀|, 1) of it). Before REG-19
`run_ols_raw` looked at column 0 only, and every entry point used the centered SST.

## Reference Implementation

R 4.3.3, sandwich 3.1.3, quantreg 6.1. Case `ols_no_intercept_grunfeld` in
[`validation/r/cases/cli_flags.R`](../r/cases/cli_flags.R), on
`validation/datasets/grunfeld.csv` (n = 200, 10 firms, 20 years):

```r
m0 <- lm(inv ~ 0 + value + capital, data = g)
summary(m0)                                          # r2, adj_r2, sigma, F
vcovHC(m0, type = "HC1")                             # se_hc1
vcovCL(m0, cluster = ~firm, type = "HC1")            # se_cluster_firm_hc1
vcovCL(m0, cluster = ~firm + year, type = "HC1")     # se_cluster_firm_year_hc1
NeweyWest(m0, lag = 4, prewhite = FALSE)             # se_nw_lag4
step(m0, direction = "backward"); extractAIC(m0); extractAIC(m0, k = log(n))
rq(inv ~ 0 + value + capital, tau = 0.5)             # rq_coef, recorded only
```

| `ols_no_intercept_grunfeld` | value | capital |
|---|---|---|
| coef | 0.10763842564502346 | 0.18320624121791276 |
| se_classical / t / p | 0.0058255828827650088 / 18.476850782343483 / 5.7847070115501921e-45 | 0.024274988583620462 / 7.5471195624590779 / 1.586907312591542e-12 |
| se_hc1 / t / p | 0.007157222743767907 / 15.039133124471881 / 1.3507255347816579e-34 | 0.045848483655002208 / 3.9959062244346311 / 9.0865394780483143e-05 |
| se_cluster_firm_hc1 / p | 0.018368739167695273 / 1.9073606787881814e-08 | 0.091581574190960788 / 0.046815654995564775 |
| se_cluster_firm_year_hc1 / p | 0.018915957603117081 / 4.513870455905945e-08 | 0.087476980292006776 / 0.037502471829740008 |
| se_nw_lag4 / t / p | 0.012024172521665591 / 8.9518364320768544 / 2.5500672258474932e-16 | 0.06422501151487825 / 2.8525684448569004 / 0.0047972150312808187 |
| rq_coef (recorded only) | 0.1116253861993205 | 0.078613841312111174 |

Scalars: R² 0.8578932101329837, adjusted R² 0.85645778801311478, sigma 98.872298425483166,
F 597.65918210272923 on (2, 198), p 1.2848962858403676e-84, logLik −1201.5484935724141, AIC
2409.0969871448283, extractAIC 1839.5215738629595 (k = 2) and 1846.1182085960556 (k = log n);
drop1 AIC none / −value / −capital 1839.5215738629595 / 2037.957430658571 / 1888.0887280465781,
so backward `step` keeps both terms.

**p2a before REG-19** on the same model: R² 0.793204443199507, adjusted R² 0.79216002119546414,
F 379.73369008363528 (the centered SST with n − 1 and k − 1). `run_ols_raw` with the ones column
last on `ols_grunfeld`: F 284.38382086987531 on 3 df against R's 426.57573130481285 on 2.

## Rust tests

| Test | Checks |
|---|---|
| `regression::ols::tests::test_validate_ols_no_intercept_grunfeld` | coefficients, classical SE/t/p, HC1 SE and p, R², adjusted R², sigma, F with its df and p, logLik, AIC |
| `regression::ols::tests::test_validate_ols_raw_intercept_in_last_column` | `run_ols_raw` with the ones column last: F, its numerator df, R², adjusted R² (case `ols_grunfeld`) |
| `regression::ols::tests::test_validate_ols_clustered_no_intercept_grunfeld` | `OlsClusteredResult::fit(.., false)`, one- and two-way: coefficients, CR1 SE and p, cluster counts, R² |
| `regression::ols::tests::test_validate_hac_no_intercept_grunfeld` | `HacResult::fit(.., false)` with Bartlett bandwidth 4: coefficients and SEs |
| `regression::ols::tests::test_clustered_and_hac_wrappers_keep_the_intercept` | `run_ols_clustered` and `run_vcov_hac` equal `fit(.., true)`; two-way SEs of `ols_grunfeld` |
| `regression::gls::tests::test_validate_gls_identity_no_intercept_grunfeld` | identity GLS without intercept: coefficients, SE, t, p, R², adjusted R², sigma; with intercept R² and adjusted R² of `ols_grunfeld` (centered) |

The CLI tests that reach these fits are listed in [../cli/flags.md](../cli/flags.md).

## Tolerance

`validation/README.md`, n = 200: coefficients `Tol::coef` (1e-8 relative), SEs and statistics
`Tol::se` / `Tol::stat` (1e-6 relative), p-values `Tol::p_value` (1e-6 relative, also 1e-12
absolute), degrees of freedom and counts `Tol::exact()`.

## Status

R-pinned (registry `validation/methods/cli_flags.json`, id `ols_no_intercept`).

## Known Differences

- `run_ols_raw` and `gls` detect the intercept as a constant non-zero column; R reads it from
  the model's terms. A design with a constant column that is not meant as an intercept is
  treated as one.
- `stats::anova_from_ols` still uses n − 1 total degrees of freedom for a no-intercept fit,
  where R uses n. Library only (no CLI or MCP route); left to the owner of `stats/anova.rs`.
- `rq` without intercept is recorded (`rq_coef`), not pinned: p2a's quantile regression is not
  the exact Barrodale-Roberts LP (REG-52).

## References

- R Core Team. `summary.lm` (stats package), the `intercept` attribute of `terms`.
- Zeileis, A., Köll, S. and Graham, N. (2020). Various Versatile Variances: An Object-Oriented
  Implementation of Clustered Covariances in R. *Journal of Statistical Software*, 95(1).
