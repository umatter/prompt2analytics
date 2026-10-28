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
relative; firm p-values within 7.3e-15 absolute. The cluster meat is summed over a
`HashMap`, so the last digits vary between runs.

Dropping the (n − 1)/(n − k) factor from the CR1 scaling makes all three tests fail (the
firm SEs then equal `se_cluster_firm_hc0`).

`test_clustered_se` (string cluster ids, cluster count) is a property test, not pinned to R.

## Known Differences

1. **Degrees of freedom**: t inference uses n − k (as `coeftest` with `vcovCL`), not
   G − 1, for one- and two-way clustering.
2. **No HC0-type option**: p2a always applies G/(G − 1) and (n − 1)/(n − k) (CR1).
3. **Missing cluster ids**: null cluster values are pooled into one cluster (REG-21,
   Tier 2). The case has no missing ids.
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
