# Validation: Panel GLS (pggls)

## Method Overview

Panel GLS (Feasible Generalized Least Squares) estimates panel data models while accounting for heteroskedasticity across periods and correlation between the periods of an entity (an unrestricted T × T error covariance, as plm's `pggls`; no N × N cross-sectional correlation). It transforms the data to achieve efficient estimation when the standard OLS assumptions are violated.

**Key Parameters:**
- `model`: Type of transformation - "within" (fixed effects), "pooling", or "fd" (first difference)
- Entity and time identifiers for panel structure

**Models Supported:**
1. **Fixed Effects GLS (FEGLS)**: Within transformation + GLS (see Known Differences 2)
2. **Pooled GLS**: Standard pooled estimation with GLS variance
3. **First Difference GLS**: FD transformation + GLS (see Known Differences 2)

**Status**: not R-pinned (ECON-36). Since T0-WP14 p2a refuses panels with more distinct
periods than entities (T > N; ECON-32). The T × T error covariance is estimated from N
residual vectors and has no inverse there, so full Grunfeld (N = 10, T = 20) is refused,
while plm's `pggls` returns a generalised-inverse fit. Only the pooled coefficients are
compared with R, on Grunfeld 1935-1944 (N = T = 10).

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| plm | R | `pggls()` | 2.6.7 (R 4.3.3) |
| linearmodels | Python | `PanelOLS` | N/A |

## Test Cases

Both use `inv ~ value + capital` on `validation/datasets/grunfeld.csv`. Values are printed
with `formatC(digits = 16, format = "g")`.

### Grunfeld 1935-1944 (N = T = 10)

The panel p2a's pooled-GLS test fits.

**R Code:**
```r
library(plm)
d <- pdata.frame(subset(read.csv("validation/datasets/grunfeld.csv"), year <= 1944), index = c("firm", "year"))
for (m in c("within", "pooling", "fd")) {
  fit <- pggls(inv ~ value + capital, data = d, model = m)
  print(formatC(coef(fit), digits = 16, format = "g")); print(formatC(sqrt(diag(vcov(fit))), digits = 16, format = "g"))
}
```

**R results** (R 4.3.3, plm 2.6.7):

| | within (n 100) | pooling (n 100) | fd (n 90) |
|---|---|---|---|
| (Intercept) | none | -13.74653219816151 (SE 1.539090790420873) | 10.03656990989108 (SE 1.338068135647484) |
| value | 0.06888727256843909 (SE 0.0007405206690987179) | 0.09591408245586269 (SE 0.0008283909896617362) | 0.06039987296684579 (SE 0.001458872991019236) |
| capital | 0.1544609271562051 (SE 0.006065620653202037) | 0.1596389387764905 (SE 0.004552539460846831) | -0.4829444560383797 (SE 0.01675773016581451) |

`validation/scripts/validate_panel_full.R` tests 6-7 write the within and pooling columns to
`validation/expected/fegls_1935_1944.csv` and `validation/expected/pggls_pooling_1935_1944.csv`.

### Full Grunfeld (N = 10, T = 20): refused by p2a

plm's generalised-inverse fit; p2a returns `InvalidSpecification` for this input. With
T > N the 20 × 20 error covariance has rank at most 10.

**R Code:**
```r
library(plm)
d <- pdata.frame(read.csv("validation/datasets/grunfeld.csv"), index = c("firm", "year"))
pggls(inv ~ value + capital, data = d, model = "within")   # also "pooling", "fd"
```

**R results** (R 4.3.3, plm 2.6.7):

| | within (n 200) | pooling (n 200) | fd (n 190) |
|---|---|---|---|
| (Intercept) | none | -40.32087035321853 (SE 3.606576997901137) | -1.76627310074304 (SE 0.07410199508316578) |
| value | 0.1096888240695295 (SE 0.0009197911431547233) | 0.115395750204538 (SE 0.0008612843838058983) | 0.08973990829009358 (SE 0.0001667942832044628) |
| capital | 0.3086506488152131 (SE 0.003522505893393277) | 0.2310418000883722 (SE 0.004088210706544555) | 0.2919898520737104 (SE 0.002313382314341975) |

## Rust tests

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with ECON-36):
- `econometrics::panel::gls_models::tests::test_validate_pooled_gls_square_grunfeld_vs_plm` (`crates/p2a-core/src/econometrics/panel/gls_models.rs`): pooled coefficients on Grunfeld 1935-1944 within 1e-8 absolute of the pggls values typed into the test (-13.7465321982, 0.0959140825, 0.1596389388). SEs are not checked. This is a quoted-value check.
- `econometrics::panel::tests::test_panel_gls_fe_full_grunfeld_is_refused` and `econometrics::panel::tests::test_pooled_gls_full_grunfeld_is_refused` (`crates/p2a-core/src/econometrics/panel/mod.rs`): full Grunfeld returns `InvalidSpecification`, and the message names N = 10 and T = 20. Their doc comments record pggls's values.
- `econometrics::panel::gls_models::tests::test_panel_gls_refuses_more_periods_than_entities` (`crates/p2a-core/src/econometrics/panel/gls_models.rs`): the T > N refusal on a 3 × 6 synthetic panel and on Grunfeld up to 1954, and acceptance of a 5 × 5 panel.
- `econometrics::panel::gls_models::tests::test_pinned_panel_gls_outputs` (`crates/p2a-core/src/econometrics/panel/gls_models.rs`): a characterisation pin of p2a's own pooled (1935-1944) and first-difference (1935-1939) coefficients, SEs, RSS and R², set against the pre-T0-WP14 code. It compares no R number. FE is not pinned (ECON-33).
- `econometrics::panel::tests::test_panel_gls_fe` and `econometrics::panel::tests::test_panel_gls_pooling` (`crates/p2a-core/src/econometrics/panel/mod.rs`): their own 25-row panel (5 entities, periods 1-5). FE checks n_obs 25, 5 groups, 5 periods and the coefficient within 0.5 of the DGP's 2.0; pooling checks structure only.
- No FE or first-difference test compares with R. The first-difference test this document cited before (test_panel_gls_first_diff) never existed.

## Numerical Precision Summary

- **Coefficients**: the pooled coefficients on Grunfeld 1935-1944 agree with pggls to within
  7e-14 relative (`test_pinned_panel_gls_outputs`: -13.746532198161622, 0.09591408245586196,
  0.15963893877650104). The FE and FD coefficients are not compared.
- **Standard errors**: no test compares an SE with R. p2a's GLS step ignores the FE/FD
  transformation and multiplies the variance by rss/df although Ω already is the error
  covariance (ECON-33). On Grunfeld 1935-1944, pooled, its SEs (110.35, 0.05939, 0.3264;
  `test_pinned_panel_gls_outputs`) are 71.7 times pggls's (1.539, 0.000828, 0.00455), the
  factor sqrt(RSS/df) = sqrt(498646.34/97).
- **R-squared**: computed differently depending on model; not compared.

## Known Differences

1. **T > N is refused**, where pggls uses a generalised inverse (ECON-32, done in T0-WP14).
   p2a estimates the unrestricted T × T Ω from the first-step residuals, as pggls does
   (`gls_models.rs`, `Omega[s,t] = (1/N) * sum_i(e_is * e_it)`).
2. **FE and FD models** (ECON-33, Tier 4): GLS runs on the untransformed data, the FD
   residuals are matched to the first n_diff rows, the SEs are scaled twice, and unbalanced
   panels use a submatrix of the full inverse.
3. **Degrees of freedom**: Slight differences in small-sample corrections
4. **First difference model**: May exclude different observations at boundaries

## Performance Comparison

No timing is claimed here. Speed comparisons with R come from the benchmark pipeline in the
paper repository (`performance/` in `../prompt2analytics-paper`), which runs the Rust
benchmarks of `crates/p2a-core/benches/` against R.

## References

- Baltagi, B. H. (2013). *Econometric Analysis of Panel Data* (5th ed.). Wiley.
- Croissant, Y., & Millo, G. (2008). "Panel Data Econometrics in R: The plm Package." *Journal of Statistical Software*, 27(2).
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.

## Implementation Notes

`run_panel_gls` (`crates/p2a-core/src/econometrics/panel/gls_models.rs`):
1. First-step OLS on the within-transformed (FE), untransformed (pooling) or
   first-differenced (FD) data. The transformation is used only for these first-step
   residuals.
2. The T × T error covariance from those residuals, `Omega[s,t] = (1/N) * sum_i(e_is * e_it)`
   over the entities observed in both periods, as pggls estimates it. A 1e-6 ridge is added
   to its diagonal when it cannot be inverted.
3. GLS, β_GLS = (Σ_i X_i'Ω_i⁻¹X_i)⁻¹ Σ_i X_i'Ω_i⁻¹y_i with Ω_i⁻¹ the submatrix of Ω⁻¹ for
   the entity's periods, on the untransformed data for all three models: FE and FD do not
   transform the GLS step (ECON-33; FE only drops the intercept column).
4. The variance (X'Ω⁻¹X)⁻¹ multiplied by rss/df, df = n - k, although Ω already is the
   error covariance (ECON-33); pggls reports (X'Ω⁻¹X)⁻¹.
