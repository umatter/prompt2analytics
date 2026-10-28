# Validation: GMM (Arellano-Bond / Blundell-Bond)

## Method Overview

Generalized Method of Moments (GMM) estimators for dynamic panel data models. The p2a-core implementation supports:

- **Arellano-Bond (1991)**: Difference GMM using lagged levels as instruments for differenced equations
- **Blundell-Bond (1998)**: System GMM adding level equations with lagged differences as instruments

Key parameters:
- `transform`: "difference" (AB) or "system" (BB)
- `step`: "onestep" or "twostep"
- `lags`: Number of lags of dependent variable to include
- `min_lag`/`max_lag`: Instrument lag limits
- `collapse`: Whether to collapse instrument matrix
- `robust`: robust standard errors for two-step (not Windmeijer-corrected; see Known Differences)

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| plm | R | `pgmm()` | 2.6.7 (R 4.3.3) |

**Status**: not R-pinned. No committed R case is compared (ECON-36 adds the R-pinned case).
The values below are R's, computed on the data and model the Rust tests use; no p2a value is
claimed.

## Test Cases

### Grunfeld, `inv ~ lag(inv, 1) + value + capital`

The data (`validation/datasets/grunfeld.csv`, N = 10 firms, T = 20 years) and model of
`test_validate_arellano_bond_full_grunfeld` and `test_validate_gmm_twostep_full_grunfeld`,
with all available lags of `inv` (2 and deeper) as GMM instruments.

**R Code**:
```r
library(plm)
pd <- pdata.frame(read.csv("validation/datasets/grunfeld.csv"), index = c("firm", "year"))
m <- pgmm(inv ~ lag(inv, 1) + value + capital | lag(inv, 2:99), data = pd,
          effect = "individual",
          model = "onestep",        # or "twosteps"
          transformation = "d")     # "d": Arellano-Bond; "ld": system GMM (Blundell-Bond)
summary(m, robust = TRUE)           # robust SEs, Sargan test, AR(1) and AR(2) tests
```

**R results** (R 4.3.3, plm 2.6.7; `formatC(digits = 16, format = "g")`):

| | difference, one-step | difference, two-step | system (`ld`), two-step |
|---|---|---|---|
| lag(inv, 1) | 0.6796680080285489 | 0.6719522974788301 | 0.904434323720773 |
| value | 0.1069946670284529 | 0.1085372918048775 | 0.0210772058500476 |
| capital | 0.1120789195185634 | 0.1167927900220365 | 0.02408375709498078 |
| robust SE lag(inv, 1) | 0.1040257764812672 | 0.1220817501058947 | 0.08635876354389636 |
| robust SE value | 0.01377301578809117 | 0.01483628441598108 | 0.01047086634052574 |
| robust SE capital | 0.0381992198503537 | 0.04971262563204494 | 0.03152250547905452 |
| Sargan (df), p | 8.99907721785226 (170), 1 | 8.198968776259962 (170), 1 | 6.519993993604108 (190), 1 |
| AR(1) z (p) | -1.341029035838593 (0.1799110255835711) | -1.364283733937975 (0.1724782832950675) | -1.378238393462307 (0.1681296969832509) |
| AR(2) z (p) | -1.362302243133932 (0.1731025200261945) | -0.6931457052533437 (0.4882181173207438) | -1.413475217229229 (0.1575160432179447) |

The difference models use 180 differenced observations and 173 instruments. pgmm warns that
the first-step matrix is singular and that a generalized inverse is used (its robust
covariance warns the same). The Sargan p-value of 1 reflects instrument proliferation with
N = 10 firms, not a good fit.

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with ECON-36):
- `econometrics::panel::tests::test_arellano_bond_convenience` (`crates/p2a-core/src/econometrics/panel/mod.rs`): the transform is Difference.
- `econometrics::panel::tests::test_validate_arellano_bond_full_grunfeld` (`crates/p2a-core/src/econometrics/panel/mod.rs`): the lag coefficient lies in (0, 1). It prints R's 0.6720/0.1085/0.1168 but asserts none of them, and passes when the estimator returns `Err`.
- `econometrics::panel::tests::test_validate_gmm_twostep_full_grunfeld` (`crates/p2a-core/src/econometrics/panel/mod.rs`): the step is TwoStep and the Sargan statistic and p-value are in range; passes when the estimator returns `Err`.
- `econometrics::panel::dynamic_panel::tests::test_pinned_gmm_outputs` (`crates/p2a-core/src/econometrics/panel/dynamic_panel.rs`, added by T0-WP14): a characterisation pin of p2a's own outputs on Grunfeld, set against the pre-Tier-0 code, for the one-step non-robust full-instrument fit (155 instrument columns per its doc comment; pgmm reports 173 instruments for the same model) and a two-step robust fit with collapsed lags 2-4. It compares no R number.
- No system-GMM test exists, and the `system` option is still ignored (ECON-30): in `dynamic_panel.rs` the configured transform is only copied into the result.

## Numerical Precision Summary

Set by the R-pinned case (ECON-36). No test compares p2a with pgmm, so no tolerance is
claimed. pgmm is closed-form given the weight matrix, so the `validation/README.md`
tolerances apply.

## Known Differences

1. **Instrument matrix**: the GMM instruments are misaligned with the differenced equations (ECON-38).
2. **Two-step robust SEs** are not Windmeijer's finite-sample corrected SEs (ECON-29).
3. **System GMM** (`transform = "system"`) is silently ignored: p2a fits difference GMM (ECON-30).
4. **Specification tests**: p2a's AR(1)/AR(2) and Sargan statistics are not the Arellano-Bond m-tests or the Sargan/Hansen J statistic (ECON-39).
5. **Unbalanced panels**: gaps are zero-filled (ECON-28).

## Performance Comparison

No timing is claimed here. Speed comparisons with R come from the benchmark pipeline in the
paper repository (`performance/` in `../prompt2analytics-paper`), which runs the Rust
benchmarks of `crates/p2a-core/benches/` against R.

## References

1. Arellano, M., & Bond, S. (1991). "Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations." *Review of Economic Studies*, 58(2), 277-297.

2. Blundell, R., & Bond, S. (1998). "Initial conditions and moment restrictions in dynamic panel data models." *Journal of Econometrics*, 87(1), 115-143.

3. Windmeijer, F. (2005). "A finite sample correction for the variance of linear efficient two-step GMM estimators." *Journal of Econometrics*, 126(1), 25-51.

4. R package `plm` (Croissant & Millo): https://cran.r-project.org/package=plm
