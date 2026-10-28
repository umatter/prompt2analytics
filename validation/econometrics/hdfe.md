# Validation: High-Dimensional Fixed Effects (HDFE)

## Method and options

`run_hdfe` absorbs one or more categorical fixed-effect dimensions with the method of
alternating projections (MAP) and runs OLS on the demeaned data, without dummy variables.

- Core: `p2a_core::run_hdfe(dataset, y, x, fe_cols, config, se_type)`; MCP tool:
  `panel_hdfe`.
- `se_type` (`CovarianceType`): `Standard` (iid) or HC0-HC3; HC1 uses `n / df_resid`. There
  is no cluster option.
- `config` (`HdfeConfig`): tolerance, iteration limit and acceleration of the MAP; invalid
  controls and non-finite data are refused.
- Degrees of freedom: `df_resid = n - K - (sum of levels - redundant levels)`.
- Fit statistics: within R-squared, adjusted within R-squared in plm's convention
  `1 - (1 - R2)(n - 1)/df_resid`, and the within F test of the slopes.

## R reference

Case `panel_hdfe_grunfeld`, defined in `validation/r/cases/panel.R`. R 4.3.3, fixest 0.14.2,
plm 2.6.7, on `validation/datasets/grunfeld.csv`:

```r
h <- feols(inv ~ value + capital | firm + year, g, vcov = "iid")
se(summary(h, vcov = "iid")); se(summary(h, vcov = "hetero"))
r2(h, "wr2"); fitstat(h, "wf")
summary(plm(inv ~ value + capital, data = pdata.frame(g, index = c("firm", "year")),
            model = "within", effect = "twoways"))$r.squared["adjrsq"]
feols(inv ~ value + capital | firm, g, vcov = "iid")   # firm effects only
```

## Pinned values (R, 12 significant digits; the case file holds 17)

| quantity | `value` | `capital` |
|---|---|---|
| coefficient (firm + year) | 0.117715855083 | 0.357916273073 |
| SE iid | 0.0137512830036 | 0.0227190108826 |
| SE hetero (HC1) | 0.0191799205213 | 0.0544026642585 |
| t (HC1) | 6.13745270486 | 6.57902104523 |
| p (HC1) | 5.78157933598e-09 | 5.69168652956e-10 |
| SE clustered by firm (not pinned) | 0.0108244294769 | 0.0478483965926 |
| coefficient (firm only) | 0.110123804121 | 0.3100653413 |
| SE iid (firm only) | 0.011856694214 | 0.0173545027756 |

| quantity | value |
|---|---|
| within R² | 0.720145212924 |
| adj. within R² (plm convention, pinned) | 0.670466848355 |
| adj. within R² (fixest `war2`, not pinned) | 0.716833321953 |
| full-model R² (not pinned) | 0.951693399681 |
| within F | 217.442306876 |
| within F df1 | 2 |
| within F df2 | 169 |
| within F p | 1.844236205e-47 |
| df_resid | 169 |
| n | 200 |


## Tests

- R-pinned (`p2a-core`, registered as method `panel_hdfe` in `validation/methods/panel.json`):
  - `econometrics::hdfe::tests::test_validate_hdfe_grunfeld`: firm + year, iid SEs:
    coefficients, SEs, within R-squared, adjusted within R-squared (plm), within F with its
    df and p-value, `df_resid`, n, and the level counts (10, 20).
  - `econometrics::hdfe::tests::test_validate_hdfe_grunfeld_hc1`: HC1 SEs, t and p
    against fixest's `hetero`.
  - `econometrics::hdfe::tests::test_validate_hdfe_grunfeld_firm_only`: firm effects only,
    coefficients and iid SEs (the one-way within estimator).
- DGP recovery (no R values compared): `econometrics::hdfe::tests::test_validate_against_felm_coefficients`,
  `econometrics::hdfe::tests::test_validate_against_felm_standard_errors`,
  `econometrics::hdfe::tests::test_validate_against_felm_r_squared`,
  `econometrics::hdfe::tests::test_validate_single_fe_against_within_estimator`,
  `econometrics::hdfe::tests::test_validate_large_panel_coefficient_recovery`. They check
  that known coefficients and degrees of freedom are recovered on simulated panels.

On a balanced firm x year panel the two demeaning projections commute, so the MAP is exact
after one sweep and the Grunfeld pins need no iterative allowance.

## Tolerance

`validation/README.md`, n = 120-300: coefficients 1e-8 relative, standard errors and
statistics 1e-6 relative, p-values 1e-6 relative or 1e-12 absolute; counts and degrees of
freedom exact.

## What changed in Tier 1

`run_hdfe` is unchanged; its tests were not pinned to R. They compared with literals R does
not produce (felm coefficients 0.11772003, 0.35792286 and SEs 0.01375339, 0.02272406, firm-only
0.11013083, 0.31004929; within R-squared 0.7201) at tolerances of 1e-4 to 1e-2, on an
inline copy of the dataset. The page also quoted a felm table for Grunfeld and an R output
for the small simulated example (x1 0.99766, SE 0.02834) that R does not give: fixest on
that data gives x1 0.9802154032 (SE 0.01506135460) and x2 0.4826382350 (SE 0.02363228724)
with 13 residual df. Both are removed.

## Known differences

- The adjusted within R-squared follows plm (0.6704668484), not fixest's `war2`
  (0.7168333220), which uses a different degrees-of-freedom convention.
- No cluster option, so fixest's firm-clustered SEs are not compared.
- HC2 and HC3 are not pinned (ECON-22).

## How to run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::hdfe::tests::test_validate
Rscript validation/r/run.R --check
```

## References

- Gaure, S. (2013). lfe: Linear Group Fixed Effects. *The R Journal*, 5(2), 104-117.
- Berge, L. (2018). Efficient estimation of maximum likelihood models with multiple
  fixed-effects: the R package FENmlm. *CREA Discussion Papers* 13 (fixest).
- Grunfeld, Y. (1958). *The Determinants of Corporate Investment*. Ph.D. dissertation,
  University of Chicago.
