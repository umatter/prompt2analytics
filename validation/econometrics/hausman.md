# Validation: Hausman Specification Test (FE vs RE)

## Method and options

H0: the entity effects are uncorrelated with the regressors (random effects is consistent
and efficient); H1: they are correlated (use fixed effects). As `plm::phtest` (method
"chisq"):

```
H = |d' D^- d|,  d = b_FE - b_RE,  D = Var(b_FE) - Var(b_RE),  H ~ chi2(rank D)
```

- Core: `p2a_core::run_hausman_test(dataset, y, x, entity)`; MCP tool: `hausman_test`.
- Both models are estimated with their classical covariances (phtest's default), whatever
  covariance the caller uses for FE or RE elsewhere; `HausmanResult.vcov_method` says so.
- The slopes are matched by name (the RE intercept is skipped).
- D is symmetrised, scaled by the FE standard errors (`D_s = S D S`,
  `S = diag(Var(b_FE)_jj^-1/2)`) and eigen-decomposed; eigenvalues with
  `|lambda| <= 100 k eps max(1, max|lambda|)` are dropped. When D is invertible the
  statistic equals phtest's exactly, and the scaling makes the rank decision independent of
  regressor units.
- `df` is the rank of D. A D that is not positive definite (phtest then takes the absolute
  value) or is singular (generalized inverse) is reported in `HausmanResult.warnings`,
  followed by the fixed-effects and random-effects warnings. A D that is zero to machine
  precision for every slope (for example, no regressor varies between entities, so FE and
  RE coincide) is refused with `InvalidSpecification`. Data that FE or RE refuses
  (non-finite values, exactly collinear regressors) is refused with their error.

## R reference

Cases `panel_hausman_grunfeld`, `panel_hausman_grunfeld_unbalanced` and
`panel_hausman_sim_corr`, defined in `validation/r/cases/panel.R`. R 4.3.3, plm 2.6.7:

```r
pd <- pdata.frame(d, index = c("firm", "year"))
fe <- plm(inv ~ value + capital, data = pd, model = "within")
re <- plm(inv ~ value + capital, data = pd, model = "random")
phtest(fe, re)
eigen(vcov(fe)[cn, cn] - vcov(re)[cn, cn])   # cn = names(coef(fe))
```

Datasets: `validation/datasets/grunfeld.csv`,
`validation/datasets/generated/grunfeld_unbalanced.csv` and
`validation/datasets/generated/panel_corr_effects.csv` (entity effects correlated with
`value`, so the test must reject).

## Pinned values (R, 12 significant digits; the case files hold 17)

| case | χ² | df | p | coef. difference `value` | coef. difference `capital` | eigenvalues of V_FE - V_RE |
|---|---|---|---|---|---|---|
| `panel_hausman_grunfeld` | 2.33036689368 | 2 | 0.311865446055 | 0.000342651888235 | 0.00195235846943 | 1.62385498269e-06, 3.48716027853e-05 |
| `panel_hausman_grunfeld_unbalanced` | 2.7465769254 | 2 | 0.253272710745 | -0.00226382783349 | 0.00341383099639 | 2.24420764387e-06, 5.91442991045e-05 |
| `panel_hausman_sim_corr` | 7203.84027461 | 2 | 0 | -0.3270946076 | 0.0158249704287 | -0.000881180363117, 1.48583977601e-05 |

At full precision (as in `validation/expected/panel_hausman_grunfeld.json`), phtest's Grunfeld
statistic is χ² = 2.330366893675463 on 2 df, p = 0.3118654460548856.


The eigenvalues are not a p2a output; the test checks that the "not positive definite"
warning appears exactly when one is negative (only `panel_hausman_sim_corr`).

## Tests

- `p2a-core`: `econometrics::panel::tests::test_validate_hausman_grunfeld`,
  `econometrics::panel::tests::test_validate_hausman_grunfeld_unbalanced`,
  `econometrics::panel::tests::test_validate_hausman_sim_corr` pin the statistic, df,
  p-value and coefficient differences, the covariance label, the warning, and (Grunfeld)
  the recommendation text.
- Unit tests of the quadratic form in
  `econometrics::panel::specification_test::tests`: `test_hausman_form_full_rank`,
  `test_hausman_form_indefinite_takes_absolute_value`, `test_hausman_form_singular_uses_rank`,
  `test_hausman_form_zero_difference_is_an_error`, `test_hausman_form_units_do_not_matter`;
  and `econometrics::panel::tests::test_hausman_refuses_when_fe_and_re_coincide`,
  `econometrics::panel::tests::test_hausman_small_panel_recommendation`,
  `econometrics::panel::tests::test_collinear_regressors_are_refused`.
- `p2a-mcp`: `tools::handlers::panel::tests::test_validate_mcp_hausman_test` checks the
  `hausman_test` text against the case at display precision.
- Registry: `validation/methods/panel.json`, method `panel_hausman`.

## Tolerance

`validation/README.md`, n = 120-300: statistics 1e-6 relative, p-values 1e-6 relative or
1e-12 absolute; df exact.

## What changed in Tier 1

- The statistic used only the diagonal of D, floored negative entries and dropped them, and
  its RE covariance was wrong (ECON-1). It returned chi2 0.0000, p 1.0000 on Grunfeld (phtest
  2.3304, p 0.3119) and recommended random effects as "FE and RE produce nearly identical
  estimates"; that p > 0.90 recommendation is removed (ECON-2). With the corrected RE
  covariance alone, the diagonal formula would still give 0.6381 (Grunfeld), 1.3499
  (unbalanced) and 7212.01 (simulated, where it drops the negative direction).
- `df` was the number of slopes; it is now the rank of D.

## Known differences

- A zero difference (FE and RE coincide) is refused. phtest returns roundoff there (for
  example chisq 2.9e-14, p = 1 on a panel whose regressor has no between variation).
- phtest uses `solve(D)` and `df = length(d)`, so it stops on a singular D (or returns
  roundoff when D is nearly singular). p2a uses a generalized inverse with `df = rank D`
  and a warning; on the three cases D has full rank and the two agree.
- A regressor that is constant within every entity, or one that is exactly collinear with
  others: plm's within model drops the aliased column and phtest compares the common
  slopes (on a 12 x 5 panel with `x1` noisy and `z = id %% 4`, chisq 0.18337, df 1, p
  0.6685). p2a refuses the test with fixed effects' `SingularMatrix` error, because FE
  refuses such a regressor, until drop-and-warn lands (ECON-72).

## How to run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::panel::tests::test_validate_hausman
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::panel::specification_test
cargo test -p p2a-mcp --lib --no-default-features --features "http,llm,auth,websocket" -- tools::handlers::panel::tests
Rscript validation/r/run.R --check
```

## References

- Hausman, J.A. (1978). Specification tests in econometrics. *Econometrica*, 46(6),
  1251-1271.
- Croissant, Y. & Millo, G. (2008). Panel data econometrics in R: The plm package.
  *Journal of Statistical Software*, 27(2), 1-43.
