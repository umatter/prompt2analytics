# Validation: Ordinary Least Squares (OLS)

## Method Overview

`run_ols` (and `run_ols_raw` on a prebuilt design matrix) estimates β̂ by least squares
and reports the inference and fit statistics of R's `summary.lm`. A full-rank,
well-conditioned design is solved by the normal equations; any other design by `lm.fit`'s
pivoted QR, which also decides which columns are aliased (see "Rank-deficient and
ill-conditioned designs" below).

**Parameters**: `y_col`, `x_cols`, `intercept`, `cov_type` (`Standard`, `HC0`-`HC3`; the
HC variants are validated in [ols_robust_se.md](ols_robust_se.md), clustered SEs in
[ols_clustered.md](ols_clustered.md)).

**Output** (`OlsResult`): per coefficient the estimate, SE, t, p and 95% CI; R², adjusted
R², residual standard error, F with its degrees of freedom and p-value, log-likelihood,
AIC, BIC, n and the residual degrees of freedom. The same numbers are available through
the `LinearEstimator` trait, which is what `p2a reg ols` and `p2a reg clustered` print.

## Conventions

With n observations and k coefficients (the intercept included):

| Quantity | Formula | R |
|---|---|---|
| t inference | t = β̂/SE on n − k df; CI β̂ ± t₀.₉₇₅,ₙ₋ₖ SE | `summary.lm`, `confint` |
| R² | 1 − SSR/SST (centred; uncentered without an intercept, see [no_intercept.md](no_intercept.md)) | `summary.lm` |
| adjusted R² | 1 − (1 − R²)(n − 1)/(n − k) | `summary.lm` |
| sigma | √(SSR/(n − k)) | `sigma` |
| F | ((SST − SSR)/(k − 1)) / (SSR/(n − k)) on (k − 1, n − k) | `summary.lm` |
| log-likelihood | −n/2 (1 + ln(2π SSR/n)), σ² at its ML estimate SSR/n | `logLik` |
| AIC | 2(k + 1) − 2 logLik (the error variance counts as a parameter) | `AIC` |
| BIC | (k + 1) ln n − 2 logLik | `BIC` |

The `LinearEstimator` defaults follow the same conventions (adjusted R² divides by
`degrees_of_freedom()`, AIC and BIC count k + 1 parameters, and p-values go through
`t_test_p_value`, so a NaN standard error gives a NaN p-value rather than a panic). The
`OlsResult` implementation returns the struct's fields.

Until Tier 1 (REG-3, REG-20), the log-likelihood was evaluated at SSR/(n − k), AIC and BIC
counted k parameters, and the trait's adjusted R² divided by n − k − 1; `p2a reg ols`
printed `Adj. R-squared: 0.809537` on Grunfeld where R gives 0.8105035.

## R reference cases

The references are computed by `validation/r/cases/regression.R`
([source](../r/cases/regression.R)) and committed as `validation/expected/<case>.json`
(17 significant digits; R 4.3.3, sandwich 3.1.3):

| Case | Model | Data |
|---|---|---|
| `ols_grunfeld` | `lm(inv ~ value + capital)`, n = 200 | `validation/datasets/grunfeld.csv` |
| `ols_mtcars` | `lm(mpg ~ wt + hp + disp)`, n = 32 | `validation/datasets/generated/mtcars.csv` (all 32 rows of `datasets::mtcars`) |
| `ols_longley` | `lm(Employed ~ GNP.deflator + GNP + Unemployed + Armed.Forces + Population + Year)`, n = 16 | `validation/datasets/generated/longley.csv` (`datasets::longley`) |

```bash
Rscript validation/r/run.R --write ols_grunfeld ols_mtcars ols_longley   # rewrite the JSON and datasets
Rscript validation/r/run.R --check                                       # recompute and compare (exit 1 on drift)
```

Each case also holds the HC0-HC3 values (see [ols_robust_se.md](ols_robust_se.md)) and
`extract_aic`/`extract_bic` (R's `extractAIC`).

### `ols_grunfeld`

| | (Intercept) | value | capital |
|---|---|---|---|
| coef | -42.714369436559 | 0.11556215636055 | 0.23067848873197 |
| SE (classical) | 9.5116760314239 | 0.0058357095572206 | 0.025475801476509 |
| t | -4.4907300559274 | 19.802588738770 | 9.0548079103488 |
| p | 1.2073565413848e-05 | 9.5427026857834e-49 | 1.3473701051204e-16 |

R² 0.8124080125447284, **adj. R² 0.8105035253624413**, sigma 94.40840333225498, F
426.57573130481285 on (2, 197) with p 2.5799247071944057e-72, **logLik
-1191.8023603678728, AIC 2391.6047207357456, BIC 2404.797990201938**, df 197.

### `ols_mtcars`

| | (Intercept) | wt | hp | disp |
|---|---|---|---|---|
| coef | 37.105505269032 | -3.8008905826376 | -0.031156550829946 | -0.00093700908148966 |
| SE (classical) | 2.1108152454988 | 1.0661906391105 | 0.011435794295664 | 0.010349744857659 |

R² 0.8268361424946444, adj. R² 0.808282872047642, sigma 2.6389302129066055, F
44.56551985573167 on (3, 28) with p 8.649587714956961e-11, logLik -74.32148638534392, AIC
158.64297277068783, BIC 165.97165228468646.

### `ols_longley`

| | coef | SE (classical) |
|---|---|---|
| (Intercept) | -3482.258634595815 | 890.4203836073756 |
| GNP.deflator | 0.015061872271372779 | 0.08491492577476738 |
| GNP | -0.035819179292591 | 0.03349100777224341 |
| Unemployed | -0.02020229803816824 | 0.004883996816517028 |
| Armed.Forces | -0.010332268671735891 | 0.002142741631616757 |
| Population | -0.051104105653579195 | 0.2260732000693726 |
| Year | 1.8291514646135503 | 0.45547849914221344 |

R² 0.9954790045772955, adj. R² 0.992465007628826, sigma 0.3048540735619657, F
330.2853392345862 on (6, 9) with p 4.984030528724918e-10, logLik 0.9066496552335839, AIC
14.186700689532833, BIC 20.36741046745108.

## Tests

All in `crates/p2a-core/src/regression/ols.rs` (registry entry `ols` in
`validation/methods/regression.json`):

| Test | Case | Pins |
|---|---|---|
| `test_validate_ols_grunfeld` | `ols_grunfeld` | `run_ols`, classical: coef, SE, t, p, CI; R², adj. R², sigma, F and its df and p, logLik, AIC, BIC, n, df; and p, CI, adj. R², sigma, logLik, AIC, BIC through `LinearEstimator` |
| `test_validate_ols_raw_grunfeld` | `ols_grunfeld` | the same through `run_ols_raw` with HC1 (the HC1 SE, t, p and CI) |
| `test_validate_ols_mtcars` | `ols_mtcars` | as `test_validate_ols_grunfeld` |
| `test_validate_ols_longley` | `ols_longley` | coef, classical SE, t, p, CI and the fit statistics (struct fields) |

Tolerances (`p2a_validation::Tol`, the table in `validation/README.md`, relative):
coefficients 1e-8 for n = 200 and 1e-6 for n < 100; SEs, t, CI bounds and fit statistics
1e-6 (n = 200) and 1e-5 (n < 100); p-values the same relative tolerance or 1e-12 absolute;
degrees of freedom and n exact.

`test_validate_linear_estimator_defaults_grunfeld` (`crates/p2a-core/src/traits/estimator.rs`,
registry entry `linear_estimator_defaults`) runs the trait's default methods on the Grunfeld
fit through a wrapper that implements only the required ones, and pins t, p, CI, adj. R²,
sigma, logLik, AIC and BIC to `ols_grunfeld`. `test_default_p_values_are_nan_safe` and
`test_default_adj_r_squared_uses_residual_df` check the NaN and df = 0 cases by hand.

`test_validate_ols_simple_regression` is a DGP-recovery check (coefficients near the true
values), not pinned to R.

## Measured agreement

Largest differences on this machine (`P2A_VALIDATION_VERBOSE=1`):

| Case | Coefficients | SE | t | CI | logLik / AIC / BIC |
|---|---|---|---|---|---|
| `ols_grunfeld` | 4.2e-15 rel | 1.6e-15 rel (classical), 5.5e-15 (HC1) | 7.5e-15 rel | 3.4e-13 rel | 1.9e-16 rel |
| `ols_mtcars` | 3.9e-12 rel (disp) | 2.3e-14 rel | 3.9e-12 rel | 2.7e-13 rel | exact |
| `ols_longley` | 6.4e-15 rel (pivoted QR) | 1.8e-15 rel | 6.6e-15 rel | 3.3e-15 rel | 1.7e-14 rel (logLik) |

p-values agree within 3e-13 absolute on Grunfeld and mtcars. Before T3-WP09 the normal
equations reached only 3.56e-8 relative on Longley's coefficients (GNP.deflator); Longley
now takes the pivoted QR and agrees to 6.4e-15 (largest over the seven coefficients).

## Rank-deficient and ill-conditioned designs (T3-WP09)

**Solve.** `fit_ols` (behind `run_ols` and `run_ols_raw`) forms X'X (`linalg::xtx`, the
CUDA path under `cuda`) and inverts it with `sym_inverse(.., SingularPolicy::Error, "X'X")`.
When that succeeds with a reciprocal condition of the column-equilibrated X'X of at least
1e-8 and a positive, finite inverse diagonal, β̂ = (X'X)⁻¹X'y. The coefficients of
that solve have a relative error of order eps/rcond (up to about 50 eps/rcond measured on
calendar-year designs), so when the reciprocal condition is below 1e-2 one step of
iterative refinement follows, β̂ += (X'X)⁻¹X'(y − Xβ̂), at O(nk); it brings the error to
that of a QR solve, within `Tol::coef(n)` (1e-10 for n > 1000); measured at 1e-10
relative against the QR at reciprocal conditions 2.7e-5 (`ols_year_trend`) and about 1e-7
(`test_ols_solve_path_threshold`, next to the 1e-8 edge). At or above 1e-2 the unrefined error is below about 1e-12 and
the solve is the one OLS always ran, bit for bit (Grunfeld). The Breusch-Pagan,
Breusch-Godfrey and VIF auxiliary fits take the same two paths, refinement included, as
RESET's augmented fit already did. The residual and wild bootstraps (fixed design) make the
same normal-equations-or-QR decision once but solve each replicate without the refinement
step: it would triple the per-replicate cost, and a replicate's error of order eps/rcond
(at most about 1e-8 relative) moves the bootstrap standard error far less than its Monte
Carlo error.
Otherwise the design goes to `linalg::solve::least_squares`, a transcription of `lm.fit`'s
dqrdc2 pivoted QR with tol = 1e-7: the later of two collinear columns is aliased, exactly
as R decides it. Measured reciprocal conditions (R): Grunfeld 0.121 (normal equations),
`ols_year_trend` 2.67e-5 (normal equations, refined), Longley 5.34e-10 and `ols_illcond`
4.09e-12 (QR); every aliased design is singular and takes the QR.
`test_validate_ols_solve_paths` asserts the path of each pinned design,
`test_validate_ols_year_trend_normal_equations` pins the refined normal equations to lm at
`Tol::coef(5000)`, and `test_ols_solve_path_threshold` checks a design on either side of
1e-8, the normal-equations side against the QR at `Tol::coef(n)`.

**What an aliased column reports.** Coefficient, SE, t, p and CI are NaN;
`OlsResult.aliased` names it and `rank()` = k − |aliased|; the warnings carry R's text
"Coefficients not defined because of singularities: <names>". df = n − rank; R², adjusted
R², sigma and F (on rank − 1 and n − rank df) use the identified columns; logLik, AIC and
BIC count rank + 1 parameters. A design of rank 0 is `SingularMatrix`; non-finite values in
`run_ols_raw` input are refused (`lm.fit` refuses NA/NaN/Inf). HC0-HC3 work on the
identified columns ([ols_robust_se.md](ols_robust_se.md)).

**Cases** (`validation/r/cases/reg_rank_qr.R`, [source](../r/cases/reg_rank_qr.R)):

| Case | Design | Decisive values |
|---|---|---|
| `ols_aliased` (n = 120) | `y ~ x1 + da + db + dc` (dummy trap: dc aliased); `y ~ x1 + x2 + x3` with x2 = 2 x1 (x2 aliased); `y ~ x1 + k5` (constant column aliased) | trap coef 0.6866082588992249, 0.45848569646633064, 0.2524436788731715, 0.5765620492704012; dup rank 3, df 117, sigma 0.5952841394745126, F 45.379581065647386 |
| `ols_illcond` (n = 80) | `y ~ x1 + x2 + x3`, κ(X) = 1.68e8, full rank | coef -521.62338184282, 1.6205851564219942, -1.0683241910594095, 0.5114453636193678; the normal equations err by 2.8e-5 relative |
| `ols_year_trend` (n = 5000) | `y ~ year + z`, year an integer in 1950..2020 (uncentred), equilibrated rcond(X'X) = 2.67e-5 | coef 2.7799542505245984, 0.29884566688664571, 0.49061624818943167; without the refinement step the normal equations miss the intercept by 5.9e-9 relative, with it by 1.4e-11 |
| `ols_longley` | as above | κ(X) = 2.38e7 |

**Tests** (`crates/p2a-core/src/regression/ols.rs`, registry entry `ols_rank_qr` in
`validation/methods/reg_rank_qr.json`): `test_validate_ols_aliased_dummy_trap`,
`test_validate_ols_aliased_duplicate`, `test_validate_ols_aliased_constant` (identified
coefficients, aliased flags, SEs, rank, df, R², adjusted R², sigma, F and its df, logLik,
AIC, BIC; HC0/HC1/HC3), `test_validate_ols_illcond_qr` (`Tol::coef(80)`),
`test_validate_ols_year_trend_normal_equations` (`Tol::coef(5000)`, SEs and statistics
`Tol::se(5000)`) and
`test_validate_ols_longley_qr` (1e-10 relative: an accuracy check, lm itself is 3.5e-14
from NIST's certified values, not a pin of dqrdc2's rounding), `test_validate_ols_solve_paths`;
`test_ols_rank_zero_is_an_error` and `test_ols_raw_rejects_non_finite` are property tests.

**Measured agreement** (`P2A_VALIDATION_VERBOSE=1`): Longley coefficients within 6.4e-15
relative; `ols_illcond` within 2.4e-12 relative (x1;
the intercept 1.4e-12). LAPACK's dgeqp3 and dqrdc2 differ by 8.1e-11 on `ols_illcond`, which
is why no tolerance tighter than the README's is applied there.

**Timing.** The common path (full rank, reciprocal condition at least 1e-8) runs the
normal equations it ran before plus one O(k²) condition certificate; no release-mode
before/after table was measured in the package lane (see the package report).

## Stepwise selection (`step`, `add1`, `drop1`)

`step` (forward, backward, both), `add1` and `drop1` rank models by R's `extractAIC`
criterion, n ln(RSS/n) + k·p, with p the number of coefficients and k = 2 (AIC) or ln n
(BIC). It differs from `AIC()` by the constant n(1 + ln 2π) + k, so the selections are
those of `stats::step`, and `StepRecord::criterion` and `TermEvaluation::criterion` are
the "AIC" column R prints. `StepResult.final_model.aic` is R's `AIC()` of the final model.
Before Tier 1 (REG-20) the criterion penalised each regressor by about 3 and dropped
regressors that `step()` keeps.

Cases (same file, `validation/r/cases/regression.R`), with candidates x1, x2, x3; forward
starts from `y ~ 1`, backward and both from the full model:

| Case | Data | Forward / backward / both (AIC) | Backward BIC | Full model logLik, AIC, BIC |
|---|---|---|---|---|
| `step_legacy` | seed 42, n = 100, the DGP of `validate_regression_diag.R` (`generated/step_legacy.csv`) | x1 + x2; -160.62381890999552 | x1 + x2; -152.80830835203125 | -58.51602053537468, 127.03204107074936, 140.05789200068983 |
| `step_borderline` | seed 5, n = 120, full-model t(x2) = 1.5661 (`generated/step_borderline.csv`) | x1 + x2; 12.510295911347013 | x1; 18.50277174659007 | -173.42647791215913, 356.85295582431826, 370.7904145382285 |

Tests (`crates/p2a-core/src/regression/step.rs`, registry entry `stepwise`):
`test_validate_step_legacy` and `test_validate_step_borderline` pin the selected
variables (exact), the final criterion of each direction and of backward BIC (through
`run_step`, the MCP/CLI wrapper), and the full model's t, logLik, AIC and BIC (1e-6
relative). Measured: criteria within 2.7e-15 relative, t within 1.7e-14.

## Known Differences

1. **Solve path** (REG-18, fixed in T3-WP09): designs whose equilibrated X'X has a
   reciprocal condition below 1e-8 are solved by pivoted QR, so the digits the normal
   equations lost there (Longley 3.6e-8, `ols_illcond` 2.8e-5) are recovered; see the
   section below. Well-conditioned designs keep the normal equations.
2. **No-intercept models** report a centred R² (REG-19, Tier 4); R uses the uncentred one.
3. **Tail p-values** below about 1e-16 are printed as 0 (X-17, Tier 3); the tests compare
   p-values with a 1e-12 absolute floor.
4. **Stepwise**: backward and both never drop to the intercept-only model (REG-54,
   Tier 4), and an unknown direction string in `run_step` becomes "both" (REG-65, Tier 4).
   The cases do not reach either.

## Running the Tests

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- regression::ols::tests::test_validate
cargo test -p p2a-core --lib --no-default-features --features ci -- regression::step::tests::test_validate
cargo test -p p2a-core --lib --no-default-features --features ci -- traits::estimator::tests
P2A_VALIDATION_VERBOSE=1 cargo test -p p2a-core --lib --no-default-features --features ci -- test_validate_ols_longley --nocapture
```

## References

- Longley, J.W. (1967). "An Appraisal of Least Squares Programs for the Electronic Computer from the Point of View of the User". *Journal of the American Statistical Association*, 62(319), 819-841.
- Akaike, H. (1974). "A new look at the statistical model identification". *IEEE Transactions on Automatic Control*, 19(6), 716-723.
- Venables, W.N. & Ripley, B.D. (2002). *Modern Applied Statistics with S* (4th ed.). Springer. (`step`, `extractAIC`)
