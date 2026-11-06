# Covariate Balancing Propensity Score (CBPS) Validation

Status: R-pinned with known deviations (registry `validation/methods/causal_cbps.json`, method
`causal_cbps`).

## Method

`run_cbps` / `cbps` (`crates/p2a-core/src/econometrics/cbps.rs`) compute
`CBPS::CBPS(treat ~ x, method, ATT, standardize = TRUE, twostep = TRUE)` of the R package CBPS
(Fong, Ratkovic and Imai; version 0.24):

| p2a | R | Moments |
|---|---|---|
| `CbpsMethod::Over` (default) | `method = "over"` (R's default) | k score and k balance conditions, two-step GMM |
| `CbpsMethod::Exact` | `method = "exact"` | k balance conditions (just identified) |
| `CbpsMethod::Logit` | `glm(treat ~ x, binomial)` | k score conditions |

`CbpsConfig.estimand` selects the balance target: `CbpsEstimand::Ate` is R's `ATT = 0` (both
groups weighted to the full sample), `CbpsEstimand::Att` is `ATT = 1` (controls weighted to the
treated). The library default is ATE (p2a's behaviour so far and the default of
`WeightIt::weightit(method = "cbps")`); `CBPS::CBPS` defaults to `ATT = 1`. Only
`standardize = TRUE` and `twostep = TRUE` are implemented.

The legacy names `CbpsMethod::ExactBalance` and `CbpsMethod::OverBalance` are constants equal to
`Over` (both always ran the score and balance GMM), `CbpsMethod::JustIdentified` equals `Logit`;
serde reads the old variant names. User strings are parsed with `FromStr`: `over`
(`overidentified`, `overbalance`), `exact` (`balance_only`), `logit` (`glm`, `just`,
`just_identified`); `ATE`, `ATT` (`treated`).

## Estimator

`x` is the n x k design with the intercept first, `t` in {0, 1}, `n_t` treated.

1. **Whitening** (`CBPS.fit`). The non-intercept columns are standardised (mean, `sd` with
   n - 1), `Xs = [1, (x_j - m_j) / s_j]`, and `Xs = U diag(d) V'` is its thin SVD. Everything
   below runs on `U`; the coefficients map back by `beta = T beta_u`, `T = M V diag(1/d)`, where
   `M` maps standardised to original coefficients (`M[0][j] = -m_j / s_j`, `M[j][j] = 1 / s_j`).
   The fit is therefore invariant to the units of a covariate.
2. **Moments** (`CBPS.2Treat`, `gmm.func`). `p = plogis(U beta)` clipped to
   `[1e-6, 1 - 1e-6]` (`probs.min`). Score `g_s = U'(t - p) / n`. Balance `g_b = U'w / n` with
   ATE `w = 1 / (p - 1 + t)` (1/p treated, -1/(1-p) controls) and ATT
   `w = (n / n_t) (t - p) / (1 - p)`. The Jacobian is the exact derivative of these moments
   (`-U' diag(p(1-p)) U / n` for the score; `U' diag(dw/dtheta) U / n` for the balance block).
3. **Moment covariance** (`gmm.func` with `invV = NULL`). With `B(h) = U' diag(h) U / n`:
   ATE `V = [[B(p(1-p)), B(1)], [B(1), B(1/(p(1-p)))]]`; ATT, `s = n / n_t`:
   `V = [[B(p(1-p)), s B(p)], [s B(p), s^2 B(p/(1-p))]]`.
4. **Start values.** `b0` is `glm(treat ~ U - 1, binomial)` at `glm.control`'s defaults; the
   start is `alpha b0`, where `alpha` minimises `f(alpha) = g' V^+ g` at `alpha b0` over
   `[0.8, 1.1]` (Brent's method, the algorithm of `optimize`, at tolerance 1e-12). The weight
   matrix `W = V(alpha b0)^+` is fixed there (`twostep = TRUE`).
5. **Exact** (`opt.bal`): damped Newton on `g_b = 0` with an Armijo line search on the balance
   loss `g_b' g_b` (R minimises the same loss by BFGS; same root).
6. **Over** (`gmm.glm.init`, `gmm.bal.init`): Gauss-Newton on `Q = gbar' W gbar` from `alpha b0`
   and from the exact-balance solution, Armijo line search on Q; the run with the smaller Q is
   kept (ties: the exact-balance start, as R). Both solvers stop when the relative Newton step is
   below `tolerance` (default 1e-8) and report a convergence record and a warning otherwise.
   Gauss-Newton converges only linearly when the over-identified model does not fit exactly,
   and on some samples `g'Wg` stops decreasing in floating point before the step reaches 1e-8
   (measured: a relative step of about 2e-8 to 6e-8 on simulated n = 100 to 10,000). It
   therefore also stops, converged, when its predicted decrease `r's` is below 1e-14 of
   `g'Wg`, or when no step lowers `g'Wg` and the predicted decrease is below 1e-10 of it
   (R's `optim` stops at a relative objective change of 1e-8).
7. **Reported.** `cbps_j` is CBPS's `J = gbar' W gbar` at the estimate (no factor n; for
   `Exact` too). For `Over`, `j_statistic = n J`, `j_df = k` and `j_p_value` is Hansen's test
   `P(chi2_k > n J)`, which CBPS does not print; a rejection at 5% adds a warning. Weights are
   CBPS's standardised weights (`standardize = TRUE`): ATE `(1/p) / sum_T(1/p)` treated,
   `(1/(1-p)) / sum_C(1/(1-p))` controls; ATT `1 / n_t` treated,
   `(p/(1-p)) / sum_C(p/(1-p))` controls; each group sums to 1. `deviance` is
   `-2 sum[t ln p + (1 - t) ln(1 - p)]`.
8. **Variance.** On the whitened scale, CBPS.2Treat's sandwich: `Over`
   `V_u = A Omega A'`, `A = (J'WJ)^-1 J'W`, `Omega = sum_i m_i m_i' / n` with
   `m_i = [u_i (t_i - p_i); u_i w_i]`; `Exact` `V_u = J_b^-1 Omega_b J_b^-1`,
   `Omega_b = sum_i w_i^2 u_i u_i' / n`. The coefficient covariance is `T V_u T' / n` (delta
   method), SEs its diagonal, z-tests two-sided normal. `Logit` reports `glm`'s model-based SEs.

## Reference

`validation/r/cases/causal_cbps.R`, case `cbps_sim_methods`
(`validation/expected/cbps_sim_methods.json`), dataset
`validation/datasets/generated/cbps_sim.csv`: n = 400, 178 treated, `set.seed(20261102)`,
`x1, x2 ~ N(0, 1)`, `x3 ~ Bernoulli(0.4)`, `p = plogis(-0.3 + 0.6 x1 - 0.4 x2 + 0.5 x3)`.
R 4.3.3, CBPS 0.24, MASS 7.3.60.0.1.

- `<method>_<estimand>_*`: `CBPS()` as shipped. Its optimisers stop early (BFGS `reltol` 1e-8,
  `optimize` at `tol = eps^0.25`), so these values cross-check p2a at about 1e-4.
- `*_tight_*` (pinned): CBPS's own `CBPS.2Treat` on CBPS.fit's whitened design with four
  patches, each asserted to apply exactly as often as expected (a new CBPS version stops the
  case): (1) `alpha` from `optimize(tol = 1e-12)` refined by bisection on the analytic
  derivative `f'(alpha)`, because the computed objective carries about 1e-12 relative rounding
  noise and `optimize` places alpha only to about 1e-7; (2) `optim(reltol = 1e-15)`; (3) the
  ATT balance block of `gmm.gradient` divided by n instead of n_t (deviation below); (4) the
  start value returned. `*_tight_se` is the delta method `T V_u T' / n` from CBPS.2Treat's
  whitened sandwich; CBPS.fit's own transform of it is recorded as `*_tight_se_reported`.
- `logit_*`: `glm(binomial)` at `epsilon = 1e-14`.

## Values

Names `(Intercept), x1, x2, x3`; full precision in the case JSON.

| Key | Value |
|---|---|
| `exact_ate_tight_coef` | -0.4544247290582656, 0.5480281384079144, -0.4258753785273447, 0.5096186071296206 |
| `exact_ate_tight_se` | 0.1445825443138091, 0.1168980556462086, 0.11496933084986505, 0.2187203464902146 |
| `exact_ate_tight_J` | 0.0005194670706120138 |
| `exact_ate_tight_deviance` | 503.59266165673125 |
| `ate_tight_alpha` | 1.0001994863243322 |
| `exact_att_tight_coef` | -0.4771129366532645, 0.5329270622575533, -0.45784691930842353, 0.5535550387284768 |
| `exact_att_tight_se` | 0.14909887970088395, 0.12633572959423592, 0.13112113229996833, 0.23152236600392156 |
| `exact_att_tight_J` | 0.0029154993467163074 |
| `exact_att_tight_deviance` | 503.68454576283995 |
| `att_tight_alpha` | 0.9873986654473881 |
| `over_ate_tight_coef` | -0.4549221925846244, 0.5449895264767033, -0.43755593064118536, 0.5106604325595238 |
| `over_ate_tight_se` | 0.1432370431526241, 0.10755594836808981, 0.10793919565134198, 0.2143246923574657 |
| `over_ate_tight_J` | 0.00047848306022312285 |
| `over_ate_tight_hansen_p` | 0.9957029830433873 |
| `over_ate_tight_deviance` | 503.5894990871496 |
| `over_att_tight_coef` | -0.47351767228241637, 0.52759664916449, -0.3983787809202583, 0.5013266791271651 |
| `over_att_tight_se` | 0.1425098149636364, 0.10679165397186108, 0.0794225994803616, 0.2088952893091055 |
| `over_att_tight_J` | 0.0017281069517493801 |
| `over_att_tight_hansen_p` | 0.9524044578559931 |
| `over_att_tight_deviance` | 503.7674534974937 |
| `logit_coef` | -0.4620686196109228, 0.5513989456977314, -0.4351705565114065, 0.5173937270404138 |
| `logit_se` | 0.14312543657323068, 0.1126734955535261, 0.11231318518073516, 0.21615702405226117 |
| `logit_deviance` | 503.58325570864935 |
| package `exact_ate_coef` (cross-check) | -0.45446797479449846, 0.5480299250717414, -0.42592384465700645, 0.5096749643009283; J 0.0005191579205108301 |
| package `over_ate_coef` (cross-check) | -0.4549231116085406, 0.5449854365246619, -0.4375532299973303, 0.5106576434503657; J 0.00047849091896059553 |
| package `over_att_coef` (recorded, not compared) | -0.43687846683737697, 0.5325494249710038, -0.46227239075016774, 0.563061490984394; J 0.0024947205855263585 |
| `exact_ate_tight_se_reported` (recorded, not compared) | 0.1436774977818093, 0.11078210010861755, 0.11507977958747449, 0.10941210623844735 |

The scores and weights of the first ten rows, the treated weight sums and the logit weights of
both estimands are pinned as well (`*_ps_head`, `*_weights_head`, `*_weights_sum_treated`,
`logit_{ate,att}_weights_head`).

## Tests and tolerances

R pins (registered; module `econometrics::cbps::tests`):

| Test | Pins |
|---|---|
| `test_validate_cbps_over_ate` | `over_ate_tight_*`, `ate_tight_alpha` |
| `test_validate_cbps_over_att` | `over_att_tight_*`, `att_tight_alpha`; J below 0.75 of CBPS's own (deviation) |
| `test_validate_cbps_exact_ate` | `exact_ate_tight_*`, `ate_tight_alpha` |
| `test_validate_cbps_exact_att` | `exact_att_tight_*`, `att_tight_alpha` |
| `test_validate_cbps_logit` | `logit_coef`, `logit_se`, `logit_deviance`, logit weights (ATE, ATT) |
| `test_validate_cbps_package_cross_check` | CBPS() as shipped: exact ATE, exact ATT, over ATE |

Tolerances (n = 400): exact and logit at the `validation/README.md` values (coefficients, scores,
weights 1e-8; SE and deviance 1e-6 relative); over (GMM, Gauss-Newton to a relative step of
1e-8 against BFGS at `reltol` 1e-15) `Tol::iterative(1e-6)`; alpha `Tol::iterative(1e-6)` and J
`Tol::iterative(5e-6)`, because CBPS's start objective carries about 1e-12 relative rounding
noise, so alpha agrees only to about 1e-7 and J through W to a few times that; the package
cross-check `Tol::iterative` 1e-4 (over ATE), 1e-3 (exact; J 2e-3) and 1e-6 (deviance), because
`CBPS()` stops its optimisers early (measured gaps to the tight refit: 1.14e-4 exact
coefficients and scores, 5.95e-4 exact J, 7.5e-6 / 1.6e-5 over ATE coefficients / J, 1.9e-7
deviance). Measured on this tree: alpha within 1.1e-9, exact coefficients within 1.1e-11, over
coefficients within 2e-8, over SEs within 1e-7, J within 3.3e-9 (relative).

Unit tests (not R pins; they read `cbps_sim.csv` directly or build their own data):
`test_cbps_methods_and_estimands_differ` (the three methods and two estimands give different
coefficients; exact balance holds to 1e-8 in the balance table; ATT treated weights are
`1 / n_t`), `test_cbps_moment_jacobian_matches_finite_differences`,
`test_brent_fmin_finds_the_minimum`, `test_cbps_scale_equivariance` (`x1 * 1e-6`, `x2 * 1e6`),
`test_cbps_over_converges_at_the_rounding_floor`,
`test_cbps_reports_non_convergence`, `test_cbps_logit_reports_glm_warnings` (separation),
`test_cbps_method_parses_aliases_and_rejects_unknown`,
`test_cbps_estimand_parses_aliases_and_rejects_unknown`, `test_cbps_weights_sum_to_one_per_group`,
`test_cbps_display`, `test_cbps_missing_column`, `test_cbps_constant_treatment`,
`test_cbps_rejects_invalid_parameters`, `test_cbps_rejects_non_binary_treatment`,
`test_cbps_rejects_collinear_covariates`, `test_cbps_rejects_too_few_observations`,
`test_balance_table_display`.

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::cbps
```

## Deviations from CBPS 0.24

1. **Over-identified ATT** (CAUS-11). CBPS's `gmm.gradient` writes the ATT balance block as
   `1/n.t * t(X * dw) %*% X`, n/n_t times the derivative of its own moment `(1/n) X'w`, so BFGS
   stops where its objective's gradient is `(3.6e-4, 2.4e-4, -4.2e-5, -3.8e-5)`; the minimiser
   of the same objective has J 0.0017281 against CBPS's 0.0024947. p2a minimises CBPS's objective
   and is pinned to CBPS.2Treat with that line corrected (`over_att_tight_*`); CBPS's own
   `over_att_*` values are recorded, not compared. Exact fits (`bal.gradient`) and ATE fits are
   not affected.
2. **Standard errors** (CAUS-55). CBPS.fit maps CBPS.2Treat's whitened-scale sandwich back with
   `Dx.inv ginv(X'X) X' U V ginv(D) ...`, which is not the Jacobian of its own coefficient map
   and not scale-equivariant: with `x3 * 10` CBPS reports the same x3 SE 0.109412 while the
   coefficient becomes 0.050967. The bootstrap SE of x3 (B = 300) is 0.219845, the delta method
   gives 0.218720 and glm 0.216157. p2a reports `T V_u T' / n`.
3. **Start scale alpha** (CAUS-55). CBPS takes `alpha = optimize(f, c(0.8, 1.1))` at
   `tol = eps^0.25`; p2a takes Brent's minimiser at tolerance 1e-12, within about 1e-7 of the
   reference's refined value (rounding noise of `f`), which moves the over-identified estimates
   by about 1e-8.
4. **Refused inputs** (CAUS-55). Collinear covariates are refused with the aliased columns named
   (pivoted QR at `LM_QR_TOL`; with `x4 = x1 + x2`, CBPS returns the minimum-norm split
   `x1 0.50063, x2 -0.45332, x4 0.030228`), as are standardised designs whose smallest singular
   value is at most 1e-5 (CBPS.fit's threshold). A treatment not coded 0/1 is refused (R treats
   a numeric two-valued treatment as continuous). `tolerance` and `balance_threshold` must be
   finite and positive, `max_iter` at least 1, and n must exceed the number of coefficients.

## Not covered

CBPS's `twostep = FALSE` (continuous updating), `ATT = 2`, `standardize = FALSE`,
`sample.weights`, multi-valued and continuous treatments (`CBPS.3Treat`, `CBPS.4Treat`,
`CBPS.Continuous`), `npCBPS`, CBPS for marginal structural models; `balance()` tables (p2a's
balance table is its own standardised-difference table).

## Legacy

`validation/expected/cbps_weightit.csv` (WeightIt `method = "cbps"`, never asserted) is
superseded by this case (`supersedes_legacy`).

## Missing values and column types

Missing values (null or NaN) in the treatment or covariates are refused, with
"Column 'x' contains k missing value(s) (null or NaN). Drop or impute them first."
(`EconError::NullValues`, naming the column and the count). A text column gives
`NonNumericColumn`, infinite values `NonFiniteValues`, an absent column `ColumnNotFound`; a
column that exists is never reported as not found. Int64 and Boolean 0/1 treatments are
read as numbers and give the same result as Float64. Results on complete data are
unchanged.

R: CBPS 0.24 drops incomplete rows (`model.frame`, `na.omit`) and runs on the rest; p2a
refuses them instead (checked for T2-WP13).

Tests: `test_validate_na_cbps_columns` (case `causal_mediation_na`, registry
`validation/methods/causal_na.json`, id `weighting_na_refusal`).

## References

- Imai, K. & Ratkovic, M. (2014). "Covariate Balancing Propensity Score."
  *Journal of the Royal Statistical Society: Series B*, 76(1), 243-263.
  DOI: 10.1111/rssb.12027
- Fong, C., Ratkovic, M., & Imai, K. (2022). CBPS: Covariate Balancing Propensity Score.
  R package version 0.24. https://CRAN.R-project.org/package=CBPS
- Hansen, L.P. (1982). "Large Sample Properties of Generalized Method of Moments Estimators."
  *Econometrica*, 50(4), 1029-1054.
