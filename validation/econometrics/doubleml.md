# DoubleML Validation

## Method: Double/Debiased Machine Learning (PLR, IRM, PLIV, IIVM)

### Implementation

- **File**: `crates/p2a-core/src/econometrics/doubleml.rs`
- **Functions**: `run_double_ml` (PLR, IRM) and `run_double_ml_iv` (PLIV, IIVM; an instrument `z`).
  Each refuses the other's models, naming the other entry point.
- **MCP Tool**: `treatment_double_ml` (PLR and IRM; the instrument, learner and fold fields are not
  in the request yet)
- **`DoubleMLConfig`**: `n_folds` (5; between 2 and n), `model_type`, `treatment_type` (declared,
  checked against d: `Binary` with a d not coded 0/1 is refused; the result reports the type
  detected from d), `ml_method` (OLS, Lasso, Ridge; parsed from `ols | lm | linear | regr.lm |
  least_squares`, `lasso | l1`, `ridge | l2`), `intercept` (true), `seed` (None: 42), `trim`
  (0.01; `0 < trim < 0.5`), `lambda` (0.05; finite, >= 0), `folds` (user labels, one per
  observation, at least two distinct; `n_folds` and `seed` are then ignored). The model parses
  from `plr | partially_linear | partially_linear_regression | doublemlplr`, `irm | interactive |
  interactive_regression | doublemlirm`, `pliv | partially_linear_iv | doublemlpliv`,
  `iivm | interactive_iv | doublemliivm`.

### Models and scores

`psi = psi_a theta + psi_b`, theta solves `mean(psi) = 0`, `J = mean(psi_a)`.

| Model | Nuisances | `psi_a` | `psi_b` | Estimand | `jacobian` (= -J) |
|---|---|---|---|---|---|
| PLR | `l = E[Y\|X]`, `m = E[D\|X]` | `-(D - m)^2` | `(Y - l)(D - m)` | theta | `E[(D - m)^2]` |
| IRM | `g0 = E[Y\|D=0,X]`, `g1 = E[Y\|D=1,X]`, `m = P(D=1\|X)` | `-1` | `g1 - g0 + D (Y - g1)/m - (1 - D)(Y - g0)/(1 - m)` | ATE | 1 |
| PLIV | `l = E[Y\|X]`, `r = E[D\|X]`, `m = E[Z\|X]` | `-(D - r)(Z - m)` | `(Y - l)(Z - m)` | theta (IV) | `E[(D - r)(Z - m)]` |
| IIVM | `g0`, `g1` = `E[Y\|Z=z,X]`; `r0`, `r1` = `P(D=1\|Z=z,X)`; `m = P(Z=1\|X)` | `-(r1 - r0 + Z (D - r1)/m - (1 - Z)(D - r0)/(1 - m))` | `g1 - g0 + Z (Y - g1)/m - (1 - Z)(Y - g0)/(1 - m)` | LATE | `-E[psi_a]` (first stage) |

A Jacobian that is (numerically) zero is a `SingularMatrix` error: PLR when `E[(D - m)^2] <= 1e-12
Var(D)`, PLIV when `|E[(D - r)(Z - m)]| <= 1e-12 sd(D) sd(Z)`, IIVM when `|E[psi_a]| <= 1e-12`.

### Learners

- **OLS**: OLS's solve decision (`regression::ols::normal_equations_inverse`: the normal equations
  on a well-conditioned design, one inverse per fold for every target; R's pivoted QR, `LM_QR_TOL`,
  otherwise). Aliased covariates are reported ("Coefficients not defined because of
  singularities: x7") and predict as 0, as `predict.lm` does.
- **Lasso**: `regression::glmnet` at `lambda`, `alpha = 1`, standardized, `thresh = 1e-14`
  (`DML_GLMNET_THRESH`; glmnet's default 1e-7 stops about 3e-4 from the minimiser).
- **Ridge**: the closed form of glmnet's ridge objective at `lambda` (no iteration): x centred
  (with an intercept) and divided by its population sd, y scaled by `ys`, penalty `lambda / ys`;
  `(Z'Z/n + lambda/ys I) b = Z'yc/n`, or the dual `n x n` system when a training set has more
  columns than rows. The standardised Gram matrix is shared by the targets of one training set.
- At `lambda = 0` lasso and ridge are least squares and use the OLS learner.
- **Logistic regression** (`econometrics::glm::fit_propensity`, logit) for every binary
  nuisance: IRM's `m`, IIVM's `r0`, `r1`, `m`. IRM and IIVM bound `m` to `[trim, 1 - trim]`.
- **Constant target**: a binary nuisance whose training outcome has no variation (IIVM's `r0` under
  one-sided noncompliance) is predicted as that constant, with the note "the training outcome has
  no variation (all 0); predicted as the constant 0" (DoubleML's `subgroups = list(always_takers
  = FALSE)` result).
- OLS and logistic training sets need more rows than coefficients (`InsufficientData`); lasso and
  ridge accept training sets with more columns than rows.
- Learner notes are merged over folds: `l = E[Y|X] by OLS, folds 1, 2, 3, 4, 5: <note>`.

### Folds

User labels (`folds`; taken in label order, so labels are names, not positions) or a seeded
shuffle: `0..n` shuffled by `ChaCha8Rng::seed_from_u64(seed)` (seed 42 when `None`), the
observation at shuffled position j in fold `j % K`. A call without a seed is reproducible.

### Inference

`dml2` (scores pooled over folds), one repetition: `se = sqrt(mean(psi^2) / J^2 / n)`,
`t = theta / se` (NaN when `se` is 0 or not finite), `p = 2 Phi(-|t|)` from the normal survival
function, 95% CI `theta -/+ qnorm(0.975) se`.

### Diagnostics

- `nuisance: Vec<NuisanceFit>` (PLR `l, m`; IRM `g0, g1, m`; PLIV `l, r, m`; IIVM `g0, g1, r0, r1,
  m`): learner, number evaluated, RMSE and R² of the out-of-fold predictions; `g0, g1, r0, r1`
  over their own arm; `m` of IRM and IIVM after bounding.
- `NuisanceDiagnostics` (all n): outcome / treatment = PLR `l` / `m`; IRM the observed arm's `g` /
  bounded `m`; PLIV `l` / `r`; IIVM `g_Z` / `r_Z`. `mean_residual_treatment` and
  `var_residual_treatment` are the mean and mean square of `D -` the treatment prediction.
- `propensity_range` and `trim` (IRM, IIVM), `lambda` (lasso, ridge), `fold_sizes`, `user_folds`,
  `seed`.
- Warnings (printed through `ResultNotes`, `Warning: ...`): the learner notes; PLR and PLIV
  `E[(D - m)^2] / Var(D) < 0.01` (weak identification) and `|mean(D - m)| > 0.1 sd(D)` (scale-free);
  IRM and IIVM more than n/10 propensities within `2 trim` of 0 or 1.

### R reference

- Case `doubleml_models` in `validation/r/cases/causal_doubleml.R`, dataset
  `validation/datasets/generated/doubleml_sim.csv` (`set.seed(20261201)`, n = 400): covariates
  `X1..X6` (N(0, 1)); continuous `dcont` with outcome `y` (theta 0.7); binary `dbin` (206 treated)
  with `yb` (ATE 1); instrument `z`, treatment `dz`, outcome `yz` (theta 0.5, confounded through an
  unobserved `u`); binary instrument `zb` (201), treatment `db`, outcome `yzb` (LATE 0.9); `fold`,
  five folds of 80.
- DoubleML is not installed (neither R nor Python), so the case is an R transcription of
  DoubleML's published scores with glmnet/glm learners on the fixed folds: `qr.coef` for OLS,
  `glmnet(lambda = 0.05, standardize = TRUE, thresh = 1e-14)` for lasso and ridge,
  `glm(binomial)` for binary nuisances, trimming 0.01, `dml2`. It is not a comparison with the
  DoubleML package.
- `*_tight`: the same estimators with each lasso / ridge fit replaced by the exact minimiser of
  glmnet's objective (lasso: KKT on glmnet's active set to 1e-12; ridge: the closed form). glmnet's
  own output is within 5.1e-12 (lasso theta) and 1.23e-11 (ridge theta) of it; the case asserts
  `|glmnet - tight| < 1e-9`.

| Key | Value | Key | Value |
|---|---|---|---|
| `plr_ols_theta` | 0.7494128231737495 | `plr_ols_se` | 0.04748384663398752 |
| `plr_ols_rmse_l`, `_rmse_m` | 1.25317591965669, 0.992459791484384 | `plr_ols_jacobian` | 0.984976437713227 |
| `plr_lasso_theta_tight` | 0.7422340535147842 | `plr_ridge_theta_tight` | 0.7495780905864603 |
| `irm_theta` | 0.9063182582178073 | `irm_se` | 0.11309006320272504 |
| `irm_ps_range` | 0.18049183708368519, 0.8606417819215899 | `irm_mean_psi_sq` | 5.115744958078538 |
| `irm_lasso_theta_tight` | 0.8875721144878681 | `irm_rmse_m` | 0.500521783630024 |
| `pliv_theta` | 0.3898800815561432 | `pliv_se` | 0.06967001640081959 |
| `iivm_theta` | 0.983389000585263 | `iivm_se` | 0.21023740784193487 |
| `iivm_jacobian` | 0.5485484170755488 | `iivm_onesided_theta` | 0.6689114428798432 |

The case has 96 keys (`validation/expected/doubleml_models.json`).

### Tests and tolerances

R pins (registered in `validation/methods/causal_doubleml.json`; each loads `doubleml_models`):

- `test_validate_doubleml_user_folds`: fold sizes (exact), PLR OLS theta and SE on the user folds;
  relabelled folds give a bit-identical theta.
- `test_validate_doubleml_plr_ols`, `test_validate_doubleml_plr_lasso`,
  `test_validate_doubleml_plr_ridge`: theta, SE, RMSE and R² of `l` and `m`, Jacobian, mean residual,
  CI, p.
- `test_validate_doubleml_irm`, `test_validate_doubleml_irm_lasso`: theta, SE, propensity range,
  every nuisance RMSE / R², the outcome and treatment diagnostics, CI, p; `mean(psi^2)` equals
  `irm_mean_psi_sq` (what the old code printed as the Jacobian) while `jacobian` is 1.
- `test_validate_doubleml_pliv`, `test_validate_doubleml_iivm`, `test_validate_doubleml_iivm_one_sided`.

Tolerances (n = 400): OLS and logistic nuisances at the README rows (`Tol::coef` 1e-8 for theta;
`Tol::se` and `Tol::stat` 1e-6 for SE, RMSE, R², CI, Jacobian, mean residuals, propensity ranges).
Ridge is the closed form, so its `*_tight` keys use the same README tolerances. Lasso is coordinate
descent at thresh 1e-14: `*_tight` at `TIGHT = Tol::new(1e-9, 1e-12)`. glmnet's own output
(`plr_lasso_*`, `plr_ridge_*`, `irm_lasso_*` without `_tight`) at `GLMNET = Tol::new(1e-8, 1e-12)`.
p-values relative, `P_REL = Tol::new(1e-6, 0.0)`: an absolute floor would accept 0 or `1 - cdf`
for tails near 1e-15. The logistic fits stop at `fit_propensity`'s deviance rule (epsilon 1e-10)
where the case's `glm` used 1e-14; the difference is below 2e-12 relative on theta.

Option tests (no case): `test_ml_method_parses_aliases_and_rejects_unknown`,
`test_dml_model_type_parses_aliases_and_rejects_unknown`.

Property tests (read `doubleml_sim.csv` without `load_case`, not registered):
`test_doubleml_default_seed_is_reproducible`, `test_doubleml_rejects_invalid_inputs`,
`test_doubleml_rejects_contradicting_treatment_type`, `test_doubleml_learners_are_not_ols`,
`test_doubleml_nuisance_rmse_is_out_of_fold`, `test_doubleml_aliased_covariate_is_reported`,
`test_doubleml_is_scale_equivariant`, `test_doubleml_display_names_learners_and_jacobian`,
`test_doubleml_ridge_closed_form_matches_glmnet` (closed form against `glmnet(alpha = 0)` at
thresh 1e-14 with and without an intercept, and the dual against the primal),
`test_doubleml_irm_jacobian_is_not_the_score_variance`,
`test_doubleml_irm_rejects_non_binary_treatment`, `test_doubleml_iv_models_need_an_instrument`.

DGP recovery and unit tests (simulated data): `test_doubleml_plr_recovers_dgp`,
`test_doubleml_crossfitting_structure`, `test_doubleml_irm_recovers_dgp`,
`test_doubleml_null_effect_recovers_dgp`, `test_plr_basic`, `test_irm_basic`,
`test_fold_assignments`, `test_add_intercept`, `test_insufficient_data_error`, `test_display`,
`test_reproducibility_with_seed`.

### Differences from DoubleML

- One repetition (`n_rep = 1`) and `dml2` only; the ATE and LATE scores only (no ATTE).
- A fixed `lambda` with no learner tuning; glmnet always standardized.
- PLR's `m` (and PLIV's `r` and `m`) are fitted by the regression learner even for a 0/1
  treatment (with OLS a linear probability model), as DoubleML does with a regression `ml_m`.
- The constant-target rule instead of DoubleML's `subgroups` option.

### What changed in T3-WP30

- **CAUS-103**: `ml_method` was never read; every nuisance was OLS while the output said "Lasso".
  On the case folds lasso and ridge gave the OLS theta 0.7494128232; they now give 0.7422340535 and
  0.7495780906. The IRM propensity was a linear probability model clipped to [0.01, 0.99] (theta
  0.9040643003, propensities [0.159, 0.904]); it is logistic now (theta 0.9063182582, [0.180,
  0.861]). A non-0/1 IRM treatment, dichotomised at 0.5 with a warning before, is refused.
- **CAUS-60**: nuisance RMSE and R² were summed over each fold's training rows (every observation
  counted K - 1 times, R² in-sample): outcome RMSE 2.4513 and treatment RMSE 1.9492 on the case
  folds, now 1.2532 and 0.9925 out of fold (R² 0.6094 / 0.2925 in-sample, now 0.5918 / 0.2670).
- **CAUS-104**: IRM printed `mean(psi^2)` (5.19 with the old propensity) as "Jacobian (Var(D-m))";
  the ATE score's Jacobian is 1 and `var_residual_treatment` is `E[(D - m)^2]` (0.2505).
- Also: PLIV and IIVM implemented; folds from the user or a seed (no entropy); option parsing;
  scale-free warnings; normal survival-function p-values and `qnorm(0.975)` CIs.

## MCP extraction (T2-WP11)

The MCP tool `treatment_double_ml` reads the outcome, the treatment and the covariates in one
request to the shared extractor (`p2a_core::data::columns`, through
`crate::tools::common::numeric_columns`) with the `Fail` policy, because DoubleML refuses
missing data. A missing (null or NaN) value in any of them is a tool error naming the column
(`Column 'x' contains 1 missing value(s) ...`); an absent column or a text column is a tool
error naming it as well, never a protocol error. Integer and boolean columns are numeric, so an
Int64 0/1 treatment is accepted. Before T2-WP11 the handler dropped nulls from y and d per
column but read missing covariates as 0, and then reshaped the covariates with the shortened
row count, so one missing outcome shifted every covariate column; an integer treatment was
refused with a protocol error (findings TOOL-5, TOOL-6).

Tests (`crates/p2a-mcp/src/tools/handlers/causal.rs`, on `validation/datasets/generated/rd_sim.csv`;
not registered, no R value): `test_mcp_treatment_double_ml_accepts_int64_treatment` (the Int64
treatment gives the same output as the same values in Float64) and
`test_mcp_treatment_double_ml_refuses_missing_values` (a missing covariate, a missing outcome and
treatment, an absent covariate).

### References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J.
  (2018). "Double/debiased machine learning for treatment and structural parameters." *The
  Econometrics Journal*, 21(1), C1-C68.
- Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). "DoubleML - An Object-Oriented
  Implementation of Double Machine Learning in Python." *Journal of Machine Learning Research*,
  23(53), 1-6.
- Bach, P., Chernozhukov, V., Kurz, M. S., Spindler, M., & Klaassen, S. (2024). "DoubleML: An
  Object-Oriented Implementation of Double Machine Learning in R." *Journal of Statistical
  Software*, 108(3).
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). "Regularization paths for generalized linear
  models via coordinate descent." *Journal of Statistical Software*, 33(1), 1-22.

### Status

- [x] PLR, IRM, PLIV, IIVM
- [x] OLS, lasso and ridge learners; logistic regression for binary nuisances
- [x] User or seeded folds; out-of-fold nuisance diagnostics
- [x] R reference: pinned to the published scores written out in R (DoubleML not installed)
- [x] MCP tool (PLR and IRM)
