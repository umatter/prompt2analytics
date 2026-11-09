# Validation: CLI causal, time-series, ML, spatial and viz commands

## Method Overview

T3-WP61 typed every option string of `p2a causal`, `p2a ts`, `p2a ml`, `p2a spatial` and
`p2a viz` (CLI-14): an option value is parsed by clap through the core option table of its
enum, so the R, CLI and MCP spellings are accepted case-insensitively ('-', '.', '_' and
spaces are equivalent) and anything else is a usage error (exit 2) that lists the accepted
values. Before, an unknown value ran a default with exit 0 (or, for `causal rd`, was refused
only after the session was opened, exit 1). The commands also print and serialise the
notes the core results carry (`Converged: ...` and `Warning: ...` lines in text; `warnings`
always, `converged` and `iterations` where reported, in JSON) and the fields the wave-2
estimators added.

The tests run the `p2a` binary (`crates/p2a-cli/tests/tier3/wp61_other.rs`) against the R
cases of the estimators' own validation (registry `validation/methods/cli_other.json`). JSON
values are compared at the `validation/README.md` tolerances for the case's n, printed text at
its printed precision, iterative fits at the tolerance of the estimator's own validation.

## Commands and pins

### `causal rd`, `causal fuzzy-rd`

| Option | Values (canonical first; aliases) | R argument |
|---|---|---|
| `--kernel` | triangular (tri), epanechnikov (epa), uniform (uni) | `rdrobust(kernel = )` |
| `--bwselect` | mserd, msetwo, msesum, msecomb1, msecomb2, cerrd, certwo, cersum, cercomb1, cercomb2 | `rdbwselect(bwselect = )` |
| `--vce` | nn, hc0, hc1, hc2, hc3 | `rdrobust(vce = )` |

`fuzzy-rd` gained `--bwselect` (it always ran mserd). Both print `Kernel:` and `Bandwidth
source:` lines and serialise `bwselect`, `bandwidth_source` and `diagnostics` (identification
warnings of severity Caution and above); `warnings` leaves out the missing-value note, which
has its own `missing` key.

| Test | Case, keys | Tolerance |
|---|---|---|
| `test_validate_cli_rd_bwselect_certwo` | `rd_bwselect_sharp` `tri_p1_nn_certwo` (h and b, left and right) | `Tol::coef(1000)` |
| `test_validate_cli_rd_kernel_alias` | `rd_bwselect_sharp` `epa_p1_nn_mserd` (`--kernel epa`) | `Tol::coef(1000)` |
| `test_validate_cli_fuzzy_rd_bwselect` | `rd_bwselect_fuzzy` `fuzzy_tri_p1_nn_certwo` (`--bwselect cer_two`) | `Tol::coef(1000)` |

`test_cli_unknown_kernel_is_rejected` checks the usage errors (no R value).

### `causal staggered-did`, `causal did`

| Option | Values (canonical first; aliases) | R argument |
|---|---|---|
| `--comparison` | never_treated (nevertreated, never), not_yet_treated (notyettreated, notyet, nyt) | `att_gt(control_group = )` |
| `--method` | outcome_regression (or, reg, regression), ipw (std_ipw), doubly_robust (dr, aipw, drdid) | `att_gt(est_method = )` |
| `--anticipation` (new) | whole number, default 0 | `att_gt(anticipation = )` |

The staggered JSON gains `covariates`, `anticipation`, `diagnostics` and the notes; `causal did`
gains `df` (n minus the rank), `diagnostics` and the notes (its `aliased` key and R's
singularity warning are the estimator's).

| Test | Case, keys | Tolerance |
|---|---|---|
| `test_validate_cli_staggered_accepts_mcp_spelling` | `did_cs_analytic` `notyet_simple_att`, `notyet_simple_se` (`--comparison not_yet_treated --bootstrap 0`) | `Tol::coef(720)`, `Tol::se(720)` |
| `test_validate_cli_staggered_anticipation` | `did_cs_covariates` `never_ant1_simple_att_dr`, `never_ant1_simple_se_dr` (`-x x1 x2 --method DR --anticipation 1 --bootstrap 0`) | `Tol::iterative(1e-8)` (glm IRLS propensity) |
| `test_validate_cli_did_aliased` | `did_2x2_training` `aliased_df_resid` (exact), `aliased_coef` and `aliased_se_hc1` of `training_group:post_training` | `Tol::coef(1200)`, `Tol::se(1200)` |

`test_cli_staggered_help_names_the_default` and `test_cli_staggered_short_help_lists_flags`
check the help (the default `or` is named next to `outcome_regression`; CLI-32).

### `causal ipw`, `doubly-robust`, `tmle`, `stdreg`, `iv`, `mediation`

| Command, option | Values (canonical first; aliases) | R argument |
|---|---|---|
| `ipw --estimand`, `doubly-robust --estimand` | ate, att | `weightit(estimand = )` |
| `doubly-robust --method` | aipw (dr, doubly_robust), ipw, regression (reg, ra, outcome_regression) | |
| `tmle --qmodel` | logistic (logit, binomial, binary), linear (gaussian, continuous, ols) | `tmle(Q.SL.library = )` family |
| `stdreg --model` | linear (ols, gaussian, continuous), logistic (logit, binomial, binary), poisson (count, counts, log_linear) | `glm(family = )` |
| `stdreg --estimand` | ate, att, atc, levels (potential_outcomes) | |
| `stdreg --se-method` | bootstrap (boot), delta (delta_method), sandwich (robust) | |

`causal ipw` defaults to `--bootstrap 0`: the M-estimation SE with the propensity score
estimated (WeightIt's `vcov = "asympt"`). `--bootstrap N` (N >= 2) bootstraps the weighted means
with the fitted propensity scores held fixed, which is conservative. `doubly-robust` keeps 999
replicates for `--method ipw` and `regression`; AIPW uses its analytic influence-function SE.
IPW and doubly robust print `SE method:` and serialise `se_method` (IPW also `se_mest`,
`se_naive`, `se_eif`); every command prints and serialises the core notes, IPW and IV the
identification diagnostics, TMLE both fluctuation coefficients.

| Test | Case, keys | Tolerance |
|---|---|---|
| `test_validate_cli_ipw_dr_se_method` | `ipw_aipw_sim` `ipw_ate_hajek`, `ipw_ate_se_mest`, `ipw_ate_se_eif`, `aipw_ate`, `aipw_ate_se` (`--trim 0`) | `Tol::coef(300)`, `Tol::se(300)` |
| `test_validate_cli_tmle_qmodel_alias` | `tmle_glm` `linear_ate`, `linear_se` (`--qmodel gaussian`) | `Tol::coef(300)`, `Tol::se(300)` |
| `test_validate_cli_stdreg_r_spellings` | `stdreg_glm` `logit_ate`, `logit_ey1`, `logit_ey0` (`--model binomial`) | `Tol::iterative(1e-8)` |

Deviation: the stdreg SEs are not pinned (CAUS-39). `test_cli_causal_json_carries_notes` checks
that the IV and mediation JSON carry `warnings` (IV also `diagnostics` and `first_stage`,
mediation `missing`); it pins no value.

### `causal matching`, `causal synth`, `causal gsynth`

| Command, option | Values (canonical first; aliases) | R argument |
|---|---|---|
| `matching --method` | nearest (nn, nearest_neighbor), cem (coarsened, coarsened_exact), full (optimal, full_matching), subclass (subclassification, stratify) | `matchit(method = )` |
| `matching --distance` | logit (glm, logistic, ps), probit, mahalanobis, euclidean | `matchit(distance = )` |
| `matching --m-order` (new) | largest (descending), smallest (ascending), data (data_order), random; default largest for logit/probit, data otherwise | `matchit(m.order = )` |
| `synth --v-method` | datadriven (data_driven, optimal, nested), equal (uniform); custom V is MCP-only | `synth(custom.v = )` |
| `gsynth --force` | none, unit (individual), time, two_way (twoway) | `gsynth(force = )` |
| `gsynth --estimator` | ife (gsynth, interactive_fixed_effects), mc (matrix_completion) | `gsynth(estimator = )` |

`--method cem` uses MatchIt's default Sturges bins (it used four quantile bins). Matching prints
its Display (both balance tables and the notes) once; its JSON gains `m_order`, `ess_treated`,
`ess_control`, `total_distance`, `caliper_abs`, `diagnostics` and the notes. Synth's JSON gains
`all_unit_weights` (every donor) and `predictor_loss`; gsynth's names the estimator and the
fixed effects by their table names (`ife`, `mc`; `two_way`; they were Rust Debug names) and
gains `beta`, `covariate_names`, `cv_mspe`, `cv_lambda`, `lambda`, `sigma2`, `mu`,
`dynamic_counts`, `dynamic_se` and `bootstrap`.

| Test | Case, keys | Tolerance |
|---|---|---|
| `test_validate_cli_matching_options` | `matchit_nearest` `ratio2_ess` (control, treated; `--ratio 2 --distance glm`) | `Tol::coef(240)` |
| `test_validate_cli_synth_v_method_alias` | `synth_weights_qp` `equal_w` (by donor), `equal_predictor_loss` (`--v-method uniform`) | `Tol::coef(14)`, `Tol::stat(14)` |
| `test_validate_cli_gsynth_r_spellings` | `gsynth_simdata` `ife_r2_cov_att_avg`, `ife_r2_cov_beta` (`--n-factors 2 -x X1 X2`) | relative 5e-4 |

Deviation: gsynth is compared at 5e-4 relative, the gap between p2a's default EM stopping rule
and R's converged fit, as in the estimator's own validation.

### `ts changepoint`, `holt-winters`, `ar`, `decompose`, `struct-ts`, `spectrum`, `granger`, `causal-impact`

The core enums of these options have no `FromStr`, so their tables live in
`commands/timeseries.rs`; they follow the same matching rules.

| Command, option | Values (canonical first; aliases) | R argument |
|---|---|---|
| `changepoint --change-type` | mean, variance (var), meanvar (both, mean_and_variance, mean_var) | `cpt.mean` / `cpt.var` / `cpt.meanvar` |
| `holt-winters --seasonal` | additive (add), multiplicative (mult, multi) | `HoltWinters(seasonal = )` |
| `ar --method` | yule-walker (yw), burg, ols | `ar(method = )` |
| `decompose --decompose-type` | additive (add), multiplicative (mult, multi) | `decompose(type = )` |
| `struct-ts --model-type` | level (local_level), trend (local_trend, llt), BSM (basic_structural) | `StructTS(type = )` |
| `spectrum --spans` | comma-separated integers; any other entry is a usage error | `spec.pgram(spans = )` |

The JSON reports the canonical names (`change_type`, `seasonal_type`, `method`,
`decompose_type`, `model_type`; they were Rust Debug names such as `Ols`). `granger`, `ar`,
`struct-ts` and `causal-impact` print and serialise the core notes; the `--penalty` help of
`changepoint` names its default, changepoint's MBIC.

| Test | Case, keys | Tolerance |
|---|---|---|
| `test_validate_cli_granger_notes` | `granger_short` `lag2_F`, `lag2_p`, `lag2_df2` (`--lags 2`) | `Tol::stat(20)`, `Tol::p_value(20)`, exact |
| `test_validate_cli_struct_ts_notes` | `structts_nile_ukgas` `ukgas_loglik_tight`; `ukgas_coef_tight` seasonal and observation variances (`--model-type bsm --period 4`) | `Tol::stat(108)`; `Tol::iterative(1e-6)` |

Not compared: the BSM level variance, which is 0 at R's optimum (a boundary a relative
tolerance cannot compare), and the slope variance next to it; the estimator's own validation
pins all four.

### `ts holt-winters --horizon`

`--horizon` was parsed and never used. With a horizon the command prints and serialises
`predict.HoltWinters`' forecasts and limits at `--level` (default 0.95), in text
`Forecasts (h = 12, 95% prediction interval):` and `  t+i: fit  [lower, upper]`, in JSON
`forecast: {horizon, level, mean, std_errors, lower, upper, residual_variance}`. New
`--alpha`, `--beta`, `--gamma` fix the smoothing parameters as `HoltWinters(alpha, beta,
gamma)` does (0 for beta or gamma fits no trend or season, R's `FALSE`); without them the
parameters are optimised.

A multiplicative seasonal fit gets point forecasts only: R's interval formula for that branch
is wrong (deviation TS-22), and the core refuses it. The text says
`Forecasts (h = 12; prediction intervals are implemented for additive and non-seasonal
fits):`, the JSON has `level`, `std_errors`, `lower` and `upper` null and a line in
`warnings`.

| Test | Case, keys | Tolerance |
|---|---|---|
| `test_validate_cli_holt_winters_horizon_prints_forecasts` | `ts_holtwinters_forecast` `airpass_mult_fit`, `airpass_mult_resvar` (alpha 0.3, beta 0.05, gamma 0.4, multiplicative); `airpass_notrend_fit`, `_lwr`, `_upr` (alpha 0.3, beta 0, gamma 0.2, additive) | JSON `Tol::coef(144)` / `Tol::stat(144)`; text `Tol::printed(4)` |

Deviations: TS-22 (`airpass_mult_lwr`, `airpass_mult_upr` not compared); TS-24 (the optimiser
is not pinned; the optimised run's forecasts are compared with the library in process).

### `ts irf`, `ts var`

`ts irf` printed `a -> b` over `irf[h][a][b]`, which is the response of `a` to a shock in `b`,
so every off-diagonal series was attributed to the wrong shock, and it showed two variables
and five horizons. It now prints the core report: one table per impulse, rows the horizons
0..steps-1, columns the responses (`--steps` counts the impact, vars' `n.ahead = steps - 1`).
The JSON keeps `irf` (`[horizon][response][impulse]`), names its axes (`axes`) and adds
`impulse_responses` (`{impulse: {response: [h = 0, 1, ...]}}`).

`ts var` prints the core VAR report (coefficient tables with SEs, t and p values per equation,
residual df, residual covariance). Its JSON adds `coef_names`, `std_errors`, `t_values`,
`p_values`, `df_residual`, `sigma_u_ml` and `log_det_sigma_ml`. `sigma_u` is vars' residual
covariance U'U / (T - Kp - 1); `sigma_u_ml` is U'U / T.

| Test | Case, keys | Tolerance |
|---|---|---|
| `test_validate_cli_irf_labels_as_vars` | `var_canada` `irf_ortho_e`, `_prod`, `_rw`, `_U` (13 horizons, text); `irf_ortho_e` (JSON, prod's response) | `Tol::printed(6)`; `Tol::stat(82)` |
| `test_validate_cli_var_json` | `var_canada` `coef_e`, `se_e`, `p_U`, `loglik`, `sigma_ml_logdet`, `df_resid` | `Tol::coef(82)`, `Tol::se(82)`, `Tol::p_value(82)`, `Tol::stat(82)`, exact |

### `ts garch --horizon`, `--distribution`

The text branch printed the last h fitted in-sample variances in reverse time order labelled
t+1..t+h, and the JSON ignored `--horizon`. Both now report `garch_forecast`, the GARCH(p, q)
variance recursion from the last fitted variance: text `  t+i: variance v, volatility s`
after the core GARCH report, JSON `forecast: {horizon, variance, volatility, mean}`. New
`--distribution`: norm (normal, gaussian) or std (t, student_t, student, studentt), fGarch's
`cond.dist`. The JSON adds `mu`, `param_names`, `estimates`, `std_errors`, `p_values`,
`shape` (null for normal errors), `distribution` and the notes.

| Test | Case, keys | Tolerance |
|---|---|---|
| `test_validate_cli_garch_horizon_forecasts` | `garch_normal_t` `norm_forecast_var_tight` (JSON and text) | relative 1e-8 (JSON), 1e-6 (text, 7 significant digits) |
| `test_validate_cli_garch_student_t` | `garch_normal_t` `std_par_tight` shape, `std_forecast_var_tight` | relative 1e-8 |

### `ts varma --long-var-order`, `ts vecm --ecdet`, `--rank`

`ts varma` takes the order m of Hannan-Rissanen's step-1 long VAR (`--long-var-order`; default
max(floor(2 ln n), 2 max(p, q)), 0 when q = 0) and prints the core VARMA report with its step-2
table. The JSON adds `estimator`, `var_names`, `long_var_order`, `regressor_names`,
`coefficients`, `std_errors` (empty for q > 0, where the step-2 OLS SEs are not valid),
`t_values`, `p_values`, `df_residual`, `n_params`, `aliased` and the notes.

`ts vecm` defaults to K = 2 (urca refuses K = 1), takes `--ecdet` (none (default), const
(constant), trend), and chooses the rank by the trace test at 5% unless `--rank` is given. It
prints urca's tables with the Osterwald-Lenum critical values and serialises every field of
the Johansen result (`"method": "VECM"` first). Only the transitory form is estimated.

| Test | Case, keys | Tolerance |
|---|---|---|
| `test_validate_cli_varma_long_var_order` | `varma_hr` `long_var_order`, `n_eff`, `df_residual`, `loglik`, `coef_y1` (m = 8); `default_long_var_order`, `default_loglik` | exact; `Tol::stat(391)`, `Tol::coef(391)`; `Tol::stat(388)` |
| `test_validate_cli_vecm_options` | `vecm_denmark` `none_trace_stat`, `none_beta` (ecdet none, rank chosen 1); `trace_stat` (ecdet const, rank chosen 0) | `Tol::stat(55)`, `Tol::coef(55)` |

### `spatial sar`, `sem`, `sac`, `moran`

| Command, option | Values (canonical first; aliases) | R argument |
|---|---|---|
| `--weight-style` (all four) | W (row, rowstd, row_standardised), B (binary), C (global, globalstd), U (unit_sum, unitsum), minmax (mm, min_max), S (varstab, variance_stabilising) | `spdep::nb2listw(style = )` |
| `moran --alternative` | two.sided (two_sided, ...), greater, less | `moran.test(alternative = )` |

An unknown style ran row standardisation and an unknown alternative the two-sided test; both
are now usage errors. The SAR impacts carry the core's warnings (text: a `Warning:` line after
the table; JSON: `impacts.warnings`, always present), so a degraded impact computation (a
singular multiplier) no longer looks like a real one (SPAT-25). SAR, SEM and SAC report the
style (`weight_style`, spdep's letter), `k_neighbors` and the optimiser's notes; Moran reports
the alternative by its R name and `n_islands`.

| Test | Case, keys | Tolerance |
|---|---|---|
| `test_validate_cli_spatial_impacts_warnings_in_json` | `spatial_sar_sdm_sem` `sar_rho`, `sar_impact_direct`, `_indirect`, `_total` | `Tol::coef(80)` |
| `test_validate_cli_spatial_weight_style_b` | `spatial_sar_sdm_sem` `sar_rho` (B: rho_W / 4; U: 80 rho_W on the 4-NN graph of 80 rows), `sar_loglik` | `Tol::coef(80)`, `Tol::stat(80)` |
| `test_validate_cli_moran_alternative` | `spatial_weights_moran` `moran` I, p_two_sided (two.sided), p_rand (greater) | `Tol::coef(80)`, `Tol::p_value(80)` |

Styles B and U reparameterise rho: on a k-nearest-neighbour graph every row of B sums to k, so
W = B / k and U = B / (n k) = W / n, and the fit is W's with rho scaled.

### `ml pca`, `ml hierarchical`, `ml random-forest`, `ml svm`, `ml kmeans`

`ml pca --transform` computed the scores and printed none of them. The scores (prcomp's `x`,
centred and unscaled) now appear in text (first 10 rows) and JSON (`scores`, every row), and
`--scores-file <CSV>` writes all of them (header `PC1,..,PCk`; implies `--transform`). The
sign of a component is arbitrary, so the test aligns each score column to R's by the sign of
their inner product.

`ml hierarchical --linkage` takes hclust's eight names: single, complete, average (upgma),
ward.D2 (ward, ward2; the default), mcquitty (weighted, wpgma), ward.D (ward1), centroid
(upgmc), median (wpgmc, gower). R's deprecated `hclust(method = "ward")` is ward.D; p2a's `ward`
is ward.D2. `ml random-forest --max-features` accepts sqrt, log2, all or an integer >= 1 and
refuses anything else (the core read it as sqrt). `ml svm` sorts its weight table on a total
order (non-finite weights last). `ml kmeans` prints and serialises its convergence and
warnings.

| Test | Case, keys | Tolerance |
|---|---|---|
| `test_validate_cli_pca_transform` | `cli_pca_scores` `scores` (60 x 3, sign-aligned), `sdev` | `Tol::coef(60)`, `Tol::stat(60)` |
| `test_validate_cli_linkage_names` | `hclust_methods` `average_cutree3` (same partition), `average_height` (`--linkage upgma`) | exact; `Tol::coef(60)` |

### `viz residuals`, `viz dendrogram`

`viz residuals` fitted the OLS model and passed no leverage to the residual diagnostics, so the
standardized residuals and Cook's distance of a model with several regressors came from the
one-predictor formula (Cook's D of the first mtcars row 0.037 instead of R's 0.019 for
`mpg ~ wt + hp + qsec`). It now fits by pivoted QR (`lm.fit`), passes the hat values
x_i' (X'X)^-1 x_i and the rank to the diagnostics, and in `-F json` prints them with the four
files it wrote (`files`, `n_obs`, `n_params`, `sigma`, `leverage`, `standardized_residuals`,
`cooks_distance`, notes). `viz dendrogram --linkage` takes hclust's eight names (it ran Ward
for anything it did not know) and its JSON names the linkage it ran.

| Test | Case, keys | Tolerance |
|---|---|---|
| `test_validate_cli_viz_residuals_hat_values` | `cli_viz_residuals` `hat`, `rstandard`, `cooks`, `sigma`, `rank`, `n` | `Tol::coef(32)`, `Tol::stat(32)`, exact |

`cli_viz_residuals` (`validation/r/cases/cli_other.R`) reads `cli_mtcars.csv`, the four model
columns of mtcars, and checks that its hat values and fitted values equal the committed
`viz_residual_diag.csv` model.

## Help and spellings

Two unit tests in `crates/p2a-cli/src/commands/causal.rs` hold the options to their tables:
`test_option_help_lists_every_choice` requires the `--help` text of every typed option to name
each canonical value of its table (so the help cannot drift when an enum grows), and
`test_options_accept_r_cli_and_mcp_spellings` parses an R, CLI or MCP spelling of each option.

## Deviations

| Finding | Quantity | Note |
|---|---|---|
| CAUS-39 | stdreg SEs | Tier 4; only point estimates are pinned |
| TS-22 | multiplicative Holt-Winters prediction intervals | R's interval formula for that branch is wrong; the core refuses it, the CLI prints point forecasts and a warning |
| TS-24 | optimised Holt-Winters parameters | the optimiser is not pinned; the pins fix alpha, beta, gamma |
| (none) | gsynth ATT and coefficients | compared at 5e-4 relative: p2a's default EM stopping rule against R's converged fit |
| (none) | StructTS BSM level and slope variances | the level variance is 0 at R's optimum (a boundary); the estimator's validation pins all four |
| (none) | `--linkage ward` | p2a's `ward` is ward.D2; R's deprecated `hclust(method = "ward")` is ward.D (`--linkage ward.D`) |

Spatial weight style U is fitted on spatialreg's full search interval and pinned through its
reparameterisation of the row-standardised fit (rho_U = 80 rho_W on the case's graph).
