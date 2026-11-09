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
