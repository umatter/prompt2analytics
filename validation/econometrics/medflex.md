# Validation: Natural Effect Models (medflex)

## Method

`run_medflex` (`crates/p2a-core/src/econometrics/medflex.rs`) is the imputation estimator of
natural effect models (Vansteelandt, Bekaert & Lange 2012) as medflex 0.6-11 implements it
(`neImpute`, `neModel`, `neEffdecomp`). For a 0/1 treatment `A`, mediator `M` and confounders `C`:

1. **Outcome working model** `Y ~ A + M + C [+ A:M]`, fitted by least squares (gaussian) or by
   the shared GLM (`econometrics::glm`, `glm.fit`'s IRLS) for poisson (log) and binomial (logit).
2. **Expanded data.** Every unit gives two rows, `A0 = 0` and `A0 = 1`, with `A1 = A` (the
   observed treatment). Each row's outcome is imputed from the working model at `A = A0` and the
   unit's own `M` and `C`.
3. **Natural effect model** `yhat ~ A0 + A1 + C [+ A0:A1 + A0:C]` on the requested scale:

   | `scale` | natural effect model | default working family (`family = None`) |
   |---|---|---|
   | `difference` | gaussian, identity | gaussian |
   | `ratio` | quasipoisson, log | poisson |
   | `odds_ratio` | quasibinomial, logit | binomial |

   `family` overrides the working model (`gaussian`, `poisson`, `binomial`), e.g. a risk ratio
   or risk difference from a binomial working model. A ratio scale needs non-negative imputed
   outcomes and the odds-ratio scale imputed outcomes in [0, 1]; otherwise the call is an error.
4. **Effects** at the confounder means `cbar` (link scale; exponentiated on the ratio scales):
   `NDE = A0 + sum_j cbar_j A0:C_j`, `NIE = A1 + A0:A1`, `TE = NDE + NIE`. medflex's
   `neEffdecomp` evaluates at `covLev = 0` by default; p2a uses the sample means and prints them.

**Why `A0:C` with an A:M working model.** In the gaussian case `Yhat(1) - Yhat(0) = bA + bAM M`.
With medflex's usual `yhat ~ A0 * A1 + C` the A0 coefficient is `bA + bAM mean(M | A = 0)`, the
raw control-group mean that CAUS-23 reports as biased under confounding. With `A0:C` added the
A0 terms are `bA + bAM (alpha_0 + alpha_C' C)` with `alpha` from `lm(M ~ A + C)`, so at `cbar` the
NDE is VanderWeele's (2015) confounder-adjusted formula. The NIE is the same in both models.
On `diff_int` the usual model gives 0.70020177518541, p2a's 0.76811165933462 (VanderWeele:
0.76811165933463).

**Proportion mediated**, never clamped: `NIE / TE` on the difference scale and
`NDE (NIE - 1) / (NDE NIE - 1)` on the ratio scales. A value outside [0, 1] (beyond 1e-8) adds a
warning; an exactly null total effect gives `NaN` with a note.

The mediator model `lm(M ~ A + C)` is reported for description only (`Treatment -> Mediator`).

## Standard errors

- **Robust (default with `bootstrap_ci = false`; the MCP tool with `n_bootstrap = 0`):** the stacked
  sandwich of both estimating equations, as medflex `neModel(se = "robust")`. With the working
  model's scores `U_i = x_i (y_i - mu_i)` and information `A_b = sum v_i x_i x_i'`, the natural
  effect model's unit scores `S_i` (both expanded rows of unit i) and information `A_t`, and the
  cross derivative `G = sum_r z_r mu.eta(eta_r) x_cf,r'`, each unit's influence is
  `A_t^{-1} (S_i + G A_b^{-1} U_i)`; `V = sum_i phi_i phi_i'` (HC0), on the non-aliased columns.
  The case checks this against medflex's own `vcov` (relative difference <= 6.6e-9) and against
  `sandwich::vcovHC(HC0)` and numerical derivatives. SEs and p-values are on the link scale
  (log, logit) on the ratio scales; intervals are mapped back.
- **Bootstrap:** units resampled with replacement, everything refitted, the effects evaluated
  at the full sample's `cbar`; normal-type intervals as medflex `confint(type = "norm")`
  (`boot::norm.ci`: `2 theta_hat - mean(draws) -/+ z sd(draws)` on the link scale). The seed is
  `seed`, 42 when `None`, so repeated calls agree. A replicate that fails (e.g. a resample with
  one arm) or gives a non-finite effect is dropped and counted in a warning naming the first
  error; fewer than 50 usable replicates is an error. Gaussian/difference replicates use the
  closed form `NDE = bA + bAM (alpha_0 + alpha_C' cbar)`, `NIE = alpha_A (bM + bAM)`, which equals
  the imputation estimator (test `test_medflex_closed_form_matches_imputation`).

## Identification and degenerate inputs

- The treatment must be coded 0/1. medflex samples hypothetical levels for a continuous exposure;
  p2a refuses non-binary exposures (registry deviation, CAUS-22).
- `n` must exceed the working model's coefficients (`InsufficientData`, CAUS-74), with at least 3
  units per arm.
- Aliased confounders (a duplicate column, say) give `NaN` coefficients and R's warning
  `Coefficients not defined because of singularities: ...` for each model (CAUS-73), and the
  estimable effects are unchanged. An aliased treatment, mediator or A:M column, or a treatment
  in the span of the confounders (no overlap), is an error.
- Non-finite values, a confidence level outside (0, 1) and fewer than 50 bootstrap replicates are
  errors.

## Reference values

Case `medflex_scales` (`validation/r/cases/causal_medflex.R`, `validation/expected/medflex_scales.json`,
dataset `validation/datasets/generated/medflex_sim.csv`, n = 500, 201 treated; registry
`validation/methods/causal_medflex.json`). The estimator is written out in R from medflex 0.6-11's
source and, for all six scale/interaction configurations, cross-checked in the case against medflex
0.6-11 itself (`neImpute` + `neModel`: coefficients to 1e-8, robust variance to 1e-6 relative).

| config (outcome, family, scale, A:M) | NDE | NIE | TE | robust SE (link) NDE / NIE / TE | PM |
|---|---|---|---|---|---|
| `diff` (yc, gaussian, difference) | 0.6323375518310344 | 0.3678489248865613 | 1.0001864767175956 | 0.10242662687767257 / 0.0627978261650129 / 0.10430136549391301 | 0.36778034241551144 |
| `ratio` (yp, poisson, ratio) | 1.6944036450938882 | 1.160629583248904 | 1.9665749964607433 | 0.09453409063311997 / 0.03501613063046932 / 0.09310885494798792 | 0.28158327327258675 |
| `odds` (yb, binomial, odds ratio) | 1.6434043805532548 | 1.2121197145572282 | 1.99200284865831 | 0.2043220286490373 / 0.06785962106456386 / 0.19419765729400623 | 0.3514087369572946 |
| `diff_int` (yi, gaussian, difference, A:M) | 0.7681116593346246 | 0.90190009327458 | 1.6700117526092046 | 0.11803782355359031 / 0.12826647248064182 / 0.1383521935929501 | 0.5400561354526177 |
| `ratio_int` (yp, poisson, ratio, A:M) | 1.7539664443747178 | 1.1400686339539357 | 1.9996421282393264 | 0.1060139026510441 / 0.037598934219085604 / 0.0974088109274076 | 0.24576363572963655 |
| `odds_int` (yb, binomial, odds ratio, A:M) | 1.7513827583364292 | 1.1492718507775597 | 2.0128149040932155 | 0.2176746397627244 / 0.09325561621071796 / 0.19426704971141517 | 0.25812430751189375 |
| `diff_opp` (yo, gaussian, difference) | -0.419777088837373 | 0.33901567613615463 | -0.0807614127012184 | 0.09271568417798906 / 0.057882567510150605 / 0.10218240541881812 | -4.197743263733673 |
| `rr_bin` (yb, binomial, ratio) | 1.311016059433763 | 1.112621121248634 | 1.4586641580221593 | 0.11142951890676614 / 0.03689177724061622 / 0.10406364243879201 | 0.32190895234779393 |
| `rd_bin` (yb, binomial, difference) | 0.11350883737128907 | 0.04477585677423907 | 0.15828469414552815 | 0.04701967468966321 / 0.015777069034525814 / 0.044379588744098676 | 0.282881784723113 |

Bootstrap (`odds_int`, 50 fixed resamples from a 32-bit LCG, seed 20261130, reproduced in the test):
SE NDE / NIE / TE 0.18444726599338046 / 0.08665215618012122 / 0.18012588777506963; normal-type 95%
intervals [1.2915470064994323, 2.6614224260533605], [0.9571555426898897, 1.3443099611975582],
[1.4775070051335104, 2.993480379334992].

**Before Tier 3** (exact R emulation of the old code: OLS component models, VanderWeele's formulas
with the raw control-group mean of M, delta SEs without covariances):

| call | before | now |
|---|---|---|
| yc, no A:M | SEs NDE / NIE 0.0980168 / 0.0588695 | 0.1024266 / 0.0627978 (same estimates) |
| yp, scale ratio | NDE 0.5299 (an OLS difference) | 1.6944 |
| yb, scale odds ratio | NDE 0.1165 (an OLS difference) | 1.6434 |
| yi, A:M | NDE 0.7002 (raw mean) | 0.7681 |
| yo | PM clamped to -2 | -4.1977 |

**Tests** (`cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::medflex`):

| test | pins | tolerance |
|---|---|---|
| `test_validate_medflex_difference` | `diff`: both models, effects, robust SE/CI/p, PM, R², mediator model, means | `Tol::coef(500)`, `Tol::se(500)` |
| `test_validate_medflex_ratio` | `ratio` | `Tol::iterative(1e-8)` |
| `test_validate_medflex_odds_ratio` | `odds` | `Tol::iterative(1e-8)` |
| `test_validate_medflex_family` | `rr_bin`, `rd_bin` | `Tol::iterative(1e-8)` |
| `test_validate_medflex_interaction` | `diff_int` (and VanderWeele), `ratio_int`, `odds_int` | as above |
| `test_validate_medflex_bootstrap` | resample rows, draws, SEs, normal intervals | `Tol::iterative(1e-8)` |
| `test_validate_medflex_proportion_mediated_unclamped` | `diff_opp` and its warning | `Tol::coef(500)` |
| `test_validate_medflex_collinear_design_warns` | `alias`, `alias_int`: NA coefficients, warnings | `Tol::coef(500)`, `Tol::se(500)` |

The poisson and binomial fits use `fit_glm`'s epsilon 1e-10 against R at 1e-14: effects, natural
effect model coefficients and robust SEs agree to 6.4e-13, hence `Tol::iterative(1e-8)`; the
working model's `summary.glm` SEs, computed from the last IRLS step's weights, converge linearly
(1.8e-7 measured) and use `Tol::iterative(1e-6)`.

Unregistered property tests (no R value): `test_medflex_closed_form_matches_imputation`,
`test_medflex_rejects_degenerate_inputs`, `test_medflex_rejects_n_le_k`,
`test_medflex_bootstrap_failures_are_reported`, `test_medflex_default_seed_is_fixed`,
`test_medflex_mediator_scale_invariance`, `test_medflex_display_reports_scale_and_method`,
`test_medflex_result_serde_round_trip`, `test_medflex_robust_se_without_bootstrap`.

## Identification assumptions

1. No unmeasured confounding of A-Y given C.
2. No unmeasured confounding of M-Y given (A, C).
3. No unmeasured confounding of A-M given C.
4. Cross-world independence: no effect of A on M-Y confounders.

## MCP tool

`natural_effects_mediation` (text = the Display): `dataset`, `outcome`, `treatment`, `mediator`,
`confounders`, `allow_interaction` (default true), `n_bootstrap` (default 1000; 0 gives robust SEs),
`confidence_level` (0.95), `scale` (`difference`, `ratio`, `odds_ratio`).

## Missing values and column types

`run_medflex_dataset` (MCP `natural_effects_mediation`) drops the rows with a missing value
(null or NaN) in the outcome, treatment, mediator or a confounder jointly, so the mediator
and outcome models are fitted on the same complete rows (R `lm`, `na.omit`). The drop is
reported in `MedflexResult.missing` (serialized only when rows were dropped) and in a last
Display line (also in the MCP text):
`Missing values: 50 of 400 rows dropped because of missing values (y: 15, m: 25, x1: 10)`.
`run_medflex` takes arrays as given and reports nothing dropped. Int64 and Boolean columns
are read as numbers; a text column gives `NonNumericColumn`, an absent one
`ColumnNotFound`.

### R reference (T2-WP13)

Case `causal_mediation_na` (`validation/r/cases/causal_mediation_na.R`, registry
`validation/methods/causal_na.json`, id `medflex_na`), the 350 complete rows of 400:

| Quantity | R (`lm` on `complete.cases(y, d, m, x1, x2)`) |
|---|---|
| treated, control | 197, 153 |
| E[M\|A=0], E[M\|A=1] | 0.3767530563055379, 1.2007356199594112 |
| `lm(m ~ d + x1 + x2)`: `d` (SE), R² | 0.6131933908612826 (0.059414385098645314), 0.5943014059209022 |
| `lm(y ~ d + m + x1 + x2)`: `d` (SE), `m` (SE), R² | 0.5104280327655459 (0.07402655405381642), 0.6554511496553376 (0.0585705531269962), 0.7489960724962549 |
| `lm(y ~ d + m + d:m + x1 + x2)`: `d` (SE), `m` (SE), `d:m` (SE), R² | 0.29803058555399 (0.09699712509259713), 0.4867447333122127 (0.0768751565077352), 0.29678541513089 (0.08929231721797547), 0.7568060755019845 |

Tolerances: `validation/README.md` for n = 350 (coefficients and means 1e-8, SEs and R² 1e-6
relative); counts exact; the MCP text at `Tol::printed(4)`. With missing values read as 0,
as before T2-WP13, the mediator coefficient of `d` was 0.6243 on 400 rows. Tests (p2a-core,
`econometrics::medflex`): `test_validate_medflex_causal_na`,
`test_validate_medflex_interaction_causal_na`, `test_validate_na_medflex_reports_missing`,
`test_validate_na_medflex_errors_name_the_column`; MCP:
`test_validate_mcp_natural_effects_mediation_causal_na`.

The component models are pinned to `lm` here; the natural effects themselves are pinned on
case `medflex_scales` (below, registry `validation/methods/causal_medflex.json`).

## References

1. Vansteelandt, S., Bekaert, M., & Lange, T. (2012). "Imputation strategies for the estimation of natural direct and indirect effects." *Epidemiologic Methods*, 1(1), 131-158.

2. Lange, T., Vansteelandt, S., & Bekaert, M. (2012). "Choice of effect measure for mediation analysis." *Epidemiology*, 23(6), 889-897. https://doi.org/10.1097/EDE.0b013e31826c2107

3. VanderWeele, T. J. (2015). *Explanation in Causal Inference: Methods for Mediation and Interaction*. Oxford University Press.

4. Steen, J., Loeys, T., Moerkerke, B., & Vansteelandt, S. (2017). "medflex: An R Package for Flexible Mediation Analysis using Natural Effect Models." *Journal of Statistical Software*, 76(11), 1-46. https://doi.org/10.18637/jss.v076.i11

5. R package medflex: https://CRAN.R-project.org/package=medflex

## Status

R-pinned on `medflex_scales` (natural effects, both models, robust and bootstrap SEs) and
`causal_mediation_na` (complete cases, component models).
