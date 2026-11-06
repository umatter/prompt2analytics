# Parametric g-formula Validation

`run_gformula` (`crates/p2a-core/src/econometrics/gformula.rs`), method registry
`validation/methods/causal_gformula.json`, R cases `validation/r/cases/causal_gformula.R`.

## Method

The parametric g-formula (Robins 1986; Hernan & Robins 2020, ch. 21) estimates the mean
outcome under a treatment regime by fitting models for the time-varying covariates, the
treatment and the outcome, and then simulating each subject's covariate path under the
regime from those models. `run_gformula` reports the natural course (treatment drawn from
the fitted treatment model), the intervention (always treat, never treat, or a threshold
rule on a time-varying covariate), their difference, ratio and odds ratio, and bootstrap
inference.

Outcome types: `Continuous` (linear end-of-follow-up outcome model), `Binary` (logistic
end-of-follow-up outcome model) and `Survival`, a discrete-time hazard g-formula (Hernan &
Robins 2020, ch. 17; Keil et al. 2014) on data built with `GFormulaData::new_survival`, which
carry each subject's last observed period and event indicator. Before CAUS-64 was fixed,
`Survival` was the binary end-of-follow-up logit under another label.

## Models

Data are wide: one row per subject, covariates and treatments per period `t = 1..T`.

| Model | Design | Fit |
|---|---|---|
| Time-varying covariate `L_t[j]` | `[1, baseline, L_(t-1), A_(t-1)]` (lags for `t > 1`) | `lm` (Normal), or `glm(binomial)` when its values are all 0 or 1 |
| Treatment `A_t` | `[1, baseline, L_t, A_(t-1)]` (lag for `t > 1`) | `glm(binomial)` |
| Outcome | `[1, baseline, L_T, A_1, ..., A_T]` | `lm` (continuous) or `glm(binomial)` (binary) |

Linear models are fitted with OLS's solve decision (normal equations on a well-conditioned
design, `lm.fit`'s pivoted QR otherwise); logistic models with the shared `glm.fit` IRLS
(`econometrics::glm::fit_glm`, `epsilon` = `tolerance`, `maxit` = `max_iter`). Aliased
columns are reported as R reports them ("Coefficients not defined because of
singularities") and predicted with their identified columns.

**Survival: risk sets and the hazard.** For survival data the models of period `t` are
fitted on the subjects at risk at its start (`time >= t`), and the outcome model is a pooled
logistic hazard on the person-periods at risk (subject `i` contributes periods `1..time_i`;
the response is 1 in the period of the event):

    logit h(t) = b0 + b1 t + b2 A_t + baseline + L_t

The form is fixed: a linear period term and the current treatment (no period dummies, no
treatment history). A period nobody is at risk in is an error (`InsufficientData`).

**Point treatment (rule P).** When the treatment never changes within any subject's
follow-up, it is a point treatment: one treatment model (period 1), its value carried
forward under every regime, and one treatment column in the outcome model. Without the
rule, per-period treatment models would be perfectly separated and the outcome model's
treatment columns aliased. A result note says when it applies (`T > 1`).

## Simulation and standardisation

For each simulated unit a baseline row is drawn with replacement. In each period every
time-varying covariate is drawn from its model: `Normal(eta, sigma)` with `sigma` the
residual SD `sqrt(RSS / df)` of the linear model, or `Bernoulli(plogis(eta))`. The
treatment is then set by the regime. The unit's value is the outcome model's expectation
(`eta` or `plogis(eta)`); the regime's risk is the mean over units.

Before CAUS-16 was fixed, continuous covariates were drawn as `eta + sigma * U(-1, 1) / 2`
(SD `sigma / sqrt(12)`), which compressed the covariate distribution: on
`gformula_natural_course` the natural-course risk was 0.166 against the exact 0.284 and an
observed mean of 0.2845.

**Common random numbers.** Every regime replays one seeded stream and consumes the same
draws (a row per unit, one draw per covariate and one treatment draw per period, used or
not). Natural against natural gives a risk difference of exactly 0, a threshold with an
extreme cutoff equals the static regime bit for bit, and a static contrast of linear models
is exact.

**Exact standardisation.** When the regime is static and there is no time-varying covariate,
nothing in the simulation is random: the intervention's risk is the mean over the `n`
observed baseline rows (`standardization = Empirical`). The natural course is always Monte
Carlo.

## Inference

Subjects are resampled with replacement `n_bootstrap` times (0, or at least 2). Replicate
`b` uses the seed `seed + 1000 b`: the resample is drawn from ChaCha8 stream 0 of that
seed and the replicate's simulations (models refitted, `n_simulations / 2` units, at least
100) from stream 1, so results do not depend on the chunked parallel evaluation or the
thread count. Replicates that fail or give a non-finite risk difference are dropped and
reported in a warning; more than half dropped is an error.

- Risk difference: SE = SD of the replicates (`n - 1`), a type-7 percentile interval
  (`stats::robust::percentile_interval`) and the two-sided normal p-value
  `2 Phi(-|RD / SE|)` computed as an upper tail.
- Risk ratio: `exp(ln RR -/+ z SE(ln RR))` with the exact `z = Phi^-1((1 + level) / 2)`.
- Without a bootstrap the SE, intervals and p-value are `NaN` and printed "not computed".
- `confidence_level` must lie strictly between 0 and 1.

## Reference implementation

gfoRmula is not installed, so the references are written out from the published formulas
in `validation/r/cases/causal_gformula.R`, with exact integrals in place of the Monte Carlo
simulation:

- `gformula_natural_course`: one period, `B ~ N(0, 1)`, `L = 0.5 B + N(0, 4)`, a confounded
  treatment, a binary outcome `Y` and a continuous outcome `Yc` (n = 2000). The models are
  `lm(L_t0 ~ B)`, `glm(A_t0 ~ B + L_t0, binomial)`, `glm(Y ~ B + L_t0 + A_t0, binomial)` and
  `lm(Yc ~ B + L_t0 + A_t0)` (`glm.control(epsilon = 1e-14)`). The natural-course, always- and
  never-treat risks integrate over `Normal(mu_i, sigma)` by 120-point Gauss-Hermite, which
  agrees with `integrate(rel.tol = 1e-12)` to 2.2e-14.

- `gformula_hazard`: n = 400, five periods, time-fixed `L ~ N(0, 1)` and `Lb ~ Bernoulli(0.4)`,
  a point treatment, events from a logistic hazard. `glm(y ~ t + A + L + Lb, binomial)` on the
  1658 person-periods; the always- and never-treat risks `1 - mean_i prod_{s <= t} (1 - h)` are
  standardised exactly over the 400 rows (no Monte Carlo); the natural course of the point
  treatment mixes the two survival curves with the period-1 treatment model's probabilities;
  the observed Kaplan-Meier curve; and the end-of-follow-up logit `glm(event ~ L + Lb + A)`,
  which is what `Survival` computed before.
- `gformula_hazard_tv`: n = 1000, three periods, a continuous time-varying covariate and a
  time-varying treatment, 10% loss to follow-up after period 2 (cells after a subject's last
  period are empty in the CSV). Covariate (`lm`), treatment and hazard (`glm`) models on the
  risk sets (1000, 894, 731; 2625 person-periods); the g-formula integral over the three
  Gaussian covariate draws by a 16-point Gauss-Hermite product rule (agrees with 24 points to
  2.5e-14), summing the eight treatment paths for the natural course.

## Values (R 4.3.3)

`gformula_hazard`:

| Quantity | Value |
|---|---|
| hazard coefficients (Intercept), t, A, L, Lb | -2.8418562733303614, 0.20850004305456832, -0.59981834178666882, 0.62349933389286327, 0.73867243632750212 |
| risk at period 5, always / never treat | 0.359537594778087 / 0.51852120218356346 |
| risk difference, ratio at period 5 | -0.15898360740547646, 0.69339034404769795 |
| natural course (point treatment), period 5 | 0.43697111971018232 |
| observed Kaplan-Meier risk, period 5 | 0.4375 |
| end-of-follow-up logit, always / never | 0.36163164924175173 / 0.51695510707905579 |

`gformula_hazard_tv`:

| Quantity | Value |
|---|---|
| risk at period 3, always / never / natural | 0.19962650752121802 / 0.38351773047382709 / 0.27790004920887373 |
| observed Kaplan-Meier risk, period 3 | 0.27797537619699053 |
| person-periods; at risk by period | 2625; 1000, 894, 731 |


`gformula_natural_course`:

| Quantity | Value |
|---|---|
| natural course, binary | 0.28396647520206436 |
| always treat, binary | 0.30555298865151798 |
| never treat, binary | 0.25845981012735386 |
| natural course, continuous | 1.3403405178404333 |
| always / never treat, continuous | 1.9007848832287226 / 0.91387882603185611 |
| observed means (binary, continuous) | 0.2845, 1.3407156957694999 |
| covariate model residual SD | 1.9416412040713367 |

## Tests and tolerances

| Test | Case | Tolerance |
|---|---|---|
| `test_validate_gformula_natural_course_models` | `gformula_natural_course` | coefficients `Tol::coef(2000)` (1e-10), residual SDs `Tol::stat(2000)` (1e-8) |
| `test_validate_gformula_natural_course_risks` | `gformula_natural_course` | Monte Carlo risks within 5 Monte Carlo SEs of the exact values (200 000 units binary, 1 000 000 continuous, seeded); natural course within 0.01 / 0.02 of the observed mean |

| `test_validate_gformula_hazard_risks` | `gformula_hazard` | hazard coefficients `Tol::coef(1658)` (1e-10), SEs `Tol::se(1658)` (1e-8); exact risks, difference, ratio and Kaplan-Meier `Tol::coef(400)` (1e-8); natural course (Monte Carlo, 200 000 units) within 5 x 0.5 / sqrt(200 000) = 0.0056 |
| `test_validate_gformula_hazard_default_tolerance` | `gformula_hazard` | the default `tolerance` 1e-8 (glm's epsilon) against the tight fit at `Tol::iterative(1e-8)` (gap <= 7.8e-10) |
| `test_validate_gformula_binary_eof_vs_survival` | `gformula_hazard` | the end-of-follow-up logit at `Tol::coef(400)`; the hazard g-formula differs from it |
| `test_validate_gformula_hazard_tv` | `gformula_hazard_tv` | risk sets and person-periods exact; models `Tol::coef` / `Tol::stat` at their risk-set sizes; hazard as above; Kaplan-Meier `Tol::coef(1000)`; Monte Carlo risk curves (200 000 units) within 0.0056 of the Gauss-Hermite values |

The Monte Carlo bound for a survival risk uses `0.5 / sqrt(n_sim)`: each unit's risk lies
in [0, 1], so its SD is at most 0.5.

Property tests (not registered; they load no case): `test_gformula_natural_vs_natural_is_exactly_zero`,
`test_gformula_static_contrast_is_exact_under_common_random_numbers`,
`test_gformula_threshold_extreme_cutoffs_equal_static`, `test_gformula_covariate_scale_invariance`,
`test_gformula_bootstrap_is_reproducible_and_thread_invariant`, `test_gformula_survival_requires_event_times`,
`test_gformula_survival_data_validation`, `test_gformula_survival_empty_risk_set_is_an_error`,
`test_gformula_display_survival_and_notes` (these two read `gformula_surv.csv` directly and pin no
value); the inference helpers against small R values: `test_gformula_rd_inference_percentile_type7`,
`test_gformula_p_value_uses_the_upper_tail`, `test_gformula_rr_interval_uses_exact_z`,
`test_gformula_without_bootstrap_reports_nan_inference`, `test_gformula_confidence_level_validated`;
data checks: `test_gformula_rejects_zero_rows`, `test_run_gformula_guards_zero_rows`,
`test_gformula_data_rejects_malformed_cells`, `test_gformula_rejects_unknown_threshold_covariate`.

## Limitations

- No censoring model and no inverse-probability-weighted natural course: the natural course
  assumes no loss to follow-up.
- The hazard is linear in the period and depends on the current treatment only.
- Binary (0/1) treatments only; no competing events.

## Running

```bash
Rscript validation/r/run.R --check gformula_hazard gformula_hazard_tv gformula_natural_course
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::gformula::
```

## References

- Robins, J.M. (1986). A new approach to causal inference in mortality studies with a sustained
  exposure period. *Mathematical Modelling*, 7, 1393-1512.
- Hernan, M.A. & Robins, J.M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC, ch. 17
  and 21.
- Keil, A.P., et al. (2014). The parametric g-formula for time-to-event data: intuition and a
  worked example. *Epidemiology*, 25(6), 889-897.
- McGrath, S., et al. (2020). gfoRmula: An R package for estimating the effects of sustained
  treatment strategies via the parametric g-formula. *Patterns*, 1(3), 100008.
