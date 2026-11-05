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

Outcome types: `Continuous` (linear end-of-follow-up outcome model) and `Binary` (logistic
end-of-follow-up outcome model).

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

## Values (R 4.3.3)

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

Property tests (not registered; they load no case): `test_gformula_natural_vs_natural_is_exactly_zero`,
`test_gformula_static_contrast_is_exact_under_common_random_numbers`,
`test_gformula_threshold_extreme_cutoffs_equal_static`, `test_gformula_covariate_scale_invariance`,
`test_gformula_bootstrap_is_reproducible_and_thread_invariant`.

## Running

```bash
Rscript validation/r/run.R --check gformula_natural_course
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::gformula::
```

## References

- Robins, J.M. (1986). A new approach to causal inference in mortality studies with a sustained
  exposure period. *Mathematical Modelling*, 7, 1393-1512.
- Hernan, M.A. & Robins, J.M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC, ch. 21.
- McGrath, S., et al. (2020). gfoRmula: An R package for estimating the effects of sustained
  treatment strategies via the parametric g-formula. *Patterns*, 1(3), 100008.
