# Validation: Hurdle Models

## Method Overview

Hurdle models (also called two-part models) handle count data with excess zeros by separating the process into two stages:

1. **Binary Part (Hurdle)**: Logit model for P(Y > 0 vs Y = 0)
2. **Count Part (Truncated)**: Truncated Poisson or Negative Binomial for P(Y = y | Y > 0)

Unlike zero-inflated models, hurdle models assume ALL zeros come from the binary part. There are no "structural zeros" mixed with "sampling zeros."

**Key Parameters:**
- `y`: Count dependent variable (non-negative integers)
- `x`: Covariates for count model
- `z`: Covariates for binary model (optional, defaults to x)
- `dist`: "poisson" or "negbin"

**Model Equations:**
- Binary: logit(P(Y > 0)) = Z'γ
- Truncated Poisson: P(Y = y | Y > 0) = exp(-μ)μ^y / (y! × (1 - exp(-μ)))
- Truncated NegBin: P(Y = y | Y > 0) = f_NB(y; μ, θ) / (1 - f_NB(0; μ, θ))

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| pscl | R | `hurdle()` | 1.5.5 |
| statsmodels | Python | `ZeroInflatedPoisson/NegativeBinomial` | N/A |

## Test Cases

### Test 1: Hurdle Poisson - Basic Model

**Data Generation:**
```r
set.seed(42)
n <- 200
x <- rnorm(n)
prob_positive <- plogis(0.5 + 1.0 * x)
is_positive <- rbinom(n, 1, prob_positive)
lambda <- exp(0.8 + 0.5 * x)
y_count <- rpois(n, lambda)
y_count[y_count == 0] <- 1
y <- ifelse(is_positive == 1, y_count, 0)
```

**R Code:**
```r
library(pscl)
model <- hurdle(y ~ x, data = data.frame(y, x), dist = "poisson", zero.dist = "binomial")
summary(model)
```

No R output of this example is committed, and no test compares with one: the zero part is
R-pinned as a binomial glm on its own case (`validation/methods/discrete_t3.json`, see
[count_models.md](count_models.md)); the truncated count part is R-pinned on `disc_hurdle_t4`
(see "Count part" below).

**Rust tests** (the zero part is R-pinned as a binomial glm, `validation/methods/discrete_t3.json`; the
truncated count part on `disc_hurdle_t4`, `validation/methods/discrete_t4.json`):
- `econometrics::discrete::tests::test_hurdle_poisson_basic` (`crates/p2a-core/src/econometrics/discrete/mod.rs`): n = 12, model type and the zero count; no `pscl::hurdle` value is compared.

### Test 2: Hurdle Negative Binomial - Overdispersed Data

**Data Generation:**
```r
set.seed(123)
n <- 200
x <- rnorm(n)
prob_positive <- plogis(-0.3 + 0.8 * x)
is_positive <- rbinom(n, 1, prob_positive)
mu <- exp(1.0 + 0.4 * x)
theta <- 2.0
y_count <- rnbinom(n, mu = mu, size = theta)
y_count[y_count == 0] <- 1
y <- ifelse(is_positive == 1, y_count, 0)
```

**R Code:**
```r
model_nb <- hurdle(y ~ x, data = data.frame(y, x), dist = "negbin", zero.dist = "binomial")
summary(model_nb)
model_nb$theta  # Dispersion parameter
```

No R output of this example is committed, and no test compares with one: the zero part is
R-pinned as a binomial glm on its own case (`validation/methods/discrete_t3.json`, see
[count_models.md](count_models.md)); the truncated count part is R-pinned on `disc_hurdle_t4`
(see "Count part" below).

**Rust tests** (the zero part is R-pinned as a binomial glm, `validation/methods/discrete_t3.json`; the
truncated count part on `disc_hurdle_t4`, `validation/methods/discrete_t4.json`):
- `econometrics::discrete::count_models::tests::test_hurdle_negbin` (`crates/p2a-core/src/econometrics/discrete/count_models.rs`): the model type and that a theta is returned.
- `econometrics::discrete::count_models::tests::test_hurdle_negbin_structure_properties` (`crates/p2a-core/src/econometrics/discrete/count_models.rs`): n, model type, theta > 0 and a finite log-likelihood; no `pscl::hurdle` value is compared.

### Test 3: Coefficient Sign Test

**Purpose:** Verify that positive relationship in data yields positive coefficients.

**Data:** Uses test dataset with y increasing with x.

**Rust tests** (the zero part is R-pinned as a binomial glm, `validation/methods/discrete_t3.json`; the
truncated count part on `disc_hurdle_t4`, `validation/methods/discrete_t4.json`):
- `econometrics::discrete::count_models::tests::test_hurdle_poisson_structure_properties` (`crates/p2a-core/src/econometrics/discrete/count_models.rs`): n, model type, no theta, the zero and positive counts, and finite log-likelihoods; no `pscl::hurdle` value is compared.

## Missing values

`run_hurdle` reads the outcome, the count regressors `x` and the binary-part regressors
`z` in one request; a row with a missing value in any of them is dropped from both parts
(listwise deletion, as `pscl::hurdle`'s default `na.action = na.omit`) and reported in
`HurdleResult::missing` (JSON key `missing`, omitted when nothing was dropped) and on the
last line of the text output. A text outcome or regressor is `NonNumericColumn`; before
T2-WP10 a missing count was read as 0 and a missing regressor was a design error.

R reference: case `disc_zeroinfl` (`validation/r/cases/discrete.R`, dataset
`validation/datasets/generated/zeroinfl.csv`): `pscl::hurdle(y_na ~ x1 + x2_na)` uses 495
of the 500 rows (`hurdle_na_n_used`; `y_na` has 3 and `x2_na` 2 missing cells).

**Rust test** (R-pinned for the rows used):
- `econometrics::discrete::count_models::tests::test_validate_hurdle_missing_rows_dropped`
  (`crates/p2a-core/src/econometrics/discrete/count_models.rs`): `n_obs` equals R's
  `hurdle_na_n_used` (exact), 5 rows are dropped, and both parts' coefficients equal those
  of the same fit on the complete rows, and so do the count-part SEs and log-likelihood
  (Poisson) and the coefficients, SEs and theta of the NB hurdle. The estimates are pinned
  against `pscl::hurdle` on complete data in the "Count part" tests.

## Count part (T4)

The count part is the zero-truncated Poisson or NB maximum-likelihood fit on the positive
counts, pscl's count hurdle. Per positive count, with `eta = x'beta`, `mu = e^eta`:

- Poisson: `y eta - mu - lnGamma(y + 1) - ln(1 - e^-mu)`; score `y - mu / (1 - p0)`;
  Hessian weight `-mu P(N >= 2) / (1 - p0)^2`, the exact truncated information
  (`P(N >= 2)` from its series below `mu = 0.1`, where `1 - e^-mu (1 + mu)` cancels).
- NB (`t = ln theta`): `ln f(y) - ln(1 - f0)`, `L = ln f0 = -theta ln(1 + mu / theta)`,
  first derivatives `lnf_a + r L_a`, second `lnf_ab + r L_ab + r (1 + r) L_a L_b` with
  `r = f0 / (1 - f0)`, in `(eta, t)`.

The log-likelihood is maximised by damped Newton steps (the zero-inflated models' loop:
Levenberg damping, step halving, at most 100 iterations), from `ln(mean y)` in the
intercept for the Poisson (the truncated-Poisson log-likelihood is concave) and from the
truncated-Poisson MLE with theta = 1 for the NB, ln theta capped at ln 1e8. It stops on
the Newton decrement, the relative log-likelihood change (1e-12) and the relative step
(the rule of `run_zip`). Standard errors are the square roots of the diagonal of the inverse
of minus the Hessian at the estimate; for NB, `SE(ln theta)` is pscl's `SE.logtheta` and
`SE(theta) = theta SE(ln theta)`. A theta above 1e6 (no overdispersion among the positive
counts) is reported as for ZINB: the coefficient SEs from the coefficient block, the theta
SEs NaN, `converged = false` and a warning. `iterations` and the convergence record describe
the count part (for NB the NB fit's, without its truncated-Poisson start); `converged` is both parts.

Case `disc_hurdle_t4` (`validation/r/cases/discrete_t4.R`, `zeroinfl.csv`, n 500): pscl's
count part (`optim`, reltol 1e-15) polished by Newton steps on the analytic truncated score,
SEs from `numDeriv::hessian`; the `*_pkg` keys are pscl's own estimate.

| Key | (Intercept) | x1 | x2 | other |
|---|---|---|---|---|
| `hp_count` / `hp_count_se` | 0.6183142914978138 / 0.055957874172556474 | 0.2035367140970712 / 0.05522726642442113 | -0.27476886758143376 / 0.05912654512240702 | loglik -688.077415295141, 241 positive |
| `hnb_count` / `hnb_count_se` (`y_nb`) | 0.3649446202808409 / 0.16163495712085688 | 0.3144150933310138 / 0.09626249227313967 | -0.3175947757804973 / 0.08990635918536229 | theta 0.9765891641238044, SE(ln theta) 0.37854615006935205, SE(theta) 0.36968406827851275 |
| `hpnb_count` (NB on `y`) | 0.6154982562698516 | 0.2035452640787471 | -0.27568737115680425 | theta 173.98441940639327 (flat profile), SE(ln theta) 10.015126250639057 |

**Rust tests** (`crates/p2a-core/src/econometrics/discrete/count_models.rs`; registry
`validation/methods/discrete_t4.json`, id `discrete_hurdle_count_part`):
- `econometrics::discrete::count_models::tests::test_validate_hurdle_poisson_count_part`:
  count coefficients 1e-8, SEs and z 1e-6, p 1e-4; count, zero and total log-likelihood, AIC
  and BIC `Tol::stat(500)`; pscl's estimate 1e-5.
- `econometrics::discrete::count_models::tests::test_validate_hurdle_negbin_count_part`: the
  same, plus theta and ln theta 1e-7, SE(ln theta) and SE(theta) 1e-5; pscl's estimate 1e-4.
- `econometrics::discrete::count_models::tests::test_validate_hurdle_negbin_flat_theta`:
  coefficients 1e-6, SEs 1e-5, ln theta 1e-4 and its SE 1e-3 (a flat profile), no warning.
- `econometrics::discrete::count_models::tests::test_validate_hurdle_intercept_only_zero_part`:
  `y ~ x1 + x2 | 1`.
- `p2a-mcp` `tools::handlers::discrete::tests::test_validate_mcp_hurdle_count_part`: the tool's
  Count Part rows at print precision.
- Not R-pinned: `test_hurdle_truncated_information_matches_central_differences` (score and
  Hessian vs central differences at mu = 1e-6, 30 and a regression point, Poisson and NB),
  `test_hurdle_small_mean_series` and `test_hurdle_negbin_theta_runs_to_poisson_warns`.

## Performance Comparison

### Hurdle Poisson

| Dataset Size | Rust (ms) | R (ms) | Speedup |
|--------------|-----------|--------|---------|
| n=100        | ~0.5      | 2.02   | ~4x     |
| n=500        | ~1.4      | 4.12   | ~3x     |
| n=1,000      | ~2.2      | 5.64   | ~2.5x   |
| n=2,000      | ~4.0      | 13.68  | ~3.4x   |

### Hurdle Negative Binomial

| Dataset Size | Rust (ms) | R (ms) | Speedup |
|--------------|-----------|--------|---------|
| n=100        | ~2.0      | 2.37   | ~1.2x   |
| n=500        | ~4.8      | 6.23   | ~1.3x   |
| n=1,000      | ~11.0     | 8.03   | ~0.7x   |
| n=2,000      | ~15.7     | 17.46  | ~1.1x   |

*Notes:*
- Hurdle Poisson is consistently ~3x faster than R
- Hurdle NegBin is comparable to R due to iterative theta estimation
- Both use Newton-Raphson MLE for parameter estimation

## Implementation Notes

The Rust implementation:
1. Separates observations into binary indicators (y > 0 vs y = 0)
2. Fits the zero hurdle as the shared binomial GLM (`fit_glm`, glm.fit's IRLS)
3. Extracts positive observations (y > 0)
4. Fits truncated Poisson/NegBin on positive subset by damped Newton steps on the analytic
   truncated log-likelihood
5. For NegBin: estimates ln theta jointly with the coefficients
6. Computes combined log-likelihood: LL = LL_binary + LL_count
7. Returns coefficients, standard errors, z-statistics, p-values for both parts

## Status

R-pinned: the rows used (registry `validation/methods/discrete_ordered_count.json`, id
`discrete_hurdle_rows`), the zero part (`validation/methods/discrete_t3.json`) and the count
part, its SEs, theta and the SEs of ln theta and theta (`validation/methods/discrete_t4.json`,
id `discrete_hurdle_count_part`).

## Zero part = binomial glm (T3-WP18)

The hurdle's zero part is `glm(I(y > 0) ~ x1 + x2, binomial)`, which is pscl's zero hurdle.
Case `disc_hurdle_zero` (registry `validation/methods/discrete_t3.json`, id
`discrete_count_domain_aliasing`) fits it on T2-WP10's `zeroinfl.csv` (n 500): `zero_coef`
(-0.0476982071, -0.2203064579, -0.3290106975 for (Intercept), x1, x2) at glm's control
1e-10, `zero_coef_tight` and `zero_se_tight` at epsilon 1e-14, `zero_loglik`, and pscl's own
`zero_coef_pkg`. `econometrics::discrete::count_models::tests::test_validate_hurdle_zero_part_is_binomial_glm`
pins it at the tight glm fit: coefficients 1e-8, SEs and logLik 1e-6, pscl's estimate 1e-6;
n, zeros and positives exact. glm's warnings (separation in the zero part) are in
`HurdleResult.warnings`, printed in the Warnings block. The truncated count part is pinned
on `disc_hurdle_t4` ("Count part" above); a singular count-part information gives `NaN`
standard errors and a warning (the `eye * 1e-6` covariance fallback is gone).

## Outcome domain and aliased regressors (T3-WP18)

Case `disc_hurdle_zero` in [`validation/r/cases/discrete_mlogit_asc.R`](../r/cases/discrete_mlogit_asc.R)
(registry `validation/methods/discrete_t3.json`, id `discrete_count_domain_aliasing`) records
what R does with a bad count outcome: `MASS::glm.nb` refuses negative counts
(`negbin_neg_refused` = 1, "negative values not allowed for the 'Poisson' family") and fits
non-integer counts with a warning (`negbin_frac_warned` = 1, "non-integer x"); `pscl::zeroinfl`
and `pscl::hurdle` refuse both (`zeroinfl_neg_refused`, `zeroinfl_frac_refused`,
`hurdle_neg_refused`, `hurdle_frac_refused` = 1, "invalid dependent variable").

p2a does the same: a negative count is `InvalidSpecification` in `run_negbin`, `run_zip`,
`run_zinb` and `run_hurdle` (naming the column, the count and the first value); a non-integer
count is `InvalidSpecification` for the pscl models and a warning for the negative binomial.
A zero is `y == 0` (the zero-inflated rule was `y < 0.5`).

Aliased regressors: the negative binomial drops a regressor that is a linear combination of the
others, as glm.nb's `glm.fitter` does at its QR tolerance (1e-11): its coefficient, SE, z and p
are `NaN`, its name is in `NegBinResult.aliased`, and the warning is R's "Coefficients not
defined because of singularities". The zero-inflated and hurdle models refuse an aliased count
or zero design with `SingularMatrix` naming the columns (pscl stops in optim with "non-finite
value supplied by optim").

Tests:

- `econometrics::discrete::count_models::tests::test_validate_count_models_refuse_negative_counts`
- `econometrics::discrete::count_models::tests::test_validate_count_models_refuse_non_integer_counts`
- `econometrics::discrete::count_models::tests::test_validate_negbin_aliased_column_is_na`
  (`x1b = 2 x1` on `disc_negbin`: the other coefficients and theta at T2-WP10's tolerance 1e-6,
  and 1e-12 against p2a's own fit without x1b)
- `econometrics::discrete::count_models::tests::test_validate_zeroinfl_and_hurdle_aliased_regressor_is_an_error`

## Timing

Release build, one P core (`taskset`, `RAYON_NUM_THREADS=1`), three interleaved rounds at 003af73a against
`ff8e7559` (before these changes); new / base as the ratio of medians (`unified_benchmarks`,
criterion `new_methods_benchmarks`).

| Fit | n | new / base |
|---|---|---|
| Poisson hurdle, two regressors (unified, `zeroinfl` data) | 100 / 1,000 / 10,000 | 1.21 / 1.19 / 1.25 |
| Poisson hurdle, one regressor (criterion `hurdle/poisson`) | 100 / 500 / 1,000 / 2,000 | 1.00 / 1.08 / 1.12 / 1.16 |
| negative binomial hurdle (criterion `hurdle/negbin`) | 100 / 500 / 1,000 / 2,000 | 0.98 / 1.03 / 1.04 / 1.06 |

The Poisson hurdle costs up to 1.25x of base at n = 10,000 and 1.21x at n = 100. This is
accepted as the cost of fitting the zero part with `glm.fit`'s IRLS (see "Zero part =
binomial glm"). The zero part's design is built and decomposed once, the aliasing QR runs
only when the normal-equations test cannot certify full rank, and the zero part's IRLS steps
take the normal equations with the QR fallback. Measured piece by piece at n = 1,000: the zero
part, R's `glm.fit` IRLS (working weights and response, the deviance at every step,
`family$aic`), takes about 170 µs where the old hand-written Newton logit took about 140; the
count part's Newton steps solve the equilibrated information by Cholesky (about 1.5 µs a step
over 19 steps) where the old code took a plain inverse. Both loops are bound by the per-row
`exp`, `ln` and divisions; fusing their passes was measured slower, and no cheaper form gives
the same values.

## References

- Mullahy, J. (1986). "Specification and Testing of Some Modified Count Data Models."
  *Journal of Econometrics*, 33(3), 341-365.
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.).
  Cambridge University Press. Chapter 4.
- Zeileis, A., Kleiber, C., & Jackman, S. (2008). "Regression Models for Count Data in R."
  *Journal of Statistical Software*, 27(8), 1-25.
