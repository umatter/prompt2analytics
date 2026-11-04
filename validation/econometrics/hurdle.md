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

**Results Comparison:**

| Part | Variable | R Coef | Rust Coef | Tolerance | Status |
|------|----------|--------|-----------|-----------|--------|
| Binary | (Intercept) | ~0.5 | ~0.5 | ±0.3 | ✅ |
| Binary | x | ~1.0 | ~1.0 | ±0.3 | ✅ |
| Count | (Intercept) | ~0.8 | ~0.8 | ±0.3 | ✅ |
| Count | x | ~0.5 | ~0.5 | ±0.3 | ✅ |

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with ECON-81):
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

**Results Comparison:**

| Part | Variable | R Coef | Rust Coef | Tolerance | Status |
|------|----------|--------|-----------|-----------|--------|
| Binary | (Intercept) | ~-0.3 | ~-0.3 | ±0.4 | ✅ |
| Binary | x | ~0.8 | ~0.8 | ±0.3 | ✅ |
| Count | (Intercept) | ~1.0 | ~1.0 | ±0.3 | ✅ |
| Count | x | ~0.4 | ~0.4 | ±0.3 | ✅ |
| - | theta | ~2.0 | ~2.0 | ±1.0 | ✅ |

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with ECON-81):
- `econometrics::discrete::count_models::tests::test_hurdle_negbin` (`crates/p2a-core/src/econometrics/discrete/count_models.rs`): the model type and that a theta is returned.
- `econometrics::discrete::count_models::tests::test_validate_hurdle_negbin_structure` (`crates/p2a-core/src/econometrics/discrete/count_models.rs`): n, model type, theta > 0 and a finite log-likelihood; no `pscl::hurdle` value is compared.

### Test 3: Coefficient Sign Test

**Purpose:** Verify that positive relationship in data yields positive coefficients.

**Data:** Uses test dataset with y increasing with x.

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with ECON-81):
- `econometrics::discrete::count_models::tests::test_validate_hurdle_poisson_structure` (`crates/p2a-core/src/econometrics/discrete/count_models.rs`): n, model type, no theta, the zero and positive counts, and finite log-likelihoods; no `pscl::hurdle` value is compared.

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
  of the same fit on the complete rows. The estimates themselves are not compared with
  `pscl::hurdle` (ECON-16, Tier 4).

## Numerical Precision Summary

- **Coefficients**: Match R within ±0.3 (stochastic data generation)
- **Standard errors**: Match R within ±30%
- **Log-likelihood**: Match R within ±5%
- **AIC/BIC**: Match R within ±5%
- **Theta (NegBin)**: Match R within ±50% (difficult to estimate)

## Known Differences

1. **Convergence tolerance**: Rust uses 1e-6, R uses 1e-8. Small samples may not converge to strict tolerance in Rust.
2. **Starting values**: Rust initializes beta from log(mean(y)), R may use different initialization.
3. **Theta estimation**: Rust uses profile likelihood gradient step, R uses full MLE.
4. **Truncation adjustment**: Both use log(1 - P(Y=0)) correction but numerical precision may differ.

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
4. Fits truncated Poisson/NegBin on positive subset by Newton steps solved on the
   equilibrated information (Cholesky first)
5. For NegBin: estimates theta using profile likelihood
6. Computes combined log-likelihood: LL = LL_binary + LL_count
7. Returns coefficients, standard errors, z-statistics, p-values for both parts

## Status

R-pinned for the rows used, known deviations (ECON-16): registry
`validation/methods/discrete_ordered_count.json`, id `discrete_hurdle_rows`. The hurdle
estimates, standard errors and theta are not R-pinned (ECON-16, Tier 4: the
truncated-count information and the NB hurdle theta).

## Zero part = binomial glm (T3-WP18)

The hurdle's zero part is `glm(I(y > 0) ~ x1 + x2, binomial)`, which is pscl's zero hurdle.
Case `disc_hurdle_zero` (registry `validation/methods/discrete_t3.json`, id
`discrete_count_domain_aliasing`) fits it on T2-WP10's `zeroinfl.csv` (n 500): `zero_coef`
(-0.0476982071, -0.2203064579, -0.3290106975 for (Intercept), x1, x2) at glm's control
1e-10, `zero_coef_tight` and `zero_se_tight` at epsilon 1e-14, `zero_loglik`, and pscl's own
`zero_coef_pkg`. `econometrics::discrete::count_models::tests::test_validate_hurdle_zero_part_is_binomial_glm`
pins it at the tight glm fit: coefficients 1e-8, SEs and logLik 1e-6, pscl's estimate 1e-6;
n, zeros and positives exact. glm's warnings (separation in the zero part) are in
`HurdleResult.warnings`, printed in the Warnings block. The truncated count part stays
unpinned (ECON-16, Tier 4); a singular count-part information gives `NaN` standard errors and a
warning (the `eye * 1e-6` covariance fallback is gone).

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

## Timing (T3-PERF2)

Release build, `taskset -c 1`, `RAYON_NUM_THREADS=1`; base is the prebuilt `fe13b889` bench
binaries of the wave-2A integration timing, head the T3-PERF2 lane; three interleaved rounds,
medians of the per-run medians (`unified_benchmarks`, criterion `new_methods_benchmarks`, and the
integration's scratch timing crate for the conditional logit). The machine was shared with other
lanes' builds; the integration re-times on a quiet machine.

| Fit | n: base µs, head/base | | | |
|---|---|---|---|---|
| Poisson hurdle, two regressors (unified, `zeroinfl` data) | 100: 88.7, 0.84 | 1,000: 377, 1.23 | 10,000: 4,003, 1.30 | |
| Poisson hurdle, one regressor (criterion `hurdle/poisson`) | 100: 92.8, 0.99 | 500: 261, 1.08 | 1,000: 438, 1.14 | 2,000: 875, 1.17 |
| negative binomial hurdle (criterion `hurdle/negbin`) | 100: 197, 0.95 | 500: 643, 0.99 | 1,000: 1,145, 1.04 | 2,000: 2,238, 1.06 |

Not signed off: the Poisson hurdle is above 1.1x at n >= 1,000 (it was 1.27-1.49 unified and
1.12-1.34 criterion at the wave-2A integration). What T3-PERF2 removed: the zero part's design
is no longer built and decomposed twice, the aliasing QR runs only when the normal-equations
test cannot certify full rank, and the zero part's IRLS steps take the normal equations with
the QR fallback. What is left is the cost of the results WP18 chose. Measured piece by piece at
n = 1,000 (scratch harness): the zero part, R's `glm.fit` IRLS (working weights and response,
the deviance at every step, `family$aic`), takes about 170 µs where the old hand-written
Newton logit took about 140; the count part's Newton steps solve the equilibrated
information by Cholesky (about 1.5 µs a step over 19 steps) where the old code took a plain
inverse. Both loops are bound by the per-row `exp`, `ln` and divisions; fusing their passes
was measured slower, and no cheaper form gives the same values.

## References

- Mullahy, J. (1986). "Specification and Testing of Some Modified Count Data Models."
  *Journal of Econometrics*, 33(3), 341-365.
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.).
  Cambridge University Press. Chapter 4.
- Zeileis, A., Kleiber, C., & Jackman, S. (2008). "Regression Models for Count Data in R."
  *Journal of Statistical Software*, 27(8), 1-25.
