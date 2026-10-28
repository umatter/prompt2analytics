# Validation: Fixed Effects Generalized Linear Model (FEGLM)

## Method Overview

Fixed Effects Generalized Linear Model (FEGLM) combines GLM estimation with high-dimensional fixed effects absorption. It uses Iteratively Reweighted Least Squares (IRLS) with weighted Method of Alternating Projections (MAP) for efficient estimation without creating dummy variables.

**Key Parameters**:
- `y_col`: Outcome variable
- `x_cols`: Predictor variables
- `fe_cols`: Fixed effect variables to absorb
- `family`: GLM family (Logit, Probit, Poisson, Gaussian)
- `config`: Optional configuration (IRLS tolerance, MAP settings)

**Supported Families**:
- **Logit**: Binary outcomes with logistic link
- **Probit**: Binary outcomes with probit link
- **Poisson**: Count data with log link
- **Gaussian**: Continuous outcomes with identity link (reduces to HDFE)

**Use Cases**:
- Binary outcomes with firm + year fixed effects
- Count data with high-dimensional FE
- Multinomial/conditional logit approximations
- Trade data with exporter + importer FE

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| alpaca | R | `feglm()` | 0.3.5 (R 4.3.3) |
| fixest | R | `feglm()`/`fepois()` | 0.14.2 (R 4.3.3) |
| fixest | R | `fepois()$ll_null`, with `stats::dpois()` for the pointwise values (Case D) | 0.14.2 (R 4.3.3) |
| ppmlhdfe | Stata | `ppmlhdfe` | 2.x |

**Status**: not R-pinned (X-10). Four tests compare quoted R values (the three
`test_validate_*` tests of Case D and `test_poisson_log_lik_large_counts`, added in Tier 0), all
for the Poisson log-likelihood convention and fixest's null log-likelihood. No test compares
an FEGLM coefficient, SE or deviance with R. Estimator issues: IRLS does not converge to the
FE-GLM MLE (ECON-20), Gaussian log-likelihood (ECON-66), quasi-Poisson SEs (ECON-67), absorbed
degrees of freedom (ECON-85).

**Primary Reference**:
> Stammann, A. (2018). "Fast and Feasible Estimation of Generalized Linear Models with High-Dimensional k-way Fixed Effects". arXiv:1707.01815.
> https://arxiv.org/abs/1707.01815

**R Package Reference**:
> Czarnowske, D. & Stammann, A. "alpaca: Fit GLM's with High-Dimensional k-Way Fixed Effects". CRAN.
> https://cran.r-project.org/package=alpaca

## Algorithm

The FEGLM algorithm iterates between:

1. **IRLS Update**: Compute working weights and working response
   ```
   w_i = (dmu/deta)^2 / V(mu_i)
   z_i = eta_i + (y_i - mu_i) * (deta/dmu)
   ```

2. **Weighted MAP**: Demean z and X using weighted alternating projections
   ```
   z_tilde = weighted_demean(z, FE, w)
   X_tilde = weighted_demean(X, FE, w)
   ```

3. **WLS Step**: Solve weighted least squares
   ```
   beta_new = (X_tilde' W X_tilde)^-1 X_tilde' W z_tilde
   ```

4. **Convergence**: Check coefficient change < tolerance

## Test Cases

Cases A-C use the datasets the Rust tests build (`crates/p2a-core/src/econometrics/feglm.rs`:
`create_validation_dataset()`, `create_count_panel()`, `create_binary_panel()`), rebuilt in R.
Values are R 4.3.3, alpaca 0.3.5 and fixest 0.14.2, printed with
`formatC(digits = 16, format = "g")`. Case D is Tier 0's log-likelihood case.

### Case A: Logit and probit with two-way fixed effects (`create_validation_dataset()`)

**R Code**:
```r
library(alpaca); library(fixest)
vd <- expand.grid(t = 0:9, i = 0:4)[, c("i", "t")]          # same row order as the Rust loop (i outer, t inner)
vd$x <- ((vd$i * 17 + vd$t * 31) %% 100) / 50 - 1
eta <- vd$x + (vd$i - 2) * 0.5 + (vd$t - 4.5) * 0.2
vd$y <- as.numeric(plogis(eta) > ((vd$i * 13 + vd$t * 7) %% 100) / 100)
vd$id <- factor(vd$i + 1); vd$time <- factor(vd$t + 1)
summary(alpaca::feglm(y ~ x | id + time, data = vd, family = binomial("logit")))
summary(fixest::feglm(y ~ x | id + time, data = vd, family = binomial("logit"), vcov = "iid"))
# the same with binomial("probit")
```

**R results** (n = 50, sum(y) = 22):

| | alpaca logit | fixest logit | alpaca probit | fixest probit |
|---|---|---|---|---|
| β(x) | 2.93615099246126 | 2.93616805637299 | 1.643750005584502 | 1.643764478565367 |
| SE | 0.9908949634735189 (inverse Hessian) | 1.180525942218305 (`vcov = "iid"`, small-sample adjusted) | 0.533199963204614 | 0.6352590440339755 |
| deviance / logLik | deviance 36.36794582454363 | logLik -18.18397291202938 | deviance 36.64308236193075 | logLik -18.32154118088564 |

- 45 of 50 observations are used, because time 2 has y = 1 throughout.
- alpaca and fixest agree to about 6e-6 relative. That is their convergence tolerance, so a
  pinned case needs `Tol::iterative`.
- The SEs differ by convention: alpaca's are inverse-Hessian, fixest `vcov = "iid"` applies a
  small-sample factor. A case must name which one it pins.
- R's logit/probit ratio is 2.93615099246126 / 1.643750005584502 = 1.786247.

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with X-10):
- `econometrics::feglm::tests::test_validate_feglm_logit_against_alpaca` (`crates/p2a-core/src/econometrics/feglm.rs`): β > 0, pseudo-R² in (0, 1), SE > 0; none of alpaca's numbers.
- `econometrics::feglm::tests::test_validate_feglm_probit_coefficient_scaling` (`crates/p2a-core/src/econometrics/feglm.rs`): the logit/probit ratio lies in (1.2, 2.2) and the signs agree.

### Case B: Poisson with two-way fixed effects (`create_count_panel()`)

**R Code**:
```r
cp <- data.frame(firm = factor(rep(c("A", "B", "C"), each = 4)), year = factor(rep(2020:2023, 3)),
                 x = c(1.0, 1.5, 1.2, 1.8, 2.0, 2.3, 2.1, 2.5, 0.5, 0.8, 0.6, 1.0),
                 count = c(2, 4, 3, 5, 8, 10, 9, 12, 1, 2, 1, 3))
alpaca::feglm(count ~ x | firm + year, data = cp, family = poisson())
fixest::fepois(count ~ x | firm + year, data = cp, vcov = "iid")
```

**R results** (n = 12): alpaca β(x) 1.160068290949554 (SE 2.803479169540262); fixest
1.160073314420888 (SE iid 4.158232662532352, logLik -19.56317883072528).

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with X-10):
- `econometrics::feglm::tests::test_feglm_poisson_basic` (`crates/p2a-core/src/econometrics/feglm.rs`): structure only (family, a coefficient, positive dispersion).

### Case C: Binary panel (`create_binary_panel()`)

The 12-row logit panel (firm + year) is quasi-separated in R, so it cannot serve as a
reference: alpaca returns β(x) = 119.1637489044198 with SE 321373.5004204401, and fixest
stops with "The only variable, 'x', is collinear with the fixed effects".

**Rust tests** (not R-pinned: no R value exists for these data):
- `econometrics::feglm::tests::test_feglm_logit_basic` (`crates/p2a-core/src/econometrics/feglm.rs`): n = 12, fixed-effect dimensions and counts, a coefficient and a finite log-likelihood.
- `econometrics::feglm::tests::test_feglm_probit_basic` (`crates/p2a-core/src/econometrics/feglm.rs`): family and a coefficient.
- `econometrics::feglm::tests::test_feglm_logit_single_fe` (`crates/p2a-core/src/econometrics/feglm.rs`): one fixed-effect dimension and a coefficient.

### Case D: Poisson Log-Likelihood and Null Log-Likelihood (fixest Convention)

**Convention**: The Poisson log-likelihood of one observation is
```
y*log(mu) - mu - lgamma(y + 1)
```
For integer `y` this equals `dpois(y, mu, log = TRUE)`. For non-integer `y`, which PPML allows, `dpois` returns `-Inf`, so p2a follows fixest and keeps the closed form. The null log-likelihood is the intercept-only model, the sum of that expression at `mu = mean(y)`, which is what `fepois(...)$ll_null` reports even when fixed effects are absorbed. The closed form is O(1) in `y`, so counts near 1e12 are evaluated directly.

**R Code** (R 4.3.3, fixest 0.14.2):
```r
dpois(0, 2.5, log = TRUE)            # -2.5
dpois(3, 2.5, log = TRUE)            # -1.5428872736055899
dpois(17, 12.3, log = TRUE)          # -3.1418859897045666
dpois(1e6, 999000, log = TRUE)       # -8.327027479053644
2.5 * log(3) - 3 - lgamma(3.5)       # -1.4544428806767997 (fixest convention; dpois gives -Inf)
dpois(1e12, 1e12 + 1e6, log = TRUE)  # -15.234448757835947

library(fixest)
i <- 0:39
d <- data.frame(id = i %/% 8, x = ((i * 7) %% 11) / 10)
d$y1 <- ((i * 37) %% 23) + 1 + 0.25 * (i %% 3)
d$y2 <- 1e12 + ((i * 7919) %% 1000003) * 1e6
fepois(y1 ~ x | id, d)$ll_null                               # -165.74049709539213
sum(d$y2 * log(mean(d$y2)) - mean(d$y2) - lgamma(d$y2 + 1))  # -145041993560.08203
fepois(y2 ~ x | id, d)$ll_null                               # -145041993560 (same, printed)
```

The Rust tests build the same 40-row panel (`id = i / 8`, `x = ((i * 7) % 11) / 10`, five groups of eight), with `y1` as the non-integer outcome and `y2` as the large-count outcome.

**Results Comparison**:

| Statistic | y | mu | R | Tolerance |
|-----------|---|----|---|-----------|
| log-likelihood | 2.5 | 3 | -1.4544428806767997 | 1e-12 × scale |
| log-likelihood | 0 | 2.5 | -2.5 | 1e-12 × scale |
| log-likelihood | 3 | 2.5 | -1.5428872736055899 | 1e-12 × scale |
| log-likelihood | 17 | 12.3 | -3.1418859897045666 | 1e-12 × scale |
| log-likelihood | 1e6 | 999000 | -8.327027479053644 | 1e-12 × scale |
| null log-likelihood, `y1` | non-integer | mean(y1) | -165.74049709539213 | 1e-10 relative |
| null log-likelihood, `y2` | integers, 1e12 to 1.31e12 | mean(y2) | -145041993560.08203 | 1e-10 relative |

`scale` is `|y*log(mu)| + mu + lgamma(y + 1)` (at least 1): the pointwise value is a difference of terms of that size, so its error is bounded relative to them rather than to the result. The `dpois(1e12, 1e12 + 1e6)` value is checked by `test_poisson_log_lik_large_counts` at 1e-14 × scale (about 0.55), the cancellation floor that fixest's closed form shares; R's `dpois` uses Loader's saddle point instead.

**Rust tests** (quoted R values typed into the tests; not R-pinned: no committed case):
- `econometrics::feglm::tests::test_validate_poisson_log_lik_against_r` (`crates/p2a-core/src/econometrics/feglm.rs`): the pointwise values at 1e-12 × scale.
- `econometrics::feglm::tests::test_validate_feglm_poisson_null_loglik_fixest_non_integer` and `econometrics::feglm::tests::test_validate_feglm_poisson_null_loglik_fixest_large_counts` (`crates/p2a-core/src/econometrics/feglm.rs`): the null log-likelihood at 1e-10 relative.
- `econometrics::feglm::tests::test_poisson_log_lik_large_counts` (`crates/p2a-core/src/econometrics/feglm.rs`): `dpois(1e12, 1e12 + 1e6)` at 1e-14 × scale.

### The previous examples, recomputed

These examples were in this document before; their DGPs and R code are kept and the values
are recomputed with R 4.3.3 and alpaca 0.3.5. `validation/r_comparison/feglm_comparison.R`
computes alpaca values on its own simulated data, which no Rust test uses; its poisson_gravity
value equals Test 3.

#### Test 1: Logit with Single Fixed Effect

**Data Generating Process**:
```
P(Y=1|X,FE) = Λ(1.0*x + α_id)
```

Where Λ(·) is the logistic CDF.

**R Code**:
```r
library(alpaca)

# Panel: 3 entities, 4 time periods
df <- data.frame(
  y = c(1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0),
  x = c(1.5, 2.0, 1.8, 0.3, 0.5, 1.5, 0.2, 0.4, 1.6, 1.9, 1.7, 0.4),
  id = factor(c(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3))
)

est <- feglm(y ~ x | id, data = df, family = binomial())
summary(est)

coef(est)
```

**R result** (alpaca 0.3.5): the data are separated, and alpaca returns β(x) =
44.53744995693865 (SE 97162.8030682735), not a value near 1.0. No Rust test runs these data.

---

#### Test 2: Probit with Two-Way Fixed Effects

**Data Generating Process**:
```
P(Y=1|X,FE) = Φ(0.5*x + α_id + γ_time)
```

Where Φ(·) is the standard normal CDF.

**R Code**:
```r
library(alpaca)

# Simulated panel
set.seed(42)
n <- 100
id <- factor(rep(1:10, each = 10))
time <- factor(rep(1:10, times = 10))
x <- rnorm(n)
id_eff <- rnorm(10)[as.numeric(id)]
time_eff <- rnorm(10)[as.numeric(time)]
latent <- 0.5 * x + id_eff + time_eff
y <- as.integer(pnorm(latent) > runif(n))

df <- data.frame(y = y, x = x, id = id, time = time)

est <- feglm(y ~ x | id + time, data = df, family = binomial("probit"))
summary(est)
```

**R result** (alpaca 0.3.5): β(x) = 0.8503073672779772 (SE 0.3284807821698389), using 81 of the
100 observations. No Rust test runs these data.

---

#### Test 3: Poisson with Fixed Effects (Trade Model)

**Data Generating Process**:
```
E[Y|X,FE] = exp(1.0*x + α_exporter + γ_importer)
```

This is the canonical gravity model specification.

**R Code**:
```r
library(alpaca)

# Simulated trade data
df <- data.frame(
  y = c(5, 10, 3, 8, 12, 4, 6, 15, 2, 9, 11, 5),
  x = c(0.5, 1.0, 0.2, 0.8, 1.2, 0.3, 0.6, 1.5, 0.1, 0.9, 1.1, 0.5),
  exporter = factor(c(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)),
  importer = factor(c(1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4))
)

est <- feglm(y ~ x | exporter + importer, data = df, family = poisson())
summary(est)

# Check log-likelihood and coefficient
logLik(est)
coef(est)
```

**R result** (alpaca 0.3.5): β(x) = 1.291533402050926 (SE 0.315821702567803), deviance
0.7057309281389577. This equals the poisson_gravity value 1.29153340205098 of
`validation/r_comparison/output/feglm_validation_results.csv`. No Rust test runs these data.

---

#### Test 4: Gaussian with HDFE (Equivalence Test)

**Data Generating Process**:
```
y = 2.0*x1 + 1.0*x2 + α_id + γ_firm + ε
```

Gaussian FEGLM should produce identical results to linear HDFE.

**R Code**:
```r
library(alpaca)
library(lfe)

set.seed(42)
n <- 200
id <- factor(rep(1:10, 20))
firm <- factor(rep(1:5, 40))
x1 <- rnorm(n)
x2 <- rnorm(n)
id_eff <- rnorm(10)[as.numeric(id)]
firm_eff <- rnorm(5)[as.numeric(firm)]
y <- 2.0 * x1 + 1.0 * x2 + id_eff + firm_eff + rnorm(n, sd = 0.5)

df <- data.frame(y = y, x1 = x1, x2 = x2, id = id, firm = firm)

# FEGLM with Gaussian
est_feglm <- feglm(y ~ x1 + x2 | id + firm, data = df, family = gaussian())

# Linear HDFE for comparison
est_felm <- felm(y ~ x1 + x2 | id + firm, data = df)

# Coefficients should match
coef(est_feglm)
coef(est_felm)
```

**R result**: alpaca 0.3.5 refuses the gaussian family ("Linear models are not supported").
`lfe::felm` (3.1.1) and `fixest::feols` (0.14.2) give x1 2.047950984428787, x2
0.9932562858243857 (felm SE 0.0398445438825219 / 0.04109260194985836; feols iid
0.04027530822722095 / 0.04153685920633746). No Rust test runs these data.

---

#### Test 5: Perfect Separation Detection

**R Code**:
```r
library(alpaca)

# Perfect separation: x perfectly predicts y within each group
df <- data.frame(
  y = c(0, 0, 1, 1, 0, 0, 1, 1),
  x = c(-1, -1, 1, 1, -1, -1, 1, 1),
  id = factor(c(1, 1, 1, 1, 2, 2, 2, 2))
)

# Should warn about separation or produce very large coefficients
est <- feglm(y ~ x | id, data = df, family = binomial())
```

**R result** (alpaca 0.3.5): β(x) = 23.20289483571356 (SE 38627.76991655862), a diverging
estimate. No Rust test runs these data.

---

#### Test 6: Coefficient Recovery Test

**Data Generating Process**:
```
P(Y=1|X) = Λ(2.0*x1 - 1.0*x2)
```

Generate data from known DGP and verify coefficient recovery.

**R Code**:
```r
library(alpaca)

set.seed(123)
n <- 1000
id <- factor(rep(1:50, 20))
x1 <- rnorm(n)
x2 <- rnorm(n)
id_eff <- rnorm(50)[as.numeric(id)]
latent <- 2.0 * x1 - 1.0 * x2 + id_eff
prob <- 1 / (1 + exp(-latent))
y <- rbinom(n, 1, prob)

df <- data.frame(y = y, x1 = x1, x2 = x2, id = id)

est <- feglm(y ~ x1 + x2 | id, data = df, family = binomial())
coef(est)
```

**R result** (alpaca 0.3.5): β(x1) = 2.319443353373118, β(x2) = -1.162249843953616 (SE
0.1577458275089327 / 0.1129665746020946). No Rust test runs these data.

---

## Link Functions

| Family | Link | η = g(μ) | μ = g⁻¹(η) | V(μ) |
|--------|------|----------|------------|------|
| Logit | logit | log(μ/(1-μ)) | exp(η)/(1+exp(η)) | μ(1-μ) |
| Probit | probit | Φ⁻¹(μ) | Φ(η) | φ(η)²/Φ(η)(1-Φ(η)) |
| Poisson | log | log(μ) | exp(η) | μ |
| Gaussian | identity | μ | η | 1 |

## Working Weights and Response

For IRLS, we compute:

**Working weights**: `w = (∂μ/∂η)² / V(μ)`
**Working response**: `z = η + (y - μ) × (∂η/∂μ)`

| Family | w | z |
|--------|---|---|
| Logit | μ(1-μ) | η + (y-μ)/(μ(1-μ)) |
| Probit | φ(η)²/[Φ(η)(1-Φ(η))] | η + (y-μ)×(1-Φ)Φ/φ |
| Poisson | μ | η + (y-μ)/μ |
| Gaussian | 1 | y |

## Numerical Precision Summary

No coefficient, SE or deviance tolerance is claimed until an R-pinned case exists (X-10).
FEGLM is iterative, so such a case uses `Tol::iterative` with the documented convergence
tolerance.

| Quantity | n | Precision |
|----------|---|-----------|
| Poisson LL / null LL (Case D, fixest convention) | 40 | 1e-12 × scale pointwise; 1e-10 relative (null) |

## Known Differences

1. **Acceleration**: p2a uses optional Gearhart-Koshy acceleration in MAP; alpaca may use different acceleration.

2. **Standard Errors**: p2a currently computes standard SEs from information matrix; alpaca supports additional SE estimators.

3. **Separated Data**: Different handling of quasi/complete separation in binary models.

4. **Convergence Criteria**: IRLS convergence based on max coefficient change; alpaca may use different criterion.

5. **Degrees of Freedom**: Both use: `df = n - k - sum(FE levels) + absorbed`

## Running the Tests

```bash
# All FEGLM tests
cargo test -p p2a-core --lib -- econometrics::feglm::tests

# The fixest log-likelihood checks (Case D)
cargo test -p p2a-core --lib -- econometrics::feglm::tests::test_validate_poisson_log_lik
cargo test -p p2a-core --lib -- econometrics::feglm::tests::test_validate_feglm_poisson_null_loglik
```

## References

- Stammann, A. (2018). "Fast and Feasible Estimation of Generalized Linear Models with High-Dimensional k-way Fixed Effects". arXiv:1707.01815.
- Santos Silva, J.M.C. & Tenreyro, S. (2006). "The Log of Gravity". *Review of Economics and Statistics*, 88(4), 641-658.
- Fernández-Val, I. & Weidner, M. (2016). "Individual and Time Effects in Nonlinear Panel Models with Large N, T". *Journal of Econometrics*, 192(1), 291-312.
- Czarnowske, D. & Stammann, A. (2020). "alpaca: Fit GLM's with High-Dimensional k-Way Fixed Effects". *R package*.
