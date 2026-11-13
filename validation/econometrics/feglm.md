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

**Status**: R-pinned (methods `feglm_mle`, `feglm_separation`, `feglm_absorbed_df` in
`validation/methods/econ_feglm_t4.json`; `feglm` in `validation/methods/econ_panel_t3.json`).
Logit, probit and Poisson with one and two fixed effects and Gaussian with two are pinned to
glm with factor dummies on the rows alpaca keeps (Case F): coefficients, inverse-Hessian SEs,
log-likelihood and null log-likelihood, AIC, BIC, deviance and null deviance, residual and
absorbed df, n. The drop rule for groups whose outcome is all 0 or all 1 is pinned in Case G,
the absorbed df on a disconnected design in Case H. The Poisson log-likelihood convention
(Case D) is compared with quoted R values, and regressors absorbed by the fixed effects are
pinned in `validation/econometrics/hdfe.md` (case `panel_absorbed_regressors`).

**Primary Reference**:
> Stammann, A. (2018). "Fast and Feasible Estimation of Generalized Linear Models with High-Dimensional k-way Fixed Effects". arXiv:1707.01815.
> https://arxiv.org/abs/1707.01815

**R Package Reference**:
> Czarnowske, D. & Stammann, A. "alpaca: Fit GLM's with High-Dimensional k-Way Fixed Effects". CRAN.
> https://cran.r-project.org/package=alpaca

## Algorithm

IRLS with weighted alternating projections of the *unscaled* working residual and regressors,
as alpaca (Stammann 2018):

1. **Separated groups**: for binomial and Poisson, every fixed-effect group whose outcome is
   all 0 (binomial and Poisson) or all 1 (binomial) is dropped, over all dimensions in turn,
   until a full round drops nothing (alpaca's and fixest's rule; their effects have no finite
   estimate). Singletons with a positive outcome stay, as alpaca keeps them (fixest drops
   them). The rows dropped are in `n_separated` and a warning; the levels left are in
   `fe_counts`.
2. **Start**: the constant linear predictor `link(mean((y + 1/2) / 2))` (binomial),
   `link(mean(y + 0.1))` (Poisson) or `mean(y)` (Gaussian), which lies in the span of the
   effects, with every slope 0.
3. **IRLS step**: with working weights `w` and the working residual `nu = z - eta`,
   ```
   MX  = weighted_demean(X, FE, w)        # warm start: the last iteration's MX
   Mnu = weighted_demean(nu, FE, w)
   step = argmin || sqrt(w) (Mnu - MX b) ||
   eta_step = nu - (Mnu - MX step)        # in the span of X and the effects
   ```
   The first iteration's warm start is the unit-weight regressor screen's demeaned X
   (`M_W (M_1 X) = M_W X`), so the screen is not an extra pass.
4. **Step halving**: `eta + rho eta_step`, `beta + rho step` with `rho` = 1, 1/2, ... (at most
   50 halvings) until the deviance is finite, the mean valid for the family, and the deviance
   no higher than the last one by more than `tolerance` (relative). alpaca asks for a decrease
   of at least the tolerance, which refuses the last Newton step near the estimate and leaves
   its coefficients about 4e-7 (relative) from glm's on Case F; p2a reaches glm's estimate.
   When no step is accepted the last iterate stays and the fit counts as converged (alpaca);
   when no step gives a finite deviance and a valid mean the fit fails with
   `ConvergenceFailure`.
5. **Convergence**: stop when `|dev - dev_old| / (|dev| + 0.1)` falls below `tolerance`
   (default 1e-8), R's `glm.fit` and fixest's `glm.tol` rule. Each weighted MAP stops on the
   relative rule of HDFE (`validation/econometrics/hdfe.md`, "Convergence of the alternating
   projections"). Reaching `max_iter` (default 25) returns the fit with `converged = false`,
   the warning "FEGLM IRLS did not converge after ..." and `Converged: no` in the output; a MAP
   that stops at its iteration limit adds one warning.
6. **Variance**: the inverse Hessian `(MX' W MX)^-1` at the final weights (X demeaned once
   more, warm-started), with dispersion 1 for binomial and Poisson (glm, alpaca, fixest) and
   deviance / df_resid for Gaussian. There is no quasi-Poisson option.
7. **Fit statistics**: the log-likelihood is glm's (`logLik`; Gaussian at sigma^2 = SSR / n);
   the null log-likelihood and deviance are the intercept-only model's on the rows used; AIC
   and BIC count the slopes, the absorbed effects and, for Gaussian, sigma, as glm with factor
   dummies does. The pseudo R-squared is McFadden's `1 - LL / LL0` (for Gaussian, on the
   normal log-likelihood).
8. **Absorbed df**: the connected-components count of `panel_hdfe` (`hdfe.md`): one way, the
   levels; two ways, the levels minus the components of the bipartite graph; each further
   dimension its levels minus 1.
9. **Warnings**: binomial fits with a fitted probability within 10 eps of 0 or 1 warn
   "fitted probabilities numerically 0 or 1 occurred" (glm's text); Poisson fits with a
   fitted rate within 10 eps of 0 warn "fitted rates numerically 0 occurred".

Regressors absorbed by the fixed effects, or collinear with others after them, are dropped
before the IRLS loop with a warning and named in `dropped` (the screen of
`validation/econometrics/hdfe.md`, "Regressors absorbed by the fixed effects").

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

**Rust tests** (not R-pinned: the data are deterministic, not an R reference):
- `econometrics::feglm::tests::test_feglm_logit_probit_ratio_plausible` (`crates/p2a-core/src/econometrics/feglm.rs`): a plausibility band, the logit/probit ratio lies in (1.2, 2.2) and the signs agree. The test that checked only β > 0 against this case was deleted in Tier 3.

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

**Rust tests** (not R-pinned: structure only on these 12 rows; the Poisson fit is pinned in Case F):
- `econometrics::feglm::tests::test_feglm_poisson_basic` (`crates/p2a-core/src/econometrics/feglm.rs`): structure only (family, a coefficient, positive dispersion).

### Case C: Binary panel (`create_binary_panel()`)

The 12-row logit panel (firm + year) is quasi-separated in R, so it cannot serve as a
reference: alpaca returns β(x) = 119.1637489044198 with SE 321373.5004204401, and fixest
stops with "The only variable, 'x', is collinear with the fixed effects".

glm with factor dummies warns "fitted probabilities numerically 0 or 1 occurred" on these rows
(R 4.3.3: β(x) = 120.65 after 24 iterations, fitted values reaching 2.2e-16), and so does
FEGLM (`econometrics::feglm::tests::test_feglm_fitted_probabilities_warning`).

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

### Case E: Gaussian, logit, probit and Poisson on a simulated panel (`feglm_families`)

`validation/r/cases/econ_panel_t3.R`, R 4.3.3, fixest 0.14.2;
`validation/datasets/generated/feglm_panel.csv` (N = 60 ids, T = 8 periods, n = 480):

```r
fixest::feglm(y_gauss ~ x + x2 | id + time, d, family = gaussian(), vcov = "iid")
fixest::feglm(y_bin ~ x + x2 | id + time, d, family = binomial("logit"), vcov = "iid")   # and "probit"
fixest::fepois(y_cnt ~ x + x2 | id + time, d, vcov = "iid")
```

| Quantity | R |
|---|---|
| Gaussian coef x, x2 (pinned) | 0.5370626024, -0.3308561515 |
| Gaussian df_resid (pinned) | 411 |
| n (pinned) | 480 |
| Gaussian SE x, x2 (`lm`) | 0.05220577326, 0.05127532555 |
| Gaussian deviance, dispersion | 429.7321626, 1.045577038 |
| Gaussian logLik (`glm`) | -654.5407439 |
| logit coef x, x2 (fixest, n = 464) | 0.8004381609, -0.4102608802 |
| probit coef x, x2 (fixest, n = 464) | 0.4865580389, -0.2515681450 |
| Poisson coef x, x2 (fixest, n = 480) | 0.2752772215, -0.1498773237 |
| Poisson SE x, x2 (`glm`) | 0.04469063846, 0.04388206037 |

Tests (method `feglm`):
- `econometrics::feglm::tests::test_validate_feglm_gaussian_families`: the Gaussian
  coefficients (README tolerance for n = 480, 1e-8; the design is balanced, so the MAP is
  exact after two sweeps), lm's SEs, the deviance and dispersion, glm's log-likelihood,
  df_resid and n.
- `econometrics::feglm::tests::test_validate_feglm_families_report_convergence`: logit,
  probit and Poisson converge, with finite coefficients and no non-convergence warning.

fixest's `vcov = "iid"` SEs (`*_se` in this case) carry fixest's small-sample factor
sqrt((n - 1) / (n - K)), which p2a, alpaca and glm do not apply, so they are not pinned; the
logit, probit and Poisson fits are pinned to glm in Case F.

### Case F: the FE-GLM MLE on `feglm_panel` (`feglm_mle`)

`validation/r/cases/econ_feglm_t4.R`, R 4.3.3, alpaca 0.3.5:
`glm(y ~ x + x2 + factor(id) [+ factor(time)], family, control = glm.control(epsilon = 1e-16))`
on the rows alpaca keeps (its drop rule, asserted against `alpaca::feglm()$nobs`); logit and
probit drop the 16 rows of entities 31 and 39, whose outcome is constant. The fits use
`tolerance = 1e-16`, `map_tolerance = 1e-12` and `max_iter = 100` (glm's epsilon: probit's
Fisher scoring converges linearly, and glm at epsilon 1e-14 still stops about 1e-8 short of
its own estimate). glm's and alpaca's inverse-Hessian SEs agree to 1e-8 relative.

| Fit | x | x2 | SE x | SE x2 |
|---|---|---|---|---|
| logit, id + time | 0.8004381700767382 | -0.41026088885247225 | 0.1339799142939801 | 0.1222242310883162 |
| logit, id | 0.7766642192084411 | -0.40827694737903536 | 0.1308781808424657 | 0.11878021203801983 |
| probit, id + time | 0.48655868581043543 | -0.25156871636428046 | 0.07764364944422303 | 0.07250365831247692 |
| probit, id | 0.4737728346532682 | -0.24731138852915355 | 0.07601623984580308 | 0.07039355962045375 |
| Poisson, id + time | 0.27527722149353595 | -0.14987732371223125 | 0.04469091775357857 | 0.04388248270963816 |
| Poisson, id | 0.29042409491583715 | -0.12788618039217853 | 0.04509796028005892 | 0.04228259036028544 |
| Gaussian, id + time | 0.5370626023737091 | -0.3308561514549285 | 0.052205773255508715 | 0.05127532554629066 |

| Fit | logLik | null logLik | AIC | BIC | deviance | null deviance | df_resid | n | df_absorbed |
|---|---|---|---|---|---|---|---|---|---|
| logit2 | -252.86653245699807 | -320.06251321479715 | 639.7330649139961 | 917.1053299131552 | 505.73306491399615 | 640.1250264295943 | 397 | 464 | 65 |
| logit1 | -258.08961019711654 | -320.06251321479715 | 636.1792203942331 | 884.5722935278084 | 516.1792203942331 | 640.1250264295943 | 404 | 464 | 58 |
| probit2 | -252.2082966027673 | -320.06251321479715 | 638.4165932055346 | 915.7888582046937 | 504.4165932055345 | 640.1250264295943 | 397 | 464 | 65 |
| probit1 | -257.45570527037376 | -320.06251321479715 | 634.9114105407475 | 883.3044836743229 | 514.9114105407475 | 640.1250264295943 | 404 | 464 | 58 |
| poisson2 | -631.5518580322216 | -770.219156764571 | 1401.1037160644432 | 1689.0949572336767 | 450.9609633411351 | 728.295560805834 | 411 | 480 | 67 |
| poisson1 | -681.4858641439072 | -770.219156764571 | 1486.9717282878144 | 1745.7464667297345 | 550.8289755645063 | 728.295560805834 | 418 | 480 | 60 |
| gauss2 | -654.5407438826985 | -860.5163956012226 | 1449.081487765397 | 1741.2465150385326 | 429.73216262558435 | 1013.7321720803457 | 411 | 480 | 67 |

n_positive: 251 for logit and probit, 330 for Poisson. Gaussian dispersion
(lm sigma^2): 1.0455770380184528. AIC = -2 logLik + 2 (k + df_absorbed), plus 2 for
Gaussian's sigma; BIC with log(n) of the rows used.

Tests (method `feglm_mle`):
- `econometrics::feglm::tests::test_validate_feglm_mle_logit`, `_probit`, `_poisson`: id + time
  and id; coefficients (README tolerance for n = 464/480), SEs, log-likelihood, null
  log-likelihood, AIC, BIC, deviance, null deviance (SE and statistics tolerances), df_resid,
  df_absorbed, n, n_positive exact; logit and probit report the 16 dropped rows. Before the
  rewrite the two-way logit gave x = 0.7002 and the two-way Poisson x = 0.4742.
- `econometrics::feglm::tests::test_validate_feglm_mle_gaussian`: the same quantities and the
  dispersion; the pseudo R-squared is McFadden's on the normal log-likelihood.
- `econometrics::feglm::tests::test_validate_feglm_mle_default_controls`: the six non-Gaussian
  fits at the default controls converge, warn only about dropped rows, and reach the
  coefficients to `Tol::iterative(1e-6)` (probit `1e-5`: Fisher scoring stops about 4e-6
  short at the deviance rule's 1e-8).

### Case G: the drop rule (`feglm_separation`)

`validation/r/cases/econ_feglm_t4.R` generates `validation/datasets/generated/feglm_sep.csv`
(181 rows, two-way): entity 1's binary outcome is all 1, entity 2's all 0, period 6 is positive
only for entity 1, entity 3's counts are all 0, and entity 31 is a singleton with a positive
outcome.

- Logit: entities 1 and 2, then period 6 (all 0 once entity 1 is gone), then 4 entities that
  leaves constant, then the singleton: 6 + 6 + 28 + 20 + 1 = 61 rows, as fixest; n = 120, 5
  periods left.
- Poisson: entity 3 (6 rows); alpaca keeps the singleton (n = 175), fixest drops it (174).
- Gaussian: nothing is dropped (n = 181).

| Fit | x | x2 | SE x | SE x2 | logLik | AIC | BIC | df_resid | df_absorbed |
|---|---|---|---|---|---|---|---|---|---|
| logit | 1.7829834942951899 | -0.7641999499383573 | 0.4009821408150966 | 0.37641505354869575 | -48.911683027708875 | 157.82336605541775 | 241.4481183388791 | 90 | 28 |
| poisson | 0.27403964028285105 | -0.29105418647955816 | 0.07698246608142499 | 0.08523053626638594 | -233.12831104263228 | 540.2566220852646 | 657.3537031204346 | 138 | 35 |
| gauss | 0.6015162527022587 | -0.10219291477899675 | 0.08068189265740196 | 0.08924948012931098 | -226.72516549843021 | 531.4503309968604 | 656.1917152162276 | 143 | 36 |

Test (method `feglm_separation`): `econometrics::feglm::tests::test_validate_feglm_separation_drop_rule`
pins the rows used and dropped, n_positive, the levels left, the warnings, and every number
above with the null log-likelihood and deviance. `test_feglm_all_rows_separated_is_refused`:
a design whose every group is constant is refused (`InsufficientData`).

### Case H: absorbed df on a disconnected design (`hdfe_disconnected`)

The two-way design of `hdfe.md`'s disconnected case (two components, n = 80): lm with dummies
gives df_absorbed 24 and df_resid 54, x 0.613056669041337, z -0.1963215773986, SE
0.11929680146935208, 0.10479347160495987. The naive rule (levels minus dimensions plus 1) gave
25 and SEs 0.12041698380526714, 0.1057774694520543. Test (method `feglm_absorbed_df`):
`econometrics::feglm::tests::test_validate_feglm_disconnected_gaussian`; and
`test_feglm_many_components_matches_hdfe` (50 disconnected 2 x 2 blocks, refused before as
`InsufficientData`) equals `panel_hdfe`'s fit.

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

| Quantity | n | Precision |
|----------|---|-----------|
| Gaussian coefficients, SEs, deviance, dispersion, logLik (Case E) | 480 | 1e-8 / 1e-6 relative |
| Logit, probit, Poisson, Gaussian coefficients (Case F) | 464 / 480 | 1e-8 relative |
| SEs, logLik, null logLik, AIC, BIC, deviance, null deviance (Cases F, G) | 120-480 | 1e-6 relative |
| Default controls (Case F) | 464 / 480 | `Tol::iterative(1e-6)`, probit `1e-5` |
| Gaussian SE, deviance, dispersion at y x 1e9 (`hdfe_map_scale`) | 1000 | `Tol::iterative(1e-6)` |
| Gaussian coefficient at y x 1e9 (`hdfe_map_scale`) | 1000 | `Tol::iterative(1e-8)` |
| Gaussian coefficients with absorbed regressors (`panel_absorbed_regressors`) | 180 | 1e-8 relative |
| Poisson LL / null LL (Case D, fixest convention) | 40 | 1e-12 × scale pointwise; 1e-10 relative (null) |

Every family's coefficients, SEs and fit statistics are pinned in Cases F-H (README
tolerances at glm's epsilon 1e-16; default controls `Tol::iterative`).

## Solve and timing

- The regressor screen and every IRLS step take OLS's solve decision: the normal equations on a
  full-rank, well-conditioned weighted demeaned design (refined once below a reciprocal
  condition of 1e-2), the pivoted QR otherwise, which still refuses a rank-deficient weighted
  design (`hdfe.md`, "Solve and timing").
- μ is computed once per update of η and serves the deviance rule, the next iteration and the
  final statistics; the final deviance is the one the convergence check computed.
- Each IRLS step's weighted demeaning of X starts from the last step's demeaned X (the first
  from the unit-weight screen's), so the alternating projections start near their fixed point;
  the final information matrix's demeaning starts from the last step's too.
- The weighted alternating projections compute the change and the norm of a sweep in one
  pass, and the acceleration terms in another, each sum accumulating in row order as before.
- (T3-PERF5) The group weight sums of a weighted demeaning do not change between sweeps,
  columns, or the z and X demeanings of one IRLS step: `WeightedProjector` computes them once
  per set of weights and demeans in place, with the same arithmetic in the same order
  (`test_weighted_projector_matches_reference_exactly`: sweeps, change, convergence and every
  value bit for bit equal to the per-pass reference, one and two unbalanced factors, zero
  weights, with and without acceleration, and a sweep limit).
- (T3-PERF5) The regressor screen's unit-weight demeaning (`hdfe::demean_matrix_map`) handed
  the columns to the rayon pool on every call, a wake-up that cost more than the work on a
  small design; one factor, one column or fewer than 2,000 rows now run on the calling thread
  (columns are independent, so the result is the same; this applies to `panel_hdfe` too).

These changed no result when made (Tier 3); the rewrite to the FE-GLM MLE (Algorithm) changed
the numbers, and the timings below predate it.

Timing: release build, one core (`taskset`, `RAYON_NUM_THREADS=1`), three interleaved rounds
against `ff8e7559` (before these changes); new / base as the ratio of medians.
Gaussian: the unified runner on R's `panel_n<n>.csv` (entity effects); logit and Poisson:
criterion's simulated panels.
The outputs the unified runner captures (coefficients, SEs, R², test statistics) equal
base's bit for bit; a pivoted-QR solve differs from them in the last one or two digits (1e-9
relative for the Hausman statistic).

| Fit | n | new / base |
|---|---|---|
| Gaussian, entity | 100 / 1,000 / 10,000 | 1.51 / 1.04 / 0.99 |
| Logit, entity | 100 / 400 / 1,000 | 1.07 / 0.93 / 0.97 |
| Logit, two-way | 100 / 400 / 1,000 | 0.80 / 0.76 / 1.14 |
| Poisson, two-way | 100 / 400 / 1,000 | 0.93 / 0.95 / 0.99 |

After T3-PERF5 (the last three bullets above), the two rows above 1.1x are within it. The
machine was shared and loaded (load average 15-70), where criterion's wall-clock rounds
scattered by up to 2x, so the lane measured user-space CPU cycles per call from the hardware
counter (`perf_event_open`; median of 11 batches, one P core, `RAYON_NUM_THREADS=1`, three
interleaved rounds) on the criterion bench's data and a generated entity panel of the same
shapes for the Gaussian rows, against `ff8e7559`. The IRLS iteration counts of the two-way
logit are 10 / 12 / 11 (base 13 / 17 / 11), and the fits equal the pre-PERF5 ones bit for bit.
The two-way logit row at n = 1,000 was re-measured with these three interleaved cycle-counter
rounds instead of the five wall-clock rounds planned; the quiet-machine re-time below decides.

| Fit | n | new / base (cycles) |
|---|---|---|
| Gaussian, entity | 100 / 1,000 / 10,000 | 0.97 / 1.02 / 0.99 |
| Logit, two-way | 100 / 400 / 1,000 | 0.50 / 0.57 / 0.90 |

Wall-clock on the unified runner (six interleaved rounds, load average about 25): Gaussian
0.99 / 0.95 / 1.01.

Wall clock on a quiet machine (2026-11-06, load average 1.2-1.6; release build, one P core, `RAYON_NUM_THREADS=1`, three interleaved rounds against `ff8e7559`, medians):

| Fit | n | new / base (wall clock) |
|---|---|---|
| Gaussian, entity (unified) | 100 / 1,000 / 10,000 | 0.96 / 0.95 / 0.94 |
| Logit, entity (criterion) | 100 / 400 / 1,000 | 0.90 / 0.91 / 0.97 |
| Logit, two-way (criterion) | 100 / 400 / 1,000 | 0.53 / 0.59 / 0.90 |
| Poisson, two-way (criterion) | 100 / 400 / 1,000 | 0.59 / 0.72 / 0.78 |

## Missing values

`run_feglm` reads y, the regressors and every fixed-effect column with one NA-aware request
and drops the rows with a missing value (null or NaN) in any of them, as fixest does; the
count is in `FeglmResult.missing` and the last line of the output. Integer columns of any
width are numbers, fixed-effect ids may be text or numeric (levels in R factor order), and a
missing id drops its row instead of becoming level 0 or "". An infinite value is refused
(`NonFiniteValues`) before the family's checks. Tests:
`econometrics::feglm::tests::test_validate_na_feglm_omit_and_missing_fe_ids` (on
`panel_grunfeld_na`: the fit equals the fit on its 193 complete rows, with the text or the
numeric firm id) and `econometrics::feglm::tests::test_feglm_non_finite_outcome`. Numbers are
not pinned to R here: they are pinned in Case F.

## Known Differences

1. **Acceleration**: p2a uses optional Gearhart-Koshy acceleration in MAP; alpaca may use different acceleration.

2. **Standard errors**: alpaca's and glm's inverse Hessian (dispersion 1 for binomial and
   Poisson), not fixest's `vcov = "iid"`, which applies a small-sample factor.

3. **Singletons**: kept, as alpaca keeps them; fixest drops them, so its n (and logLik, AIC,
   BIC) differ when a singleton has a positive outcome.

4. **Separation**: groups whose outcome is all 0 or all 1 are dropped until none is left, as
   alpaca and fixest; a fit whose probabilities still reach 0 or 1 warns as glm does.

5. **Degrees of freedom**: absorbed df by connected components for two fixed effects (exact);
   for three or more, the HDFE rule (each further dimension its levels minus 1).

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
