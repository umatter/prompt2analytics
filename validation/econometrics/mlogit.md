# Validation: conditional (McFadden) and multinomial logit

R reference: `validation/r/cases/discrete_mlogit_asc.R` (T3-WP18).

## Method Overview

The McFadden conditional logit model is used for discrete choice analysis where individuals choose among a set of alternatives. It differs from standard multinomial logit by supporting:

1. **Alternative-specific variables** (e.g., price, quality) - same coefficient across alternatives
2. **Individual-specific variables** (e.g., income, age) - different coefficient for each alternative

**Model Specification** (mlogit's default, with alternative-specific constants):
```
U_ij = alpha_j + beta'x_ij + gamma_j'z_i + e_ij,   alpha_ref = gamma_ref = 0

- x_ij = alternative-specific variables (vary by both i and j), generic beta
- z_i = individual-specific variables, one gamma_j per non-reference alternative
- alpha_j = the J - 1 alternative-specific constants (mlogit "(Intercept):j")
- e_ij ~ Type I extreme value (Gumbel)
```

`run_mlogit` and `run_conditional_logit` include the constants by default (`MlogitConfig { asc:
true, .. }`), as `mlogit(y ~ x | z)` does; they are an individual-specific `(Intercept)` column
with mlogit's names `(Intercept):boat`. `asc = false` is mlogit's `| 0 + z` (with no `z`,
`survival::clogit`'s model). `MlogitConfig::reflevel` is mlogit's `reflevel`; the default
reference is the first alternative in R `factor()` order. An unknown reference is an error
that lists the alternatives.

## Reference Implementations

| Package | Language | Function |
|---------|----------|----------|
| mlogit | R | `mlogit()` (2.0.0) |
| nnet | R | `multinom()` |
| survival | R | `clogit()` (= `asc = false`, no individual-specific variables) |

## Fishing pins (`disc_mlogit_fishing`)

Case `disc_mlogit_fishing` in [`validation/r/cases/discrete_mlogit_asc.R`](../r/cases/discrete_mlogit_asc.R)
(registry `validation/methods/discrete_t3.json`, id `discrete_mlogit_asc`), dataset
`validation/datasets/generated/mlogit_fishing.csv` (mlogit's `Fishing` in long format: 1182
choice situations x 4 alternatives; choices beach 134, boat 418, charter 452, pier 178).

```r
m1 <- mlogit(chosen ~ price + catch | income,     dl, reflevel = "beach")   # asc_*
m0 <- mlogit(chosen ~ price + catch | 0 + income, dl, reflevel = "beach")   # noasc_*
m2 <- mlogit(chosen ~ price + catch,              dl, reflevel = "beach")   # generic_asc_*
m3 <- mlogit(chosen ~ price + catch | 0,          dl, reflevel = "beach")   # generic_noasc_*
m4 <- mlogit(chosen ~ price + catch | income,     dl, reflevel = "pier")    # pier_*
```

Each model is re-solved by Newton's method on the analytic score and information (the `*_tight`
keys; SEs from the inverse information) and mlogit's own estimates are kept (`asc_coef`,
`asc_se`, `noasc_coef`, `generic_asc_coef`).

| Key | price | catch | (Intercept):boat | (Intercept):charter | (Intercept):pier | logLik |
|---|---|---|---|---|---|---|
| `asc_coef_tight` | -0.02511657127 | 0.3577819542 | 0.5272787696 | 1.694365736 | 0.7779593984 | -1215.13760391 |
| `noasc_coef_tight` | -0.02159738598 | 0.7120786078 | | | | -1257.129278831 |
| `generic_asc_coef_tight` | -0.02478955088 | 0.3771688521 | 0.8713749155 | 1.498888411 | 0.3070552453 | -1230.783830415 |
| `generic_noasc_coef_tight` | -0.02047653464 | 0.9530985061 | | | | -1311.979617108 |

With constants, income is `income:boat` 8.943982072e-05, `income:charter` -3.329172664e-05,
`income:pier` -1.275771503e-04 (SEs about 5.0e-05); `asc_aic` 2446.275207819. Without
constants, price and catch are -0.0216 and 0.712 (ll -1257.13); with them, -0.0251 and 0.358
(ll -1215.14): the fit p2a reported before T3-WP18 was the `| 0` model.

Tests (coefficients `Tol::iterative(1e-9)`, SEs 1e-8 relative at the Newton-polished optimum,
logLik and AIC `Tol::stat(1182)`; mlogit's own estimates at 1e-6; counts exact):

- `econometrics::discrete::conditional_logit::tests::test_validate_mlogit_fishing_asc`
- `econometrics::discrete::conditional_logit::tests::test_validate_mlogit_fishing_noasc` (`asc = false`)
- `econometrics::discrete::conditional_logit::tests::test_validate_conditional_logit_fishing_generic`
  (generic attributes only, with and without constants)
- `econometrics::discrete::conditional_logit::tests::test_validate_mlogit_fishing_reflevel`
  (`reflevel = "pier"`: `pier_*`; the log-likelihood is the same)
- `econometrics::discrete::conditional_logit::tests::test_validate_mlogit_convergence_is_scale_free`
- `econometrics::discrete::conditional_logit::tests::test_validate_mlogit_aliased_attribute_is_an_error`

## Identification

An attribute that is a linear combination of the others, or constant within every choice
situation (it cancels from the choice probabilities), is refused with `SingularMatrix` naming
it; mlogit refuses `price2 = 2 price` too (`aliased_refused` = 1). The check is a QR of the
features centred within each situation, at `LM_QR_TOL`.

## Convergence

Newton's method on the centred information, solved on the equilibrated matrix by Cholesky (a
generalized inverse, with a warning, only when Cholesky refuses it). It stops when the Newton
decrement is below 1e-10 on `glm.fit`'s relative-deviance scale, then takes that last full step;
a line search that finds no increase is reported and the fit is not converged. Newton's method
is affine invariant: with income scaled by 1e-6 or 1e6 the iteration counts agree within one
across scales (p2a's own invariance; the coefficients change only by the scale). Not
R-pinned: iteration counts.

## Separation

When some fitted choice probability is numerically 0 or 1 (within 10 eps), the result sets
`separation` and warns; no maximum likelihood estimate exists
(`econometrics::discrete::conditional_logit::tests::test_mlogit_separation_warns`, not R-pinned).
A singular information at the estimates gives `NaN` standard errors and a warning.

## Multinomial logit (`run_multinom`, `disc_multinom_overlap`)

Case `disc_multinom_overlap` (registry id `discrete_multinom_nnet`; n = 300, counts A 90, B 134,
C 76): `nnet::multinom(y ~ x1 + x2)` re-solved by Newton's method on the analytic score and
information (`*_tight`; nnet's own vcov at reltol 1e-16 agrees to 1e-6), with nnet at its
defaults as `coef_pkg` (8.3e-6 off the MLE).

| Key | (Intercept) | x1 | x2 |
|---|---|---|---|
| `coef_tight` B vs A | 0.3785881018 | 0.7659098536 | -0.5457958153 |
| `coef_tight` C vs A | -0.3097380458 | 0.2766702345 | 0.5587418141 |
| `se_tight` B vs A | 0.1474184159 | 0.1700127214 | 0.1676288313 |
| `se_tight` C vs A | 0.1788186365 | 0.1773426071 | 0.1794021489 |
| `refB_coef_tight` C vs B | -0.6883261477 | -0.4892396191 | 1.104537629 |

`loglik` -287.6434018959, `aic` 587.2868037917, `null_loglik` -320.705605012.

- `econometrics::discrete::multinomial_logit::tests::test_validate_multinom_overlap`: reference A;
  coefficients `Tol::iterative(1e-8)`, SEs 1e-6, logLik, AIC and null logLik `Tol::stat(300)`,
  `coef_pkg` 1e-4; converged, no warning.
- `validation_new_methods::test_validate_multinom_overlapping` (`crates/p2a-core/tests/`):
  reference B through the crate-root API (`refB_*`).
- `econometrics::discrete::multinomial_logit::tests::test_validate_multinom_convergence_is_scale_free`:
  x1 scaled by 1e-6 and 1e6, iteration counts within one, coefficients back to `coef_tight`.
- `econometrics::discrete::multinomial_logit::tests::test_validate_multinom_aliased_regressor_is_an_error`:
  `x1b = 2 x1` is `SingularMatrix` naming x1b (nnet returns an arbitrary split).
- Separation (not R-pinned): when an observation's fitted probability of its own category is
  numerically 0 or 1, `separation` is set, a warning counts the observations and the fit is not
  converged (`econometrics::discrete::multinomial_logit::tests::test_multinom_complete_separation_warns`);
  a tiny probability of a category not chosen is an ordinary tail (`disc_multinom_sep5`). One
  observation per category gives a singular information: `NaN` standard errors and a warning
  (`econometrics::discrete::multinomial_logit::tests::test_multinom_one_row_per_category_does_not_converge`).
  The Newton iteration is the one above, with one polishing step after the last full step.

The tight `nnet` pins on 4, 5 and 8 categories are in [discrete_newton.md](discrete_newton.md).

---

### Not R-pinned: small fixtures

- `econometrics::discrete::conditional_logit::tests::test_mlogit_basic` (a copy is
  `econometrics::discrete::tests::test_mlogit_basic` in `discrete/mod.rs`): 5 choice situations,
  3 alternatives, shapes and a negative cost coefficient.

## Conditional logit on `mixed_logit.csv` (Newton-polished mlogit, no constants)

Case `disc_mixed_logit` in [`validation/r/cases/discrete_mixed.R`](../r/cases/discrete_mixed.R)
(registry `validation/methods/discrete_mixed.json`, id `conditional_logit_choice_data`;
300 choice situations x 3 alternatives a, b, c). mlogit's own conditional logit stops with
`|gradient|` ~ 1e-3, 1.5e-5 relative from the MLE, so the case re-solves the same
likelihood with Newton steps on numDeriv's gradient and Hessian (`clogit_*_tight`; SEs from
`numDeriv::hessian`) and records mlogit's estimate as `clogit_coef`.

```r
cl <- mlogit(choice ~ x + w | 0, data = dfidx(d, idx = c("id", "alt"), choice = "choice"))
```

The model has no constants, so the test fits it with `run_conditional_logit_with` and
`asc = false`.

| Key | x | w |
|---|---|---|
| `clogit_coef_tight` | 0.5894383442012155 | -0.5523202005586988 |
| `clogit_se_tight` | 0.0839638977069438 | 0.08344026152700368 |
| `clogit_loglik_tight` | -281.77611853131634 | |
| `clogit_coef` (mlogit's own) | 0.5894296210995991 | -0.5523119773028726 |

**Rust test** (R-pinned): `econometrics::discrete::conditional_logit::tests::test_validate_conditional_logit_mixed_logit_data`:
beta 1e-6 and SEs 1e-5 relative, log-likelihood 1e-6, mlogit's own estimate at 1e-4,
`converged`, 300 situations. The mixed logit with no random coefficient gives the same fit
([mixed_logit.md](mixed_logit.md)).

---

## Missing values

Every column (choice-situation id, alternative id, choice indicator, alternative- and
individual-specific variables) comes from one request with `NaPolicy::Fail`: a missing
cell is refused with `NullValues` naming the column, as R's mlogit refuses it (case
`disc_choice_na`: `mlogit_refuses` = [1, 1, 1] for a missing choice, attribute and id;
mlogit's messages are "impossible to coerce the choice variable to a logical" and "missing
value where TRUE/FALSE needed"). Before, a missing attribute was read as 0 and a missing id
grouped the rows into one situation "".

Not R-pinned tests (`crates/p2a-core/src/econometrics/discrete/conditional_logit.rs`):

- `econometrics::discrete::conditional_logit::tests::test_na_conditional_logit_missing_cells_refused`:
  `run_conditional_logit` and `run_mlogit` refuse a missing choice, attribute and id.
- `econometrics::discrete::conditional_logit::tests::test_na_conditional_logit_int64_attribute`:
  an Int64 attribute gives the fit of its Float64 copy.
- `econometrics::discrete::conditional_logit::tests::test_na_conditional_logit_text_attribute_refused`:
  a text attribute is `NonNumericColumn`.

## Data types

The choice indicator and the attributes may be Float64, any integer type or Boolean. The
ids may be text, integers or floats; situations and alternatives follow R `factor()` order
(numbers numerically, text by byte), so integer alternative ids 2..12 are ordered 2 < 10
and the default reference alternative is the first in that order (it followed string
order). The likelihood is summed over the situations in that order.

## Timing (T3-PERF2)

Release build, `taskset -c 1`, `RAYON_NUM_THREADS=1`; base is the prebuilt `fe13b889` bench
binaries of the wave-2A integration timing, head the T3-PERF2 lane; three interleaved rounds,
medians of the per-run medians (`unified_benchmarks`, criterion `new_methods_benchmarks`, and the
integration's scratch timing crate for the conditional logit). The machine was shared with other
lanes' builds; the integration re-times on a quiet machine. Both models now evaluate the accepted
Newton step with its derivatives, which the next iteration and the standard errors reuse,
count separation in that pass, and run the aliasing QR only when the normal-equations test
(`certainly_full_rank`) cannot certify full rank.

| Fit | n: base µs, head/base | | | |
|---|---|---|---|---|
| multinomial logit (criterion `multinom`) | 100: 80.6, 0.97 | 500: 371, 0.92 | 1,000: 736, 0.92 | 5,000: 3,695, 0.91 |
| multinomial logit (unified) | 100: 81.5, 1.01 | 1,000: 718, 0.93 | 10,000: 7,322, 0.91 | |
| conditional logit, constants (3 alternatives, situations) | 300: 229, 1.06 | 1,000: 792, 1.03 | 3,000: 2,402, 1.01 | |
| conditional logit, no constants (head; base fits constants) | 300: 230, 0.92 | 1,000: 790, 0.85 | 3,000: 2,400, 0.84 | |

Within 1.1x of `fe13b889` at every size.

## References

- McFadden, D. (1974). "Conditional Logit Analysis of Qualitative Choice Behavior."
  In P. Zarembka (Ed.), *Frontiers in Econometrics* (pp. 105-142). Academic Press.
- Train, K. E. (2009). *Discrete Choice Methods with Simulation* (2nd ed.). Cambridge University Press.
- R Package: `mlogit` (Yves Croissant). https://cran.r-project.org/package=mlogit
