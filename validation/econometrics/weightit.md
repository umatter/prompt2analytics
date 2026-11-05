# WeightIt Validation

## Method Overview

`weightit()` (`crates/p2a-core/src/econometrics/weightit.rs`) computes balancing weights for
the ATE, ATT or ATC with four methods, each reproducing an R function:

| Method | What is computed | R reference |
|--------|------------------|-------------|
| `Logistic` | inverse propensity weights from a logistic regression (the shared GLM, `glm::fit_propensity`) | `WeightIt::weightit(method = "glm")` |
| `Entropy` | entropy balancing (Hainmueller 2012): the convex dual `min log sum exp(lambda'(x_i - t))` by Newton with an Armijo search, on scores in SD units | `WeightIt::weightit(method = "ebal")`; the exact dual (Newton to `|grad| < 1e-14`) |
| `Energy` | energy balancing (Huling & Mak 2022) at lambda = 0, solved exactly as a quadratic program | the energy-distance QP solved with `quadprog::solve.QP` (WeightIt's `energy` needs `osqp`) |
| `Stable` | stable balancing weights with exact balance, through `sbw::run_sbw_inner` | `WeightIt::weightit(method = "optweight", tols = 0)`; the SBW QP solved with `quadprog::solve.QP` |

`entropy_balance()` (ATT) and `entropy_balance_with()` (ATT, ATC, ATE; targets, iteration
controls) expose entropy balancing directly. `stabilize` applies to the logistic weights only
(WeightIt's convention); with a balancing method it is ignored with a warning.

**Weight scale (decision WP23-B).** For the balancing methods (entropy, energy, stable) and
`entropy_balance`, target units get weight 1 and each reweighted group sums to the size of the
target population: ATT controls sum to n1, ATC treated sum to n0, ATE each group sums to n.
This is WeightIt's `ebal` convention (`att_weights_sum_c` = 151 = n1 on the case). Effects,
per-group ESS and balance do not depend on the scale; the pooled `effective_sample_size` does.

## Reference Implementation

- R code: `validation/r/cases/causal_weighting.R` (cases `weights_ebal`, `weights_sbw`,
  `weights_energy`) and `validation/r/cases/causal_ipw_aipw.R` (case `ipw_aipw_sim`, the `glm`
  weights).
- Data: `validation/datasets/generated/weighting_sim.csv` (300 units, 151 treated; x1, x2,
  x3 with x3 binary; seed 20261104), written by `weights_ebal`.
- Packages: WeightIt 2.0.0, quadprog (both listed in `validation/run_validation.sh`).
- Registry: `validation/methods/causal_weighting.json` (`weightit_glm`, `weightit_ebal`,
  `weightit_energy`, `weightit_stable`).

## Reference values (weighting_sim, n = 300)

| Quantity | R value |
|----------|---------|
| ebal ATT effect (WeightIt / exact dual) | 1.5169783470990001 / 1.5169783593391748 |
| ebal ATT control ESS (exact dual) | 86.72313506696332 |
| ebal ATE effect (WeightIt / exact dual) | 1.4331592875919263 / 1.4331582180997464 |
| ebal ATC effect (exact dual) | 1.3441773322039805 |
| ebal weight sums (WeightIt): ATE treated, ATT controls, ATC treated | 300.00001136675013, 150.9999971783664, 148.99994348161445 |
| energy ATT effect, energy distance, zero weights | 1.5225448615582375, 0.007118441959559796, 23 |
| energy ATC effect, zero weights | 1.2854853751038673, 34 |
| energy ATE effect, zero weights | 1.3755492901548088, 1 |
| stable (SBW exact) ATT / ATC / ATE effect | 1.5617368270324565 / 1.3128019372669482 / 1.4307248209844157 |

## Tests and tolerances

| Test | Case | Checks | Tolerance |
|------|------|--------|-----------|
| `test_validate_weightit_logistic` | `ipw_aipw_sim` | propensity coefficients, glm ATE/ATT/ATC/stabilised weights | `Tol::coef(n)` |
| `test_validate_ebal_ate`, `test_validate_ebal_att`, `test_validate_ebal_atc` | `weights_ebal` | weights, ESS, effects against the exact dual; weights, sums and means against WeightIt | exact dual `Tol::coef(n)` (1e-8 relative), ESS `Tol::stat(n)`; WeightIt cross-check 1e-4 relative |
| `test_validate_entropy_balance_ess` | `weights_ebal` | `entropy_balance` ESS is a count (CAUS-94) | `Tol::stat(n)` |
| `test_validate_energy_att`, `test_validate_energy_atc`, `test_validate_energy_ate` | `weights_energy` | weights (mean-1 scale), weighted means, effect, zero weights, energy distance | weights 1e-8 relative + 1e-10 absolute; means 1e-8 + 1e-12; effect and energy `Tol::coef(n)`; zero count exact |
| `test_validate_stable_weights` | `weights_sbw` | ATT, ATC, ATE weights and effects | weights 1e-8 relative + 1e-12 absolute; effects `Tol::coef(n)` |

Property tests (no R values, not registered) cover the error paths and invariants:
`test_ebal_infeasible_is_error`, `test_entropy_on_infeasible_test_data_is_error`,
`test_ebal_scale_invariant`, `test_ebal_drops_collinear_covariate`,
`test_entropy_balance_with_matches_weightit`, `test_ess_properties` (all four methods),
`test_energy_estimands_differ`, `test_energy_merges_duplicate_rows` (hand-sized quadprog
reference), `test_energy_refuses_too_many_units`, `test_energy_honours_cancellation`,
`test_weightit_stable_on_infeasible_test_data_is_error`, the dense-QP tests
(`test_dense_qp_matches_quadprog`, `test_dense_qp_detects_infeasibility`,
`test_dense_qp_kkt_on_seeded_problems`) and the option and control checks.

## Implementation Notes

1. **Entropy balancing.** Each reweighted group (ATT controls, ATC treated, ATE both, to the
   pooled means) is solved as the convex dual of Hainmueller (2012) with log-sum-exp scores in
   SD units, so the solve is scale-free. A target outside a covariate's range, a negative dual
   objective (the target lies outside the convex hull) and a singular Hessian on the boundary
   are `InvalidSpecification` errors. Covariates that are linearly dependent within a group are
   dropped, as WeightIt's `make_full_rank`, and their balance is checked after the fit.
2. **Energy balancing.** The energy distance on covariates standardised over all units (R's
   `scale()`) is a QP in the weights with one sum constraint per group and `w >= 0`. Identical
   standardised rows are merged and share their weight (their Hessian rows would be equal);
   each group's sum constraint is removed by a Householder reflector, and the reduced strictly
   convex QP is solved by the dual active-set method of Goldfarb & Idnani (1983), in the
   crate-private `weightit::dense_qp`. If the reduced Hessian is numerically singular
   (near-duplicate rows), WeightIt's ridge lambda = 1e-4 is added once, with a warning.
   Licence: the solver is written from the published algorithm; no code from the GPL-2.0
   quadprog packages (R or Rust) is used (decision WP23-E).
3. **Stable balancing** is the SBW QP with exact balance (`validation/econometrics/sbw.md`); an
   infeasible exact problem is an error that points to `sbw()` with `balance_tol > 0`.
4. **Trimming** at a quantile is applied after the weights are computed (sort by
   `f64::total_cmp`).

## Known Differences

- WeightIt's `energy` defaults to `lambda = 1e-4` and `min.w = 1e-8` and solves with `osqp`;
  p2a solves the lambda = 0 problem exactly (on the case the ATT effect differs by about
  1.3e-5 from WeightIt's default).
- Infeasible balancing targets (entropy, stable) are errors; WeightIt returns zero or
  degenerate weights with a warning.
- Energy balancing reweights at most 3000 units (`ENERGY_MAX_UNITS`, decision WP23-A): the QP
  is dense (two m x m buffers, O(m^3) factorisation). Larger samples get an
  `InvalidSpecification` suggesting entropy or stable weights.
- Entropy balancing's WeightIt cross-check holds to 1e-4: WeightIt stops at its own
  tolerance, the exact dual at `|grad| < 1e-14`.

## Missing values and column types

Missing values (null or NaN) in the treatment or covariates are refused, with
"Column 'x' contains k missing value(s) (null or NaN). Drop or impute them first."
(`EconError::NullValues`, naming the column and the count). A text column gives
`NonNumericColumn`, infinite values `NonFiniteValues`, an absent column `ColumnNotFound`; a
column that exists is never reported as not found. Int64 and Boolean 0/1 treatments are
read as numbers and give the same result as Float64. Results on complete data are
unchanged.

R: WeightIt 2.0.0 refuses a missing treatment ("Missing values are not allowed in the
treatment variable.") and, for missing covariates (methods `glm` and `ebal`), warns and adds
missingness indicators; p2a refuses missing covariates instead (checked for T2-WP13). The
same holds for `entropy_balance`.

Tests: `test_validate_na_weightit_columns`, `test_validate_na_entropy_balance_columns` (case
`causal_mediation_na`, registry `validation/methods/causal_na.json`,
id `weighting_na_refusal`).

## References

- Horvitz, D.G. & Thompson, D.J. (1952). "A Generalization of Sampling Without Replacement from a Finite Universe". *JASA*, 47(260), 663-685.

- Rosenbaum, P.R. & Rubin, D.B. (1983). "The Central Role of the Propensity Score in Observational Studies for Causal Effects". *Biometrika*, 70(1), 41-55.

- Hainmueller, J. (2012). "Entropy Balancing for Causal Effects: A Multivariate Reweighting Method to Produce Balanced Samples in Observational Studies". *Political Analysis*, 20(1), 25-46.

- Zubizarreta, J.R. (2015). "Stable Weights that Balance Covariates for Estimation with Incomplete Outcome Data". *JASA*, 110(511), 910-922.

- Greifer, N. (2024). WeightIt: Weighting for Covariate Balance in Observational Studies. R package. https://ngreifer.github.io/WeightIt/

- Huling, J.D. & Mak, S. (2022). "Energy Balancing of Covariate Distributions". *Journal of Causal Inference*, 12(1).

- Goldfarb, D. & Idnani, A. (1983). "A numerically stable dual method for solving strictly convex quadratic programs". *Mathematical Programming*, 27, 1-33.
