# Marginal Treatment Effects (MTE) by local IV

**File:** `crates/p2a-core/src/econometrics/ivmte.rs` (`run_ivmte`, `run_ivmte_multi_z`, `ivmte`; MCP tool `iv_mte`).

**Status**: R-pinned (registry `validation/methods/econ_gmm_mte.json`, id `ivmte`), to the Heckman-Vytlacil
formulas written out in R, not to an R package.

## Method

Selection `D = 1{P(Z) >= U_D}`, `U_D ~ U(0, 1)`; `MTE(u) = dE[Y | P = u] / du` (local IV).

1. First stage: P(Z) by probit (default) or logit through the shared GLM (`glm.fit`'s deviance rule; the probit
   also iterates to a relative coefficient change of 1e-10), or a linear probability model clipped to [0, 1].
   Regressors `[1, Z]`, plus X with `propensity_covariates`.
2. Second stage: least squares of Y on `[1, P, .., P^d]`, plus X with `include_covariates`.
3. `MTE(u) = sum_j j gamma_j u^(j-1)` on the grid `u_i = (i + 0.5) / n_grid`.
4. Estimands (Heckman & Vytlacil 2005, Table IB), weights normalised to sum to one:
   - ATE `h(u) = 1`;
   - ATT `h(u) = Pr(P > u) / E[P]`;
   - ATU `h(u) = Pr(P <= u) / E[1 - P]`;
   - LATE(p_min,p_max) `h(u) = 1{p_min <= u <= p_max}`;
   - PRTE: the user's `prte_weights` (for a policy P -> P', `Pr(P' > u) - Pr(P > u)`).
5. Delta method `se = sqrt(g' V g)`, `g = sum_i h(u_i) [(j + 1) u_i^j]`, V the HC1 covariance of the polynomial
   coefficients (`sandwich::vcovHC(type = "HC1")`); first-stage error not included. Pointwise curve SEs and CIs.
6. Bootstrap (`bootstrap_se`, `n_bootstrap >= 2`): units resampled with ChaCha8 seeded by `seed` (default 42),
   both stages refitted, SD and type-7 percentile intervals for every estimand and every curve point; failed
   replicates are counted and reported, resource stops propagate.
7. ATE, ATT and ATU integrate the polynomial over all of [0, 1]; outside [p_min, p_max] it is extrapolated. The
   result reports each estimand's weight share outside the support and warns above 1%. A support that contains no
   grid point is an error (LATE undefined).

R equivalent: no single function. The point-identified local-IV MTE of Brinch, Mogstad & Wiswall (2017), as Stata
`mtefe` with a polynomial; R `ivmte` (Shea & Torgovitsky) also point-identifies with a parametric MTR but solves an
LP and estimates something else (bounds), so it is not the reference. Validated against the formulas written out in
R (`validation/r/cases/econ_mte.R`).

## Cases

- `mte_local_iv`: `mte_sim.csv` (n = 2000, `MTE(u) = 2.5 - 2u`), probit on (z1, z2), cubic in P, 1001-point grid;
  also the narrow-support subsample |z1| < 0.5 (719 rows) and the PRTE of P' = min(P + 0.1, 1).
- `mte_bootstrap`: rows 1-400 of `mte_sim.csv`, 25 replicates from the committed index matrix
  `mte_boot_indices.csv`.

## Values (four digits; the JSON is canonical)

| Estimand | Estimate | HC1 delta SE |
|---|---|---|
| ATE | 1.3861 | 0.1624 |
| ATT | 1.8301 | 0.2054 |
| ATU | 0.8578 | 0.1996 |
| LATE(p_min,p_max) | 1.3847 | 0.1617 |
| PRTE (P + 0.1) | 1.1977 | 0.1444 |

Narrow support (|z1| < 0.5): support [0.2914, 0.8321], 541 of 1001 grid points; weight outside the support
ATE 46.0%, ATT 51.1%, ATU 39.1%; ATE 2.7264 (SE 3.7287).

Bootstrap (n = 400, B = 25): ATE 1.3031, SD 0.3246, percentile interval [0.7371, 1.8398].

**Tolerance.** `validation/README.md`: n = 2000 estimates 1e-10, SEs and shares 1e-8; n = 719 and n = 400 estimates
1e-8, SEs and CIs 1e-6 (relative); first-stage coefficients `Tol::iterative(1e-10)` against the probit MLE (the R
case polishes glm's IRLS result, which stops about 5e-11 short of the MLE, with three Newton steps); the
second-stage polynomial coefficients and MTE(u) `Tol::iterative(1e-8)`, because the cubic amplifies the first
stage's convergence error (1.6e-11 in a probit coefficient moves the P^3 coefficient by 4.7e-9 relative).

## Tests (`econometrics::ivmte::tests`)

R-pinned: `test_validate_mte_local_iv_estimands`, `test_validate_mte_narrow_support`, `test_validate_mte_prte_shift`,
`test_validate_mte_bootstrap_indices`. Properties: `test_mte_rejects_degenerate_grid_and_support`,
`test_mte_estimand_and_prte_validation`, `test_mte_bootstrap_seeded`, `test_mte_bootstrap_rejects_one_replication`,
`test_mte_stops_when_cancelled_in_bootstrap`, the option-parsing tests and the module's earlier tests.

## References

- Heckman, J.J., & Vytlacil, E. (2005). Structural equations, treatment effects, and econometric policy
  evaluation. *Econometrica*, 73(3), 669-738. https://doi.org/10.1111/j.1468-0262.2005.00594.x
- Brinch, C.N., Mogstad, M., & Wiswall, M. (2017). Beyond LATE with a discrete instrument. *Journal of Political
  Economy*, 125(4), 985-1039. https://doi.org/10.1086/692712
- Cornelissen, T., Dustmann, C., Raute, A., & Schönberg, U. (2016). From LATE to MTE. *Labour Economics*, 41,
  47-60. https://doi.org/10.1016/j.labeco.2016.06.004
