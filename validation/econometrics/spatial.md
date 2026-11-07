# SAR, spatial Durbin, SEM and SAC models

`crates/p2a-core/src/econometrics/spatial.rs`: `run_sar` (with `SarConfig::durbin` for the SDM),
`run_sem`, `run_sac`. Registry entry `spatial_ml` in `validation/methods/spatial_ml_weights.json`.

R-pinned to the formulas of `spatialreg::lagsarlm(method = "eigen", Durbin = FALSE/TRUE)`,
`errorsarlm`, `sacsarlm` and `impacts`, written out in base R, because spatialreg is not installed.
The case does not run spatialreg itself.

## Models

- SAR: y = ρWy + Xβ + ε. SDM: X also holds WX as `lag.<name>`.
- SEM: y = Xβ + u, u = λWu + ε.
- SAC: y = ρWy + Xβ + u, u = λWu + ε.

The concentrated log-likelihood is

    L(ρ) = -(n/2)(1 + ln 2π + ln σ²(ρ)) + log|I - ρW|,   σ²(ρ) = RSS(ρ) / n

The SAC adds log|I - λW|. The log-determinant is Re Σ log(1 - ρλ_i) over the complex eigenvalues of
W (`validation/spatial/weights.md`). The SAR and the SDM use Ord's regressions of y and Wy on X, so
RSS is a quadratic in ρ.

## Search

ρ and λ are searched by golden section on spatialreg's `1/range(Re(eig))`, less 1e-8 at each end.
The SAC runs a grid over the same interval, then coordinate descent with a golden section per
coordinate. A W whose interval is not finite is `InvalidSpecification`; spatialreg's `eig.range` is
not finite there either.

Results carry `convergence` and `warnings` (`ResultNotes`):

- `Convergence` records the steps and the final bracket width, or for the SAC the outer iterations
  and the last change in (ρ, λ).
- Stopping at the limit adds the standard non-convergence note.
- An estimate on the interval bound adds "rho on interval bound (spatialreg: results should not be
  used)", or the same for lambda.

## Standard errors

From the observed information of the full log-likelihood at the MLE:

- SAR and SDM: analytic, in (β, ρ, σ²).
- SEM: `sqrt(s2 diag((X*'X*)^-1))` for β (errorsarlm's block-diagonal form); the (λ, σ²) block for λ.
- SAC: central differences of the analytic gradient, symmetrised (steps 1e-5 of each
  parameter's own size, so that they follow the units of y).

An information matrix that is not positive definite at the estimate is `EconError::Computation`; the
message says when the estimate is on the interval bound. That is what a complete-graph W gives
(k = n - 1): the SAR likelihood grows without limit towards the lower end of the interval.

## Impacts

With S_k = (I - ρW)^-1 (β_k I + θ_k W), where θ_k = 0 for the SAR: direct = mean(diag(S_k)),
total = mean(rowSums(S_k)), indirect = total - direct (`lagImpactsExact`, `mixedImpactsExact`).

The traces come from the cached eigenvalues. The total has a closed form when the rows of W have one
sum, and otherwise takes a linear solve per term. When min |1 - ρλ| is at most 1e-10 (where R's
`solve()` stops), or the solve fails, the impacts are NaN with a message in `SpatialImpacts.warnings`.

## Errors

| Input | Error |
|---|---|
| n no larger than the parameter count (k + spatial parameters + σ²) | `InsufficientData` |
| Collinear design ("There are aliased coefficients in the model") | `PerfectMulticollinearity`, naming the columns |
| SAC `grid_resolution` below 2 | `InvalidSpecification` |
| Dataset rows differ from the units of W | `InvalidSpecification` |

## Reference values

Case `spatial_sar_sdm_sem`, dataset `validation/datasets/generated/spatial_sim.csv` (n = 80, 4 nearest
neighbours on (cx, cy), row-standardised). The R maximiser is polished on the analytic score.

| Model | ρ or λ | logLik | Observed-information SE of ρ or λ |
|---|---|---|---|
| SAR | ρ 0.5949166262 | -85.8884260108 | 0.081813 |
| SDM | ρ 0.5216218948 | -85.3568616765 | 0.114182 |
| SEM | λ 0.6490127555 | -90.1532160317 | 0.094025 |
| SAC | ρ 0.6552859239, λ -0.2092131616 | -85.5900773491 | ρ 0.096191, λ 0.263641 |

The observed-information SEs are checked inside the case against `stats::optimHess`.

SDM impacts (x1, x2): direct 0.940402, -0.480573; indirect 1.359341, -0.645566; total 2.299743,
-1.126139.

## Tolerances

From `validation/README.md` at n = 80:

- Golden-section quantities (ρ, λ, coefficients, impacts, SAR, SDM and SEM SEs): `Tol::iterative(1e-6)`.
  The flatness bound of the concentrated likelihood is 1.3e-7 in ρ.
- SAR residuals: 1e-6 absolute (1.3e-7 times max |de/dρ| = 2.24).
- σ², logLik, AIC, BIC and the SEM coefficient SE, z and p: the statistic, SE and p-value tolerances.
- SAC by coordinate descent: `Tol::iterative(1e-5)` at tol 1e-10, 1e-4 for ρ and λ at the default tol.

## Known differences

- **SE convention.** Observed information at the MLE; spatialreg's method "eigen" reports the expected
  information (`sar_se` in the case), which differs at finite n and is not compared.
- **SEM residuals.** p2a reports u = y - Xβ; spatialreg reports e = (I - λW)u.
- **SDM on styles other than W.** spatialreg's `create_WX` adds `lag.(Intercept)`; p2a does not. This
  comes from the spatialreg source and could not be run here.

## Tests

In `econometrics::spatial::tests`:

- R-pinned: `test_validate_sar_sdm_sem_concentrated_ml`, `test_validate_sac_concentrated_ml`,
  `test_validate_sdm_impacts`, `test_validate_spatial_ml_se_observed_information`.
- Properties: `test_spatial_ml_se_scale_with_y`, `test_sac_rho_and_lambda_se_differ`,
  `test_spatial_ml_refuses_samples_no_larger_than_the_parameter_count`,
  `test_spatial_ml_refuses_aliased_regressors`, `test_sac_refuses_a_grid_below_two`,
  `test_spatial_ml_reports_convergence`, `test_spatial_ml_invariant_to_the_scale_of_y`,
  `test_sar_search_covers_the_spatialreg_interval`, `test_golden_section_flags_a_bound_optimum`,
  `test_spatial_ml_refuses_an_unbounded_search_interval`, `test_impacts_report_a_singular_multiplier`.

The paper's Makefile filters do not select spatial tests.
