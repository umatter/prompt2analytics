# Validation: Bootstrap Covariance (vcovBS)

## Method Overview

Bootstrap covariance estimation provides an alternative to asymptotic variance estimation when the standard assumptions may not hold. The bootstrap creates many resampled datasets and estimates the model on each, then uses the empirical distribution of coefficients to estimate the covariance matrix.

**Key Parameters:**
- `n_boot`: Number of bootstrap replications (default: 999); at least 10, fewer is an `InvalidSpecification` error (Tier 0, `c3e808e`)
- `bootstrap_type`: Method for resampling

**Bootstrap Types:**
1. **Pairs (xy) bootstrap**: Resamples (y_i, x_i) pairs together. Most robust to misspecification.
2. **Residual bootstrap**: Resamples residuals keeping X fixed. More efficient under correct specification.
3. **Wild bootstrap**: Multiplies residuals by random weights (Rademacher ±1). Robust to heteroskedasticity.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| sandwich | R | `vcovBS()` | 3.1.3 (legacy script only) |
| statsmodels | Python | `OLSResults.get_robustcov_results()` | N/A |

## Validation

Not R-pinned: R draws with Mersenne Twister, p2a with ChaCha8Rng, so no seed matches.
`sandwich::vcovBS` cannot be reproduced from another RNG, and no R reference case exists for
this method. The legacy script `validation/scripts/validate_regression_diag.R` (Test 9) writes
`vcovBS(lm(y ~ x), type = "xy" | "wild", R = 200)` on its own simulated data to
`validation/expected/bootstrap_test.csv`; that file is not asserted by any test.

Property tests (`crates/p2a-core/src/regression/ols.rs`):

- `test_bootstrap_pairs`, `test_bootstrap_residual`, `test_bootstrap_wild`: each resampling
  scheme returns positive standard errors and reports its type
- `test_bootstrap_vcov_positive_semidefinite`: the diagonal of the covariance matrix is
  non-negative and equals the squared standard errors
- `test_bootstrap_type_parsing`, `test_bootstrap_displays_correctly`
- `test_bootstrap_pairs_and_wild_on_heteroskedastic_data`: pairs and wild standard errors on
  heteroskedastic data are within a factor of 2 of each other, and more than 90% of the
  replications converge
- `test_pinned_hac_dk_bootstrap_outputs` (Tier 0, `df73f11`): characterisation pin of the
  seeded output, so a change to the resampling code is visible
- `test_bootstrap_rejects_too_few_replications`,
  `test_bootstrap_impossible_replication_count_is_an_error`,
  `test_bootstrap_replicate_storage_respects_alloc_ceiling` and
  `test_bootstrap_stops_when_cancelled` (Tier 0, `c3e808e`): the `n_boot >= 10` bound, the
  replicate-storage ceiling and cancellation

## Known Differences

1. **Random number generation**: Rust uses ChaCha8Rng, R uses Mersenne Twister. Results with same seed will differ.
2. **Wild bootstrap weights**: R may use different weight distributions (Rademacher, Mammen). Rust uses Rademacher.
3. **Handling of singularities**: Rust skips singular bootstrap samples; R may fail entirely.

## References

- Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife."
  *Annals of Statistics*, 7(1), 1-26.
- Wu, C. F. J. (1986). "Jackknife, Bootstrap and Other Resampling Methods
  in Regression Analysis." *Annals of Statistics*, 14(4), 1261-1295.
- MacKinnon, J. G. (2006). "Bootstrap Methods in Econometrics."
  *Economic Record*, 82, S2-S18.
- Zeileis, A. (2004). "Econometric Computing with HC and HAC Covariance
  Matrix Estimators." *Journal of Statistical Software*, 11(10), 1-17.

## Implementation Notes

The Rust implementation:
1. Runs standard OLS to get original coefficients, fitted values, and residuals
2. For each bootstrap replication:
   - Pairs: Resample indices, build new (y*, X*) with replacement
   - Residual: y* = Xβ̂ + ε*, where ε* sampled from residuals
   - Wild: y* = Xβ̂ + w*ε̂, where w = ±1 (Rademacher)
3. Estimate OLS on bootstrap sample
4. Collect coefficient vectors
5. Compute sample covariance: Cov(β̂*) = (1/(B-1)) Σ(β̂*ᵢ - β̄*)(β̂*ᵢ - β̄*)'
