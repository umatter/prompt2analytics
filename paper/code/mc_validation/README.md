# Monte Carlo Validation Pipeline

Standalone evaluation pipeline that validates the statistical properties of
p2a-core methods using Monte Carlo simulation. This is a paper-level evaluation
artifact — it is not part of the core library, CI, or application code.

## What it tests

| Property | Description | Example |
|----------|-------------|---------|
| Type I error | Rejection rate under H0 ≈ α | t-test rejects ~5% when H0 is true |
| CI coverage | 95% CI contains true θ ~95% of runs | OLS β̂ CI covers true β |
| Estimator bias | E[θ̂] ≈ θ | OLS is unbiased for β |
| SE accuracy | Mean reported SE ≈ empirical SD(θ̂) | Robust SEs are well-calibrated |
| Power | Rejection rate increases with effect size | t-test power curve |

## Method categories

- **Regression**: OLS, robust SEs (HC0–HC3), HAC, clustered, GLS, quantile
- **Panel**: FE, RE, HDFE, Hausman, Arellano-Bond
- **Discrete**: Logit, probit, Poisson, ordered logit, multinomial
- **Causal**: IV/2SLS, DiD, RD, matching, IPW, TMLE, DoubleML
- **Hypothesis tests**: t-test, ANOVA, chi-squared, Wilcoxon, KS, Shapiro-Wilk, ...
- **Time series**: ARIMA, VAR, Granger causality
- **Survival**: Cox PH, Kaplan-Meier, log-rank

## Running

```bash
cd paper/code/mc_validation

# Full pipeline (all methods, 1000 simulations each)
cargo run --release

# Quick check (100 simulations)
cargo run --release -- --sims 100

# Specific category only
cargo run --release -- --category regression
cargo run --release -- --category hypothesis

# Results are written to results/mc_validation_<timestamp>.json
```

## Output

Each method produces:

```json
{
  "method": "OLS",
  "property": "ci_coverage",
  "dgp": "homoskedastic",
  "n": 1000,
  "n_sims": 1000,
  "observed": 0.948,
  "expected": 0.95,
  "within_tolerance": true,
  "tolerance_ci": [0.936, 0.964]
}
```

Tolerance bounds use a binomial confidence interval: for 1000 simulations at
nominal 5%, the 95% acceptance region for the rejection rate is [3.6%, 6.4%].

## Design principles

1. **Known DGPs**: Every simulation uses a data-generating process with known
   true parameters, so we can check whether estimators recover them.
2. **Binomial tolerance**: MC tests use proper statistical tolerance — we don't
   demand exact 5% rejection, we check that the observed rate falls within the
   expected sampling variability.
3. **Reproducible**: All simulations use sequential seeds derived from a master
   seed for full reproducibility.
4. **Self-contained**: No external dependencies beyond p2a-core. No R required.
