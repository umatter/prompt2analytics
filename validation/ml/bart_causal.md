# BART Causal Inference Validation

## Method Overview

BART-based Causal Inference for Heterogeneous Treatment Effects, implementing a simplified frequentist approximation to the R package `bartCause`.

**Rust implementation**: `p2a_core::ml::bart_causal`

**R reference**: `bartCause::bartc()`

## Methodology

### Estimation Approach (T-Learner)

The implementation uses the T-learner approach:

1. **Fit response surface for treated**: mu_1(x) = E[Y | W=1, X=x]
2. **Fit response surface for control**: mu_0(x) = E[Y | W=0, X=x]
3. **Estimate CATE**: tau(x) = mu_1(x) - mu_0(x)

Each response surface is fitted using an ensemble of shallow regression trees (similar to BART's sum-of-trees model, but without MCMC).

### Uncertainty Quantification

Since we cannot do full Bayesian MCMC, uncertainty is quantified via bootstrap:

1. Resample data with replacement B times
2. Re-estimate CATE for each bootstrap sample, with forests of `max(1, n_trees / 4)` trees per response surface
3. Compute confidence intervals from bootstrap distribution: percentiles by R's `quantile(type = 7)`, which the CATE lines of the summary use too

The quarter-size bootstrap forests add their own Monte Carlo variance to the replicates, so the standard errors and
intervals are conservative (never narrower): about 1% of the standard error at the default 200 trees, more with few
trees. A replicate with fewer than 5 treated or 5 control rows is skipped; fewer than 2 usable replicates is an
`InsufficientData` error. A standard error of 0 (a constant outcome) reports `t` and `p` as NaN.

### Propensity score

With `include_propensity`, a tree ensemble on the treatment indicator gives each row a propensity score, clipped to
[0.01, 0.99], used as one more covariate. It is estimated once and held fixed across the bootstrap replicates
(bartCause's `p.scoreAsCovariate`). The fitted propensity model is kept with the result, and
`bart_causal_predict_arrays` scores new rows with it (they are passed without the propensity column), so the
training rows reproduce the in-sample CATEs exactly.

### Variable importance

Permutation importance of the estimated CATE: for each covariate (the propensity score included), the mean over
the rows of the squared change of `tau(x) = mu_1(x) - mu_0(x)` when that covariate's column is permuted, reported as
shares that sum to 1 (all zeros when no permutation changes the CATE). It ranks the covariates the effect varies
with, not those that predict the outcome; the permutations use their own random stream, so the estimates do not
depend on them. Only the trees that split on the permuted covariate can change, and in such a tree a row's
prediction changes only where a split on that covariate sends the permuted value the other way, so the pass walks
each row through those trees once (`bart_causal_importance_matches_the_full_permutation_pass` checks it against
permuting the whole column and predicting both ensembles everywhere).

### Differences from Full BART

| Feature | p2a-core Implementation | bartCause (R) |
|---------|-------------------------|---------------|
| Uncertainty | Bootstrap CI | Posterior credible intervals |
| Trees | Random forest ensemble | BART (sum of trees with MCMC) |
| Estimation | Frequentist | Bayesian |
| Speed | Fast | Slower (MCMC) |
| Regularization | Tree depth limit | BART priors |

## Test Cases

### Test Case 1: Synthetic Data with Known Treatment Effect

**Data Generation** (Rust):
```rust
// True DGP:
// Y = 5 + 2*X0 + 0.5*X1 + tau(X)*W + noise
// tau(X) = 1 + 2*X0 (heterogeneous treatment effect)
// P(W=1|X) = 0.3 + 0.4*X0 (confounding)

fn generate_test_data(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let mut rng = seed;

    let mut x = Array2::zeros((n, 3));
    let mut y = Array1::zeros(n);
    let mut w = Array1::zeros(n);

    for i in 0..n {
        x[[i, 0]] = (lcg_random(&mut rng) % 100) as f64 / 100.0;
        x[[i, 1]] = (lcg_random(&mut rng) % 100) as f64 / 100.0;
        x[[i, 2]] = (lcg_random(&mut rng) % 100) as f64 / 100.0;

        let p_treat = 0.3 + 0.4 * x[[i, 0]];
        w[i] = if (lcg_random(&mut rng) % 100) as f64 / 100.0 < p_treat {
            1.0
        } else {
            0.0
        };

        let tau = 1.0 + 2.0 * x[[i, 0]];
        let noise = ((lcg_random(&mut rng) % 100) as f64 - 50.0) / 50.0;
        y[i] = 5.0 + 2.0 * x[[i, 0]] + 0.5 * x[[i, 1]] + tau * w[i] + noise;
    }

    (y, w, x)
}
```

**Expected Results**:
- True ATE = E[tau(X)] = E[1 + 2*X0] = 1 + 2*0.5 = 2.0
- CATE should be positively correlated with X0
- Variable importance should rank X0 high (X0 is the effect modifier here)

**Rust Test**:
```rust
#[test]
fn test_bart_causal_basic() {
    let (y, w, x) = generate_test_data(200, 42);

    let config = BartCausalConfig {
        n_trees: 50,
        max_depth: 3,
        n_bootstrap: 20,
        seed: Some(42),
        ..Default::default()
    };

    let result = bart_causal_arrays(
        y.view(), w.view(), x.view(),
        vec!["x0".to_string(), "x1".to_string(), "x2".to_string()],
        config,
    ).unwrap();

    // ATE should be positive (true ATE is ~2)
    assert!(result.ate > 0.5);
    assert!(result.ate < 5.0);
}
```

### Test Case 2: Treatment Effect Heterogeneity Detection

**Objective**: Verify that CATE estimates vary with the true effect modifier (X0).

**Expected**: Positive correlation between estimated CATE and X0.

**Rust Test**:
```rust
#[test]
fn test_bart_causal_heterogeneity() {
    let (y, w, x) = generate_test_data(300, 456);

    let config = BartCausalConfig {
        n_trees: 100,
        max_depth: 4,
        n_bootstrap: 30,
        seed: Some(456),
        ..Default::default()
    };

    let result = bart_causal_arrays(
        y.view(), w.view(), x.view(),
        vec!["x0".to_string(), "x1".to_string(), "x2".to_string()],
        config,
    ).unwrap();

    // Compute correlation between CATE and X0
    let corr = compute_correlation(&result.cate, &x.column(0).to_vec());

    // Should be positively correlated
    assert!(corr > 0.0);
}
```

## Comparison with R bartCause

### R Reference Code

```r
library(bartCause)

# Generate similar data
set.seed(42)
n <- 200
x0 <- runif(n)
x1 <- runif(n)
x2 <- runif(n)
p_treat <- 0.3 + 0.4 * x0
w <- rbinom(n, 1, p_treat)
tau <- 1 + 2 * x0
y <- 5 + 2*x0 + 0.5*x1 + tau * w + rnorm(n, sd=0.5)

# Fit BART causal model
fit <- bartc(y, w, cbind(x0, x1, x2),
             method.rsp = "bart",
             n.samples = 1000,
             n.burn = 500,
             n.trees = 200)

# Results
summary(fit)
# ATE estimate and credible interval
extract(fit, "ate")
# CATE estimates
cate <- fitted(fit)
```

### Expected Tolerance

Due to the methodological differences (bootstrap vs MCMC), we expect:

| Metric | Tolerance |
|--------|-----------|
| ATE | Within 50% of true value |
| CATE correlation with X0 | > 0 (positive) |
| CI coverage | Approximately 95% |

Note: We do not expect exact numerical agreement with bartCause because:
1. Our implementation uses frequentist bootstrap, not Bayesian MCMC
2. Tree ensemble methodology differs from BART's sum-of-trees
3. Different regularization approaches

## Validation Tests

Not R-pinned: `bartCause` is a different model (see the table above). Regression and probit BART are in
[bart.md](bart.md).

### Unit Tests (Rust)

Location: `crates/p2a-core/src/ml/bart_causal.rs` (module tests)

| Test | Description | Status |
|------|-------------|--------|
| `test_bart_causal_basic` | Basic estimation with synthetic data | Pass |
| `test_bart_causal_with_propensity` | Estimation with propensity adjustment | Pass |
| `test_bart_causal_heterogeneity` | CATE varies with effect modifier | Pass |
| `test_bart_causal_insufficient_data` | Error handling for small samples | Pass |
| `test_bootstrap_se` | Bootstrap SE calculation | Pass |
| `test_percentile_ci` | Percentile CI calculation against `quantile(0:99, c(0.025, 0.975))` = 2.475, 96.525 | Pass |
| `test_tree_ensemble_fit_predict` | Tree ensemble mechanics | Pass |
| `test_display` | Display trait implementation | Pass |
| `bart_causal_with_one_to_three_trees_has_a_finite_se` | The bootstrap forests have at least one tree | Pass |
| `bart_causal_refuses_when_fewer_than_two_replicates_fit` | Fewer than 2 usable replicates is an error | Pass |
| `bart_causal_reports_nan_t_and_p_for_a_zero_se` | A zero standard error gives NaN t and p | Pass |
| `memory_bound_counts_the_bootstrap_floor` | The memory bound counts the bootstrap's tree floor | Pass |
| `bart_causal_predict_repeats_the_in_sample_cate_with_propensity` | Prediction scores the propensity with the fitted model | Pass |
| `bart_causal_predict_arrays_checks_the_column_count` | Prediction checks the covariate count | Pass |
| `bart_causal_importance_ranks_the_effect_modifier_first` | Importance ranks the effect modifier above a prognostic covariate | Pass |
| `bart_causal_importance_is_zero_for_a_constant_cate` | A constant CATE has zero importance | Pass |
| `bart_causal_importance_matches_the_full_permutation_pass` | The tree-restricted pass equals the full permutation pass | Pass |
| `bart_causal_outputs_unchanged_by_the_sweep` | Golden: estimates unchanged by the importance pass | Pass |

### Integration Tests

To run validation tests:

```bash
cargo test -p p2a-core bart_causal
```

## Wall-clock

`release`, `ci` features, base `f80b0131` against this tree, measured as for [bart.md](bart.md) (one core,
`RAYON_NUM_THREADS=1`, base and this tree interleaved with the order alternating, 2026-11-14, load average 4-42 from
the other lanes' builds). Defaults (200 trees, depth 4, 100 bootstrap replicates); n x p uniform design, a 0/1
treatment with probability 0.5, y = 2 x1 + 0.5 x2 + (1 + x2) w + noise. The change is mostly the CATE
permutation-importance pass. Five rounds at p = 5, nine at p = 20.

| p | n | base (ms) | this tree (ms) | ratio | min ratio |
|---:|---:|---:|---:|---:|---:|
| 5 | 100 | 314.3 | 338.5 | 1.08 | 1.04 |
| 5 | 1,000 | 4,948.9 | 5,153.7 | 1.04 | 1.05 |
| 5 | 10,000 | 56,301.2 | 57,736.0 | 1.03 | 1.09 |
| 20 | 100 | 300.6 | 311.6 | 1.04 | 1.04 |
| 20 | 1,000 | 4,779.4 | 5,044.1 | 1.06 | 1.07 |

A first version of the pass predicted both ensembles at every row for every permuted covariate; at p = 20 it took
1.12x the base at n = 100 (nine rounds, min ratio 1.12). A cap on the rows it uses would not have helped there,
since every row is used below 2,000. The pass now walks only the trees that split on the permuted covariate and
leaves a row's path only where a split on it sends the permuted value the other way, which gives the same
importance (to rounding) at a fraction of the traversals. Every row is within 1.1x, so the pass uses every row and
there is no row cap.

Peak RSS (`/usr/bin/time -v`, one fit per process), defaults at n = 10,000, p = 5: base 14,744 kB, this tree
14,828 kB (1.01x). The importance pass holds one more vector of n (the permuted column), which `memory_bound`
counts.

## API Reference

### Function Signature

```rust
pub fn bart_causal(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: BartCausalConfig,
) -> EconResult<BartCausalResult>
```

### Configuration

```rust
pub struct BartCausalConfig {
    pub n_trees: usize,           // Default: 200
    pub max_depth: usize,         // Default: 4
    pub min_node_size: usize,     // Default: 5
    pub n_bootstrap: usize,       // Default: 100
    pub include_propensity: bool, // Default: false
    pub confidence_level: f64,    // Default: 0.95
    pub seed: Option<u64>,
    pub sample_fraction: f64,     // Default: 0.632
    pub mtry: Option<usize>,
}
```

### Result Structure

```rust
pub struct BartCausalResult {
    pub ate: f64,
    pub ate_se: f64,
    pub ate_t_stat: f64,
    pub ate_p_value: f64,
    pub ate_ci_lower: f64,
    pub ate_ci_upper: f64,
    pub ate_significance: SignificanceLevel,
    pub cate: Vec<f64>,
    pub cate_lower: Vec<f64>,
    pub cate_upper: Vec<f64>,
    pub cate_se: Vec<f64>,
    pub y1_pred: Vec<f64>,
    pub y0_pred: Vec<f64>,
    pub variable_importance: Vec<(String, f64)>,
    // ... additional fields
}
```

## MCP Tool

### Tool Name

`ml_bart_causal`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| dataset | string | Yes | Dataset name |
| outcome | string | Yes | Outcome variable |
| treatment | string | Yes | Binary treatment (0/1) |
| covariates | [string] | Yes | Covariate columns |
| n_trees | int | No | Trees per ensemble (default: 200) |
| max_depth | int | No | Max tree depth (default: 4) |
| n_bootstrap | int | No | Bootstrap samples (default: 100) |
| include_propensity | bool | No | Add propensity (default: false) |
| seed | int | No | Random seed |

### Example Usage

```json
{
  "tool": "ml_bart_causal",
  "arguments": {
    "dataset": "experiment_data",
    "outcome": "revenue",
    "treatment": "treated",
    "covariates": ["age", "income", "region"],
    "n_trees": 200,
    "n_bootstrap": 100
  }
}
```

## References

- Hill, J. L. (2011). Bayesian Nonparametric Modeling for Causal Inference. *Journal of Computational and Graphical Statistics*, 20(1), 217-240.
- Chipman, H. A., George, E. I., & McCulloch, R. E. (2010). BART: Bayesian Additive Regression Trees. *Annals of Applied Statistics*, 4(1), 266-298.
- Hahn, P. R., Murray, J. S., & Carvalho, C. M. (2020). Bayesian Regression Tree Models for Causal Inference. *Bayesian Analysis*, 15(3), 965-1056.
- Kunzel, S. R., et al. (2019). Metalearners for Estimating Heterogeneous Treatment Effects. *PNAS*, 116(10), 4156-4165.
- R package `bartCause`: Dorie, V. (2020). https://CRAN.R-project.org/package=bartCause
