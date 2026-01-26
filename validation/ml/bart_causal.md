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
2. Re-estimate CATE for each bootstrap sample
3. Compute confidence intervals from bootstrap distribution (percentile method)

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
- Variable importance should rank X0 high

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

### Unit Tests (Rust)

Location: `crates/p2a-core/src/ml/bart_causal.rs` (module tests)

| Test | Description | Status |
|------|-------------|--------|
| `test_bart_causal_basic` | Basic estimation with synthetic data | Pass |
| `test_bart_causal_with_propensity` | Estimation with propensity adjustment | Pass |
| `test_bart_causal_heterogeneity` | CATE varies with effect modifier | Pass |
| `test_bart_causal_insufficient_data` | Error handling for small samples | Pass |
| `test_bootstrap_se` | Bootstrap SE calculation | Pass |
| `test_percentile_ci` | Percentile CI calculation | Pass |
| `test_tree_ensemble_fit_predict` | Tree ensemble mechanics | Pass |
| `test_display` | Display trait implementation | Pass |

### Integration Tests

To run validation tests:

```bash
cargo test -p p2a-core bart_causal
```

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
