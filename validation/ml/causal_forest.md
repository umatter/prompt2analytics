# Causal Forest Validation

## Method Description

Causal forests estimate heterogeneous treatment effects (CATE) using random forests adapted for causal inference. The method is based on Wager & Athey (2018).

Key features:
- **Honest splitting**: Uses separate data for determining tree structure vs. estimation
- **Causal splitting criterion**: Maximizes variance of treatment effects across child nodes
- **Bootstrap variance estimation**: Uncertainty quantification via across-tree variance

## Implementation Details

**File**: `crates/p2a-core/src/ml/causal_forest.rs`

### Core Algorithm

1. For each tree:
   a. Draw a subsample from the data
   b. If honest splitting: split subsample into structure and estimation samples
   c. Build tree structure using structure sample (or full subsample if not honest)
   d. Estimate treatment effects in leaves using estimation sample
   e. Store out-of-bag predictions

2. For predictions:
   a. Average CATE estimates across all trees
   b. Compute variance from bootstrap (across-tree) variation

3. For ATE:
   a. Average CATE across all observations
   b. Standard error combines bootstrap variance and sampling variance

### Key Functions

```rust
pub fn causal_forest(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: CausalForestConfig,
) -> EconResult<CausalForestResult>
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_trees` | 2000 | Number of trees |
| `min_node_size` | 5 | Minimum observations per leaf |
| `max_depth` | 10 | Maximum tree depth |
| `honesty` | true | Use honest splitting |
| `honesty_fraction` | 0.5 | Fraction for estimation |
| `sample_fraction` | 0.5 | Subsample fraction per tree |
| `mtry` | sqrt(p) | Variables per split |

## Reference Implementation

**R package**: `grf` (Generalized Random Forests)

```r
library(grf)

# Generate data
n <- 500
p <- 5
X <- matrix(runif(n * p), n, p)
W <- rbinom(n, 1, 0.5)
tau <- 1 + 2 * X[, 1]  # True CATE
Y <- 5 + X[, 1] + 0.5 * X[, 2] + tau * W + rnorm(n)

# Fit causal forest
cf <- causal_forest(X, Y, W,
                    num.trees = 2000,
                    min.node.size = 5,
                    honesty = TRUE)

# Get predictions
predictions <- predict(cf)$predictions
ate <- average_treatment_effect(cf)
```

## Validation Test Cases

### Test Case 1: Basic Functionality (n=200)

**Data Generation**:
```rust
// CATE: tau(x) = 1 + 2*x0
// Outcome: y = 5 + x0 + 0.5*x1 + tau*w + noise
```

**Expected Results**:
- ATE should be approximately 2 (1 + 2*E[x0] where E[x0]=0.5)
- Variable x0 should have highest importance
- All predictions should be finite

### Test Case 2: Honest vs Non-Honest Splitting

Both configurations should produce finite results, with honest splitting generally providing better coverage.

### Test Case 3: Subset ATE

Verify that `average_treatment_effect()` correctly computes ATE for observation subsets.

## Tolerances

| Metric | Tolerance | Notes |
|--------|-----------|-------|
| ATE | +/- 0.5 | Due to bootstrap variability |
| Predictions | finite | No NaN or Inf values |
| Variable importance | >0 for true predictors | x0 should dominate |

## Comparison with R `grf`

Due to differences in:
- Random number generation
- Bootstrap sampling
- Tree construction details

We expect qualitative rather than exact numerical agreement. Key checks:
1. ATE sign and approximate magnitude match
2. Variable importance rankings match
3. CATE heterogeneity patterns similar

## References

- Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous
  Treatment Effects using Random Forests. *Journal of the American Statistical
  Association*, 113(523), 1228-1242.
  https://doi.org/10.1080/01621459.2017.1319839

- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests.
  *Annals of Statistics*, 47(2), 1148-1178.
  https://doi.org/10.1214/18-AOS1709

- R package `grf`: https://grf-labs.github.io/grf/

## Rust Test Functions

- `test_causal_forest_basic` - Basic functionality with heterogeneous effects
- `test_causal_forest_no_honesty` - Without honest splitting
- `test_causal_tree_honest_split` - Verifies honest sample splitting
- `test_treatment_effect_estimation` - Leaf-level treatment effect calculation
- `test_average_treatment_effect_subset` - Subset ATE computation
- `test_insufficient_data` - Error handling for small samples

## Status

| Component | Status |
|-----------|--------|
| Core implementation | Complete |
| Unit tests | Complete |
| MCP tool | Complete |
| Documentation | Complete |
| R comparison | Qualitative |
