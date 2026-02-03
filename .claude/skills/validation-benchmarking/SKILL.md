# Skill: Validation & Benchmarking

## Purpose

Document validation of prompt2analytics implementations against reference implementations (R, Python, Julia) and establish performance benchmarks. This skill supports the `implement_metrics` Phase 6 workflow.

---

## When to Use

- After implementing a new econometric/ML method
- When adding validation documentation for existing methods
- When running performance benchmarks
- When creating cross-language comparison tests

---

## Validation Workflow

### Step 1: Identify Reference Implementations

**Primary References by Domain**:

| Domain | R Package | Python Package | Function |
|--------|-----------|----------------|----------|
| Regression | stats | statsmodels | `lm()` / `OLS()` |
| Robust SEs | sandwich | statsmodels | `vcovHC()` / `get_robustcov_results()` |
| Clustered SEs | sandwich | linearmodels | `vcovCL()` / `fit(cov_type='clustered')` |
| Panel FE | plm, lfe | linearmodels | `plm()`, `felm()` / `PanelOLS()` |
| Panel RE | plm | linearmodels | `plm(model='random')` |
| IV/2SLS | AER | linearmodels | `ivreg()` / `IV2SLS()` |
| Logit/Probit | stats | statsmodels | `glm()` / `Logit()`, `Probit()` |
| VAR | vars | statsmodels | `VAR()` / `VAR()` |
| VECM | urca, vars | statsmodels | `ca.jo()`, `vec2var()` / `VECM()` |
| ARIMA | forecast | statsmodels | `auto.arima()` / `ARIMA()` |
| MSTL | forecast | N/A | `mstl()` |
| K-means | stats | sklearn | `kmeans()` / `KMeans()` |
| PCA | stats | sklearn | `prcomp()` / `PCA()` |
| Random Forest | randomForest | sklearn | `randomForest()` / `RandomForestClassifier()` |

### Step 2: Create Test Cases

**Minimum Requirements**:
- At least 2 test cases per method
- One synthetic data case (known DGP for coefficient verification)
- One real/standard dataset case (e.g., Grunfeld, Longley, iris)

**Test Data Sources**:
- `validation/datasets/grunfeld.csv` - Panel data
- `validation/datasets/longley.csv` - Collinear regression data
- R built-in datasets: `mtcars`, `iris`, `AirPassengers`

**R Code Template**:
```r
# Set seed for reproducibility
set.seed(42)

# Generate data
n <- 1000
x <- rnorm(n)
y <- 2 + 3*x + rnorm(n, 0, 0.5)
data <- data.frame(y = y, x = x)

# Fit model
fit <- lm(y ~ x, data = data)
summary(fit)

# Extract results
coef(fit)
sqrt(diag(vcov(fit)))  # Standard errors
```

### Step 3: Document Results

**Create validation document at**: `validation/[category]/[method].md`

**Required Sections**:
1. **Method Overview** - Brief description
2. **Reference Implementations** - Table of packages
3. **Test Cases** - With R/Python code
4. **Results Comparison** - Table with tolerances
5. **Numerical Precision Summary**
6. **Known Differences** - Any intentional deviations
7. **References** - Academic citations

**Results Comparison Table Format**:
```markdown
| Statistic | R | p2a Rust | Difference | Tolerance |
|-----------|---|----------|------------|-----------|
| β₀ | 2.0012 | 2.0012 | 1e-10 | 1e-8 |
| β₁ | 2.9987 | 2.9987 | 2e-11 | 1e-8 |
| SE(β₁) | 0.0158 | 0.0158 | 5e-12 | 1e-6 |
```

### Step 4: Add Rust Validation Tests

**Naming Convention**: `test_validate_[method]_[scenario]`

**Test Location**: Same file as implementation, in `#[cfg(test)]` module

**Template**:
```rust
#[test]
fn test_validate_ols_longley() {
    // Load reference data
    let dataset = Dataset::from_csv("validation/datasets/longley.csv").unwrap();

    // Run p2a implementation
    let result = run_ols(&dataset, "y", &["x1", "x2"], true, CovarianceType::Standard).unwrap();

    // Compare to R results (from validation/regression/ols.md)
    let expected_coef = vec![2.0012, 2.9987];
    for (i, (actual, expected)) in result.coefficients().iter().zip(expected_coef.iter()).enumerate() {
        assert!((actual - expected).abs() < 1e-4,
            "Coefficient {} differs: {} vs {}", i, actual, expected);
    }
}
```

### Step 5: Update Indexes

1. Add entry to `validation/README.md` method index
2. Mark status as "Complete" with link to validation doc
3. If new R/Python package used, add to `validation/reference_implementations.md`

---

## Benchmarking Workflow

### Step 1: Write Criterion Benchmark

**File Location**: `crates/p2a-core/benches/[category]_benchmarks.rs`

**Template**:
```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use p2a_core::dataset::Dataset;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn generate_data(n: usize, k: usize, rng: &mut ChaCha8Rng) -> Dataset {
    // Generate n observations with k predictors
    // Return Dataset
}

fn method_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("method_name");
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    for n in [100, 1000, 10000] {
        let dataset = generate_data(n, 5, &mut rng);

        group.bench_with_input(
            BenchmarkId::new("variant", n),
            &dataset,
            |b, d| b.iter(|| {
                // Call method being benchmarked
            })
        );
    }
    group.finish();
}

criterion_group!(benches, method_benchmark);
criterion_main!(benches);
```

### Step 2: Run Benchmarks

```bash
# Run all benchmarks
cargo bench -p p2a-core

# Run specific benchmark
cargo bench -p p2a-core -- method_name

# Save results with timestamp
cargo bench -p p2a-core -- --save-baseline $(date +%Y-%m-%d)
```

### Step 3: Cross-Language Comparison

**R Benchmark Script** (`performance/comparisons/r_comparison/benchmark_[method].R`):
```r
library(microbenchmark)
library(lfe)  # or relevant package

# Generate same data as Rust (same seed, same n)
set.seed(42)
n <- 10000
data <- data.frame(
    y = rnorm(n),
    x1 = rnorm(n),
    x2 = rnorm(n)
)

# Benchmark
result <- microbenchmark(
    lm(y ~ x1 + x2, data = data),
    times = 100
)

# Save results
write.csv(
    summary(result),
    "results/method_benchmark.csv"
)
```

### Step 4: Document Performance

**Update**: `performance/reports/[category]_performance.md`

**Include**:
- Benchmark results table (mean, median, std)
- Scaling analysis (how time grows with n)
- Cross-language comparison chart
- Memory usage if applicable

---

## Tolerance Guidelines

| Sample Size | Coefficient Tolerance | SE Tolerance | p-value Tolerance |
|-------------|----------------------|--------------|-------------------|
| n < 100 | 1e-6 | 1e-4 | 0.01 |
| n = 100-1000 | 1e-8 | 1e-6 | 0.001 |
| n > 1000 | 1e-10 | 1e-8 | 0.0001 |

**Note**: Iterative methods (HDFE, MLE) may have larger tolerances (1e-5).

---

## Checklist

After completing validation and benchmarking, verify:

### Required (All items mandatory)
- [ ] `validation/[category]/[method].md` created
- [ ] At least 2 test cases documented (synthetic + real)
- [ ] R/Python reproduction code included
- [ ] Results comparison table with explicit tolerances
- [ ] Rust `test_validate_*` tests pass
- [ ] `validation/README.md` index updated

### Performance (All items mandatory for new methods)
- [ ] Criterion benchmark added to `crates/p2a-core/benches/`
- [ ] R benchmark script created in `performance/comparisons/r_comparison/`
- [ ] **Rust benchmarks executed** (`cargo bench -p p2a-core -- [method]`)
- [ ] **R benchmarks executed** (`Rscript benchmark_[method].R`)
- [ ] Performance table in validation doc filled with **actual numbers**
- [ ] Speedup factors calculated and documented

---

## Quick Commands

```bash
# Run validation tests for a specific method
cargo test -p p2a-core -- test_validate_[method] --nocapture

# Run all validation tests
cargo test -p p2a-core -- test_validate --nocapture

# Run Rust benchmarks for a specific method (MANDATORY)
cargo bench -p p2a-core -- [method_name]

# Run all benchmarks
cargo bench -p p2a-core

# View HTML benchmark report
open target/criterion/report/index.html

# Run R benchmarks for a specific method (MANDATORY)
Rscript performance/comparisons/r_comparison/benchmark_[method].R

# Fallback if microbenchmark not available - use system.time() in R
```

## Phase 7 Execution Template

When implementing a new method, execute these commands in order:

```bash
# 1. Run Rust validation tests
cargo test -p p2a-core -- test_validate_[method] --nocapture

# 2. Run Rust performance benchmarks
cargo bench -p p2a-core -- [method]

# 3. Run R performance benchmarks
Rscript performance/comparisons/r_comparison/benchmark_[method].R

# 4. Update validation document with actual results
# Edit: validation/[category]/[method].md
# Fill in the Performance Comparison table with actual numbers
```

---

## References

- Criterion documentation: https://bheisler.github.io/criterion.rs/book/
- R microbenchmark package: https://cran.r-project.org/package=microbenchmark
- Python timeit module: https://docs.python.org/3/library/timeit.html
