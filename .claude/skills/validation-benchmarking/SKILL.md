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
| Panel FE, HDFE | plm, fixest | linearmodels | `plm()`, `feols()` / `PanelOLS()` |
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

### Step 2: Create the R Reference Case

A method is validated by an R reference case that a Rust test loads, not by values copied into
a test. See `validation/README.md` ("Adding an R-pinned validation") for the full rules.

**Test data**:
- Committed data in `validation/datasets/*.csv` (`grunfeld.csv` for panels), read with
  `ctx$dataset("datasets/grunfeld.csv")`.
- R's own datasets and simulated data go through a seeded `ctx$gen_dataset()` in the case,
  which writes `validation/datasets/generated/<name>.csv` and reads it back
  (e.g. `longley` is `validation/datasets/generated/longley.csv`, written by case `ols_longley`).
- Data must have noise; at least one real/standard dataset and, where useful, a simulated one
  that exercises the property the method must get right.

**R case template**: `.claude/tooling/validation/templates/r_validation.R.template`. Copy it into
`validation/r/cases/<area>.R`, then:

```bash
Rscript validation/r/run.R --write <case_id>
Rscript validation/r/run.R --check <case_id>
```

### Step 3: Document Results

**Create validation document at**: `validation/[category]/[method].md`

**Required Sections**:
1. **Method Overview** - Brief description
2. **Reference Implementations** - Table of packages, with versions
3. **Test Cases** - With R code and R's values
4. **R reference** - `validation/r/cases/<file>.R`, case `<id>`
5. **Rust tests** - libtest paths and what each checks
6. **Tolerance** - with the `Tol` used
7. **Status** - R-pinned (registry `validation/methods/<area>.json`, id `<id>`) or not R-pinned
   (with the finding that adds the case)
8. **Known Differences** - each with its finding id
9. **References** - Academic citations

No `~` values in comparison tables and no hand-written `#[test]` code: cite the tests that exist.

### Step 4: Add the Rust Validation Test

**Naming Convention**: `test_validate_<method>_<dataset>[_<variant>]`

**Test Location**: Same file as implementation, in `#[cfg(test)]` module

**Template**:
```rust
#[test]
fn test_validate_ols_grunfeld() {
    use p2a_validation::{Check, Tol};
    let (case, ds) = crate::test_support::load_case("ols_grunfeld");
    let r = run_ols(&ds, "inv", &["value", "capital"], true, CovarianceType::HC1).unwrap();
    let n = r.n_obs;
    let names = r.variable_names.as_slice();
    let coef: Vec<f64> = r.coefficients.iter().map(|c| c.estimate).collect();
    let se: Vec<f64> = r.coefficients.iter().map(|c| c.std_error).collect();
    Check::new(&case)
        .named("coef", names, &coef, Tol::coef(n))
        .named("se_hc1", names, &se, Tol::se(n))
        .scalar("adj_r2", r.adj_r_squared, Tol::stat(n))
        .finish();
}
```

Pin every number the method reports for that call. p2a-cli and p2a-mcp tests use
`p2a_validation::Case::load`. No `if let Ok`, no print-only checks. A mismatch is a defect:
fix it, or record a registry deviation and leave the quantity out with a
`// deviation <finding id>` comment; never widen a tolerance to make it pass.

### Step 5: Register and Update Indexes

1. Add the registry entry in `validation/methods/<area>.json` (cases, tests, quantities,
   tolerance, deviations; schema in `validation/methods/README.md`).
2. Add or update the row in the `validation/README.md` method index. Its status follows the
   registry (`R-pinned`, `R-pinned, known deviations (...)` or `not R-pinned`), never "Complete".
3. If a new R package is used, add it to `validation/reference_implementations.md` and to the
   prerequisites in `validation/README.md`.
4. Run `python3 validation/tools/validation_status.py lint --strict --tests target/*-tests.txt`.
5. Adding, removing or renaming a `test_validate` test changes the counts CI checks:
   regenerate `validation/VALIDATION_STATUS.md` with `validation/run_validation.sh
   --write-status` before the change reaches main (during the review campaign: at the wave
   integration, not in a lane).

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

**R Benchmark Script** (`../prompt2analytics-paper/performance/comparisons/r_comparison/benchmark_[method].R`):
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

**Update**: `../prompt2analytics-paper/performance/reports/[category]_performance.md`

**Include**:
- Benchmark results table (mean, median, std)
- Scaling analysis (how time grows with n)
- Cross-language comparison chart
- Memory usage if applicable

---

## Tolerance Guidelines

R-pinned tests use `p2a_validation::Tol` (`crates/p2a-validation/src/check.rs`, the table in
`validation/README.md`):

| Quantity | `Tol` | n < 100 | 100 ≤ n ≤ 1000 | n > 1000 | abs |
|---|---|---|---|---|---|
| coefficients | `Tol::coef(n)` | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
| standard errors | `Tol::se(n)` | 1e-5 | 1e-6 | 1e-8 | 1e-14 |
| statistics, R², CI bounds | `Tol::stat(n)` | 1e-5 | 1e-6 | 1e-8 | 1e-14 |
| p-values | `Tol::p_value(n)` | 1e-5 | 1e-6 | 1e-8 | 1e-12 |
| iterative (MAP, MLE) | `Tol::iterative(rel)` | documented per test | | | 1e-12 |
| CLI text, d decimals | `Tol::printed(d)` | | | | 0.5·10^-d + 1e-12 |
| counts, df | `Tol::exact()` | exact | | | |

---

## Checklist

After completing validation and benchmarking, verify:

### Required (All items mandatory)
- [ ] R case written in `validation/r/cases/<area>.R`, and `Rscript validation/r/run.R --check` is clean
- [ ] `test_validate_*` test loads the case with `load_case` and compares with `Check`/`Tol`
- [ ] Test registered in `validation/methods/<area>.json`
- [ ] `python3 validation/tools/validation_status.py lint --strict --tests target/*-tests.txt` is clean
- [ ] `validation/run_validation.sh --rust-only --category <cat>` passes
- [ ] `validation/[category]/[method].md` created or updated, and the `validation/README.md` index row follows the registry

### Performance (All items mandatory for new methods)
- [ ] Criterion benchmark added to `crates/p2a-core/benches/`
- [ ] R benchmark script created in `../prompt2analytics-paper/performance/comparisons/r_comparison/`
- [ ] **Rust benchmarks executed** (`cargo bench -p p2a-core -- [method]`)
- [ ] **R benchmarks executed** (`Rscript benchmark_[method].R`)
- [ ] Performance table in validation doc filled with **actual numbers**
- [ ] Speedup factors calculated and documented

---

## Quick Commands

```bash
# Write and check an R reference case
Rscript validation/r/run.R --write <case_id>
Rscript validation/r/run.R --check

# Run the validation tests of one category (CI feature sets)
validation/run_validation.sh --rust-only --category <cat>

# Run validation tests for a specific method, printing every comparison
P2A_VALIDATION_VERBOSE=1 cargo test -p p2a-core -- test_validate_[method] --nocapture

# Registry, case and doc lint (needs the cargo --list outputs in target/*-tests.txt)
python3 validation/tools/validation_status.py lint --strict --tests target/*-tests.txt

# Run Rust benchmarks for a specific method (MANDATORY)
cargo bench -p p2a-core -- [method_name]

# Run all benchmarks
cargo bench -p p2a-core

# View HTML benchmark report
open target/criterion/report/index.html

# Run R benchmarks for a specific method (MANDATORY)
Rscript ../prompt2analytics-paper/performance/comparisons/r_comparison/benchmark_[method].R

# Fallback if microbenchmark not available - use system.time() in R
```

## Phase 7 Execution Template

When implementing a new method, execute these commands in order:

```bash
# 1. Check the R case and run the Rust validation tests
Rscript validation/r/run.R --check
validation/run_validation.sh --rust-only --category <cat>

# 2. Run Rust performance benchmarks
cargo bench -p p2a-core -- [method]

# 3. Run R performance benchmarks
Rscript ../prompt2analytics-paper/performance/comparisons/r_comparison/benchmark_[method].R

# 4. Update validation document with actual results
# Edit: validation/[category]/[method].md
# Fill in the Performance Comparison table with actual numbers
```

---

## References

- Criterion documentation: https://bheisler.github.io/criterion.rs/book/
- R microbenchmark package: https://cran.r-project.org/package=microbenchmark
- Python timeit module: https://docs.python.org/3/library/timeit.html
