# Validation Tooling for p2a-core

This directory contains tooling to support the `/validate-method` slash command, which provides unified R vs Rust validation and benchmarking.

## Directory Structure

```
.claude/tooling/validation/
├── README.md                      # This file
├── method_registry.json           # Central index of all methods and artifacts
└── templates/
    ├── r_benchmark.R.template     # Template for R benchmark scripts
    └── r_validation.R.template    # Template for R validation scripts
```

## Method Registry

The `method_registry.json` file maps each p2a-core method to its artifacts:

```json
{
  "methods": {
    "ols": {
      "display_name": "Ordinary Least Squares",
      "r_name": "lm",
      "r_packages": ["stats"],
      "category": "regression",
      "rust_module": "crates/p2a-core/src/regression/ols.rs",
      "rust_function": "run_ols",
      "r_benchmark_script": "performance/comparisons/r_comparison/benchmark_regression.R",
      "criterion_benchmark": "crates/p2a-core/benches/regression_benchmarks.rs",
      "criterion_group": "OLS_Standard",
      "validation_doc": "validation/regression/ols.md",
      "test_patterns": ["test_validate_ols", "test_validate_longley"]
    }
    // ... 70+ more methods
  }
}
```

### Registry Fields

| Field | Description |
|-------|-------------|
| `display_name` | Human-readable method name |
| `r_name` | R function(s) being compared |
| `r_packages` | Required R packages |
| `category` | Method category (regression, panel, causal, etc.) |
| `rust_module` | Path to Rust source file |
| `rust_function` | Rust function name(s) |
| `r_benchmark_script` | Path to R benchmark script (may not exist) |
| `criterion_benchmark` | Path to Criterion benchmark file |
| `criterion_group` | Criterion benchmark group name |
| `validation_doc` | Path to validation documentation |
| `test_patterns` | Rust test name patterns for validation |

### Categories

- **regression**: OLS, NLS, LOESS, GLS, smooth splines, diagnostics
- **panel**: Fixed effects, random effects, HDFE, Hausman test
- **causal**: IV/2SLS, DiD, synthetic control, RD, IPW, doubly robust
- **discrete**: Logit, probit
- **timeseries**: VAR, VECM, IRF, ACF/PACF, spectral analysis
- **survival**: Kaplan-Meier, Cox PH, AFT, competing risks
- **hypothesis**: t-tests, ANOVA, chi-squared, Fisher, Wilcoxon, etc.
- **multivariate**: MANOVA, factor analysis, canonical correlation
- **correlation**: Correlation tests
- **power**: Power analysis for t-tests, proportions, ANOVA
- **forecasting**: ARIMA, Holt-Winters, STL, MSTL, Kalman filter
- **ml**: K-means, DBSCAN, hierarchical clustering, PCA, t-SNE, random forest

## Templates

### R Benchmark Template

The `r_benchmark.R.template` provides a standardized structure for R benchmark scripts:

1. **Setup**: Load required packages, set seed
2. **Data Generation**: Generate test data matching Rust's DGP
3. **Benchmark Function**: Use `microbenchmark` for timing
4. **Output**: Save CSV and JSON results to `results/` directory

**Template Variables:**
- `{{METHOD_NAME}}` - Display name
- `{{METHOD_ID}}` - Short identifier
- `{{R_FUNCTION}}` - R function being benchmarked
- `{{R_PACKAGES}}` - Required packages
- `{{DATA_GENERATION_CODE}}` - R code to generate test data
- `{{BENCHMARK_CODE}}` - R code being benchmarked

### R Validation Template

The `r_validation.R.template` provides structure for validation scripts:

1. **Setup**: Load packages, set tolerances
2. **Test Cases**: Generate synthetic data with known DGP
3. **R Model Fitting**: Fit R model and extract results
4. **Comparison**: Compare against Rust results
5. **Output**: Generate markdown tables for validation docs

**Template Variables:**
- `{{TOLERANCE_COEF}}` - Coefficient tolerance (e.g., 1e-8)
- `{{TOLERANCE_SE}}` - Standard error tolerance (e.g., 1e-6)
- `{{SYNTHETIC_DATA_CODE}}` - R code for synthetic data
- `{{R_FIT_CODE}}` - R code to fit model
- `{{COMPARISON_CODE}}` - R code to compare results

## Using the Templates

When the `/validate-method` command encounters a missing R benchmark script:

1. It reads the appropriate template
2. Substitutes method-specific values for template variables
3. Prompts the user for confirmation
4. Writes the generated script to the appropriate location

**Example generated script path:**
```
performance/comparisons/r_comparison/benchmark_new_method.R
```

## Tolerance Guidelines

From the registry's `tolerances` section:

| Sample Size | Coefficients | Std Errors | P-values |
|-------------|--------------|------------|----------|
| n < 100     | 1e-6         | 1e-4       | 0.01     |
| n = 100-1000| 1e-8         | 1e-6       | 0.001    |
| n > 1000    | 1e-10        | 1e-8       | 0.0001   |

**Iterative methods** (HDFE, MLE-based): 1e-5 tolerance

## Performance Requirements

From the registry's `performance_requirements`:

| Sample Size | Required Speedup |
|-------------|------------------|
| n=1,000     | 1.5x faster      |
| n=10,000    | 2.0x faster      |
| n=100,000   | 3.0x faster      |

If Rust is slower than R, the validation reports a warning.

## Adding a New Method

1. **Add entry to `method_registry.json`**:
   ```json
   "new_method": {
     "display_name": "New Method Name",
     "r_name": "r_function",
     "r_packages": ["required_package"],
     "category": "category_name",
     "rust_module": "crates/p2a-core/src/module/file.rs",
     "rust_function": "function_name",
     "test_patterns": ["test_new_method"]
   }
   ```

2. **Run `/validate-method new_method`**:
   - Command will report missing artifacts
   - Offer to generate R benchmark from template
   - Run validation and benchmarks

3. **Fill in remaining fields** after artifacts are created

## Related Files

- **Slash Command**: `.claude/commands/validate-method.md`
- **Validation Skill**: `.claude/skills/validation-benchmarking/SKILL.md`
- **R Benchmarks**: `performance/comparisons/r_comparison/`
- **Criterion Benchmarks**: `crates/p2a-core/benches/`
- **Validation Docs**: `validation/`
- **Validation Reports**: `validation/reports/`

## Maintenance

### Updating the Registry

When new methods are added to p2a-core:

1. Add entry to `method_registry.json`
2. Create R benchmark script (or use template)
3. Add Criterion benchmark
4. Create validation documentation
5. Run `/validate-method` to verify

### Keeping Benchmarks Current

Periodically run benchmarks to track performance:

```bash
# Run all R benchmarks
cd performance/comparisons/r_comparison
Rscript run_all_benchmarks.R

# Run all Criterion benchmarks
cargo bench -p p2a-core
```

Results are saved with timestamps for historical comparison.
