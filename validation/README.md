# Validation Framework

This directory contains validation documentation comparing prompt2analytics implementations against reference implementations in R, Python, and Julia.

## Purpose

Systematic validation is essential for:
1. **Correctness verification**: Ensure our implementations produce numerically accurate results
2. **Publication readiness**: Provide evidence for Journal of Statistical Software submission
3. **User confidence**: Give users assurance that results match established packages

## Structure

```
validation/
├── README.md                          # This file
├── reference_implementations.md       # Catalog of reference packages
├── regression/                        # OLS, robust SEs, clustered SEs
├── econometrics/                      # Panel, IV, DiD, discrete choice
│   └── timeseries/                    # VAR, VARMA, VECM, IRF
├── forecasting/                       # ARIMA, MSTL, changepoint
├── ml/                                # Clustering, dimensionality reduction (cmdscale, cutree)
├── stats/                             # Statistical methods (medpolish, isoreg, loglin)
├── diagnostics/                       # Regression diagnostics
└── datasets/                          # Reference datasets for testing
```

## Validation Document Format

Each method has a dedicated markdown file following a consistent template:

1. **Method Overview**: Brief description and key parameters
2. **Reference Implementations**: Table of packages used for comparison
3. **Test Cases**: Multiple scenarios with:
   - Dataset description
   - R/Python code for reproduction
   - Results comparison table
   - Link to Rust test function
4. **Numerical Precision Summary**: Expected precision by sample size
5. **Known Differences**: Any intentional deviations from references
6. **References**: Academic citations

## Running Validation Tests

`validation/run_validation.sh` runs the whole suite and writes a report:

```bash
validation/run_validation.sh                     # Rust test_validate tests, registry lint, R drift check
validation/run_validation.sh --rust-only         # Rust only (tests and registry lint)
validation/run_validation.sh --r-only            # R drift check only: Rscript validation/r/run.R --check
validation/run_validation.sh --category stats    # one category: stats|regression|econometrics|forecasting|ml|cli|mcp
validation/run_validation.sh --r-only --legacy-r # also run the legacy validation/scripts/*.R (reported, never counted)
validation/run_validation.sh --write-status      # also regenerate validation/VALIDATION_STATUS.md
```

The Rust step lists the tests of p2a-core, p2a-cli and p2a-mcp on CI's feature sets
(`cargo test -p <pkg> --tests -- --list`), keeps the `test_validate` tests, selects a
category by module path and test binary (`validation/tools/validation_status.py select`),
and runs them with `--exact`, plus every self-test of `p2a-validation`. A compile error, a
failing test, fewer tests run than selected, or no selected `test_validate` test is a
failure. Full runs also lint the methods registry. The R step recomputes every R reference
case (see "R reference harness" below). `--legacy-r` runs each legacy script in a scratch
directory with a copy of `validation/datasets`, so tracked files are never written, and
compares the CSVs it produces with `validation/expected/*.csv`.

Exit codes: 0 PASS, 1 FAIL, 2 usage error, 3 INCOMPLETE (`cargo` or `Rscript` missing for a
requested step). Results go to `validation/reports/<timestamp>-<pid>/` (gitignored): the
logs, `results.json` and `report.md`. `--release`, `--verbose` (`-v`) and the environment
variables `P2A_RSCRIPT`, `P2A_LEGACY_R_TIMEOUT` (seconds, default 300) and `PYTHON` are
also accepted.

Direct cargo filters (libtest filters are substrings of the test path, not regexes):

```bash
cargo test -p p2a-core -- test_validate                                 # every test_validate test
cargo test -p p2a-core -- regression::ols::tests::test_validate          # the OLS validation tests
cargo test -p p2a-core -- econometrics::hdfe::tests::test_validate       # the HDFE validation tests
cargo test -p p2a-core --lib --no-default-features --features ci -- test_validate   # CI's features, no duckdb build
```

## Reference Datasets

The `datasets/` directory contains standard econometric and ML datasets:

| Dataset | n | k | Use Case |
|---------|---|---|----------|
| grunfeld.csv | 200 | 5 | Panel data (firm + year FE) |
| longley.csv | 16 | 6 | Multicollinearity testing |
| iris.csv | 150 | 5 | Classification, clustering |

## Tolerance Guidelines

Numerical precision varies by sample size and method:

| Sample Size | Coefficient Tolerance | SE Tolerance |
|-------------|----------------------|--------------|
| n < 100 | 1e-6 | 1e-5 |
| n = 100-1000 | 1e-8 | 1e-6 |
| n > 1000 | 1e-10 | 1e-8 |

For iterative methods (HDFE, MLE), expect slightly larger differences due to convergence criteria.

## R reference harness

New validations pin a method's numbers to committed R output instead of quoting R values in
comments.

Layout:

```
validation/
├── r/harness.R, r/run.R      # case registry, CSV/JSON writers, drift comparator
├── r/cases/<area>.R          # the R cases: one case(...) call per reference
├── expected/<id>.json        # one case per file (schema 1), written by run.R --write
├── datasets/generated/       # datasets that cases simulate, written and read back by R
└── methods/<area>.json       # methods registry: which tests pin which cases (methods/README.md)
```

Writing a case (`validation/r/cases/<area>.R`):

- One `case(id, area, method, function(ctx) ...)` per reference; `id` matches
  `^[a-z0-9_]+$`, is the JSON file name and starts with the area's prefix (`ols_`, `panel_`,
  `diag_`, `step_`, ...).
- Read committed data with `ctx$dataset("datasets/grunfeld.csv")` (relative to
  `validation/`). Simulated data: `set.seed(<fixed>)` inside the case, then
  `ctx$gen_dataset(name, df)`, which writes `validation/datasets/generated/<name>.csv`
  with round-trip precision and reads it back, so R computes on the doubles Rust will
  read. Data must have noise.
- Return `list(values = ..., reference = ..., call = ..., packages = ...)`. Named
  vectors (coefficients, SEs, CI bounds) go through `named(x)`, unnamed arrays through
  `vec(x)`, scalars as plain numbers; `NA` becomes `null` (NaN in Rust).
- Use the R function the method is validated against, never a re-implementation, and add
  `stopifnot()` checks for the property the case exists to exercise.

Commands:

```bash
Rscript validation/r/run.R --list            # the defined cases
Rscript validation/r/run.R --check [ids]     # recompute into a temp dir, compare with the committed files
Rscript validation/r/run.R --write [ids]     # (re)write validation/expected/<id>.json and generated datasets
```

`--check` fails (exit 1) when a recomputed value differs from the committed file by more
than 1e-12 relative, when a generated dataset differs, when a committed case has no
definition, or when a generated dataset belongs to no case. That is what fails when an R
package upgrade changes a reference value; R writes tracked files only under `--write`.
The harness needs R >= 4.3 with `jsonlite`, and the cases use `plm`, `fixest`,
`sandwich`, `lmtest`, `car`, `tseries`, `quantreg`, `nlme`, `aod`, `strucchange`, `nnet`
and `MASS`.

Pinning a method in Rust (dev-dependency `p2a-validation`; in p2a-core through
`crate::test_support`):

```rust
#[test]
fn test_validate_panel_re_grunfeld() {
    use p2a_validation::{Check, Tol};
    let (case, ds) = crate::test_support::load_case("panel_re_grunfeld");
    let r = run_random_effects(&ds, "inv", &["value", "capital"], "firm").unwrap();
    let n = r.n_obs;
    Check::new(&case)
        .named("coef", &r.variables, &r.coefficients, Tol::coef(n))
        .named("se", &r.variables, &r.std_errors, Tol::se(n))
        .scalar("df_resid", r.df as f64, Tol::exact())
        .finish();
}
```

- `load_case` reads the case and loads its dataset with `DataLoader::load_csv`, the loader
  `p2a data load` uses. `Check` compares every value, collects all failures and panics once
  listing them; it also panics when nothing was compared or when `finish()` is missing.
- Tolerances come from the table above: `Tol::coef(n)`, `Tol::se(n)`, `Tol::stat(n)`
  (statistics, R², CI bounds) and `Tol::p_value(n)`; `Tol::iterative(rel)` (MAP demeaning,
  MLE) and `Tol::new` only with a comment giving the reason; `Tol::printed(decimals)` for
  numbers parsed from CLI text; `Tol::exact()` for counts and degrees of freedom.
- A mismatch is a defect. Never widen a tolerance to make it pass: fix it, or leave the
  quantity out of the `Check` with a `// deviation <finding id>` comment and record it in
  the registry's `deviations`.
- Name R-pinned tests `test_validate_<method>_<dataset>[_<variant>]`, keep them in the
  module of the code they test, and assert: no print-only tests, no `if let Ok`.
- CLI exhibit tests compare their output with committed golden files through
  `p2a_validation::golden::assert_golden`; regenerate with `P2A_UPDATE_GOLDEN=1` only after
  the numbers in the output were checked against R (`p2a_validation::text` parses the
  coefficient table).
- Register every test that loads a case in `validation/methods/<area>.json`
  (`validation/methods/README.md`); `validation_status.py lint` fails otherwise.
- `P2A_VALIDATION_VERBOSE=1` prints every comparison; `P2A_REPO_ROOT` points the crate at
  another checkout.

## Adding New Validations

When implementing a new method:

1. Create `validation/[category]/[method].md` using the template
2. Add at least two test cases:
   - Synthetic data with known DGP
   - Real dataset (if available)
3. Document R/Python code for reproduction
4. Create corresponding Rust test with `test_validate_` prefix
5. Update this README's method index

## Method Index

### Regression
| Method | File | Status |
|--------|------|--------|
| OLS | [ols.md](regression/ols.md) | Pending |
| Robust SEs (HC0-HC3) | [ols_robust_se.md](regression/ols_robust_se.md) | Pending |
| Clustered SEs | [ols_clustered.md](regression/ols_clustered.md) | Pending |
| Tukey's Resistant Line | [line.md](regression/line.md) | **Complete** |
| Friedman's SuperSmoother | [supsmu.md](regression/supsmu.md) | **Complete** |
| Sensitivity Analysis (sensemakr) | [sensemakr.md](regression/sensemakr.md) | **Complete** |
| E-Value (unmeasured confounding) | [evalue.md](regression/evalue.md) | **Complete** |

### Econometrics
| Method | File | Status |
|--------|------|--------|
| Fixed Effects | [panel_fe.md](econometrics/panel_fe.md) | Pending |
| Random Effects | [panel_re.md](econometrics/panel_re.md) | Pending |
| Hausman Test | [hausman.md](econometrics/hausman.md) | Pending |
| IV/2SLS | [iv_2sls.md](econometrics/iv_2sls.md) | Pending |
| Diff-in-Diff | [did.md](econometrics/did.md) | Pending |
| Regression Discontinuity (Sharp) | [rd.md](econometrics/rd.md) | Pending |
| Regression Discontinuity (Fuzzy) | [rd.md](econometrics/rd.md) | Pending |
| IPW Treatment Effects | [treatment_ipw.md](econometrics/treatment_ipw.md) | Pending |
| Doubly Robust (AIPW) | [treatment_aipw.md](econometrics/treatment_aipw.md) | Pending |
| TMLE | [tmle.md](econometrics/tmle.md) | **Complete** |
| LTMLE (Longitudinal TMLE) | [ltmle.md](econometrics/ltmle.md) | **Complete** |
| CBPS | [cbps.md](econometrics/cbps.md) | **Complete** |
| SBW (Stable Balancing Weights) | [sbw.md](econometrics/sbw.md) | **Complete** |
| Mediation Analysis | [mediation.md](econometrics/mediation.md) | Pending |
| Natural Effect Models (medflex) | [medflex.md](econometrics/medflex.md) | **Complete** |
| Logit | [logit.md](econometrics/logit.md) | Pending |
| Probit | [probit.md](econometrics/probit.md) | Pending |
| HDFE | [hdfe.md](econometrics/hdfe.md) | **Complete** |
| Multi-Cutoff RD (rdmulti) | [rdmulti.md](econometrics/rdmulti.md) | **Complete** |

### Survival Analysis
| Method | File | Status |
|--------|------|--------|
| Kaplan-Meier | [survival.md](econometrics/survival.md) | **Complete** |
| Log-Rank Test | [survival.md](econometrics/survival.md) | **Complete** |
| Cox PH | [survival.md](econometrics/survival.md) | **Complete** |
| AFT Models | [survival.md](econometrics/survival.md) | **Complete** |
| Competing Risks | [survival.md](econometrics/survival.md) | **Complete** |

### Time Series
| Method | File | Status |
|--------|------|--------|
| VAR | [var.md](econometrics/timeseries/var.md) | Pending |
| VARMA | [varma.md](econometrics/timeseries/varma.md) | Pending |
| VECM | [vecm.md](econometrics/timeseries/vecm.md) | Pending |
| IRF | [irf.md](econometrics/timeseries/irf.md) | Pending |
| Cumulative Periodogram | [cpgram.md](timeseries/cpgram.md) | **Complete** |

### Forecasting
| Method | File | Status |
|--------|------|--------|
| ARIMA | [arima.md](forecasting/arima.md) | Pending |
| MSTL | [mstl.md](forecasting/mstl.md) | Pending |
| Changepoint | [changepoint.md](forecasting/changepoint.md) | Pending |

### Machine Learning
| Method | File | Status |
|--------|------|--------|
| K-means | [kmeans.md](ml/kmeans.md) | Pending |
| DBSCAN | [dbscan.md](ml/dbscan.md) | Pending |
| Hierarchical | [hierarchical.md](ml/hierarchical.md) | Pending |
| Cutree | [cutree.md](ml/cutree.md) | **Complete** |
| PCA | [pca.md](ml/pca.md) | Pending |
| t-SNE | [tsne.md](ml/tsne.md) | Pending |
| Classical MDS (cmdscale) | [cmdscale.md](ml/cmdscale.md) | **Complete** |
| Projection Pursuit Regression | [ppr.md](ml/ppr.md) | **Complete** |
| Random Forest | [random_forest.md](ml/random_forest.md) | Pending |
| SVM | [svm.md](ml/svm.md) | Pending |
| Causal Forest | [causal_forest.md](ml/causal_forest.md) | Pending |
| BART Causal (bartCause) | [bart_causal.md](ml/bart_causal.md) | **Complete** |

### Statistical Methods
| Method | File | Status |
|--------|------|--------|
| Median Polish | [medpolish.md](stats/medpolish.md) | **Complete** |
| Isotonic Regression | [isoreg.md](stats/isoreg.md) | **Complete** |
| Log-Linear Models | [loglin.md](stats/loglin.md) | **Complete** |
| Constrained Optimization | [constroptim.md](stats/constroptim.md) | **Complete** |
| SE for Contrasts | [secontrast.md](stats/secontrast.md) | **Complete** |
| Model Tables | [modeltables.md](stats/modeltables.md) | **Complete** |

### Spatial Econometrics
| Method | File | Status |
|--------|------|--------|
| Moran's I Test | [moran.md](spatial/moran.md) | Pending |
| SAR (Spatial Lag) | [sar.md](spatial/sar.md) | Pending |
| SEM (Spatial Error) | [sem.md](spatial/sem.md) | Pending |
| SAR Probit | [spatialprobit.md](econometrics/spatialprobit.md) | **Complete** |
| SEM Probit | [spatialprobit.md](econometrics/spatialprobit.md) | **Complete** |
| Spatial GMM (sphet) | [sphet.md](econometrics/sphet.md) | **Complete** |

### Linear Algebra
| Method | File | Status |
|--------|------|--------|
| Toeplitz Matrix | [toeplitz.md](linalg/toeplitz.md) | **Complete** |

### Diagnostics
| Method | File | Status |
|--------|------|--------|
| Regression Diagnostics | [regression_diagnostics.md](diagnostics/regression_diagnostics.md) | Pending |

## Contributing

When adding validations:
- Use consistent formatting
- Include reproducible R/Python code
- Test on multiple platforms if possible
- Document any known limitations
