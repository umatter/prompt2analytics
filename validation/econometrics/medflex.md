# Validation: Natural Effect Models (medflex)

## Method Overview

Natural Effect Models (NEM) provide a regression-based approach to causal mediation analysis that allows for treatment-mediator interactions. This implementation follows the methodology of:

- Lange, T., Vansteelandt, S., & Bekaert, M. (2012). "Choice of effect measure for mediation analysis." *Epidemiology*, 23(6), 889-897.
- VanderWeele, T. J. (2015). *Explanation in Causal Inference: Methods for Mediation and Interaction*. Oxford University Press.
- Steen, J., et al. (2017). "medflex: An R Package for Flexible Mediation Analysis using Natural Effect Models." *Journal of Statistical Software*, 76(11), 1-46.

## Implementation Details

**File:** `crates/p2a-core/src/econometrics/medflex.rs`

**Key Functions:**
- `run_medflex()` - Core implementation using ArrayViews
- `run_medflex_dataset()` - Dataset-based interface

**Key Structures:**
- `MedflexConfig` - Configuration (interaction, bootstrap, confidence level, scale)
- `MedflexResult` - Complete results including NDE, NIE, TE, SEs, CIs, p-values

## Mathematical Framework

### Models

1. **Mediator Model:**
   ```
   M = alpha_0 + alpha_1*A + alpha_2'*C + epsilon_M
   ```

2. **Outcome Model (with interaction):**
   ```
   Y = beta_0 + beta_1*A + beta_2*M + beta_3*A*M + beta_4'*C + epsilon_Y
   ```

### Effect Decomposition

**Without interaction (beta_3 = 0):**
- NDE = beta_1
- NIE = alpha_1 * beta_2
- TE = NDE + NIE

**With interaction:**
- NDE = beta_1 + beta_3 * E[M|A=0]
- NIE = alpha_1 * (beta_2 + beta_3)
- TE = NDE + NIE

## Reference Implementation

### R Package: medflex

```r
# Install
install.packages("medflex")
library(medflex)

# Simulate data
set.seed(42)
n <- 200
x <- rnorm(n)
a <- rbinom(n, 1, plogis(0.5*x))
m <- 0.5 + 0.6*a + 0.3*x + rnorm(n, sd=0.3)
y <- 1.0 + 0.4*a + 0.5*m + 0.2*a*m + 0.3*x + rnorm(n, sd=0.5)
data <- data.frame(y=y, a=a, m=m, x=x)

# Fit natural effect model
library(medflex)
expData <- neWeight(a ~ x, data = data)
neMod <- neModel(y ~ a0 + a1 + x, expData = expData, se = "robust")
summary(neMod)

# Get effects
neEffdecomp(neMod)
```

### Test Case 1: Basic Mediation

**Simulated Data (n=100):**
- Treatment effect on mediator: alpha_1 ~ 0.6
- Direct effect: beta_1 ~ 0.4
- Mediator effect: beta_2 ~ 0.5
- Interaction: beta_3 ~ 0.2

**Expected Results (R medflex):**
| Effect | Estimate | SE |
|--------|----------|-----|
| TE | ~0.85 | ~0.15 |
| NDE | ~0.50 | ~0.12 |
| NIE | ~0.35 | ~0.08 |

**Rust Implementation:**
```rust
let config = MedflexConfig {
    allow_interaction: true,
    bootstrap_ci: true,
    n_bootstrap: 1000,
    confidence_level: 0.95,
    scale: EffectScale::Difference,
    seed: Some(42),
};

let result = run_medflex(&y.view(), &a.view(), &m.view(), &x.view(), config)?;
assert!((result.total_effect - 0.85).abs() < 0.2);
```

### Test Case 2: No Interaction

**Expected Results (product method):**
- NDE = beta_1 (direct coefficient)
- NIE = alpha_1 * beta_2 (product of coefficients)

**Rust Implementation:**
```rust
let config = MedflexConfig {
    allow_interaction: false,
    ..Default::default()
};

let result = run_medflex(&y.view(), &a.view(), &m.view(), &x.view(), config)?;
// NDE should equal the direct coefficient from outcome regression
// NIE should equal alpha_1 * beta_2
```

## Validation Tests

### Unit Tests

```bash
cargo test -p p2a-core econometrics::medflex
```

**Results:**
- `test_medflex_basic` - Basic effect decomposition
- `test_medflex_with_interaction` - Treatment-mediator interaction
- `test_medflex_with_confounders` - Confounder adjustment
- `test_medflex_delta_method` - Delta method standard errors
- `test_medflex_dataset_interface` - Dataset-based API
- `test_medflex_display` - Result formatting
- `test_medflex_insufficient_data` - Error handling
- `test_proportion_mediated_bounds` - Proportion mediated bounds

### Tolerance Guidelines

| Statistic | Tolerance |
|-----------|-----------|
| Coefficients | 0.1 (with noise) |
| Standard Errors | 0.05 |
| p-values | 0.05 |
| Proportion mediated | 0.1 |

## Identification Assumptions

1. **No unmeasured confounding of A-Y relationship** given C
2. **No unmeasured confounding of M-Y relationship** given (A, C)
3. **No unmeasured confounding of A-M relationship** given C
4. **Cross-world independence**: No effect of A on M-Y confounders

## MCP Tool

**Tool Name:** `natural_effects_mediation`

**Parameters:**
- `dataset` - Dataset name
- `outcome` - Outcome variable
- `treatment` - Treatment variable
- `mediator` - Mediator variable
- `confounders` - Confounder columns (optional)
- `allow_interaction` - Include A*M interaction (default: true)
- `n_bootstrap` - Bootstrap samples (default: 1000)
- `confidence_level` - CI level (default: 0.95)
- `scale` - Effect scale: "difference", "ratio", "odds_ratio"

## Missing values and column types

`run_medflex_dataset` (MCP `natural_effects_mediation`) drops the rows with a missing value
(null or NaN) in the outcome, treatment, mediator or a confounder jointly, so the mediator
and outcome models are fitted on the same complete rows (R `lm`, `na.omit`). The drop is
reported in `MedflexResult.missing` (serialized only when rows were dropped) and in a last
Display line (also in the MCP text):
`Missing values: 50 of 400 rows dropped because of missing values (y: 15, m: 25, x1: 10)`.
`run_medflex` takes arrays as given and reports nothing dropped. Int64 and Boolean columns
are read as numbers; a text column gives `NonNumericColumn`, an absent one
`ColumnNotFound`.

### R reference (T2-WP13)

Case `causal_mediation_na` (`validation/r/cases/causal_mediation_na.R`, registry
`validation/methods/causal_na.json`, id `medflex_na`), the 350 complete rows of 400:

| Quantity | R (`lm` on `complete.cases(y, d, m, x1, x2)`) |
|---|---|
| treated, control | 197, 153 |
| E[M\|A=0], E[M\|A=1] | 0.3767530563055379, 1.2007356199594112 |
| `lm(m ~ d + x1 + x2)`: `d` (SE), R² | 0.6131933908612826 (0.059414385098645314), 0.5943014059209022 |
| `lm(y ~ d + m + x1 + x2)`: `d` (SE), `m` (SE), R² | 0.5104280327655459 (0.07402655405381642), 0.6554511496553376 (0.0585705531269962), 0.7489960724962549 |
| `lm(y ~ d + m + d:m + x1 + x2)`: `d` (SE), `m` (SE), `d:m` (SE), R² | 0.29803058555399 (0.09699712509259713), 0.4867447333122127 (0.0768751565077352), 0.29678541513089 (0.08929231721797547), 0.7568060755019845 |

Tolerances: `validation/README.md` for n = 350 (coefficients and means 1e-8, SEs and R² 1e-6
relative); counts exact; the MCP text at `Tol::printed(4)`. With missing values read as 0,
as before T2-WP13, the mediator coefficient of `d` was 0.6243 on 400 rows. Tests (p2a-core,
`econometrics::medflex`): `test_validate_medflex_causal_na`,
`test_validate_medflex_interaction_causal_na`, `test_validate_na_medflex_reports_missing`,
`test_validate_na_medflex_errors_name_the_column`; MCP:
`test_validate_mcp_natural_effects_mediation_causal_na`.

The component models are pinned to `lm`. medflex 0.6-11 is installed, but p2a's NDE/NIE
differ from it (CAUS-23, Tier 4), so they are not pinned.

## References

1. Lange, T., Vansteelandt, S., & Bekaert, M. (2012). "Choice of effect measure for mediation analysis." *Epidemiology*, 23(6), 889-897. https://doi.org/10.1097/EDE.0b013e31826c2107

2. VanderWeele, T. J. (2015). *Explanation in Causal Inference: Methods for Mediation and Interaction*. Oxford University Press.

3. Steen, J., Loeys, T., Moerkerke, B., & Vansteelandt, S. (2017). "medflex: An R Package for Flexible Mediation Analysis using Natural Effect Models." *Journal of Statistical Software*, 76(11), 1-46. https://doi.org/10.18637/jss.v076.i11

4. R package medflex: https://CRAN.R-project.org/package=medflex

## Status

**Implementation:** Complete
**Tests:** 8/8 passing
**Validation:** In progress
**MCP Tool:** Added

Last updated: 2026-01-25
