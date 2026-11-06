# Validation: Difference-in-Differences (DiD)

## Method Overview

Difference-in-Differences is a quasi-experimental design that estimates causal effects by comparing changes over time between a treatment group and a control group.

**Model**:
```
Y = β₀ + β₁×Treatment + β₂×Post + β₃×(Treatment × Post) + ε
```

**ATT (Average Treatment Effect on Treated)** = β₃

**Key Assumption**: Parallel trends - absent treatment, treated and control groups would have followed the same trend.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats + sandwich | R | `lm`, `vcovHC(type = "HC1")`, `vcovCL(type = "HC1")` | R 4.x (case `did_2x2_training`) |
| did | R | `att_gt()` for staggered DiD | 2.1-x |
| statsmodels | Python | Manual interaction regression | 0.14.x |

## R reference: case `did_2x2_training`

`validation/r/cases/causal_did_cov.R`, case `did_2x2_training`, on
`validation/datasets/employee_training.csv` (1200 rows, 200 employees; the paper's
`diff_in_diff` example):

```r
lm(performance ~ training_group + post_training + training_group:post_training
     [+ tenure [+ years_since_hire]], data = employee_training)
sandwich::vcovHC(fit, type = "HC1")
sandwich::vcovCL(fit, cluster = ~employee_id, type = "HC1")
# t and p on df.residual
```

Three specifications: `base`, `tenure`, and `aliased` (`tenure` and `years_since_hire`, which is
`tenure + 0.5`: `lm` reports `NA` for `years_since_hire`).

| Quantity | base | tenure | aliased |
|---|---|---|---|
| ATT (interaction) | 0.21296162104133362 | 0.21296162104133187 | 0.21296162104133187 |
| CR1 SE of the ATT | 0.035166867334917605 | | 0.03518157842835946 |
| HC1 SE of the ATT | 0.06453459673051398 | | |
| years_since_hire | | | NA |
| df.residual | 1196 | 1195 | 1195 |
| R2 | 0.21322320903407338 | | 0.22340065513795904 |

Group means (control pre, control post, treated pre, treated post): 3.0021376811594203,
3.127355072463768, 3.424351851851852, 3.762530864197531; 200 clusters. Every coefficient, SE,
t and p of each specification is in `validation/expected/did_2x2_training.json`.

**Test**: `test_validate_did_2x2_training` (`crates/p2a-core/src/econometrics/did.rs`; registry
`validation/methods/causal_did_cov.json`, method `did_2x2`): coefficients `Tol::coef(1200)`
(1e-10 relative), HC1 and CR1 SEs `Tol::se(1200)`, t, R2 and adjusted R2 `Tol::stat(1200)`,
p-values `Tol::p_value(1200)`, df exact, the aliased name and R's warning, the headline fields
against the interaction's entry, and the group means.

**Estimation rules** (CAUS-102):
- the regression is solved as OLS is: the normal equations on a full-rank, well-conditioned
  design, otherwise `lm`'s pivoted QR (`dqrdc2`, tolerance 1e-7); an aliased control is `NA`
  (NaN; Display prints `NA`) and the result carries `aliased` and R's warning "Coefficients not
  defined because of singularities: ...";
- an aliased treatment x post interaction (for example only treated units) is an
  `InvalidSpecification` error ("not identified");
- df = n - rank by checked arithmetic (n <= rank is `InsufficientData`);
- HC1: `B (sum e_i^2 x_i x_i') B n / df`; CR1: `B meat B (G / (G - 1)) ((n - 1) / df)`, with B the
  inverse cross-product of the identified columns; clustering needs at least 2 clusters
  (`vcovCL` fails with one).

Before the fix `run_did` inverted X'X: the aliased control got a generalized-inverse coefficient
with df 1194, an unidentified interaction got a number, and a single cluster fell back to a
correction of 1.

Argument tests: `test_run_did_rejects_n_le_k`, `test_run_did_rejects_single_cluster`,
`test_run_did_rejects_unidentified_att`.

## DGP property tests (not R-pinned)

`test_did_classic_2x2_recovers_dgp` (ATT within 0.5 of the DGP's 5.0, group means, ATT equal to
the manual difference in differences), `test_did_with_covariates_recovers_dgp` and
`test_did_null_effect_recovers_dgp` (p > 0.05; on its deterministic data `lm` +
`vcovHC(type = "HC1")` gives ATT -0.028054724238937286, SE 0.21315963502411217, p
0.89535670480011142, which the test quotes).

## Further designs (documentation, not tested)

### Test 3: Panel DiD with Entity Fixed Effects

**R Code**:
```r
library(plm)

set.seed(42)
n_entities <- 100
n_periods <- 10
treatment_period <- 6

data <- expand.grid(
  id = 1:n_entities,
  t = 1:n_periods
)

# Half are treated (after period 6)
data$treat <- ifelse(data$id <= 50, 1, 0)
data$post <- ifelse(data$t >= treatment_period, 1, 0)

# Entity fixed effects
entity_fe <- rnorm(n_entities)
time_fe <- seq(0, 1, length.out = n_periods)

data$y <- entity_fe[data$id] + time_fe[data$t] +
          5 * data$treat * data$post +  # Treatment effect
          rnorm(nrow(data), 0, 0.5)

pdata <- pdata.frame(data, index = c("id", "t"))

# DiD with entity FE
did_fe <- plm(y ~ treat:post, data = pdata, model = "within")
summary(did_fe)
```

Not tested: `run_did` has no entity fixed effects (use the panel estimators).

---

## ATT Calculation

The ATT can be computed directly from group means:

```
ATT = (Ȳ_treat,post - Ȳ_treat,pre) - (Ȳ_control,post - Ȳ_control,pre)
```

| Group | Pre | Post | Δ |
|-------|-----|------|---|
| Control | Ȳ₀₀ | Ȳ₀₁ | Ȳ₀₁ - Ȳ₀₀ |
| Treated | Ȳ₁₀ | Ȳ₁₁ | Ȳ₁₁ - Ȳ₁₀ |

ATT = (Ȳ₁₁ - Ȳ₁₀) - (Ȳ₀₁ - Ȳ₀₀)

## Known Differences

1. **Standard errors**: Default vs. clustered by entity.
2. **Fixed effects**: Some implementations require explicit FE specification.
3. **Weights**: Unweighted vs. weighted by group size.

## Parallel Trends Assumption

Cannot be tested directly, but can assess plausibility:

**R Code for Pre-Trend Test**:
```r
# Check pre-treatment trends are parallel
pre_data <- subset(data, post == 0)
trend_test <- lm(y ~ treat + as.factor(t) + treat:as.factor(t), data = pre_data)
# Interaction terms should be insignificant
```

## Missing values, column types and clusters

`run_did` (MCP `diff_in_diff`, CLI `causal did`) reads the outcome, treatment, post indicator,
controls and the cluster id through one `ColumnRequest`:

- a row with a missing value (null or NaN) in any of them is dropped from all of them
  (`NaPolicy::Omit`, as `lm`/`fixest` drop incomplete rows), and `DiDResult::missing` and the
  last Display line report it ("Missing values: 2 of 200 rows dropped because of missing values
  (y: 1, cl_int: 1)");
- cluster ids are factors of any type (integers, fractional numbers, strings, categoricals,
  dates): every distinct value is one cluster for the CR1 sandwich. Numeric ids were keyed by
  `format!("{:.0}")` before, which merged fractional ids (0.0 .. 1.9 became 3 clusters);
- a text outcome is `NonNumericColumn`, an absent column `ColumnNotFound`, and fewer complete
  rows than regressors plus one is `InsufficientData`.

Fixture tests, not R-pinned (`crates/p2a-core/src/econometrics/did.rs`):
`test_na_did_cluster_ids_are_factors` (integer, string and fractional ids of 20 clusters give one
result), `test_na_did_omits_missing_rows`, `test_na_did_errors_name_the_column`.

## Running the Tests

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::did
Rscript validation/r/run.R --check          # did_2x2_training still reproduces
```

## References

- Card, D. & Krueger, A.B. (1994). "Minimum Wages and Employment". *American Economic Review*, 84(4), 772-793.
- Angrist, J.D. & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press. Chapter 5.
- Goodman-Bacon, A. (2021). "Difference-in-Differences with Variation in Treatment Timing". *Journal of Econometrics*, 225(2), 254-277.
