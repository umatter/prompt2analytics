# Validation: Clustered Standard Errors

## Method Overview

Clustered standard errors account for correlation within groups (clusters) such as firms, states, or individuals over time. They are essential when observations within a cluster are not independent.

**Key Parameters**:
- `cluster1`: Primary clustering variable
- `cluster2`: Optional second clustering variable (for two-way clustering)

**Formula (One-way)**:
```
V̂_cluster = (X'X)⁻¹ (Σ_g X'_g û_g û'_g X_g) (X'X)⁻¹
```
where g indexes clusters and û_g is the vector of residuals for cluster g.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| sandwich | R | `vcovCL()` | 3.0-x |
| lfe | R | `felm(..., cluster = ~var)` | 2.8-x |
| linearmodels | Python | `PanelOLS(..., cluster_entity=True)` | 5.x |

## Test Cases

### Test 1: Basic Cluster-Robust SEs

**Data Generating Process**:
Observations within clusters share a common error component:
```
y_ig = β₀ + β₁x_ig + u_g + ε_ig
```
where u_g is the cluster-level error and ε_ig is individual-level error.

**R Code**:
```r
library(sandwich)
library(lmtest)

set.seed(42)
n_clusters <- 50
obs_per_cluster <- 10
n <- n_clusters * obs_per_cluster

# Generate clustered data
cluster <- rep(1:n_clusters, each = obs_per_cluster)
u_cluster <- rnorm(n_clusters, 0, 1)  # Cluster-level shock
x <- rnorm(n)
y <- 1.0 + 2.0 * x + u_cluster[cluster] + rnorm(n, 0, 0.5)

data <- data.frame(y = y, x = x, cluster = factor(cluster))

# OLS
fit <- lm(y ~ x, data = data)

# Standard SEs (incorrect)
se_standard <- coef(summary(fit))[, "Std. Error"]

# Cluster-robust SEs
se_cluster <- sqrt(diag(vcovCL(fit, cluster = data$cluster)))

# Print comparison
cbind(Standard = se_standard, Clustered = se_cluster)
```

**Validation Criteria**:
- Clustered SEs > Standard SEs (when clustering matters)
- Number of clusters correctly computed
- Coefficients unchanged (only SEs differ)

---

### Test 2: Comparison with R's sandwich Package

**R Code**:
```r
library(sandwich)
library(plm)

# Use Grunfeld panel data
data(Grunfeld)

# Pooled OLS
fit <- lm(inv ~ value + capital, data = Grunfeld)

# Cluster by firm
se_firm <- sqrt(diag(vcovCL(fit, cluster = Grunfeld$firm)))

# Cluster by year
se_year <- sqrt(diag(vcovCL(fit, cluster = Grunfeld$year)))

# Two-way clustering (firm + year)
se_twoway <- sqrt(diag(vcovCL(fit, cluster = ~ firm + year, multi0 = TRUE)))

print(cbind(Firm = se_firm, Year = se_year, TwoWay = se_twoway))
```

**Results Comparison**:

| Coefficient | R Firm-Clustered | p2a Firm | Tolerance |
|-------------|-----------------|----------|-----------|
| (Intercept) | X.XX | X.XX | 1e-4 |
| value | X.XX | X.XX | 1e-6 |
| capital | X.XX | X.XX | 1e-6 |

---

### Test 3: Few Clusters (Small-Cluster Corrections)

When the number of clusters is small (< 50), small-sample corrections matter.

**R Code**:
```r
library(sandwich)

set.seed(42)
n_clusters <- 10
obs_per_cluster <- 50
n <- n_clusters * obs_per_cluster

cluster <- rep(1:n_clusters, each = obs_per_cluster)
x <- rnorm(n)
y <- 1.0 + 2.0 * x + rnorm(n_clusters)[cluster] + rnorm(n, 0, 0.5)

data <- data.frame(y = y, x = x, cluster = factor(cluster))
fit <- lm(y ~ x, data = data)

# Default clustering (with small-sample correction)
se_corrected <- sqrt(diag(vcovCL(fit, cluster = data$cluster)))

# Without correction
se_uncorrected <- sqrt(diag(vcovCL(fit, cluster = data$cluster,
                                    cadjust = FALSE)))

cbind(Corrected = se_corrected, Uncorrected = se_uncorrected)
```

**Validation Criteria**:
- Corrected SEs > Uncorrected SEs
- Difference is larger with fewer clusters

---

### Test 4: Two-Way Clustering (Firm + Year)

**R Code**:
```r
library(sandwich)
library(plm)

data(Grunfeld)
fit <- lm(inv ~ value + capital, data = Grunfeld)

# Two-way clustering using Cameron-Gelbach-Miller formula:
# V_twoway = V_firm + V_year - V_intersection
se_twoway <- sqrt(diag(vcovCL(fit, cluster = ~ firm + year, multi0 = TRUE)))

print(se_twoway)
```

**Mathematical Formula**:
```
V̂_two-way = V̂_cluster1 + V̂_cluster2 - V̂_intersection
```

where V̂_intersection clusters by the interaction of both variables.

**Validation Criteria**:
- Two-way SEs differ from one-way SEs
- Correct handling of intersection term

---

## Numerical Precision Summary

| Clustering Type | n_clusters | SE Precision vs R |
|-----------------|------------|------------------|
| One-way | 50+ | < 1e-8 |
| One-way | 10-50 | < 1e-6 |
| Two-way | 50+ | < 1e-6 |

## Known Differences

1. **Small-sample correction**: R's sandwich uses G/(G-1) by default; p2a uses same correction.
2. **Degrees of freedom**: p2a uses min(G1, G2) for two-way clustering as per Cameron-Gelbach-Miller.
3. **Singleton clusters**: R may handle differently; p2a includes them in count.

## Running the Tests

```bash
# Run clustered SE tests
cargo test -p p2a-core -- clustered

# Run with output
cargo test -p p2a-core -- test_clustered --nocapture
```

## References

- Arellano, M. (1987). "Computing Robust Standard Errors for Within-Groups Estimators". *Oxford Bulletin of Economics and Statistics*, 49(4), 431-434.
- Cameron, A.C., Gelbach, J.B., & Miller, D.L. (2011). "Robust Inference with Multiway Clustering". *Journal of Business & Economic Statistics*, 29(2), 238-249.
- Petersen, M.A. (2009). "Estimating Standard Errors in Finance Panel Data Sets: Comparing Approaches". *Review of Financial Studies*, 22(1), 435-480.
