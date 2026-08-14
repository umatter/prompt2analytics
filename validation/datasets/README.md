# Reference Datasets

This directory contains standard datasets used for validation testing.

## Available Datasets

### Grunfeld (1958)
**File**: `grunfeld.csv`

| Variable | Description |
|----------|-------------|
| firm | Firm identifier (1-10) |
| year | Year (1935-1954) |
| inv | Gross investment |
| value | Market value |
| capital | Stock of capital |

**Size**: 200 observations (10 firms × 20 years)

**Use Cases**:
- Panel data methods (FE, RE, HDFE)
- Two-way fixed effects validation

**Source**: Grunfeld, Y. (1958). "The Determinants of Corporate Investment". Unpublished Ph.D. dissertation, University of Chicago. Available in R's `plm` package.

**R Code**:
```r
library(plm)
data(Grunfeld)
write.csv(Grunfeld, "grunfeld.csv", row.names = FALSE)
```

---

### Longley (1967)
**File**: `longley.csv`

| Variable | Description |
|----------|-------------|
| GNP.deflator | GNP deflator |
| GNP | Gross National Product |
| Unemployed | Number unemployed |
| Armed.Forces | Size of armed forces |
| Population | Population |
| Year | Year |
| Employed | Number employed (outcome) |

**Size**: 16 observations

**Use Cases**:
- Multicollinearity testing
- VIF and condition number validation
- Small-sample regression

**Source**: Longley, J.W. (1967). "An Appraisal of Least Squares Programs for the Electronic Computer from the Point of View of the User". Journal of the American Statistical Association, 62, 819-841.

**R Code**:
```r
data(longley)
write.csv(longley, "longley.csv", row.names = FALSE)
```

---

### Iris
**File**: `iris.csv`

| Variable | Description |
|----------|-------------|
| Sepal.Length | Sepal length in cm |
| Sepal.Width | Sepal width in cm |
| Petal.Length | Petal length in cm |
| Petal.Width | Petal width in cm |
| Species | Species name (setosa, versicolor, virginica) |

**Size**: 150 observations (50 per species)

**Use Cases**:
- Classification methods
- Clustering validation
- PCA validation

**Source**: Fisher, R.A. (1936). "The use of multiple measurements in taxonomic problems". Annals of Eugenics, 7, 179-188.

**R Code**:
```r
data(iris)
write.csv(iris, "iris.csv", row.names = FALSE)
```

---

### Quarterly GDP growth (synthetic)
**File**: `gdp_quarterly.csv`

| Variable | Description |
|----------|-------------|
| date | Quarter label, `YYYY-Qn` |
| year | Year (1954-2023) |
| quarter | Quarter within year (1-4) |
| gdp_growth | Quarterly growth rate (synthetic) |

**Size**: 280 observations (quarters labeled 1954-Q1 to 2023-Q4)

**Use Cases**:
- Time series workflow demonstration (stationarity testing, ARIMA, residual diagnostics)
- Used in the paper's §4.3 time series example

**Source**: **Synthetic.** This is *not* real national accounts data. The values
are a simulated stationary series (roughly mean 0.5, sd 0.8, no recession
structure) dressed with real-looking quarterly date labels; e.g. 2020-Q2 shows
no COVID collapse, so it cannot be matched to any actual GDP series. It exists
only to exercise the time series tools on a GDP-shaped, stationary series. The
original generation script/seed was not preserved; treat the checked-in CSV as
the canonical artifact. Do not cite it as empirical GDP data.

---

## Adding New Datasets

When adding a new reference dataset:

1. Include the CSV file with appropriate headers
2. Document in this README:
   - Variable descriptions
   - Size and structure
   - Use cases
   - Source citation
   - R/Python code to regenerate

3. Use consistent formatting:
   - UTF-8 encoding
   - Comma delimiters
   - Header row with variable names
   - No row names/indices

## Dataset Integrity

To verify dataset integrity, compare checksums:

```bash
# Generate checksums
md5sum validation/datasets/*.csv

# Expected values (update when adding files)
# abc123... grunfeld.csv
# def456... longley.csv
# ghi789... iris.csv
```
