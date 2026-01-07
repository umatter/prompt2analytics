# Desktop App Test Scenario

This document provides a step-by-step test scenario for validating the prompt2analytics desktop application.

## Prerequisites

1. Build the application:
   ```bash
   cd /path/to/prompt2analytics
   cargo build --release -p p2a-mcp
   cargo build --release -p p2a-desktop
   ```

2. Ensure system dependencies are installed (Linux):
   ```bash
   sudo apt install libwebkit2gtk-4.1-dev libsoup-3.0-dev libjavascriptcoregtk-4.1-dev
   ```

## Test Data

Use the provided `sample_sales.csv` file in this directory. It contains 20 rows of fictional sales data with columns:

| Column | Type | Description |
|--------|------|-------------|
| id | integer | Row identifier |
| region | string | Sales region (North, South, East, West) |
| product | string | Product name (Widget A, B, C) |
| units_sold | integer | Number of units sold |
| revenue | float | Total revenue in dollars |
| cost | float | Total cost in dollars |
| date | string | Sale date (YYYY-MM-DD) |

---

## Test Scenario

### Step 1: Launch the Application

```bash
./target/release/p2a-desktop
```

**Expected:** A window opens with three panels:
- Left: Chat panel with input field
- Center: Data panel (empty initially)
- Right: Results panel (empty initially)

Console output should show:
```
Using MCP binary: "/path/to/target/release/p2a-mcp"
MCP server started successfully
```

---

### Step 2: Load the Test Dataset

1. Click the **Import** button in the Data panel
2. Navigate to `docs/testing/sample_sales.csv`
3. Select the file and click Open

**Expected:**
- The Data panel shows a dropdown with "sample_sales" selected
- A table preview appears showing the first few rows
- Chat shows a message confirming the dataset was loaded

---

### Step 3: Describe the Dataset

Type in the chat input:
```
/describe_dataset dataset:sample_sales
```

**Expected Output (Results Panel):**

```
Dataset: sample_sales (20 rows x 7 columns)

Numeric Columns:
┌────────────┬───────┬─────────┬──────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ column     │ count │ mean    │ std      │ min     │ 25%     │ 50%     │ 75%     │ max     │
├────────────┼───────┼─────────┼──────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ id         │ 20    │ 10.5    │ 5.92     │ 1       │ 5.75    │ 10.5    │ 15.25   │ 20      │
│ units_sold │ 20    │ 132.75  │ 40.49    │ 75      │ 97.5    │ 127.5   │ 163.75  │ 220     │
│ revenue    │ 20    │ 5022.50 │ 1701.88  │ 2700    │ 3862.5  │ 4625    │ 5925    │ 8800    │
│ cost       │ 20    │ 2739.25 │ 1050.60  │ 1350    │ 2062.5  │ 2512.5  │ 3285    │ 5280    │
└────────────┴───────┴─────────┴──────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

Categorical Columns:
┌─────────┬────────┬─────────────────────────────────┐
│ column  │ unique │ top values                      │
├─────────┼────────┼─────────────────────────────────┤
│ region  │ 4      │ North(5), South(5), East(5)...  │
│ product │ 3      │ Widget A(7), Widget B(7)...     │
│ date    │ 5      │ 2024-01-15(4), 2024-01-22(4)... │
└─────────┴────────┴─────────────────────────────────┘
```

**Key Values to Verify:**
- Row count: 20
- Mean units_sold: ~132.75
- Mean revenue: ~5022.50
- 4 unique regions, 3 unique products

---

### Step 4: Compute Correlations

Type in the chat input:
```
/compute_correlation dataset:sample_sales
```

**Expected Output:**

A correlation matrix showing relationships between numeric columns:

```
Correlation Matrix:
┌────────────┬──────┬────────────┬─────────┬───────┐
│            │ id   │ units_sold │ revenue │ cost  │
├────────────┼──────┼────────────┼─────────┼───────┤
│ id         │ 1.00 │ 0.08       │ 0.05    │ 0.05  │
│ units_sold │ 0.08 │ 1.00       │ 0.91    │ 0.91  │
│ revenue    │ 0.05 │ 0.91       │ 1.00    │ 0.99  │
│ cost       │ 0.05 │ 0.91       │ 0.99    │ 1.00  │
└────────────┴──────┴────────────┴─────────┴───────┘
```

**Key Values to Verify:**
- Strong correlation (0.91) between units_sold and revenue
- Very strong correlation (0.99) between revenue and cost
- Weak correlation (~0.05-0.08) between id and other variables

---

### Step 5: Run OLS Regression

Type in the chat input:
```
/regression_ols dataset:sample_sales y:revenue x:units_sold,cost
```

**Expected Output:**

```
OLS Regression Results
======================
Dependent Variable: revenue
Observations: 20
R-squared: 0.999+

Coefficients:
┌────────────┬───────────┬──────────┬─────────┬─────────┐
│ Variable   │ Coef      │ Std Err  │ t-value │ p-value │
├────────────┼───────────┼──────────┼─────────┼─────────┤
│ const      │ ~0        │ ...      │ ...     │ ...     │
│ units_sold │ ~10       │ ...      │ ...     │ < 0.001 │
│ cost       │ ~1.2      │ ...      │ ...     │ < 0.001 │
└────────────┴───────────┴──────────┴─────────┴─────────┘

F-statistic: very high (p < 0.001)
```

**Key Values to Verify:**
- R-squared should be very high (>0.99) since revenue = units × price and cost is proportional
- Coefficients should be significant (p < 0.05)

---

### Step 6: Generate a Histogram

Type in the chat input:
```
/viz_histogram dataset:sample_sales column:revenue
```

**Expected Output:**
- A histogram image appears in the Results panel
- Shows distribution of revenue values
- Should show bars spanning roughly $2,700 to $8,800

---

### Step 7: Generate a Scatter Plot

Type in the chat input:
```
/viz_scatter dataset:sample_sales x_column:units_sold y_column:revenue
```

**Expected Output:**
- A scatter plot image appears in the Results panel
- Shows positive linear relationship between units_sold and revenue
- Points should form a roughly linear pattern

---

### Step 8: K-Means Clustering

Type in the chat input:
```
/ml_kmeans dataset:sample_sales columns:units_sold,revenue k:3
```

**Expected Output:**

```
K-Means Clustering Results
==========================
Number of clusters: 3
Iterations: ...
Inertia: ...

Cluster Centroids:
┌─────────┬────────────┬─────────┐
│ Cluster │ units_sold │ revenue │
├─────────┼────────────┼─────────┤
│ 0       │ ~95        │ ~3500   │
│ 1       │ ~145       │ ~5200   │
│ 2       │ ~195       │ ~7600   │
└─────────┴────────────┴─────────┘

Cluster assignments added to dataset.
```

**Key Values to Verify:**
- 3 distinct clusters
- Centroids should reflect low/medium/high sales groups

---

## Troubleshooting

### App doesn't start
- Check that `p2a-mcp` binary exists in `target/release/`
- Check console output for error messages

### "MCP server not running" error
- The server may have crashed; restart the app
- Check that no other instance is running

### No charts appearing
- Verify the Results panel is visible
- Check that the dataset name is correct
- Look for error messages in the chat

### Dataset not loading
- Verify the CSV file exists and is valid
- Check file permissions

---

## Success Criteria

The test is successful if:

1. App launches without errors
2. CSV file loads and displays in Data panel
3. `describe_dataset` returns correct statistics
4. `compute_correlation` shows expected correlations
5. `regression_ols` runs and shows coefficients
6. `viz_histogram` displays a chart image
7. `viz_scatter` displays a chart image
8. `ml_kmeans` clusters the data into 3 groups

Report any deviations from expected output as bugs.
