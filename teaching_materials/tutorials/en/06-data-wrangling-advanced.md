# Advanced Data Wrangling

**Difficulty: Advanced**

## Learning Objectives

After completing this tutorial, you will be able to:
- [ ] Perform complex multi-table joins
- [ ] Reshape data between wide and long formats
- [ ] Use window functions for running calculations
- [ ] Build complete analytical datasets from multiple sources
- [ ] Apply data quality checks and validation
- [ ] Create lag and lead variables for panel/time series data

## Prerequisites

- Completed [Data Munging Basics](04-data-munging-basics.md)
- Completed [Intermediate Data Operations](05-data-operations-intermediate.md)
- Understanding of basic joins and aggregation
- Sample datasets from `teaching_data/business_analytics/`

---

## Section 1: Complex Multi-Table Joins

Real-world analysis often requires combining data from many sources.

### The Business Case

You want to analyze how firm performance varies with macroeconomic conditions while controlling for firm characteristics. This requires:

1. **Financial data** (balance_sheet, income_statement)
2. **Company characteristics** (company_info)
3. **Economic context** (economic_indicators)

### Step-by-Step Multi-Source Join

**Chat Prompt:**
```
Load teaching_data/business_analytics/balance_sheet.csv as "balance"
Load teaching_data/business_analytics/income_statement.csv as "income"
Load teaching_data/business_analytics/company_info.csv as "companies"
Load teaching_data/business_analytics/economic_indicators.csv as "economy"
```

**Step 1: Join Financial Statements**

**Chat Prompt:**
```
Merge balance with income on firm_id, year, and quarter. Call it "financials"
```

**Step 2: Add Company Characteristics**

**Chat Prompt:**
```
Left join financials with companies on firm_id. Call it "firm_data"
```

**Step 3: Add Economic Context**

**Chat Prompt:**
```
Left join firm_data with economy on year and quarter. Call it "analysis_data"
```

**Verify the Result:**

**Chat Prompt:**
```
Show the columns in analysis_data
How many rows are in analysis_data?
Show the first 5 rows
```

### Self-Joins

Sometimes you need to join a dataset with itself. This is useful for comparing entities or creating lagged values.

**Chat Prompt:**
```
Create a dataset comparing each firm's Q4 performance to their Q1 performance
```

---

## Section 2: Reshaping Data (Wide to Long, Long to Wide)

Data comes in two common formats:

**Wide format:** Each variable is a column
```
firm_id  |  q1_revenue  |  q2_revenue  |  q3_revenue  |  q4_revenue
ACME     |  1200000     |  1350000     |  1280000     |  1450000
```

**Long format:** Each observation is a row
```
firm_id  |  quarter  |  revenue
ACME     |  Q1       |  1200000
ACME     |  Q2       |  1350000
ACME     |  Q3       |  1280000
ACME     |  Q4       |  1450000
```

### Converting Wide to Long (Melt/Unpivot)

**Chat Prompt:**
```
Reshape the data from wide to long format, keeping firm_id as identifier and creating quarter and revenue columns
```

### Converting Long to Wide (Pivot)

**Chat Prompt:**
```
Pivot the income data so each quarter becomes a column, with firm_id as rows and revenue as values
```

**Chat Prompt:**
```
Create a wide-format table showing each firm's quarterly net_income across columns
```

### When to Use Each Format

| Format | Best For |
|--------|----------|
| Long | Panel data regression, time series analysis, plotting |
| Wide | Correlation matrices, side-by-side comparisons, reporting |

---

## Section 3: Window Functions and Rolling Calculations

Window functions compute values based on a set of related rows.

### Lag and Lead Variables

Essential for panel and time series analysis.

**Chat Prompt:**
```
Add a column "prev_quarter_revenue" that shows each firm's revenue from the previous quarter
```

**Chat Prompt:**
```
Create a "revenue_growth" column as (revenue - previous revenue) / previous revenue
```

**Chat Prompt:**
```
Add a "next_quarter_revenue" column showing the following quarter's revenue
```

### Running Totals and Averages

**Chat Prompt:**
```
Calculate a running total of revenue by firm
```

**Chat Prompt:**
```
Add a 4-quarter moving average of net_income for each firm
```

### Ranking Within Groups

**Chat Prompt:**
```
Rank firms by total_assets within each quarter
```

**Chat Prompt:**
```
Add a percentile rank for ROE within each quarter
```

---

## Section 4: Data Validation and Quality Checks

Before analysis, validate your data.

### Checking for Completeness

**Chat Prompt:**
```
Check if all firms have data for all four quarters
```

**Chat Prompt:**
```
Find any firm-quarter combinations that are missing from the panel
```

### Consistency Checks

**Chat Prompt:**
```
Verify that total_assets equals current_assets plus fixed_assets for each row
```

**Chat Prompt:**
```
Check if any rows have negative values for revenue or total_assets
```

### Outlier Detection

**Chat Prompt:**
```
Find any observations where ROE is more than 3 standard deviations from the mean
```

**Chat Prompt:**
```
Show the distribution of debt_to_equity ratio and flag potential outliers
```

---

## Section 5: Building a Complete Analytical Dataset

Let's build a research-ready panel dataset step by step.

### Goal

Create a quarterly firm-level panel with:
- Financial metrics (from balance sheet and income statement)
- Calculated ratios
- Company characteristics
- Macroeconomic controls
- Lagged variables for dynamic analysis

### Complete Workflow

**Step 1: Load all source data**

**Chat Prompt:**
```
Load balance_sheet.csv as "balance"
Load income_statement.csv as "income"
Load company_info.csv as "companies"
Load economic_indicators.csv as "economy"
```

**Step 2: Create the base panel**

**Chat Prompt:**
```
Merge balance with income on firm_id, year, quarter. Call it "panel"
```

**Step 3: Calculate derived metrics**

**Chat Prompt:**
```
Add to panel:
- "roa" as net_income divided by total_assets
- "leverage" as total_liabilities divided by total_assets
- "current_ratio" as current_assets divided by current_liabilities
```

**Step 4: Add time-varying characteristics**

**Chat Prompt:**
```
Add lagged_roa (previous quarter's ROA) to panel, by firm
```

**Step 5: Merge with firm characteristics**

**Chat Prompt:**
```
Left join panel with companies on firm_id
```

**Step 6: Merge with economic conditions**

**Chat Prompt:**
```
Left join panel with economy on year and quarter
```

**Step 7: Validate the final dataset**

**Chat Prompt:**
```
Show summary statistics for the final panel dataset
Check for missing values
How many observations per firm?
```

---

## Practice Exercises

### Exercise 1: Multi-Source Join

Build an employee analysis dataset by:
1. Loading employee_data.csv and company_info.csv
2. Joining them on firm_id
3. Adding the average ROE for each firm from financial_ratios.csv (hint: aggregate first, then join)

<details>
<summary>Solution</summary>

```
Load teaching_data/business_analytics/employee_data.csv as "employees"
Load teaching_data/business_analytics/company_info.csv as "companies"
Load teaching_data/business_analytics/financial_ratios.csv as "ratios"

Calculate average roe by firm_id from ratios. Call it "firm_roe"
Merge employees with companies on firm_id
Left join the result with firm_roe on firm_id
```

</details>

### Exercise 2: Create Growth Variables

Using the income statement data:
1. Sort by firm_id and quarter
2. Create a lagged revenue variable (previous quarter)
3. Calculate quarter-over-quarter revenue growth
4. Calculate year-over-year growth (Q1 2022 to Q1 2023... though our data is only 2022, so demonstrate the concept)

<details>
<summary>Solution</summary>

```
Load teaching_data/business_analytics/income_statement.csv as "income"
Sort income by firm_id, year, quarter
Add lagged_revenue as the previous quarter's revenue for each firm
Create revenue_growth as (revenue - lagged_revenue) / lagged_revenue
```

</details>

### Exercise 3: Data Validation

Using your merged dataset:
1. Check that all firms have exactly 4 quarterly observations
2. Verify that shareholders_equity = total_assets - total_liabilities (within rounding)
3. Find any quarters where a firm's revenue decreased compared to the previous quarter

<details>
<summary>Solution</summary>

```
Count observations by firm_id and verify each has 4 rows
Create a column "equity_check" as total_assets - total_liabilities - shareholders_equity
Show rows where abs(equity_check) > 1
Show rows where revenue_growth < 0
```

</details>

### Exercise 4: Complete Panel Construction

Build a comprehensive firm-quarter panel for regression analysis:
1. Merge balance sheet, income statement, and company info
2. Add ROE, ROA, and leverage ratios
3. Create lagged ROE and lagged leverage
4. Add economic indicators
5. Validate: no missing values in key variables, all firms balanced

<details>
<summary>Solution</summary>

```
# Load all data
Load balance_sheet.csv as "balance"
Load income_statement.csv as "income"
Load company_info.csv as "companies"
Load economic_indicators.csv as "economy"

# Merge financials
Merge balance with income on firm_id, year, quarter as "panel"

# Calculate ratios
Add "roe" as net_income / shareholders_equity to panel
Add "roa" as net_income / total_assets to panel
Add "leverage" as total_liabilities / total_assets to panel

# Add lags
Add lagged_roe (previous quarter) by firm_id
Add lagged_leverage (previous quarter) by firm_id

# Merge characteristics
Left join panel with companies on firm_id
Left join result with economy on year, quarter

# Validate
Show missing value counts
Count observations by firm_id
Show summary statistics for roe, roa, leverage
```

</details>

---

## Advanced Tips

### 1. Order of Operations Matters

Join order affects results:
- Inner joins first (removes rows) - understand what you're losing
- Left joins after (preserves rows) - for reference data

### 2. Aggregation Before Joining

To avoid many-to-many problems:
```
Instead of joining daily stock data directly...
First aggregate to quarterly averages, then join with quarterly financials
```

### 3. Document Your Pipeline

Keep track of:
- Source datasets and their row counts
- Join types used and resulting row counts
- Calculated variables and their definitions
- Data quality issues found and how resolved

### 4. Creating Analysis-Ready Variables

For regression, you often need:
- **Logged variables**: `log_assets` = log(total_assets)
- **Scaled variables**: `assets_millions` = total_assets / 1000000
- **Winsorized variables**: Trim extreme outliers
- **Standardized variables**: (x - mean) / std

**Chat Prompt:**
```
Create log_revenue as the natural log of revenue
Scale total_assets to millions
Winsorize ROE at the 1st and 99th percentiles
```

---

## Key Takeaways

- **Multi-table joins**: Plan your join sequence; aggregate where needed to avoid row explosion
- **Reshaping**: Long format for regression, wide format for reporting
- **Window functions**: Essential for panel data (lags, leads, running calculations)
- **Validation**: Always check row counts, missing values, and logical consistency
- **Documentation**: Track your data pipeline for reproducibility

---

## Summary: Data Wrangling Pipeline

```
1. LOAD raw data sources

2. INSPECT each source
   - Row counts
   - Column types
   - Missing values
   - Key columns

3. CLEAN individual datasets
   - Handle missing values
   - Fix data types
   - Remove duplicates

4. JOIN datasets
   - Start with core entity table
   - Add dimensions one at a time
   - Verify row counts after each join

5. CALCULATE derived variables
   - Ratios and metrics
   - Lags and leads
   - Growth rates

6. VALIDATE final dataset
   - Completeness checks
   - Consistency checks
   - Outlier review

7. EXPORT for analysis
```

---

## What's Next?

With your cleaned, merged dataset, you're ready for:
- [Data Analysis Workflow](03-data-analysis-workflow.md) - Run regressions and visualize results
- Panel data analysis with fixed effects
- Time series forecasting
