# Intermediate Data Operations

**Difficulty: Intermediate**

## Learning Objectives

After completing this tutorial, you will be able to:
- [ ] Aggregate data using group-by operations
- [ ] Create calculated columns (derived variables)
- [ ] Perform basic joins between two datasets
- [ ] Understand different join types (inner, left, right)
- [ ] Rename and reorder columns
- [ ] Handle duplicates

## Prerequisites

- Completed [Data Munging Basics](04-data-munging-basics.md)
- Familiarity with basic filtering and selection
- Sample datasets from `teaching_data/business_analytics/`

---

## Section 1: Aggregation and Group-By Operations

Aggregation lets you summarize data by groups, computing statistics like sums, averages, and counts.

### Setup: Load the Data

**Chat Prompt:**
```
Load teaching_data/business_analytics/sales_data.csv as "sales"
Load teaching_data/business_analytics/financial_ratios.csv as "ratios"
```

### Simple Aggregation

**Chat Prompt:**
```
What is the total revenue in sales?
```

**Chat Prompt:**
```
Calculate the average advertising spend in sales
```

### Group-By Aggregation

Group-by operations split your data into groups, apply a function, and combine the results.

**Chat Prompt:**
```
Calculate the total revenue by region in sales
```

**Chat Prompt:**
```
What is the average revenue for each product in sales?
```

**Chat Prompt:**
```
Show the sum of units sold by product and region
```

### Multiple Aggregations

**Chat Prompt:**
```
For each region in sales, show the total revenue, average units, and count of records
```

### Common Aggregation Functions

| Function | Description | Example Use |
|----------|-------------|-------------|
| sum | Total of values | Total revenue |
| mean/avg | Average value | Average price |
| count | Number of rows | Number of transactions |
| min | Minimum value | Lowest sale |
| max | Maximum value | Highest sale |
| std | Standard deviation | Variability measure |

---

## Section 2: Calculated Columns

Create new columns based on existing data.

### Arithmetic Operations

**Chat Prompt:**
```
Add a column called "revenue_per_unit" to sales calculated as revenue divided by units
```

**Chat Prompt:**
```
Create a "profit_margin" column in ratios as (gross_margin - operating_margin)
```

### Conditional Columns

**Chat Prompt:**
```
Add a column "high_revenue" to sales that is true when revenue > 12000, false otherwise
```

**Chat Prompt:**
```
Create a "size_category" column in sales: "Large" if revenue > 15000, "Medium" if revenue > 10000, otherwise "Small"
```

---

## Section 3: Basic Joins (Merging Datasets)

Joins combine data from multiple datasets based on matching keys.

### Setup: Load Related Datasets

**Chat Prompt:**
```
Load teaching_data/business_analytics/company_info.csv as "companies"
Load teaching_data/business_analytics/balance_sheet.csv as "balance"
```

### Understanding Your Keys

Before joining, identify which columns link the datasets.

**Chat Prompt:**
```
Show the first few rows of companies
Show the first few rows of balance
```

You'll see that both have `firm_id` as a common column.

### Inner Join

An inner join keeps only rows where the key exists in both datasets.

**Chat Prompt:**
```
Merge balance with companies on firm_id
```

This combines the balance sheet data with company information. Only firms appearing in both datasets are included.

### Left Join

A left join keeps all rows from the first (left) dataset, filling in missing values where there's no match.

**Chat Prompt:**
```
Left join companies with balance on firm_id
```

This keeps all companies, even if they don't have balance sheet data. (In this case, DELTA and EPSILON have company info but no financial data.)

### Right Join

A right join keeps all rows from the second (right) dataset.

**Chat Prompt:**
```
Right join companies with balance on firm_id
```

### Viewing Join Results

After a join, always inspect the result:

**Chat Prompt:**
```
Show the first 10 rows of the merged data
How many rows are in the merged dataset?
```

---

## Section 4: Working with Multiple Keys

Sometimes you need to join on more than one column.

### Time-Based Joins

**Chat Prompt:**
```
Load teaching_data/business_analytics/income_statement.csv as "income"
Load teaching_data/business_analytics/economic_indicators.csv as "economy"
```

**Chat Prompt:**
```
Merge income with economy on both year and quarter
```

This matches each firm's quarterly income statement with the economic conditions at that time.

---

## Section 5: Handling Duplicates

### Identifying Duplicates

**Chat Prompt:**
```
Are there any duplicate rows in sales?
```

**Chat Prompt:**
```
Check for duplicate firm_id and quarter combinations in balance
```

### Removing Duplicates

**Chat Prompt:**
```
Remove duplicate rows from the dataset
```

**Chat Prompt:**
```
Keep only the first occurrence of each region in sales
```

---

## Section 6: Renaming and Reordering

### Renaming Columns

**Chat Prompt:**
```
Rename the column "avg_wage" to "average_wage" in the labor dataset
```

**Chat Prompt:**
```
In balance, rename total_assets to assets and total_liabilities to liabilities
```

### Reordering Columns

**Chat Prompt:**
```
Reorder balance so firm_id, year, quarter come first, followed by total_assets and shareholders_equity
```

---

## Practice Exercises

### Exercise 1: Aggregation

Using the financial_ratios.csv dataset:

1. Load the data as "ratios"
2. Calculate the average ROE for each firm
3. Find the maximum and minimum debt_to_equity ratio by firm
4. Count how many quarterly observations each firm has

<details>
<summary>Solution</summary>

```
Load teaching_data/business_analytics/financial_ratios.csv as "ratios"
Calculate average roe by firm_id in ratios
Show the max and min debt_to_equity by firm_id in ratios
Count records by firm_id in ratios
```

</details>

### Exercise 2: Calculated Columns

Using the income_statement.csv dataset:

1. Load the data as "income"
2. Create an "ebitda" column (operating_income + depreciation... but we'll use operating_income + interest_expense as a proxy)
3. Create a "tax_rate" column as (tax_expense / (net_income + tax_expense))
4. Create a "profitable" column that is true if net_income > 100000

<details>
<summary>Solution</summary>

```
Load teaching_data/business_analytics/income_statement.csv as "income"
Add a column "ebitda_proxy" to income calculated as operating_income + interest_expense
Create "tax_rate" column as tax_expense divided by (net_income + tax_expense) in income
Add column "profitable" that is true when net_income > 100000 in income
```

</details>

### Exercise 3: Joining Datasets

1. Load company_info.csv as "companies"
2. Load employee_data.csv as "employees"
3. Join employees with companies on firm_id
4. After joining, calculate the average salary by industry

<details>
<summary>Solution</summary>

```
Load teaching_data/business_analytics/company_info.csv as "companies"
Load teaching_data/business_analytics/employee_data.csv as "employees"
Merge employees with companies on firm_id
Calculate average salary by industry from the merged data
```

</details>

### Exercise 4: Multi-Key Join

1. Load balance_sheet.csv and income_statement.csv
2. Join them on firm_id, year, and quarter
3. After joining, calculate the return on assets (net_income / total_assets) for each observation

<details>
<summary>Solution</summary>

```
Load teaching_data/business_analytics/balance_sheet.csv as "balance"
Load teaching_data/business_analytics/income_statement.csv as "income"
Merge balance with income on firm_id, year, and quarter
Add column "calculated_roa" as net_income divided by total_assets in the merged data
```

</details>

---

## Common Join Pitfalls

### 1. Many-to-Many Joins (Cartesian Product)

If your key columns have duplicate values in both datasets, you can get an explosion of rows.

**Example:** If firm ACME appears 4 times in balance (Q1-Q4) and you join with a dataset where ACME appears 4 times, you could get 16 rows for ACME.

**Prevention:** Ensure at least one side of the join has unique keys, or aggregate first.

### 2. Missing Keys After Join

Always check row counts before and after joining:

**Chat Prompt:**
```
How many rows were in balance before the merge?
How many rows are in the merged dataset?
```

### 3. Column Name Conflicts

If both datasets have columns with the same name (other than the key), they'll typically get suffixes like `_x` and `_y`.

**Chat Prompt:**
```
Rename revenue_x to balance_revenue and revenue_y to income_revenue
```

---

## Key Takeaways

- **Aggregation**: Use `by` or `group by` to calculate statistics per group
- **Calculated columns**: Add new columns using arithmetic or conditional logic
- **Inner join**: Only matching rows from both datasets
- **Left join**: All rows from left dataset, matching rows from right
- **Multi-key joins**: Join on multiple columns when needed (e.g., firm + quarter)
- **Always verify**: Check row counts and preview results after operations

---

## Next Steps

Ready for advanced topics? Continue to:
- [Advanced Data Wrangling](06-data-wrangling-advanced.md) - Complex joins, reshaping, and building analytical datasets
