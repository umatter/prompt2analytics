# Data Munging Basics

**Difficulty: Beginner**

## Learning Objectives

After completing this tutorial, you will be able to:
- [ ] Load CSV data into the analytics environment
- [ ] View and inspect dataset structure
- [ ] Select specific columns
- [ ] Filter rows based on simple conditions
- [ ] Sort data by one or more columns
- [ ] Handle basic missing value scenarios

## Prerequisites

- Access to the p2a chat interface
- Sample datasets from `teaching_data/business_analytics/`

## Introduction

Data munging (also called data wrangling) is the process of transforming raw data into a format suitable for analysis. Before you can run regressions or build visualizations, you need clean, well-structured data.

This tutorial covers the fundamentals using the chat interface. You'll learn to express data operations in plain English and let the system handle the technical details.

---

## Section 1: Loading Data

### Your First Dataset

Let's start by loading a simple sales dataset.

**Chat Prompt:**
```
Load the data from teaching_data/business_analytics/sales_data.csv and call it "sales"
```

The system will confirm the data is loaded and show basic information like row count and column names.

### Viewing Your Data

Once loaded, you can inspect what you're working with.

**Chat Prompt:**
```
Show me the first 10 rows of sales
```

**Chat Prompt:**
```
What columns are in the sales dataset?
```

**Chat Prompt:**
```
Give me summary statistics for sales
```

### Understanding the Output

The summary statistics will show you:
- **Count**: Number of non-missing values
- **Mean**: Average value for numeric columns
- **Std**: Standard deviation (spread of the data)
- **Min/Max**: Range of values
- **Quartiles**: 25th, 50th (median), and 75th percentiles

---

## Section 2: Selecting Columns

Often you only need a subset of columns for your analysis.

### Selecting Specific Columns

**Chat Prompt:**
```
From sales, show me only the date, region, and revenue columns
```

**Chat Prompt:**
```
Create a new dataset called "sales_subset" with just product, units, and revenue from sales
```

### Excluding Columns

**Chat Prompt:**
```
Show sales data without the advertising_spend column
```

---

## Section 3: Filtering Rows

Filtering lets you focus on specific subsets of your data.

### Simple Conditions

**Chat Prompt:**
```
Show me all rows from sales where region equals "North"
```

**Chat Prompt:**
```
Filter sales to only include rows where revenue is greater than 10000
```

**Chat Prompt:**
```
Get all sales data for Widget A
```

### Combining Conditions

You can combine multiple conditions using "and" or "or".

**Chat Prompt:**
```
Show sales where region is "North" and revenue is greater than 12000
```

**Chat Prompt:**
```
Filter sales where product is "Widget A" or product is "Widget B"
```

### Negation

**Chat Prompt:**
```
Show all sales that are not in the South region
```

---

## Section 4: Sorting Data

Sorting helps you identify patterns and extremes in your data.

### Basic Sorting

**Chat Prompt:**
```
Sort sales by revenue in descending order
```

**Chat Prompt:**
```
Show sales ordered by date
```

### Multi-Column Sorting

**Chat Prompt:**
```
Sort sales by region first, then by revenue within each region
```

---

## Section 5: Basic Data Inspection

### Checking for Missing Values

**Chat Prompt:**
```
Are there any missing values in the sales dataset?
```

**Chat Prompt:**
```
Count the missing values in each column of sales
```

### Unique Values

**Chat Prompt:**
```
What are the unique products in the sales data?
```

**Chat Prompt:**
```
How many unique regions are there in sales?
```

### Value Counts

**Chat Prompt:**
```
How many sales records are there for each region?
```

**Chat Prompt:**
```
Show the count of sales by product
```

---

## Practice Exercises

### Exercise 1: Load and Explore

1. Load `teaching_data/business_analytics/customer_segments.csv` as "customers"
2. View the first 5 rows
3. Get summary statistics
4. List all unique customer segments

<details>
<summary>Solution</summary>

```
Load teaching_data/business_analytics/customer_segments.csv as "customers"
Show the first 5 rows of customers
Give me summary statistics for customers
What are the unique segments in customers?
```

</details>

### Exercise 2: Filtering

Using the sales dataset:

1. Find all sales where units sold is greater than 100
2. Find all Widget A sales in the North region
3. Find sales where advertising spend is between 1500 and 2000

<details>
<summary>Solution</summary>

```
Show sales where units is greater than 100
Filter sales where product is "Widget A" and region is "North"
Show sales where advertising_spend is between 1500 and 2000
```

</details>

### Exercise 3: Sorting and Selection

1. Sort sales by advertising_spend in ascending order
2. Show only the top 5 sales by revenue
3. Create a subset with date, product, and revenue, sorted by date

<details>
<summary>Solution</summary>

```
Sort sales by advertising_spend ascending
Show the top 5 rows of sales sorted by revenue descending
Create a dataset with date, product, and revenue from sales, sorted by date
```

</details>

---

## Common Mistakes to Avoid

1. **Forgetting to load data first**: Always load your CSV before trying to work with it

2. **Case sensitivity**: Column names and string values may be case-sensitive
   - Use exact spelling: "North" not "north"

3. **Numeric vs. string comparisons**:
   - For numbers: `revenue > 10000`
   - For text: `region equals "North"` (use quotes)

4. **Overwriting data**: If you filter data, save it to a new name if you need the original later

---

## Key Takeaways

- **Loading**: Use `load` with the file path and give your dataset a name
- **Viewing**: Use `show`, `first N rows`, or `summary statistics`
- **Selecting**: Specify columns by name to create subsets
- **Filtering**: Use conditions like `equals`, `greater than`, `between`
- **Sorting**: Use `sort by` with optional `ascending` or `descending`

---

## Next Steps

Ready for more? Continue to:
- [Intermediate Data Operations](05-data-operations-intermediate.md) - Learn about aggregation and basic merging
- [Advanced Data Wrangling](06-data-wrangling-advanced.md) - Master complex joins and reshaping
