# Teaching Datasets

This folder contains sample datasets for use in tutorials and exercises.

## Dataset Categories

### Economics (`economics/`)

Datasets for economic analysis and econometrics:

| File | Description | Key Variables |
|------|-------------|---------------|
| `gdp_growth.csv` | GDP growth data by country/year | country, year, gdp_growth, inflation, unemployment |
| `labor_market.csv` | Labor market indicators | region, year, employment_rate, wages, education_level |

### Business Analytics (`business_analytics/`)

Datasets for business analysis:

#### Sales & Marketing

| File | Description | Key Variables |
|------|-------------|---------------|
| `sales_data.csv` | Sales transactions | date, product, region, revenue, units, advertising_spend |
| `customer_segments.csv` | Customer segmentation data | customer_id, segment, age, income, purchase_frequency |

#### Financial Statements (Panel Data)

Three firms (ACME, BETA, GAMMA) with quarterly data for 2022:

| File | Description | Key Variables |
|------|-------------|---------------|
| `balance_sheet.csv` | Balance sheet data | firm_id, year, quarter, total_assets, current_assets, cash, accounts_receivable, inventory, fixed_assets, total_liabilities, current_liabilities, long_term_debt, shareholders_equity |
| `income_statement.csv` | Income statement data | firm_id, year, quarter, revenue, cost_of_goods_sold, gross_profit, operating_expenses, rd_expenses, sga_expenses, operating_income, interest_expense, tax_expense, net_income |
| `cash_flow.csv` | Cash flow statement | firm_id, year, quarter, operating_cash_flow, depreciation, changes_working_capital, capex, investing_cash_flow, debt_issued, debt_repaid, dividends_paid, financing_cash_flow, net_cash_flow |

#### Financial Analysis

| File | Description | Key Variables |
|------|-------------|---------------|
| `financial_ratios.csv` | Computed financial ratios | firm_id, year, quarter, current_ratio, quick_ratio, debt_to_equity, interest_coverage, roa, roe, gross_margin, operating_margin, net_margin, asset_turnover |
| `stock_returns.csv` | Daily stock data | firm_id, date, open_price, close_price, high, low, volume, daily_return, market_return, excess_return |

#### Company & Employee Data

| File | Description | Key Variables |
|------|-------------|---------------|
| `company_info.csv` | Company characteristics | firm_id, company_name, industry, founded_year, headquarters, employee_count, is_public |
| `employee_data.csv` | Employee records | employee_id, firm_id, department, hire_date, salary, performance_score, education_level, is_manager |

#### Macroeconomic Data

| File | Description | Key Variables |
|------|-------------|---------------|
| `economic_indicators.csv` | Quarterly economic indicators | year, quarter, gdp_growth_rate, interest_rate, inflation_rate, unemployment_rate, consumer_confidence |

## Example Analyses

### Financial Statement Analysis
```
Load the balance sheet data and calculate the average debt-to-equity ratio by firm
```

### Regression with Panel Data
```
Run a fixed effects regression of ROE on debt_to_equity and asset_turnover using firm_id as the entity
```

### Stock Return Analysis
```
Calculate the correlation between daily returns and market returns for each firm
```

### Multi-Table Joins
```
Merge balance_sheet with income_statement on firm_id, year, and quarter,
then join with company_info to add industry classification
```

### Panel with Economic Controls
```
Join firm financials with economic_indicators on year and quarter
to analyze firm performance under different macroeconomic conditions
```

### Employee Analysis
```
Join employee_data with company_info and calculate average salary by industry
```

## Usage

### With the Chat Interface

```
Load the balance sheet data from teaching_data/business_analytics/balance_sheet.csv
```

### With the CLI

```bash
p2a data load teaching_data/business_analytics/balance_sheet.csv --name balance
p2a data load teaching_data/business_analytics/income_statement.csv --name income
p2a panel fe balance -y shareholders_equity -x total_assets --entity firm_id
```

## Data Sources

These are synthetic datasets created for educational purposes. They are designed to demonstrate various analytical techniques without containing real personal or proprietary information.

## Notes

- All datasets are in CSV format with headers
- Missing values are represented as empty cells
- Dates are in YYYY-MM-DD format
- Numeric values use period (.) as decimal separator
- Financial data follows standard accounting conventions
- Panel data uses firm_id as entity identifier and year/quarter for time dimension

## Related Tutorials

These datasets are used in the following tutorials:

1. **[Data Munging Basics](../tutorials/en/04-data-munging-basics.md)** - Loading, filtering, sorting
2. **[Intermediate Data Operations](../tutorials/en/05-data-operations-intermediate.md)** - Aggregation, calculated columns, basic joins
3. **[Advanced Data Wrangling](../tutorials/en/06-data-wrangling-advanced.md)** - Complex joins, reshaping, building analytical panels
