# Lehrdatensätze

Dieser Ordner enthält Beispieldatensätze für Tutorials und Übungen.

## Datensatz-Kategorien

### Volkswirtschaft (`economics/`)

Datensätze für volkswirtschaftliche Analysen und Ökonometrie:

| Datei | Beschreibung | Wichtige Variablen |
|-------|--------------|-------------------|
| `gdp_growth.csv` | BIP-Wachstumsdaten nach Land/Jahr | country, year, gdp_growth, inflation, unemployment |
| `labor_market.csv` | Arbeitsmarktindikatoren | region, year, employment_rate, wages, education_level |

### Business Analytics (`business_analytics/`)

Datensätze für betriebswirtschaftliche Analysen:

#### Verkauf & Marketing

| Datei | Beschreibung | Wichtige Variablen |
|-------|--------------|-------------------|
| `sales_data.csv` | Verkaufstransaktionen | date, product, region, revenue, units, advertising_spend |
| `customer_segments.csv` | Kundensegmentierungsdaten | customer_id, segment, age, income, purchase_frequency |

#### Jahresabschlüsse (Paneldaten)

Drei Unternehmen (ACME, BETA, GAMMA) mit Quartalsdaten für 2022:

| Datei | Beschreibung | Wichtige Variablen |
|-------|--------------|-------------------|
| `balance_sheet.csv` | Bilanzdaten | firm_id, year, quarter, total_assets, current_assets, cash, accounts_receivable, inventory, fixed_assets, total_liabilities, current_liabilities, long_term_debt, shareholders_equity |
| `income_statement.csv` | Erfolgsrechnung | firm_id, year, quarter, revenue, cost_of_goods_sold, gross_profit, operating_expenses, rd_expenses, sga_expenses, operating_income, interest_expense, tax_expense, net_income |
| `cash_flow.csv` | Geldflussrechnung | firm_id, year, quarter, operating_cash_flow, depreciation, changes_working_capital, capex, investing_cash_flow, debt_issued, debt_repaid, dividends_paid, financing_cash_flow, net_cash_flow |

#### Finanzanalyse

| Datei | Beschreibung | Wichtige Variablen |
|-------|--------------|-------------------|
| `financial_ratios.csv` | Berechnete Finanzkennzahlen | firm_id, year, quarter, current_ratio, quick_ratio, debt_to_equity, interest_coverage, roa, roe, gross_margin, operating_margin, net_margin, asset_turnover |
| `stock_returns.csv` | Tägliche Aktienkursdaten | firm_id, date, open_price, close_price, high, low, volume, daily_return, market_return, excess_return |

## Beispielanalysen

### Jahresabschlussanalyse
```
Lade die Bilanzdaten und berechne das durchschnittliche Verschuldungsverhältnis pro Unternehmen
```

### Regression mit Paneldaten
```
Führe eine Fixed-Effects-Regression von ROE auf debt_to_equity und asset_turnover mit firm_id als Entity durch
```

### Aktienrendite-Analyse
```
Berechne die Korrelation zwischen täglichen Renditen und Marktrenditen für jedes Unternehmen
```

## Verwendung

### Mit dem Chat-Interface

```
Lade die Bilanzdaten aus teaching_data/business_analytics/balance_sheet.csv
```

### Mit dem CLI

```bash
p2a data load teaching_data/business_analytics/balance_sheet.csv --name bilanz
p2a data load teaching_data/business_analytics/income_statement.csv --name erfolg
p2a panel fe bilanz -y shareholders_equity -x total_assets --entity firm_id
```

## Datenquellen

Dies sind synthetische Datensätze, die für Lehrzwecke erstellt wurden. Sie dienen der Demonstration verschiedener Analysetechniken ohne echte persönliche oder vertrauliche Informationen.

## Hinweise

- Alle Datensätze sind im CSV-Format mit Kopfzeilen
- Fehlende Werte werden als leere Zellen dargestellt
- Datumsangaben im Format JJJJ-MM-TT
- Numerische Werte verwenden Punkt (.) als Dezimaltrennzeichen
- Finanzdaten folgen Standard-Buchhaltungskonventionen
- Paneldaten verwenden firm_id als Entity-Identifikator und year/quarter für die Zeitdimension
