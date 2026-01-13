# Fortgeschrittene Datenoperationen

**Schwierigkeitsgrad: Fortgeschritten**

## Lernziele

Nach Abschluss dieses Tutorials können Sie:
- [ ] Daten mit Group-By-Operationen aggregieren
- [ ] Berechnete Spalten (abgeleitete Variablen) erstellen
- [ ] Grundlegende Joins zwischen zwei Datensätzen durchführen
- [ ] Verschiedene Join-Typen verstehen (Inner, Left, Right)
- [ ] Spalten umbenennen und neu anordnen
- [ ] Mit Duplikaten umgehen

## Voraussetzungen

- [Grundlagen der Datenaufbereitung](04-data-munging-basics.md) abgeschlossen
- Vertrautheit mit grundlegendem Filtern und Auswählen
- Beispieldatensätze aus `teaching_data/business_analytics/`

---

## Abschnitt 1: Aggregation und Group-By-Operationen

Aggregation ermöglicht es Ihnen, Daten nach Gruppen zusammenzufassen und Statistiken wie Summen, Durchschnitte und Anzahlen zu berechnen.

### Vorbereitung: Daten laden

**Chat-Eingabe:**
```
Lade teaching_data/business_analytics/sales_data.csv als "sales"
Lade teaching_data/business_analytics/financial_ratios.csv als "ratios"
```

### Einfache Aggregation

**Chat-Eingabe:**
```
Was ist der Gesamtumsatz in sales?
```

**Chat-Eingabe:**
```
Berechne die durchschnittlichen Werbeausgaben in sales
```

### Group-By-Aggregation

Group-By-Operationen teilen Ihre Daten in Gruppen auf, wenden eine Funktion an und kombinieren die Ergebnisse.

**Chat-Eingabe:**
```
Berechne den Gesamtumsatz nach Region in sales
```

**Chat-Eingabe:**
```
Was ist der durchschnittliche Umsatz für jedes Produkt in sales?
```

**Chat-Eingabe:**
```
Zeige die Summe der verkauften Einheiten nach Produkt und Region
```

### Mehrfache Aggregationen

**Chat-Eingabe:**
```
Für jede Region in sales, zeige den Gesamtumsatz, durchschnittliche Einheiten und Anzahl der Datensätze
```

### Häufige Aggregationsfunktionen

| Funktion | Beschreibung | Beispielverwendung |
|----------|--------------|-------------------|
| sum | Summe der Werte | Gesamtumsatz |
| mean/avg | Durchschnittswert | Durchschnittspreis |
| count | Anzahl der Zeilen | Anzahl Transaktionen |
| min | Minimalwert | Niedrigster Verkauf |
| max | Maximalwert | Höchster Verkauf |
| std | Standardabweichung | Variabilitätsmass |

---

## Abschnitt 2: Berechnete Spalten

Erstellen Sie neue Spalten basierend auf bestehenden Daten.

### Arithmetische Operationen

**Chat-Eingabe:**
```
Füge eine Spalte namens "revenue_per_unit" zu sales hinzu, berechnet als revenue geteilt durch units
```

**Chat-Eingabe:**
```
Erstelle eine "profit_margin"-Spalte in ratios als (gross_margin - operating_margin)
```

### Bedingte Spalten

**Chat-Eingabe:**
```
Füge eine Spalte "high_revenue" zu sales hinzu, die true ist wenn revenue > 12000, sonst false
```

**Chat-Eingabe:**
```
Erstelle eine "size_category"-Spalte in sales: "Large" wenn revenue > 15000, "Medium" wenn revenue > 10000, sonst "Small"
```

---

## Abschnitt 3: Grundlegende Joins (Datensätze zusammenführen)

Joins kombinieren Daten aus mehreren Datensätzen basierend auf übereinstimmenden Schlüsseln.

### Vorbereitung: Verwandte Datensätze laden

**Chat-Eingabe:**
```
Lade teaching_data/business_analytics/company_info.csv als "companies"
Lade teaching_data/business_analytics/balance_sheet.csv als "balance"
```

### Ihre Schlüssel verstehen

Bevor Sie joinen, identifizieren Sie, welche Spalten die Datensätze verbinden.

**Chat-Eingabe:**
```
Zeige die ersten Zeilen von companies
Zeige die ersten Zeilen von balance
```

Sie werden sehen, dass beide `firm_id` als gemeinsame Spalte haben.

### Inner Join

Ein Inner Join behält nur Zeilen, wo der Schlüssel in beiden Datensätzen existiert.

**Chat-Eingabe:**
```
Führe balance mit companies zusammen über firm_id
```

Dies kombiniert die Bilanzdaten mit Unternehmensinformationen. Nur Firmen, die in beiden Datensätzen vorkommen, werden eingeschlossen.

### Left Join

Ein Left Join behält alle Zeilen aus dem ersten (linken) Datensatz und füllt fehlende Werte aus, wo es keine Übereinstimmung gibt.

**Chat-Eingabe:**
```
Left Join von companies mit balance über firm_id
```

Dies behält alle Unternehmen, auch wenn sie keine Bilanzdaten haben. (In diesem Fall haben DELTA und EPSILON Unternehmensinformationen, aber keine Finanzdaten.)

### Right Join

Ein Right Join behält alle Zeilen aus dem zweiten (rechten) Datensatz.

**Chat-Eingabe:**
```
Right Join von companies mit balance über firm_id
```

### Join-Ergebnisse anzeigen

Nach einem Join immer das Ergebnis prüfen:

**Chat-Eingabe:**
```
Zeige die ersten 10 Zeilen der zusammengeführten Daten
Wie viele Zeilen sind im zusammengeführten Datensatz?
```

---

## Abschnitt 4: Arbeiten mit mehreren Schlüsseln

Manchmal müssen Sie über mehr als eine Spalte joinen.

### Zeitbasierte Joins

**Chat-Eingabe:**
```
Lade teaching_data/business_analytics/income_statement.csv als "income"
Lade teaching_data/business_analytics/economic_indicators.csv als "economy"
```

**Chat-Eingabe:**
```
Führe income mit economy zusammen über year und quarter
```

Dies ordnet die quartalsweise Gewinn- und Verlustrechnung jeder Firma den wirtschaftlichen Bedingungen zu diesem Zeitpunkt zu.

---

## Abschnitt 5: Mit Duplikaten umgehen

### Duplikate identifizieren

**Chat-Eingabe:**
```
Gibt es doppelte Zeilen in sales?
```

**Chat-Eingabe:**
```
Prüfe auf doppelte firm_id und quarter Kombinationen in balance
```

### Duplikate entfernen

**Chat-Eingabe:**
```
Entferne doppelte Zeilen aus dem Datensatz
```

**Chat-Eingabe:**
```
Behalte nur das erste Vorkommen jeder Region in sales
```

---

## Abschnitt 6: Umbenennen und Neuordnen

### Spalten umbenennen

**Chat-Eingabe:**
```
Benenne die Spalte "avg_wage" um in "average_wage" im labor-Datensatz
```

**Chat-Eingabe:**
```
In balance, benenne total_assets um in assets und total_liabilities in liabilities
```

### Spalten neuordnen

**Chat-Eingabe:**
```
Ordne balance neu, sodass firm_id, year, quarter zuerst kommen, gefolgt von total_assets und shareholders_equity
```

---

## Übungsaufgaben

### Übung 1: Aggregation

Mit dem financial_ratios.csv-Datensatz:

1. Lade die Daten als "ratios"
2. Berechne die durchschnittliche ROE für jede Firma
3. Finde das Maximum und Minimum des debt_to_equity-Verhältnisses nach Firma
4. Zähle, wie viele Quartalsbeobachtungen jede Firma hat

<details>
<summary>Lösung</summary>

```
Lade teaching_data/business_analytics/financial_ratios.csv als "ratios"
Berechne durchschnittliche roe nach firm_id in ratios
Zeige max und min debt_to_equity nach firm_id in ratios
Zähle Datensätze nach firm_id in ratios
```

</details>

### Übung 2: Berechnete Spalten

Mit dem income_statement.csv-Datensatz:

1. Lade die Daten als "income"
2. Erstelle eine "ebitda"-Spalte (operating_income + Abschreibungen... wir verwenden operating_income + interest_expense als Näherung)
3. Erstelle eine "tax_rate"-Spalte als (tax_expense / (net_income + tax_expense))
4. Erstelle eine "profitable"-Spalte, die true ist wenn net_income > 100000

<details>
<summary>Lösung</summary>

```
Lade teaching_data/business_analytics/income_statement.csv als "income"
Füge eine Spalte "ebitda_proxy" zu income hinzu, berechnet als operating_income + interest_expense
Erstelle "tax_rate"-Spalte als tax_expense geteilt durch (net_income + tax_expense) in income
Füge Spalte "profitable" hinzu, die true ist wenn net_income > 100000 in income
```

</details>

### Übung 3: Datensätze zusammenführen

1. Lade company_info.csv als "companies"
2. Lade employee_data.csv als "employees"
3. Führe employees mit companies über firm_id zusammen
4. Nach dem Zusammenführen, berechne das durchschnittliche Gehalt nach Branche

<details>
<summary>Lösung</summary>

```
Lade teaching_data/business_analytics/company_info.csv als "companies"
Lade teaching_data/business_analytics/employee_data.csv als "employees"
Führe employees mit companies zusammen über firm_id
Berechne durchschnittliches Gehalt nach industry aus den zusammengeführten Daten
```

</details>

### Übung 4: Multi-Key-Join

1. Lade balance_sheet.csv und income_statement.csv
2. Führe sie zusammen über firm_id, year und quarter
3. Nach dem Zusammenführen, berechne die Gesamtkapitalrendite (net_income / total_assets) für jede Beobachtung

<details>
<summary>Lösung</summary>

```
Lade teaching_data/business_analytics/balance_sheet.csv als "balance"
Lade teaching_data/business_analytics/income_statement.csv als "income"
Führe balance mit income zusammen über firm_id, year und quarter
Füge Spalte "calculated_roa" hinzu als net_income geteilt durch total_assets in den zusammengeführten Daten
```

</details>

---

## Häufige Join-Fallstricke

### 1. Many-to-Many-Joins (Kartesisches Produkt)

Wenn Ihre Schlüsselspalten doppelte Werte in beiden Datensätzen haben, können Sie eine Explosion von Zeilen erhalten.

**Beispiel:** Wenn Firma ACME 4 mal in balance vorkommt (Q1-Q4) und Sie mit einem Datensatz joinen, wo ACME 4 mal vorkommt, könnten Sie 16 Zeilen für ACME erhalten.

**Prävention:** Stellen Sie sicher, dass mindestens eine Seite des Joins eindeutige Schlüssel hat, oder aggregieren Sie zuerst.

### 2. Fehlende Schlüssel nach Join

Prüfen Sie immer die Zeilenanzahl vor und nach dem Join:

**Chat-Eingabe:**
```
Wie viele Zeilen waren in balance vor dem Zusammenführen?
Wie viele Zeilen sind im zusammengeführten Datensatz?
```

### 3. Spaltennamen-Konflikte

Wenn beide Datensätze Spalten mit dem gleichen Namen haben (ausser dem Schlüssel), erhalten sie typischerweise Suffixe wie `_x` und `_y`.

**Chat-Eingabe:**
```
Benenne revenue_x um in balance_revenue und revenue_y in income_revenue
```

---

## Wichtigste Erkenntnisse

- **Aggregation**: Verwenden Sie `nach` oder `gruppiert nach`, um Statistiken pro Gruppe zu berechnen
- **Berechnete Spalten**: Fügen Sie neue Spalten mit arithmetischer oder bedingter Logik hinzu
- **Inner Join**: Nur übereinstimmende Zeilen aus beiden Datensätzen
- **Left Join**: Alle Zeilen aus dem linken Datensatz, übereinstimmende Zeilen aus dem rechten
- **Multi-Key-Joins**: Über mehrere Spalten joinen wenn nötig (z.B. Firma + Quartal)
- **Immer verifizieren**: Prüfen Sie Zeilenanzahlen und schauen Sie Ergebnisse nach Operationen an

---

## Nächste Schritte

Bereit für fortgeschrittene Themen? Fahren Sie fort mit:
- [Fortgeschrittenes Data Wrangling](06-data-wrangling-advanced.md) - Komplexe Joins, Umstrukturierung und Aufbau analytischer Datensätze
