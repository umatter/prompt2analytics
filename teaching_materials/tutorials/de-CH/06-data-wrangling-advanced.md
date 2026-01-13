# Fortgeschrittenes Data Wrangling

**Schwierigkeitsgrad: Fortgeschritten**

## Lernziele

Nach Abschluss dieses Tutorials können Sie:
- [ ] Komplexe Multi-Table-Joins durchführen
- [ ] Daten zwischen Wide- und Long-Format umstrukturieren
- [ ] Window-Funktionen für laufende Berechnungen verwenden
- [ ] Vollständige analytische Datensätze aus mehreren Quellen aufbauen
- [ ] Datenqualitätsprüfungen und Validierung anwenden
- [ ] Lag- und Lead-Variablen für Panel-/Zeitreihendaten erstellen

## Voraussetzungen

- [Grundlagen der Datenaufbereitung](04-data-munging-basics.md) abgeschlossen
- [Fortgeschrittene Datenoperationen](05-data-operations-intermediate.md) abgeschlossen
- Verständnis von grundlegenden Joins und Aggregation
- Beispieldatensätze aus `teaching_data/business_analytics/`

---

## Abschnitt 1: Komplexe Multi-Table-Joins

Reale Analysen erfordern oft die Kombination von Daten aus vielen Quellen.

### Der Business Case

Sie möchten analysieren, wie die Unternehmensleistung mit makroökonomischen Bedingungen variiert, unter Kontrolle für Unternehmensmerkmale. Dies erfordert:

1. **Finanzdaten** (balance_sheet, income_statement)
2. **Unternehmensmerkmale** (company_info)
3. **Wirtschaftlicher Kontext** (economic_indicators)

### Schritt-für-Schritt Multi-Source-Join

**Chat-Eingabe:**
```
Lade teaching_data/business_analytics/balance_sheet.csv als "balance"
Lade teaching_data/business_analytics/income_statement.csv als "income"
Lade teaching_data/business_analytics/company_info.csv als "companies"
Lade teaching_data/business_analytics/economic_indicators.csv als "economy"
```

**Schritt 1: Finanzberichte zusammenführen**

**Chat-Eingabe:**
```
Führe balance mit income zusammen über firm_id, year und quarter. Nenne es "financials"
```

**Schritt 2: Unternehmensmerkmale hinzufügen**

**Chat-Eingabe:**
```
Left Join von financials mit companies über firm_id. Nenne es "firm_data"
```

**Schritt 3: Wirtschaftlichen Kontext hinzufügen**

**Chat-Eingabe:**
```
Left Join von firm_data mit economy über year und quarter. Nenne es "analysis_data"
```

**Ergebnis verifizieren:**

**Chat-Eingabe:**
```
Zeige die Spalten in analysis_data
Wie viele Zeilen sind in analysis_data?
Zeige die ersten 5 Zeilen
```

### Self-Joins

Manchmal müssen Sie einen Datensatz mit sich selbst joinen. Dies ist nützlich zum Vergleich von Entitäten oder zur Erstellung von verzögerten Werten.

**Chat-Eingabe:**
```
Erstelle einen Datensatz, der die Q4-Leistung jeder Firma mit ihrer Q1-Leistung vergleicht
```

---

## Abschnitt 2: Daten umstrukturieren (Wide zu Long, Long zu Wide)

Daten kommen in zwei gängigen Formaten:

**Wide-Format:** Jede Variable ist eine Spalte
```
firm_id  |  q1_revenue  |  q2_revenue  |  q3_revenue  |  q4_revenue
ACME     |  1200000     |  1350000     |  1280000     |  1450000
```

**Long-Format:** Jede Beobachtung ist eine Zeile
```
firm_id  |  quarter  |  revenue
ACME     |  Q1       |  1200000
ACME     |  Q2       |  1350000
ACME     |  Q3       |  1280000
ACME     |  Q4       |  1450000
```

### Wide zu Long konvertieren (Melt/Unpivot)

**Chat-Eingabe:**
```
Strukturiere die Daten von Wide zu Long um, behalte firm_id als Identifikator und erstelle quarter- und revenue-Spalten
```

### Long zu Wide konvertieren (Pivot)

**Chat-Eingabe:**
```
Pivotiere die income-Daten, sodass jedes Quartal eine Spalte wird, mit firm_id als Zeilen und revenue als Werte
```

**Chat-Eingabe:**
```
Erstelle eine Wide-Format-Tabelle, die das quartalsweise net_income jeder Firma über Spalten zeigt
```

### Wann welches Format verwenden

| Format | Am besten für |
|--------|---------------|
| Long | Panel-Daten-Regression, Zeitreihenanalyse, Plotting |
| Wide | Korrelationsmatrizen, Seite-an-Seite-Vergleiche, Reporting |

---

## Abschnitt 3: Window-Funktionen und laufende Berechnungen

Window-Funktionen berechnen Werte basierend auf einer Menge verwandter Zeilen.

### Lag- und Lead-Variablen

Essentiell für Panel- und Zeitreihenanalyse.

**Chat-Eingabe:**
```
Füge eine Spalte "prev_quarter_revenue" hinzu, die den Umsatz jeder Firma vom vorherigen Quartal zeigt
```

**Chat-Eingabe:**
```
Erstelle eine "revenue_growth"-Spalte als (revenue - vorheriger Umsatz) / vorheriger Umsatz
```

**Chat-Eingabe:**
```
Füge eine "next_quarter_revenue"-Spalte hinzu, die den Umsatz des folgenden Quartals zeigt
```

### Laufende Summen und Durchschnitte

**Chat-Eingabe:**
```
Berechne eine laufende Summe des Umsatzes nach Firma
```

**Chat-Eingabe:**
```
Füge einen gleitenden 4-Quartals-Durchschnitt des net_income für jede Firma hinzu
```

### Rangbildung innerhalb von Gruppen

**Chat-Eingabe:**
```
Rangiere Firmen nach total_assets innerhalb jedes Quartals
```

**Chat-Eingabe:**
```
Füge einen Perzentilrang für ROE innerhalb jedes Quartals hinzu
```

---

## Abschnitt 4: Datenvalidierung und Qualitätsprüfungen

Vor der Analyse validieren Sie Ihre Daten.

### Auf Vollständigkeit prüfen

**Chat-Eingabe:**
```
Prüfe, ob alle Firmen Daten für alle vier Quartale haben
```

**Chat-Eingabe:**
```
Finde alle Firma-Quartal-Kombinationen, die im Panel fehlen
```

### Konsistenzprüfungen

**Chat-Eingabe:**
```
Verifiziere, dass total_assets gleich current_assets plus fixed_assets für jede Zeile ist
```

**Chat-Eingabe:**
```
Prüfe, ob irgendwelche Zeilen negative Werte für revenue oder total_assets haben
```

### Ausreisser-Erkennung

**Chat-Eingabe:**
```
Finde alle Beobachtungen, wo ROE mehr als 3 Standardabweichungen vom Mittelwert entfernt ist
```

**Chat-Eingabe:**
```
Zeige die Verteilung des debt_to_equity-Verhältnisses und markiere potentielle Ausreisser
```

---

## Abschnitt 5: Aufbau eines vollständigen analytischen Datensatzes

Lassen Sie uns Schritt für Schritt einen forschungsfertigen Panel-Datensatz aufbauen.

### Ziel

Erstellen Sie ein quartalsweises Firmen-Panel mit:
- Finanzkennzahlen (aus Bilanz und Gewinn- und Verlustrechnung)
- Berechneten Kennzahlen
- Unternehmensmerkmalen
- Makroökonomischen Kontrollvariablen
- Verzögerten Variablen für dynamische Analyse

### Vollständiger Workflow

**Schritt 1: Alle Quelldaten laden**

**Chat-Eingabe:**
```
Lade balance_sheet.csv als "balance"
Lade income_statement.csv als "income"
Lade company_info.csv als "companies"
Lade economic_indicators.csv als "economy"
```

**Schritt 2: Das Basis-Panel erstellen**

**Chat-Eingabe:**
```
Führe balance mit income zusammen über firm_id, year, quarter. Nenne es "panel"
```

**Schritt 3: Abgeleitete Metriken berechnen**

**Chat-Eingabe:**
```
Füge zu panel hinzu:
- "roa" als net_income geteilt durch total_assets
- "leverage" als total_liabilities geteilt durch total_assets
- "current_ratio" als current_assets geteilt durch current_liabilities
```

**Schritt 4: Zeitvariierende Merkmale hinzufügen**

**Chat-Eingabe:**
```
Füge lagged_roa (ROA des vorherigen Quartals) zu panel hinzu, nach Firma
```

**Schritt 5: Mit Unternehmensmerkmalen zusammenführen**

**Chat-Eingabe:**
```
Left Join von panel mit companies über firm_id
```

**Schritt 6: Mit wirtschaftlichen Bedingungen zusammenführen**

**Chat-Eingabe:**
```
Left Join von panel mit economy über year und quarter
```

**Schritt 7: Den finalen Datensatz validieren**

**Chat-Eingabe:**
```
Zeige zusammenfassende Statistiken für den finalen Panel-Datensatz
Prüfe auf fehlende Werte
Wie viele Beobachtungen pro Firma?
```

---

## Übungsaufgaben

### Übung 1: Multi-Source-Join

Bauen Sie einen Mitarbeiteranalyse-Datensatz auf durch:
1. Laden von employee_data.csv und company_info.csv
2. Zusammenführen über firm_id
3. Hinzufügen der durchschnittlichen ROE für jede Firma aus financial_ratios.csv (Hinweis: erst aggregieren, dann joinen)

<details>
<summary>Lösung</summary>

```
Lade teaching_data/business_analytics/employee_data.csv als "employees"
Lade teaching_data/business_analytics/company_info.csv als "companies"
Lade teaching_data/business_analytics/financial_ratios.csv als "ratios"

Berechne durchschnittliche roe nach firm_id aus ratios. Nenne es "firm_roe"
Führe employees mit companies zusammen über firm_id
Left Join des Ergebnisses mit firm_roe über firm_id
```

</details>

### Übung 2: Wachstumsvariablen erstellen

Mit den Gewinn- und Verlustrechnungsdaten:
1. Sortiere nach firm_id und quarter
2. Erstelle eine verzögerte Umsatzvariable (vorheriges Quartal)
3. Berechne das Quartal-zu-Quartal-Umsatzwachstum
4. Berechne das Jahr-zu-Jahr-Wachstum (Q1 2022 zu Q1 2023... obwohl unsere Daten nur 2022 sind, demonstrieren Sie das Konzept)

<details>
<summary>Lösung</summary>

```
Lade teaching_data/business_analytics/income_statement.csv als "income"
Sortiere income nach firm_id, year, quarter
Füge lagged_revenue als den Umsatz des vorherigen Quartals für jede Firma hinzu
Erstelle revenue_growth als (revenue - lagged_revenue) / lagged_revenue
```

</details>

### Übung 3: Datenvalidierung

Mit Ihrem zusammengeführten Datensatz:
1. Prüfe, dass alle Firmen genau 4 Quartalsbeobachtungen haben
2. Verifiziere, dass shareholders_equity = total_assets - total_liabilities (innerhalb der Rundung)
3. Finde alle Quartale, in denen der Umsatz einer Firma im Vergleich zum Vorquartal gesunken ist

<details>
<summary>Lösung</summary>

```
Zähle Beobachtungen nach firm_id und verifiziere, dass jede 4 Zeilen hat
Erstelle eine Spalte "equity_check" als total_assets - total_liabilities - shareholders_equity
Zeige Zeilen, wo abs(equity_check) > 1
Zeige Zeilen, wo revenue_growth < 0
```

</details>

### Übung 4: Vollständige Panel-Konstruktion

Bauen Sie ein umfassendes Firma-Quartal-Panel für Regressionsanalyse:
1. Führe Bilanz, Gewinn- und Verlustrechnung und Unternehmensinformationen zusammen
2. Füge ROE, ROA und Leverage-Verhältnisse hinzu
3. Erstelle verzögerte ROE und verzögertes Leverage
4. Füge Wirtschaftsindikatoren hinzu
5. Validiere: keine fehlenden Werte in Schlüsselvariablen, alle Firmen ausbalanciert

<details>
<summary>Lösung</summary>

```
# Alle Daten laden
Lade balance_sheet.csv als "balance"
Lade income_statement.csv als "income"
Lade company_info.csv als "companies"
Lade economic_indicators.csv als "economy"

# Finanzdaten zusammenführen
Führe balance mit income zusammen über firm_id, year, quarter als "panel"

# Kennzahlen berechnen
Füge "roe" hinzu als net_income / shareholders_equity zu panel
Füge "roa" hinzu als net_income / total_assets zu panel
Füge "leverage" hinzu als total_liabilities / total_assets zu panel

# Verzögerungen hinzufügen
Füge lagged_roe (vorheriges Quartal) nach firm_id hinzu
Füge lagged_leverage (vorheriges Quartal) nach firm_id hinzu

# Merkmale zusammenführen
Left Join von panel mit companies über firm_id
Left Join des Ergebnisses mit economy über year, quarter

# Validieren
Zeige Anzahl fehlender Werte
Zähle Beobachtungen nach firm_id
Zeige zusammenfassende Statistiken für roe, roa, leverage
```

</details>

---

## Fortgeschrittene Tipps

### 1. Reihenfolge der Operationen ist wichtig

Join-Reihenfolge beeinflusst Ergebnisse:
- Inner Joins zuerst (entfernen Zeilen) - verstehen Sie, was Sie verlieren
- Left Joins danach (bewahren Zeilen) - für Referenzdaten

### 2. Aggregation vor dem Joinen

Um Many-to-Many-Probleme zu vermeiden:
```
Anstatt tägliche Aktiendaten direkt zu joinen...
Zuerst auf Quartalsdurchschnitte aggregieren, dann mit quartalsweisen Finanzdaten joinen
```

### 3. Dokumentieren Sie Ihre Pipeline

Behalten Sie im Blick:
- Quelldatensätze und ihre Zeilenanzahlen
- Verwendete Join-Typen und resultierende Zeilenanzahlen
- Berechnete Variablen und ihre Definitionen
- Gefundene Datenqualitätsprobleme und wie sie gelöst wurden

### 4. Analyse-bereite Variablen erstellen

Für Regression benötigen Sie oft:
- **Logarithmierte Variablen**: `log_assets` = log(total_assets)
- **Skalierte Variablen**: `assets_millions` = total_assets / 1000000
- **Winsorisierte Variablen**: Extreme Ausreisser trimmen
- **Standardisierte Variablen**: (x - Mittelwert) / Standardabweichung

**Chat-Eingabe:**
```
Erstelle log_revenue als den natürlichen Logarithmus von revenue
Skaliere total_assets auf Millionen
Winsorisiere ROE beim 1. und 99. Perzentil
```

---

## Wichtigste Erkenntnisse

- **Multi-Table-Joins**: Planen Sie Ihre Join-Sequenz; aggregieren Sie wo nötig, um Zeilenexplosion zu vermeiden
- **Umstrukturierung**: Long-Format für Regression, Wide-Format für Reporting
- **Window-Funktionen**: Essentiell für Panel-Daten (Lags, Leads, laufende Berechnungen)
- **Validierung**: Prüfen Sie immer Zeilenanzahlen, fehlende Werte und logische Konsistenz
- **Dokumentation**: Verfolgen Sie Ihre Daten-Pipeline für Reproduzierbarkeit

---

## Zusammenfassung: Data-Wrangling-Pipeline

```
1. LADEN der Rohdatenquellen

2. INSPIZIEREN jeder Quelle
   - Zeilenanzahlen
   - Spaltentypen
   - Fehlende Werte
   - Schlüsselspalten

3. BEREINIGEN einzelner Datensätze
   - Fehlende Werte behandeln
   - Datentypen korrigieren
   - Duplikate entfernen

4. ZUSAMMENFÜHREN der Datensätze
   - Mit Kern-Entitätstabelle beginnen
   - Dimensionen einzeln hinzufügen
   - Zeilenanzahlen nach jedem Join verifizieren

5. BERECHNEN abgeleiteter Variablen
   - Verhältnisse und Metriken
   - Lags und Leads
   - Wachstumsraten

6. VALIDIEREN des finalen Datensatzes
   - Vollständigkeitsprüfungen
   - Konsistenzprüfungen
   - Ausreisser-Überprüfung

7. EXPORTIEREN für Analyse
```

---

## Was kommt als Nächstes?

Mit Ihrem bereinigten, zusammengeführten Datensatz sind Sie bereit für:
- [Data-Analyse-Workflow](03-data-analysis-workflow.md) - Regressionen durchführen und Ergebnisse visualisieren
- Panel-Daten-Analyse mit Fixed Effects
- Zeitreihen-Prognose
