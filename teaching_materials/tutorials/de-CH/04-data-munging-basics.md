# Grundlagen der Datenaufbereitung

**Schwierigkeitsgrad: Anfänger**

## Lernziele

Nach Abschluss dieses Tutorials können Sie:
- [ ] CSV-Daten in die Analyseumgebung laden
- [ ] Datensatzstruktur anzeigen und untersuchen
- [ ] Bestimmte Spalten auswählen
- [ ] Zeilen nach einfachen Bedingungen filtern
- [ ] Daten nach einer oder mehreren Spalten sortieren
- [ ] Grundlegende Szenarien mit fehlenden Werten behandeln

## Voraussetzungen

- Zugang zur p2a Chat-Oberfläche
- Beispieldatensätze aus `teaching_data/business_analytics/`

## Einführung

Datenaufbereitung (auch Data Wrangling genannt) ist der Prozess der Umwandlung von Rohdaten in ein für die Analyse geeignetes Format. Bevor Sie Regressionen durchführen oder Visualisierungen erstellen können, benötigen Sie saubere, gut strukturierte Daten.

Dieses Tutorial behandelt die Grundlagen mit der Chat-Oberfläche. Sie lernen, Datenoperationen in natürlicher Sprache auszudrücken, und das System kümmert sich um die technischen Details.

---

## Abschnitt 1: Daten laden

### Ihr erster Datensatz

Beginnen wir mit dem Laden eines einfachen Verkaufsdatensatzes.

**Chat-Eingabe:**
```
Lade die Daten aus teaching_data/business_analytics/sales_data.csv und nenne sie "sales"
```

Das System bestätigt, dass die Daten geladen wurden, und zeigt grundlegende Informationen wie Zeilenanzahl und Spaltennamen.

### Daten anzeigen

Nach dem Laden können Sie untersuchen, womit Sie arbeiten.

**Chat-Eingabe:**
```
Zeige mir die ersten 10 Zeilen von sales
```

**Chat-Eingabe:**
```
Welche Spalten sind im sales-Datensatz?
```

**Chat-Eingabe:**
```
Gib mir zusammenfassende Statistiken für sales
```

### Die Ausgabe verstehen

Die zusammenfassenden Statistiken zeigen Ihnen:
- **Count**: Anzahl der nicht-fehlenden Werte
- **Mean**: Durchschnittswert für numerische Spalten
- **Std**: Standardabweichung (Streuung der Daten)
- **Min/Max**: Wertebereich
- **Quartile**: 25., 50. (Median) und 75. Perzentil

---

## Abschnitt 2: Spalten auswählen

Oft benötigen Sie nur eine Teilmenge der Spalten für Ihre Analyse.

### Bestimmte Spalten auswählen

**Chat-Eingabe:**
```
Zeige mir von sales nur die Spalten date, region und revenue
```

**Chat-Eingabe:**
```
Erstelle einen neuen Datensatz namens "sales_subset" mit nur product, units und revenue aus sales
```

### Spalten ausschliessen

**Chat-Eingabe:**
```
Zeige sales-Daten ohne die Spalte advertising_spend
```

---

## Abschnitt 3: Zeilen filtern

Filtern ermöglicht es Ihnen, sich auf bestimmte Teilmengen Ihrer Daten zu konzentrieren.

### Einfache Bedingungen

**Chat-Eingabe:**
```
Zeige mir alle Zeilen aus sales, wo region gleich "North" ist
```

**Chat-Eingabe:**
```
Filtere sales, um nur Zeilen einzuschliessen, wo revenue grösser als 10000 ist
```

**Chat-Eingabe:**
```
Hole alle Verkaufsdaten für Widget A
```

### Bedingungen kombinieren

Sie können mehrere Bedingungen mit "und" oder "oder" kombinieren.

**Chat-Eingabe:**
```
Zeige sales, wo region "North" ist und revenue grösser als 12000 ist
```

**Chat-Eingabe:**
```
Filtere sales, wo product "Widget A" oder "Widget B" ist
```

### Verneinung

**Chat-Eingabe:**
```
Zeige alle Verkäufe, die nicht in der Region South sind
```

---

## Abschnitt 4: Daten sortieren

Sortieren hilft Ihnen, Muster und Extreme in Ihren Daten zu identifizieren.

### Einfache Sortierung

**Chat-Eingabe:**
```
Sortiere sales nach revenue in absteigender Reihenfolge
```

**Chat-Eingabe:**
```
Zeige sales sortiert nach date
```

### Mehrspaltige Sortierung

**Chat-Eingabe:**
```
Sortiere sales zuerst nach region, dann nach revenue innerhalb jeder Region
```

---

## Abschnitt 5: Grundlegende Dateninspektion

### Auf fehlende Werte prüfen

**Chat-Eingabe:**
```
Gibt es fehlende Werte im sales-Datensatz?
```

**Chat-Eingabe:**
```
Zähle die fehlenden Werte in jeder Spalte von sales
```

### Eindeutige Werte

**Chat-Eingabe:**
```
Was sind die eindeutigen Produkte in den Verkaufsdaten?
```

**Chat-Eingabe:**
```
Wie viele eindeutige Regionen gibt es in sales?
```

### Werteanzahl

**Chat-Eingabe:**
```
Wie viele Verkaufsdatensätze gibt es für jede Region?
```

**Chat-Eingabe:**
```
Zeige die Anzahl der Verkäufe nach Produkt
```

---

## Übungsaufgaben

### Übung 1: Laden und Erkunden

1. Lade `teaching_data/business_analytics/customer_segments.csv` als "customers"
2. Zeige die ersten 5 Zeilen
3. Hole zusammenfassende Statistiken
4. Liste alle eindeutigen Kundensegmente auf

<details>
<summary>Lösung</summary>

```
Lade teaching_data/business_analytics/customer_segments.csv als "customers"
Zeige die ersten 5 Zeilen von customers
Gib mir zusammenfassende Statistiken für customers
Was sind die eindeutigen Segmente in customers?
```

</details>

### Übung 2: Filtern

Mit dem sales-Datensatz:

1. Finde alle Verkäufe, wo die verkauften Einheiten grösser als 100 sind
2. Finde alle Widget A-Verkäufe in der Region North
3. Finde Verkäufe, wo die Werbeausgaben zwischen 1500 und 2000 liegen

<details>
<summary>Lösung</summary>

```
Zeige sales, wo units grösser als 100 ist
Filtere sales, wo product "Widget A" und region "North" ist
Zeige sales, wo advertising_spend zwischen 1500 und 2000 liegt
```

</details>

### Übung 3: Sortieren und Auswählen

1. Sortiere sales nach advertising_spend in aufsteigender Reihenfolge
2. Zeige nur die Top 5 Verkäufe nach revenue
3. Erstelle eine Teilmenge mit date, product und revenue, sortiert nach date

<details>
<summary>Lösung</summary>

```
Sortiere sales nach advertising_spend aufsteigend
Zeige die Top 5 Zeilen von sales sortiert nach revenue absteigend
Erstelle einen Datensatz mit date, product und revenue aus sales, sortiert nach date
```

</details>

---

## Häufige Fehler vermeiden

1. **Vergessen, zuerst Daten zu laden**: Laden Sie immer Ihre CSV, bevor Sie damit arbeiten

2. **Gross-/Kleinschreibung**: Spaltennamen und Zeichenkettenwerte können gross-/kleinschreibungssensitiv sein
   - Verwenden Sie die genaue Schreibweise: "North" nicht "north"

3. **Numerische vs. Zeichenketten-Vergleiche**:
   - Für Zahlen: `revenue > 10000`
   - Für Text: `region gleich "North"` (Anführungszeichen verwenden)

4. **Daten überschreiben**: Wenn Sie Daten filtern, speichern Sie sie unter einem neuen Namen, falls Sie das Original später benötigen

---

## Wichtigste Erkenntnisse

- **Laden**: Verwenden Sie `lade` mit dem Dateipfad und geben Sie Ihrem Datensatz einen Namen
- **Anzeigen**: Verwenden Sie `zeige`, `erste N Zeilen` oder `zusammenfassende Statistiken`
- **Auswählen**: Geben Sie Spalten nach Namen an, um Teilmengen zu erstellen
- **Filtern**: Verwenden Sie Bedingungen wie `gleich`, `grösser als`, `zwischen`
- **Sortieren**: Verwenden Sie `sortiere nach` mit optionalem `aufsteigend` oder `absteigend`

---

## Nächste Schritte

Bereit für mehr? Fahren Sie fort mit:
- [Fortgeschrittene Datenoperationen](05-data-operations-intermediate.md) - Lernen Sie Aggregation und grundlegendes Zusammenführen
- [Fortgeschrittenes Data Wrangling](06-data-wrangling-advanced.md) - Meistern Sie komplexe Joins und Umstrukturierung
