# Erste Schritte mit dem p2a Command-Line Interface

## Lernziele

Nach Abschluss dieses Tutorials können Sie:
- [ ] Das p2a CLI installieren und ausführen
- [ ] Datensätze aus verschiedenen Dateiformaten laden
- [ ] Grundlegende Datenexploration durchführen
- [ ] Einfache Regressionsanalysen ausführen
- [ ] Ergebnisse für Berichte exportieren

## Voraussetzungen

- p2a CLI installiert (siehe Abschnitt Installation)
- Zugang zu Terminal/Eingabeaufforderung
- Beispieldatensätze aus `teaching_data/`

## Installation

<!-- TODO: Installationsanleitung ergänzen -->

## Abschnitt 1: Ihre ersten Befehle

### Einen Datensatz laden

<!-- TODO: Schritt-für-Schritt-Anleitung ergänzen -->

```bash
p2a data load pfad/zu/daten.csv --name meinedaten
```

### Daten anzeigen

<!-- TODO: Beispiele ergänzen -->

```bash
p2a data describe meinedaten
p2a data head meinedaten
```

## Abschnitt 2: Grundlegende Statistik

<!-- TODO: Korrelation, deskriptive Statistik ergänzen -->

## Abschnitt 3: Regressionen durchführen

<!-- TODO: OLS-Regressionsbeispiele ergänzen -->

```bash
p2a reg ols meinedaten -y abhaengige_var -x unabhaengige_var1 unabhaengige_var2
```

## Abschnitt 4: Visualisierungen erstellen

<!-- TODO: Visualisierungsbeispiele ergänzen -->

## Abschnitt 5: Session-Verwaltung

<!-- TODO: Session-Aufzeichnung und Skript-Export ergänzen -->

## Übungsaufgaben

1. <!-- TODO: Übung 1 -->
2. <!-- TODO: Übung 2 -->
3. <!-- TODO: Übung 3 -->

## Zusammenfassung

<!-- TODO: Wichtigste Punkte ergänzen -->

## Nächste Schritte

- Weiter mit der [Chat-Interface-Anleitung](02-chat-interface-guide.md) für den webbasierten Ansatz
- Probieren Sie den [Datenanalyse-Workflow](03-data-analysis-workflow.md) für ein vollständiges Beispiel
