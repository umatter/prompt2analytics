# Teaching Materials for prompt2analytics

This folder contains teaching materials for business analytics courses using prompt2analytics (p2a).

**Requirements**: Rust 1.85+ (for building from source), or use pre-built binaries

**Available languages:** English (en), German/Swiss (de-CH)

## Contents

### Slides (`slides/`)

LaTeX/Beamer presentations using the Metropolis theme.

| Language | Directory | Build Command |
|----------|-----------|---------------|
| English | `slides/en/` | `make en` |
| Deutsch (CH) | `slides/de-CH/` | `make de-CH` |

Build all languages: `make` or `make all`

Requires: texlive with beamer and metropolis packages

### Tutorials (`tutorials/`)

Step-by-step markdown guides:

#### English (`tutorials/en/`)
- `01-cli-getting-started.md` - Introduction to the p2a command-line interface
- `02-chat-interface-guide.md` - Using the web-based chat interface (primary focus)
- `03-data-analysis-workflow.md` - Complete data analysis workflow examples

#### Deutsch/Schweiz (`tutorials/de-CH/`)
- `01-cli-getting-started.md` - Einführung in das p2a Command-Line Interface
- `02-chat-interface-guide.md` - Das Chat-Interface verwenden (Hauptfokus)
- `03-data-analysis-workflow.md` - Vollständiger Datenanalyse-Workflow

### Teaching Data (`teaching_data/`)

Sample datasets for tutorials and exercises (shared across languages):

- `economics/` - Economic datasets (GDP, labor market)
- `business_analytics/` - Business datasets (sales, customer segments, financial statements)

Documentation:
- `README.en.md` - English dataset documentation
- `README.de-CH.md` - Deutsche Datensatz-Dokumentation

## Target Audience

College students in business analytics courses with:
- Basic statistics knowledge
- Familiarity with data analysis concepts
- No prior programming experience required (for chat interface)

## Getting Started

### English
1. Start with `tutorials/en/02-chat-interface-guide.md` for the chat-based approach
2. Optionally explore `tutorials/en/01-cli-getting-started.md` for command-line usage
3. Use datasets in `teaching_data/` for hands-on practice

### Deutsch (Schweiz)
1. Beginnen Sie mit `tutorials/de-CH/02-chat-interface-guide.md` für den Chat-Ansatz
2. Optional: `tutorials/de-CH/01-cli-getting-started.md` für die Kommandozeile
3. Verwenden Sie die Datensätze in `teaching_data/` für praktische Übungen

## Language Notes

### German (Swiss) - de-CH
- Uses Swiss orthography (no ß, use ss instead)
- Example: "Strasse" not "Straße"
- Guillemets for quotation marks: «Beispiel»

## Contributing Translations

To add a new language:
1. Create `slides/<lang>/main.tex`
2. Create `tutorials/<lang>/` with translated markdown files
3. Add `teaching_data/README.<lang>.md`
4. Update the Makefile with new language target
