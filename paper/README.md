# prompt2analytics Paper

This directory contains the manuscript and supporting code for the Journal of Statistical Software (JSS) submission.

## Quick Start

```bash
# Build JSS format (for journal submission)
make jss

# Build arXiv format (for preprint)
make arxiv

# Build both formats
make all
```

## Prerequisites

### LaTeX

A full LaTeX distribution is required (TeX Live, MiKTeX, or MacTeX):
- `pdflatex`
- `bibtex`

### R Packages

Install required packages:
```r
source("code/requirements.R")
```

Or manually:
```r
install.packages(c(
  "tidyverse", "ggplot2", "jsonlite", "xtable",
  "patchwork", "scales", "viridis"
))
```

### Rust CLI (for regenerating examples)

```bash
cargo build --release -p p2a-cli
```

## Directory Structure

```
paper/
├── article-jss.tex      # JSS format wrapper
├── article-arxiv.tex    # arXiv format wrapper
├── paper.tex            # Shared document content
├── references.bib       # Bibliography
├── Makefile             # Build automation
│
├── code/                # Exhibit generation scripts
│   ├── fig_*.R          # Figure scripts (→ figures/)
│   ├── tab_*.R          # Table scripts (→ tables/)
│   ├── helpers.R        # Shared utility functions
│   ├── requirements.R   # R package dependencies
│   ├── generate_*.sh    # CLI output generation
│   ├── rust_validation/ # R vs Rust benchmarking
│   └── llm_eval/        # LLM evaluation framework
│
├── sections/            # LaTeX chapter files
├── figures/             # Generated figures (PDF + PNG)
├── tables/              # Generated LaTeX tables
├── generated/           # CLI output captures
└── data/                # Source datasets
```

## Build Targets

### Paper Building

| Target | Description |
|--------|-------------|
| `make jss` | Build JSS format (`article-jss.pdf`) |
| `make arxiv` | Build arXiv format (`article-arxiv.pdf`) |
| `make all` | Build both formats |
| `make quick-jss` | Quick build without bibtex (for drafting) |

### Exhibit Generation (from cached results)

| Target | Description |
|--------|-------------|
| `make figures` | Regenerate figures from existing JSON results |
| `make tables` | Regenerate tables from existing JSON results |
| `make generate` | Regenerate CLI output examples |
| `make exhibits` | All of the above |

### Full Reproducibility (re-run experiments)

| Target | Description | Time |
|--------|-------------|------|
| `make rust-validation` | Run Rust vs R validation + benchmarks | ~30 min |
| `make llm-eval MODEL=... PROVIDER=...` | Run LLM evaluation (requires API keys) | ~1-2 hours |
| `make reproduce-all CONFIRM=yes` | Full pipeline (NO LLM eval) | ~45 min |

### Cleanup

| Target | Description |
|--------|-------------|
| `make clean` | Remove auxiliary files (keep PDFs) |
| `make cleanall` | Remove all generated files |
| `make help` | Show all available targets |

## Regenerating Exhibits

### All Figures

```bash
make figures
```

Or individually:
```bash
cd code
Rscript fig_benchmark_speedup.R
Rscript fig_accuracy_comparison.R
# etc.
```

### All Tables

```bash
make tables
```

### CLI Examples

```bash
make generate
```

This runs `code/generate_outputs.sh` which executes actual CLI commands and captures output for the paper examples.

## Full Reproducibility

To reproduce all paper results from scratch:

```bash
# Full pipeline: run experiments → generate exhibits → build paper
# Requires explicit confirmation (takes ~45 minutes)
make reproduce-all CONFIRM=yes
```

This runs:
1. `rust-validation` - Rust vs R accuracy validation and performance benchmarks (~30 min)
2. `exhibits` - Regenerate all figures and tables from fresh results
3. `jss` - Build the paper

**Note:** LLM evaluation is NOT included in `reproduce-all` (requires API keys and takes 1-2 hours).

### LLM Evaluation (optional, requires API keys)

```bash
export OPENAI_API_KEY="your-key"
make llm-eval MODEL=gpt-4o PROVIDER=openai
```

Supported providers: `openai`, `anthropic`, `openrouter`, `ollama`

## Individual Validation and Benchmarks

### Validation Tests

```bash
make validate          # All validation tests
make validate-ols      # OLS only (Longley dataset)
make validate-panel    # Panel FE/RE (Grunfeld dataset)
```

### Benchmarks

```bash
make benchmark         # Rust benchmarks (Criterion)
make benchmark-r       # R comparison benchmarks
make benchmark-full    # Both Rust and R
```

## Naming Conventions

| Type | Pattern | Output Location |
|------|---------|-----------------|
| Figure scripts | `fig_*.R` | `figures/*.pdf` |
| Table scripts | `tab_*.R` | `tables/*.tex` |
| Shell scripts | `verb_noun.sh` | varies |
| R methods | `snake_case.R` | N/A |

## Workflow

1. **Edit content** in `sections/*.tex`
2. **Regenerate exhibits** if data changed: `make figures tables`
3. **Rebuild paper**: `make jss` or `make arxiv`
4. **Full rebuild**: `make cleanall all`

## Subdirectory Documentation

- **rust_validation/**: See `code/rust_validation/README.md` for R vs Rust benchmarking
- **llm_eval/**: See `code/llm_eval/README.md` for LLM evaluation framework

## Troubleshooting

### LaTeX errors

```bash
# Check log files
cat article-jss.log | grep -i error

# Clean and rebuild
make cleanall jss
```

### Missing R packages

```bash
Rscript code/requirements.R
```

### CLI binary not found

```bash
cargo build --release -p p2a-cli
export P2A=./target/release/p2a
make generate
```

## Citation

If you use this software, please cite:

```bibtex
@article{prompt2analytics2026,
  title = {prompt2analytics: LLM-Assisted Econometric Analysis via Rust},
  author = {...},
  journal = {Journal of Statistical Software},
  year = {2026}
}
```
