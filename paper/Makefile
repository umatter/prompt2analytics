# Makefile for prompt2analytics paper
# Supports both JSS (journal) and arXiv (preprint) formats
#
# Usage:
#   make jss      - Build JSS format (article-jss.pdf)
#   make arxiv    - Build arXiv format (article-arxiv.pdf)
#   make all      - Build both formats
#   make clean    - Remove auxiliary files (keep PDFs)
#   make cleanall - Remove everything including PDFs

BIBTEX = references

# Style files location (for JSS)
TEXINPUTS := .:./style:
BSTINPUTS := .:./style:
export TEXINPUTS BSTINPUTS

.PHONY: all jss arxiv clean cleanall quick-jss quick-arxiv watch help generate clean-generated \
        figures tables exhibits \
        validate validate-ols validate-panel validate-hdfe \
        benchmark benchmark-regression benchmark-r benchmark-full \
        rust-validation llm-eval reproduce-all

# Default target: build both versions
all: jss arxiv

# JSS format (for Journal of Statistical Software submission)
jss: article-jss.pdf

# Section files
SECTIONS = sections/introduction_new.tex sections/defining.tex sections/tools.tex \
           sections/examples.tex sections/evaluation_new.tex sections/deployment.tex \
           sections/discussion.tex sections/appendices.tex

# Generated output files (from CLI commands)
GENERATED = generated/cli_data_describe.tex generated/cli_data_head.tex \
            generated/cli_ols.tex generated/cli_diagnostics.tex \
            generated/cli_fe.tex generated/cli_re.tex \
            generated/cli_hausman.tex generated/cli_hdfe.tex \
            generated/cli_ols_json.tex

# Generate CLI output files
generate: $(GENERATED)

$(GENERATED): code/generate_outputs.sh
	./code/generate_outputs.sh

# Clean generated files
clean-generated:
	rm -rf generated/*.tex

# Regenerate all figures from R scripts
figures:
	@echo "Regenerating figures..."
	cd code && Rscript fig_benchmark_speedup.R
	cd code && Rscript fig_benchmark_histogram.R
	cd code && Rscript fig_benchmark_execution.R
	cd code && Rscript fig_benchmark_boxplots.R
	cd code && Rscript fig_accuracy_comparison.R
	cd code && Rscript fig_accuracy_by_category.R
	cd code && Rscript fig_latency_comparison.R
	cd code && Rscript fig_extended_eval_comparison.R
	cd code && Rscript fig_robustness_radar.R
	@echo "Figures regenerated in figures/"

# Regenerate all tables from R scripts
tables:
	@echo "Regenerating tables..."
	cd code && Rscript tab_benchmark_summary.R
	cd code && Rscript tab_category_accuracy.R
	cd code && Rscript tab_extended_eval_summary.R
	cd code && Rscript tab_model_summary.R
	@echo "Tables regenerated in tables/"

# Regenerate all exhibits (figures + tables + CLI outputs)
exhibits: figures tables generate
	@echo "All exhibits regenerated."

article-jss.pdf: article-jss.tex paper.tex $(SECTIONS) $(GENERATED) $(BIBTEX).bib jss.cls jss.bst
	pdflatex -interaction=nonstopmode article-jss.tex
	bibtex article-jss
	pdflatex -interaction=nonstopmode article-jss.tex
	pdflatex -interaction=nonstopmode article-jss.tex

# arXiv format (for preprint submission)
arxiv: article-arxiv.pdf

article-arxiv.pdf: article-arxiv.tex paper.tex $(SECTIONS) $(GENERATED) $(BIBTEX).bib
	pdflatex -interaction=nonstopmode article-arxiv.tex
	bibtex article-arxiv
	pdflatex -interaction=nonstopmode article-arxiv.tex
	pdflatex -interaction=nonstopmode article-arxiv.tex

# Quick builds (no bibtex, for drafting)
quick-jss:
	pdflatex -interaction=nonstopmode article-jss.tex

quick-arxiv:
	pdflatex -interaction=nonstopmode article-arxiv.tex

# Continuous compilation with latexmk (if available)
watch:
	@echo "Use: latexmk -pdf -pvc article-jss.tex"
	@echo "  or: latexmk -pdf -pvc article-arxiv.tex"

# Clean auxiliary files (keep PDFs)
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot
	rm -f *.fdb_latexmk *.fls *.synctex.gz

# Clean everything including PDFs
cleanall: clean
	rm -f article-jss.pdf article-arxiv.pdf article.pdf

# Validation and benchmarks (run separately, not part of normal build)
validate:
	@echo "Running validation tests against R reference implementations..."
	cd .. && cargo test -p p2a-core -- test_validate --nocapture

validate-ols:
	@echo "Running OLS validation tests (Longley dataset)..."
	cd .. && cargo test -p p2a-core -- regression::ols::tests::test_validate --nocapture

validate-panel:
	@echo "Running Panel data validation tests (Grunfeld dataset)..."
	cd .. && cargo test -p p2a-core -- econometrics::panel::tests::test_validate --nocapture

validate-hdfe:
	@echo "Running HDFE validation tests..."
	cd .. && cargo test -p p2a-core -- econometrics::hdfe::tests::test_validate --nocapture

benchmark:
	@echo "Running Rust benchmarks (Criterion)..."
	cd .. && cargo bench -p p2a-core -- --sample-size 10

benchmark-regression:
	@echo "Running regression benchmarks..."
	cd .. && cargo bench -p p2a-core -- OLS --sample-size 10

benchmark-r:
	@echo "Running R comparison benchmarks..."
	@echo "Requires: R with microbenchmark, sandwich, plm, lfe packages"
	cd ../performance/comparisons/r_comparison && Rscript run_all_benchmarks.R

benchmark-full:
	@echo "Running full benchmark suite (Rust + R)..."
	@echo "Step 1: Rust benchmarks"
	cd .. && cargo bench -p p2a-core 2>&1 | tee performance/rust_benchmark_results.txt
	@echo ""
	@echo "Step 2: R benchmarks"
	cd ../performance/comparisons/r_comparison && Rscript run_all_benchmarks.R

# ============================================================
# Full Reproducibility Pipeline
# ============================================================
# WARNING: These targets run expensive computations.
# - rust-validation: ~30 minutes
# - llm-eval: ~1-2 hours, requires API keys
# - reproduce-all: ~45 minutes (does NOT include LLM eval)
#
# To prevent accidental execution, reproduce-all requires CONFIRM=yes
# ============================================================

# Run complete Rust vs R validation and benchmarking (generates JSON results)
# This runs: synthetic data generation → validation → benchmarks → report
rust-validation:
	@echo "Running complete Rust vs R validation and benchmark suite..."
	@echo "This may take 10-30 minutes depending on hardware."
	@echo ""
	cd code/rust_validation && ./scripts/run_all.sh
	@echo ""
	@echo "Results written to code/rust_validation/results/"

# Run LLM tool selection evaluation (requires API keys)
# Usage: make llm-eval MODEL=gpt-4o PROVIDER=openai
# Supported: openai (gpt-4o, gpt-4o-mini), anthropic (claude-3-5-sonnet), openrouter, ollama
MODEL ?= gpt-4o
PROVIDER ?= openai
llm-eval:
	@echo "Running LLM tool selection evaluation..."
	@echo "Model: $(MODEL), Provider: $(PROVIDER)"
	@echo "Requires API key: OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY"
	@echo ""
	cd code/llm_eval && ./scripts/run_eval.sh $(MODEL) $(PROVIDER) all
	cd code/llm_eval && ./scripts/run_multi_turn_eval.sh $(MODEL) $(PROVIDER) all
	cd code/llm_eval && ./scripts/run_naturalistic_eval.sh $(MODEL) $(PROVIDER) all
	@echo ""
	@echo "Results written to code/llm_eval/results/"

# Full reproducibility: Rust validation + exhibits + paper (NO LLM eval)
# Requires explicit confirmation: make reproduce-all CONFIRM=yes
reproduce-all:
ifndef CONFIRM
	@echo "============================================"
	@echo "WARNING: reproduce-all runs expensive computations (~45 min)"
	@echo "============================================"
	@echo ""
	@echo "This will:"
	@echo "  1. Run Rust vs R validation and benchmarks (~30 min)"
	@echo "  2. Regenerate all figures and tables"
	@echo "  3. Rebuild the paper"
	@echo ""
	@echo "To proceed, run:"
	@echo "  make reproduce-all CONFIRM=yes"
	@echo ""
	@echo "Note: LLM evaluation is NOT included (requires API keys)."
	@echo "For full reproduction including LLM eval, see 'make help'."
	@exit 1
else
	@echo "Starting full reproduction pipeline..."
	$(MAKE) rust-validation
	$(MAKE) exhibits
	$(MAKE) jss
	@echo ""
	@echo "============================================"
	@echo "Full reproduction complete!"
	@echo "============================================"
	@echo ""
	@echo "Generated:"
	@echo "  - Validation results: code/rust_validation/results/"
	@echo "  - Figures: figures/"
	@echo "  - Tables: tables/"
	@echo "  - Paper: article-jss.pdf"
	@echo ""
	@echo "Note: LLM evaluation was NOT included."
	@echo "Run 'make llm-eval MODEL=gpt-4o PROVIDER=openai' separately if needed."
endif

# Help
help:
	@echo "Available targets:"
	@echo ""
	@echo "  Paper building:"
	@echo "    make jss             - Build JSS format (article-jss.pdf)"
	@echo "    make arxiv           - Build arXiv format (article-arxiv.pdf)"
	@echo "    make all             - Build both formats (default)"
	@echo "    make quick-jss       - Quick JSS build (no bibtex)"
	@echo "    make quick-arxiv     - Quick arXiv build (no bibtex)"
	@echo ""
	@echo "  Exhibit generation (from cached results):"
	@echo "    make figures         - Regenerate figures from existing JSON"
	@echo "    make tables          - Regenerate tables from existing JSON"
	@echo "    make generate        - Regenerate CLI output examples"
	@echo "    make exhibits        - All of the above"
	@echo ""
	@echo "  Full reproducibility (re-run experiments):"
	@echo "    make rust-validation - Rust vs R validation + benchmarks (~30 min)"
	@echo "    make llm-eval        - LLM evaluation (requires API keys, ~1-2 hrs)"
	@echo "    make reproduce-all CONFIRM=yes - Full pipeline, NO LLM (~45 min)"
	@echo ""
	@echo "  Individual validation tests:"
	@echo "    make validate        - Run ALL Rust validation tests"
	@echo "    make validate-ols    - OLS validation (Longley dataset)"
	@echo "    make validate-panel  - Panel FE/RE validation (Grunfeld)"
	@echo "    make validate-hdfe   - HDFE validation"
	@echo ""
	@echo "  Individual benchmarks:"
	@echo "    make benchmark       - Run Rust benchmarks (Criterion)"
	@echo "    make benchmark-r     - Run R benchmarks for comparison"
	@echo "    make benchmark-full  - Both Rust and R benchmarks"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean           - Remove auxiliary files"
	@echo "    make clean-generated - Remove generated CLI output files"
	@echo "    make cleanall        - Remove all generated files including PDFs"
	@echo ""
	@echo "  Examples:"
	@echo "    make reproduce-all CONFIRM=yes       # Full reproduction (no LLM)"
	@echo "    make llm-eval MODEL=claude-3-5-sonnet PROVIDER=anthropic"
