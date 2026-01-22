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
        validate validate-ols validate-panel validate-hdfe \
        benchmark benchmark-regression benchmark-r benchmark-full

# Default target: build both versions
all: jss arxiv

# JSS format (for Journal of Statistical Software submission)
jss: article-jss.pdf

# Section files
SECTIONS = sections/introduction.tex sections/methods.tex sections/architecture.tex \
           sections/examples.tex sections/performance.tex sections/validation.tex \
           sections/conclusion.tex

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

# Help
help:
	@echo "Available targets:"
	@echo ""
	@echo "  Paper building:"
	@echo "    make jss           - Build JSS format (article-jss.pdf)"
	@echo "    make arxiv         - Build arXiv format (article-arxiv.pdf)"
	@echo "    make all           - Build both formats (default)"
	@echo "    make generate      - Regenerate CLI output examples"
	@echo "    make quick-jss     - Quick JSS build (no bibtex)"
	@echo "    make quick-arxiv   - Quick arXiv build (no bibtex)"
	@echo ""
	@echo "  Validation (run separately):"
	@echo "    make validate      - Run ALL validation tests against R"
	@echo "    make validate-ols  - Run OLS validation (Longley dataset)"
	@echo "    make validate-panel- Run Panel FE/RE validation (Grunfeld)"
	@echo "    make validate-hdfe - Run HDFE validation"
	@echo ""
	@echo "  Benchmarks (run separately):"
	@echo "    make benchmark     - Run Rust benchmarks (Criterion)"
	@echo "    make benchmark-r   - Run R benchmarks for comparison"
	@echo "    make benchmark-full- Run both Rust and R benchmarks"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean         - Remove auxiliary files"
	@echo "    make clean-generated - Remove generated CLI output files"
	@echo "    make cleanall      - Remove all generated files including PDFs"
