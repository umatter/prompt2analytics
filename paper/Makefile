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

.PHONY: all jss arxiv clean cleanall quick-jss quick-arxiv watch help

# Default target: build both versions
all: jss arxiv

# JSS format (for Journal of Statistical Software submission)
jss: article-jss.pdf

article-jss.pdf: article-jss.tex content.tex $(BIBTEX).bib style/jss.cls style/jss.bst
	pdflatex -interaction=nonstopmode article-jss.tex
	bibtex article-jss
	pdflatex -interaction=nonstopmode article-jss.tex
	pdflatex -interaction=nonstopmode article-jss.tex

# arXiv format (for preprint submission)
arxiv: article-arxiv.pdf

article-arxiv.pdf: article-arxiv.tex content.tex $(BIBTEX).bib
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

# Help
help:
	@echo "Available targets:"
	@echo "  make jss        - Build JSS format (article-jss.pdf)"
	@echo "  make arxiv      - Build arXiv format (article-arxiv.pdf)"
	@echo "  make all        - Build both formats (default)"
	@echo "  make quick-jss  - Quick JSS build (no bibtex)"
	@echo "  make quick-arxiv- Quick arXiv build (no bibtex)"
	@echo "  make clean      - Remove auxiliary files"
	@echo "  make cleanall   - Remove all generated files including PDFs"
