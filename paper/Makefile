# Makefile for JSS paper compilation
# Usage: make          - Build the PDF
#        make clean    - Remove generated files
#        make watch    - Continuous compilation (requires latexmk)

MAIN = article
BIBTEX = references

# Style files location
TEXINPUTS := .:./style:
BSTINPUTS := .:./style:
export TEXINPUTS BSTINPUTS

# Default target
all: $(MAIN).pdf

# Full build with bibliography
$(MAIN).pdf: $(MAIN).tex $(BIBTEX).bib style/jss.cls style/jss.bst
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

# Quick build (no bibtex, for drafting)
quick: $(MAIN).tex
	pdflatex $(MAIN)

# Continuous compilation with latexmk (if available)
watch:
	latexmk -pdf -pvc $(MAIN).tex

# Clean generated files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.pdf *.toc *.lof *.lot
	rm -f *.fdb_latexmk *.fls *.synctex.gz

# Clean but keep PDF
mostlyclean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot
	rm -f *.fdb_latexmk *.fls *.synctex.gz

.PHONY: all quick watch clean mostlyclean
