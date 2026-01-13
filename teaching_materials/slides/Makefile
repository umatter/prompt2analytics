# Makefile for building Beamer slides (multilingual)
# Requires: texlive with beamer and metropolis packages

TEX = pdflatex
BIB = bibtex

# Language directories
LANGS = en de-CH

.PHONY: all clean distclean en de-CH

# Build all languages
all: en de-CH

# Build English slides
en:
	cd en && $(TEX) main && $(TEX) main

# Build German (Swiss) slides
de-CH:
	cd de-CH && $(TEX) main && $(TEX) main

# Build with bibliography (specify LANG=en or LANG=de-CH)
with-bib:
	cd $(LANG) && $(TEX) main && $(BIB) main && $(TEX) main && $(TEX) main

# Clean auxiliary files from all language directories
clean:
	for lang in $(LANGS); do \
		rm -f $$lang/*.aux $$lang/*.log $$lang/*.nav $$lang/*.out \
		      $$lang/*.snm $$lang/*.toc $$lang/*.vrb $$lang/*.bbl \
		      $$lang/*.blg $$lang/*.synctex.gz; \
	done

# Clean everything including PDFs
distclean: clean
	for lang in $(LANGS); do \
		rm -f $$lang/main.pdf; \
	done

# Individual language targets for convenience
.PHONY: english german swiss
english: en
german: de-CH
swiss: de-CH
