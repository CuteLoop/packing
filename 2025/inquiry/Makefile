LATEXMK := $(shell command -v latexmk 2>/dev/null)
PDFL := pdflatex

TEX := main.tex
PDF := main.pdf
CHAPTERS := $(wildcard chapters/*.tex)

.PHONY: all clean distclean

all: $(PDF)

ifeq ($(LATEXMK),)
$(PDF): $(TEX) $(CHAPTERS)
	@echo "latexmk not found; falling back to pdflatex"
	$(PDFL) -interaction=nonstopmode $(TEX)
	@if [ -f main.aux ]; then bibtex main || true; fi
	$(PDFL) -interaction=nonstopmode $(TEX)
	$(PDFL) -interaction=nonstopmode $(TEX)

clean:
	rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.toc *.synctex.gz

distclean: clean
	rm -f $(PDF)

else
$(PDF): $(TEX) $(CHAPTERS)
	$(LATEXMK) -pdf -interaction=nonstopmode $(TEX)

clean:
	$(LATEXMK) -c $(TEX) || true

distclean: clean
	rm -f $(PDF)

endif

# Convenience: open the PDF (Unix)
.PHONY: view
view: $(PDF)
	@if command -v xdg-open >/dev/null 2>&1; then xdg-open $(PDF) &>/dev/null || true; fi
