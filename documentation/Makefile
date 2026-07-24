# Build every Compresr documentation PDF.
#
# usage:
#   make            # build all PDFs into pdf/
#   make compresr_core
#   make python_litellm
#   make clean      # remove aux files
#   make distclean  # also remove generated PDFs

LATEX_DIR := latex
PDF_DIR   := pdf
BUILD_DIR := build

DOCS := \
	compresr_core \
	python_langchain \
	python_langgraph \
	python_llamaindex \
	python_litellm \
	typescript_langchain \
	typescript_langgraph \
	typescript_llamaindex

PDFS := $(addprefix $(PDF_DIR)/,$(addsuffix .pdf,$(DOCS)))

.PHONY: all clean distclean $(DOCS)

all: $(PDFS)

$(DOCS): %: $(PDF_DIR)/%.pdf

$(PDF_DIR)/%.pdf: $(LATEX_DIR)/%.tex $(LATEX_DIR)/compresr.sty
	@mkdir -p $(BUILD_DIR) $(PDF_DIR)
	cd $(LATEX_DIR) && latexmk -pdf -interaction=nonstopmode \
		-output-directory=../$(BUILD_DIR) $*.tex
	@cp $(BUILD_DIR)/$*.pdf $(PDF_DIR)/$*.pdf
	@echo "→ $(PDF_DIR)/$*.pdf"

clean:
	rm -rf $(BUILD_DIR)

distclean: clean
	rm -f $(PDFS)
