#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build-pass1.log
bibtex main > build-bibtex.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build-pass2.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build-pass3.log
