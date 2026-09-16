#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root/paper"
command -v latexmk >/dev/null || { echo 'Install TeX Live/MacTeX with latexmk to build the manuscript.' >&2; exit 1; }
exec latexmk -pdf -interaction=nonstopmode -halt-on-error COLI_template.tex
