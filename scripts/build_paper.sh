#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_root/output/pdf"
cd "$repo_root/paper"
if command -v tectonic >/dev/null; then
  for source in COLI_template.tex response_letter.tex; do
    tectonic "$source" --outdir "$repo_root/output/pdf" --keep-logs
  done
elif command -v latexmk >/dev/null; then
  for source in COLI_template.tex response_letter.tex; do
    latexmk -pdf -interaction=nonstopmode -halt-on-error \
      -outdir="$repo_root/output/pdf" "$source"
  done
else
  echo 'Install Tectonic or TeX Live/MacTeX with latexmk to build the manuscript and response.' >&2
  exit 1
fi
