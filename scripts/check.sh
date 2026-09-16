#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python -m compileall -q hide tests
python -m unittest discover -s tests -v
for script in scripts/*.sh; do bash -n "$script"; done
