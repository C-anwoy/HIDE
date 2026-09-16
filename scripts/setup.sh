#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
printf "Activate with: source .venv/bin/activate\n"
