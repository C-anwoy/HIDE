#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ $# -lt 1 ]]; then echo "Usage: bash scripts/export.sh SNAPSHOT_NAME [--suite review ...]" >&2; exit 2; fi
snapshot="$1"
shift
if [[ ! "$snapshot" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then echo "Use a simple snapshot name" >&2; exit 2; fi
python -m hide.export_results --source "${HIDE_RESULTS_ROOT:-$PWD/outputs/final}" --output "results/$snapshot" "$@"
python -m hide.export_results --verify "results/$snapshot"
