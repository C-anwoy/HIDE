#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="$1"
model="$2"
if [[ -f configs/env.local.sh ]]; then source configs/env.local.sh; fi
export PATH="$(dirname "$python_bin"):$PATH"
"$python_bin" -u scripts/answer_subset.py run --model "$model" --gpu 1 \
  --hours 8 --hard-hours 9 2>&1 | tee -a "outputs/answer-detection/logs/subset-$model.log"
