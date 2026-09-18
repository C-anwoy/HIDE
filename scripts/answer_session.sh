#!/usr/bin/env bash
# Internal tmux entry point. Use the interpreter selected in the activated environment.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="$1"
gpu_uuid="$2"
action="$3"
result_root="$4"
model="${5:-}"
if [[ -f configs/env.local.sh ]]; then source configs/env.local.sh; fi
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export HIDE_DEVICE=cuda:0
export PATH="$(dirname "$python_bin"):$PATH"
mkdir -p "$result_root/logs"
args=("$action" --root "$result_root" --hours 18 --hard-hours 20)
if [[ "$action" == detection ]]; then
  args+=(--models "$model")
else
  args+=(--wait-for-detection)
fi
"$python_bin" -u -m hide.answer_runs "${args[@]}" 2>&1 | tee -a "$result_root/logs/$action-${model:-all}.log"
