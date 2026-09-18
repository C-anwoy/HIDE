#!/usr/bin/env bash
# Two detection GPUs plus a timing waiter; all jobs use bounded, resumable workers.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
llama_gpu=0
gemma_gpu=1
timing_gpu=1
result_root=outputs/answer-boundary
prefix=hide-answer
while [[ $# -gt 0 ]]; do
  case "$1" in
    --llama-gpu) llama_gpu="$2"; shift 2 ;;
    --gemma-gpu) gemma_gpu="$2"; shift 2 ;;
    --timing-gpu) timing_gpu="$2"; shift 2 ;;
    --root) result_root="$2"; shift 2 ;;
    --prefix) prefix="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
command -v tmux >/dev/null
python_bin="$(command -v python)"
"$python_bin" -c 'import sys, torch, transformers; assert sys.version_info >= (3,10); print("Interpreter:",sys.executable)'
gpu_uuid() {
  local value
  value="$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader)"
  [[ "$value" == GPU-* && "$value" != *$'\n'* ]] || { echo 'Expected one GPU UUID' >&2; return 1; }
  printf '%s' "$value"
}
llama_uuid="$(gpu_uuid "$llama_gpu")"
gemma_uuid="$(gpu_uuid "$gemma_gpu")"
timing_uuid="$(gpu_uuid "$timing_gpu")"
[[ "$llama_uuid" != "$gemma_uuid" ]] || { echo 'Choose two distinct detection GPUs, or use run_answers.sh manually.' >&2; exit 2; }
for suffix in llama gemma timing; do
  if tmux has-session -t "$prefix-$suffix" 2>/dev/null; then
    echo "Session $prefix-$suffix already exists; attach to it instead of launching a duplicate." >&2
    exit 2
  fi
done
"$python_bin" -u -m hide.answer_runs init --root "$result_root"
start_session() {
  local name="$1" gpu="$2" action="$3" model="$4" command_text
  printf -v command_text '%q ' bash "$repo_root/scripts/answer_session.sh" "$python_bin" "$gpu" "$action" "$result_root" "$model"
  tmux new-session -d -s "$name" -c "$repo_root" "$command_text"
}
start_session "$prefix-llama" "$llama_uuid" detection llama3-8b
start_session "$prefix-gemma" "$gemma_uuid" detection gemma-2-9b
start_session "$prefix-timing" "$timing_uuid" timing ''
printf 'Started %s-llama, %s-gemma and %s-timing.\n' "$prefix" "$prefix" "$prefix"
printf 'Timing waits for all detection parts. Each session has an 18h soft / 20h hard budget, including pilot or waiting time.\n'
printf 'Saved logs: %s/logs/\n' "$result_root"
