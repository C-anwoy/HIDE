#!/usr/bin/env bash
# Run only the fixed 2,000-example comparison, two models on physical GPU1.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="$(command -v python)"
"$python_bin" -c 'import torch, transformers; print("Environment ready")'
for model in llama gemma; do
  if tmux has-session -t "hide-2000-$model" 2>/dev/null; then
    echo "hide-2000-$model exists; inspect/attach instead of launching duplicates." >&2
    exit 2
  fi
done
mkdir -p outputs/answer-detection/logs
for model in llama gemma; do
  if [[ "$model" == llama ]]; then model_name=llama3-8b; else model_name=gemma-2-9b; fi
  printf -v command_text '%q ' bash scripts/subset_session.sh "$python_bin" "$model_name"
  tmux new-session -d -s "hide-2000-$model" -c "$repo_root"
  tmux send-keys -t "hide-2000-$model" -l "$command_text"
  tmux send-keys -t "hide-2000-$model" Enter
done
echo 'Started hide-2000-llama and hide-2000-gemma on GPU1; 8h soft / 9h hard budget each.'
echo 'Shells remain open after success or failure. Logs: outputs/answer-detection/logs/subset-*.log'
