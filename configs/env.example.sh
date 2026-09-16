# Copy to configs/env.local.sh and edit. The local file is ignored by Git.
export HIDE_MODEL_ROOT=/home/models
export HIDE_DATA_ROOT="$PWD/data"
export HIDE_RESULTS_ROOT="$PWD/outputs/final"
export HIDE_DEVICE=cuda:0
export HIDE_DTYPE=bfloat16
export TOKENIZERS_PARALLELISM=false
