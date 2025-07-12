#!/bin/bash

# Configuration
MODEL="llama3-3b"
DATASET="race"
DEVICE="cuda:0"
SEED=25
KEYWORDS=20
LAYER=14
KERNEL="rbf"

echo "${MODEL}, ${DATASET}, ${LAYER}"

CMD="python generate.py \
  --model $MODEL \
  --dataset $DATASET \
  --device $DEVICE \
  --seed $SEED \
  --keywords $KEYWORDS \
  --layer $LAYER \
  --kernel $KERNEL"

# Run
echo "Running: $CMD"
eval $CMD
echo "Completed: ${MODEL}, ${DATASET}"