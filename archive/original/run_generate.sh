#!/bin/bash

# Configuration
MODEL="llama3-3b"
DATASET="halueval"
DEVICE="cuda:0"
SEED=25
KEYWORDS=20
LAYER=14
KERNEL="rbf"
PROJECT_IND=0

# MODEL="llama3-8b"
# DATASET="SQuAD"
# DEVICE="cuda:0"
# SEED=25
# KEYWORDS=20
# LAYER=16
# KERNEL="rbf"
# PROJECT_IND=1

# MODEL="gemma2"
# DATASET="halueval"
# DEVICE="cuda:0"
# SEED=25
# KEYWORDS=20
# LAYER=21
# KERNEL="rbf"
# PROJECT_IND=0

echo "${MODEL}, ${DATASET}, ${LAYER}"

CMD="python generate.py \
  --model $MODEL \
  --dataset $DATASET \
  --device $DEVICE \
  --seed $SEED \
  --keywords $KEYWORDS \
  --layer $LAYER \
  --kernel $KERNEL \
  --project_ind $PROJECT_IND"

# Run
echo "Running: $CMD"
eval $CMD
echo "Completed: ${MODEL}, ${DATASET}"

# Configuration
MODEL="llama3-3b-instruct"
DATASET="halueval"
DEVICE="cuda:0"
SEED=25
KEYWORDS=20
LAYER=14
KERNEL="rbf"
PROJECT_IND=0

echo "${MODEL}, ${DATASET}, ${LAYER}"

CMD="python generate.py \
  --model $MODEL \
  --dataset $DATASET \
  --device $DEVICE \
  --seed $SEED \
  --keywords $KEYWORDS \
  --layer $LAYER \
  --kernel $KERNEL \
  --project_ind $PROJECT_IND"

# Run
echo "Running: $CMD"
eval $CMD
echo "Completed: ${MODEL}, ${DATASET}"