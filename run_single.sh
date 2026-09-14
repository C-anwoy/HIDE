#!/bin/bash

# Configuration
# MODEL="llama3-3b"
# DATASET="race"
# DEVICE="cuda:0"
# SEED=25
# KEYWORDS=20
# LAYER=14
# KERNEL="rbf"

# MODEL="llama3-8b"
# DATASET="SQuAD"
# DEVICE="cuda:0"
# SEED=25
# KEYWORDS=20
# LAYER=16
# KERNEL="rbf"
# PROJECT_IND=2

MODEL="gemma2"
DATASET="SQuAD"
DEVICE="cuda:0"
SEED=25
KEYWORDS=20
LAYER=21
KERNEL="rbf"
PROJECT_IND=3

echo "${MODEL}, ${DATASET}, ${LAYER}"

CMD="python generate_single_exp.py \
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

# MODEL="llama3-8b"
# DATASET="nq_open"
# DEVICE="cuda:0"
# SEED=25
# KEYWORDS=20
# LAYER=16
# KERNEL="rbf"
# PROJECT_IND=2

MODEL="gemma2"
DATASET="nq_open"
DEVICE="cuda:0"
SEED=25
KEYWORDS=20
LAYER=21
KERNEL="rbf"
PROJECT_IND=3

echo "${MODEL}, ${DATASET}, ${LAYER}"

CMD="python generate_single_exp.py \
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