#!/usr/bin/env bash
set -e

SCRIPT="prune_train.py"
EPOCHS=20
WD=1e-6

# Learning rates to test
EGR=(1.1 1.2 1.0 0.9 0.8 0.7 0.75 0.77 )
GDR=(0.008 0.009 0.01 0.02 0.03 0.04 0.05 0.06)
# EG runs
for LR in "${EGR[@]}"; do
  echo "=== EG | lr=${LR} ==="
  python "$SCRIPT" \
    --alg eg \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --weight_decay "${WD}" \
    --save_prefix "eg_lr${LR}"
done
#

# GD runs
for LR in "${GDR[@]}"; do
  echo "=== GD | lr=${LR} ==="
  python "$SCRIPT" \
    --alg gd \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --weight_decay "${WD}" \
    --save_prefix "gd_lr${LR}"
done
