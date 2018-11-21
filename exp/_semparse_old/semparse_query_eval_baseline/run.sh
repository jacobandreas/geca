#!/bin/sh

python -u ../../eval.py \
  --dataset semparse \
  --seed 1 \
  --semparse_split query \
  --n_epochs 150 \
  --n_enc 512 \
  --sched_factor 0.5 \
  --dropout 0.5 \
  --lr 0.001 \
  --copy_sup \
  > eval.val1.out 2> eval.val1.err

# 1 has SEED 1
