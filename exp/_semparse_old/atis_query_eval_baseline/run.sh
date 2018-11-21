#!/bin/sh

python -u ../../eval.py \
  --dataset semparse \
  --semparse_dataset atis \
  --semparse_split query \
  --n_epochs 300 \
  --n_enc 512 \
  --sched_factor 0.5 \
  --dropout 0.1 \
  --lr 0.003 \
  --copy_sup \
  > eval.val.out 2> eval.val.err
