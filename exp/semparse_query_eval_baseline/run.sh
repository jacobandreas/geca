#!/bin/sh

python -u ../../eval.py \
  --dataset semparse \
  --semparse_dataset geography-logic.txt \
  --semparse_mrl logic \
  --seed 4 \
  --semparse_split query \
  --n_epochs 150 \
  --n_enc 512 \
  --sched_factor 0.5 \
  --dropout 0.5 \
  --lr 0.001 \
  --copy_sup \
  --TEST \
  > eval4.out 2> eval4.err
