#!/bin/sh

python -u ../../train.py \
  --dataset semparse \
  --semparse_split query \
  --n_epochs 150 \
  --n_enc 512 \
  --sched_factor 1.0 \
  --dropout 0.5 \
  --lr 0.001 \
  > train.out 2> train.err
