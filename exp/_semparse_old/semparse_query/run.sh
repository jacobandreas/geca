#!/bin/sh

python -u ../../train.py \
  --dataset semparse \
  --compute_adjacencies \
  --wug_limit 10 \
  --semparse_split query \
  --n_epochs 300 \
  --n_enc 512 \
  --sched_factor 1.0 \
  --dropout 0.2 \
  --lr 0.0003 \
  > train.out 2> train.err
