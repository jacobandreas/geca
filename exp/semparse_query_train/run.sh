#!/bin/sh

python -u ../../train.py \
  --dataset semparse \
  --semparse_dataset geography-logic.txt \
  --semparse_mrl logic \
  --semparse_split query \
  --seed 0 \
  --compute_adjacencies \
  --wug_limit 50 \
  --n_epochs 300 \
  --n_enc 512 \
  --sched_factor 1.0 \
  --dropout 0.2 \
  --lr 0.0003 \
  > train.out 2> train.err
