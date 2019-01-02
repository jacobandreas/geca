#!/bin/sh

python -u ../../train_ctx.py \
  --dataset semparse \
  --semparse_dataset geography-logic.txt \
  --semparse_mrl logic \
  --semparse_split query \
  --seed 0 \
  --hard_comp_table \
  --wug_limit 50 \
  --wug_size 4 \
  --n_epochs 100 \
  --n_enc 512 \
  --sched_factor 1.0 \
  --dropout 0.2 \
  --lr 0.0003 \
  > train.out 2> train.err
