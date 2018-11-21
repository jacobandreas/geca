#!/bin/sh

python -u ../../compose.py \
  --dataset semparse \
  --semparse_dataset atis \
  --semparse_split query \
  --compute_adjacencies \
  --wug_limit 25 \
  --model_dir ../semparse_query/model \
  --model "model.00299.chk" \
  --n_sample 3000 \
  --write "composed.json" \
  > compose.out 2> compose.err

python -u ../../eval.py \
  --dataset semparse \
  --semparse_dataset atis \
  --semparse_split query \
  --augment "composed.json" \
  --aug_ratio 0.332 \
  --n_epochs 150 \
  --n_enc 512 \
  --sched_factor 0.5 \
  --dropout 0.5 \
  --lr 0.001 \
  --copy_sup \
  > eval.out 2> eval.err

# <null> has wug_limit 25, aug_ratio .332
