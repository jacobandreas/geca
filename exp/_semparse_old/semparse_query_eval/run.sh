#!/bin/sh

python -u ../../compose.py \
  --dataset semparse \
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
  --seed 1 \
  --semparse_split query \
  --augment "composed.json" \
  --aug_ratio 0.5 \
  --n_epochs 150 \
  --n_enc 512 \
  --sched_factor 0.5 \
  --dropout 0.5 \
  --lr 0.001 \
  --copy_sup \
  > eval5.out 2> eval5.err

# <null> has wug_limit 50, aug_ratio .333
# 1 has wug_limit 25
# 2 has aug_ratio 0.1
# 3 has aug_ratio 0.332
# 4 has aug_ratio 0.5
# 5 has seed 1
