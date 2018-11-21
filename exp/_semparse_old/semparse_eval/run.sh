#!/bin/sh

#python -u ../../compose.py \
#  --dataset semparse \
#  --compute_adjacencies \
#  --wug_limit 30 \
#  --model_dir ../semparse/model \
#  --model "model.00299.chk" \
#  --n_sample 2000 \
#  --write "composed.json" \
#  > compose.out 2> compose.err

python -u ../../eval.py \
  --dataset semparse \
  --n_epochs 150 \
  --n_enc 512 \
  --sched_factor 0.5 \
  --dropout 0.5 \
  --lr 0.001 \
  --copy_sup \
  > eval.retrieve.out 2> eval.retrieve.err
  #--TEST \
  #--augment "composed.json" \
  #--aug_ratio 0 \
