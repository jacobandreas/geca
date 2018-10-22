#!/bin/sh

python -u ../../compose.py \
  --dataset semparse \
  --model_dir ../semparse/model \
  --model "model.00149.chk" \
  --n_sample 500 \
  --wug_limit 10 \
  --write "composed.json" \
#  > compose.out 2> compose.err
#
#python -u ../../eval.py \
#  --dataset semparse \
#  --augment "composed.json" \
#  --n_epochs 150 \
#  --n_enc 512 \
#  --sched_factor 0.5 \
#  --dropout 0.5 \
#  --lr 0.001 \
#  --copy_sup \
#  > eval.comp.out 2> eval.comp.err
#  #--TEST \
