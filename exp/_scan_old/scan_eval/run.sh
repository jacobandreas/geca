#!/bin/sh

python -u ../../compose.py \
  --dataset scan \
  --dedup \
  --model_dir ../scan/model \
  --model "model.00149.chk" \
  --n_sample 1000 \
  --write "composed.json" \
  --wug_limit 2 \
  > compose.out 2> compose.err

python -u ../../eval.py \
  --dataset scan \
  --dedup \
  --augment "composed.json" \
  > eval.out 2> eval.err
