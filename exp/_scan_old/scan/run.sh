#!/bin/sh

python -u ../../train.py \
  --dataset scan \
  --dedup \
  --sched_factor 1.0 \
  >train.out 2>train.err
