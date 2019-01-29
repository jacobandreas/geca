#!/bin/sh

home="../../.."

for i in `seq 0 9`
do

  python -u $home/eval.py \
    --dataset scan \
    --invert \
    --seed $i \
    --scan_data_dir /data/jda/SCAN \
    --augment ../../scan_jump/retrieval/composed.$i.json \
    --dedup \
    --aug_ratio 0.3 \
    --n_epochs 150 \
    --n_enc 512 \
    --sched_factor 0.5 \
    --dropout 0.5 \
    --lr 0.001 \
    --notest_curve \
    --TEST \
    > eval.$i.out 2> eval.$i.err

done
