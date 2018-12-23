#!/bin/sh

home="../../.."

for i in `seq 0 9`
do

  python -u $home/eval.py \
    --dataset semparse \
    --semparse_dataset /x/jda/data/text2sql-data/data/geography.json \
    --semparse_mrl sql \
    --seed $i \
    --semparse_split query \
    --n_epochs 150 \
    --n_enc 512 \
    --sched_factor 0.5 \
    --dropout 0.5 \
    --lr 0.001 \
    --copy_sup \
    > eval.val$i.out 2> eval.val$i.err
    #--TEST \

done