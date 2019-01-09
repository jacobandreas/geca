#!/bin/sh

for i in `seq 0 9`
do

  python -u ../../../eval.py \
    --dataset semparse \
    --semparse_dataset /data/jda/text2sql-data/data/non-sql-data/geography-logic.txt \
    --semparse_mrl logic \
    --seed $i \
    --semparse_split query \
    --n_epochs 150 \
    --n_enc 512 \
    --sched_factor 0.5 \
    --dropout 0.5 \
    --lr 0.001 \
    --copy_sup \
    --TEST \
    > eval.$i.out 2> eval.$i.err

done
