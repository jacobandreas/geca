#!/bin/sh

home="../../.."
dataset=/data/jda/text2sql-data/data/non-sql-data/geography-logic.txt \

for i in `seq 0 9`
do

  python -u $home/compose.py \
    --dataset semparse \
    --semparse_dataset $dataset \
    --semparse_mrl logic \
    --semparse_split question \
    --seed $i \
    --model_type retrieval \
    --wug_limit 50 \
    --compute_adjacency \
    --n_sample 1000 \
    --write "composed.$i.json" \
    --TEST \
    > compose.$i.out 2> compose.$i.err

done
