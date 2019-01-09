#!/bin/sh

home="../../.."
dataset=/data/jda/text2sql-data/data/non-sql-data/geography-logic.txt \

python -u $home/info.py \
  --dataset semparse \
  --semparse_dataset $dataset \
  --semparse_mrl logic \
  --semparse_split query \
  > info.out 2> /dev/null

for i in `seq 0 9`
do

  python -u $home/compose.py \
    --dataset semparse \
    --semparse_dataset $dataset \
    --semparse_mrl logic \
    --semparse_split query \
    --seed $i \
    --model_type retrieval \
    --wug_limit 50 \
    --compute_adjacency \
    --n_sample 1000 \
    --write "composed.$i.json" \
    --TEST \
    > compose.$i.out 2> compose.$i.err

  python -u $home/eval.py \
    --dataset semparse \
    --semparse_dataset $dataset \
    --semparse_mrl logic \
    --semparse_split query \
    --seed $i \
    --augment "composed.$i.json" \
    --aug_ratio 0.3 \
    --n_epochs 150 \
    --n_enc 512 \
    --sched_factor 0.5 \
    --dropout 0.5 \
    --lr 0.001 \
    --copy_sup \
    --TEST \
    > eval.$i.out 2> eval.$i.err

done
