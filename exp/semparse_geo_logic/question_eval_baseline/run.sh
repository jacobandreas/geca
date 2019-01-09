#!/bin/sh

home="../../.."
dataset=/data/jda/text2sql-data/data/non-sql-data/geography-logic.txt \

python -u $home/info.py \
  --dataset semparse \
  --semparse_dataset $dataset \
  --semparse_mrl logic \
  --semparse_split question \
  > info.out 2> /dev/null

for i in `seq 0 9`
do

  python -u $home/eval.py \
    --dataset semparse \
    --semparse_dataset $dataset \
    --semparse_mrl logic \
    --semparse_split question \
    --seed $i \
    --n_epochs 150 \
    --n_enc 200 \
    --n_emb 100 \
    --sched_factor 0.5 \
    --dropout 0.5 \
    --lr 0.001 \
    --copy_sup \
    --beam 5 \
    --TEST \
    > eval.$i.out 2> eval.$i.err

done
