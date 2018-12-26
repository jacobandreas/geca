#!/bin/sh

home="../../.."
dataset=/x/jda/data/text2sql-data/data/geography.json

python -u $home/info.py \
  --dataset semparse \
  --semparse_dataset $dataset \
  --semparse_mrl sql \
  --semparse_split query \
  > info.out 2> /dev/null

for i in `seq 0 9`
do

  python -u $home/compose.py \
    --dataset semparse \
    --semparse_dataset $dataset \
    --semparse_mrl sql \
    --semparse_split query \
    --seed $i \
    --model_type retrieval \
    --wug_limit 50 \
    --wug_size 4 \
    --wug_count 3 \
    --compute_adjacency \
    --n_sample 2000 \
    --write "composed.val$i.json" \
    > compose.val$i.out 2> compose.val$i.err
    #--TEST \

  python -u $home/eval.py \
    --dataset semparse \
    --semparse_dataset $dataset \
    --semparse_mrl sql \
    --semparse_split query \
    --seed $i \
    --augment "composed.val$i.json" \
    --aug_ratio 0.3 \
    --n_epochs 150 \
    --n_enc 512 \
    --sched_factor 0.5 \
    --dropout 0.5 \
    --lr 0.001 \
    --copy_sup \
    > eval.val$i.out 2> eval.val$i.err
    #--TEST \

done
