#!/bin/sh


for i in `seq 0 9`
do

  python -u ../../compose.py \
    --dataset semparse \
    --semparse_dataset geography-logic.txt \
    --semparse_mrl logic \
    --semparse_split question \
    --seed $i \
    --model_type retrieval \
    --wug_limit 50 \
    --build_comp_table \
    --n_sample 1000 \
    --write "composed.val$i.json" \
    > compose.val$i.out 2> compose.val$i.err
    #--TEST \

  python -u ../../eval.py \
    --dataset semparse \
    --semparse_dataset geography-logic.txt \
    --semparse_mrl logic \
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
