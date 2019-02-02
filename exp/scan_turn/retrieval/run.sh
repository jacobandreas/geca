#!/bin/sh

home="../../.."

for i in `seq 0 9`
do

  python -u $home/compose.py \
    --dataset scan \
    --scan_data_dir /data/jda/SCAN \
    --scan_split template_split \
    --scan_file template_around_right \
    --dedup \
    --wug_size 1 \
    --seed $i \
    --model_type retrieval \
    --compute_adjacency \
    --n_sample 1000 \
    --write "composed.$i.json" \
    --nouse_trie \
    --max_comp_len 40 \
    --max_adjacencies 1000 \
    --TEST \
    > compose.$i.out 2> compose.$i.err

  python -u $home/eval.py \
    --dataset scan \
    --seed $i \
    --scan_data_dir /data/jda/SCAN \
    --scan_split template_split \
    --scan_file template_around_right \
    --augment composed.$i.json \
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
