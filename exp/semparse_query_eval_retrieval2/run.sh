#!/bin/sh

python -u ../../compose.py \
  --dataset semparse \
  --semparse_dataset geography-logic.txt \
  --semparse_mrl logic \
  --semparse_split query \
  --seed 0 \
  --compute_adjacencies \
  --wug_limit 50 \
  --n_sample 2000 \
  --write "composed.json" \
  > compose.out 2> compose.err
  #--model_dir ../semparse_query/model \
  #--model "model.00299.chk" \
  #--TEST \

for i in `seq 0 4`
do

  python -u ../../eval.py \
    --dataset semparse \
    --semparse_dataset geography-logic.txt \
    --semparse_mrl logic \
    --semparse_split query \
    --seed $i \
    --augment "composed.json" \
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
