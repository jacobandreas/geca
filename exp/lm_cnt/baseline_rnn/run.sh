#!/bin/sh

home="../../.."

for i in `seq 0 9`
do

  python -u $home/eval_lm.py \
    --dataset lm \
    --seed $i \
    --lm_data_dir /Users/jda/code/babel/cnt \
    --n_epochs 100 \
    --n_epoch_batches 5 \
    --n_enc 512 \
    --dropout 0.5 \
    --lr 0.001 \
    #> eval.$i.out 2> eval.$i.err

  exit 0

done
