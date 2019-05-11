#!/bin/sh

home="../../.."
dataset="/data/jda/wikidata/formatted/eng"
kenlm="../../../../kenlm/build"

context=2
order=5

for seed in `seq 0 0`
do

  python -u $home/compose.py \
    --dataset lm \
    --lm_data_dir $dataset \
    --seed $seed \
    --model_type retrieval \
    --wug_limit 20 \
    --wug_size 2 \
    --wug_count 1 \
    --variants 1 \
    --template_sim window \
    --sim_window_size $context \
    --compute_adjacency \
    --n_sample 5000 \
    --write "composed.$seed.json" \
    --output_only \
    --max_comp_len 100 \
    > compose.$seed.out 2> compose.$seed.err

  python -u $home/fake_corpus.py \
    --dataset lm \
    --lm_data_dir $dataset \
    --augment "composed.$seed.json" \
    --write "augmented.$seed.txt" \
    > comb.$seed.out \
    2> comb.$seed.err

  $kenlm/bin/lmplz \
    -o $order \
    <$dataset/train.txt \
    >lm_base.$seed.arpa \
    2>train_base.$seed.err

  $kenlm/bin/lmplz \
    -o $order \
    < "augmented.$seed.txt" \
    > lm_aug.$seed.arpa \
    2> train_aug.$seed.err

  rm -f eval.$seed.out eval.$seed.err

  for ratio in 0 0.1 0.2 1
  do

    python -u $home/eval_lm.py \
      --dataset lm \
      --seed $seed \
      --lm_data_dir $dataset \
      --lm_file lm_base.$seed.arpa \
      --aug_lm_file lm_aug.$seed.arpa \
      --aug_ratio $ratio \
      --test_curve \
      --use_mkn \
      --TEST \
      >> eval.$seed.out 2>> eval.$seed.err

  done

done
