#!/bin/sh

dataset=/data/jda/text2sql-data/data/non-sql-data/geography-logic.txt \
python=~/anaconda2/bin/python2
code=../../../3p/jia/src

OMP_NUM_THREADS=1
THEANO_FLAGS=blas.ldflags=-lopenblas 

for i in `seq 0 9`
do

  $python $code/py/main.py \
    -d 200 \
    -i 100 \
    -o 100 \
    -p attention \
    -u 1 \
    -t 15,5,5,5 \
    -c lstm \
    -m attention \
    --stats-file stats.$i.json \
    --domain geoquery \
    -k 5 \
    --dev-seed $i \
    --model-seed $i \
    --data $dataset \
    --split query \
    --save-file params.$i \
    > train.$i.out 2> train.$i.err

done
