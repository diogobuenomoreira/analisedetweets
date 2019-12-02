#!/bin/bash


python3 lstm-glove.py 2048 0.5 300 250 0 1000 1 1> /dev/null 2> /dev/null
python3 lstm-glove.py 1024 0.5 50 250 0 1000 1 1> /dev/null 2> /dev/null

python3 lstm-glove.py 512 0.7 100 250 1 1000 1 1> /dev/null 2> /dev/null
python3 lstm-glove.py 512 0.7 300 250 1 1000 1 1> /dev/null 2> /dev/null
python3 lstm-glove.py 512 0.3 200 250 1 1000 1 1> /dev/null 2> /dev/null
python3 lstm-glove.py 1024 0.7 200 196 1 1000 1 1> /dev/null 2> /dev/null
python3 lstm-glove.py 512 0.7 200 196 1 1000 1 1> /dev/null 2> /dev/null

VOCABSIZE=(2000 2500)
TRAINABLE=(0 1)

for j in ${TRAINABLE[@]};
do
    echo -e "\tTRAINABLE "  $j
    for i in ${VOCABSIZE[@]};
    do
      echo -e "VOCABSIZE "  $i
      python3 lstm-glove.py 512 0.7 50 250 0 $i $j 1> /dev/null 2> /dev/null
      python3 lstm-glove.py 1024 0.3 100 196 0 $i $j 1> /dev/null 2> /dev/null
      python3 lstm-glove.py 512 0.5 100 196 0 $i $j 1> /dev/null 2> /dev/null
      python3 lstm-glove.py 2048 0.5 300 250 0 $i $j 1> /dev/null 2> /dev/null
      python3 lstm-glove.py 1024 0.5 50 250 0 $i $j 1> /dev/null 2> /dev/null

      python3 lstm-glove.py 512 0.7 100 250 1 $i $j 1> /dev/null 2> /dev/null
      python3 lstm-glove.py 512 0.7 300 250 1 $i $j 1> /dev/null 2> /dev/null
      python3 lstm-glove.py 512 0.3 200 250 1 $i $j 1> /dev/null 2> /dev/null
      python3 lstm-glove.py 1024 0.7 200 196 1 $i $j 1> /dev/null 2> /dev/null
      python3 lstm-glove.py 512 0.7 200 196 1 $i $j 1> /dev/null 2> /dev/null
  done
done
