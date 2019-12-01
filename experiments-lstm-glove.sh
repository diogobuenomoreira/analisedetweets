#!/bin/bash

VOCABSIZE=(1000 2000 10000 20000)
TRAINABLE=(0 1)

for i in ${VOCABSIZE[@]};
do
  echo -e "VOCABSIZE "  $i
  for j in ${TRAINABLE[@]};
  do
    echo -e "\tLSTM_OUT "  $j
    python3 lstm-glove.py 250 50 0.7 512 0 $i $j 1> /dev/null 2> /dev/null
  done
done
