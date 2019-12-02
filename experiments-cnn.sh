#!/bin/bash

DATABASE=(0 1)
FILTERS=(196 250)
EMBED_DIM=(50 100 200 300)
DROPOUT=(0.3 0.5 0.7)
BATCHSIZE=(512 1024 2048)
for i in ${DATABASE[@]};
do
  echo -e "DATABASE "  $i
  for j in ${FILTERS[@]};
  do
    echo -e "\tFILTERS "  $j
    for k in ${EMBED_DIM[@]};
    do
      echo -e "\t\tEMBED_DIM "  $k
      for l in ${DROPOUT[@]};
      do
        echo -e "\t\t\tDROPOUT "  $l
        for m in ${BATCHSIZE[@]};
        do
          echo -e "\t\t\t\tBATCHSIZE "  $m
          python3 cnn.py $m $l $k $j $i 1> /dev/null 2> /dev/null
        done
      done
    done
  done
done
