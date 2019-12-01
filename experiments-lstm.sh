#!/bin/bash

DATABASE=(0 1)
LSTM_OUT=(196 250)
EMBED_DIM=(50 100 200 300)
DROPOUT=(0.3 0.5 0.7)
BATCHSIZE=(512 1024 2048)
for i in ${DATABASE[@]};
do
  echo -e "DATABASE "  $i
  for j in ${LSTM_OUT[@]};
  do
    echo -e "\tLSTM_OUT "  $j
    for k in ${EMBED_DIM[@]};
    do
      echo -e "\t\tEMBED_DIM "  $k
      for l in ${DROPOUT[@]};
      do
        echo -e "\t\t\tDROPOUT "  $l
        for m in ${BATCHSIZE[@]};
        do
          echo -e "\t\t\t\tBATCHSIZE "  $m
          python3 lstm.py $m $l $k $k $i 1> /dev/null 2> /dev/null
        done
      done
    done
  done
done

exit

echo "DATABASE 0"
echo "LSTM_OUT 196"
echo "\tEMBED_DIM 50"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.3 50 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.3 50 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.3 50 196 0 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 50 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 50 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 50 196 0 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.7 50 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.7 50 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.7 50 196 0 1> /dev/null 2> /dev/null


echo "\tEMBED_DIM 100"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.3 100 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.3 100 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.3 100 196 0 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 100 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 100 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 100 196 0 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.7 100 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.7 100 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.7 100 196 0 1> /dev/null 2> /dev/null


echo "\tEMBED_DIM 200"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.3 200 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.3 200 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.3 200 196 0 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 200 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 200 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 200 196 0 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.7 200 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.7 200 196 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.7 200 196 0 1> /dev/null 2> /dev/null

echo "LSTM_OUT 250"
echo "\tEMBED_DIM 50"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.3 50 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.3 50 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.3 50 250 0 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 50 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 50 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 50 250 0 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.7 50 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.7 50 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.7 50 250 0 1> /dev/null 2> /dev/null

echo "\tEMBED_DIM 100"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.3 100 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.3 100 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.3 100 250 0 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 100 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 100 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 100 250 0 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.7 100 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.7 100 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.7 100 250 0 1> /dev/null 2> /dev/null

echo "\tEMBED_DIM 200"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.3 200 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.3 200 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.3 200 250 0 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 200 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 200 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 200 250 0 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.7 200 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.7 200 250 0 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.7 200 250 0 1> /dev/null 2> /dev/null

echo "DATABASE 1"
echo "LSTM_OUT 196"
echo "\tEMBED_DIM 50"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.3 50 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.3 50 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.3 50 196 1 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 50 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 50 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 50 196 1 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.7 50 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.7 50 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.7 50 196 1 1> /dev/null 2> /dev/null


echo "\tEMBED_DIM 100"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.3 100 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.3 100 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.3 100 196 1 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 100 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 100 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 100 196 1 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.7 100 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.7 100 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.7 100 196 1 1> /dev/null 2> /dev/null


echo "\tEMBED_DIM 200"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.3 200 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.3 200 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.3 200 196 1 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 200 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 200 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 200 196 1 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.7 200 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.7 200 196 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.7 200 196 1 1> /dev/null 2> /dev/null

echo "LSTM_OUT 250"
echo "\tEMBED_DIM 50"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.3 50 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.3 50 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.3 50 250 1 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 50 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 50 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 50 250 1 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.7 50 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.7 50 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.7 50 250 1 1> /dev/null 2> /dev/null

echo "\tEMBED_DIM 100"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.3 100 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.3 100 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.3 100 250 1 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 100 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 100 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 100 250 1 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.7 100 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.7 100 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.7 100 250 1 1> /dev/null 2> /dev/null

echo "\tEMBED_DIM 200"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.3 200 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.3 200 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.3 200 250 1 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 200 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 200 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 200 250 1 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.7 200 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.7 200 250 1 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.7 200 250 1 1> /dev/null 2> /dev/null
