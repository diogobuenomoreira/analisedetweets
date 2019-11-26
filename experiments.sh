#!/bin/bash

echo "LSTM_OUT 196"
echo "\tEMBED_DIM 64"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
#python3 lstm.py 512 0.3 64 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
#python3 lstm.py 1024 0.3 64 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
#python3 lstm.py 2048 0.3 64 196 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 64 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 64 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 64 196 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
#python3 lstm.py 512 0.7 64 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
#python3 lstm.py 1024 0.7 64 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
#python3 lstm.py 2048 0.7 64 196 1> /dev/null 2> /dev/null


echo "\tEMBED_DIM 128"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
#python3 lstm.py 512 0.3 128 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
#python3 lstm.py 1024 0.3 128 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
#python3 lstm.py 2048 0.3 128 196 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 128 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 128 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 128 196 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
#python3 lstm.py 512 0.7 128 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
#python3 lstm.py 1024 0.7 128 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
#python3 lstm.py 2048 0.7 128 196 1> /dev/null 2> /dev/null


echo "\tEMBED_DIM 256"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
#python3 lstm.py 512 0.3 256 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
#python3 lstm.py 1024 0.3 256 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
#python3 lstm.py 2048 0.3 256 196 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 256 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 256 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 256 196 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
#python3 lstm.py 512 0.7 256 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
#python3 lstm.py 1024 0.7 256 196 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
#python3 lstm.py 2048 0.7 256 196 1> /dev/null 2> /dev/null

echo "LSTM_OUT 250"
echo "\tEMBED_DIM 64"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
#python3 lstm.py 512 0.3 64 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
#python3 lstm.py 1024 0.3 64 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
#python3 lstm.py 2048 0.3 64 250 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 64 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 64 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 64 250 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
#python3 lstm.py 512 0.7 64 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
#python3 lstm.py 1024 0.7 64 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
#python3 lstm.py 2048 0.7 64 250 1> /dev/null 2> /dev/null

echo "\tEMBED_DIM 128"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
#python3 lstm.py 512 0.3 128 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
#python3 lstm.py 1024 0.3 128 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
#python3 lstm.py 2048 0.3 128 250 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 128 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 128 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 128 250 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
#python3 lstm.py 512 0.7 128 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
#python3 lstm.py 1024 0.7 128 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
#python3 lstm.py 2048 0.7 128 250 1> /dev/null 2> /dev/null

echo "\tEMBED_DIM 256"
echo "\t\tDROPOUT 0.3"
echo "\t\t\tBATCHSIZE 512"
#python3 lstm.py 512 0.3 256 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
#python3 lstm.py 1024 0.3 256 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
#python3 lstm.py 2048 0.3 256 250 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.5"
echo "\t\t\tBATCHSIZE 512"
python3 lstm.py 512 0.5 256 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
python3 lstm.py 1024 0.5 256 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
python3 lstm.py 2048 0.5 256 250 1> /dev/null 2> /dev/null

echo "\t\tDROPOUT 0.7"
echo "\t\t\tBATCHSIZE 512"
#python3 lstm.py 512 0.7 256 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 1024"
#python3 lstm.py 1024 0.7 256 250 1> /dev/null 2> /dev/null
echo "\t\t\tBATCHSIZE 2048"
#python3 lstm.py 2048 0.7 256 250 1> /dev/null 2> /dev/null
