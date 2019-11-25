#####ESTE PROGRAMA REALIZA O TREINAMENTO DE UMA REDE NEURAL PARA CLASSIFICAR
#####TWEETS COM DISCURSO DE ODIO RACISTA OU SEXISTA UTILIZANDO A ARQUITETURA lstm


# Importing libraries
import sys
import os
import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from time import time

#Leitura dos dados de treinamento e teste
train_df = pd.read_csv('train.csv')
#train_df = pd.read_csv('classificados.csv')
test_df = pd.read_csv('test.csv')

#x_tr possui os dados de entrada para o treinamento
x_tr = train_df['tweet']
#y_tr possui os dados de saida (rotulo) para o treinamento
y_tr = train_df['label']

print(len(y_tr.values.tolist()))
print(y_tr.values.tolist().count(1))
print(y_tr.values.tolist().count(0))
print(y_tr.values.tolist().count(1)+y_tr.values.tolist().count(0))
