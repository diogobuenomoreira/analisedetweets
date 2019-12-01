#####ESTE PROGRAMA REALIZA O TREINAMENTO DE UMA REDE NEURAL PARA CLASSIFICAR

import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, Flatten, MaxPooling1D
from keras.callbacks.callbacks import EarlyStopping
from keras.preprocessing import text

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

#Argumentos deste programa
print("\t\t\t\tpython3 cnn.py [batch size] [dropout] [embed_dim] [filters] [database] [vocabsize]")

#Argumentos
batchsize = 1024
if len(sys.argv) > 1:
    batchsize = int(sys.argv[1])

dropout = 0.3
if len(sys.argv) > 2:
    dropout = float(sys.argv[2])

embed_dim = 100
if len(sys.argv) > 3:
    embed_dim = int(sys.argv[3])

filters = 196
if len(sys.argv) > 4:
    filters = int(sys.argv[4])

vocabsize = 2000
if len(sys.argv) > 6:
    vocabsize = int(sys.argv[6])

#Leitura dos dados
train_df = pd.read_csv('data/train-kaggle.csv')
resultsdir = 'results-kaggle'
if len(sys.argv) > 5:
    if sys.argv[5] == '1':
        train_df = pd.read_csv('data/train-git.csv')
        resultsdir = 'results-git'
    if sys.argv[5] == '0':
        train_df = pd.read_csv('data/train-kaggle.csv')
        resultsdir = 'results-kaggle'
if not os.path.exists(resultsdir):
    os.makedirs("./" + resultsdir)

#x_tr possui os dados de entrada para o treinamento
x_tr = train_df['tweet']
#y_tr possui os dados de saida (rotulo) para o treinamento
y_tr = train_df['label']

#Funcao para limpar o texto
def clean_up(text):
    #Retira mencao a outros usuarios
    text = re.sub('@\S*','',text)
    #Retira urls
    text = re.sub('http\S*','',text)
    #Retira numeros
    text = re.sub('\d+','',text)
    #Retira os termos 'user' e 'RT'
    text = re.sub('user','',text)
    text = re.sub('RT','',text)
    #Retira caracteres especiais e emojis
    text = re.sub(r'[^\w\s]','',text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text


#x_tr possui todos os dados de entrada e realiza a limpeza dos textos
x_tr = x_tr.apply(clean_up)

#Funcao para realizar a tokenizacao
def tokenize(text_frame):
    tokenizer = Tokenizer(num_words=vocabsize, split=' ')
    #cria um vocabulario com base nas strings do argumento em ordem de frequencia
    tokenizer.fit_on_texts(text_frame.values)
    #troca as palavras pelos seus identificadores
    tokenized = tokenizer.texts_to_sequences(text_frame.values)
    #padroniza o tamanho dos tweets
    padded_seq = pad_sequences(tokenized)
    return padded_seq

#realiza tokenizacao em todos os tweets
x_tr = tokenize(x_tr)

#Criacao da Rede Neural
hiddendims = filters
kernelsize = 3
model = Sequential()
#sem usar Transfer Learning
model.add(Embedding(vocabsize, embed_dim, input_length = x_tr.shape[1]))
model.add(Conv1D(filters, kernelsize, padding = 'Valid', activation = 'relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(hiddendims, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#Callback para o treinamento parar quando a acuracia parar de deminiuir
callback = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='max', baseline=None, restore_best_weights=False)


#Divisao dos dados de treinamento em treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(x_tr,y_tr, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

#Treinamento do Modelo
starttime=time()
r = model.fit(X_train, Y_train, batch_size = batchsize, epochs = 100000000000,
          validation_data=(X_test, Y_test), callbacks=[callback])
time = time() - starttime

#Teste do Modelo
preds = model.predict(X_test)

#Resultados
fpr, tpr, thresh = roc_curve(Y_test, preds)
print('Accuracy: {}'.format(round(accuracy_score(Y_test, preds.round()),4)))
print('F1 score: {}'.format(round(f1_score(Y_test, preds.round()),4)))
print('AUC-ROC: {}'.format(round(roc_auc_score(Y_test, preds.round()),4)))
print('Confusion matrix')
print(confusion_matrix(Y_test,preds.round()))

file = resultsdir + "/cnn-" + str(embed_dim) + "x" + str(filters) + "_" + str(dropout) + "_" + str(batchsize) + ".out"
f = open(file,"w+")
f.write(str(embed_dim) + ";" + str(filters) + ";" + str(dropout) + ";" + str(batchsize) + ";" + str(len(r.history["accuracy"])) + ";" + str(round(time,4)) + ";" + str(round(accuracy_score(Y_test, preds.round()),4)) + ";" + str(round(f1_score(Y_test, preds.round()),4)) + ";" + str(round(roc_auc_score(Y_test, preds.round()),4)) + ";" + str(confusion_matrix(Y_test,preds.round())[0][0]) + ";" + str(confusion_matrix(Y_test,preds.round())[0][1]) + ";" + str(confusion_matrix(Y_test,preds.round())[1][0]) + ";" + str(confusion_matrix(Y_test,preds.round())[1][1]) + "\n")
