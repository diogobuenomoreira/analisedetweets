#####ESTE PROGRAMA REALIZA O TREINAMENTO DE UMA REDE NEURAL PARA CLASSIFICAR
#####TWEETS COM DISCURSO DE ODIO RACISTA OU SEXISTA UTILIZANDO A ARQUITETURA cnn com glove
import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, Flatten, MaxPooling1D
from keras.preprocessing import text

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

#Argumentos deste programa
print("\t\t\t\tpython3 cnn-glove.py [batch size] [dropout] [embed_dim] [filters] [database] [vocabsize] [trainable]")

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

trainable = False
if len(sys.argv) > 7:
    if sys.argv[7] == '1':
        trainable = True
    if sys.argv[7] == '0':
        trainable = False

#Leitura dos dados
train_df = pd.read_csv('data/train-kaggle.csv')
resultsdir = 'results-glove-kaggle'
if len(sys.argv) > 5:
    if sys.argv[5] == '1':
        train_df = pd.read_csv('data/train-git.csv')
        resultsdir = 'results-glove-git'
    if sys.argv[5] == '0':
        train_df = pd.read_csv('data/train-kaggle.csv')
        resultsdir = 'results-glove-kaggle'
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

#total_data possui todos os dados de entrada e realiza a limpeza dos textos
x_tr = x_tr.apply(clean_up)

#Realiza a tokenizacao
tokenizer = text.Tokenizer(num_words = vocabsize,split=' ')
tokenizer.fit_on_texts(x_tr)
x_tr = tokenizer.texts_to_matrix(x_tr)
#padroniza o tamanho dos tweets
x_tr = sequence.pad_sequences(x_tr, maxlen = x_tr.shape[1])

embeddings_index = dict()
f = open('data/glove.6B.'+str(embed_dim)+'d.txt',encoding="utf8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((vocabsize, embed_dim))

# prepare embedding matrix

embedding_matrix = np.zeros((vocabsize, embed_dim))
for word, index in tokenizer.word_index.items():
  if index > vocabsize - 1:
      break
  else:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[index] = embedding_vector


#Criacao da Rede Neural
hiddendims = filters
kernelsize = 3
model = Sequential()
model.add(Embedding(vocabsize, embed_dim, input_length = x_tr.shape[1],
                weights = [embedding_matrix], trainable = trainable))
model.add(Conv1D(filters, kernelsize, padding = 'Valid', activation = 'relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(hiddendims, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#Divisao dos dados de treinamento em treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(x_tr,y_tr, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

#Treinamento do modelo
starttime=time()
r = model.fit(X_train, Y_train, batch_size = batchsize, epochs = 7,
          validation_data=(X_test, Y_test))
time = time() - starttime

#Teste do modelo
preds = model.predict(X_test)

#Resultados
fpr, tpr, thresh = roc_curve(Y_test, preds)
print('Accuracy: {}'.format(round(accuracy_score(Y_test, preds.round()),4)))
print('F1 score: {}'.format(round(f1_score(Y_test, preds.round()),4)))
print('AUC-ROC: {}'.format(round(roc_auc_score(Y_test, preds.round()),4)))
print('Confusion matrix')
print(confusion_matrix(Y_test,preds.round()))

if trainable == True:
    trainable = 1
else:
    trainable = 0

file = resultsdir + "/cnn-glove-" + str(embed_dim) + "x" + str(filters) + "_" + str(dropout) + "_" + str(batchsize) + "_" +str(vocabsize) + "_"+str(trainable)+".out"
f = open(file,"w+")
f.write(str(embed_dim) + ";" + str(filters) + ";" + str(dropout) + ";" + str(batchsize) + ";" + str(vocabsize) + ";" + str(trainable) + ";" + str(len(r.history["accuracy"])) + ";" + str(round(time,4)) + ";" + str(round(accuracy_score(Y_test, preds.round()),4)) + ";" + str(round(f1_score(Y_test, preds.round()),4)) + ";" + str(round(roc_auc_score(Y_test, preds.round()),4)) + ";" + str(confusion_matrix(Y_test,preds.round())[0][0]) + ";" + str(confusion_matrix(Y_test,preds.round())[0][1]) + ";" + str(confusion_matrix(Y_test,preds.round())[1][0]) + ";" + str(confusion_matrix(Y_test,preds.round())[1][1]) + "\n")
