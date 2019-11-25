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

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks.callbacks import EarlyStopping

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

#Argumentos deste programa
print("\t\t\t\tpython3 baseline.py [batch size] [dropout]")

resultsdir = 'results3'
if not os.path.exists(resultsdir):
    os.makedirs("./" + resultsdir)

#Leitura dos dados de treinamento e teste
train_df = pd.read_csv('data/train.csv')
train_df = pd.read_csv('data/train3.csv')
#train_df = pd.read_csv('classificados.csv')
test_df = pd.read_csv('data/test.csv')

#print("DADOS DE ENTRADA ANTES DA LIMPEZA")
#for i in range(0,10):
#    print(train_df['tweet'][i])
#print("\n=================\n")

#x_tr possui os dados de entrada para o treinamento
x_tr = train_df['tweet']
#y_tr possui os dados de saida (rotulo) para o treinamento
y_tr = train_df['label']
#holdout_test os dados de entrada para o teste
holdout_test = test_df['tweet']

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
#total_data = x_tr.append(holdout_test)
total_data = x_tr
total_data = total_data.apply(clean_up)

#print("DADOS DE ENTRADA DEPOIS DA LIMPEZA")
#for i in range(0,10):
#    print(clean_up(train_df['tweet'][i]))
#print("\n=================\n")

#numero de tweets com cada um dos dois rotulos
#print(train_df.groupby('label')['label'].count())
#numero de tweets de treinamento
#print('The shape of train is {}'.format(x_tr.shape))
#numero de tweets de teste
#print('The shape of holdout test is {}'.format(holdout_test.shape))

#for i in range(0,10):
#    print(x_tr[i])

#Funcao para realizar a tokenizacao
max_fatures = 2000
def tokenize(text_frame):
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    #Updates internal vocabulary based on a list of texts.
    #This method creates the vocabulary index based on word frequency.
    #So if you give it something like, "The cat sat on the mat."
    #It will create a dictionary s.t. word_index["the"] = 1;
    #word_index["cat"] = 2 it is word -> index dictionary so every word gets
    #a unique integer value. 0 is reserved for padding.
    #So lower integer means more frequent word
    #(often the first few are stop words because they appear a lot).
    tokenizer.fit_on_texts(text_frame.values)
    #Transforms each text in texts to a sequence of integers.
    #So it basically takes each word in the text and replaces it with its
    #corresponding integer value from the word_index dictionary.
    tokenized = tokenizer.texts_to_sequences(text_frame.values)
    #Padroniza o tamanho dos tweets
    padded_seq = pad_sequences(tokenized)
    return padded_seq

#realiza tokenizacao em todos os tweets
total_data = tokenize(total_data)

#atribui a tokenizacao nos vetores de treinamento e teste
x_tr = total_data
#x_tr = total_data[:x_tr.shape[0]]
#holdout_test = total_data[x_tr.shape[0]:]

#numero de tweets com cada um dos dois rotulos
print(train_df.groupby('label')['label'].count())
#numero de tweets de treinamento
print('The shape of train is {}'.format(x_tr.shape))
#numero de tweets de teste
print('The shape of holdout test is {}'.format(holdout_test.shape))


#Argumentos
batchsize = 1024
if len(sys.argv) > 1:
    batchsize = int(sys.argv[1])

dropout = 0.3
if len(sys.argv) > 2:
    dropout = float(sys.argv[2])

embed_dim = 128
if len(sys.argv) > 3:
    embed_dim = int(sys.argv[3])

lstm_out = 196
if len(sys.argv) > 4:
    lstm_out = int(sys.argv[4])

#Criacao da Rede Neural
model = Sequential()
model.add(Embedding(max_fatures, embed_dim, input_length = x_tr.shape[1]))
model.add(LSTM(lstm_out, dropout=dropout, recurrent_dropout=dropout))
model.add(Dense(lstm_out,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())

#Callback para o treinamento parar quando a acuracia parar de deminiuir
callback = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='max', baseline=None, restore_best_weights=False)

#Divisao dos dados de treinamento em treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(x_tr,y_tr, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

#Treinamento do modelo
starttime=time()
r = model.fit(X_train, Y_train, epochs = 100000000000, batch_size=batchsize, verbose = 1, validation_data = (X_test,Y_test), callbacks=[callback])
time = time() - starttime

#Teste do modelo
preds = model.predict(X_test)


fpr, tpr, thresh = roc_curve(Y_test, preds)
# Accuracy tests
print('Accuracy: {}'.format(round(accuracy_score(Y_test, preds.round()),4)))
print('F1 score: {}'.format(round(f1_score(Y_test, preds.round()),4)))
print('AUC-ROC: {}'.format(round(roc_auc_score(Y_test, preds.round()),4)))
print('Confusion matrix')
print(confusion_matrix(Y_test,preds.round()))

file = resultsdir + "/lstm-" + str(embed_dim) + "x" + str(lstm_out) + "_" + str(dropout) + "_" + str(batchsize) + ".out"
f = open(file,"w+")
f.write(str(embed_dim) + ";" + str(lstm_out) + ";" + str(dropout) + ";" + str(batchsize) + ";" + str(len(r.history["accuracy"])) + ";" + str(round(time,4)) + ";" + str(round(accuracy_score(Y_test, preds.round()),4)) + ";" + str(round(f1_score(Y_test, preds.round()),4)) + ";" + str(round(roc_auc_score(Y_test, preds.round()),4)) + ";" + str(confusion_matrix(Y_test,preds.round())[0][0]) + ";" + str(confusion_matrix(Y_test,preds.round())[0][1]) + ";" + str(confusion_matrix(Y_test,preds.round())[1][0]) + ";" + str(confusion_matrix(Y_test,preds.round())[1][1]) + "\n")

plt.title('AUC-ROC Curves')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr,tpr)

holdout_preds = model.predict(holdout_test)
