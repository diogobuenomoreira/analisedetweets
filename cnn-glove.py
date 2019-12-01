import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
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

#some configuration
EMBEDDING_DIMS = 50
FILTERS = 250
KERNEL_SIZE = 3
HIDDEN_DIMS = 100
EPOCHS = 7
DROPOUT_RATE = 0.5

# import data
train_df = pd.read_csv('train.csv')


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

#Tamanho do vocabulario
VOCAB_SIZE = 2000
MAXLEN = 100

#Realiza a tokenizacao
tokenizer = text.Tokenizer(num_words=VOCAB_SIZE,split=' ')
tokenizer.fit_on_texts(x_tr)
x_tr = tokenizer.texts_to_sequences(x_tr)
#X_test= tokenizer.texts_to_sequences(X_test.values)
#padroniza o tamanho dos tweets
x_tr = sequence.pad_sequences(x_tr, maxlen=MAXLEN)
#X_test = sequence.pad_sequences(X_test, maxlen=MAXLEN)

embeddings_index = dict()
f = open('glove.6B.100d.txt',encoding="utf8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((VOCAB_SIZE, 100))

# prepare embedding matrix

embedding_matrix = np.zeros((VOCAB_SIZE, 100))
for word, index in tokenizer.word_index.items():
  if index > VOCAB_SIZE - 1:
      break
  else:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[index] = embedding_vector


#Criacao da Rede Neural
model = Sequential()
model.add(Embedding(VOCAB_SIZE, 100, input_length = MAXLEN, weights = [embedding_matrix], trainable = False))
model.add(Conv1D(FILTERS, KERNEL_SIZE, padding = 'Valid', activation = 'relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(HIDDEN_DIMS, activation='relu'))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#Callback para o treinamento parar quando a acuracia parar de deminiuir
#callback = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='max', baseline=None, restore_best_weights=False)

#Divisao dos dados de treinamento em treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(x_tr,y_tr, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

#Treinamento do modelo
starttime=time()
r = model.fit(X_train, Y_train, batch_size = batch_size, epochs = EPOCHS,
          validation_data=(X_test, Y_test))
time = time() - starttime

#Grafico
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

#Acuracia
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

#Teste do modelo
preds = model.predict(X_test)

#Resultados
fpr, tpr, thresh = roc_curve(Y_test, preds)
print('Accuracy: {}'.format(round(accuracy_score(Y_test, preds.round()),4)))
print('F1 score: {}'.format(round(f1_score(Y_test, preds.round()),4)))
print('AUC-ROC: {}'.format(round(roc_auc_score(Y_test, preds.round()),4)))
print('Confusion matrix')
print(confusion_matrix(Y_test,preds.round()))

file = resultsdir + "/cnn-glove" + str(embed_dim) + "x" + str(lstm_out) + "_" + str(dropout) + "_" + str(batchsize) + ".out"
f = open(file,"w+")
f.write(str(embed_dim) + ";" + str(lstm_out) + ";" + str(dropout) + ";" + str(batchsize) + ";" + str(len(r.history["accuracy"])) + ";" + str(round(time,4)) + ";" + str(round(accuracy_score(Y_test, preds.round()),4)) + ";" + str(round(f1_score(Y_test, preds.round()),4)) + ";" + str(round(roc_auc_score(Y_test, preds.round()),4)) + ";" + str(confusion_matrix(Y_test,preds.round())[0][0]) + ";" + str(confusion_matrix(Y_test,preds.round())[0][1]) + ";" + str(confusion_matrix(Y_test,preds.round())[1][0]) + ";" + str(confusion_matrix(Y_test,preds.round())[1][1]) + "\n")
