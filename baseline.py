# Importing libraries
import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

# import data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

for i in range(0,10):
    print(train_df['tweet'][i])

x_tr = train_df['tweet']
y_tr = train_df['label']
holdout_test = test_df['tweet']

def clean_up(text):
    text = re.sub('\d+','',text)
    text = re.sub(r'[^\w\s]','',text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

total_data = x_tr.append(holdout_test)
total_data = total_data.apply(clean_up)

for i in range(0,10):
    print(clean_up(train_df['tweet'][i]))


print(train_df.groupby('label')['label'].count())
print('The shape of train is {}'.format(x_tr.shape))
print('The shape of holdout test is {}'.format(holdout_test.shape))

for i in range(0,10):
    print(x_tr[i])

max_fatures = 2000
def tokenize(text_frame):
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(text_frame.values)
    tokenized = tokenizer.texts_to_sequences(text_frame.values)
    padded_seq = pad_sequences(tokenized)
    return padded_seq

total_data = tokenize(total_data)
x_tr = total_data[:x_tr.shape[0]]
holdout_test = total_data[x_tr.shape[0]:]
print(train_df.groupby('label')['label'].count())
print('The shape of train is {}'.format(x_tr.shape))
print('The shape of holdout test is {}'.format(holdout_test.shape))

print(x_tr.shape[1])

embed_dim = 128
lstm_out = 196

# creating the Neural Network
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = x_tr.shape[1]))

model.add(LSTM(lstm_out, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

X_train, X_test, Y_train, Y_test = train_test_split(x_tr,y_tr, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

# Training the model
model.fit(X_train, Y_train, epochs = 7, batch_size=1024, verbose = 1,
validation_data = (X_test,Y_test))


# Testing the model
preds = model.predict(X_test)

fpr, tpr, thresh = roc_curve(Y_test, preds)

# Accuracy tests
print('Accuracy: {}'.format(round(accuracy_score(Y_test, preds.round()),4)))
print('F1 score: {}'.format(round(f1_score(Y_test, preds.round()),4)))
print('AUC-ROC: {}'.format(round(roc_auc_score(Y_test, preds.round()),4)))
print('Confusion matrix')
print(confusion_matrix(Y_test,preds.round()))

plt.plot(fpr,tpr)
plt.title('AUC-ROC Curves')
plt.xlabel('FPR')
plt.ylabel('TPR')

holdout_preds = model.predict(holdout_test)
