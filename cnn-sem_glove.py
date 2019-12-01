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
VOCAB_SIZE = 20000
MAXLEN = 100
BATCH_SIZE = 32
EMBEDDING_DIMS = 50
FILTERS = 250
KERNEL_SIZE = 3
HIDDEN_DIMS = 100
EPOCHS = 7

# import data
train_df = pd.read_csv('train.csv')
#test_df = pd.read_csv('test.csv')

for i in range(0,10):
    print(train_df['tweet'][i])

x_tr = train_df['tweet']
y_tr = train_df['label']

def clean_up(text):
    text = re.sub('\d+','',text)
    text = re.sub(r'[^\w\s]','',text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

x_tr = x_tr.apply(clean_up)

for i in range(0,10):
    print(clean_up(train_df['tweet'][i]))
    
x_tr_shape = x_tr[:x_tr.shape[0]]
print(train_df.groupby('label')['label'].count())
print('The shape of train is {}'.format(x_tr_shape.shape))

for i in range(0,10):
    print(x_tr[i])

tokenizer = text.Tokenizer(num_words=VOCAB_SIZE,split=' ')
tokenizer.fit_on_texts(x_tr)
x_tr = tokenizer.texts_to_sequences(x_tr)
#X_test= tokenizer.texts_to_sequences(X_test.values)
 
x_tr = sequence.pad_sequences(x_tr, maxlen=MAXLEN)
#X_test = sequence.pad_sequences(X_test, maxlen=MAXLEN) 

# creating the Model
      
model = Sequential()
#sem usar Transfer Learning
model.add(Embedding(VOCAB_SIZE, 100, input_length = MAXLEN))
model.add(Conv1D(FILTERS,
                 KERNEL_SIZE,
                 padding = 'Valid',
                 activation = 'relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(HIDDEN_DIMS, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

#spliting the data in train and test 
X_train, X_test, Y_train, Y_test = train_test_split(x_tr,y_tr, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

#Training the model
r = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = EPOCHS,
          validation_data=(X_test, Y_test))

# plot some data

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

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
