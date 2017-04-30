'''Trains a simple deep RNN on the MNIST dataset.
Christopher Holbert
'''
import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from ZRNN import LSTM3
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.preprocessing import sequence

#load the Amazon dataset
filename = 'Amazon_Unlocked_Mobile_125k.csv'

#dtype1 = np.dtype([('rating', 'f2'),('reviews', 'S10')])
#dataset = np.loadtxt(filename, delimiter=",", skiprows=1, dtype='S10', comments=None)

dataset = pd.read_csv(filename, delimiter = ",")

train,test = train_test_split(dataset, test_size = 0.2)



y_train = np.array(train['Rating']-1)
y_test = np.array(test['Rating']-1)



# vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=2000)
tokenizer.fit_on_texts(train['Reviews'])
sequences_train = tokenizer.texts_to_sequences(train['Reviews'])
sequences_test = tokenizer.texts_to_sequences(test['Reviews'])

X_train = sequence.pad_sequences(sequences_train, maxlen=40)
X_test = sequence.pad_sequences(sequences_test, maxlen=40)

Y_train = np_utils.to_categorical(y_train, 5)
Y_test = np_utils.to_categorical(y_test, 5)

batch_size = 32
nb_epoch = 20

#parameters for LSTM network
nb_lstm_outputs = 200
#nb_time_steps = img_rows
#dim_input_vector = img_cols

#load MNIST dataset
#(X_train,y_train),(X_test,y_test) = mnist.load_data()
#input_shape = (nb_time_steps,dim_input_vector)
#print X_train.shape
#X_train = X_train.astype('float32')/255
#X_test = X_test.astype('float32')/255
#Y_train = np_utils.to_categorical(y_train,nb_classes = nb_classes)
#Y_test = np_utils.to_categorical(y_test,nb_classes = nb_classes)

model = Sequential()
model.add(Embedding(2000, 100, dropout=0.2))
model.add(LSTM3(nb_lstm_outputs)) 
model.add(Dropout(0.2))
model.add(Dense(5))
model.add(Activation('softmax'))
model.summary()

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
history = model.fit(X_train,Y_train,nb_epoch = nb_epoch,batch_size=batch_size,shuffle = True,validation_split = 0.1)
score = model.evaluate(X_test,Y_test)
print 'test loss',score[0]
print 'test accuracy',score[1]
