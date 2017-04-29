# -*- coding: utf-8 -*-
import numpy as np #导入Numpy
import pickle
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
in_pickle_file_path = 'data/'
print "load start"
infile = open(in_pickle_file_path, 'rb')
all_data_list = pickle.load(infile)
all_label_list = pickle.load(infile)
out_params = pickle.load(infile)
print "load complete"
infile.close()

data_dim = out_params["data_dim"]
timesteps = out_params["n_time_steps"]



# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32

model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', #loss :rmse?
              optimizer='rmsprop',# optimizer: adam?
              metrics=['accuracy'])

# Generate dummy training data
num_classes = 10
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))
batch_size = 64
model.fit(x_train, y_train,
          batch_size=batch_size, epochs= 5,
          validation_data=(x_val, y_val))

score, acc = model.evaluate(x_val, y_val,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)