# -*- coding: utf-8 -*-
import numpy as np #导入Numpy
import pickle
import keras
from keras.models import Sequential,Model
from keras.layers import LSTM, Dense, Dropout, Input
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
LSTM_dim = 32
region_dim = 12
dense_dim = 64
# Input tensor for sequences of 20 timesteps,
# each containing a 784-dimensional vector
input_sequences = Input(shape=(timesteps, data_dim))

lstm1 = LSTM(LSTM_dim, return_sequences=True)(input_sequences)
lstm2 = LSTM(LSTM_dim, return_sequences=True)(lstm1)
lstm3 = LSTM(LSTM_dim)

region_input = Input(shape=(region_dim, ), name='region_input')
concat_layer = keras.layers.concatenate([lstm3, region_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(concat_layer)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[input_sequences, region_input], outputs=[main_output])

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
batch_size = 1280
epochs = 50
model.fit(x_train, y_train,
          batch_size=batch_size, epochs= epochs,
          validation_data=(x_val, y_val))

score, acc = model.evaluate(x_val, y_val,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)