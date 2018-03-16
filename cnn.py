from keras.layers import Input, Dense, convolutional, pooling, Flatten
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.regularizers import *
import numpy as np
import h5py
import tensorflow as tf


class Config:
    def __init__(self):
        self.training_x_path = "./data/PCA16_5000.npy"
        self.training_y_path = "./data/y_train_5000.npy"
        self.validation_x_path = "./data/PCA16_val.npy"
        self.validation_y_path = "./data/y_val.npy"
        self.testing_x_path = "./data/PCA16_test.npy"
        self.batch_size = 16
        self.input_dim = (4, 4)
        self.max_epochs = 20
        self.lr = 1e-3
        self.cnn_model_weights_path = "./model/cnn_model_weights.h5"
        self.cnn_model_structure_path = "./model/cnn_model_structure.json"


cnn_config = Config()

inputs = Input(shape=cnn_config.input_dim)

x = convolutional.Conv1D(32, kernel_size=3,
                         strides=1, padding='same',
                         activation='relu')(inputs)
x = pooling.MaxPool1D(2)(x)
x = convolutional.Conv1D(64, kernel_size=3,
                         strides=1, padding='same',
                         activation='relu')(x)
x = pooling.MaxPool1D(2)(x)

# model.add(convolutional.Conv1D(32, kernel_size=3,
#                                strides=1, padding='same',
#                                activation='relu',
#                                input_shape=cnn_config.input_dim))
# model.add(pooling.MaxPool1D(2))
# model.add(convolutional.Conv1D(64, kernel_size=3,
#                                strides=1, padding='same',
#                                activation='relu'))
# model.add(pooling.MaxPool1D(2))
# tf.reshape(x, [-1])
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
prediction = Dense(1)(x)

model = Model(input=inputs, output=prediction)


model.compile(optimizer=Adam(lr=cnn_config.lr), loss='mse', metrics=['mse'])

train_x = np.load(cnn_config.training_x_path).reshape((-1, 4, 4))
train_y = np.load(cnn_config.training_y_path)
validation_x = np.load(cnn_config.validation_x_path).reshape((-1, 4, 4))
validation_y = np.load(cnn_config.validation_y_path)
test_x = np.load(cnn_config.testing_x_path).reshape((-1, 4, 4))

model.fit(train_x, train_y,
          batch_size=cnn_config.batch_size,
          epochs=cnn_config.max_epochs)

model.save_weights(cnn_config.cnn_model_weights_path)
model_structure = model.to_json()
with open(cnn_config.cnn_model_structure_path, 'w') as f:
    f.write(model_structure)

prediction_train = model.predict(train_x)
prediction_val = model.predict(validation_x)

score = [np.mean(np.square(prediction_train - train_y)) / 2,
         np.mean(np.square(prediction_val - validation_y)) / 2]
print(score)
