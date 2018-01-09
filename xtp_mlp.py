from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import *
from keras.regularizers import *
import keras
import numpy as np
import h5py


class Config:
    def __init__(self):
        self.input_dim = [16]
        self.training_x_path = "./data/embeddings16_5000.npy"
        self.training_y_path = "./data/y_train_5000.npy"
        self.validation_x_path = "./data/embeddings16_val.npy"
        self.validation_y_path = "./data/y_val.npy"
        self.testing_x_path = "./data/PCA16_test.npy"
        # self.testing_y_path = "./data/high_y.npy"
        self.batch_size = 32
        self.max_epochs = 30

        self.lr = 1e-3
        self.regularization = 1e-2
        self.xtp_mlp_model_path = "./model/xtp_mlp"


xtp_mlp_config = Config()


inputs = Input(shape=xtp_mlp_config.input_dim)

shortcut_1 = inputs
x = Dense(16, activation='relu',
          kernel_regularizer=l2(xtp_mlp_config.regularization))(inputs)
# x = Dense(16, activation='relu',
#           kernel_regularizer=l2(xtp_mlp_config.regularization))(x)
x = keras.layers.add([x, shortcut_1])
x = Dense(8, activation='relu',
          kernel_regularizer=l2(xtp_mlp_config.regularization))(x)

shortcut_2 = x
x = Dense(8, activation='relu',
          kernel_regularizer=l2(xtp_mlp_config.regularization))(x)
# x = Dense(8, activation='relu',
#           kernel_regularizer=l2(xtp_mlp_config.regularization))(x)
x = keras.layers.add([x, shortcut_2])
x = Dense(4, activation='relu',
          kernel_regularizer=l2(xtp_mlp_config.regularization))(x)

outputs = Dense(1)(x)

model = Model(input=inputs, output=outputs)

model.compile(optimizer=Adam(xtp_mlp_config.lr), loss='mse', metrics=['mse', 'mae'])


train_x = np.load(xtp_mlp_config.training_x_path)
train_y = np.load(xtp_mlp_config.training_y_path)
validation_x = np.load(xtp_mlp_config.validation_x_path)
validation_y = np.load(xtp_mlp_config.validation_y_path)

model.fit(train_x, train_y,
          batch_size=xtp_mlp_config.batch_size,
          epochs=xtp_mlp_config.max_epochs)

predict = model.predict(validation_x)
mse = np.mean(np.square(predict - validation_y) / 2)
print(mse)