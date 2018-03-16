from keras.layers import Input, Dense
from keras.models import Sequential
from keras.optimizers import *
from keras.regularizers import *
import numpy as np
import h5py


class Config:
    def __init__(self):
        self.input_dim = 32
        self.training_x_path = "./data/enhanced/x_train_enhanced.npy"
        self.training_y_path = "./data/enhanced/y_train_enhanced.npy"
        self.validation_x_path = "./data/enhanced/x_val_enhanced.npy"
        self.validation_y_path = "./data/enhanced/y_val_enhanced.npy"
        self.testing_x_path = "./data/test_x.npy"
        # self.testing_y_path = "./data/high_y.npy"
        self.batch_size = 16
        self.max_epochs = 550
        self.lr = 1e-3
        self.mlp_model_weights_path = "./model/mlp_model_weights.h5"
        self.mlp_model_structure_path = "./model/mlp_model_structure.json"


mlp_config = Config()


model = Sequential()
model.add(Dense(128, activation='sigmoid', input_dim=mlp_config.input_dim, kernel_regularizer=l2(0.01)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='sigmoid'))
# model.add(Dense(128, activation='sigmoid'))
# model.add(Dense(64, activation='sigmoid'))
# model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1))

model.compile(optimizer=Adam(lr=mlp_config.lr), loss='mse', metrics=['mse'])


train_x = np.load(mlp_config.training_x_path)
train_y = np.load(mlp_config.training_y_path)
validation_x = np.load(mlp_config.validation_x_path)
validation_y = np.load(mlp_config.validation_y_path)
test_x = np.load(mlp_config.testing_x_path)
# test_y = np.load(mlp_config.testing_y_path)

model.fit(train_x, train_y,
          batch_size=mlp_config.batch_size,
          epochs=mlp_config.max_epochs)

model.save_weights(mlp_config.mlp_model_weights_path)
model_structure = model.to_json()
with open(mlp_config.mlp_model_structure_path, 'w') as f:
    f.write(model_structure)

# score = model.evaluate(validation_x, validation_y)
# print(score)

prediction_train = model.predict(train_x)
prediction_val = model.predict(validation_x)

score = [np.mean(np.square(prediction_train - train_y)) / 2,
         np.mean(np.square(prediction_val - validation_y)) / 2]
print(score)
predict_y = model.predict(test_x)
np.savetxt('./data/predict/predict.txt', predict_y)
