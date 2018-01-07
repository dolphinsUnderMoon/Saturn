from keras.layers import Input, Dense
from keras.models import Sequential
from keras.optimizers import *
from keras.regularizers import *
import numpy as np
import h5py


class Config:
    def __init__(self):
        self.input_dim = 16
        self.training_x_path = "./data/PCA16_5000.npy"
        self.training_y_path = "./data/y_train_5000.npy"
        self.validation_x_path = "./data/PCA16_val.npy"
        self.validation_y_path = "./data/y_val.npy"
        self.testing_x_path = "./data/PCA16_test.npy"
        # self.testing_y_path = "./data/high_y.npy"
        self.batch_size = 16
        self.max_epochs = 200
        self.lr = 1e-3
        self.mlp_model_weights_path = "./model/mlp_model_weights.h5"
        self.mlp_model_structure_path = "./model/mlp_model_structure.json"


mlp_config = Config()


model = Sequential()
model.add(Dense(32, activation='sigmoid', input_dim=mlp_config.input_dim, kernel_regularizer=l2(0.01)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='sigmoid'))
# model.add(Dense(128, activation='sigmoid'))
# model.add(Dense(64, activation='sigmoid'))
# model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1))

model.compile(optimizer=Adam(lr=mlp_config.lr), loss='mse', metrics=['mae', 'mse'])


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
validation_prediction = model.predict(validation_x)
mean_absolute_error = np.mean(np.abs(validation_prediction - validation_y))
print(mean_absolute_error)
predict_y = model.predict(test_x)
np.savetxt('./data/predict/predict.txt', predict_y)
