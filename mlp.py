from keras.layers import Input, Dense
from keras.models import Sequential
import numpy as np
import h5py


class Config:
    def __init__(self):
        self.input_dim = 39
        self.training_data_X_path = "./data/embeddings.npy"
        self.training_data_Y_path = "./data/ground_truth.npy"
        self.batch_size = 16
        self.max_epochs = 10

        self.mlp_model_weights_path = "./model/mlp_model_weights.h5"
        self.mlp_model_structure_path = "./model/mlp_model_structure.json"


mlp_config = Config()


model = Sequential()
model.add(Dense(128, activation='relu', input_dim=mlp_config.input_dim))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

train_x = np.load(mlp_config.training_data_X_path)
train_y = np.load(mlp_config.training_data_Y_path)
test_x = np.load("./data/test_x.npy")
test_y = np.load("./data/test_y.npy")

model.fit(train_x, train_y,
          batch_size=mlp_config.batch_size,
          epochs=mlp_config.max_epochs)

model.save_weights(mlp_config.mlp_model_weights_path)
model_structure = model.to_json()
with open(mlp_config.mlp_model_structure_path, 'w') as f:
    f.write(model_structure)

score = model.evaluate(test_x, test_y)
print(score)