from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import h5py


class Config:
    def __init__(self):
        self.input_dim = 39
        self.training_data_path = "./data/"
        self.batch_size = 16
        self.max_epochs = 10

        self.mlp_model_weights_path = "./model/ae_model_weights.h5"
        self.mlp_model_structure_path = "./model/ae_model_structure.json"


mlp_config = Config()


inputs = Input(shape=(mlp_config.input_dim, ))

x = Dense(128, activation='relu')(inputs)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

train_x = np.load("./data/train_x.npy")
train_y = np.load("./data/train_y.npy")
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