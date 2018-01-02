from keras.layers import Input, Dense
from keras.models import Model
import numpy as np


class Config:
    def __init__(self):
        self.input_dim = 39
        self.data_path = "./data/"
        self.batch_size = 16
        self.max_epochs = 10

        self.ae_model_weights_path = "./model/ae_model_weights.h5"
        self.ae_model_structure_path = "./model/ae_model_structure.json"


ae_config = Config()

inputs = Input(shape=(ae_config.input_dim,))

x = Dense(32, activation='relu')(inputs)
x = Dense(16)(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(ae_config.input_dim, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

training_data = np.load(ae_config.data_path)

model.fit(training_data, training_data,
          batch_size=ae_config.batch_size,
          epochs=ae_config.max_epochs)

model.save_weights(ae_config.ae_model_weights_path)
model_structure = model.to_json()
with open(ae_config.ae_model_structure_path, 'w') as f:
    f.write(model_structure)
