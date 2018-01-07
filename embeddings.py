from keras.models import Model
import numpy as np
from keras.models import model_from_json


class Config:
    def __init__(self):
        self.data_path = "./data/train_x.npy"
        self.ae_model_weights_path = "./model/ae_model_weights.h5"
        self.ae_model_structure_path = "./model/ae_model_structure.json"
        self.embeddings_path = "./data/embeddings3.npy"


ae_config = Config()

with open(ae_config.ae_model_structure_path, 'r') as f:
    model_structure = f.read()

model = model_from_json(model_structure)
model.load_weights(ae_config.ae_model_weights_path)

embedding_model = Model(inputs=model.input, outputs=model.get_layer('embd').output)
embedding_outputs = embedding_model.predict(np.load(ae_config.data_path))
np.save(ae_config.embeddings_path, embedding_outputs)
