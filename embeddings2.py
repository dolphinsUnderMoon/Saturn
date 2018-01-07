from keras.models import Model
import numpy as np
from keras.models import model_from_json


class Config:
    def __init__(self):
        self.data_path = "./data/embeddings_test.npy"
        self.emlp_model_path = "./model/emlp_model"
        self.alphas = np.load("./model/emlp_model/alphas.npy")


emlp_config = Config()


def predict(x):
    index = 0
    temp = np.zeros([x.shape[0], 1])
    for id in range(30):
        model_weights_path = emlp_config.emlp_model_path + "/mlp_model_weights_" + str(id) + ".h5"
        model_structure_path = emlp_config.emlp_model_path + "/mlp_model_structure_" + str(id) + ".json"

        with open(model_structure_path, 'r') as f:
            model_structure = f.read()
        model = model_from_json(model_structure)
        model.load_weights(model_weights_path)
        each_result = model.predict(x)
        temp += emlp_config.alphas[id] * each_result
        index += 1

    return temp


if __name__ == '__main__':
    test_x = np.load(emlp_config.data_path)
    predict_y = predict(test_x)
    np.savetxt("./data/predict.txt", predict_y)