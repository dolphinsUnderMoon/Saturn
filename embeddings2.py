from keras.models import Model
import numpy as np
from keras.models import model_from_json


class Config:
    def __init__(self):
        self.data_path = "./data/embeddings_test.npy"
        self.res_mlp_model_path = "./model/res_mlp_model"
        # self.alphas = np.load("./model/emlp_model/alphas.npy")


res_mlp_config = Config()


def predict(x):
    model_weights_path = res_mlp_config.res_mlp_model_path + "/regression_mlp_structure.h5"
    model_structure_path = res_mlp_config.res_mlp_model_path + "/regression_mlp_structure.json"
    with open(model_structure_path, 'r') as f:
        model_structure = f.read()
    model = model_from_json(model_structure)
    model.load_weights(model_weights_path)
    regression_result = model.predict(x)
    temp = regression_result
    for index in range(30):
        model_weights_path = res_mlp_config.res_mlp_model_path + "/residual_mlp_weights_" + str(id) + ".h5"
        model_structure_path = res_mlp_config.res_mlp_model_path + "/residual_mlp_structure_" + str(id) + ".json"

        with open(model_structure_path, 'r') as f:
            model_structure = f.read()
        model = model_from_json(model_structure)
        model.load_weights(model_weights_path)
        residual_result = model.predict(x)
        temp += residual_result

    return temp


if __name__ == '__main__':
    test_x = np.load(res_mlp_config.data_path)
    predict_y = predict(test_x)
    np.savetxt("./data/predict.txt", predict_y)