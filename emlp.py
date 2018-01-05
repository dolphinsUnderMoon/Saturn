from keras.layers import Input, Dense
from keras.models import Sequential, Model
import numpy as np
from keras import backend as K
import h5py
import os


class Config:
    def __init__(self):
        self.input_dim = 16
        self.training_data_x_path = "./data/embeddings16_5000.npy"
        self.training_data_y_path = "./data/y_train_5000.npy"
        self.validation_data_x_path = "./data/embeddings16_val.npy"
        self.validation_data_y_path = "./data/y_val.npy"
        self.max_epochs = 20

        self.num_weak_mlps = 10
        self.weak_mlp_structure = [32, 64, 32]
        self.num_training_samples = 5000

        self.emlp_model_saved_path = "./model/emlp_model"


emlp_config = Config()


def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


class MultiLayerPerception:
    def __init__(self, input_dim, num_hidden_neural, weights, number):
        self.id = str(number)
        self.weights = weights

        self.model = Sequential()
        self.model.add(Dense(num_hidden_neural[0],
                             activation='relu', input_dim=input_dim))
        for i in range(len(num_hidden_neural) - 1):
            self.model.add(Dense(num_hidden_neural[i + 1], activation='relu'))

        self.model.add(Dense(1))

        self.model.compile(optimizer='rmsprop', loss=self.emlp_loss,
                           metrics=['mae', 'mse'])

    def emlp_loss(self, y_true, y_pred):
        square_error = K.square(y_true - y_pred) / 2
        return K.sum(self.weights * square_error)

    def train(self, train_x, train_y, model_path):
        self.model.fit(train_x, train_y, batch_size=train_x.shape[0],
                       epochs=emlp_config.max_epochs)

        model_weights_path = model_path + "/mlp_model_weights_" + self.id + ".h5"
        model_structure_path = model_path + "/mlp_model_structure_" + self.id + ".json"

        self.model.save_weights(model_weights_path)

        model_structure = self.model.to_json()
        if not os.path.exists(model_structure_path):
            os.system("touch " + model_structure_path)

        with open(model_structure_path, 'w') as f:
            f.write(model_structure)

    def predict(self, x):
        return self.model.predict(x)


class EnsembleMultiLayerPerception:
    def __init__(self, num_weak_mlps, num_hidden_neural, model_saved_directory):
        self.num_weak_mlps = num_weak_mlps
        self.weak_mlps = []
        self.alphas = []
        self.model_saved_directory = model_saved_directory
        if not os.path.exists(model_saved_directory):
            os.mkdir(model_saved_directory)

        weights = K.ones([emlp_config.num_training_samples]) / emlp_config.num_training_samples
        for i in range(num_weak_mlps):
            self.weak_mlps.append(MultiLayerPerception(emlp_config.input_dim,
                                                       num_hidden_neural,
                                                       weights,
                                                       i))

    def train(self, train_x, train_y):
        weights = K.ones([emlp_config.num_training_samples]) / emlp_config.num_training_samples
        weak_mlp_maes = []

        for weak_mlp in self.weak_mlps:
            weak_mlp.weights = weights
            weak_mlp.train(train_x, train_y, self.model_saved_directory)

            prediction_for_training = weak_mlp.predict(train_x)
            weak_mlp_maes.append(compute_mae(train_y, prediction_for_training))
            square_error = (prediction_for_training - train_y) ** 2 / 2
            alpha = 1. / (np.mean(square_error) + 1e-9)
            self.alphas.append(alpha)

            weights *= np.exp(square_error)
            weights /= np.sum(weights)

        alpha_np = np.array(self.alphas)
        alpha_np /= np.sum(alpha_np)
        self.alphas = alpha_np.tolist()

        np.save(self.model_saved_directory + "/alphas.npy", np.array(self.alphas))
        return weak_mlp_maes

    def predict(self, x):
        index = 0
        temp = np.zeros([x.shape[0], 1])
        for weak_mlp in self.weak_mlps:
            each_result = weak_mlp.predict(x)
            temp += self.alphas[index] * each_result
            index += 1

        return temp


if __name__ == '__main__':
    test = EnsembleMultiLayerPerception(num_weak_mlps=emlp_config.num_weak_mlps,
                                        num_hidden_neural=emlp_config.weak_mlp_structure,
                                        model_saved_directory=emlp_config.emlp_model_saved_path)

    train_x = np.load(emlp_config.training_data_x_path)
    train_y = np.load(emlp_config.training_data_y_path)
    validation_x = np.load(emlp_config.validation_data_x_path)
    validation_y = np.load(emlp_config.validation_data_y_path)

    weak_mlp_maes = test.train(train_x, train_y)
    print(weak_mlp_maes)

    final_training_prediction = test.predict(train_x)
    final_validation_prediction = test.predict(validation_x)

    training_mae = compute_mae(train_y, final_training_prediction)
    validation_mae = compute_mae(validation_y, final_validation_prediction)
    print("training mae: [%.4f], validation mae: [%.4f]" % (training_mae, validation_mae))
