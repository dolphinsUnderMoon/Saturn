from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.regularizers import *
import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split


class Config:
    def __init__(self):
        self.input_dim = 32
        # self.training_x_path = "./data/low.npy"
        # self.training_y_path = "./data/low_y.npy"
        # self.validation_x_path = "./data/high.npy"
        # self.validation_y_path = "./data/high_y.npy"
        self.trainingdata_path = "./data2/trainingdata.npy"
        self.testing_x_path = "./data2/test_x.npy"
        # self.testing_y_path = "./data/high_y.npy"
        self.num_residual_units = 2
        self.batch_size = 32
        self.max_epochs = 70
        self.lr = 1e-3
        self.regularization = 1e-2
        self.res_mlp_model_path = "./model/res_mlp_model"


res_mlp_config = Config()


class ResidualMultiLayerPerception:
    def __init__(self, config):
        if not os.path.exists(config.res_mlp_model_path):
            os.mkdir(config.res_mlp_model_path)

        # the regression model
        self.regression_mlp = Sequential()
        self.regression_mlp.add(Dense(16, activation='relu', input_dim=config.input_dim,
                                kernel_regularizer=l2(config.regularization)))
        self.regression_mlp.add(Dense(8, activation='relu',
                                kernel_regularizer=l2(config.regularization)))
        self.regression_mlp.add(Dense(16, activation='relu',
                                kernel_regularizer=l2(config.regularization)))
        self.regression_mlp.add(Dense(32, activation='relu',
                                      kernel_regularizer=l2(config.regularization)))
        self.regression_mlp.add(Dense(1))

        self.regression_mlp.compile(optimizer=Adam(lr=config.lr), loss='mse', metrics=['mse'])

        # the residual regression model
        self.residual_mlp_collections = []
        self.discount_factor = 1
        for i in range(config.num_residual_units):
            residual_mlp = Sequential()
            residual_mlp.add(Dense(64, activation='relu', input_dim=config.input_dim,
                                   kernel_regularizer=l2(config.regularization)))
            residual_mlp.add(Dense(32, activation='relu',
                                   kernel_regularizer=l2(config.regularization)))
            residual_mlp.add(Dense(1))

            residual_mlp.compile(optimizer=Adam(lr=config.lr), loss='mse', metrics=['mse'])
            self.residual_mlp_collections.append(residual_mlp)

    def train(self, train_x, train_y, config):
        self.regression_mlp.fit(train_x, train_y,
                                batch_size=config.batch_size,
                                epochs=config.max_epochs)

        predict = self.regression_mlp.predict(train_x)
        residual_train = train_y - predict

        self.regression_mlp.save_weights(config.res_mlp_model_path + "/regression_mlp_weights.h5")
        regression_model_structure = self.regression_mlp.to_json()
        regression_model_structure_path = config.res_mlp_model_path + "/regression_mlp_structure.json"
        if not os.path.exists(regression_model_structure_path):
            os.system("type nul>" + regression_model_structure_path)
        with open(regression_model_structure_path, 'w') as f:
            f.write(regression_model_structure)

        count = 1
        for residual_mlp in self.residual_mlp_collections:
            residual_mlp.fit(train_x, residual_train,
                             batch_size=config.batch_size,
                             epochs=config.max_epochs)

            residual_mlp.save_weights(config.res_mlp_model_path + "/residual_mlp_weights_" + str(count) + ".h5")
            residual_model_structure = residual_mlp.to_json()
            residual_model_structure_path = config.res_mlp_model_path \
                                            + "/residual_mlp_structure_" + str(count) + ".json"

            if not os.path.exists(residual_model_structure_path):
                os.system("type nul>" + residual_model_structure_path)
            with open(residual_model_structure_path, 'w') as f:
                f.write(residual_model_structure)

            count += 1
            residual_train -= residual_mlp.predict(train_x)

    def predict(self, x):
        output = self.regression_mlp.predict(x)
        for res_mlp in self.residual_mlp_collections:
            output += self.discount_factor * res_mlp.predict(x)

        return output


if __name__ == '__main__':
    # train_x = np.load(res_mlp_config.training_x_path)
    # train_y = np.load(res_mlp_config.training_y_path)
    # validation_x = np.load(res_mlp_config.validation_x_path)
    # validation_y = np.load(res_mlp_config.validation_y_path)
    trainingdata = np.load(res_mlp_config.trainingdata_path)
    print(trainingdata.shape)
    x = trainingdata[:, 0:-1]
    y = trainingdata[:, -1].reshape(-1, 1)
    print(x.shape)
    print(y.shape)
    train_x, validation_x, train_y, validation_y = train_test_split(x, y, test_size=0.368, random_state=None)


    res_mlp = ResidualMultiLayerPerception(res_mlp_config)

    res_mlp.train(train_x, train_y, res_mlp_config)

    prediction_train = res_mlp.predict(train_x)
    prediction_val = res_mlp.predict(validation_x)

    mae = [np.mean(np.square(prediction_train - train_y)) / 2,
           np.mean(np.square(prediction_val - validation_y)) / 2]
    print(mae)

    test_x = np.load(res_mlp_config.testing_x_path)
    # label = np.loadtxt("./data/predict/predict_label.txt")
    # low_index = np.where(label == 1)[0].tolist()
    # predict = [0] * test_x.shape[0]
    # for index in low_index:
    #     predict[index] = res_mlp.predict(test_x[index, :].reshape(-1, 32))
    # predict = np.array(predict)
    predict = res_mlp.predict(test_x)
    print(predict)
    np.savetxt('./data2/predict.txt', predict)
