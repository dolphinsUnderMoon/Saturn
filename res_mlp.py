from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.regularizers import *
import numpy as np
import h5py
import os


class Config:
    def __init__(self):
        self.input_dim = 16
        self.training_x_path = "./data/embeddings16_5000.npy"
        self.training_y_path = "./data/y_train_5000.npy"
        self.validation_x_path = "./data/embeddings16_val.npy"
        self.validation_y_path = "./data/y_val.npy"
        self.testing_x_path = "./data/PCA16_test.npy"
        # self.testing_y_path = "./data/high_y.npy"
        self.num_residual_units = 3
        self.batch_size = 32
        self.max_epochs = 30
        self.lr = 1e-3
        self.regularization = 1e-3
        self.discount = 1e-1
        self.res_mlp_model_path = "./model/res_mlp_model"


res_mlp_config = Config()


class ResidualMultiLayerPerception:
    def __init__(self, config):
        if not os.path.exists(config.res_mlp_model_path):
            os.mkdir(config.res_mlp_model_path)

        # the regression model
        self.regression_mlp = Sequential()
        self.regression_mlp.add(Dense(32, activation='sigmoid', input_dim=config.input_dim,
                                kernel_regularizer=l2(config.regularization)))
        # self.regression_mlp.add(Dense(64, activation='sigmoid',
        #                         kernel_regularizer=l2(config.regularization)))
        # self.regression_mlp.add(Dense(32, activation='sigmoid',
        #                         kernel_regularizer=l2(config.regularization)))
        self.regression_mlp.add(Dense(1))

        self.regression_mlp.compile(optimizer=Adam(lr=config.lr), loss='mse', metrics=['mse'])

        # the residual regression model
        self.residual_mlp_collections = []
        for i in range(config.num_residual_units):
            residual_mlp = Sequential()
            residual_mlp.add(Dense(32, activation='tanh', input_dim=config.input_dim,
                                   kernel_regularizer=l2(config.regularization)))
            # residual_mlp.add(Dense(16, activation='tanh',
            #                        kernel_regularizer=l2(config.regularization)))
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
            os.system("touch " + regression_model_structure_path)
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
                os.system("touch " + residual_model_structure_path)
            with open(residual_model_structure_path, 'w') as f:
                f.write(residual_model_structure)

            count += 1
            residual_train -= residual_mlp.predict(train_x)

    def predict(self, x):
        output = self.regression_mlp.predict(x)
        for res_mlp in self.residual_mlp_collections:
            output += 1e-1 * res_mlp.predict(x)

        return output


if __name__ == '__main__':
    train_x = np.load(res_mlp_config.training_x_path)
    train_y = np.load(res_mlp_config.training_y_path)
    validation_x = np.load(res_mlp_config.validation_x_path)
    validation_y = np.load(res_mlp_config.validation_y_path)

    res_mlp = ResidualMultiLayerPerception(res_mlp_config)

    res_mlp.train(train_x, train_y, res_mlp_config)

    prediction_val = res_mlp.predict(validation_x)
    mse = np.mean(np.square(prediction_val - validation_y) / 2)
    print(mse)