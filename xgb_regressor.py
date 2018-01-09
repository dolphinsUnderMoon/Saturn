import numpy as np
import xgboost as xgb


class Config:
    def __init__(self):
        self.input_dim = 16
        self.training_x_path = "./data/embeddings16_5000.npy"
        self.training_y_path = "./data/classification_train_y.npy"
        self.validation_x_path = "./data/embeddings16_val.npy"
        self.validation_y_path = "./data/classification_validation_y.npy"
        self.testing_x_path = "./data/PCA16_test.npy"
        # self.testing_y_path = "./data/high_y.npy"
        self.batch_size = 32
        self.max_epochs = 600
        self.lr = 1e-3
        self.regularization = 1e-2
        self.model_saving_path = "./model/xgboost_model"


xgb_config = Config()


train_x = np.load(xgb_config.training_x_path)
train_y = np.load(xgb_config.training_y_path)
validation_x = np.load(xgb_config.validation_x_path)
validation_y = np.load(xgb_config.validation_y_path)

training_data = xgb.DMatrix(train_x, train_y)
validation_data = xgb.DMatrix(validation_x)

parameters = {
    'booster': 'gbtree',
    'tree_method': 'hist',
    'objective': 'binary:logistic', #gamma
    'gamma': 50,
    'max_depth': 8,
    'max_leaves': 4,
    'grow_policy': 'lossguide',
    'lambda': 30,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.012,
    'seed': 1000,
    'nthread': 1,
    'metric': 'error'
}

plst = parameters.items()
model = xgb.train(plst, training_data, xgb_config.max_epochs)
model.save_model(xgb_config.model_saving_path)


ans = model.predict(validation_data)
# mae = np.mean(np.abs(ans - validation_y))
# mse = np.mean(np.square(ans - validation_y) / 2)
# print(mse, mae)
print(ans)
accuracy = np.mean(ans == validation_y)
print(accuracy)