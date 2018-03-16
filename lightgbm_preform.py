from lightgbm import *
import numpy as np


class Config:
    def __init__(self):
        self.trainingdata_path = "./data/trainingdata_no_outliers.npy"
        self.testing_x_path = "./data/test_x.npy"
        self.param = {'boost': 'gbdt',
                      'num_leaves': 8,
                      'learning_rate': 1,
                      'objective': 'mse',
                      'bagging_fraction': 1.0,
                      'bagging_freq': 1,
                      'is_unbalance': True,
                      'metric': 'mse'}
        self.num_round = 1000
        self.early_stopping_round = 30


lgb_config = Config()

if __name__ == '__main__':
    trainingdata = np.load(lgb_config.trainingdata_path)
    x = trainingdata[:, 0:-1]
    y = trainingdata[:, -1]
    dataset = Dataset(x, label=y)
    # train_x, validation_x, train_y, validation_y = train_test_split(x, y, test_size=0.368, random_state=None)
    num_round = lgb_config.num_round
    cv = cv(lgb_config.param, dataset, num_round, nfold=5, early_stopping_rounds=lgb_config.early_stopping_round)
    print(cv)
    best = train(lgb_config.param, dataset, num_boost_round=len(cv['l2-mean']))
    best.save_model('model.txt', num_iteration=best.best_iteration)

    testingdata = np.load(lgb_config.testing_x_path)
    # label = np.loadtxt("./data/predict/predict_lgb_label.txt")
    # low_index = np.where(label == 1)[0].tolist()
    # predict = [0] * testingdata.shape[0]
    # for index in low_index:
    #     predict[index] = best.predict(testingdata[index, :].reshape(-1, 32), num_iteration=best.best_iteration)
    # predict = np.array(predict)
    predict = best.predict(testingdata, num_iteration=best.best_iteration)
    print(predict)
    # np.savetxt("./data/predict/predict_high.txt", predict)
