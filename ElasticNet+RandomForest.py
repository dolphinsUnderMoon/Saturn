import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from tpot.builtins import StackingEstimator


class Config:
    def __init__(self):
        self.trainingdata_path = "./data/trainingdata.npy"
        self.testingdata_path = "./data/test_x.npy"


EN_RF_config = Config()

if __name__ == '__main__':
    trainingdata = np.load(EN_RF_config.trainingdata_path)
    x = trainingdata[:, 0:-1]
    y = trainingdata[:, -1].reshape(-1, 1)
    train_x, validation_x, train_y, validation_y = train_test_split(x, y, test_size=0.368, random_state=None)

    pipeline = make_pipeline(
        StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.75, tol=0.0001)),
        StandardScaler(),
        RandomForestRegressor(bootstrap=False, max_features=0.45, min_samples_leaf=13, min_samples_split=12,
                              n_estimators=100)
    )

    pipeline.fit(train_x, train_y)
    prediction_train = pipeline.predict(train_x)
    prediction_val = pipeline.predict(validation_x)

    mse = [np.mean(np.square(prediction_train - train_y)) / 2,
           np.mean(np.square(prediction_val - validation_y)) / 2]
    print(mse)

    testingdata = np.load(EN_RF_config.testingdata_path)
    # high_list = [2, 179, 209, 248, 404, 456, 565, 632, 845]
    # testingdata = testingdata[high_list]
    predict = pipeline.predict(testingdata)
    np.savetxt('./data/predict/EN_RF_predict.txt', predict)