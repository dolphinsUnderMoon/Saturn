import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

class Config:
    def __init__(self):
        self.trainingdata_path = "./data/trainingdata_low.npy"
        self.testingdata_path = "./data/test_x.npy"


RET_config = Config()

if __name__ == '__main__':
    trainingdata = np.load(RET_config.trainingdata_path)
    x = trainingdata[:, 0:-1]
    y = trainingdata[:, -1].reshape(-1, 1)
    train_x, validation_x, train_y, validation_y = train_test_split(x, y, test_size=0.368, random_state=None)

    pipeline = make_pipeline(
    StackingEstimator(estimator=RidgeCV()),
    ExtraTreesRegressor(bootstrap=True, max_features=0.35000000000000003, min_samples_leaf=13, min_samples_split=2,
                        n_estimators=100)
)

    pipeline.fit(train_x, train_y)
    prediction_train = pipeline.predict(train_x)
    prediction_val = pipeline.predict(validation_x)

    mse = [np.mean(np.square(prediction_train - train_y)) / 2,
           np.mean(np.square(prediction_val - validation_y)) / 2]
    print(mse)

    testingdata = np.load(RET_config.testingdata_path)
    label = np.loadtxt("./data/predict/predict_label.txt")
    low_index = np.where(label == 0)[0].tolist()
    predict = [0] * testingdata.shape[0]
    for index in low_index:
        predict[index] = pipeline.predict(testingdata[index, :].reshape(-1, 32))
    predict = np.array(predict)
    np.savetxt('./data/predict/RET_predict.txt', predict)