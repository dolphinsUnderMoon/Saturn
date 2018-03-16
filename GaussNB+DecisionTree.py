import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier


class Config:
    def __init__(self):
        self.trainingdata_path = "./data/trainingdata_bool.npy"
        self.testingdata_path = "./data/test_x.npy"


RBF_SVC_config = Config()

if __name__ == '__main__':
    trainingdata = np.load(RBF_SVC_config.trainingdata_path)
    x = trainingdata[:, 0:-1]
    y = trainingdata[:, -1].reshape(-1, 1)
    train_x, validation_x, train_y, validation_y = train_test_split(x, y, test_size=0.368, random_state=None)

    pipeline = make_pipeline(
    StackingEstimator(estimator=LogisticRegression(C=0.0001, dual=True, penalty="l2")),
    XGBClassifier(learning_rate=1.0, max_depth=4, min_child_weight=11, n_estimators=100, nthread=1, subsample=0.45)
)

    pipeline.fit(train_x, train_y)
    score_train = pipeline.score(train_x, train_y)
    score_test = pipeline.score(validation_x, validation_y)
    print(score_train, score_test)

    testingdata = np.load(RBF_SVC_config.testingdata_path)
    pipeline.fit(x, y)
    predict = pipeline.predict(testingdata)
    print(predict)
    np.savetxt('./data/predict/predict_label.txt', predict)

