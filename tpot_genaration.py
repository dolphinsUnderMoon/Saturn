from tpot import *
import numpy as np
from sklearn.model_selection import train_test_split


class Config:
    def __init__(self):
        self.trainingdata_path = "./data2/trainingdata.npy"


tpot_config = Config()

if __name__ == '__main__':
    trainingdata = np.load(tpot_config.trainingdata_path)
    x = trainingdata[:, 0:-1]
    y = trainingdata[:, -1]
    # x = np.load(tpot_config.train_x_path)
    # y = np.load(tpot_config.train_y_path).reshape(-1)
    train_x, validation_x, train_y, validation_y = train_test_split(x, y, test_size=0.368, random_state=None)

    model = TPOTRegressor(generations=100, population_size=10,
                          verbosity=2, n_jobs=-1,
                          early_stop=10)
    model.fit(x, y)
    print(model.score(x, y))

    model.export('./data2/tpot_new_pipeline.py')
