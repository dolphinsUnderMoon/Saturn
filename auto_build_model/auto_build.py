from tpot import TPOTClassifier, TPOTRegressor
import numpy as np


data = np.load("../data/low.npy")
training_data = data[:4000]
validation_data = data[4000:]

auto_regressor = TPOTRegressor()
auto_regressor.fit(training_data[:, :-1], training_data[:, -1])
auto_regressor.score(validation_data[:, :-1], validation_data[:, -1])

auto_regressor.export("./auto_regression_low.py")