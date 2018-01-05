import numpy as np
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor


x_train = np.load("./data/embeddings16_train.npy")
y_train = np.load("./data/y_train.npy")
x_test = np.load("./data/embeddings16_val.npy")
y_test = np.load("./data/y_val.npy")


def try_different_method(model):
    model.fit(x_train, y_train)

    # x_test = x_train
    # y_test = y_train
    # score = model.score(x_test, y_test)
    result_train = model.predict(x_train)
    result_test = model.predict(x_test)
    mean_absolute_error_train = np.mean(np.abs(result_train - y_train))
    mean_absolute_error_test = np.mean(np.abs(result_test - y_test))
    # print(mean_absolute_error)
    return mean_absolute_error_train, mean_absolute_error_test
    # plt.figure()
    # plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
    # plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    # plt.title('score: %f' % score)
    # plt.legend()
    # plt.show()


# Decision Tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
# linear regression
model_LinearRegression = linear_model.LinearRegression()
# SVR
model_SVR = svm.SVR()
# KNN regression
model_KNeighborsRegressor = neighbors.KNeighborsRegressor(weights='uniform')
# random forest
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
# Adaboost regression
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
# GBRT regression
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
# Bagging regression
model_BaggingRegressor = BaggingRegressor()
# ExtraTree regression
model_ExtraTreeRegressor = ExtraTreeRegressor()


if __name__ == '__main__':
    algorithm_collections = [model_DecisionTreeRegressor,
                             model_LinearRegression,
                             model_SVR,
                             model_KNeighborsRegressor,
                             model_RandomForestRegressor,
                             model_AdaBoostRegressor,
                             model_GradientBoostingRegressor,
                             model_BaggingRegressor,
                             model_ExtraTreeRegressor]

    # algorithm_collections = [model_KNeighborsRegressor]

    maes = []

    for model in algorithm_collections:
        maes.append(try_different_method(model))

    print(maes)
