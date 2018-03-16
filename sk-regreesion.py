import numpy as np
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


data = np.load("./data/classification/training_data_classification.npy")
x = data[:, 0:-1]
y = data[:, -1].reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.368, random_state=42)
# x_train = np.load("./data/PCA16_5000.npy")
# y_train = np.load("./data/y_train_5000_bool.npy")
# x_test = np.load("./data/PCA16_val.npy")
# y_test = np.load("./data/y_val_bool.npy")


def try_different_method(model):
    model.fit(x_train, y_train)
    print(model.class_prior_)
    # x_test = x_train
    # y_test = y_train
    # score = model.score(x_test, y_test)
    # result_train = model.predict(x_train)
    # result_test = model.predict(x_test)
    # mean_square_error_train = np.mean(np.square(result_train - y_train)) / 2
    # mean_square_error_test = np.mean(np.square(result_test - y_test)) / 2
    score_train = model.score(x_train, y_train)
    score_test = model.score(x_test, y_test)
    # print(mean_absolute_error)
    return score_train, score_test
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
# model_SVC = svm.LinearSVC()
# KNN regression
model_KNeighborsRegressor = neighbors.KNeighborsRegressor(weights='uniform')
# random forest
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=200)  # 这里使用20个决策树
# Adaboost regression
# model_AdaBoostRegressor = ensemble.AdaBoostRegressor(base_estimator=svm.SVR(),
#                                                      n_estimators=2, learning_rate=1)  # 这里使用50个决策树
model_AdaBoostClassifier = ensemble.AdaBoostClassifier(base_estimator=svm.LinearSVC(),
                                                       algorithm='SAMME', n_estimators=100)
# GBRT regression
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=3)  # 这里使用100个决策树
# Bagging regression
model_BaggingRegressor = BaggingRegressor(base_estimator=svm.SVR(),
                                          n_estimators=12,
                                          max_samples=1/5,
                                          max_features=1.0,
                                          bootstrap=True,
                                          bootstrap_features=False,
                                          random_state=42,
                                          verbose=10)
# ExtraTree regression
model_ExtraTreeRegressor = ExtraTreeRegressor()
# GaussianNB
# model_GaussionNB = GaussianNB()


if __name__ == '__main__':
    # algorithm_collections = [model_DecisionTreeRegressor,
    #                          model_LinearRegression,
    #                          model_SVR,
    #                          model_KNeighborsRegressor,
    #                          model_RandomForestRegressor,
    #                          model_AdaBoostRegressor,
    #                          model_GradientBoostingRegressor,
    #                          model_BaggingRegressor,
    #                          model_ExtraTreeRegressor]

    algorithm_collections = [model_AdaBoostClassifier]

    maes = []

    for model in algorithm_collections:
        maes.append(try_different_method(model))
        x_test2 = np.load("./data/embeddings_test.npy")
        y_predict = model.predict(x_test2)
        print(y_predict)
        # np.save("./data/predict/predict.txt", y_predict)
        # np.savetxt("./data/predict/predict_label.txt", y_predict)
    print(maes)
