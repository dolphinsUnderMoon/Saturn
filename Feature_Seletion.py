from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVR
import numpy as np


x = np.load('./data/train_x.npy')
y = np.load('./data/train_y.npy')
x_test = np.load('./data/test_x.npy')

rfe = RFE(estimator=LinearSVR(), n_features_to_select=16)
rfe.fit(x, y)
rfe16_x_train = rfe.transform(x)
rfe16_x_test = rfe.transform(x_test)
# var = np.var(x, axis=0)
# 计算每一列的方差
# print(var)
'''
for i in range(x.shape[1]-2):
    for j in range(10):
        x_temp = x[:, i] * j/10 + x[:, i+2] * (10-j)/10
        x_temp = x_temp - np.mean(x_temp)
        y_temp = y - np.mean(y)
        correlation = np.dot(x_temp, y_temp) / \
                (np.linalg.norm(x_temp)*np.linalg.norm(y_temp))
    # if correlation > 0.1:
    #    feature_list.append(i)
        print(i, j, correlation)
'''
# x_selected = x[:, feature_list]
# x_test_selected = x_test[:, feature_list]
np.save('./data/x_selected_5000.npy', rfe16_x_train[0:5000][0:])
np.save('./data/x_selected_val.npy', rfe16_x_train[5000:][0:])
np.save('./data/x_selected_test.npy', rfe16_x_test)


