import numpy as np
import pandas as pd
'''
x = np.load('./data/low.npy')
y = np.load('./data/low_y.npy')
high_x = np.load('./data/high.npy')
high_y = np.load('./data/high_y.npy')
data_shape = high_x.shape
data_rows = data_shape[0]
data_cols = data_shape[1]

high_x = high_x + np.random.normal(0, 1/12, data_shape)
# high_y = high_y + np.random.normal(0, 1, (data_rows, 1))
high_x = np.vstack((high_x, )*10)
high_y = np.vstack((high_y, )*10)
# for i in range(9):
#     low = np.vstack((x[(500 * i):(500 * (i+1))][0:], high_x))
#     low_y = np.vstack((y[(500 * i):(500 * (i+1))][0:], high_y))
#     np.save('./data/low' + str(i+1) + '+high.npy', low)
#     np.save('./data/low_y' + str(i+1) + '+high.npy', low_y)
# low = np.vstack((x[4500:][0:], high_x))
# low_y = np.vstack((y[4500:][0:], high_y))
x_enhanced = np.vstack((x, high_x))
y_enhanced = np.vstack((y, high_y))
np.save('./data/enhanced/x_enhanced.npy', x_enhanced)
np.save('./data/enhanced/y_enhanced.npy', y_enhanced)


for i in range(data_rows):
    if y[i] > 6.7:
        high.append(y[i].tolist())
    else:
        low.append(y[i].tolist())

np.save("./data/high_y.npy", np.array(high))
np.save("./data/low_y.npy", np.array(low))
#train_data = np.split(train_data, [-1], axis=1)
#train_x = train_data[0]
#train_y = train_data[1]
'''




data_train = pd.read_csv('./data2/train.csv', header=None)
# data_train.drop([data_train.columns[[0,3,8,11,19,20,21,22,23]]],axis=1,inplace=True)
data_train.fillna(0, inplace=True)
# data_train.to_csv('./data/train_no_blank.csv')
# for i in range(1, 32):
#     data_train[i] = data_train[i].astype(np.float64)
# for i in range(1, 31):
#    data_train[i] = data_train[i].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
data_train.to_csv('./data2/train_no_blank.csv')






