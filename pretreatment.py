import numpy as np
#
# x = np.load('./data/low.npy')
# y = np.load('./data/low_y.npy')
# high_x = np.load('./data/high.npy')
# high_y = np.load('./data/high_y.npy')
# data_shape = x.shape
# data_rows = data_shape[0]
# data_cols = data_shape[1]
#
#
# for i in range(9):
#     low = np.vstack((x[(500 * i):(500 * (i+1))][0:], high_x))
#     low_y = np.vstack((y[(500 * i):(500 * (i+1))][0:], high_y))
#     np.save('./data/low' + str(i+1) + '+high.npy', low)
#     np.save('./data/low_y' + str(i+1) + '+high.npy', low_y)
# low = np.vstack((x[4500:][0:], high_x))
# low_y = np.vstack((y[4500:][0:], high_y))
# np.save('./data/low' + str(10) + '+high.npy', low)
# np.save('./data/low_y' + str(10) + '+high.npy', low_y)
#
# '''
# for i in range(data_rows):
#     if y[i] > 6.7:
#         high.append(y[i].tolist())
#     else:
#         low.append(y[i].tolist())
#
# np.save("./data/high_y.npy", np.array(high))
# np.save("./data/low_y.npy", np.array(low))
# #train_data = np.split(train_data, [-1], axis=1)
# #train_x = train_data[0]
# #train_y = train_data[1]
#
#
#
# '''
# '''
# data_train = pd.read_csv('./data/test.csv', header=None)
# # data_train.drop([data_train.columns[[0,3,8,11,19,20,21,22,23]]],axis=1,inplace=True)
# data_train.fillna(data_train.mean(), inplace=True)
# # data_train.to_csv('./data/train_no_blank.csv')
# # for i in range(1, 32):
# #     data_train[i] = data_train[i].astype(np.float64)
# # for i in range(1, 31):
# #    data_train[i] = data_train[i].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# data_train.to_csv('./data/test_no_blank.csv')
# '''

a = np.load("./data/embeddings16_5000.npy")
aa = np.load("./data/y_train_5000.npy")
b = np.load("./data/embeddings16_val.npy")
bb = np.load("./data/y_val.npy")
high_x = []
high_y = []
low_x = []
low_y = []
classification_train = []
classification_validation = []

for i in range(a.shape[0]):
    if aa[i] > 6.7:
        classification_train.append(1)
    else:
        classification_train.append(0)

for i in range(b.shape[0]):
    if bb[i] > 6.7:
        classification_validation.append(1)
    else:
        classification_validation.append(0)

np.save("./data/classification_train_y.npy", classification_train)
np.save("./data/classification_validation_y.npy", classification_validation)

# np.save("./data/high_x.npy", np.array(high_x))
# np.save("./data/high_y.npy", np.array(high_y))
# np.save("./data/low_x.npy", np.array(low_x))
# np.save("./data/low_y.npy", np.array(low_y))