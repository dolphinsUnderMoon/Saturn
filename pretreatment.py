import numpy as np

data = np.loadtxt('./data/test.txt')
data_shape = data.shape
data_rows = data_shape[0]
data_cols = data_shape[1]

for i in range(0, data_rows):
    for j in range(1, data_cols):
        data[i][j] =\
            (data[i][j] - data.min(axis=0)[j] + 0.1) /\
            (data.max(axis=0)[j] - data.min(axis=0)[j] + 0.2)
    print(data[i])
#train_data = np.split(train_data, [-1], axis=1)
#train_x = train_data[0]
#train_y = train_data[1]
np.save('./data/test_x.npy', data)
#np.save('./data/train_y.npy', train_y)
'''


data_train = pd.read_csv('./data/test.csv', header=None)
# data_train.drop([data_train.columns[[0,3,8,11,19,20,21,22,23]]],axis=1,inplace=True)
data_train.fillna(data_train.mean(), inplace=True)
# data_train.to_csv('./data/train_no_blank.csv')
# for i in range(1, 32):
#     data_train[i] = data_train[i].astype(np.float64)
# for i in range(1, 31):
#    data_train[i] = data_train[i].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
data_train.to_csv('./data/test_no_blank.csv')
'''





