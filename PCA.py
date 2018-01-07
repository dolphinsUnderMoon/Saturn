from sklearn.decomposition import PCA
import numpy as np

train_x = np.load('./data/train_x.npy')
test_x = np.load('./data/test_x.npy')

pca = PCA(n_components=16)
pca.fit(train_x)
ratio = pca.explained_variance_ratio_
print(np.cumsum(ratio))
temp = pca.transform(np.load('./data/high.npy'))
np.save('./data/PCA16_high.npy', temp)
# train_x = pca.transform(train_x)
# test_x = pca.transform(test_x)
# np.save('./data/PCA16_5000.npy', train_x[0:5000][0:])
# np.save('./data/PCA16_val.npy', train_x[5000:][0:])
# np.save('./data/PCA16_test.npy', test_x)

