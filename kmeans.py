from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


high = np.load("./data/high.npy")
low = np.load("./data/low.npy")
y = np.load("./data/train_y.npy")

model = KMeans(n_clusters=2)
x = np.vstack((high, low))
pca = PCA(n_components=2)
x = pca.fit_transform(x)
cluster = model.fit(x)
mark = ['or', 'ob']
for i in range(x.shape[0]):
    plt.plot(x[i][1], y[i], mark[cluster.labels_[i]])
plt.show()
# print(x.shape)
