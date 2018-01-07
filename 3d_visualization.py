import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


high = np.load('./data/embeddings3_high.npy')
low = np.load('./data/embeddings3_low.npy')

x_high = high[:, 0]
y_high = high[:, 1]
z_high = high[:, 2]

x_low = low[:, 0]
y_low = low[:, 1]
z_low = low[:, 2]

plot_3d = plt.subplot(111, projection='3d')
plot_3d.scatter(x_high, y_high, z_high, c='b')
plot_3d.scatter(x_low, y_low, z_low, c='r')
plot_3d.set_zlabel('Z')
plot_3d.set_ylabel('Y')
plot_3d.set_xlabel('X')

plt.show()
