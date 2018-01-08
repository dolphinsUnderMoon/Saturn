import numpy as np
from matplotlib import pyplot as plt

ground_truth = np.load("./data/y_val.npy")
mlp_predict = np.load("./data/val_predict_mlp.npy")
svr_predict = np.load("./data/val_predict_rfe_adaboost_svr.npy")

mlp_residual = mlp_predict - ground_truth
svr_residual = svr_predict - ground_truth
print(np.mean(np.square(svr_residual) / 2))
print(np.mean(np.square(mlp_residual) / 2))


# num_show = ground_truth.shape[0]
# num_show = 50
#
# indices = np.arange(0, num_show)
#
# plt.plot(indices, mlp_residual[0:num_show, :], 'r', label="mlp")
# plt.plot(indices, svr_residual[0:num_show, :], 'b', label="svr")
# plt.legend()
# plt.show()