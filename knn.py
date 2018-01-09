import numpy as np


class Config:
    def __init__(self):
        self.input_dim = 16
        self.training_x_path = "./data/high_x_train.npy"
        self.training_y_path = "./data/high_y_train.npy"
        self.validation_x_path = "./data/high_x_validation.npy"
        self.validation_y_path = "./data/high_y_validation.npy"
        self.testing_x_path = "./data/PCA16_test.npy"
        # self.testing_y_path = "./data/high_y.npy"
        self.batch_size = 32
        self.max_epochs = 600
        self.lr = 1e-3
        self.regularization = 1e-2
        self.model_saving_path = "./model/xgboost_model"
        self.k = 3
        self.distance_metric = 'cosine'
        self.weighted_mode = 'distance'


knn_config = Config()


def compute_square_error(y_true, y_pred):
    return np.square(y_pred - y_true) / 2

# ses = np.zeros(shape=[4])
# num_val = validation_x.shape[0]
# for sample_index in range(num_val // 64):
#     sample = validation_x[sample_index, :]
#     sample_y = validation_y[sample_index, :]
#
#     l1_distances = []
#     l2_distances = []
#     cosines = []
#     for i in range(train_x.shape[0]):
#         train_sample = train_x[i, :]
#
#         l1_distance = np.sum(np.abs(sample - train_sample))
#         l2_distance = np.sum(np.square(sample - train_sample))
#         cosine = np.sum(train_sample * sample) / np.sqrt(np.sum(sample * sample) * np.sum(train_sample * train_sample))
#
#         l1_distances.append(l1_distance)
#         l2_distances.append(l2_distance)
#         cosines.append(cosine)
#
#     l1_nearest = np.argmin(l1_distances)
#     l2_nearest = np.argmin(l2_distances)
#     cosine_nearest = np.argmax(cosines)
#
#     l1_predict = train_y[l1_nearest, :]
#     l2_predict = train_y[l2_nearest, :]
#     cosine_predict = train_y[cosine_nearest, :]
#     average_predict = (l1_predict + l2_predict + cosine_predict) / 3
#
#     l1_se = compute_square_error(l1_predict, sample_y)
#     l2_se = compute_square_error(l2_predict, sample_y)
#     cosine_se = compute_square_error(cosine_predict, sample_y)
#     average_se = compute_square_error(average_predict, sample_y)
#
#     print("%d: truth value: %f \n L1 predict: %f, SE: %f \n L2 predict: %f, SE: %f \n cosine predict: %f, SE: %f "
#           "\n average: %f, SE: %f"
#           % (sample_index, sample_y, l1_predict, l1_se, l2_predict, l2_se,
#              cosine_predict, cosine_se, average_predict, average_se))
#
#     this_se = np.array([l1_se, l2_se, cosine_se, average_se]).reshape([4])
#     ses += this_se
#
# print(ses / num_val)

# print("truth value: %f \n L1 predict: %f, SE: %f \n L2 predict: %f, SE: %f \n cosine predict: %f, SE: %f "
#       "\n average: %f, SE: %f"
#       % (sample_y, l1_predict, l1_se, l2_predict, l2_se,
#          cosine_predict, cosine_se, average_predict, average_se))


class KNearestNeighbourRegression:
    def __init__(self, k, distance_metric, train_x, train_y):
        self.k = k
        self.distance_metric = distance_metric
        self.train_x = train_x
        self.train_y = train_y

    @staticmethod
    def compute_distance(op1, op2, distance_metric):
        if distance_metric == 'l1':
            return np.sum(np.abs(op1 - op2))
        elif distance_metric == 'l2':
            return np.sum(np.square(op1 - op2))
        elif distance_metric == 'cosine':
            return np.sum(op1 * op2) / np.sqrt(np.sum(op1 ** 2) * np.sum(op2 ** 2))

    def predict(self, x, weighted_mode):
        distances = []
        for i in range(self.train_x.shape[0]):
            training_sample_x = self.train_x[i, :]
            distances.append(self.compute_distance(x, training_sample_x, self.distance_metric))

        indices = []
        corre_distances = []
        for i in range(self.k):
            temp = max(distances) if self.distance_metric == 'cosine' else min(distances)

            corre_distances.append(temp)
            indices.append(distances.index(temp))
            distances[indices[-1]] = float('-Inf') if self.distance_metric == 'cosine' else float('Inf')

        corre_distances = np.array(corre_distances)

        weights = None
        if weighted_mode == 'uniform':
            weights = np.ones([self.k]) / self.k
        elif weighted_mode == 'distance':
            weights = np.exp(corre_distances) if self.distance_metric == 'cosine' else np.exp(-corre_distances)

        weights /= np.sum(weights)
        candidates = np.array([self.train_y[index] for index in indices]).reshape([self.k])

        return np.sum(candidates * weights)


if __name__ == '__main__':
    train_x = np.load(knn_config.training_x_path)
    train_y = np.load(knn_config.training_y_path)
    validation_x = np.load(knn_config.validation_x_path)
    validation_y = np.load(knn_config.validation_y_path)

    knn = KNearestNeighbourRegression(knn_config.k, knn_config.distance_metric,
                                      train_x, train_y)

    mse = 0
    num_val = 4  # validation_x.shape[0]
    for val_index in range(num_val):
        predict = knn.predict(validation_x[val_index, :], knn_config.weighted_mode)
        mse += compute_square_error(predict, validation_y[val_index, :])
    print(mse / num_val)