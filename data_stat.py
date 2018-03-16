import numpy as np
from scipy import stats
from scipy.stats import chisquare, laplace

data = np.load("./data/train_y.npy")


def pro_bet_a_b(a, b, miu, sigma):
    pro = laplace(miu, sigma).cdf(b) - laplace(miu, sigma).cdf(a)
    return pro


threshold = 38
while(True):
    low = data[data < threshold]
    miu = np.median(low)
    sigma = np.mean(np.abs(low - miu))
    (n, bins) = np.histogram(low, bins=10, normed=False)
# print((n, bins))
    n_observed = n.tolist()
    n_expected = [0] * n.shape[0]
    for i in range(n.shape[0]):
        n_expected[i] = pro_bet_a_b(bins[i], bins[i+1], miu, sigma) * low.shape[0]
    chisq, p = chisquare(n_observed, n_expected)
    print(chisq, p)
    if p > 0.01:
        break
    threshold = miu + 4.606 * sigma
    print(threshold)




