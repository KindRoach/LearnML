import numpy as np
import matplotlib.pyplot as plt

FUNC_C = 10
POINT_N = 1000
UPPER = 1.
LOWER = -UPPER
SIGMA = 0.5


def sample_once():
    inc = (UPPER - LOWER) / POINT_N
    x = np.arange(start=LOWER, stop=UPPER + inc, step=inc).reshape(-1, 1)

    # Use radial basis function kernel
    # https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    gamma = -1 / SIGMA ** 2
    x_norm = np.sum(x ** 2, axis=-1)
    k = np.exp(gamma * (x_norm[:, None] + x_norm[None, :] - 2 * np.dot(x, x.T)))

    # Use 0 mean function.
    mean = np.zeros([len(x), 1])
    u, s, v = np.linalg.svd(k)
    a = np.matmul(u, np.diag(np.sqrt(s)))
    z = np.random.normal(size=[len(x), 1])
    y = mean + np.matmul(a, z)
    return x.T[0], y.T[0]


for i in range(FUNC_C):
    x, y = sample_once()
    plt.plot(x, y)

plt.show()
