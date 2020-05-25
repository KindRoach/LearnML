import numpy as np
import matplotlib.pyplot as plt

DATA_SIZE = 10000


class Cluster(object):
    def __init__(self, init_x: np.ndarray, index: int):
        self.samples = []
        self.mean = init_x.copy()
        self.index = index

    def update_mean(self):
        self.mean = np.mean(self.samples, axis=0)

    def calculate_distance(self, sample: np.ndarray) -> float:
        return np.linalg.norm(self.mean - sample)

    def add_sample(self, sample: np.ndarray):
        self.samples.append(sample)

    def clear_sample(self):
        self.samples.clear()


def random_set(x: float, y: float) -> np.ndarray:
    data = np.random.normal(0, .5, [DATA_SIZE, 2])
    return data + [x, y]


x = np.concatenate([random_set(0, 0), random_set(3, 2), random_set(-3, 2)])
y = np.zeros(x.shape[0])

K = 2
clusters = []
init_samples = x[np.random.randint(0, x.shape[0], [K])]
for idx, val in enumerate(init_samples):
    clusters.append(Cluster(val, idx))

STEP_NUM = 5
for step in range(STEP_NUM):
    for cluster in clusters:
        cluster.clear_sample()

    for sample in x:
        closest_cluster = min(clusters, key=lambda c: c.calculate_distance(sample))
        closest_cluster.add_sample(sample)
    for cluster in clusters:
        cluster.update_mean()

    for cluster in clusters:
        dots = np.array(cluster.samples)
        plt.scatter(dots[:, 0], dots[:, 1])
        plt.scatter(cluster.mean[0], cluster.mean[1], c="black", marker="x")
    plt.show()
