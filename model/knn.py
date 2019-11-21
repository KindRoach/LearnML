from collections import namedtuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from helper.data_helper import get_mnist_data
from helper.eval_helper import eval_classification
from helper.log_helper import get_logger

logger = get_logger(__name__)

KNN_N = 10


def predict_one(digit: np.ndarray, x_train: np.ndarray, y_train: np.ndarray) -> bool:
    Neighbor = namedtuple('Neighbor', 'x y dis')
    neighbors = list()
    for i in range(x_train.shape[0]):
        xi = x_train[i]
        yi = y_train[i]
        neighbors.append(Neighbor(xi, yi, np.linalg.norm(digit - xi)))
    neighbors = sorted(neighbors, key=lambda neighbor: neighbor.dis)[:KNN_N]
    neighbors = [neighbor.y for neighbor in neighbors]
    return max(neighbors, key=neighbors.count)


def predict_many(x_test: np.ndarray, x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    logger.info("knn predicting test set...")
    y_pred = list()
    for i in range(x_test.shape[0]):
        xi = x_test[i]
        y_pred.append(predict_one(xi, x_train, y_train))

    logger.info("knn finished")
    return np.array(y_pred)


if __name__ == "__main__":
    x, y = get_mnist_data(n=1000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    y_pred = predict_many(x_test, x_train, y_train)
    eval_classification("my", y_test, y_pred)

    neigh = KNeighborsClassifier(n_neighbors=KNN_N)
    neigh.fit(x_train, y_train)
    y_pred_sk = neigh.predict(x_test)
    eval_classification("my", y_test, y_pred_sk)
