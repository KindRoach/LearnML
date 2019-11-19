from collections import namedtuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from loghelper import get_logger

logger = get_logger("mnist.py")


def get_data() -> (np.ndarray, np.ndarray):
    """ Get mnist data from openml.org by sklearn.datasets
    :return: two vectors as x and y.
    """
    logger.info("collecting data...")
    mnist = fetch_openml('mnist_784')
    x, y = mnist["data"], mnist["target"]
    logger.info("data collected.")
    return x, y


def get_data_binary() -> (np.ndarray, np.ndarray):
    """ Get mnist data from openml.org by sklearn.datasets with binary target label
    :return: two vectors as x and y.
    """
    x, y = get_data()
    y = y == '5'
    return x, y


def predict_one(digit: np.ndarray, x_train: np.ndarray, y_train: np.ndarray) -> bool:
    Neighbor = namedtuple('Neighbor', 'x y dis')
    neighbors = list()
    for i in range(x_train.shape[0]):
        xi = x_train[i]
        yi = y_train[i]
        neighbors.append(Neighbor(xi, yi, np.linalg.norm(digit - xi)))
    neighbors = sorted(neighbors, key=lambda neighbor: neighbor.dis)[:9]
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


x, y = get_data_binary()
x = x[:5000]
y = y[:5000]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
y_pred_knn = predict_many(x_test, x_train, y_train)
logger.info(f"knn result:\n{classification_report(y_test, y_pred_knn)}")

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train)
y_pred_sgd = sgd_clf.predict(x_test)
logger.info(f"sgd result:\n{classification_report(y_test, y_pred_sgd)}")
