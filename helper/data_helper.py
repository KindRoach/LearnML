import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_openml

from helper.log_helper import get_logger

logger = get_logger(__name__)


def get_mnist_data(n: int = 70000) -> (np.ndarray, np.ndarray):
    """ Get mnist data from openml.org by sklearn.datasets
    :return: data as x and target as y.
    """
    logger.info("collecting data...")
    mnist = fetch_openml('mnist_784')
    x, y = mnist["data"], mnist["target"]
    logger.info("data collected.")
    n = min(n, 70000)
    return x[:n], y[:n]


def get_iris_data(n: int = 150):
    logger.info("collecting data...")
    iris = datasets.load_iris()
    x, y = iris["data"], iris["target"]
    logger.info("data collected.")
    n = min(n, 70000)
    return x[:n], y[:n]
