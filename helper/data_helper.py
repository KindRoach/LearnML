import numpy as np
from sklearn.datasets import fetch_openml

from model.knn import logger


def get_mnist_data() -> (np.ndarray, np.ndarray):
    """ Get mnist data from openml.org by sklearn.datasets
    :return: two vectors as x and y.
    """
    logger.info("collecting data...")
    mnist = fetch_openml('mnist_784')
    x, y = mnist["data"], mnist["target"]
    logger.info("data collected.")
    return x, y


def get_mnist_data_binary_is5() -> (np.ndarray, np.ndarray):
    """ Get mnist data from openml.org by sklearn.datasets with binary target label
    :return: two vectors as x and y.
    """
    x, y = get_mnist_data()
    y = y == '5'
    return x, y
